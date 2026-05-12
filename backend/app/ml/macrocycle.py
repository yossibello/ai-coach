"""
Macrocycle planner — reverse-periodization from event date to today.

Unlike `inference.generate_recommendation` (which answers "what's the best
*next* workout?"), this module answers:

    "Given an event on date D and today's CTL, what weekly TSS and phase
     should I be doing each week between now and then so I peak ON event day?"

Algorithm
─────────
1. Compute target peak CTL for event day from event_type + current CTL.
2. Walk backwards from event day to today, assigning a phase per week
   (Friel/Coggan periodization: taper → peak → build → base).
3. For each phase, compute a target weekly TSS that ramps CTL toward the
   peak with a safe slope (max +5 TSS/day per week, so CTL grows ~+5/week).
4. Return the weekly schedule + computed peak CTL + ramp info.

This is rule-based (not model-driven) on purpose — it's a deterministic
"skeleton" that the per-day model recommendations slot into. The user can
trust the macrocycle is sane; the day-to-day model adds the texture.

References
──────────
- Coggan & Allen (2012) — Training and Racing with a Power Meter, ch. 7-9
- Friel (2018) — The Cyclist's Training Bible, ch. 7
- Banister & Calvert (1980) — Performance modelling (CTL/ATL)
- Bannister IF model: TSB optimum for race day ≈ +15 to +25
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Any, Optional


# ── Event-specific peak CTL targets (TSS/day) ────────────────────────────────
# Floor: amateur completing distance. Ceiling: competitive amateur.
# These are *targets*; planner clamps them to [current_ctl + 5, current_ctl + 30]
# to avoid prescribing 14-week 50-point ramps that risk injury.
EVENT_PEAK_CTL_TARGETS: dict[str, int] = {
    "crit":            70,   # Short, max-effort: high CTL helps repeated efforts
    "tt":              75,   # Sustained threshold: high CTL critical
    "long_road":       80,   # 4-7h hard pack racing
    "stage_race":      85,   # Multi-day hardest demand
    "gran_fondo":      75,   # 4-8h endurance + climbs
    "climbing_camp":   80,   # Multi-day big climbing volume
    "mtb_marathon":    70,   # 3-6h variable power off-road
    "ultra_endurance": 90,   # 8h+ steady aerobic
    "triathlon_70_3":  70,   # 90km bike leg
    "triathlon_140_6": 85,   # 180km bike leg
}
DEFAULT_PEAK_CTL = 70

# Maximum safe weekly CTL ramp (TSS/day per week). Above this, injury/illness
# risk rises sharply (Gabbett 2016, ACWR research).
MAX_CTL_RAMP_PER_WEEK = 5.0
MIN_CTL_RAMP_PER_WEEK = 1.5

# Recovery-week frequency: every Nth week is a recovery week (-30% TSS).
RECOVERY_WEEK_INTERVAL = 4


# ── Phase definitions ────────────────────────────────────────────────────────
@dataclass
class WeekPlan:
    week_index: int               # 0 = next Monday, 1 = following week …
    week_start: str               # ISO date of Monday of that week
    weeks_to_event: int           # at end of this week
    phase: str                    # base | build | peak | taper | event_week
    target_weekly_tss: int        # cumulative TSS target for the week
    target_ctl_end: float         # projected CTL at end of week
    workout_focus: list[str]      # types to emphasize (e.g. ["sweetspot", "long_ride"])
    is_recovery_week: bool
    notes: str


def _phase_for_weeks_out(weeks_to_event: int) -> str:
    if weeks_to_event <= 0:  return "event_week"
    if weeks_to_event <= 2:  return "taper"
    if weeks_to_event <= 8:  return "peak"
    if weeks_to_event <= 16: return "build"
    return "base"


def _focus_for_phase(phase: str, event_type: Optional[str]) -> list[str]:
    """Workout types this phase should emphasize.

    Event type biases the mix (e.g. climbing_camp = more sweetspot+long_ride;
    crit = more vo2max). Mirrors `_EVENT_BIAS` in cold_start.py.
    """
    base = {
        "base":       ["endurance", "long_ride", "tempo"],
        "build":      ["sweetspot", "tempo", "long_ride", "endurance"],
        "peak":       ["threshold", "vo2max", "sweetspot", "endurance"],
        "taper":      ["threshold", "endurance", "easy"],
        "event_week": ["easy", "recovery"],
    }.get(phase, ["endurance"])

    if not event_type or phase in ("taper", "event_week"):
        return base

    # Event-specific overlay
    if event_type in ("climbing_camp", "ultra_endurance", "gran_fondo"):
        if phase in ("build", "peak"):
            return ["sweetspot", "long_ride", "endurance", "tempo"]
    if event_type in ("crit", "tt"):
        if phase == "peak":
            return ["vo2max", "threshold", "sweetspot", "endurance"]
    if event_type in ("triathlon_70_3", "triathlon_140_6"):
        if phase == "peak":
            return ["threshold", "tempo", "long_ride", "endurance"]
    if event_type == "stage_race":
        if phase == "peak":
            return ["threshold", "long_ride", "sweetspot", "endurance"]
    return base


def _phase_tss_multiplier(phase: str) -> float:
    """How weekly TSS scales relative to the ramp baseline."""
    return {
        "base":       0.85,   # build aerobic base, lower TSS
        "build":      1.00,   # progressive overload
        "peak":       1.10,   # quality + maintained volume
        "taper":      0.55,   # cut volume, keep intensity
        "event_week": 0.30,   # event itself counts here
    }.get(phase, 1.0)


def build_macrocycle(
    *,
    current_ctl: float,
    current_atl: float,
    event_date: datetime,
    today: Optional[datetime] = None,
    event_type: Optional[str] = None,
    event_name: Optional[str] = None,
    training_days_per_week: int = 5,
    ftp: float = 250.0,
) -> dict[str, Any]:
    """
    Build a week-by-week macrocycle from today through event day.

    Returns a dict with:
      - peak_ctl_target: float  (planned CTL on event day)
      - planned_tsb_event: float (taper aims here, typically +15 to +25)
      - weeks: list[WeekPlan]
      - summary: human-readable strings
      - feasibility: "comfortable" | "ambitious" | "unrealistic"
      - confidence: 0..1 — how trustworthy this plan is
    """
    today = today or datetime.now(timezone.utc)
    if event_date.tzinfo is None:
        event_date = event_date.replace(tzinfo=timezone.utc)
    if today.tzinfo is None:
        today = today.replace(tzinfo=timezone.utc)

    days_to_event = (event_date.date() - today.date()).days
    if days_to_event <= 0:
        return {
            "error": "Event date is today or in the past",
            "weeks": [],
            "peak_ctl_target": current_ctl,
        }

    weeks_to_event = max(1, days_to_event // 7)

    # ── 1. Peak CTL target ──────────────────────────────────────────────────
    raw_target = EVENT_PEAK_CTL_TARGETS.get(event_type or "", DEFAULT_PEAK_CTL)
    # Clamp: must be ≥ current+5 (some growth) and ≤ current + max ramp possible.
    max_possible = current_ctl + (weeks_to_event * MAX_CTL_RAMP_PER_WEEK)
    min_growth = current_ctl + 5
    peak_ctl_target = max(min_growth, min(raw_target, max_possible))

    # Feasibility check
    needed_ramp = (peak_ctl_target - current_ctl) / max(1, weeks_to_event)
    if needed_ramp > MAX_CTL_RAMP_PER_WEEK * 0.9:
        feasibility = "ambitious"
    elif needed_ramp < MIN_CTL_RAMP_PER_WEEK:
        feasibility = "comfortable"
    else:
        feasibility = "balanced"
    if raw_target > max_possible:
        feasibility = "unrealistic"  # event too soon for desired peak

    # ── 2. Walk forward week by week ────────────────────────────────────────
    weeks: list[WeekPlan] = []
    ctl = current_ctl
    # Find Monday of this week
    monday = today - timedelta(days=today.weekday())

    for w_idx in range(weeks_to_event + 1):
        week_start = monday + timedelta(weeks=w_idx)
        weeks_remaining = weeks_to_event - w_idx
        phase = _phase_for_weeks_out(weeks_remaining)
        is_recovery = (
            w_idx > 0
            and (w_idx + 1) % RECOVERY_WEEK_INTERVAL == 0
            and phase not in ("taper", "event_week")
        )

        # Target weekly TSS to drive CTL toward peak.
        # CTL ≈ 7-day rolling avg of daily TSS (42-day EWMA actually, but
        # 7-day approximates well for planning).
        # If we want CTL to grow by `needed_ramp` per week → daily TSS ≈
        # current_ctl + needed_ramp + small overshoot.
        target_daily_tss = ctl + needed_ramp + 2.0
        target_weekly_tss = int(target_daily_tss * 7 * _phase_tss_multiplier(phase))
        if is_recovery:
            target_weekly_tss = int(target_weekly_tss * 0.65)

        # Project end-of-week CTL (simple linear approximation).
        if phase == "event_week":
            ctl_delta = -1.0  # taper through event
        elif phase == "taper":
            ctl_delta = -2.0  # planned reduction for freshness
        elif is_recovery:
            ctl_delta = -1.5
        else:
            # Aim for ~needed_ramp growth, but cap at MAX_CTL_RAMP_PER_WEEK
            ctl_delta = min(needed_ramp, MAX_CTL_RAMP_PER_WEEK)
        ctl = max(0, ctl + ctl_delta)

        focus = _focus_for_phase(phase, event_type)

        notes = ""
        if w_idx == 0:
            notes = "Current week — partial schedule"
        elif phase == "taper":
            notes = "Cut volume 40-50%, keep intensity, sharpen"
        elif phase == "event_week":
            notes = f"Event week{' — ' + event_name if event_name else ''}. Pre-race openers + rest."
        elif is_recovery:
            notes = "Recovery week — drop intensity, focus on Z1/Z2"
        elif phase == "peak":
            notes = "Peak phase — quality intervals, maintain volume"
        elif phase == "build":
            notes = "Build phase — progressive overload, sweetspot/threshold"
        elif phase == "base":
            notes = "Base phase — aerobic foundation, low intensity"

        weeks.append(WeekPlan(
            week_index=w_idx,
            week_start=week_start.date().isoformat(),
            weeks_to_event=weeks_remaining,
            phase=phase,
            target_weekly_tss=target_weekly_tss,
            target_ctl_end=round(ctl, 1),
            workout_focus=focus,
            is_recovery_week=is_recovery,
            notes=notes,
        ))

    # ── 3. Confidence ───────────────────────────────────────────────────────
    # Reconcile target with what the schedule actually projects.
    # The "peak" of the macrocycle is the highest CTL reached *before* taper
    # begins (taper intentionally sheds CTL for race-day freshness).
    pre_taper_ctls = [w.target_ctl_end for w in weeks if w.phase not in ("taper", "event_week")]
    peak_ctl_projected = round(max(pre_taper_ctls) if pre_taper_ctls else current_ctl, 1)

    # If projection falls noticeably short of the desired peak, downgrade.
    if peak_ctl_projected < raw_target - 5 and feasibility != "unrealistic":
        feasibility = "ambitious"

    if feasibility == "unrealistic":
        confidence = 0.30
    elif feasibility == "ambitious":
        confidence = 0.65
    elif weeks_to_event < 4:
        confidence = 0.70  # short window: limited room to adjust
    else:
        confidence = 0.85

    summary_lines = [
        f"Projected peak CTL: {peak_ctl_projected} TSS/day "
        f"(from current {round(current_ctl, 1)}, target {round(raw_target, 1)})",
        f"Required ramp: +{round(needed_ramp, 1)} TSS/day per week "
        f"over {weeks_to_event} weeks",
        f"Feasibility: {feasibility}",
    ]
    if event_type:
        summary_lines.append(f"Event-specific focus: {event_type.replace('_', ' ')}")

    return {
        "event_date":          event_date.date().isoformat(),
        "event_name":          event_name,
        "event_type":          event_type,
        "current_ctl":         round(current_ctl, 1),
        "current_atl":         round(current_atl, 1),
        "peak_ctl_target":     peak_ctl_projected,         # what plan projects
        "peak_ctl_desired":    round(raw_target, 1),       # event-type ideal
        "planned_tsb_event":   20.0,   # standard taper target +15..+25
        "weeks_to_event":      weeks_to_event,
        "days_to_event":       days_to_event,
        "feasibility":         feasibility,
        "confidence":          confidence,
        "weeks":               [asdict(w) for w in weeks],
        "summary":             summary_lines,
        "method":              "reverse_periodization_v1",
    }
