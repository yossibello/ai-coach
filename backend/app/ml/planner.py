"""
Weekly TSS constraint solver.

Given the athlete's current Performance Management Chart state (CTL/ATL/TSB)
and a planned 7-day sequence of workout types, compute per-day target TSS
values that:

  1. Sum to a safe weekly TSS budget (Friel safe-ramp rule):
       desired_weekly_tss = clamp(7·CTL · ramp_factor)
       where ramp_factor depends on phase and current TSB.

  2. Leave end-of-week projected TSB ≥ -25 (Coggan/Allen overreaching limit).
     Hard sessions are throttled if the projection violates this.

  3. Respect each workout type's baseline TSS shape — relative spending stays
     proportional to the cold_start library so a "vo2max" day still costs more
     than a "recovery" day.

Returns per-day TSS (and back-derived duration) plus a list of safety_notes
explaining any clamps applied.

References:
  - Coggan & Allen, *Training and Racing with a Power Meter* (3rd ed.),
    chap. 7-9 (PMC, ATL/CTL/TSB, ramp rate).
  - Friel J., *The Cyclist's Training Bible* (5th ed.), Chap. 7
    ("Building Fitness", safe weekly ramp ≈ 5–7 CTL units).
  - Plews & Buchheit (2013) — HRV-based intensity gating.
"""
from __future__ import annotations

from typing import Iterable

from app.ml.cold_start import WORKOUT_LIBRARY

# CTL EMA time constants (Banister / Coggan):
#   CTL alpha = 2 / (42+1)  (42-day exponential moving average)
#   ATL alpha = 2 / (7+1)   (7-day  exponential moving average)
_ALPHA_CTL = 2.0 / 43.0
_ALPHA_ATL = 2.0 / 8.0

# Safe weekly CTL ramp by phase (Friel: 5–7 TSS/week typical; up to 10 in build
# for advanced athletes; 0 or negative in taper/recovery).
_RAMP_BY_PHASE: dict[str, float] = {
    "base":          5.0,
    "base_build":    6.0,
    "build":         7.0,
    "peak":          3.0,   # consolidate, don't build
    "taper":        -8.0,   # actively shed fatigue
    "recovery_week": -10.0,
}

# Hard-effort workout types (used for safety throttling).
_HARD_TYPES = {"threshold", "vo2max", "sprint", "race", "sweetspot"}

# TSB lower bound; below this we throttle hard work (Coggan racing window
# typically requires TSB > -10; -25 is the absolute "overreaching cliff").
_TSB_MIN = -25.0


def _baseline_tss(workout_type: str) -> float:
    tmpl = WORKOUT_LIBRARY.get(workout_type)
    if not tmpl:
        return 50.0
    return float(tmpl.get("target_tss", 50.0))


def _project_pmc(
    ctl: float, atl: float, daily_tss: list[float]
) -> tuple[float, float, float]:
    """Return (CTL, ATL, TSB) at the END of the week given today's PMC and
    the next 7 days of TSS."""
    for tss in daily_tss:
        ctl += _ALPHA_CTL * (tss - ctl)
        atl += _ALPHA_ATL * (tss - atl)
    return ctl, atl, ctl - atl


def desired_weekly_tss(
    ctl: float, tsb: float, phase: str, *, max_increase_pct: float = 0.30
) -> float:
    """Compute target weekly TSS given current fitness, freshness, and phase.

    The starting point is "maintain current CTL" (which requires 7·CTL TSS
    per week, since CTL is itself a TSS-per-day EMA). We then add the
    phase-dependent safe ramp, and clamp the increase to max_increase_pct
    so we never jump more than 30% over last week's load.
    """
    maintain = max(7.0 * ctl, 100.0)  # never plan less than 100 TSS/week
    ramp = _RAMP_BY_PHASE.get(phase, 0.0)
    target = maintain + 7.0 * ramp

    # If athlete is fresh (TSB > +10), we can safely push the upper bound a bit.
    if tsb > 10:
        target *= 1.05
    # If athlete is fatigued (TSB < -15), pull back regardless of phase.
    if tsb < -15:
        target *= 0.85
    if tsb < -25:
        target *= 0.7

    # Cap absolute increase vs current maintenance level.
    return float(min(target, maintain * (1.0 + max_increase_pct)))


def solve_week(
    workouts: list[str],
    *,
    ctl: float,
    atl: float,
    tsb: float,
    phase: str,
    hrv_z: float | None = None,
) -> tuple[list[float], list[str]]:
    """Compute per-day TSS for a planned 7-workout week.

    Args:
        workouts: list of workout-type names, length 7 (one per day; use
            "rest"/"recovery" for off days).
        ctl, atl, tsb: current PMC values.
        phase: periodization phase name.
        hrv_z: latest HRV z-score vs baseline; if < -1.0, hard sessions get
            an extra 30% throttle (Plews & Buchheit 2013).

    Returns:
        (daily_tss, safety_notes)
    """
    notes: list[str] = []
    n = len(workouts)
    if n == 0:
        return [], notes

    # 1. Baseline TSS per day from the workout library.
    base = [_baseline_tss(w) for w in workouts]

    # 2. Compute desired weekly total and the scale factor.
    target = desired_weekly_tss(ctl, tsb, phase)
    base_total = sum(base) or 1.0
    scale = target / base_total
    # Don't over-shrink (preserves intensity feel) or over-grow (preserves
    # recovery between hard days).
    scale = max(0.6, min(scale, 1.4))
    if abs(scale - 1.0) > 0.05:
        notes.append(
            f"Weekly TSS scaled by {scale:.2f}× to hit target "
            f"{target:.0f} TSS (phase={phase}, current CTL={ctl:.0f})."
        )
    daily = [b * scale for b in base]

    # 3. HRV throttle for hard sessions.
    if hrv_z is not None and hrv_z < -1.0:
        for i, w in enumerate(workouts):
            if w in _HARD_TYPES:
                daily[i] *= 0.7
        notes.append(
            f"Hard sessions throttled 30% due to suppressed HRV (z={hrv_z:.2f})."
        )

    # 4. TSB safety projection — iteratively shrink hardest day if end-of-week
    #    TSB falls below the cliff.
    for _ in range(5):
        end_ctl, end_atl, end_tsb = _project_pmc(ctl, atl, daily)
        if end_tsb >= _TSB_MIN:
            break
        # Find the hardest scheduled day and trim 15% off it.
        hard_indices = [i for i, w in enumerate(workouts) if w in _HARD_TYPES]
        if not hard_indices:
            break
        worst = max(hard_indices, key=lambda i: daily[i])
        daily[worst] *= 0.85
    end_ctl, end_atl, end_tsb = _project_pmc(ctl, atl, daily)
    if end_tsb < _TSB_MIN:
        notes.append(
            f"Even after throttling, projected end-of-week TSB={end_tsb:.0f} "
            "is below safe range. Consider taking an extra rest day."
        )

    return daily, notes


def tss_to_duration_minutes(tss: float, intensity_factor: float) -> int:
    """Invert ``TSS = (duration_h · IF² · 100)`` to recover duration in minutes.

    Rounded and clamped to a sane training window [20, 360].
    """
    if intensity_factor <= 0:
        return 60
    duration_h = tss / (100.0 * (intensity_factor ** 2))
    return int(max(20, min(360, round(duration_h * 60))))
