"""
Strength session scheduler.

Rules:
  1. Never on the same day as a hard cycling session (threshold, vo2max, race, sprint)
  2. Never the day BEFORE a hard cycling session (legs need to be fresh)
  3. Never the day AFTER a hard cycling session (still recovering)
  4. Among valid gap days, prefer those FURTHEST from any hard session
  5. When no gap days exist (all 7 days have cycling), add as addon:
       recovery days first, then easy, then endurance — never hard days
  6. GTG co-exists with any non-hard day
  7. Never exceed the approach's sessions_per_week target

Key design: the plan is SPARSE — a 4-day/week athlete has only 4 entries,
so rest days between training sessions have no entry in the list. This
scheduler explicitly detects those gap offsets and prefers them over
stacking gym on a cycling day.
"""
from __future__ import annotations

from app.strength.approaches import APPROACHES, StrengthSession

HARD_TYPES = {"threshold", "vo2max", "race", "sprint"}
EASY_TYPES = {"recovery", "easy", "endurance", "rest"}

# Within easy days, prefer lower-fatigue types for gym addon.
_EASY_PRIORITY = {"recovery": 0, "easy": 1, "rest": 2, "endurance": 3}


def _is_hard(day: dict) -> bool:
    return day.get("workout_type", "rest") in HARD_TYPES


def _is_easy_or_rest(day: dict) -> bool:
    return day.get("workout_type", "rest") in EASY_TYPES


def _is_day_before_hard(plan_by_offset: dict[int, dict], offset: int) -> bool:
    """True if offset+1 is a hard cycling session."""
    nxt = plan_by_offset.get(offset + 1)
    return nxt is not None and _is_hard(nxt)


def _is_day_after_hard(plan_by_offset: dict[int, dict], offset: int) -> bool:
    """True if offset-1 is a hard cycling session (still recovering)."""
    prv = plan_by_offset.get(offset - 1)
    return prv is not None and _is_hard(prv)


def _dist_to_nearest_hard(plan_by_offset: dict[int, dict], offset: int) -> int:
    """Calendar-day distance from offset to the nearest hard session."""
    hard_offsets = [o for o, d in plan_by_offset.items() if _is_hard(d)]
    if not hard_offsets:
        return 99
    return min(abs(offset - h) for h in hard_offsets)


def _strength_workout_entry(day_offset: int, session: StrengthSession, approach_key: str) -> dict:
    return {
        "day_offset":       day_offset,
        "workout_type":     "strength",
        "duration_minutes": session.duration_minutes,
        "target_tss":       _estimate_tss(session),
        "description":      session.notes,
        "structure":        [],
        "rationale":        _rationale(session, approach_key),
        "is_strength":      True,
        "strength_session": session.to_dict(),
    }


def _estimate_tss(session: StrengthSession) -> int:
    if "GTG" in session.name or session.duration_minutes <= 10:
        return 10
    if "Maintenance" in session.phase_label:
        return 20
    if "Anatomical" in session.phase_label:
        return 35
    if "Maximum" in session.phase_label:
        return 50
    return 30


def _rationale(session: StrengthSession, approach_key: str) -> str:
    labels = {
        "friel":             "Friel periodized strength",
        "minimum_dose":      "Minimum effective dose",
        "grease_the_groove": "Grease the Groove daily practice",
    }
    return f"{labels.get(approach_key, 'Strength')} · {session.phase_label}"


def add_strength_to_plan(
    weekly_plan: list[dict],
    phase: str,
    approach_key: str = "friel",
    max_sessions: int | None = None,
) -> list[dict]:
    """
    Insert strength sessions into a cycling weekly plan.

    Args:
        weekly_plan:   Sparse list (only training days, not all 7 days).
        phase:         Cycling periodization phase.
        approach_key:  'friel' | 'minimum_dose' | 'grease_the_groove'.
        max_sessions:  Hard cap on sessions this week.

    Returns:
        weekly_plan sorted by day_offset with strength sessions inserted/annotated.
    """
    approach = APPROACHES.get(approach_key)
    if approach is None:
        return weekly_plan

    sessions = approach.get_sessions(phase)
    if not sessions or not weekly_plan:
        return weekly_plan

    is_gtg = approach_key == "grease_the_groove"

    plan_by_offset: dict[int, dict] = {d["day_offset"]: d for d in weekly_plan}
    occupied = set(plan_by_offset.keys())
    max_offset = max(occupied)

    # ── Gap days (true rest — no cycling entry) ───────────────────────────
    # Search up to one day past the last training day, capped at day 7.
    gap_offsets = [
        d for d in range(0, min(max_offset + 2, 8))
        if d not in occupied
    ]

    # Filter gaps: exclude day-before-hard AND day-after-hard.
    # Then sort by DISTANCE from nearest hard session (furthest first).
    valid_gaps = []
    for off in gap_offsets:
        if is_gtg:
            valid_gaps.append(off)
            continue
        if _is_day_before_hard(plan_by_offset, off):
            continue
        if _is_day_after_hard(plan_by_offset, off):
            continue
        valid_gaps.append(off)

    valid_gaps.sort(key=lambda o: _dist_to_nearest_hard(plan_by_offset, o), reverse=True)

    # ── Cycling-day addons (when gaps run out or don't exist) ─────────────
    # Sort easy days: recovery → easy → endurance (lowest fatigue cost first).
    # Hard days are always excluded.
    easy_days: list[tuple[int, int]] = []   # (priority, offset)
    last_resort: list[tuple[int, int]] = []  # non-hard non-easy (sweetspot, tempo…)

    for off in sorted(occupied):
        day = plan_by_offset[off]
        if _is_hard(day):
            continue
        if not is_gtg:
            if _is_day_before_hard(plan_by_offset, off):
                continue
            if _is_day_after_hard(plan_by_offset, off):
                continue
        wt = day.get("workout_type", "rest")
        if _is_easy_or_rest(day):
            easy_days.append((_EASY_PRIORITY.get(wt, 3), off))
        else:
            last_resort.append((0, off))

    easy_days.sort()
    last_resort.sort()

    # Final candidate list: (source, offset)
    candidates: list[tuple[str, int]] = (
        [("gap", o) for o in valid_gaps] +
        [("easy", o) for _, o in easy_days] +
        [("other", o) for _, o in last_resort]
    )

    if not candidates:
        return weekly_plan

    cap = min(len(sessions), max_sessions) if max_sessions is not None else len(sessions)
    slots = candidates[:cap]
    session_list = sessions[:len(slots)]

    for (source, off), session in zip(slots, session_list):
        if is_gtg:
            day = plan_by_offset.get(off)
            if day is not None:
                day["gtg_practice"] = session.to_dict()
                day["rationale"] = (
                    (day.get("rationale") or "") +
                    " · GTG: 5 KB swings + 5 goblet squats every 1-2h throughout the day."
                )
        elif source == "gap":
            new_entry = _strength_workout_entry(off, session, approach_key)
            weekly_plan.append(new_entry)
            plan_by_offset[off] = new_entry
        else:
            day = plan_by_offset[off]
            day["strength_addon"] = session.to_dict()
            day["rationale"] = (
                (day.get("rationale") or "") +
                f" · Add {session.name} ({session.duration_minutes} min) after this ride."
            )

    weekly_plan.sort(key=lambda d: d["day_offset"])
    return weekly_plan
