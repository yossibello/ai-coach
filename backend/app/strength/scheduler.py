"""
Strength session scheduler.

Takes a completed cycling weekly_plan (list of 7 day dicts) and inserts
strength sessions from the chosen approach, respecting these rules:

  1. Never on the same day as a hard cycling session (threshold, vo2max, race, sprint)
  2. Never the day BEFORE a hard cycling session (legs need to be fresh)
  3. Prefer rest days, then easy/recovery days
  4. For GTG: mark applicable days as GTG-active (GTG co-exists with easy rides)
  5. Never exceed the approach's sessions_per_week target

The weekly_plan days that were already 'rest' days keep their rest status;
the scheduler only promotes days that have no cycling or easy cycling.
"""
from __future__ import annotations

from app.strength.approaches import APPROACHES, StrengthSession

HARD_TYPES = {"threshold", "vo2max", "race", "sprint"}
EASY_TYPES = {"recovery", "easy", "endurance", "rest"}


def _is_hard(day: dict) -> bool:
    return day.get("workout_type", "rest") in HARD_TYPES


def _is_easy_or_rest(day: dict) -> bool:
    return day.get("workout_type", "rest") in EASY_TYPES


def _day_before_hard(plan: list[dict], idx: int) -> bool:
    """True if plan[idx+1] is a hard cycling session."""
    if idx + 1 >= len(plan):
        return False
    return _is_hard(plan[idx + 1])


def _candidate_days(plan: list[dict], approach_key: str) -> list[int]:
    """
    Return day indices eligible for a strength session, ordered by preference:
      1. Rest days (no cycling at all)
      2. Recovery / easy ride days
    Hard days and the day before a hard day are excluded.
    GTG can co-exist with easy days so it uses a looser filter.
    """
    is_gtg = approach_key == "grease_the_groove"

    preferred: list[int] = []
    acceptable: list[int] = []

    for i, day in enumerate(plan):
        wt = day.get("workout_type", "rest")
        if _is_hard(day):
            continue
        if _day_before_hard(plan, i) and not is_gtg:
            continue
        if wt == "rest":
            preferred.append(i)
        elif _is_easy_or_rest(day):
            acceptable.append(i)
        elif is_gtg:
            # GTG can go on any non-hard day
            acceptable.append(i)

    return preferred + acceptable


def _strength_workout_entry(
    day_offset: int,
    session: StrengthSession,
    approach_key: str,
) -> dict:
    """Build the weekly_plan dict for a strength session."""
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
    """
    Rough TSS equivalent for strength work.
    GTG is negligible. Maintenance ≈ 20. AA ≈ 35. MS ≈ 50.
    """
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
) -> list[dict]:
    """
    Insert strength sessions into a 7-day cycling weekly plan.

    Args:
        weekly_plan:  List of 7 day dicts (day_offset 0-6).
        phase:        Cycling periodization phase (base_build, build, peak, recovery_week).
        approach_key: One of 'friel', 'minimum_dose', 'grease_the_groove'.

    Returns:
        The same weekly_plan with strength sessions added.
        Days that already have a cycling workout get a combined entry;
        rest days are replaced with the strength session.
    """
    approach = APPROACHES.get(approach_key)
    if approach is None:
        return weekly_plan

    sessions = approach.get_sessions(phase)
    if not sessions:
        return weekly_plan

    candidates = _candidate_days(weekly_plan, approach_key)
    if not candidates:
        return weekly_plan

    # Pick slots — up to len(sessions) candidate days
    slots = candidates[: len(sessions)]

    # Deduplicate session list if approach returns the same session twice (min_dose)
    session_list = sessions[: len(slots)]

    for idx, session in zip(slots, session_list):
        day = weekly_plan[idx]
        wt = day.get("workout_type", "rest")

        if approach_key == "grease_the_groove":
            # GTG: annotate the existing day rather than replacing it
            day["gtg_practice"] = session.to_dict()
            day["rationale"] = (
                (day.get("rationale") or "") +
                " · GTG: 5 KB swings + 5 goblet squats every 1-2h throughout the day."
            )
        elif wt == "rest":
            # Replace the rest day with the strength session
            weekly_plan[idx] = _strength_workout_entry(day["day_offset"], session, approach_key)
        else:
            # Easy ride day — add a note that strength follows
            day["strength_addon"] = session.to_dict()
            day["rationale"] = (
                (day.get("rationale") or "") +
                f" · Add {session.name} ({session.duration_minutes} min) after this ride."
            )

    return weekly_plan
