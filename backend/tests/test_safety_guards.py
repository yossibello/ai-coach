"""
Tests for the safety guard layer (`app.safety.guards`).

These prove that the system will refuse to:
  - Recommend a high-intensity workout when the athlete is too fatigued.
  - Suggest a workout TSS far above the athlete's current fitness (CTL).
  - Allow weekly load to ramp faster than +10% (injury risk).
  - Recommend an unsafe supplement dose.
  - Recommend a supplement contraindicated by user condition.
  - Recommend two supplements that are dangerous in combination.
"""
from __future__ import annotations

import pytest

from app.safety.guards import (
    apply_workout_safety,
    apply_weekly_plan_safety,
    apply_supplement_safety,
    check_weekly_ramp_safe,
    MAX_DAILY_TSS,
    MAX_SINGLE_SESSION_MINUTES,
    HARD_RECOVERY_TSB,
    SUPPLEMENT_HARD_LIMITS,
)


# ═══════════════════════ Workout safety ═══════════════════════════════════════

def test_deeply_fatigued_athlete_forced_to_recovery():
    workout = {"workout_type": "vo2max", "duration_minutes": 90, "target_tss": 150}
    safe, notes = apply_workout_safety(workout, ctl=60, tsb=-40)
    assert safe["workout_type"] == "recovery"
    assert safe["target_tss"] <= 30
    assert safe["duration_minutes"] <= 45
    assert any("Forced recovery" in n for n in notes)


def test_moderately_fatigued_athlete_blocked_from_threshold():
    workout = {"workout_type": "threshold", "duration_minutes": 75, "target_tss": 90}
    safe, notes = apply_workout_safety(workout, ctl=60, tsb=-20)
    assert safe["workout_type"] == "sweetspot"
    assert any("Downgraded threshold" in n for n in notes)


def test_session_duration_capped_at_six_hours():
    workout = {"workout_type": "endurance", "duration_minutes": 600, "target_tss": 200}
    safe, _ = apply_workout_safety(workout, ctl=80, tsb=10)
    assert safe["duration_minutes"] == MAX_SINGLE_SESSION_MINUTES


def test_session_tss_capped_to_ctl_multiple():
    """Don't suggest a 300 TSS ride to a beginner with CTL=20."""
    workout = {"workout_type": "long_ride", "duration_minutes": 300, "target_tss": 300}
    safe, notes = apply_workout_safety(workout, ctl=20, tsb=5)
    # 2.5 × CTL = 50
    assert safe["target_tss"] <= 50
    assert any("× CTL" in n for n in notes)


def test_per_type_tss_ceiling_enforced():
    """A 'recovery' workout cannot be 200 TSS."""
    workout = {"workout_type": "recovery", "duration_minutes": 45, "target_tss": 200}
    safe, _ = apply_workout_safety(workout, ctl=80, tsb=5)
    assert safe["target_tss"] <= 40


def test_absolute_tss_ceiling():
    workout = {"workout_type": "race", "duration_minutes": 360, "target_tss": 600}
    safe, _ = apply_workout_safety(workout, ctl=120, tsb=10)
    assert safe["target_tss"] <= MAX_DAILY_TSS


def test_well_rested_workout_unchanged():
    workout = {"workout_type": "endurance", "duration_minutes": 90, "target_tss": 80}
    safe, notes = apply_workout_safety(workout, ctl=60, tsb=8)
    assert safe["workout_type"] == "endurance"
    assert safe["target_tss"] == 80
    assert notes == []


def test_none_workout_passes_through():
    safe, notes = apply_workout_safety(None, ctl=50, tsb=0)
    assert safe is None and notes == []


# ═══════════════════════ Weekly ramp guard ════════════════════════════════════

def test_ramp_under_10pct_is_safe():
    ok, msg = check_weekly_ramp_safe(last_week_tss=400, proposed_week_tss=430)
    assert ok and msg is None


def test_ramp_over_10pct_blocked():
    ok, msg = check_weekly_ramp_safe(last_week_tss=400, proposed_week_tss=500)
    assert not ok and "ramp" in msg.lower()


def test_ramp_safe_when_no_history():
    ok, _ = check_weekly_ramp_safe(last_week_tss=0, proposed_week_tss=400)
    assert ok


def test_weekly_plan_scaled_down_when_ramp_too_aggressive():
    plan = [
        {"workout_type": "endurance", "duration_minutes": 120, "target_tss": 150},
        {"workout_type": "threshold", "duration_minutes": 90,  "target_tss": 120},
        {"workout_type": "long_ride", "duration_minutes": 240, "target_tss": 250},
    ]
    safe_plan, notes = apply_weekly_plan_safety(
        plan, last_week_tss=200, ctl=50, tsb=5
    )
    total = sum(w["target_tss"] for w in safe_plan)
    assert total <= 200 * 1.10 + 0.5
    assert any("scaled down" in n for n in notes)


# ═══════════════════════ Supplement safety ════════════════════════════════════

def test_caffeine_dose_capped_at_safety_limit():
    stack = [{
        "supplement_key": "caffeine",
        "label": "Caffeine",
        "dose": 12.0,                # absurd
        "dose_unit": "mg/kg",
        "score": 0.9,
    }]
    safe, warnings = apply_supplement_safety(stack)
    assert safe[0]["dose"] == SUPPLEMENT_HARD_LIMITS["caffeine"]["max_dose"]
    assert any(w["warning_key"] == "clamped_caffeine" for w in warnings)


def test_iron_blocked_for_high_ferritin_user():
    stack = [{"supplement_key": "iron", "label": "Iron", "dose": 30, "dose_unit": "mg", "score": 0.7}]
    safe, warnings = apply_supplement_safety(stack, user_conditions=["ferritin_high"])
    assert safe == []
    assert any(w["warning_key"] == "blocked_iron" for w in warnings)


def test_caffeine_blocked_during_pregnancy():
    stack = [{"supplement_key": "caffeine", "label": "Caffeine", "dose": 3, "dose_unit": "mg/kg", "score": 0.9}]
    safe, warnings = apply_supplement_safety(stack, user_conditions=["pregnancy"])
    assert safe == []
    assert any("pregnancy" in w["message"] for w in warnings)


def test_creatine_blocked_for_kidney_disease():
    stack = [{"supplement_key": "creatine_monohydrate", "label": "Creatine",
              "dose": 5, "dose_unit": "g", "score": 0.8}]
    safe, _ = apply_supplement_safety(stack, user_conditions=["chronic_kidney_disease"])
    assert safe == []


def test_dangerous_pair_caffeine_plus_bicarb_drops_lower_score():
    stack = [
        {"supplement_key": "caffeine", "label": "Caffeine",
         "dose": 3, "dose_unit": "mg/kg", "score": 0.9},
        {"supplement_key": "sodium_bicarbonate", "label": "Sodium Bicarb",
         "dose": 0.2, "dose_unit": "g/kg", "score": 0.6},
    ]
    safe, warnings = apply_supplement_safety(stack)
    keys = {s["supplement_key"] for s in safe}
    assert "caffeine" in keys                        # higher score retained
    assert "sodium_bicarbonate" not in keys
    assert any("Removed sodium_bicarbonate" in w["message"] for w in warnings)


def test_safe_stack_passes_through_unchanged():
    stack = [{"supplement_key": "vitamin_d3", "label": "Vit D3",
              "dose": 2000, "dose_unit": "IU", "score": 0.7}]
    safe, warnings = apply_supplement_safety(stack)
    assert safe[0]["dose"] == 2000
    assert warnings == []
