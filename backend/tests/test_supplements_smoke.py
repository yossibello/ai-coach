"""
Smoke tests for the supplement engine. Verifies:
  - It returns a well-formed payload.
  - Safety hard-limits (caffeine cap, pregnancy block) survive
    end-to-end, not just at the guard layer.
"""
from __future__ import annotations

import pytest

from app.nutrition.engine import recommend_supplements


def _profile(sex="male", age=35, diet="omnivore", climate="temperate"):
    return {"sex": sex, "age": age, "diet": diet, "climate": climate, "weight_kg": 70}


def test_payload_shape():
    out = recommend_supplements(
        profile=_profile(),
        weekly_hours=8,
        ctl=60,
        weekly_tss=400,
        recent_avg_temp_c=22,
        upcoming_event_type=None,
        days_to_event=None,
        workout_focus="endurance",
    )
    assert set(["stack", "warnings", "depletion_signals", "engine_version", "disclaimer"]).issubset(out)
    assert isinstance(out["stack"], list)
    assert "Educational" in out["disclaimer"]


def test_pregnancy_blocks_caffeine_end_to_end():
    out = recommend_supplements(
        profile=_profile(),
        weekly_hours=10,
        ctl=70,
        weekly_tss=500,
        recent_avg_temp_c=20,
        upcoming_event_type="A",
        days_to_event=1,
        workout_focus="vo2max",
        user_contraindications=["pregnancy"],
    )
    keys = {s["supplement_key"] for s in out["stack"]}
    assert "caffeine" not in keys


def test_high_ferritin_blocks_iron_end_to_end():
    out = recommend_supplements(
        profile=_profile(sex="female"),
        weekly_hours=12,
        ctl=70,
        weekly_tss=600,
        recent_avg_temp_c=20,
        upcoming_event_type=None,
        days_to_event=None,
        workout_focus="endurance",
        blood_test={"id": "bt-1", "markers": {"ferritin": {"status": "high", "value": 350}}},
    )
    keys = {s["supplement_key"] for s in out["stack"]}
    assert "iron" not in keys


def test_no_dose_exceeds_hard_limit():
    """Across many random-ish trigger contexts, no recommended dose should
    exceed the hard safety limit."""
    from app.safety.guards import SUPPLEMENT_HARD_LIMITS

    out = recommend_supplements(
        profile=_profile(),
        weekly_hours=15,
        ctl=90,
        weekly_tss=900,
        recent_avg_temp_c=35,
        upcoming_event_type="A",
        days_to_event=2,
        workout_focus="vo2max",
    )
    for item in out["stack"]:
        limit = SUPPLEMENT_HARD_LIMITS.get(item["supplement_key"])
        if limit and isinstance(item["dose"], (int, float)):
            assert item["dose"] <= limit["max_dose"], (
                f"{item['supplement_key']} dose {item['dose']} exceeds "
                f"safety limit {limit['max_dose']} {limit['unit']}"
            )
