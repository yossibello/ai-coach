"""
Compute depletion / context signals from training load + profile + blood markers.

Signals are floats in [0, 1] representing how strongly a need is indicated.
The supplement engine consumes these to decide which catalog items to recommend.

Signal sources, in order of authority:
  1. Blood markers (when available) — direct measurement, highest weight
  2. Training load proxies (CTL, weekly hours, recent TSS) — indirect
  3. Profile flags (sex, age, diet, climate) — context only
"""
from __future__ import annotations

from typing import Optional

from app.nutrition.markers import status_for_value


# ── Tunable thresholds (athletic literature) ─────────────────────────────────
HIGH_VOLUME_HOURS_PER_WEEK = 12      # >12 h/wk = high volume
VERY_HIGH_VOLUME_HOURS_PER_WEEK = 18
HIGH_CTL = 75
HOT_CLIMATE_TEMP_C = 26
LONG_SESSION_MINUTES = 150


def compute_signals(
    *,
    blood_markers: dict,                # {marker_key: {"value": float, "status": str}}
    profile: dict,                      # {sex, age, diet, climate, training_days_per_week, ...}
    weekly_hours: float,                # last 4-week average hours/week
    ctl: float,
    weekly_tss: float,
    recent_avg_temp_c: Optional[float],
    upcoming_event_type: Optional[str], # "long_road", "crit", "tt", "stage_race", None
    days_to_event: Optional[int],
    workout_focus: Optional[str],       # "vo2max", "endurance", "threshold", ...
    has_recent_blood_test: bool,
) -> dict[str, float]:
    """
    Returns a dict of {signal_key: strength_0_1}.

    Signal keys must match the `triggers` lists in supplements.SUPPLEMENTS
    (see app/nutrition/supplements.py).
    """
    s: dict[str, float] = {}

    # ── Blood-marker driven signals ──────────────────────────────────────────
    bm = blood_markers or {}

    def status(key: str) -> str:
        m = bm.get(key)
        return m.get("status") if m else "unknown"

    # Iron — hierarchy: critical > deficient > subclinical
    if status("ferritin") in ("critical_low", "low"):
        s["iron_deficient"] = 1.0
    elif status("ferritin") == "suboptimal":
        s["iron_subclinical_low"] = 0.7
    if status("ferritin") in ("high", "critical_high"):
        s["ferritin_high"] = 1.0

    if status("vitamin_d") in ("critical_low", "low"):
        s["vitamin_d_deficient"] = 1.0
    elif status("vitamin_d") == "suboptimal":
        s["vitamin_d_low"] = 0.6

    if status("vitamin_b12") in ("critical_low", "low"):
        s["b12_low"] = 1.0
    elif status("vitamin_b12") == "suboptimal":
        s["b12_low"] = 0.5

    if status("magnesium") in ("critical_low", "low") or status("magnesium_rbc") in ("critical_low", "low"):
        s["magnesium_low"] = 1.0
    elif status("magnesium") == "suboptimal" or status("magnesium_rbc") == "suboptimal":
        s["magnesium_low"] = 0.5

    if status("zinc") in ("critical_low", "low"):
        s["zinc_low"] = 1.0
    elif status("zinc") == "suboptimal":
        s["zinc_low"] = 0.5

    if status("omega3_index") in ("critical_low", "low", "suboptimal"):
        s["omega3_low"] = 0.8

    if status("crp") in ("high", "critical_high"):
        s["high_inflammation"] = min(1.0, 0.6 + 0.1 * (bm["crp"]["value"] - 3))

    if status("cortisol") == "high" or status("testosterone_total") == "low":
        s["high_cortisol"] = 0.8

    # ── Training-load driven signals ─────────────────────────────────────────
    if weekly_hours >= VERY_HIGH_VOLUME_HOURS_PER_WEEK:
        s["very_high_volume"] = 1.0
        s["high_weekly_volume"] = 1.0
    elif weekly_hours >= HIGH_VOLUME_HOURS_PER_WEEK:
        s["high_weekly_volume"] = 0.8

    if ctl >= HIGH_CTL:
        s["high_weekly_volume"] = max(s.get("high_weekly_volume", 0), 0.7)

    if recent_avg_temp_c is not None and recent_avg_temp_c >= HOT_CLIMATE_TEMP_C:
        s["hot_climate"] = min(1.0, (recent_avg_temp_c - HOT_CLIMATE_TEMP_C) / 10 + 0.6)

    # ── Workout / event context ──────────────────────────────────────────────
    if workout_focus == "vo2max":
        s["vo2max_focus"] = 1.0
        s["high_intensity_focus"] = 1.0
    if workout_focus in ("threshold", "sweetspot"):
        s["high_intensity_focus"] = 0.7

    if upcoming_event_type and days_to_event is not None and 0 <= days_to_event <= 7:
        s["pre_a_event"] = 1.0
        if upcoming_event_type in ("crit", "tt"):
            s["short_max_effort_event"] = 1.0
            s["crit_race_focus"] = 1.0
        # Long endurance events: extra carb-loading + iron + electrolyte focus.
        if upcoming_event_type in (
            "gran_fondo", "long_road", "stage_race", "climbing_camp",
            "mtb_marathon", "ultra_endurance", "triathlon_140_6",
        ):
            s["long_endurance_event"] = 1.0
            s["carb_loading_focus"] = 1.0
        # Multi-day / climbing-heavy events: protect quad muscle damage + iron.
        if upcoming_event_type in ("stage_race", "climbing_camp", "ultra_endurance"):
            s["muscle_damage_focus"] = 1.0
            s["multi_day_recovery_focus"] = 1.0

    if upcoming_event_type and days_to_event is not None and 0 <= days_to_event <= 1:
        s["pre_key_session"] = 1.0

    # ── Profile signals ──────────────────────────────────────────────────────
    diet = (profile.get("diet") or "").lower()
    if diet == "vegan":
        s["vegan_diet"] = 1.0
        s["vegetarian_diet"] = 1.0
    elif diet == "vegetarian":
        s["vegetarian_diet"] = 1.0

    age = profile.get("age")
    if age and age >= 40:
        s["masters_athlete"] = min(1.0, (age - 40) / 30 + 0.5)

    climate = (profile.get("climate") or "").lower()
    if climate in ("northern_winter", "indoor_only"):
        s["winter_indoor_training"] = 0.8

    if (profile.get("recent_illness_count_3m") or 0) >= 2:
        s["frequent_illness"] = 0.8

    # ── Safety signal ────────────────────────────────────────────────────────
    if not has_recent_blood_test and weekly_hours >= HIGH_VOLUME_HOURS_PER_WEEK:
        s["no_blood_test_with_high_load"] = 1.0

    return s
