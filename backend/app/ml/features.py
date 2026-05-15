"""
Feature engineering: thin adapter that converts ORM Activity / AthleteProfile
objects into the RAW dict shape expected by `app.ml.norm` encoders.

All actual normalization lives in `norm.py` so that training (synthetic data)
and inference (live DB) share a single source of truth.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from app.models.activity import Activity
from app.models.user import AthleteProfile
from app.ml.norm import (
    encode_activity_row,
    encode_profile_row,
    WORKOUT_TYPE_IDX,
    GOAL_TYPE_IDX,
)


def encode_activity(
    act: Activity,
    profile: AthleteProfile | None,
    ctl: float,
    atl: float,
    tsb: float,
    days_since_last: int,
    *,
    hrv_z: float = 0.0,
    rhr_delta: float = 0.0,
    sleep_score: float = 50.0,
    body_battery: float = 50.0,
) -> np.ndarray:
    """ORM Activity → 1-D float32 vector of length ACTIVITY_DIM.

    Health params (HRV z-score, RHR delta, sleep score, body battery) are
    optional kwargs — the caller is expected to look them up from HealthMetric
    on the activity's date. Defaults centre each signal in its bound, which
    is equivalent to telling the model "no health data available".
    """
    ftp = (profile.ftp if profile and profile.ftp else 200) or 200
    max_hr = (profile.max_hr if profile and profile.max_hr else 190) or 190
    resting_hr = (profile.resting_hr if profile and profile.resting_hr else 55) or 55

    raw = {
        "date": act.date,
        "duration_seconds": act.duration_seconds,
        "distance_meters": act.distance_meters,
        "elevation_gain_meters": act.elevation_gain_meters,
        "avg_power": act.avg_power,
        "normalized_power": act.normalized_power,
        "max_power": act.max_power,
        "intensity_factor": act.intensity_factor,
        "tss": act.tss,
        "variability_index": act.variability_index,
        "avg_hr": act.avg_hr,
        "max_hr": act.max_hr,
        "hr_drift": act.hr_drift,
        "aerobic_efficiency": act.aerobic_efficiency,
        "avg_cadence": act.avg_cadence,
        "time_in_zones": act.time_in_zones,
        "temperature_c": act.temperature_c,
        "humidity_pct": act.humidity_pct,
        "wind_speed_kmh": act.wind_speed_kmh,
        "perceived_exertion": act.perceived_exertion,
        "workout_type": act.workout_type,
        # Health & recovery (passed through to encoder via act dict)
        "hrv_z": hrv_z,
        "rhr_delta": rhr_delta,
        "sleep_score": sleep_score,
        "body_battery": body_battery,
        # Power curve (W/kg from stream extraction; 0 if no power meter)
        "pc_5s_wkg":   act.pc_5s_wkg   or 0.0,
        "pc_1min_wkg": act.pc_1min_wkg  or 0.0,
        "pc_5min_wkg": act.pc_5min_wkg  or 0.0,
        "pc_20min_wkg": act.pc_20min_wkg or 0.0,
    }
    return encode_activity_row(
        raw, ftp=ftp, max_hr_athlete=max_hr, resting_hr=resting_hr,
        ctl=ctl, atl=atl, tsb=tsb, days_since_last=days_since_last,
    )


def encode_profile(profile: AthleteProfile | None, current_ftp: float = 200) -> np.ndarray:
    """ORM AthleteProfile → 1-D float32 vector of length PROFILE_DIM."""
    p = profile

    days_to_event = 365.0
    if p and p.goal_event_date:
        dte = (p.goal_event_date.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).days
        days_to_event = max(0, dte)

    raw = {
        "age": p.age if p else None,
        "weight_kg": p.weight_kg if p else None,
        "height_cm": p.height_cm if p else None,
        "sex": p.sex if p else None,
        "ftp": current_ftp,
        "athlete_max_hr": p.max_hr if p else None,
        "resting_hr": p.resting_hr if p else None,
        "cycling_experience_years": p.cycling_experience_years if p else None,
        "primary_goal": p.primary_goal if p else None,
        "days_to_event": days_to_event,
        "training_days_per_week": p.training_days_per_week if p else None,
    }
    return encode_profile_row(raw)


__all__ = ["encode_activity", "encode_profile", "WORKOUT_TYPE_IDX", "GOAL_TYPE_IDX"]
