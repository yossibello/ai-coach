"""
Single source of truth for feature normalization.

Both the inference path (features.py — encodes ORM Activity objects from the DB)
and the synthetic-data generator (ml/training/generate_synthetic.py — produces
parquet for pre-training) MUST go through the encoders defined here, otherwise
the model's training distribution will not match what it sees at inference time.

Everything works on plain dicts of RAW values (watts, bpm, seconds, °C, …).
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from app.ml.model import ACTIVITY_FEATURES, PROFILE_FEATURES, ACTIVITY_DIM, PROFILE_DIM


# ─── Categorical maps ─────────────────────────────────────────────────────────
WORKOUT_TYPES: list[str] = [
    "recovery", "easy", "endurance", "tempo", "sweetspot",
    "threshold", "vo2max", "sprint", "race", "long_ride",
]
WORKOUT_TYPE_IDX: dict[str, int] = {n: i for i, n in enumerate(WORKOUT_TYPES)}

GOAL_TYPES: list[str] = [
    "general_fitness", "ftp_improvement", "weight_loss",
    "event_specific", "gran_fondo", "criterium",
    "climbing", "triathlon",
]
GOAL_TYPE_IDX: dict[str, int] = {n: i for i, n in enumerate(GOAL_TYPES)}


# ─── Normalization bounds (raw value → [0, 1]) ────────────────────────────────
# Wide enough to cover real-world amateur AND elite values without clipping
# the typical training distribution. Used by both train and inference.
NORM_BOUNDS: dict[str, tuple[float, float]] = {
    # Load
    "tss":               (0.0, 250.0),
    "ctl":               (0.0, 150.0),
    "atl":               (0.0, 200.0),
    "tsb":               (-100.0, 100.0),
    "duration_h":        (0.0, 8.0),
    "distance_km":       (0.0, 300.0),
    "elevation_km":      (0.0, 5.0),
    # Power (as fraction of FTP)
    "avg_power_pct_ftp": (0.0, 1.6),
    "np_pct_ftp":        (0.0, 1.6),
    "intensity_factor":  (0.0, 1.5),
    "max_power_pct_ftp": (0.0, 3.0),
    "variability_index": (1.0, 1.5),
    # HR (as fraction of HRmax)
    "avg_hr_pct_max":    (0.4, 1.0),
    "max_hr_pct_max":    (0.4, 1.05),
    "hr_drift":          (-5.0, 25.0),
    "aerobic_efficiency":(0.0, 4.0),     # watts / bpm  (1.5–3.5 typical)
    # Cadence
    "avg_cadence":       (50.0, 120.0),
    # Environment
    "temp_c":            (-10.0, 45.0),
    "humidity_pct":      (0.0, 100.0),
    "wind_speed_kmh":    (0.0, 60.0),
    # Temporal
    "days_since_last":   (0.0, 21.0),
    # Subjective
    "rpe":               (1.0, 10.0),
    # Profile
    "age":               (15.0, 80.0),
    "weight_kg":         (40.0, 120.0),
    "height_cm":         (140.0, 210.0),
    "ftp":               (50.0, 600.0),
    "max_hr_athlete":    (140.0, 220.0),
    "resting_hr":        (30.0, 90.0),
    "experience_years":  (0.0, 30.0),
    "days_to_event":     (0.0, 730.0),
}


def n01(value: float, key: str) -> float:
    """Min-max clamp to [0, 1] using the canonical bounds for `key`."""
    lo, hi = NORM_BOUNDS[key]
    if hi == lo:
        return 0.0
    return float(max(0.0, min(1.0, (value - lo) / (hi - lo))))


# ─── Cyclical (sin/cos) encoding kept in [-1, 1] consistently ────────────────
def _sincos(value: float, period: float) -> tuple[float, float]:
    a = 2.0 * math.pi * value / period
    return math.sin(a), math.cos(a)


def encode_activity_row(
    act: Mapping,
    ftp: float,
    max_hr_athlete: float,
    resting_hr: float,
    ctl: float,
    atl: float,
    tsb: float,
    days_since_last: float,
) -> np.ndarray:
    """
    Encode one activity (raw values) → 1-D vector of length ACTIVITY_DIM.

    `act` is a mapping with raw fields. Missing fields default sensibly.
    The CTL/ATL/TSB/days_since_last are passed in because they are STATE
    derived from the surrounding sequence, not the activity itself.
    """
    ftp = max(float(ftp), 1.0)
    max_hr_athlete = max(float(max_hr_athlete), 1.0)

    # ── Field accessors with safe defaults ────────────────────────────────
    def g(key, default=0.0):
        v = act.get(key, default)
        return default if v is None else v

    duration_h   = g("duration_seconds", 0.0) / 3600.0
    distance_km  = g("distance_meters", 0.0) / 1000.0
    elevation_km = g("elevation_gain_meters", 0.0) / 1000.0

    avg_power = g("avg_power")
    np_power  = g("normalized_power") or avg_power
    max_power = g("max_power")
    raw_if    = g("intensity_factor") or (avg_power / ftp if avg_power else 0.0)
    vi        = g("variability_index") or 1.0

    avg_hr = g("avg_hr")
    max_hr_ride = g("max_hr") or avg_hr
    hr_drift = g("hr_drift")
    aero_eff = g("aerobic_efficiency") or (
        avg_power / avg_hr if (avg_power and avg_hr) else 0.0
    )

    cadence = g("avg_cadence") or 85.0

    # Zones can come from either:
    #   * `time_in_zones` JSON column (live ORM Activity from the DB)
    #   * flat z1..z7 columns (synthetic-data parquet rows)
    zones = act.get("time_in_zones") or {}
    def zone(short, long):
        # Prefer flat key if present on the row, then named JSON key, then short
        if act.get(short) is not None:
            return act.get(short) or 0.0
        return zones.get(long, zones.get(short, 0.0)) or 0.0

    z1 = zone("z1", "z1_recovery")
    z2 = zone("z2", "z2_endurance")
    z3 = zone("z3", "z3_tempo")
    z4 = zone("z4", "z4_threshold")
    z5 = zone("z5", "z5_vo2max")
    z6 = zone("z6", "z6_anaerobic")
    z7 = zone("z7", "z7_neuromuscular")

    temp_c   = g("temperature_c", 18.0)
    humidity = g("humidity_pct", 50.0)
    wind     = g("wind_speed_kmh", 0.0)

    rpe = g("perceived_exertion", 5.0) or 5.0

    date = act["date"]
    dow_sin, dow_cos = _sincos(date.weekday(), 7.0)
    mon_sin, mon_cos = _sincos(date.month - 1, 12.0)

    # Workout type one-hot
    wt_vec = [0.0] * len(WORKOUT_TYPES)
    wt = act.get("workout_type")
    if wt and wt in WORKOUT_TYPE_IDX:
        wt_vec[WORKOUT_TYPE_IDX[wt]] = 1.0

    avg_hr_pct = (avg_hr / max_hr_athlete) if avg_hr else 0.0
    max_hr_pct = (max_hr_ride / max_hr_athlete) if max_hr_ride else 0.0

    vec = [
        n01(g("tss"), "tss"),
        n01(ctl, "ctl"),
        n01(atl, "atl"),
        n01(tsb, "tsb"),
        n01(duration_h, "duration_h"),
        n01(distance_km, "distance_km"),
        n01(elevation_km, "elevation_km"),
        n01(avg_power / ftp if avg_power else 0.0, "avg_power_pct_ftp"),
        n01(np_power / ftp if np_power else 0.0, "np_pct_ftp"),
        n01(raw_if, "intensity_factor"),
        n01(max_power / ftp if max_power else 0.0, "max_power_pct_ftp"),
        n01(vi, "variability_index"),
        n01(avg_hr_pct, "avg_hr_pct_max"),
        n01(max_hr_pct, "max_hr_pct_max"),
        n01(hr_drift, "hr_drift"),
        n01(aero_eff, "aerobic_efficiency"),
        n01(cadence, "avg_cadence"),
        float(z1), float(z2), float(z3), float(z4), float(z5), float(z6), float(z7),
        n01(temp_c, "temp_c"),
        n01(humidity, "humidity_pct"),
        n01(wind, "wind_speed_kmh"),
        dow_sin, dow_cos, mon_sin, mon_cos,
        n01(days_since_last, "days_since_last"),
        *wt_vec,
        n01(rpe, "rpe"),
    ]
    arr = np.asarray(vec, dtype=np.float32)
    assert arr.shape[0] == ACTIVITY_DIM, (
        f"encode_activity_row produced {arr.shape[0]} features, expected {ACTIVITY_DIM}"
    )
    return arr


def _norm_array(arr: np.ndarray, key: str) -> np.ndarray:
    lo, hi = NORM_BOUNDS[key]
    span = max(hi - lo, 1e-9)
    return np.clip((arr - lo) / span, 0.0, 1.0).astype(np.float32)


def encode_activity_dataframe(df) -> np.ndarray:
    """
    Vectorized encoder: take a DataFrame with raw columns and return a
    (N, ACTIVITY_DIM) float32 matrix. Same bounds & ordering as
    `encode_activity_row` — verified by `verify_encoder_parity()`.

    Required columns:
      duration_seconds, distance_meters, elevation_gain_meters,
      avg_power, normalized_power, max_power, intensity_factor, tss,
      variability_index, avg_hr, max_hr, hr_drift, aerobic_efficiency,
      avg_cadence, z1..z7, temperature_c, humidity_pct, wind_speed_kmh,
      perceived_exertion, ctl, atl, tsb, days_since_last_ride,
      ftp, athlete_max_hr, workout_type, date.
    """
    import pandas as pd  # local: keep top-level light

    ftp = np.maximum(df["ftp"].to_numpy(dtype=np.float32), 1.0)
    max_hr_a = np.maximum(df["athlete_max_hr"].to_numpy(dtype=np.float32), 1.0)

    avg_p = df["avg_power"].fillna(0).to_numpy(dtype=np.float32)
    np_p  = df["normalized_power"].fillna(df["avg_power"]).fillna(0).to_numpy(dtype=np.float32)
    max_p = df["max_power"].fillna(0).to_numpy(dtype=np.float32)
    avg_hr = df["avg_hr"].fillna(0).to_numpy(dtype=np.float32)
    max_hr = df["max_hr"].fillna(df["avg_hr"]).fillna(0).to_numpy(dtype=np.float32)

    dates = pd.to_datetime(df["date"])
    dow = dates.dt.weekday.to_numpy(dtype=np.float32)
    month = dates.dt.month.to_numpy(dtype=np.float32)
    dow_a = 2 * np.pi * dow / 7.0
    mon_a = 2 * np.pi * (month - 1) / 12.0

    # Workout type one-hot
    wt_series = df["workout_type"].fillna("endurance").astype(str)
    wt_oh = np.zeros((len(df), len(WORKOUT_TYPES)), dtype=np.float32)
    for i, name in enumerate(WORKOUT_TYPES):
        wt_oh[:, i] = (wt_series == name).to_numpy(dtype=np.float32)

    cols = [
        _norm_array(df["tss"].fillna(0).to_numpy(dtype=np.float32), "tss"),
        _norm_array(df["ctl"].fillna(0).to_numpy(dtype=np.float32), "ctl"),
        _norm_array(df["atl"].fillna(0).to_numpy(dtype=np.float32), "atl"),
        _norm_array(df["tsb"].fillna(0).to_numpy(dtype=np.float32), "tsb"),
        _norm_array(df["duration_seconds"].fillna(0).to_numpy(dtype=np.float32) / 3600.0, "duration_h"),
        _norm_array(df["distance_meters"].fillna(0).to_numpy(dtype=np.float32) / 1000.0, "distance_km"),
        _norm_array(df["elevation_gain_meters"].fillna(0).to_numpy(dtype=np.float32) / 1000.0, "elevation_km"),
        _norm_array(avg_p / ftp, "avg_power_pct_ftp"),
        _norm_array(np_p / ftp, "np_pct_ftp"),
        _norm_array(df["intensity_factor"].fillna(0).to_numpy(dtype=np.float32), "intensity_factor"),
        _norm_array(max_p / ftp, "max_power_pct_ftp"),
        _norm_array(df["variability_index"].fillna(1.0).to_numpy(dtype=np.float32), "variability_index"),
        _norm_array(avg_hr / max_hr_a, "avg_hr_pct_max"),
        _norm_array(max_hr / max_hr_a, "max_hr_pct_max"),
        _norm_array(df["hr_drift"].fillna(0).to_numpy(dtype=np.float32), "hr_drift"),
        _norm_array(df["aerobic_efficiency"].fillna(0).to_numpy(dtype=np.float32), "aerobic_efficiency"),
        _norm_array(df["avg_cadence"].fillna(85).to_numpy(dtype=np.float32), "avg_cadence"),
        df["z1"].fillna(0).to_numpy(dtype=np.float32),
        df["z2"].fillna(0).to_numpy(dtype=np.float32),
        df["z3"].fillna(0).to_numpy(dtype=np.float32),
        df["z4"].fillna(0).to_numpy(dtype=np.float32),
        df["z5"].fillna(0).to_numpy(dtype=np.float32),
        df["z6"].fillna(0).to_numpy(dtype=np.float32),
        df["z7"].fillna(0).to_numpy(dtype=np.float32),
        _norm_array(df["temperature_c"].fillna(18).to_numpy(dtype=np.float32), "temp_c"),
        _norm_array(df["humidity_pct"].fillna(50).to_numpy(dtype=np.float32), "humidity_pct"),
        _norm_array(df["wind_speed_kmh"].fillna(0).to_numpy(dtype=np.float32), "wind_speed_kmh"),
        np.sin(dow_a).astype(np.float32),
        np.cos(dow_a).astype(np.float32),
        np.sin(mon_a).astype(np.float32),
        np.cos(mon_a).astype(np.float32),
        _norm_array(df["days_since_last_ride"].fillna(0).to_numpy(dtype=np.float32), "days_since_last"),
    ]
    cols.extend(wt_oh[:, i] for i in range(len(WORKOUT_TYPES)))
    cols.append(_norm_array(df["perceived_exertion"].fillna(5).to_numpy(dtype=np.float32), "rpe"))

    out = np.stack(cols, axis=1).astype(np.float32)
    assert out.shape[1] == ACTIVITY_DIM, f"got {out.shape[1]} cols, expected {ACTIVITY_DIM}"
    return out


def encode_profile_dataframe(df) -> np.ndarray:
    """Vectorized encoder for profile features. Returns (N, PROFILE_DIM) float32."""
    sex_str = df["sex"].astype(str).str.lower()
    sex_bin = (sex_str == "male").to_numpy(dtype=np.float32)

    goal_str = df["primary_goal"].fillna("general_fitness").astype(str)
    goal_idx = goal_str.map(GOAL_TYPE_IDX).fillna(0).to_numpy(dtype=np.float32)
    goal_norm = goal_idx / max(len(GOAL_TYPES) - 1, 1)

    cols = [
        _norm_array(df["age"].fillna(35).to_numpy(dtype=np.float32), "age"),
        _norm_array(df["weight_kg"].fillna(70).to_numpy(dtype=np.float32), "weight_kg"),
        _norm_array(df["height_cm"].fillna(175).to_numpy(dtype=np.float32), "height_cm"),
        sex_bin,
        _norm_array(df["ftp"].fillna(200).to_numpy(dtype=np.float32), "ftp"),
        _norm_array(df["athlete_max_hr"].fillna(190).to_numpy(dtype=np.float32), "max_hr_athlete"),
        _norm_array(df["resting_hr"].fillna(55).to_numpy(dtype=np.float32), "resting_hr"),
        _norm_array(df["experience_years"].fillna(3).to_numpy(dtype=np.float32), "experience_years"),
        goal_norm,
        _norm_array(df["days_to_event"].fillna(365).to_numpy(dtype=np.float32), "days_to_event"),
    ]
    out = np.stack(cols, axis=1).astype(np.float32)
    assert out.shape[1] == PROFILE_DIM
    return out


def encode_profile_row(profile: Mapping) -> np.ndarray:
    """Encode an athlete profile (raw fields) → vector of length PROFILE_DIM."""
    sex = profile.get("sex")
    if isinstance(sex, str):
        sex_bin = 1.0 if sex.lower() == "male" else 0.0
    else:
        sex_bin = float(sex) if sex is not None else 0.0

    goal = profile.get("primary_goal") or profile.get("goal_type")
    if isinstance(goal, str):
        goal_idx = GOAL_TYPE_IDX.get(goal, 0)
    else:
        goal_idx = int(goal) if goal is not None else 0

    # Athlete max HR can arrive under either key:
    #   * `athlete_max_hr` for synth/parquet rows that also carry a ride max_hr
    #   * `max_hr` for ORM AthleteProfile via features.py
    max_hr_a = (profile.get("athlete_max_hr")
                if profile.get("athlete_max_hr") is not None
                else profile.get("max_hr"))
    if max_hr_a is None:
        max_hr_a = 190

    vec = [
        n01(profile.get("age") or 35, "age"),
        n01(profile.get("weight_kg") or 70, "weight_kg"),
        n01(profile.get("height_cm") or 175, "height_cm"),
        sex_bin,
        n01(profile.get("ftp") or 200, "ftp"),
        n01(max_hr_a, "max_hr_athlete"),
        n01(profile.get("resting_hr") or 55, "resting_hr"),
        n01(profile.get("experience_years") or profile.get("cycling_experience_years") or 3,
            "experience_years"),
        goal_idx / max(len(GOAL_TYPES) - 1, 1),
        n01(profile.get("days_to_event") if profile.get("days_to_event") is not None else 365,
            "days_to_event"),
    ]
    arr = np.asarray(vec, dtype=np.float32)
    assert arr.shape[0] == PROFILE_DIM
    return arr
