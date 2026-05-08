"""
Synthetic pre-training data generator for the CyclingTransformer.

Three evidence-based training philosophies are simulated:

  COGGAN / FRIEL   — Classic periodization (base → build → peak → taper).
                     Macro-cycles ~12/8/4 weeks. Sweetspot is king in build phase.
                     Friel "Cyclist's Training Bible"; Coggan & Allen.

  POGAČAR Z2       — UAE Team Emirates / San Millán method.
                     ~80% true Zone 2 aerobic volume (NOT sweetspot, NOT tempo).
                     Selective VO2max blocks and sprint work. Minimal middle
                     intensities. High fat oxidation, high long-term FTP ceiling.
                     San Millán & Brooks (2018).

  POLARIZED        — Seiler 80/20.
                     80% sub-LT1 (Z1/Z2), 20% supra-LT (Z4/Z5+).
                     Deliberately avoids "no-man's-land" moderate intensity.
                     Seiler & Tønnessen (2009); Stöggl & Sperlich (2014).

Output schema:
  Parquet of RAW values (watts, bpm, seconds, °C, %). Both training-time
  normalization (ml/training/dataset.py) and inference-time normalization
  (backend/app/ml/features.py) go through `app.ml.norm.encode_*`. This is
  the only way to guarantee the model sees the same distribution at train
  and inference time.

Usage:
  python -m ml.training.generate_synthetic --athletes 50000 --output ./ml/data/synthetic.parquet
  python -m ml.training.generate_synthetic --athletes 500 --weeks 26 \\
      --output ./ml/data/synthetic_small.parquet
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw): return x  # type: ignore


# ── Training philosophies ─────────────────────────────────────────────────────
COGGAN_FRIEL = "coggan_friel"
POGACAR_Z2   = "pogacar_z2"
POLARIZED    = "polarized"

WT = ["recovery", "easy", "endurance", "tempo", "sweetspot",
      "threshold", "vo2max", "sprint", "race", "long_ride"]


# ── Per-workout-type physiological templates ──────────────────────────────────
TEMPLATES = {
    "recovery":  dict(if_range=(0.44, 0.54), np_bonus=0.01,
                      dur_h=(0.75, 1.25),
                      zones=[0.08, 0.82, 0.10, 0, 0, 0, 0],
                      rpe=(1, 3), hr_drift=(0, 2), vi=(1.01, 1.04)),
    "easy":      dict(if_range=(0.54, 0.65), np_bonus=0.02,
                      dur_h=(1.0, 2.0),
                      zones=[0.04, 0.82, 0.13, 0.01, 0, 0, 0],
                      rpe=(2, 4), hr_drift=(0, 3), vi=(1.02, 1.07)),
    "endurance": dict(if_range=(0.60, 0.73), np_bonus=0.03,
                      dur_h=(1.5, 3.5),
                      zones=[0.02, 0.74, 0.21, 0.03, 0, 0, 0],
                      rpe=(3, 5), hr_drift=(1, 6), vi=(1.03, 1.10)),
    "tempo":     dict(if_range=(0.74, 0.84), np_bonus=0.04,
                      dur_h=(1.0, 2.5),
                      zones=[0, 0.18, 0.50, 0.27, 0.05, 0, 0],
                      rpe=(5, 7), hr_drift=(2, 9), vi=(1.04, 1.12)),
    "sweetspot": dict(if_range=(0.84, 0.93), np_bonus=0.05,
                      dur_h=(1.0, 2.25),
                      zones=[0, 0.04, 0.18, 0.36, 0.37, 0.05, 0],
                      rpe=(6, 8), hr_drift=(3, 11), vi=(1.04, 1.12)),
    "threshold": dict(if_range=(0.91, 1.03), np_bonus=0.06,
                      dur_h=(0.75, 1.75),
                      zones=[0, 0.04, 0.10, 0.10, 0.52, 0.20, 0.04],
                      rpe=(7, 9), hr_drift=(4, 13), vi=(1.05, 1.16)),
    "vo2max":    dict(if_range=(0.95, 1.12), np_bonus=0.12,
                      dur_h=(0.75, 1.50),
                      zones=[0, 0.20, 0.18, 0.04, 0.14, 0.36, 0.08],
                      rpe=(8, 10), hr_drift=(2, 7), vi=(1.10, 1.28)),
    "sprint":    dict(if_range=(0.85, 1.06), np_bonus=0.20,
                      dur_h=(0.75, 1.25),
                      zones=[0, 0.40, 0.28, 0.04, 0.04, 0.06, 0.18],
                      rpe=(7, 10), hr_drift=(1, 6), vi=(1.15, 1.45)),
    "race":      dict(if_range=(0.92, 1.12), np_bonus=0.15,
                      dur_h=(2.0, 6.0),
                      zones=[0, 0.08, 0.24, 0.15, 0.21, 0.22, 0.10],
                      rpe=(8, 10), hr_drift=(3, 13), vi=(1.12, 1.32)),
    "long_ride": dict(if_range=(0.57, 0.73), np_bonus=0.04,
                      dur_h=(3.5, 7.0),
                      zones=[0.02, 0.70, 0.23, 0.04, 0.01, 0, 0],
                      rpe=(4, 7), hr_drift=(4, 18), vi=(1.04, 1.13)),
}


# ── Weekly schedule templates per philosophy × phase ──────────────────────────
SCHEDULES = {
    COGGAN_FRIEL: {
        "base": [
            ["endurance", "endurance", "easy", "long_ride", "recovery"],
            ["endurance", "endurance", "endurance", "long_ride", "easy"],
            ["endurance", "tempo", "easy", "long_ride", "recovery"],
        ],
        "build": [
            ["sweetspot", "endurance", "easy", "sweetspot", "long_ride"],
            ["sweetspot", "threshold", "endurance", "long_ride", "recovery"],
            ["sweetspot", "sweetspot", "endurance", "threshold", "long_ride"],
        ],
        "peak": [
            ["threshold", "endurance", "vo2max", "endurance", "threshold"],
            ["vo2max", "endurance", "threshold", "easy", "long_ride"],
            ["threshold", "vo2max", "endurance", "recovery", "threshold"],
        ],
        "taper": [
            ["endurance", "threshold", "recovery", "easy"],
            ["easy", "vo2max", "recovery", "endurance"],
        ],
        "recovery_week": [
            ["easy", "recovery", "endurance", "easy"],
            ["recovery", "easy", "recovery", "easy"],
        ],
    },
    POGACAR_Z2: {
        "base": [
            ["endurance", "sprint", "long_ride", "endurance", "easy"],
            ["long_ride", "endurance", "sprint", "endurance", "recovery"],
            ["endurance", "endurance", "sprint", "long_ride", "recovery"],
        ],
        "build": [
            ["endurance", "vo2max", "long_ride", "sprint", "endurance"],
            ["long_ride", "endurance", "vo2max", "endurance", "sprint"],
            ["endurance", "sprint", "vo2max", "long_ride", "endurance"],
        ],
        "peak": [
            ["endurance", "threshold", "vo2max", "long_ride", "sprint"],
            ["vo2max", "endurance", "threshold", "endurance", "sprint"],
        ],
        "taper": [
            ["easy", "vo2max", "recovery", "sprint"],
            ["endurance", "sprint", "easy", "recovery"],
        ],
        "recovery_week": [
            ["easy", "endurance", "recovery", "easy"],
            ["recovery", "easy", "easy", "recovery"],
        ],
    },
    POLARIZED: {
        "base": [
            ["endurance", "endurance", "vo2max", "long_ride", "easy"],
            ["long_ride", "endurance", "easy", "vo2max", "recovery"],
        ],
        "build": [
            ["endurance", "vo2max", "endurance", "vo2max", "long_ride"],
            ["long_ride", "vo2max", "endurance", "threshold", "easy"],
        ],
        "peak": [
            ["endurance", "vo2max", "threshold", "endurance", "vo2max"],
            ["vo2max", "threshold", "endurance", "long_ride", "easy"],
        ],
        "taper": [
            ["easy", "vo2max", "recovery", "endurance"],
            ["endurance", "threshold", "easy", "recovery"],
        ],
        "recovery_week": [
            ["easy", "endurance", "recovery", "easy"],
            ["recovery", "easy", "easy", "recovery"],
        ],
    },
}


# ── Athlete profile ───────────────────────────────────────────────────────────
@dataclass
class Athlete:
    athlete_id: int
    age: int
    weight: float
    height: int
    sex: str              # "male"|"female"
    ftp: float            # current FTP (updated each week)
    ftp_ceiling: float    # genetic maximum
    max_hr: int
    resting_hr: int
    experience: float     # years
    adaptation_rate: float
    philosophy: str
    goal: str             # "general_fitness"|"event_specific"|"ftp_improvement"
    training_days: int
    n_weeks: int
    event_week: int
    start_date: datetime
    ctl: float = 0.0
    atl: float = 0.0


def _make_athlete(rng: np.random.Generator, athlete_id: int) -> Athlete:
    sex_bin = int(rng.integers(0, 2))
    sex     = "male" if sex_bin == 0 else "female"
    age     = int(rng.integers(18, 58))
    weight  = float(rng.uniform(52, 95))
    height  = int(rng.integers(160, 196))
    exp     = float(rng.uniform(0, 12))

    ftp_base = rng.uniform(130, 300) if sex == "male" else rng.uniform(105, 250)
    ftp      = float(np.clip(ftp_base + exp * 6, 100, 420))

    wkg_ceil = rng.uniform(3.4, 5.6) if sex == "male" else rng.uniform(2.9, 4.9)
    ceiling  = float(np.clip(wkg_ceil * weight, ftp + 20, 600))

    max_hr     = int(np.clip(208 - 0.7 * age + rng.integers(-8, 8), 155, 210))
    resting_hr = int(np.clip(60 - exp * 1.5 + rng.integers(-5, 8), 35, 75))
    adapt_rate = float(np.clip(rng.normal(1.0, 0.22), 0.45, 1.9))

    philosophy = rng.choice([COGGAN_FRIEL, POGACAR_Z2, POLARIZED],
                            p=[0.45, 0.30, 0.25])
    goal_str = rng.choice(["general_fitness", "event_specific", "ftp_improvement"],
                           p=[0.45, 0.35, 0.20])
    n_weeks    = int(rng.integers(40, 105))
    event_week = (int(rng.integers(20, n_weeks))
                  if goal_str == "event_specific" else n_weeks)

    start_date = datetime(2021, 1, 1) + timedelta(days=int(rng.integers(0, 365 * 4)))
    ctl = float(rng.uniform(8, 45))

    return Athlete(
        athlete_id=athlete_id,
        age=age, weight=weight, height=height, sex=sex,
        ftp=ftp, ftp_ceiling=ceiling,
        max_hr=max_hr, resting_hr=resting_hr,
        experience=exp, adaptation_rate=adapt_rate,
        philosophy=str(philosophy), goal=str(goal_str),
        training_days=int(rng.integers(3, 7)),
        n_weeks=n_weeks, event_week=event_week,
        start_date=start_date,
        ctl=ctl, atl=ctl * float(rng.uniform(0.85, 1.15)),
    )


def _phase(week: int, event_week: int) -> str:
    weeks_to_event = event_week - week
    if weeks_to_event > 20:  return "base"
    if weeks_to_event > 8:   return "build"
    if weeks_to_event > 3:   return "peak"
    if weeks_to_event > 0:   return "taper"
    return "base"


# ── Simulate a single ride: produce RAW values ────────────────────────────────
def _simulate_ride(
    wt: str, athlete: Athlete, rng: np.random.Generator,
    date: datetime, days_since_last: int,
    ctl: float, atl: float,
) -> tuple[dict, float]:
    """Return (raw_dict, tss). raw_dict keys match the synthetic parquet schema."""
    T = TEMPLATES[wt]

    raw_if   = float(rng.uniform(*T["if_range"]))
    tsb = ctl - atl
    fatigue_penalty = max(0.0, -tsb / 120)  # up to 5% penalty when TSB=-60
    raw_if = max(0.3, raw_if * (1 - fatigue_penalty * 0.05))

    dur_h    = float(rng.uniform(*T["dur_h"]))
    avg_pwr  = raw_if * athlete.ftp
    np_pwr   = avg_pwr * (1 + T["np_bonus"] * float(rng.uniform(0.6, 1.4)))
    max_pwr  = avg_pwr * (float(rng.uniform(1.4, 2.2)) if wt in ("sprint", "race")
                          else float(rng.uniform(1.2, 1.7)))
    vi       = np_pwr / max(avg_pwr, 1)

    hr_pct_target = 0.50 + raw_if * 0.40
    avg_hr_pct = float(np.clip(hr_pct_target + rng.normal(0, 0.03), 0.45, 0.98))
    max_hr_pct = float(np.clip(avg_hr_pct + rng.uniform(0.05, 0.18), 0.55, 1.0))
    avg_hr = avg_hr_pct * athlete.max_hr
    max_hr_ride = max_hr_pct * athlete.max_hr

    hr_drift = float(rng.uniform(*T["hr_drift"]))
    if dur_h > 3.0:
        hr_drift *= 1.3
    if athlete.philosophy == POGACAR_Z2 and wt in ("endurance", "long_ride"):
        hr_drift *= float(rng.uniform(0.5, 0.8))

    aero_eff = avg_pwr / max(avg_hr, 1)

    tss = (dur_h * raw_if ** 2) * 100

    zones = np.array(T["zones"], dtype=np.float32)
    zones += rng.uniform(0, 0.03, size=7)
    zones = np.clip(zones, 0, None)
    zones /= zones.sum() + 1e-9

    month = date.month
    season_temp_c = 20 + 12 * math.sin((month - 4) / 12 * 2 * math.pi)
    temp_c   = season_temp_c + float(rng.normal(0, 5))
    humidity = float(rng.uniform(30, 90))
    wind_kmh = float(rng.uniform(0, 40))

    cadence = (float(rng.uniform(82, 100)) if wt not in ("sprint", "race")
               else float(rng.uniform(90, 110)))

    rpe = float(rng.uniform(*T["rpe"]))
    if tsb < -25:
        rpe = min(10.0, rpe + 1.0)

    dist_km = avg_pwr / max(athlete.weight, 1.0) * dur_h * 18  # rough power-to-speed
    elev_m  = float(rng.uniform(0, 800.0 * dur_h))             # up to 800 m/hr

    raw = {
        # identification / target labels
        "athlete_id":            athlete.athlete_id,
        "philosophy":            athlete.philosophy,
        "date":                  date,
        "workout_type":          wt,
        # session
        "duration_seconds":      dur_h * 3600.0,
        "distance_meters":       dist_km * 1000.0,
        "elevation_gain_meters": elev_m,
        # power
        "avg_power":             avg_pwr,
        "normalized_power":      np_pwr,
        "max_power":             max_pwr,
        "intensity_factor":      raw_if,
        "tss":                   tss,
        "variability_index":     vi,
        # HR
        "avg_hr":                avg_hr,
        "max_hr":                max_hr_ride,
        "hr_drift":              hr_drift,
        "aerobic_efficiency":    aero_eff,
        # cadence
        "avg_cadence":           cadence,
        # zones (raw fractions, sum≈1) — store as 7 separate columns
        "z1": float(zones[0]), "z2": float(zones[1]), "z3": float(zones[2]),
        "z4": float(zones[3]), "z5": float(zones[4]),
        "z6": float(zones[5]), "z7": float(zones[6]),
        # environment
        "temperature_c":         temp_c,
        "humidity_pct":          humidity,
        "wind_speed_kmh":        wind_kmh,
        # subjective
        "perceived_exertion":    rpe,
        # state at the moment of the ride (pre-update)
        "ctl":                   ctl,
        "atl":                   atl,
        "tsb":                   tsb,
        "days_since_last_ride":  float(days_since_last),
        # athlete profile (snapshot)
        "ftp":                   athlete.ftp,
        "age":                   athlete.age,
        "weight_kg":             athlete.weight,
        "height_cm":             athlete.height,
        "sex":                   athlete.sex,
        "athlete_max_hr":        athlete.max_hr,
        "resting_hr":            athlete.resting_hr,
        "experience_years":      athlete.experience,
        "primary_goal":          athlete.goal,
        "days_to_event":         max(0.0, (athlete.event_week - (date - athlete.start_date).days / 7.0)) * 7.0,
    }
    return raw, tss


# ── Physiological FTP adaptation model ────────────────────────────────────────
def _adapt_ftp(
    athlete: Athlete,
    week_tss_by_type: dict[str, float],
    tsb: float,
    rng: np.random.Generator,
) -> float:
    """Return weekly FTP delta in watts (can be negative)."""
    tss_z2  = sum(week_tss_by_type.get(w, 0) for w in ("easy", "endurance", "long_ride"))
    tss_hi  = sum(week_tss_by_type.get(w, 0) for w in ("threshold", "vo2max"))
    tss_spr = week_tss_by_type.get("sprint", 0)
    tss_mod = sum(week_tss_by_type.get(w, 0) for w in ("tempo", "sweetspot"))

    z2_signal_base = min(tss_z2 / 220.0, 1.0) * 0.45
    if athlete.philosophy == POGACAR_Z2:
        z2_volume_bonus = min(tss_z2 / 400.0, 0.4) * 0.20
        z2_signal = z2_signal_base + z2_volume_bonus
        hi_signal = min(tss_hi / 70.0, 1.0) * 1.10
        mod_signal = 0.0
    elif athlete.philosophy == POLARIZED:
        z2_signal = z2_signal_base
        hi_signal = min(tss_hi / 70.0, 1.0) * 1.00
        mod_signal = 0.0
    else:  # COGGAN_FRIEL
        z2_signal = z2_signal_base * 0.90
        hi_signal = min(tss_hi / 70.0, 1.0) * 0.90
        mod_signal = min(tss_mod / 100.0, 1.0) * 0.55  # sweetspot effective for Coggan

    spr_signal = min(tss_spr / 40.0, 1.0) * 0.08
    raw_signal = z2_signal + hi_signal + mod_signal + spr_signal

    if tsb >= -10:
        recovery_factor = 1.0
    elif tsb >= -30:
        recovery_factor = 1.0 + (tsb + 10) / 20 * 0.5
    else:
        recovery_factor = max(0.05, 0.5 + (tsb + 30) / 60)

    ceiling_distance = max(0.05, (athlete.ftp_ceiling - athlete.ftp) / athlete.ftp_ceiling)
    ceiling_factor = math.sqrt(ceiling_distance)
    genetic_factor = athlete.adaptation_rate

    delta = raw_signal * recovery_factor * ceiling_factor * genetic_factor

    if rng.random() < 0.03:               # illness
        delta = float(rng.uniform(-15, -5))

    delta += float(rng.normal(0, 0.15))    # measurement noise
    return delta


# ── Simulate one athlete ──────────────────────────────────────────────────────
def _simulate_athlete(
    athlete: Athlete, rng: np.random.Generator
) -> list[dict]:
    rows: list[dict] = []
    ctl  = athlete.ctl
    atl  = athlete.atl
    date = athlete.start_date
    days_since_last = 3

    for week_num in range(athlete.n_weeks):
        phase = _phase(week_num, athlete.event_week)

        recovery_interval = 4 if athlete.philosophy == POGACAR_Z2 else 3
        if week_num > 0 and week_num % recovery_interval == 0:
            phase = "recovery_week"

        # Illness / life disruption ~5% chance — skip the week
        if rng.random() < 0.05:
            date += timedelta(days=7)
            ctl *= (1 - 2 / 43)
            atl *= (1 - 2 / 8)
            days_since_last += 7
            continue

        schedule_options = SCHEDULES[athlete.philosophy].get(
            phase, SCHEDULES[athlete.philosophy]["base"]
        )
        weekly_wts: list[str] = list(rng.choice(schedule_options))

        if len(weekly_wts) > athlete.training_days:
            weekly_wts = weekly_wts[: athlete.training_days]

        weeks_to_event = athlete.event_week - week_num
        if athlete.goal == "event_specific" and 1 <= weeks_to_event <= 12 and rng.random() < 0.12:
            weekly_wts[-1] = "race"

        spread_days = sorted(rng.choice(7, size=len(weekly_wts), replace=False))

        week_tss_by_type: dict[str, float] = {}

        for i, wt in enumerate(weekly_wts):
            ride_date = date + timedelta(days=int(spread_days[i]))
            row, tss = _simulate_ride(
                wt, athlete, rng, ride_date, days_since_last, ctl, atl
            )

            alpha_ctl = 2 / 43
            alpha_atl = 2 / 8
            ctl = ctl + alpha_ctl * (tss - ctl)
            atl = atl + alpha_atl * (tss - atl)

            week_tss_by_type[wt] = week_tss_by_type.get(wt, 0) + tss
            days_since_last = 1
            rows.append(row)

        # End-of-week FTP adaptation
        tsb = ctl - atl
        delta = _adapt_ftp(athlete, week_tss_by_type, tsb, rng)
        athlete.ftp = float(np.clip(athlete.ftp + delta, 80, athlete.ftp_ceiling))

        date += timedelta(days=7)

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def generate(n_athletes: int, n_weeks: int | None, seed: int, output: str):
    rng = np.random.default_rng(seed)
    all_rows: list[dict] = []

    for i in tqdm(range(n_athletes), desc="Simulating athletes"):
        athlete = _make_athlete(rng, athlete_id=i + 1)
        if n_weeks:
            athlete.n_weeks = n_weeks
            athlete.event_week = min(athlete.event_week, n_weeks)
        rows = _simulate_athlete(athlete, rng)
        all_rows.extend(rows)

    print(f"\nTotal rides simulated: {len(all_rows):,}")
    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    df.to_parquet(output, index=False)

    print(f"\nSaved → {output}")
    print(f"File size: {os.path.getsize(output) / 1e6:.1f} MB")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"FTP range: {df['ftp'].min():.0f} – {df['ftp'].max():.0f} W")
    print(f"\nWorkout type distribution:\n{df['workout_type'].value_counts().to_string()}")
    print(f"\nPhilosophy distribution:\n{df.drop_duplicates('athlete_id')['philosophy'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic cycling training data for transformer pre-training"
    )
    parser.add_argument("--athletes", type=int, default=50_000)
    parser.add_argument("--weeks",    type=int, default=None,
                        help="Override weeks per athlete (default: random 40–104)")
    parser.add_argument("--output",   default="./ml/data/synthetic.parquet")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()
    generate(args.athletes, args.weeks, args.seed, args.output)
