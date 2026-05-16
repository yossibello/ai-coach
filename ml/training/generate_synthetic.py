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
import multiprocessing
import os
import tempfile
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
    goal: str             # see GOAL_TYPES in app.ml.norm — must match.
    event_type: str | None  # "climbing_camp" | "gran_fondo" | "crit" | "tt" | ...
    training_days: int
    n_weeks: int
    event_week: int
    start_date: datetime
    ctl: float = 0.0
    atl: float = 0.0
    # ── Power curve profile (stable per athlete) ─────────────────────────
    sprint_factor:    float = 3.0   # personal-best 5-sec W/kg ÷ FTP W/kg (1.8–5.0)
    anaerobic_factor: float = 1.65  # personal-best 1-min W/kg ÷ FTP W/kg (1.3–2.1)  — mutable
    vo2max_factor:    float = 1.18  # personal-best 5-min W/kg ÷ FTP W/kg (1.08–1.35) — mutable
    anaerobic_ceiling_factor: float = 2.0   # genetic max anaerobic_factor
    vo2max_ceiling_factor:    float = 1.30  # genetic max vo2max_factor
    # ── Health/recovery state (updated daily) ────────────────────────────
    # Plews & Buchheit (2013): each athlete has an individual HRV baseline
    # (in ms) that drifts slowly with fitness. Day-to-day RMSSD oscillates
    # around it under autonomic load.
    hrv_baseline: float = 50.0     # rolling mean of overnight RMSSD (ms)
    hrv_today: float = 50.0
    rhr_today: int = 55            # today's resting HR
    sleep_score_today: float = 75.0
    body_battery_today: float = 80.0


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
    # Sample from the full goal taxonomy so the trained model learns each
    # one's distinctive training signature. Distribution skews towards
    # general_fitness (most amateurs) but covers all event-driven goals.
    goal_str = rng.choice(
        [
            "general_fitness", "ftp_improvement", "weight_loss",
            "event_specific", "gran_fondo", "criterium",
            "climbing", "triathlon",
        ],
        p=[0.30, 0.15, 0.05, 0.15, 0.12, 0.08, 0.10, 0.05],
    )
    # Event-driven goals get a concrete event_type so we can bias the schedule.
    if goal_str == "criterium":
        ev_type = "crit"
    elif goal_str == "gran_fondo":
        ev_type = str(rng.choice(["gran_fondo", "long_road", "mtb_marathon"], p=[0.6, 0.25, 0.15]))
    elif goal_str == "climbing":
        ev_type = str(rng.choice(["climbing_camp", "stage_race", "ultra_endurance"], p=[0.6, 0.25, 0.15]))
    elif goal_str == "triathlon":
        ev_type = str(rng.choice(["triathlon_70_3", "triathlon_140_6"], p=[0.65, 0.35]))
    elif goal_str == "event_specific":
        ev_type = str(rng.choice(["long_road", "tt", "stage_race", "gran_fondo"], p=[0.4, 0.2, 0.2, 0.2]))
    else:
        ev_type = None
    has_event = ev_type is not None
    n_weeks    = int(rng.integers(40, 105))
    event_week = (int(rng.integers(20, n_weeks))
                  if has_event else n_weeks)

    start_date = datetime(2021, 1, 1) + timedelta(days=int(rng.integers(0, 365 * 4)))
    ctl = float(rng.uniform(8, 45))

    # Power curve profile: sprint/anaerobic factors capture fast-twitch fraction;
    # vo2max factor captures VO2max:FTP ratio. Both are weakly correlated with goal.
    if goal_str == "criterium":
        sprint_factor    = float(rng.uniform(3.2, 5.0))
        anaerobic_factor = float(rng.uniform(1.65, 2.10))
        vo2max_factor    = float(rng.uniform(1.10, 1.20))
    elif goal_str in ("climbing", "triathlon"):
        sprint_factor    = float(rng.uniform(1.8, 2.8))
        anaerobic_factor = float(rng.uniform(1.30, 1.60))
        vo2max_factor    = float(rng.uniform(1.18, 1.35))
    else:
        sprint_factor    = float(rng.uniform(2.2, 3.8))
        anaerobic_factor = float(rng.uniform(1.45, 1.82))
        vo2max_factor    = float(rng.uniform(1.12, 1.28))

    # Genetic ceilings: how high each factor can grow with optimal training.
    anaerobic_ceil = float(np.clip(anaerobic_factor + rng.uniform(0.10, 0.40), anaerobic_factor + 0.05, 2.60))
    vo2max_ceil    = float(np.clip(vo2max_factor    + rng.uniform(0.03, 0.12), vo2max_factor    + 0.02, 1.40))

    # Initial HRV baseline scales with fitness/recovery (typical 30–120 ms RMSSD).
    # Trained endurance athletes sit higher; older athletes lower
    # (Plews & Buchheit 2013, Aubert et al. 2003).
    hrv_baseline = float(np.clip(
        70 - 0.4 * age + 4.0 * exp + rng.normal(0, 8),
        25, 130,
    ))

    return Athlete(
        athlete_id=athlete_id,
        age=age, weight=weight, height=height, sex=sex,
        ftp=ftp, ftp_ceiling=ceiling,
        max_hr=max_hr, resting_hr=resting_hr,
        experience=exp, adaptation_rate=adapt_rate,
        philosophy=str(philosophy), goal=str(goal_str),
        event_type=ev_type,
        training_days=int(rng.integers(3, 7)),
        n_weeks=n_weeks, event_week=event_week,
        start_date=start_date,
        ctl=ctl, atl=ctl * float(rng.uniform(0.85, 1.15)),
        sprint_factor=sprint_factor,
        anaerobic_factor=anaerobic_factor,
        vo2max_factor=vo2max_factor,
        anaerobic_ceiling_factor=anaerobic_ceil,
        vo2max_ceiling_factor=vo2max_ceil,
        hrv_baseline=hrv_baseline,
        hrv_today=hrv_baseline,
        rhr_today=resting_hr,
        sleep_score_today=75.0,
        body_battery_today=80.0,
    )


def _phase(week: int, event_week: int) -> str:
    weeks_to_event = event_week - week
    if weeks_to_event > 20:  return "base"
    if weeks_to_event > 8:   return "build"
    if weeks_to_event > 3:   return "peak"
    if weeks_to_event > 0:   return "taper"
    return "base"


# Event-specific schedule bias. During build/peak (and partly taper), athletes
# preparing for these events skew their workout mix to match event demands.
# Each entry maps a generic workout type → preferred replacement for that
# event_type. The trained model picks up these patterns via the
# (primary_goal, days_to_event, workout_type) feature triple.
_SYNTH_EVENT_BIAS: dict[str, dict[str, str]] = {
    "climbing_camp":   {"vo2max": "sweetspot", "threshold": "sweetspot", "tempo": "long_ride"},
    "gran_fondo":      {"vo2max": "sweetspot", "threshold": "tempo"},
    "ultra_endurance": {"vo2max": "endurance", "threshold": "tempo", "sweetspot": "endurance"},
    "mtb_marathon":    {"threshold": "sweetspot", "tempo": "sweetspot"},
    "stage_race":      {"vo2max": "threshold"},
    "crit":            {"sweetspot": "vo2max", "tempo": "vo2max"},
    "tt":              {"vo2max": "threshold", "sweetspot": "threshold"},
    "long_road":       {"tempo": "sweetspot"},
    "triathlon_70_3":  {"vo2max": "threshold"},
    "triathlon_140_6": {"vo2max": "tempo", "threshold": "sweetspot"},
}


def _apply_event_bias(weekly_wts: list[str], event_type: str | None, phase: str) -> list[str]:
    """Apply event-specific workout substitutions during build/peak phase."""
    if not event_type or phase not in ("build", "peak"):
        return weekly_wts
    bias = _SYNTH_EVENT_BIAS.get(event_type)
    if not bias:
        return weekly_wts
    return [bias.get(w, w) for w in weekly_wts]


# ── Horizon probe sequences ───────────────────────────────────────────────────
# Goal: teach the model that the same FITNESS STATE should yield DIFFERENT
# recommendations depending on how far the event is.  This is the Friel/Coggan
# periodization principle that the normal simulation can't provide alone —
# because in the natural trajectory an athlete in base always has high DTE and
# an athlete in taper always has low DTE, so the model conflates fitness state
# with DTE and can't separate them.
#
# Solution: append a structured "probe season" after each real simulation.
# The probe season contains 4 phase blocks (one per Friel phase), each 3 weeks
# long.  Within a block ALL rides have the same fixed days_to_event AND
# phase-appropriate workout types.  The dataset then samples windows FROM
# WITHIN these blocks — so both the history context AND the look-ahead target
# are phase-consistent.  The model sees the contrastive signal it needs.
#
# Friel/Coggan phase boundaries (weeks_to_event → phase):
#   > 20 weeks (140d+)  → base
#   8–20 weeks (56–140d) → build
#   3–8 weeks  (21–56d)  → peak
#   0–3 weeks  (0–21d)   → taper

# Representative DTE per phase (mid-phase, clearly inside each zone)
_PROBE_PHASE_CONFIG: list[tuple[str, int]] = [
    ("base",  182),   # 26 weeks out — deep aerobic base
    ("build",  84),   # 12 weeks out — progressive intensity build
    ("peak",   35),   # 5 weeks out  — quality sharpening
    ("taper",  10),   # 10 days out  — race-week freshening
]

# Each philosophy's most representative workout mix per phase.
# Two workouts per phase so the model sees the characteristic PAIR, not just
# one type.  Friel/Coggan values guide COGGAN_FRIEL; the others follow their
# own sport-science logic.
_PROBE_SCHEDULES: dict[str, dict[str, list[str]]] = {
    COGGAN_FRIEL: {
        "base":          ["endurance", "endurance", "tempo",     "long_ride",  "easy"],
        "build":         ["sweetspot", "endurance", "sweetspot", "threshold",  "long_ride"],
        "peak":          ["threshold", "endurance", "vo2max",    "easy",       "threshold"],
        "taper":         ["easy",      "threshold", "recovery",  "easy",       "endurance"],
        "recovery_week": ["recovery",  "easy",      "recovery",  "easy"],
    },
    POGACAR_Z2: {
        "base":          ["endurance", "endurance", "sprint",    "long_ride",  "easy"],
        "build":         ["endurance", "vo2max",    "long_ride", "sprint",     "endurance"],
        "peak":          ["vo2max",    "endurance", "sprint",    "threshold",  "easy"],
        "taper":         ["easy",      "sprint",    "recovery",  "easy",       "endurance"],
        "recovery_week": ["recovery",  "easy",      "recovery",  "easy"],
    },
    POLARIZED: {
        "base":          ["endurance", "endurance", "vo2max",    "long_ride",  "easy"],
        "build":         ["endurance", "vo2max",    "endurance", "vo2max",     "long_ride"],
        "peak":          ["vo2max",    "endurance", "threshold", "easy",       "vo2max"],
        "taper":         ["easy",      "vo2max",    "recovery",  "easy",       "endurance"],
        "recovery_week": ["recovery",  "easy",      "recovery",  "easy"],
    },
}


def _simulate_probe_season(
    athlete: "Athlete",
    rng: np.random.Generator,
    season_start: datetime,
    ctl: float,
    atl: float,
) -> list[dict]:
    """Append a structured horizon-probe season after the real simulation.

    Produces 4 phase blocks × 3 weeks × (up to training_days) rides.
    Within each block every ride has days_to_event fixed to the mid-phase value
    so BOTH the history window AND the look-ahead target window (used by
    CyclingDataset) are phase-consistent.  CTL/ATL evolve realistically within
    each block but are not carried across blocks (each block starts from the
    same snapshot) to avoid one phase's fatigue contaminating the next.
    """
    rows: list[dict] = []
    sched = _PROBE_SCHEDULES.get(athlete.philosophy, _PROBE_SCHEDULES[COGGAN_FRIEL])

    for phase_name, probe_dte in _PROBE_PHASE_CONFIG:
        # Each block starts from the same fitness snapshot — the point is to
        # show the model that the SAME CTL/ATL can be associated with different
        # training depending solely on days_to_event.
        block_ctl = ctl
        block_atl = atl
        block_date = season_start
        days_since_last = 2
        yesterday_workout: str | None = None
        yesterday_tss = 0.0
        days_since_hard = 3

        phase_options = sched.get(phase_name, sched["base"])

        for week_num in range(3):    # 3 weeks per phase block
            # Every 3rd week is a recovery week (Friel: 3+1 pattern)
            if week_num == 2:
                weekly_wts = list(sched.get("recovery_week", ["easy", "recovery", "easy", "easy"]))
            else:
                weekly_wts = list(rng.choice(phase_options)
                                  if isinstance(phase_options[0], list)
                                  else phase_options)

            if len(weekly_wts) > athlete.training_days:
                weekly_wts = weekly_wts[: athlete.training_days]

            spread_days = sorted(
                rng.choice(7, size=len(weekly_wts), replace=False).tolist()
            )

            for i, wt in enumerate(weekly_wts):
                ride_date = block_date + timedelta(days=int(spread_days[i]))

                hrv_today, rhr_today, sleep_score, body_battery = _simulate_health_for_day(
                    athlete, rng,
                    yesterday_workout=yesterday_workout,
                    yesterday_tss=yesterday_tss,
                    days_since_hard=days_since_hard,
                    atl=block_atl, ctl=block_ctl,
                )

                row, tss = _simulate_ride(
                    wt, athlete, rng, ride_date, days_since_last,
                    block_ctl, block_atl,
                    hrv_today=hrv_today,
                    rhr_today=rhr_today,
                    sleep_score=sleep_score,
                    body_battery=body_battery,
                )

                # Override days_to_event: this is the core of the probe season.
                row["days_to_event"] = float(probe_dte)

                block_ctl = block_ctl + (2 / 43) * (tss - block_ctl)
                block_atl = block_atl + (2 / 8)  * (tss - block_atl)

                days_since_last = 1
                yesterday_workout = wt
                yesterday_tss = tss
                days_since_hard = 0 if wt in ("threshold", "vo2max", "race", "sprint", "sweetspot") else days_since_hard + 1

                rows.append(row)

            block_date += timedelta(days=7)

        # Gap between phase blocks so the dataset never samples a window that
        # straddles two probe phases.
        season_start = block_date + timedelta(days=21)

    return rows


# ── HRV / RHR / sleep / body-battery simulation ───────────────────────────────
# Implements an autonomic-recovery model based on:
#   • Plews & Buchheit (2013) — Ln(RMSSD) drops with sympathetic dominance and
#     fluctuates ~5–15 ms day to day around an individual baseline.
#   • Stanley, Peake & Buchheit (2013) — HR recovery time-courses by intensity:
#     Z1/Z2 < 24h, threshold 24–48h, VO2max/race 48–72h.
#   • Buchheit (2014) — RHR rises 3–10 bpm with accumulated fatigue; returns
#     to baseline with rest.
#   • Vesterinen et al. (2016) — fitter athletes have higher Ln(RMSSD) baselines.
# Returned values are RAW (ms / bpm / 0–100) so the encoder will normalize.

# Fraction of personal-best power curve achieved at each duration in each
# workout type: (5s_range, 1min_range, 5min_range, 20min_range).
# Reflects what portion of the athlete's neuromuscular/aerobic ceiling is
# actually expressed during a given session type.
_PC_EFFORT: dict[str, tuple] = {
    "recovery":  ((0.10, 0.25), (0.10, 0.22), (0.15, 0.30), (0.28, 0.45)),
    "easy":      ((0.15, 0.30), (0.18, 0.30), (0.28, 0.45), (0.45, 0.62)),
    "endurance": ((0.15, 0.30), (0.20, 0.35), (0.38, 0.58), (0.58, 0.75)),
    "tempo":     ((0.20, 0.38), (0.30, 0.48), (0.58, 0.75), (0.75, 0.90)),
    "sweetspot": ((0.20, 0.38), (0.30, 0.48), (0.65, 0.82), (0.85, 0.97)),
    "threshold": ((0.20, 0.38), (0.32, 0.50), (0.75, 0.92), (0.92, 1.00)),
    "vo2max":    ((0.35, 0.62), (0.62, 0.85), (0.85, 1.00), (0.72, 0.88)),
    "sprint":    ((0.80, 1.00), (0.75, 0.98), (0.45, 0.65), (0.48, 0.65)),
    "race":      ((0.55, 0.88), (0.65, 0.88), (0.80, 0.98), (0.85, 0.98)),
    "long_ride": ((0.12, 0.28), (0.18, 0.30), (0.35, 0.52), (0.55, 0.72)),
}

# Per-workout autonomic stress (relative units; calibrated against
# Stanley 2013 figure 1 recovery curves).
_AUTO_STRESS = {
    "recovery":  0.10,
    "easy":      0.20,
    "endurance": 0.40,
    "tempo":     0.65,
    "sweetspot": 0.75,
    "threshold": 1.00,
    "vo2max":    1.30,
    "sprint":    0.95,
    "race":      1.50,
    "long_ride": 0.85,
}


def _simulate_health_for_day(
    athlete: Athlete,
    rng: np.random.Generator,
    yesterday_workout: str | None,
    yesterday_tss: float,
    days_since_hard: int,
    atl: float,
    ctl: float,
) -> tuple[float, int, float, float]:
    """
    Simulate today's morning health metrics BEFORE today's ride. Returns
    (hrv_overnight_ms, rhr_bpm, sleep_score, body_battery_high).

    Updates `athlete.hrv_baseline` slowly (≈0.1× exponential blend) so the
    7-day rolling baseline behaviour matches Plews & Buchheit's smoothing.
    """
    # Acute autonomic stress from yesterday's ride (decays exponentially).
    stress = _AUTO_STRESS.get(yesterday_workout or "easy", 0.0) if yesterday_workout else 0.0
    # Decay across days since last hard session (Stanley 2013).
    decay = math.exp(-days_since_hard / 1.8)
    acute = stress * decay

    # Chronic fatigue contribution from ATL/CTL ratio (Buchheit 2014).
    chronic = max(0.0, (atl - ctl) / max(ctl, 20.0))
    chronic = min(chronic, 1.5)

    # Drift HRV baseline upward with fitness, downward with chronic fatigue.
    fitness_pull = 0.005 * (40 - (athlete.hrv_baseline - 50))   # very slow regression to mean
    fatigue_pull = -0.4 * chronic
    athlete.hrv_baseline = float(np.clip(
        athlete.hrv_baseline + fitness_pull + fatigue_pull + rng.normal(0, 0.5),
        20.0, 140.0,
    ))

    # Today's overnight HRV: baseline × (1 - acute_drop) × noise
    acute_drop = 0.18 * acute + 0.10 * chronic
    noise = rng.normal(1.0, 0.07)
    hrv_today = float(np.clip(
        athlete.hrv_baseline * (1 - acute_drop) * noise,
        10.0, 160.0,
    ))

    # Resting HR delta (Buchheit 2014): up with acute + chronic load.
    rhr_bump = 4.0 * acute + 3.0 * chronic + rng.normal(0, 1.5)
    rhr_today = int(np.clip(athlete.resting_hr + rhr_bump, 32, 95))

    # Sleep score: degrades with stress; baseline ~ 75.
    sleep_score = float(np.clip(
        82 - 12 * acute - 8 * chronic + rng.normal(0, 7),
        20.0, 100.0,
    ))

    # Body Battery: starts ~ 90, drained by stress + poor sleep.
    body_battery = float(np.clip(
        92 - 25 * acute - 18 * chronic + (sleep_score - 75) * 0.3 + rng.normal(0, 5),
        5.0, 100.0,
    ))

    return hrv_today, rhr_today, sleep_score, body_battery


# ── Simulate a single ride: produce RAW values ────────────────────────────────
def _simulate_ride(
    wt: str, athlete: Athlete, rng: np.random.Generator,
    date: datetime, days_since_last: int,
    ctl: float, atl: float,
    *,
    hrv_today: float = None,
    rhr_today: int = None,
    sleep_score: float = 75.0,
    body_battery: float = 80.0,
) -> tuple[dict, float]:
    """Return (raw_dict, tss). raw_dict keys match the synthetic parquet schema."""
    T = TEMPLATES[wt]

    raw_if   = float(rng.uniform(*T["if_range"]))
    tsb = ctl - atl
    fatigue_penalty = max(0.0, -tsb / 120)  # up to 5% penalty when TSB=-60
    raw_if = max(0.3, raw_if * (1 - fatigue_penalty * 0.05))

    # ── Health-driven intensity penalty ─────────────────────────────────
    # When HRV is suppressed or RHR elevated, the same nominal IF "costs"
    # more — and athletes self-regulate down. Plews & Buchheit (2013) show
    # ≈5–10% reduction in tolerable workload on low-HRV days.
    if hrv_today is not None and hrv_today > 0:
        hrv_drop_pct = max(0.0, (athlete.hrv_baseline - hrv_today) / max(athlete.hrv_baseline, 1))
        raw_if *= (1 - 0.10 * min(hrv_drop_pct, 0.5))
    if rhr_today is not None:
        rhr_delta = rhr_today - athlete.resting_hr
        if rhr_delta > 5:
            raw_if *= (1 - 0.005 * min(rhr_delta - 5, 10))

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

    # ── Risk labels (target supervision for the risk heads) ──────────────
    # OT class is mutually exclusive: 0=overtraining, 1=undertraining, 2=neither.
    # Rules align with PMC interpretation (Coggan / Friel) and Banister model:
    #   overtraining   when TSB very negative or ATL >> CTL (acute spike)
    #   undertraining  when TSB highly positive AND chronic load is light
    #   neither        otherwise
    if tsb < -30 or (atl > 50 and atl > 1.4 * max(ctl, 1.0)):
        risk_ot_class = 0  # overtraining
    elif tsb > 25 and atl < ctl * 0.6:
        risk_ot_class = 1  # undertraining
    else:
        risk_ot_class = 2  # neither
    # Injury / illness risk: count autonomic stress signals (Plews & Buchheit
    # 2013, Buchheit 2014). \u22652 of {HRV suppressed, RHR elevated, sleep low}
    # \u2192 elevated risk. Plus a small baseline illness rate.
    inj_signals = 0
    if hrv_today is not None and athlete.hrv_baseline > 0:
        if (athlete.hrv_baseline - hrv_today) / max(athlete.hrv_baseline, 1.0) > 0.10:
            inj_signals += 1
    if rhr_today is not None and (rhr_today - athlete.resting_hr) > 6:
        inj_signals += 1
    if sleep_score < 60:
        inj_signals += 1
    risk_inj_target = 1 if inj_signals >= 2 else 0
    if rng.random() < 0.02:    # baseline illness/injury surprise
        risk_inj_target = 1

    # Power curve: peak W/kg achieved at each duration in THIS workout.
    # Personal bests (pb_*) are athlete constants; per-workout values are a
    # fraction of the pb, scaled by workout type (sprint → high 5s, threshold
    # → high 20min, etc.).
    ftp_wkg  = athlete.ftp / max(athlete.weight, 1.0)
    pb_5s    = ftp_wkg * athlete.sprint_factor
    pb_1min  = ftp_wkg * athlete.anaerobic_factor
    pb_5min  = ftp_wkg * athlete.vo2max_factor
    pb_20min = ftp_wkg * 1.05  # 20-min ≈ 105% of FTP W/kg

    ef = _PC_EFFORT.get(wt, ((0.20, 0.40), (0.30, 0.50), (0.50, 0.70), (0.60, 0.80)))
    pc_5s_wkg    = float(rng.uniform(pb_5s    * ef[0][0], pb_5s    * ef[0][1]))
    pc_1min_wkg  = float(rng.uniform(pb_1min  * ef[1][0], pb_1min  * ef[1][1]))
    pc_5min_wkg  = float(rng.uniform(pb_5min  * ef[2][0], pb_5min  * ef[2][1]))
    pc_20min_wkg = float(rng.uniform(pb_20min * ef[3][0], pb_20min * ef[3][1]))

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
        "event_type":            athlete.event_type,
        "days_to_event":         max(0.0, (athlete.event_week - (date - athlete.start_date).days / 7.0)) * 7.0,
        "training_days":         float(athlete.training_days),
        # ── Health & recovery (raw values + derived z-score / delta) ────
        "hrv_overnight_ms":      hrv_today if hrv_today is not None else athlete.hrv_baseline,
        "hrv_z":                 (
            (hrv_today - athlete.hrv_baseline) / max(athlete.hrv_baseline * 0.10, 1.0)
            if hrv_today is not None else 0.0
        ),
        "rhr_bpm":               rhr_today if rhr_today is not None else athlete.resting_hr,
        "rhr_delta":             (
            float(rhr_today - athlete.resting_hr) if rhr_today is not None else 0.0
        ),
        "sleep_score":           sleep_score,
        "body_battery":          body_battery,
        # ── Power curve (peak W/kg per duration in this workout) ───────
        "pc_5s_wkg":             pc_5s_wkg,
        "pc_1min_wkg":           pc_1min_wkg,
        "pc_5min_wkg":           pc_5min_wkg,
        "pc_20min_wkg":          pc_20min_wkg,
        # Current best capacity (personal record) — used as adaptation target.
        # Distinct from per-ride peaks which are a noisy fraction of capacity.
        "pc1min_capacity_wkg":   athlete.ftp / max(athlete.weight, 1.0) * athlete.anaerobic_factor,
        "pc5min_capacity_wkg":   athlete.ftp / max(athlete.weight, 1.0) * athlete.vo2max_factor,
        # ── Risk targets (training supervision for risk heads) ──────────
        "risk_ot_class":         int(risk_ot_class),       # 0=over, 1=under, 2=neither
        "risk_inj_target":       int(risk_inj_target),     # 0/1
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


def _adapt_vo2max_factor(
    athlete: "Athlete",
    week_tss_by_type: dict[str, float],
    tsb: float,
    rng: np.random.Generator,
) -> None:
    """Update athlete.vo2max_factor in-place based on weekly training mix.

    VO2max power / FTP ratio responds primarily to supra-threshold intervals
    (4-6 min efforts). Timescale ~6-12 weeks. Without stimulus the ratio drifts
    down as FTP catches up toward the VO2max ceiling.
    """
    tss_v   = week_tss_by_type.get("vo2max", 0.0)
    tss_thr = week_tss_by_type.get("threshold", 0.0) * 0.25  # secondary stimulus

    stimulus = min((tss_v + tss_thr) / 55.0, 1.0) * 0.007   # up to +0.007/week
    decay    = 0.002                                           # natural compression

    recovery_factor = 1.0 if tsb >= -10 else max(0.3, 1.0 + (tsb + 10) / 30)
    net = stimulus * recovery_factor - decay

    ceiling = athlete.vo2max_ceiling_factor
    gap = max(0.0, ceiling - athlete.vo2max_factor)
    net *= math.sqrt(gap / max(ceiling - 1.05, 0.01))         # plateau near ceiling

    athlete.vo2max_factor = float(np.clip(
        athlete.vo2max_factor + net + rng.normal(0, 0.001),
        1.05, ceiling,
    ))


def _adapt_anaerobic_factor(
    athlete: "Athlete",
    week_tss_by_type: dict[str, float],
    tsb: float,
    rng: np.random.Generator,
) -> None:
    """Update athlete.anaerobic_factor in-place based on weekly training mix.

    Peak 1-min power / FTP ratio responds to sprint and short anaerobic work.
    Adapts faster than VO2max (~4-8 weeks) but also detrains faster — pure
    aerobic blocks slowly compress the ratio as FTP grows without anaerobic work.
    """
    tss_spr = week_tss_by_type.get("sprint", 0.0)
    tss_vo2 = week_tss_by_type.get("vo2max", 0.0) * 0.15     # secondary stimulus

    stimulus = min((tss_spr + tss_vo2) / 30.0, 1.0) * 0.010  # up to +0.010/week
    decay    = 0.003                                            # faster detraining

    recovery_factor = 1.0 if tsb >= -15 else max(0.2, 1.0 + (tsb + 15) / 25)
    net = stimulus * recovery_factor - decay

    ceiling = athlete.anaerobic_ceiling_factor
    gap = max(0.0, ceiling - athlete.anaerobic_factor)
    net *= math.sqrt(gap / max(ceiling - 1.20, 0.01))

    athlete.anaerobic_factor = float(np.clip(
        athlete.anaerobic_factor + net + rng.normal(0, 0.001),
        1.20, ceiling,
    ))


# ── Simulate one athlete ──────────────────────────────────────────────────────
def _simulate_athlete(
    athlete: Athlete, rng: np.random.Generator
) -> list[dict]:
    rows: list[dict] = []
    ctl  = athlete.ctl
    atl  = athlete.atl
    date = athlete.start_date
    days_since_last = 3

    yesterday_workout: str | None = None
    yesterday_tss = 0.0
    days_since_hard = 7

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
            days_since_hard += 7
            continue

        schedule_options = SCHEDULES[athlete.philosophy].get(
            phase, SCHEDULES[athlete.philosophy]["base"]
        )
        weekly_wts: list[str] = list(rng.choice(schedule_options))

        # Event-specific bias: re-shape the week so the model sees
        # "athletes preparing for X tend to do more of Y".
        weekly_wts = _apply_event_bias(weekly_wts, athlete.event_type, phase)

        if len(weekly_wts) > athlete.training_days:
            weekly_wts = weekly_wts[: athlete.training_days]

        weeks_to_event = athlete.event_week - week_num
        if athlete.goal == "event_specific" and 1 <= weeks_to_event <= 12 and rng.random() < 0.12:
            weekly_wts[-1] = "race"

        spread_days = sorted(rng.choice(7, size=len(weekly_wts), replace=False))

        week_tss_by_type: dict[str, float] = {}

        for i, wt in enumerate(weekly_wts):
            ride_date = date + timedelta(days=int(spread_days[i]))

            # Compute today's morning health BEFORE the ride.
            hrv_today, rhr_today, sleep_score, body_battery = _simulate_health_for_day(
                athlete, rng,
                yesterday_workout=yesterday_workout,
                yesterday_tss=yesterday_tss,
                days_since_hard=days_since_hard,
                atl=atl, ctl=ctl,
            )

            row, tss = _simulate_ride(
                wt, athlete, rng, ride_date, days_since_last, ctl, atl,
                hrv_today=hrv_today,
                rhr_today=rhr_today,
                sleep_score=sleep_score,
                body_battery=body_battery,
            )

            alpha_ctl = 2 / 43
            alpha_atl = 2 / 8
            ctl = ctl + alpha_ctl * (tss - ctl)
            atl = atl + alpha_atl * (tss - atl)

            week_tss_by_type[wt] = week_tss_by_type.get(wt, 0) + tss
            days_since_last = 1
            yesterday_workout = wt
            yesterday_tss = tss
            if wt in ("threshold", "vo2max", "race", "sprint", "sweetspot"):
                days_since_hard = 0
            else:
                days_since_hard += 1
            rows.append(row)

        # End-of-week multi-system adaptation
        tsb = ctl - atl
        delta = _adapt_ftp(athlete, week_tss_by_type, tsb, rng)
        athlete.ftp = float(np.clip(athlete.ftp + delta, 80, athlete.ftp_ceiling))
        _adapt_vo2max_factor(athlete, week_tss_by_type, tsb, rng)
        _adapt_anaerobic_factor(athlete, week_tss_by_type, tsb, rng)

        date += timedelta(days=7)

    # Horizon probe season (20 % of athletes): append a structured 4-phase
    # block after the real simulation.  Each block has consistent days_to_event
    # + phase-appropriate workouts so the dataset samples fully phase-coherent
    # windows.  This gives the model the contrastive signal it needs to learn
    # Friel periodization: same fitness state, different DTE → different plan.
    if rng.random() < 0.35:
        rows.extend(
            _simulate_probe_season(athlete, rng, date, ctl, atl)
        )

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def _generate_chunk(args: tuple) -> str:
    """Worker function: simulate a contiguous slice of athlete IDs.

    Writes results directly to a temp parquet file and returns the path.
    This avoids pickling millions of dicts through the multiprocessing queue
    which causes OOM on machines with many workers.
    """
    start_id, end_id, n_weeks, base_seed, tmp_dir = args
    rng = np.random.default_rng(base_seed + start_id)
    rows: list[dict] = []
    for i in range(start_id, end_id):
        athlete = _make_athlete(rng, athlete_id=i + 1)
        if n_weeks:
            athlete.n_weeks = n_weeks
            athlete.event_week = min(athlete.event_week, n_weeks)
        rows.extend(_simulate_athlete(athlete, rng))
    # Write to temp file immediately — nothing large leaves this process
    tmp_path = os.path.join(tmp_dir, f"chunk_{start_id}_{end_id}.parquet")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(tmp_path, index=False)
    return tmp_path


def generate(n_athletes: int, n_weeks: int | None, seed: int, output: str,
             workers: int | None = None):
    """Generate synthetic data using multiple CPU cores.

    Parameters
    ----------
    workers : int | None
        Number of parallel processes.  None = auto (logical CPU count – 1,
        min 1).  Set to 1 to disable multiprocessing (simpler stack traces).
    """
    if workers is None:
        workers = max(1, (multiprocessing.cpu_count() or 2) - 1)
    workers = max(1, workers)

    # Use small fixed-size chunks so the progress bar updates frequently.
    # The Pool reuses workers across many chunks (work-stealing), so parallelism
    # is maintained even though there are far more chunks than workers.
    chunk_size = 500
    tmp_dir = tempfile.mkdtemp(prefix="aicoach_gen_")
    chunks = []
    start = 0
    while start < n_athletes:
        end = min(start + chunk_size, n_athletes)
        chunks.append((start, end, n_weeks, seed, tmp_dir))
        start = end

    print(f"Generating {n_athletes:,} athletes using {len(chunks)} workers "
          f"(~{chunk_size} athletes each) …")

    if workers == 1 or len(chunks) == 1:
        # Single-process path — keeps nice tqdm progress bar.
        tmp_files: list[str] = []
        for chunk in tqdm(chunks, desc="Simulating athletes (single-process)"):
            tmp_files.append(_generate_chunk(chunk))
    else:
        # Multi-process path — workers write to temp files, return paths only.
        with multiprocessing.Pool(processes=workers) as pool:
            tmp_files = list(
                tqdm(
                    pool.imap(_generate_chunk, chunks),
                    total=len(chunks),
                    desc=f"Simulating athletes ({workers} workers)",
                )
            )

    print(f"\nMerging {len(tmp_files)} chunk files…")

    # Stream-merge: write one chunk at a time so we never hold 14M rows in RAM.
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    writer = None
    total_rows = 0
    for f in tmp_files:
        table = pq.read_table(f)
        if writer is None:
            writer = pq.ParquetWriter(output, table.schema, compression="snappy")
        writer.write_table(table)
        total_rows += len(table)
        del table
        try:
            os.remove(f)
        except OSError:
            pass

    if writer:
        writer.close()
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(f"Total rides simulated: {total_rows:,}")

    # Sort in one pass using pyarrow (avoids a second full pandas DataFrame).
    print("Sorting by athlete_id + date…")
    full = pq.read_table(output)
    full = full.sort_by([("athlete_id", "ascending"), ("date", "ascending")])
    pq.write_table(full, output, compression="snappy")

    print(f"\nSaved → {output}")
    print(f"File size: {os.path.getsize(output) / 1e6:.1f} MB")

    df = full.to_pandas()
    del full
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"FTP range: {df['ftp'].min():.0f} – {df['ftp'].max():.0f} W")
    print(f"\nWorkout type distribution:\n{df['workout_type'].value_counts().to_string()}")
    print(f"\nPhilosophy distribution:\n{df.drop_duplicates('athlete_id')['philosophy'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    # Required on Windows: multiprocessing spawns new interpreter instances,
    # so the entry-point must be guarded by if __name__ == "__main__".
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Generate synthetic cycling training data for transformer pre-training"
    )
    parser.add_argument("--athletes", type=int, default=50_000)
    parser.add_argument("--weeks",    type=int, default=None,
                        help="Override weeks per athlete (default: random 40–104)")
    parser.add_argument("--output",   default="./ml/data/synthetic.parquet")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--workers",  type=int, default=None,
                        help="Parallel workers (default: CPU count – 1)")
    args = parser.parse_args()
    generate(args.athletes, args.weeks, args.seed, args.output, args.workers)

