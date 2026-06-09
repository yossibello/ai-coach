"""
Model sanity check: two synthetic athletes, three horizons each.

Athlete A — BASE PERIOD
  90 days of endurance/easy rides, CTL ~35, low intensity.
  Expected: long horizon → base→build→peak→taper arc
            short horizon → sharpen quickly

Athlete B — HIGH INTENSITY BLOCK
  90 days mixing VO2max/threshold/sweetspot, CTL ~60.
  Expected: different workout mix, model should recognise already-hard load
            and not pile on more VO2max on all horizons.

Run from repo root:
    PYTHONPATH=backend python ml/sanity_check.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
# Kaggle doesn't have asyncpg — set a dummy URL so database.py doesn't crash on import
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///tmp/dummy.db")

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np

# ── Minimal stubs so we can import without a real DB ──────────────────────────

class FakeActivity:
    def __init__(self, date, workout_type, tss, avg_power, duration_s, ftp=220):
        self.date                  = date
        self.workout_type          = workout_type
        self.tss                   = tss
        self.avg_power             = avg_power
        self.normalized_power      = avg_power * 1.05
        self.intensity_factor      = avg_power / ftp
        self.duration_seconds      = duration_s
        self.distance_meters       = avg_power * duration_s / 40  # rough proxy
        self.elevation_gain_meters = 200
        self.avg_hr                = 130 + int(avg_power / ftp * 40)
        self.max_hr                = self.avg_hr + 20
        self.max_power             = avg_power * 1.3
        self.cadence               = 88
        self.temperature_c         = 18.0
        self.humidity_pct          = 55.0
        self.hr_drift_pct          = 3.0
        self.hr_drift              = 3.0
        self.variability_index     = avg_power * 1.05 / avg_power if avg_power else 1.05
        self.aerobic_efficiency    = avg_power / max(130 + int(avg_power / ftp * 40), 1)
        self.avg_cadence           = 88
        self.time_in_zones         = None
        self.wind_speed_kmh        = None
        self.perceived_exertion    = None
        self.aerobic_efficiency    = None
        self.rpe                   = None
        self.health_modifier       = 0.0
        self.pc_5s_wkg             = None
        self.pc_1min_wkg           = None
        self.pc_5min_wkg           = None
        self.pc_20min_wkg          = None
        self.review_status         = "confirmed"
        self.external_id           = None

class FakeProfile:
    def __init__(self, ftp=220, goal_event_date=None, event_type="gran_fondo",
                 training_days_per_week=5, weight_kg=70, age=35):
        self.ftp                    = ftp
        self.goal_event_date        = goal_event_date
        self.event_type             = event_type
        self.training_days_per_week = training_days_per_week
        self.weight_kg              = weight_kg
        self.height_cm              = 175
        self.age                    = age
        self.max_hr                 = 185
        self.resting_hr             = 50
        self.pc5min_capacity_wkg    = None
        self.pc1min_capacity_wkg    = None
        self.goal_type              = "event"
        self.sex                    = "male"
        self.cycling_experience_years = 5
        self.primary_goal           = "event"


def make_base_athlete(n_days=90, ftp=220):
    """Zone 2 / endurance focus. Low IF, medium TSS, steady."""
    base = datetime.now(timezone.utc) - timedelta(days=n_days)
    rides = []
    for i in range(n_days):
        if i % 7 == 6:          # rest day — skip
            continue
        if i % 7 == 5:          # long ride saturday
            wt, power, dur, tss = "long_ride", int(ftp*0.65), 4*3600, 110
        elif i % 7 == 2:        # mid-week easy
            wt, power, dur, tss = "easy", int(ftp*0.58), 1*3600, 38
        else:                   # endurance default
            wt, power, dur, tss = "endurance", int(ftp*0.65), 90*60, 70
        rides.append(FakeActivity(base + timedelta(days=i), wt, tss, power, dur, ftp))
    return rides


def make_intensity_athlete(n_days=90, ftp=250):
    """VO2max + threshold block. High IF, high CTL."""
    base = datetime.now(timezone.utc) - timedelta(days=n_days)
    week = [
        ("vo2max",    int(ftp*1.10), 70*60,  95),   # Mon
        ("endurance", int(ftp*0.65), 90*60,  70),   # Tue
        ("threshold", int(ftp*1.00), 75*60,  90),   # Wed
        ("recovery",  int(ftp*0.45), 50*60,  25),   # Thu
        ("sweetspot", int(ftp*0.90), 80*60,  85),   # Fri
        ("long_ride", int(ftp*0.65), 4*3600, 120),  # Sat
        None,                                         # Sun rest
    ]
    rides = []
    for i in range(n_days):
        slot = week[i % 7]
        if slot is None:
            continue
        wt, power, dur, tss = slot
        rides.append(FakeActivity(base + timedelta(days=i), wt, tss, power, dur, ftp))
    return rides


def compute_ctl_atl(rides):
    """Simple EMA forward pass to get CTL/ATL/TSB."""
    ctl, atl = 0.0, 0.0
    for r in sorted(rides, key=lambda x: x.date):
        tss = r.tss or 0
        ctl = ctl + (2/43) * (tss - ctl)
        atl = atl + (2/8)  * (tss - atl)
    return round(ctl, 1), round(atl, 1), round(ctl - atl, 1)


def run_check():
    import os, torch
    from app.ml.model import CyclingTransformer
    from app.ml.inference import _transformer_recommendation

    model_path = os.path.join(os.path.dirname(__file__), "..", "backend", "models", "cycling_coach.pt")
    print(f"Loading model from {model_path}…")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg  = ckpt.get("config", {})
    model = CyclingTransformer(
        d_model         = cfg.get("d_model", 256),
        nhead           = cfg.get("nhead", 8),
        num_layers      = cfg.get("num_layers", 8),
        dim_feedforward = cfg.get("dim_feedforward", 1024),
    )
    sd = ckpt["state_dict"]
    if next(iter(sd)).startswith("_orig_mod."):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params\n")

    event_date = datetime.now(timezone.utc) + timedelta(days=90)
    horizons   = {"short (7d)": 7, "medium (28d)": 28, "event (90d)": 90}

    athletes = {
        "BASE athlete  (endurance block, FTP 220)": (
            make_base_athlete(ftp=220),
            FakeProfile(ftp=220, goal_event_date=event_date)
        ),
        "INTENSITY athlete (VO2max block, FTP 250)": (
            make_intensity_athlete(ftp=250),
            FakeProfile(ftp=250, goal_event_date=event_date)
        ),
    }

    for athlete_name, (rides, profile) in athletes.items():
        ctl, atl, tsb = compute_ctl_atl(rides)
        print("=" * 70)
        print(f"  {athlete_name}")
        print(f"  CTL={ctl}  ATL={atl}  TSB={tsb}  rides={len(rides)}")
        print("=" * 70)

        for h_label, h_days in horizons.items():
            payload = _transformer_recommendation(
                model, rides[-90:], profile,
                ctl=ctl, atl=atl, tsb=tsb,
                horizon_override_days=h_days,
            )
            nw   = payload["next_workout"]
            plan = payload["weekly_plan"]
            wt_types = [w["workout_type"] for w in plan]
            print(f"\n  ── Horizon: {h_label} ──────────────────────────────────")
            print(f"  Next workout : {nw['workout_type']:15s}  IF≈{nw.get('target_tss',0)/max(nw['duration_minutes']/60,0.1):.2f}  TSS={nw['target_tss']}  {nw['duration_minutes']}min")
            print(f"  Weekly plan  : {' → '.join(wt_types)}")
            print(f"  Rationale    : {nw['rationale'][:80]}")
        print()


if __name__ == "__main__":
    run_check()
