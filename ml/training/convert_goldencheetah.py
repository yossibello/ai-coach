"""
Convert GoldenCheetah OpenData → our training parquet schema.

Handles TWO formats automatically:

  FORMAT A — Kaggle summary CSVs (markliversedge/goldencheetah-opendata-athlete-activity-and-mmp)
    activities.csv      : one row per activity, pre-aggregated metrics
    activities_mmp.csv  : matching MMP (peak power) per activity, columns like "1s","5s","1m","5m","20m","60m"
    athletes.csv        : athlete metadata (weight, height, sex, yob)

  FORMAT B — Raw GoldenCheetah OpenData GitHub repo
    <athlete_id>/RIDES/<date>.csv  : second-by-second (secs, km, power, hr, cad, alt)
    <athlete_id>/athlete.json      : metadata

Usage:
  # Kaggle summary format:
  python -m ml.training.convert_goldencheetah \\
      --input  /kaggle/input/goldencheetah-opendata-athlete-activity-and-mmp \\
      --output ml/data/goldencheetah.parquet

  # Raw per-second format:
  python -m ml.training.convert_goldencheetah \\
      --input  /path/to/OpenData \\
      --output ml/data/goldencheetah.parquet \\
      --format raw

  # Preview available columns without writing output:
  python -m ml.training.convert_goldencheetah --input /path --preview
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)


# ── Workout-type inference from IF + duration ─────────────────────────────────
def _infer_workout_type(if_: float, dur_h: float, vi: float = 1.0) -> str:
    if dur_h >= 3.0 and if_ < 0.78:
        return "long_ride"
    if vi >= 1.30 and if_ >= 0.85:
        return "sprint"
    if if_ < 0.50:
        return "recovery"
    if if_ < 0.65:
        return "easy"
    if if_ < 0.75:
        return "endurance"
    if if_ < 0.85:
        return "tempo"
    if if_ < 0.93:
        return "sweetspot"
    if if_ < 1.05:
        return "threshold"
    return "vo2max"


# ── Coggan zone fractions from IF (simplified) ───────────────────────────────
# Returns [z1, z2, z3, z4, z5, z6, z7] summing to 1.0
_ZONE_PROFILES: dict[str, list[float]] = {
    "recovery":   [0.08, 0.82, 0.10, 0.00, 0.00, 0.00, 0.00],
    "easy":       [0.04, 0.82, 0.13, 0.01, 0.00, 0.00, 0.00],
    "endurance":  [0.02, 0.74, 0.21, 0.03, 0.00, 0.00, 0.00],
    "tempo":      [0.00, 0.18, 0.50, 0.27, 0.05, 0.00, 0.00],
    "sweetspot":  [0.00, 0.04, 0.18, 0.36, 0.37, 0.05, 0.00],
    "threshold":  [0.00, 0.04, 0.10, 0.10, 0.52, 0.20, 0.04],
    "vo2max":     [0.00, 0.20, 0.18, 0.04, 0.14, 0.36, 0.08],
    "sprint":     [0.00, 0.40, 0.28, 0.04, 0.04, 0.06, 0.18],
    "long_ride":  [0.02, 0.70, 0.23, 0.04, 0.01, 0.00, 0.00],
}


def _zone_fractions(wt: str) -> list[float]:
    return _ZONE_PROFILES.get(wt, _ZONE_PROFILES["endurance"])


# ── CTL / ATL / TSB computation ───────────────────────────────────────────────
def _compute_fitness(tss_series: pd.Series, dates: pd.Series) -> pd.DataFrame:
    """Compute CTL (42d), ATL (7d), TSB for one athlete's sorted activity list."""
    # Fill in missing days with TSS=0 so EWM is day-accurate
    idx = pd.date_range(dates.min(), dates.max(), freq="D")
    daily = (pd.Series(tss_series.values, index=pd.DatetimeIndex(dates))
             .groupby(level=0).sum()
             .reindex(idx, fill_value=0.0))

    ctl = daily.ewm(span=42, adjust=False).mean()
    atl = daily.ewm(span=7, adjust=False).mean()
    tsb = ctl - atl

    result = pd.DataFrame({"date": idx, "ctl": ctl.values,
                            "atl": atl.values, "tsb": tsb.values})
    return result.set_index("date")


# ── FTP estimation ────────────────────────────────────────────────────────────
def _estimate_ftp(mmp_60m: float | None, mmp_20m: float | None,
                  mmp_5m: float | None) -> float | None:
    if mmp_60m and mmp_60m > 0:
        return mmp_60m
    if mmp_20m and mmp_20m > 0:
        return mmp_20m * 0.95
    if mmp_5m and mmp_5m > 0:
        return mmp_5m * 0.78   # rough proxy
    return None


# ── Column name aliases ───────────────────────────────────────────────────────
# GoldenCheetah exports use many different names depending on version / locale.
def _col(df: pd.DataFrame, *candidates, default=None):
    """Return the first matching column (case-insensitive), or default."""
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return df[low[c.lower()]]
    if default is not None:
        return pd.Series([default] * len(df), index=df.index)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT A — Kaggle summary CSVs
# ─────────────────────────────────────────────────────────────────────────────
def convert_summary(input_dir: Path, output: Path, min_rides: int = 20) -> pd.DataFrame:
    print("Format: Kaggle summary CSVs")

    # ── Load files ───────────────────────────────────────────────────────────
    act_path = next(input_dir.glob("*activit*.csv"), None)
    mmp_path = next((p for p in input_dir.glob("*.csv")
                     if "mmp" in p.name.lower()), None)
    ath_path = next((p for p in input_dir.glob("*.csv")
                     if "athlete" in p.name.lower()), None)

    if act_path is None:
        raise FileNotFoundError(f"No activities CSV found in {input_dir}")

    print(f"  Activities: {act_path.name}")
    acts = pd.read_csv(act_path, low_memory=False)
    print(f"  Loaded {len(acts):,} rows, {acts.shape[1]} columns")
    print(f"  Columns: {list(acts.columns[:20])} ...")

    mmp = pd.read_csv(mmp_path, low_memory=False) if mmp_path else None
    if mmp is not None:
        print(f"  MMP:        {mmp_path.name}  ({len(mmp):,} rows)")
        print(f"  MMP cols:   {list(mmp.columns[:15])}")

    athletes = pd.read_csv(ath_path, low_memory=False) if ath_path else None
    if athletes is not None:
        print(f"  Athletes:   {ath_path.name}  ({len(athletes):,} rows)")
        print(f"  Ath cols:   {list(athletes.columns)}")

    # ── Normalise activity columns ───────────────────────────────────────────
    # Athlete ID
    aid = _col(acts, "athlete_id", "athlete", "id", "rider_id")
    if aid is None:
        raise ValueError("Cannot find athlete ID column in activities CSV")
    acts["_aid"] = aid.astype(str)

    # Date
    date_col = _col(acts, "date", "activity_date", "start_date", "Date")
    if date_col is None:
        raise ValueError("Cannot find date column in activities CSV")
    acts["_date"] = pd.to_datetime(date_col, infer_datetime_format=True, errors="coerce")
    acts = acts.dropna(subset=["_date"])

    # Sport — keep cycling only
    sport_col = _col(acts, "sport", "Sport", "activity_type", "type")
    if sport_col is not None:
        mask = sport_col.astype(str).str.lower().str.contains("bike|cycl|ride", na=False)
        acts = acts[mask].copy()
        print(f"  Cycling rows: {len(acts):,}")
    else:
        print("  No sport column found — keeping all rows")

    # Core metrics
    dur_s  = _col(acts, "duration", "Duration", "total_time", "secs",
                  "moving_time", "elapsed_time", default=0.0).astype(float)
    dist_m = _col(acts, "distance", "Distance", "total_distance",
                  "km", default=0.0).astype(float)
    # Convert km → meters if values look like km (median < 500)
    if dist_m.median() < 500:
        dist_m = dist_m * 1000.0

    elev_m = _col(acts, "elevation", "Elevation Gain", "elevation_gain",
                  "total_elevation_gain", "ascent", default=0.0).astype(float)

    avg_p  = _col(acts, "power", "average_power", "avg_power", "Average Power",
                  "power_avg", default=0.0).astype(float)
    np_p   = _col(acts, "normalized_power", "np", "NP", "Normalized Power",
                  "power_np", default=None)
    if np_p is None:
        np_p = avg_p.copy()
    else:
        np_p = np_p.astype(float).fillna(avg_p)

    max_p  = _col(acts, "max_power", "power_max", "Max Power",
                  default=0.0).astype(float)

    avg_hr = _col(acts, "avg_hr", "average_hr", "heart_rate", "Average HR",
                  "hr", "avg_heart_rate", default=0.0).astype(float)
    max_hr = _col(acts, "max_hr", "Max HR", "max_heart_rate",
                  default=0.0).astype(float).fillna(avg_hr)

    cad    = _col(acts, "avg_cadence", "cadence", "average_cadence", "Cadence",
                  default=85.0).astype(float)

    tss_c  = _col(acts, "tss", "TSS", "training_stress_score",
                  default=None)

    if_c   = _col(acts, "if", "intensity_factor", "IF",
                  default=None)

    vi_c   = _col(acts, "vi", "variability_index", "VI",
                  default=None)

    # ── Merge MMP data ───────────────────────────────────────────────────────
    mmp_merged = None
    if mmp is not None:
        # Find join key in MMP
        mmp_aid = _col(mmp, "athlete_id", "athlete", "id", "rider_id")
        mmp_date = _col(mmp, "date", "activity_date", "Date")
        if mmp_aid is not None and mmp_date is not None:
            mmp["_aid"] = mmp_aid.astype(str)
            mmp["_date"] = pd.to_datetime(mmp_date, infer_datetime_format=True, errors="coerce")
            acts_key = acts[["_aid", "_date"]].copy()
            mmp_merged = acts_key.merge(mmp, on=["_aid", "_date"], how="left")

    def _mmp_col(duration_label: str) -> pd.Series:
        """Try to extract MMP at given duration from the merged MMP dataframe."""
        if mmp_merged is None:
            return pd.Series([np.nan] * len(acts), index=acts.index)
        candidates = [duration_label, f"mmp_{duration_label}",
                      f"p{duration_label}", duration_label.replace("m", "min")]
        for c in candidates:
            if c in mmp_merged.columns:
                return mmp_merged[c].astype(float).values
            cl = c.lower()
            match = [col for col in mmp_merged.columns if col.lower() == cl]
            if match:
                return mmp_merged[match[0]].astype(float).values
        return pd.Series([np.nan] * len(acts), index=acts.index)

    mmp_5s  = _mmp_col("5s")
    mmp_1m  = _mmp_col("1m")
    mmp_5m  = _mmp_col("5m")
    mmp_20m = _mmp_col("20m")
    mmp_60m = _mmp_col("60m")

    # ── Athlete metadata ─────────────────────────────────────────────────────
    ath_meta: dict[str, dict] = {}
    if athletes is not None:
        ath_id_col = _col(athletes, "athlete_id", "athlete", "id", "rider_id")
        if ath_id_col is not None:
            athletes["_aid"] = ath_id_col.astype(str)
            for _, row in athletes.iterrows():
                aid_val = str(row["_aid"])
                weight = float(row.get("weight", row.get("Weight", 70)) or 70)
                height = float(row.get("height", row.get("Height", 175)) or 175)
                sex_raw = str(row.get("sex", row.get("gender", row.get("Sex", "M"))) or "M")
                sex = "male" if sex_raw.upper().startswith("M") else "female"
                yob = int(row.get("yob", row.get("year_of_birth", 1985)) or 1985)
                age = 2020 - yob   # approximate mid-dataset year
                ath_meta[aid_val] = {"weight_kg": weight, "height_cm": height,
                                     "sex": sex, "age": age}

    # ── Build output rows ────────────────────────────────────────────────────
    athlete_weight_col = _col(acts, "athlete_weight", "weight", default=70.0).astype(float)

    rows = []
    acts_reset = acts.reset_index(drop=True)

    # Group by athlete to compute CTL/ATL/TSB and rolling FTP
    for aid_val, grp_idx in acts_reset.groupby("_aid").groups.items():
        grp = acts_reset.loc[grp_idx].sort_values("_date").copy()
        if len(grp) < min_rides:
            continue

        meta = ath_meta.get(str(aid_val), {})
        weight_kg = meta.get("weight_kg", 70.0)
        height_cm = meta.get("height_cm", 175.0)
        sex       = meta.get("sex", "male")
        age       = meta.get("age", 35)
        max_hr_ath = int(208 - 0.7 * age)
        resting_hr = 55

        # Per-activity values for this group
        g_dur   = dur_s.iloc[grp_idx].values
        g_dist  = dist_m.iloc[grp_idx].values
        g_elev  = elev_m.iloc[grp_idx].values
        g_avgp  = avg_p.iloc[grp_idx].values
        g_npp   = np_p.iloc[grp_idx].values
        g_maxp  = max_p.iloc[grp_idx].values
        g_avghr = avg_hr.iloc[grp_idx].values
        g_maxhr = max_hr.iloc[grp_idx].values
        g_cad   = cad.iloc[grp_idx].values

        g_mmp5s  = np.asarray(mmp_5s.iloc[grp_idx]  if hasattr(mmp_5s,  "iloc") else [np.nan]*len(grp))
        g_mmp1m  = np.asarray(mmp_1m.iloc[grp_idx]  if hasattr(mmp_1m,  "iloc") else [np.nan]*len(grp))
        g_mmp5m  = np.asarray(mmp_5m.iloc[grp_idx]  if hasattr(mmp_5m,  "iloc") else [np.nan]*len(grp))
        g_mmp20m = np.asarray(mmp_20m.iloc[grp_idx] if hasattr(mmp_20m, "iloc") else [np.nan]*len(grp))
        g_mmp60m = np.asarray(mmp_60m.iloc[grp_idx] if hasattr(mmp_60m, "iloc") else [np.nan]*len(grp))

        # FTP per ride: rolling best 60-min (or 20-min×0.95), expanded forward
        ftp_series = []
        ftp_rolling = None
        for i in range(len(grp)):
            ride_ftp = _estimate_ftp(
                g_mmp60m[i] if not np.isnan(g_mmp60m[i]) else None,
                g_mmp20m[i] if not np.isnan(g_mmp20m[i]) else None,
                g_mmp5m[i]  if not np.isnan(g_mmp5m[i])  else None,
            )
            if ride_ftp and ride_ftp > 80:
                ftp_rolling = ride_ftp
            ftp_series.append(ftp_rolling)

        # Backfill first rides with first known FTP
        first_ftp = next((f for f in ftp_series if f), 200.0)
        ftp_series = [f if f else first_ftp for f in ftp_series]
        ftp_arr = np.array(ftp_series, dtype=np.float32)

        # TSS — compute from NP + FTP if not in dataset
        tss_arr = tss_c.iloc[grp_idx].astype(float).values if tss_c is not None else np.full(len(grp), np.nan)
        for i in range(len(grp)):
            if np.isnan(tss_arr[i]) or tss_arr[i] == 0:
                f = ftp_arr[i]
                np_w = g_npp[i]
                dur  = g_dur[i]
                if f > 0 and np_w > 0 and dur > 0:
                    if_ = np_w / f
                    tss_arr[i] = (dur / 3600) * if_ * if_ * 100
                else:
                    tss_arr[i] = 0.0

        # IF
        if_arr = (if_c.iloc[grp_idx].astype(float).values
                  if if_c is not None else np.full(len(grp), np.nan))
        for i in range(len(grp)):
            if np.isnan(if_arr[i]) or if_arr[i] == 0:
                f = ftp_arr[i]
                if_arr[i] = g_npp[i] / f if (f > 0 and g_npp[i] > 0) else 0.0

        # VI
        vi_arr = (vi_c.iloc[grp_idx].astype(float).values
                  if vi_c is not None else np.full(len(grp), np.nan))
        for i in range(len(grp)):
            if np.isnan(vi_arr[i]) or vi_arr[i] < 1.0:
                vi_arr[i] = g_npp[i] / g_avgp[i] if g_avgp[i] > 0 else 1.0

        # CTL / ATL / TSB
        dates_arr = grp["_date"].values
        fitness = _compute_fitness(pd.Series(tss_arr), pd.Series(pd.DatetimeIndex(dates_arr)))

        ctl_arr = np.array([fitness.loc[d, "ctl"] if d in fitness.index else 0.0
                            for d in pd.DatetimeIndex(dates_arr)], dtype=np.float32)
        atl_arr = np.array([fitness.loc[d, "atl"] if d in fitness.index else 0.0
                            for d in pd.DatetimeIndex(dates_arr)], dtype=np.float32)
        tsb_arr = ctl_arr - atl_arr

        # Days since last ride
        dates_dt = pd.DatetimeIndex(dates_arr)
        days_since = np.zeros(len(grp), dtype=np.float32)
        for i in range(1, len(grp)):
            days_since[i] = (dates_dt[i] - dates_dt[i - 1]).days

        # Rolling personal-best power curves
        pc1min_best = np.zeros(len(grp), dtype=np.float32)
        pc5min_best = np.zeros(len(grp), dtype=np.float32)
        best_1m = best_5m = 0.0
        for i in range(len(grp)):
            wkg_1m = g_mmp1m[i] / weight_kg if (not np.isnan(g_mmp1m[i]) and weight_kg > 0) else 0.0
            wkg_5m = g_mmp5m[i] / weight_kg if (not np.isnan(g_mmp5m[i]) and weight_kg > 0) else 0.0
            best_1m = max(best_1m, wkg_1m)
            best_5m = max(best_5m, wkg_5m)
            pc1min_best[i] = best_1m
            pc5min_best[i] = best_5m

        # Build rows
        n_athlete_rides = 0
        for i in range(len(grp)):
            ftp   = float(ftp_arr[i])
            dur_h = g_dur[i] / 3600.0
            if_v  = float(if_arr[i])
            vi_v  = float(vi_arr[i])
            wt    = _infer_workout_type(if_v, dur_h, vi_v)
            zf    = _zone_fractions(wt)

            wkg_5s  = g_mmp5s[i]  / weight_kg if (not np.isnan(g_mmp5s[i])  and weight_kg > 0) else 0.0
            wkg_1m  = g_mmp1m[i]  / weight_kg if (not np.isnan(g_mmp1m[i])  and weight_kg > 0) else 0.0
            wkg_5m  = g_mmp5m[i]  / weight_kg if (not np.isnan(g_mmp5m[i])  and weight_kg > 0) else 0.0
            wkg_20m = g_mmp20m[i] / weight_kg if (not np.isnan(g_mmp20m[i]) and weight_kg > 0) else 0.0

            rows.append({
                "athlete_id":            str(aid_val),
                "date":                  dates_arr[i],
                # Activity
                "duration_seconds":      float(g_dur[i]),
                "distance_meters":       float(g_dist[i]),
                "elevation_gain_meters": float(g_elev[i]),
                "avg_power":             float(g_avgp[i]),
                "normalized_power":      float(g_npp[i]),
                "max_power":             float(g_maxp[i]),
                "intensity_factor":      if_v,
                "variability_index":     vi_v,
                "tss":                   float(tss_arr[i]),
                "avg_hr":                float(g_avghr[i]),
                "max_hr":                float(g_maxhr[i]),
                "avg_cadence":           float(g_cad[i]),
                "hr_drift":              0.0,
                "aerobic_efficiency":    (g_avgp[i] / g_avghr[i]
                                          if g_avghr[i] > 0 and g_avgp[i] > 0 else 0.0),
                # Zones (inferred)
                "z1": zf[0], "z2": zf[1], "z3": zf[2], "z4": zf[3],
                "z5": zf[4], "z6": zf[5], "z7": zf[6],
                # Environment defaults
                "temperature_c":   18.0,
                "humidity_pct":    50.0,
                "wind_speed_kmh":  0.0,
                "perceived_exertion": 5.0,
                # Fitness state
                "ctl":                float(ctl_arr[i]),
                "atl":                float(atl_arr[i]),
                "tsb":                float(tsb_arr[i]),
                "days_since_last_ride": float(days_since[i]),
                # Workout classification
                "workout_type":      wt,
                # Profile (repeated per row — same as synthetic parquet)
                "ftp":               ftp,
                "athlete_max_hr":    float(max_hr_ath),
                "resting_hr":        float(resting_hr),
                "weight_kg":         float(weight_kg),
                "height_cm":         float(height_cm),
                "sex":               sex,
                "age":               float(age),
                "experience_years":  3.0,
                "primary_goal":      "general_fitness",
                "training_days":     5,
                "days_to_event":     -1.0,
                # Power curve
                "pc_5s_wkg":         float(np.nan_to_num(wkg_5s)),
                "pc_1min_wkg":       float(np.nan_to_num(wkg_1m)),
                "pc_5min_wkg":       float(np.nan_to_num(wkg_5m)),
                "pc_20min_wkg":      float(np.nan_to_num(wkg_20m)),
                # Rolling personal bests
                "pc1min_capacity_wkg": float(pc1min_best[i]),
                "pc5min_capacity_wkg": float(pc5min_best[i]),
                # Risk labels — unknown for real data, use safe defaults
                "risk_ot_class":     2,    # "neither" (no overtraining signal)
                "risk_inj_target":   0.0,
                # Health/recovery — not in dataset, use neutral defaults
                "hrv_z":             0.0,
                "rhr_delta":         0.0,
                "sleep_score":       50.0,
                "body_battery":      50.0,
            })
            n_athlete_rides += 1

    print(f"  Converted {len(rows):,} rides from {len(set(r['athlete_id'] for r in rows))} athletes")
    return pd.DataFrame(rows)


# ── FORMAT B — raw per-second CSVs ────────────────────────────────────────────
def convert_raw(input_dir: Path, output: Path, min_rides: int = 20) -> pd.DataFrame:
    """Convert raw GoldenCheetah per-second CSVs (secs, km, power, hr, cad, alt)."""
    print("Format: raw per-second CSVs")
    athlete_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    print(f"  Found {len(athlete_dirs)} athlete directories")

    rows = []
    for ath_dir in athlete_dirs:
        ride_dir = ath_dir / "RIDES"
        if not ride_dir.exists():
            ride_dir = ath_dir
        ride_files = sorted(ride_dir.glob("*.csv"))
        if len(ride_files) < min_rides:
            continue

        # Load optional metadata
        meta_file = ath_dir / "athlete.json"
        meta = {}
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
            except Exception:
                pass

        weight_kg  = float(meta.get("weight",  70))
        height_cm  = float(meta.get("height",  175))
        sex_raw    = str(meta.get("sex", meta.get("gender", "M")))
        sex        = "male" if sex_raw.upper().startswith("M") else "female"
        yob        = int(meta.get("dob", meta.get("yob", 1985)) or 1985)
        if yob > 1900:
            age = 2020 - yob
        else:
            age = 35
        max_hr_ath = int(208 - 0.7 * age)

        aid = ath_dir.name
        ride_data = []

        for f in ride_files:
            try:
                df = pd.read_csv(f, low_memory=False)
            except Exception:
                continue
            if df.empty or "secs" not in df.columns:
                continue

            dur_s  = float(df["secs"].max())
            dist_m = float(df["km"].max() * 1000) if "km" in df.columns else 0.0
            elev_m = float((df["alt"].diff().clip(lower=0).sum())
                           if "alt" in df.columns else 0.0)
            power  = df["power"].fillna(0) if "power" in df.columns else pd.Series([0.0]*len(df))
            hr     = df["hr"].fillna(0)    if "hr"    in df.columns else pd.Series([0.0]*len(df))
            cad    = df["cad"].fillna(85)  if "cad"   in df.columns else pd.Series([85.0]*len(df))

            avg_p = float(power[power > 0].mean() or 0)
            # Normalized power: 30-sec rolling RMS
            if len(power) > 30 and power.max() > 0:
                roll = power.rolling(30, min_periods=1).mean()
                np_w = float((roll**4).mean()**0.25)
            else:
                np_w = avg_p
            max_p  = float(power.max())
            avg_hr = float(hr[hr > 0].mean() or 0)
            max_hr = float(hr.max() or 0)
            avg_cad = float(cad[cad > 0].mean() or 85)

            # MMP: peak mean power over each duration
            def _mmp(seconds):
                if len(power) < seconds or power.max() == 0:
                    return np.nan
                return float(power.rolling(seconds).mean().max())

            mmp5s  = _mmp(5)
            mmp1m  = _mmp(60)
            mmp5m  = _mmp(300)
            mmp20m = _mmp(1200)
            mmp60m = _mmp(3600)

            # Parse date from filename (YYYY-MM-DD or similar)
            date_str = f.stem[:10]
            try:
                date = pd.Timestamp(date_str)
            except Exception:
                continue

            ride_data.append({
                "date": date, "dur_s": dur_s, "dist_m": dist_m, "elev_m": elev_m,
                "avg_p": avg_p, "np_w": np_w, "max_p": max_p,
                "avg_hr": avg_hr, "max_hr": max_hr, "avg_cad": avg_cad,
                "mmp5s": mmp5s, "mmp1m": mmp1m, "mmp5m": mmp5m,
                "mmp20m": mmp20m, "mmp60m": mmp60m,
            })

        if len(ride_data) < min_rides:
            continue

        ride_df = pd.DataFrame(ride_data).sort_values("date").reset_index(drop=True)

        # FTP rolling best
        ftp_series = []
        ftp_rolling = None
        for _, r in ride_df.iterrows():
            ftp_est = _estimate_ftp(
                r["mmp60m"] if not np.isnan(r["mmp60m"]) else None,
                r["mmp20m"] if not np.isnan(r["mmp20m"]) else None,
                r["mmp5m"]  if not np.isnan(r["mmp5m"])  else None,
            )
            if ftp_est and ftp_est > 80:
                ftp_rolling = ftp_est
            ftp_series.append(ftp_rolling)
        first_ftp = next((f for f in ftp_series if f), 200.0)
        ftp_arr = np.array([f if f else first_ftp for f in ftp_series], dtype=np.float32)

        for i, (_, r) in enumerate(ride_df.iterrows()):
            ftp   = float(ftp_arr[i])
            np_w  = float(r["np_w"])
            avg_p = float(r["avg_p"])
            dur_h = float(r["dur_s"]) / 3600.0
            if_v  = np_w / ftp if ftp > 0 and np_w > 0 else 0.0
            vi_v  = np_w / avg_p if avg_p > 0 else 1.0
            tss   = dur_h * if_v * if_v * 100 if if_v > 0 else 0.0
            wt    = _infer_workout_type(if_v, dur_h, vi_v)
            zf    = _zone_fractions(wt)

            wkg_5s  = r["mmp5s"]  / weight_kg if not np.isnan(r["mmp5s"])  and weight_kg > 0 else 0.0
            wkg_1m  = r["mmp1m"]  / weight_kg if not np.isnan(r["mmp1m"])  and weight_kg > 0 else 0.0
            wkg_5m  = r["mmp5m"]  / weight_kg if not np.isnan(r["mmp5m"])  and weight_kg > 0 else 0.0
            wkg_20m = r["mmp20m"] / weight_kg if not np.isnan(r["mmp20m"]) and weight_kg > 0 else 0.0

            rows.append({
                "athlete_id": aid, "date": r["date"],
                "duration_seconds": r["dur_s"], "distance_meters": r["dist_m"],
                "elevation_gain_meters": r["elev_m"],
                "avg_power": avg_p, "normalized_power": np_w,
                "max_power": r["max_p"], "intensity_factor": if_v,
                "variability_index": vi_v, "tss": tss,
                "avg_hr": r["avg_hr"], "max_hr": r["max_hr"],
                "avg_cadence": r["avg_cad"],
                "hr_drift": 0.0,
                "aerobic_efficiency": (avg_p / r["avg_hr"]
                                       if r["avg_hr"] > 0 and avg_p > 0 else 0.0),
                "z1": zf[0], "z2": zf[1], "z3": zf[2], "z4": zf[3],
                "z5": zf[4], "z6": zf[5], "z7": zf[6],
                "temperature_c": 18.0, "humidity_pct": 50.0, "wind_speed_kmh": 0.0,
                "perceived_exertion": 5.0,
                "ctl": 0.0, "atl": 0.0, "tsb": 0.0, "days_since_last_ride": 1.0,
                "workout_type": wt,
                "ftp": ftp, "athlete_max_hr": float(max_hr_ath),
                "resting_hr": 55.0, "weight_kg": weight_kg,
                "height_cm": height_cm, "sex": sex, "age": float(age),
                "experience_years": 3.0, "primary_goal": "general_fitness",
                "training_days": 5, "days_to_event": -1.0,
                "pc_5s_wkg": float(np.nan_to_num(wkg_5s)),
                "pc_1min_wkg": float(np.nan_to_num(wkg_1m)),
                "pc_5min_wkg": float(np.nan_to_num(wkg_5m)),
                "pc_20min_wkg": float(np.nan_to_num(wkg_20m)),
                "pc1min_capacity_wkg": 0.0, "pc5min_capacity_wkg": 0.0,
                "risk_ot_class": 2, "risk_inj_target": 0.0,
                "hrv_z": 0.0, "rhr_delta": 0.0,
                "sleep_score": 50.0, "body_battery": 50.0,
            })

    df = pd.DataFrame(rows)
    # Compute CTL/ATL/TSB properly for raw format
    for aid_val in df["athlete_id"].unique():
        mask = df["athlete_id"] == aid_val
        grp = df[mask].sort_values("date")
        fitness = _compute_fitness(grp["tss"], grp["date"])
        for i, (idx, row) in enumerate(grp.iterrows()):
            d = pd.Timestamp(row["date"]).normalize()
            if d in fitness.index:
                df.loc[idx, "ctl"] = fitness.loc[d, "ctl"]
                df.loc[idx, "atl"] = fitness.loc[d, "atl"]
                df.loc[idx, "tsb"] = fitness.loc[d, "tsb"]

    print(f"  Converted {len(df):,} rides from {df['athlete_id'].nunique()} athletes")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",    required=True, help="Path to GoldenCheetah data directory")
    ap.add_argument("--output",   default="ml/data/goldencheetah.parquet")
    ap.add_argument("--format",   choices=["auto", "summary", "raw"], default="auto",
                    help="Dataset format. 'auto' tries summary first, then raw.")
    ap.add_argument("--min-rides", type=int, default=20,
                    help="Minimum rides per athlete (default 20)")
    ap.add_argument("--preview",   action="store_true",
                    help="Print available columns and exit without writing output")
    args = ap.parse_args()

    input_dir = Path(args.input)

    if args.preview:
        csv_files = list(input_dir.glob("*.csv"))
        for f in csv_files[:5]:
            df = pd.read_csv(f, nrows=2)
            print(f"\n{f.name}:")
            print("  ", list(df.columns))
        return

    fmt = args.format
    if fmt == "auto":
        has_csvs = any(input_dir.glob("*.csv"))
        fmt = "summary" if has_csvs else "raw"

    if fmt == "summary":
        df = convert_summary(input_dir, Path(args.output), args.min_rides)
    else:
        df = convert_raw(input_dir, Path(args.output), args.min_rides)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"\nWriting {len(df):,} rows → {args.output}")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, args.output, compression="snappy")
    print(f"✓ Done  ({Path(args.output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
