"""
Lazy training dataset for the CyclingTransformer.

Reads the RAW parquet produced by `ml/training/generate_synthetic.py` (or any
real-DB export with the same raw schema) and exposes sliding-window
(history → next-workout) samples for the transformer.

Memory model:
  - Pre-normalize the FULL dataframe ONCE via the shared encoder
    (app.ml.norm.encode_activity_dataframe / encode_profile_dataframe).
  - Hold one float32 matrix of shape (N_rows, ACTIVITY_DIM + PROFILE_DIM).
  - Sliding-window samples are stored as (athlete_idx, end_idx) tuples and
    materialized lazily in __getitem__. Memory ≈ N_rows × INPUT_DIM × 4 bytes.

Targets per sample (predicting the NEXT activity):
  target_wt      : workout-type class index (10)
  target_if      : raw intensity factor of next ride          (regression)
  target_dur     : raw duration of next ride in HOURS         (regression)
  ftp_delta      : fractional FTP change over 4 weeks, e.g. 0.05 = +5 %
  pc1min_delta   : fractional 1-min power capacity change over 4 weeks
  pc5min_delta   : fractional 5-min power capacity change over 4 weeks
  goal_idx       : integer goal class (0-7) for goal-weighted loss weighting
  policy_weight  : how much to IMITATE this sample's prescription (wt/if/dur).
                   1.0 for synthetic/expert data (always trust the coach).
                   For REAL data it is an outcome-advantage in [0,1]: ~1.0 when
                   the rider's FTP actually improved over the forecast window,
                   ~0.0 when they stagnated or got worse. This is the key to
                   "learn good-from-bad" — we only copy the prescriptions of
                   riders who got faster, never the average/declining ones. The
                   FORECAST heads (ftp/pc/risk) ignore this weight entirely and
                   train on ALL data, good and bad — declines are the negative
                   examples the physiology model needs.
  All fractional deltas are NaN when the capacity column is absent (old data).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from app.ml.model import ACTIVITY_DIM, PROFILE_DIM, INPUT_DIM, HORIZON_DIM, encode_horizon
from app.ml.norm import (
    WORKOUT_TYPE_IDX,
    GOAL_TYPE_IDX,
    encode_activity_dataframe,
    encode_profile_dataframe,
)

FTP_FORECAST_DAYS = 28  # 4 calendar weeks

# ── Horizon planning windows (calendar days into the FUTURE of the history) ──
# When a sample is emitted with horizon="short", the target reflects the
# next-7-day workload; "medium" → next 28 days; "event" → window centered on
# the athlete's true days_to_event (clamped to a reasonable forecast range).
HORIZON_WINDOW_DAYS = {
    "short": 7,
    "medium": 28,
    # "event" is per-sample (uses days_to_event from the row)
}


@dataclass
class _AthleteBlock:
    start: int                  # absolute row index of first ride
    end: int                    # absolute row index of last ride + 1
    ftp_future: np.ndarray      # FTP ≥28d ahead (watts); NaN if no future ride
    pc1min_future: np.ndarray   # 1-min capacity W/kg ≥28d ahead; NaN if absent
    pc5min_future: np.ndarray   # 5-min capacity W/kg ≥28d ahead; NaN if absent


class CyclingDataset(Dataset):
    """One sample = (history window of length ≤ seq_len, target = next ride)."""

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 90,
        min_history: int = 10,
        athlete_ids: Iterable[int] | None = None,
        horizon_aware: bool = True,
        already_sorted: bool = False,
        outcome_weighting: bool = True,
        outcome_scale: float = 0.05,
        outcome_deadband: float = 0.0,
        synthetic_forecast_weight: float = 1.0,
        forecast_min_delta: float = 1e-4,
    ):
        if athlete_ids is not None:
            df = df[df["athlete_id"].isin(set(athlete_ids))]

        if not already_sorted:
            df = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)
        self.seq_len = seq_len
        self.horizon_aware = horizon_aware
        # Outcome-weighting controls for the POLICY heads on real data.
        #   outcome_scale    : FTP gain (fractional) that earns full imitation
        #                      weight 1.0. 0.05 ⇒ a +5% FTP gain over the 4-week
        #                      forecast window is treated as a clear "winner".
        #   outcome_deadband : minimum FTP gain before any imitation weight is
        #                      given. 0.0 ⇒ any improvement counts; decliners → 0.
        self.outcome_weighting = outcome_weighting
        self.outcome_scale = max(outcome_scale, 1e-6)
        self.outcome_deadband = outcome_deadband
        # FORECAST heads (FTP/power-curve deltas) learn from MEASURED outcomes.
        # Synthetic deltas are simulated (a formula), so during fine-tuning they
        # must be excluded or they re-teach fictional physiology and wash out the
        # real dose-response. synthetic_forecast_weight=1.0 during pure-synthetic
        # PRETRAIN (it's all we have); set to 0.0 when fine-tuning with real data
        # present so the forecast heads train on real rows only.
        self.synthetic_forecast_weight = synthetic_forecast_weight
        # Minimum absolute ftp_delta for a REAL row to contribute to forecast loss.
        # Excludes the ~47% of GC sequences where FTP never changed (athlete didn't
        # test) — those would teach the model "training has no effect on FTP".
        self.forecast_min_delta = forecast_min_delta

        # Per-row "is this real (logged) data?" flag. Synthetic/expert data has
        # a trusted prescription policy → always imitated (weight 1.0). Real
        # data is only imitated in proportion to the realized outcome. Source is
        # taken from a `source` column ("real" vs anything else); absent column
        # ⇒ treated as synthetic for backward compatibility with old parquets.
        if "source" in df.columns:
            self.is_real = (df["source"].astype(str).to_numpy() == "real")
        else:
            self.is_real = np.zeros(len(df), dtype=bool)

        # ── Pre-normalize the entire dataframe ────────────────────────────
        act_mat  = encode_activity_dataframe(df)   # (N, ACTIVITY_DIM) float32
        prof_mat = encode_profile_dataframe(df)    # (N, PROFILE_DIM)  float32
        # Allocate float16 directly and copy column-wise — avoids the 2.6 GB
        # temporary float32 concat that np.concatenate(...).astype(float16) creates.
        N = len(df)
        self.tokens = np.empty((N, INPUT_DIM), dtype=np.float16)
        self.tokens[:, :ACTIVITY_DIM] = act_mat;  del act_mat
        self.tokens[:, ACTIVITY_DIM:] = prof_mat; del prof_mat

        # ── Per-row scalar arrays for fast lookup ─────────────────────────
        self.dates = df["date"].to_numpy(dtype="datetime64[ns]")
        self.ftp   = df["ftp"].to_numpy(dtype=np.float32)
        self.if_   = df["intensity_factor"].fillna(0).to_numpy(dtype=np.float32)
        self.dur_h = (df["duration_seconds"].fillna(0).to_numpy(dtype=np.float32) / 3600.0)
        self.wt_idx = (
            df["workout_type"].fillna("endurance").astype(str)
              .map(WORKOUT_TYPE_IDX).fillna(2).to_numpy(dtype=np.int64)
        )
        # Goal index for goal-weighted loss (0-7 matching GOAL_TYPE_IDX).
        self.goal_idx = (
            df["primary_goal"].fillna("general_fitness").astype(str)
              .map(GOAL_TYPE_IDX).fillna(0).to_numpy(dtype=np.int64)
        )
        # Power-curve capacity columns — zeros if not present (old parquets).
        self.pc1min_cap = (
            df["pc1min_capacity_wkg"].fillna(0.0).to_numpy(dtype=np.float32)
            if "pc1min_capacity_wkg" in df.columns
            else np.zeros(len(df), dtype=np.float32)
        )
        self.pc5min_cap = (
            df["pc5min_capacity_wkg"].fillna(0.0).to_numpy(dtype=np.float32)
            if "pc5min_capacity_wkg" in df.columns
            else np.zeros(len(df), dtype=np.float32)
        )
        # days_to_event: raw days (0 = race day, may be NaN if no event)
        if "days_to_event" in df.columns:
            self.days_to_event = df["days_to_event"].fillna(-1).to_numpy(dtype=np.float32)
        else:
            self.days_to_event = np.full(len(df), -1.0, dtype=np.float32)

        # Risk supervision (per-row state at the moment of the ride). For
        # samples whose history ends at row i we read the labels at i-1 —
        # i.e. "given the rider's state right now, what is the risk?". Older
        # parquets without these columns get the safe default of 2=neither
        # (no risk) and 0=no injury so training still runs but contributes
        # zero gradient signal.
        if "risk_ot_class" in df.columns:
            self.risk_ot = df["risk_ot_class"].fillna(2).to_numpy(dtype=np.int64)
        else:
            self.risk_ot = np.full(len(df), 2, dtype=np.int64)
        if "risk_inj_target" in df.columns:
            self.risk_inj = df["risk_inj_target"].fillna(0).to_numpy(dtype=np.float32)
        else:
            self.risk_inj = np.zeros(len(df), dtype=np.float32)

        # ── Athlete blocks + per-row 4-week-ahead targets ─────────────────
        self.blocks: dict[int, _AthleteBlock] = {}
        starts = []
        for athlete_id, group in df.groupby("athlete_id", sort=False):
            start = int(group.index[0])
            end = int(group.index[-1]) + 1
            block_dates  = self.dates[start:end]
            block_ftp    = self.ftp[start:end]
            block_pc1min = self.pc1min_cap[start:end]
            block_pc5min = self.pc5min_cap[start:end]

            # For each ride, find the first future ride ≥28 days ahead.
            target_dates = block_dates + np.timedelta64(FTP_FORECAST_DAYS, "D")
            future_idx   = np.searchsorted(block_dates, target_dates, side="left")
            valid        = future_idx < (end - start)

            ftp_future_local   = np.full(end - start, np.nan, dtype=np.float32)
            pc1min_future_local = np.full(end - start, np.nan, dtype=np.float32)
            pc5min_future_local = np.full(end - start, np.nan, dtype=np.float32)

            ftp_future_local[valid]    = block_ftp[future_idx[valid]]
            pc1min_future_local[valid] = block_pc1min[future_idx[valid]]
            pc5min_future_local[valid] = block_pc5min[future_idx[valid]]

            self.blocks[int(athlete_id)] = _AthleteBlock(
                start=start, end=end,
                ftp_future=ftp_future_local,
                pc1min_future=pc1min_future_local,
                pc5min_future=pc5min_future_local,
            )
            starts.append((int(athlete_id), start, end))

        # ── Build sample index ────────────────────────────────────────────
        # Stored as 4 compact numpy arrays instead of a list of Python tuples.
        # Python tuples cost ~200 bytes each; numpy int32/float32 costs 4 bytes.
        # For 12M samples this saves ~6 GB of RAM.
        # horizon_idx encoding: 0=short (7d), 1=medium (28d), 2=event (dte days)
        _sa: list[int]   = []  # athlete_id
        _se: list[int]   = []  # end_row
        _sh: list[int]   = []  # horizon_idx
        _sd: list[float] = []  # horizon_days
        for athlete_id, start, end in starts:
            block = self.blocks[athlete_id]
            for end_row in range(start + min_history, end):
                history_last = end_row - 1
                if np.isnan(block.ftp_future[history_last - start]):
                    continue  # no FTP measurement 4w ahead → drop sample

                if not self.horizon_aware:
                    _sa.append(athlete_id); _se.append(end_row)
                    _sh.append(0);          _sd.append(7.0)
                    continue

                _sa.append(athlete_id); _se.append(end_row)
                _sh.append(0);          _sd.append(7.0)
                _sa.append(athlete_id); _se.append(end_row)
                _sh.append(1);          _sd.append(28.0)
                dte = float(self.days_to_event[history_last])
                if dte > 0 and dte <= 200:
                    _sa.append(athlete_id); _se.append(end_row)
                    _sh.append(2);          _sd.append(dte)

        self._s_athlete  = np.array(_sa, dtype=np.int32)
        self._s_end_row  = np.array(_se, dtype=np.int32)
        self._s_h_idx    = np.array(_sh, dtype=np.uint8)
        self._s_h_days   = np.array(_sd, dtype=np.float32)
        # Release the DataFrame — all needed data is in the arrays above.
        # This frees ~3 GB for 50K athletes.
        del df

    _HORIZON_LABELS = ("short", "medium", "event")

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._s_athlete)

    def __getitem__(self, idx: int) -> dict:
        athlete_id    = int(self._s_athlete[idx])
        end_row       = int(self._s_end_row[idx])
        horizon_label = self._HORIZON_LABELS[int(self._s_h_idx[idx])]
        horizon_days  = float(self._s_h_days[idx])
        block = self.blocks[athlete_id]

        win_start = max(block.start, end_row - self.seq_len)
        x = self.tokens[win_start:end_row].astype(np.float32)  # float16→float32 for model

        # day index = days since first token in window
        win_dates = self.dates[win_start:end_row]
        day_idx = ((win_dates - win_dates[0]).astype("timedelta64[D]")
                   .astype(np.int64))

        # ── Horizon-conditioned targets ──────────────────────────────────
        # Find all future rides of this athlete whose date falls within
        # `horizon_days` calendar days after the last history ride. Use the
        # modal workout type and mean IF / mean duration of those rides as
        # the target — this is the "what should the rider plan over the
        # next H days" signal the model must learn to differentiate.
        last_date = self.dates[end_row - 1]
        cutoff = last_date + np.timedelta64(int(round(horizon_days)), "D")
        # Search within this athlete's block.
        block_dates_slice = self.dates[end_row:block.end]
        # Future indices within the horizon (relative to end_row).
        horizon_end_offset = int(np.searchsorted(block_dates_slice, cutoff, side="right"))
        horizon_end = end_row + max(horizon_end_offset, 1)  # at least 1 ride
        horizon_end = min(horizon_end, block.end)

        future_wt = self.wt_idx[end_row:horizon_end]
        future_if = self.if_[end_row:horizon_end]
        future_dur = self.dur_h[end_row:horizon_end]

        # Modal workout type over the window (most common). For ties, prefer
        # the FIRST upcoming ride to keep targets consistent for short horizons.
        if len(future_wt) == 0:
            target_wt = int(self.wt_idx[end_row])
            target_if = float(self.if_[end_row])
            target_dur = float(self.dur_h[end_row])
        else:
            counts = np.bincount(future_wt)
            target_wt = int(counts.argmax())
            # Mean IF / duration weighted toward the next ride for short horizons.
            target_if = float(future_if.mean())
            target_dur = float(future_dur.mean())

        # Fractional fitness deltas over the next 4 weeks.
        history_last = end_row - 1
        local_idx    = history_last - block.start

        ftp_now    = float(self.ftp[history_last])
        ftp_future = float(block.ftp_future[local_idx])
        ftp_delta  = (ftp_future - ftp_now) / max(ftp_now, 1.0)

        pc1min_now    = float(self.pc1min_cap[history_last])
        pc1min_future = float(block.pc1min_future[local_idx])
        if pc1min_now > 0.01 and not np.isnan(pc1min_future):
            pc1min_delta = (pc1min_future - pc1min_now) / pc1min_now
        else:
            pc1min_delta = float("nan")

        pc5min_now    = float(self.pc5min_cap[history_last])
        pc5min_future = float(block.pc5min_future[local_idx])
        if pc5min_now > 0.01 and not np.isnan(pc5min_future):
            pc5min_delta = (pc5min_future - pc5min_now) / pc5min_now
        else:
            pc5min_delta = float("nan")

        goal_idx = int(self.goal_idx[history_last])

        # ── Policy imitation weight ──────────────────────────────────────────
        # Synthetic/expert prescriptions are always trusted (weight 1.0). Real
        # prescriptions are weighted by the realized FTP outcome over the
        # forecast window: a clear improver → ~1.0, flat/decliner → 0.0. This
        # turns plain imitation (which would copy amateur mistakes) into
        # advantage-weighted imitation (copy only what made riders faster).
        # `ftp_delta` is always finite here — NaN-future samples are dropped at
        # index-build time.
        is_real = bool(self.is_real[history_last])
        if self.outcome_weighting and is_real:
            adv = (ftp_delta - self.outcome_deadband) / self.outcome_scale
            policy_weight = float(min(max(adv, 0.0), 1.0))
        else:
            policy_weight = 1.0

        # Forecast imitation weight: real rows = 1.0; synthetic rows controlled
        # by synthetic_forecast_weight (1.0 pretrain, 0.0 fine-tune). Keeps the
        # forecast heads learning physiology from measured data only.
        # Additionally, real rows where FTP didn't actually change (athlete never
        # tested in this window) are excluded — they'd teach "training = no effect".
        forecast_weight = 1.0 if is_real else self.synthetic_forecast_weight
        if is_real and abs(ftp_delta) < self.forecast_min_delta:
            forecast_weight = 0.0

        # Risk targets reflect the rider's state at the END of the history window.
        target_risk_ot  = int(self.risk_ot[history_last])
        target_risk_inj = float(self.risk_inj[history_last])

        horizon_vec = np.asarray(
            encode_horizon(horizon_label, horizon_days), dtype=np.float32
        )

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "day_idx": torch.from_numpy(day_idx),
            "horizon_query": torch.from_numpy(horizon_vec),
            "target_wt":  torch.tensor(target_wt,  dtype=torch.long),
            "target_if":  torch.tensor(target_if,  dtype=torch.float32),
            "target_dur": torch.tensor(target_dur, dtype=torch.float32),
            "ftp_delta":    torch.tensor(ftp_delta,    dtype=torch.float32),
            "pc1min_delta": torch.tensor(pc1min_delta, dtype=torch.float32),
            "pc5min_delta": torch.tensor(pc5min_delta, dtype=torch.float32),
            "goal_idx":     torch.tensor(goal_idx,     dtype=torch.long),
            "policy_weight": torch.tensor(policy_weight, dtype=torch.float32),
            "forecast_weight": torch.tensor(forecast_weight, dtype=torch.float32),
            "target_risk_ot":  torch.tensor(target_risk_ot,  dtype=torch.long),
            "target_risk_inj": torch.tensor(target_risk_inj, dtype=torch.float32),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Right-pad sequences to max length in the batch."""
    max_len = max(item["x"].shape[0] for item in batch)
    B = len(batch)
    feat_dim = batch[0]["x"].shape[1]

    x_padded     = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
    day_padded   = torch.zeros(B, max_len, dtype=torch.long)
    padding_mask = torch.ones(B, max_len, dtype=torch.bool)  # True = padded

    targets_wt   = torch.zeros(B, dtype=torch.long)
    targets_if   = torch.zeros(B, dtype=torch.float32)
    targets_dur  = torch.zeros(B, dtype=torch.float32)
    ftp_delta    = torch.full((B,), float("nan"), dtype=torch.float32)
    pc1min_delta = torch.full((B,), float("nan"), dtype=torch.float32)
    pc5min_delta = torch.full((B,), float("nan"), dtype=torch.float32)
    goal_idx     = torch.zeros(B, dtype=torch.long)
    policy_weight = torch.ones(B, dtype=torch.float32)     # default: full trust
    forecast_weight = torch.ones(B, dtype=torch.float32)   # default: full
    risk_ot      = torch.full((B,), 2, dtype=torch.long)   # default "neither"
    risk_inj     = torch.zeros(B, dtype=torch.float32)
    has_horizon = "horizon_query" in batch[0]
    if has_horizon:
        horizon_dim = batch[0]["horizon_query"].shape[0]
        horizon_query = torch.zeros(B, horizon_dim, dtype=torch.float32)

    for i, item in enumerate(batch):
        T = item["x"].shape[0]
        x_padded[i, :T] = item["x"]
        day_padded[i, :T] = item["day_idx"]
        padding_mask[i, :T] = False
        targets_wt[i]   = item["target_wt"]
        targets_if[i]   = item["target_if"]
        targets_dur[i]  = item["target_dur"]
        ftp_delta[i]    = item["ftp_delta"]
        pc1min_delta[i] = item["pc1min_delta"]
        pc5min_delta[i] = item["pc5min_delta"]
        goal_idx[i]     = item["goal_idx"]
        if "policy_weight" in item:
            policy_weight[i] = item["policy_weight"]
        if "forecast_weight" in item:
            forecast_weight[i] = item["forecast_weight"]
        if "target_risk_ot" in item:
            risk_ot[i]  = item["target_risk_ot"]
        if "target_risk_inj" in item:
            risk_inj[i] = item["target_risk_inj"]
        if has_horizon:
            horizon_query[i] = item["horizon_query"]

    out = {
        "x": x_padded,
        "day_idx": day_padded,
        "padding_mask": padding_mask,
        "target_wt":  targets_wt,
        "target_if":  targets_if,
        "target_dur": targets_dur,
        "ftp_delta":    ftp_delta,
        "pc1min_delta": pc1min_delta,
        "pc5min_delta": pc5min_delta,
        "goal_idx":     goal_idx,
        "policy_weight": policy_weight,
        "forecast_weight": forecast_weight,
        "target_risk_ot":  risk_ot,
        "target_risk_inj": risk_inj,
    }
    if has_horizon:
        out["horizon_query"] = horizon_query
    return out


def athlete_split(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42
                  ) -> tuple[list[int], list[int]]:
    """Split athletes (NOT rows) into train/val to prevent leakage."""
    rng = np.random.default_rng(seed)
    ids = df["athlete_id"].unique()
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    return list(ids[n_val:]), list(ids[:n_val])
