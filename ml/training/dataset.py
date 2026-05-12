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
  target_wt    : workout-type class index (10)
  target_if    : raw intensity factor of next ride          (regression)
  target_dur   : raw duration of next ride in HOURS         (regression)
  ftp_delta    : (FTP four CALENDAR weeks after window) − (FTP at window end)
                 in WATTS. Sample is dropped if no future ride exists in that
                 horizon — keeps the regression target meaningful.
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
    start: int                # absolute row index of first ride
    end: int                  # absolute row index of last ride + 1
    ftp_future: np.ndarray    # raw watts: FTP measured >=28 days AFTER each row's date.
                              # NaN where no future ride exists in horizon.


class CyclingDataset(Dataset):
    """One sample = (history window of length ≤ seq_len, target = next ride)."""

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 90,
        min_history: int = 10,
        athlete_ids: Iterable[int] | None = None,
        horizon_aware: bool = True,
    ):
        if athlete_ids is not None:
            df = df[df["athlete_id"].isin(set(athlete_ids))]

        df = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)
        self.df = df
        self.seq_len = seq_len
        self.horizon_aware = horizon_aware

        # ── Pre-normalize the entire dataframe ────────────────────────────
        act_mat = encode_activity_dataframe(df)        # (N, ACTIVITY_DIM)
        prof_mat = encode_profile_dataframe(df)        # (N, PROFILE_DIM)
        self.tokens = np.concatenate([act_mat, prof_mat], axis=1).astype(np.float32)
        assert self.tokens.shape[1] == INPUT_DIM

        # ── Per-row scalar arrays for fast lookup ─────────────────────────
        self.dates = df["date"].to_numpy(dtype="datetime64[ns]")
        self.ftp   = df["ftp"].to_numpy(dtype=np.float32)
        self.if_   = df["intensity_factor"].fillna(0).to_numpy(dtype=np.float32)
        self.dur_h = (df["duration_seconds"].fillna(0).to_numpy(dtype=np.float32) / 3600.0)
        self.wt_idx = (
            df["workout_type"].fillna("endurance").astype(str)
              .map(WORKOUT_TYPE_IDX).fillna(2).to_numpy(dtype=np.int64)
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

        # ── Athlete blocks + per-row FTP-in-4-weeks ───────────────────────
        self.blocks: dict[int, _AthleteBlock] = {}
        starts = []
        for athlete_id, group in df.groupby("athlete_id", sort=False):
            start = int(group.index[0])
            end = int(group.index[-1]) + 1
            block_dates = self.dates[start:end]
            block_ftp = self.ftp[start:end]

            # For each ride, find first future ride at date+28d via searchsorted
            target_dates = block_dates + np.timedelta64(FTP_FORECAST_DAYS, "D")
            future_idx = np.searchsorted(block_dates, target_dates, side="left")
            ftp_future_local = np.full(end - start, np.nan, dtype=np.float32)
            valid = future_idx < (end - start)
            ftp_future_local[valid] = block_ftp[future_idx[valid]]
            self.blocks[int(athlete_id)] = _AthleteBlock(
                start=start, end=end, ftp_future=ftp_future_local,
            )
            starts.append((int(athlete_id), start, end))

        # ── Build sample index: (athlete_id, end_row, horizon_label, horizon_days) ─
        # `end_row` is the absolute index of the FIRST future ride. The history
        # is [start, end_row). For each base anchor we emit up to 3 samples
        # corresponding to short/medium/event horizons, each carrying a
        # horizon-specific target (the modal workout type of the future window
        # and its mean IF/duration).
        self.samples: list[tuple[int, int, str, float]] = []
        for athlete_id, start, end in starts:
            block = self.blocks[athlete_id]
            for end_row in range(start + min_history, end):
                history_last = end_row - 1
                if np.isnan(block.ftp_future[history_last - start]):
                    continue  # no FTP measurement 4w ahead → drop sample

                if not self.horizon_aware:
                    self.samples.append((athlete_id, end_row, "short", 7.0))
                    continue

                # Always emit short + medium horizons.
                self.samples.append((athlete_id, end_row, "short", 7.0))
                self.samples.append((athlete_id, end_row, "medium", 28.0))
                # Emit event horizon only if a real event is upcoming.
                dte = float(self.days_to_event[history_last])
                if dte > 0 and dte <= 200:
                    self.samples.append((athlete_id, end_row, "event", dte))

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        athlete_id, end_row, horizon_label, horizon_days = self.samples[idx]
        block = self.blocks[athlete_id]

        win_start = max(block.start, end_row - self.seq_len)
        x = self.tokens[win_start:end_row]                  # (T, INPUT_DIM)

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

        # FTP delta over next 4 weeks, measured from the LAST history ride.
        history_last = end_row - 1
        ftp_now    = float(self.ftp[history_last])
        ftp_future = float(block.ftp_future[history_last - block.start])
        ftp_delta  = ftp_future - ftp_now

        # Risk targets reflect the rider's state at the END of the history
        # window — the model is asked "what is the risk RIGHT NOW given
        # everything you have seen so far?".
        target_risk_ot  = int(self.risk_ot[history_last])
        target_risk_inj = float(self.risk_inj[history_last])

        # Encode horizon descriptor.
        horizon_vec = np.asarray(
            encode_horizon(horizon_label, horizon_days), dtype=np.float32
        )

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "day_idx": torch.from_numpy(day_idx),
            "horizon_query": torch.from_numpy(horizon_vec),
            "target_wt": torch.tensor(target_wt, dtype=torch.long),
            "target_if": torch.tensor(target_if, dtype=torch.float32),
            "target_dur": torch.tensor(target_dur, dtype=torch.float32),
            "ftp_delta": torch.tensor(ftp_delta, dtype=torch.float32),
            "target_risk_ot":  torch.tensor(target_risk_ot, dtype=torch.long),
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

    targets_wt  = torch.zeros(B, dtype=torch.long)
    targets_if  = torch.zeros(B, dtype=torch.float32)
    targets_dur = torch.zeros(B, dtype=torch.float32)
    ftp_delta   = torch.zeros(B, dtype=torch.float32)
    risk_ot     = torch.full((B,), 2, dtype=torch.long)   # default "neither"
    risk_inj    = torch.zeros(B, dtype=torch.float32)
    has_horizon = "horizon_query" in batch[0]
    if has_horizon:
        horizon_dim = batch[0]["horizon_query"].shape[0]
        horizon_query = torch.zeros(B, horizon_dim, dtype=torch.float32)

    for i, item in enumerate(batch):
        T = item["x"].shape[0]
        x_padded[i, :T] = item["x"]
        day_padded[i, :T] = item["day_idx"]
        padding_mask[i, :T] = False
        targets_wt[i]  = item["target_wt"]
        targets_if[i]  = item["target_if"]
        targets_dur[i] = item["target_dur"]
        ftp_delta[i]   = item["ftp_delta"]
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
        "target_wt": targets_wt,
        "target_if": targets_if,
        "target_dur": targets_dur,
        "ftp_delta": ftp_delta,
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
