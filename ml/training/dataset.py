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

from app.ml.model import ACTIVITY_DIM, PROFILE_DIM, INPUT_DIM
from app.ml.norm import (
    WORKOUT_TYPE_IDX,
    encode_activity_dataframe,
    encode_profile_dataframe,
)

FTP_FORECAST_DAYS = 28  # 4 calendar weeks


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
    ):
        if athlete_ids is not None:
            df = df[df["athlete_id"].isin(set(athlete_ids))]

        df = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)
        self.df = df
        self.seq_len = seq_len

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

        # ── Build sample index: (athlete_id, end_row) ─────────────────────
        # `end_row` is the absolute index of the TARGET ride (i.e. the next
        # workout we want to predict). The history is [start, end_row).
        self.samples: list[tuple[int, int]] = []
        for athlete_id, start, end in starts:
            block = self.blocks[athlete_id]
            for end_row in range(start + min_history, end):
                history_last = end_row - 1
                if np.isnan(block.ftp_future[history_last - start]):
                    continue  # no FTP measurement 4w ahead → drop sample
                self.samples.append((athlete_id, end_row))

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        athlete_id, end_row = self.samples[idx]
        block = self.blocks[athlete_id]

        win_start = max(block.start, end_row - self.seq_len)
        x = self.tokens[win_start:end_row]                  # (T, INPUT_DIM)

        # day index = days since first token in window
        win_dates = self.dates[win_start:end_row]
        day_idx = ((win_dates - win_dates[0]).astype("timedelta64[D]")
                   .astype(np.int64))

        # Targets — read from the NEXT (target) activity row
        target_wt  = int(self.wt_idx[end_row])
        target_if  = float(self.if_[end_row])
        target_dur = float(self.dur_h[end_row])

        # FTP delta over next 4 weeks, measured from the LAST history ride
        history_last = end_row - 1
        ftp_now    = float(self.ftp[history_last])
        ftp_future = float(block.ftp_future[history_last - block.start])
        ftp_delta  = ftp_future - ftp_now

        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "day_idx": torch.from_numpy(day_idx),
            "target_wt": torch.tensor(target_wt, dtype=torch.long),
            "target_if": torch.tensor(target_if, dtype=torch.float32),
            "target_dur": torch.tensor(target_dur, dtype=torch.float32),
            "ftp_delta": torch.tensor(ftp_delta, dtype=torch.float32),
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

    for i, item in enumerate(batch):
        T = item["x"].shape[0]
        x_padded[i, :T] = item["x"]
        day_padded[i, :T] = item["day_idx"]
        padding_mask[i, :T] = False
        targets_wt[i]  = item["target_wt"]
        targets_if[i]  = item["target_if"]
        targets_dur[i] = item["target_dur"]
        ftp_delta[i]   = item["ftp_delta"]

    return {
        "x": x_padded,
        "day_idx": day_padded,
        "padding_mask": padding_mask,
        "target_wt": targets_wt,
        "target_if": targets_if,
        "target_dur": targets_dur,
        "ftp_delta": ftp_delta,
    }


def athlete_split(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 42
                  ) -> tuple[list[int], list[int]]:
    """Split athletes (NOT rows) into train/val to prevent leakage."""
    rng = np.random.default_rng(seed)
    ids = df["athlete_id"].unique()
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    return list(ids[n_val:]), list(ids[:n_val])
