"""
CyclingTransformer — temporal transformer that learns from sequences of cycling activities.

Architecture:
  - Each activity → feature vector of ~50 training/physiology parameters
  - Positional encoding = time-aware (days since first activity, day-of-week)
  - Athlete embedding = static profile concatenated to every token
  - N transformer encoder layers
  - Output heads:
      1. WorkoutType classification  (next workout type)
      2. Intensity regression        (target IF % FTP)
      3. Duration regression         (target minutes)
      4. FTP delta regression        (expected FTP change in 4 weeks)
      5. Risk classification         (overtraining risk score)

Training objective (future):
  - Self-supervised: predict next activity features given history
  - Supervised (fine-tune): given activity sequence + outcome, predict FTP change

Cold-start fallback (< 50 activities):
  Rule-based periodization + Friel training zones — see cold_start.py
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


# ─── Activity Feature Vector ────────────────────────────────────────────────────
# These are the fields we embed per activity (in order).
ACTIVITY_FEATURES = [
    # Load
    "tss", "ctl", "atl", "tsb",
    "duration_h", "distance_km", "elevation_km",
    # Power
    "avg_power_pct_ftp", "np_pct_ftp", "intensity_factor",
    "max_power_pct_ftp", "variability_index",
    # HR
    "avg_hr_pct_max", "max_hr_pct_max", "hr_drift", "aerobic_efficiency",
    # Cadence
    "avg_cadence_norm",
    # Zones (% time in each)
    "z1", "z2", "z3", "z4", "z5", "z6", "z7",
    # Environment
    "temp_c_norm", "humidity_norm", "wind_norm",
    # Temporal
    "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
    "days_since_last_ride",
    # Workout type (one-hot 10 classes)
    "wt_recovery", "wt_easy", "wt_endurance", "wt_tempo",
    "wt_sweetspot", "wt_threshold", "wt_vo2max", "wt_sprint",
    "wt_race", "wt_long_ride",
    # Perceived exertion / notes
    "rpe_norm",
]

ACTIVITY_DIM = len(ACTIVITY_FEATURES)  # ~46

# Static athlete profile features
PROFILE_FEATURES = [
    "age_norm", "weight_norm", "height_norm", "sex_bin",
    "ftp_norm", "max_hr_norm", "resting_hr_norm",
    "experience_norm", "goal_type_norm",
    "days_to_event_norm",
]
PROFILE_DIM = len(PROFILE_FEATURES)  # 10

INPUT_DIM = ACTIVITY_DIM + PROFILE_DIM  # total token dim


class PositionalEncoding(nn.Module):
    """
    Time-aware positional encoding using actual elapsed days (not just sequence index).
    """
    def __init__(self, d_model: int, max_days: int = 1500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_days, d_model)
        position = torch.arange(0, max_days, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor, day_indices: Tensor) -> Tensor:
        """
        x: (batch, seq_len, d_model)
        day_indices: (batch, seq_len) — actual day offset for each token
        """
        pe = self.pe[day_indices.clamp(0, self.pe.size(0) - 1)]  # type: ignore
        return self.dropout(x + pe)


class CyclingTransformer(nn.Module):
    """
    Transformer encoder that processes a sequence of past activities and
    outputs coaching predictions for the NEXT workout.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 180,  # up to 180 activities
        num_workout_types: int = 10,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Output heads
        # 1. Next workout type (classification)
        self.workout_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_workout_types),
        )

        # 2. Next workout intensity (IF target, 0-2)
        self.intensity_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # → [0, 1] * 2 in inference
        )

        # 3. Next workout duration (normalized hours, 0-6h)
        self.duration_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # 4. FTP delta in 4 weeks (watts, can be negative)
        self.ftp_delta_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # 5. Peak CTL forecast
        self.ctl_forecast_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.ReLU(),
        )

        # 6. Risk score (overtraining risk, 0-1)
        self.risk_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 3),      # [overtraining, undertraining, injury]
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: Tensor,                      # (B, T, INPUT_DIM)
        day_indices: Tensor,            # (B, T) int
        padding_mask: Tensor | None = None,  # (B, T) bool — True = padded
    ) -> dict[str, Tensor]:
        B, T, _ = x.shape

        # Project to model dim
        h = self.input_proj(x)            # (B, T, d_model)
        h = self.pos_enc(h, day_indices)  # (B, T, d_model)

        # Transformer encoder (causal mask so model only sees past)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=padding_mask)

        # Use last non-padded token as "current state"
        if padding_mask is not None:
            # Find last valid token per batch item
            lengths = T - padding_mask.sum(dim=1)  # (B,)
            last_idx = (lengths - 1).clamp(0)
            last_h = h[torch.arange(B, device=x.device), last_idx]  # (B, d_model)
        else:
            last_h = h[:, -1, :]  # (B, d_model)

        return {
            "workout_logits": self.workout_head(last_h),        # (B, num_types)
            "intensity":      self.intensity_head(last_h) * 2,  # (B, 1) → [0, 2] IF
            "duration":       self.duration_head(last_h) * 6,   # (B, 1) → [0, 6] hours
            "ftp_delta":      self.ftp_delta_head(last_h),      # (B, 1) watts
            "ctl_peak":       self.ctl_forecast_head(last_h),   # (B, 1)
            "risks":          self.risk_head(last_h),            # (B, 3)
        }
