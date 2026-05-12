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
    # ── Health & recovery (Garmin daily wellness on the day of the ride) ──
    # Plews & Buchheit 2013 (HRV-guided), Buchheit 2014 (RHR + HRV combined),
    # Stanley et al. 2013 (HR recovery time-courses).
    "hrv_z_norm",         # z-score of overnight HRV vs 7d baseline, mapped to [0,1]
    "rhr_delta_norm",     # delta vs 30d baseline RHR, mapped to [0,1]
    "sleep_score_norm",   # Garmin sleep score 0–100 → [0,1]
    "body_battery_norm",  # Garmin body battery 0–100 → [0,1]
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


# ─── Horizon Query token ──────────────────────────────────────────────────────
# A separate virtual token appended to the end of the sequence whose final
# hidden state is what the prediction heads consume. This gives the model a
# strong, dedicated signal of "what horizon are we forecasting for", instead
# of burying days_to_event inside the per-token profile vector.
#
# Encoding (HORIZON_DIM = 6):
#   [is_short, is_medium, is_event,        # one-hot bucket
#    horizon_days_norm,                    # min(days, 365) / 365
#    log_horizon_norm,                     # log1p(days) / log(366)  (compresses long horizons)
#    weeks_to_event_norm]                  # min(weeks, 52) / 52
HORIZON_DIM = 6
HORIZON_BUCKETS = ["short", "medium", "event"]
HORIZON_BUCKET_IDX = {n: i for i, n in enumerate(HORIZON_BUCKETS)}


def encode_horizon(label: str, days: float) -> list[float]:
    """Encode a planning horizon as a 6-dim float vector.

    Args:
        label: one of {"short", "medium", "event"} — coarse semantic bucket.
        days: planning horizon in days (clamped to [0, 365] for normalization).

    Returns:
        Length-6 list, intended to be projected into the model dim and fed in
        as a learnable virtual token.
    """
    import math
    one_hot = [0.0, 0.0, 0.0]
    if label in HORIZON_BUCKET_IDX:
        one_hot[HORIZON_BUCKET_IDX[label]] = 1.0
    d = max(0.0, float(days))
    d_clamped = min(d, 365.0)
    return [
        *one_hot,
        d_clamped / 365.0,
        math.log1p(d_clamped) / math.log(366.0),
        min(d / 7.0, 52.0) / 52.0,
    ]


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
        horizon_dim: int = HORIZON_DIM,
    ):
        super().__init__()
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Horizon Query token projection ───────────────────────────────
        # Maps the 6-dim horizon descriptor to d_model. Combined with a
        # learnable bias (`horizon_query_bias`) so that even when the input
        # horizon is "blank" (training without horizon supervision), the
        # model has a stable query position to read from.
        self.horizon_proj = nn.Sequential(
            nn.Linear(horizon_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.horizon_query_bias = nn.Parameter(torch.zeros(d_model))

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

        # 6. Risk heads — split for mutual exclusion + independent injury.
        #   risk_ot_head:  3-class LOGITS [overtraining, undertraining, neither]
        #                   trained with CrossEntropyLoss; softmax applied at
        #                   inference for the `risks` probability output.
        #   risk_inj_head: 1 LOGIT for injury probability
        #                   trained with BCEWithLogitsLoss; sigmoid applied at
        #                   inference for the `risks` probability output.
        # Activations are intentionally kept OUT of the head so the loss can
        # use the numerically-stable `*_with_logits` formulations.
        self.risk_ot_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, 3),      # logits: [overtraining, undertraining, neither]
        )
        self.risk_inj_head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Linear(16, 1),      # logit: injury
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
        horizon_query: Tensor | None = None, # (B, HORIZON_DIM) optional
    ) -> dict[str, Tensor]:
        """Forward pass.

        If `horizon_query` is supplied the model appends a learnable
        horizon-conditioned virtual token at the END of every sequence and
        reads its hidden state for predictions. This lets the same trained
        weights produce DIFFERENT outputs for different horizons of the same
        athlete history — which is required for multi-horizon coaching.

        If `horizon_query` is None, falls back to the legacy behaviour of
        reading the last non-padded ride token. This keeps older checkpoints
        loadable.
        """
        B, T, _ = x.shape

        # Project rides to model dim
        h = self.input_proj(x)            # (B, T, d_model)
        h = self.pos_enc(h, day_indices)  # (B, T, d_model)

        if horizon_query is not None:
            # Project horizon descriptor → d_model and append as virtual token.
            hq = self.horizon_proj(horizon_query) + self.horizon_query_bias  # (B, d_model)
            hq = hq.unsqueeze(1)                                            # (B, 1, d_model)
            h = torch.cat([h, hq], dim=1)                                   # (B, T+1, d_model)

            # Padding mask: virtual token is never padded.
            if padding_mask is not None:
                pad_q = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
                pad_full = torch.cat([padding_mask, pad_q], dim=1)
            else:
                pad_full = None

            # Causal mask over T+1 tokens. The query token (last) attends to
            # ALL ride tokens (full bidirectional within encoder), but every
            # ride token still has the standard causal mask over its peers.
            T_full = T + 1
            causal = torch.triu(
                torch.ones(T_full, T_full, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            # Allow the query token (row T) to see every previous token.
            causal[T, :] = False

            h_out = self.transformer(h, mask=causal, src_key_padding_mask=pad_full)
            # Read the query token's hidden state.
            last_h = h_out[:, T, :]  # (B, d_model)
        else:
            # Legacy path: standard causal encoder, read last non-padded token.
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            h_out = self.transformer(h, mask=causal_mask, src_key_padding_mask=padding_mask)

            if padding_mask is not None:
                lengths = T - padding_mask.sum(dim=1)  # (B,)
                last_idx = (lengths - 1).clamp(0)
                last_h = h_out[torch.arange(B, device=x.device), last_idx]
            else:
                last_h = h_out[:, -1, :]

        risk_ot_logits = self.risk_ot_head(last_h)        # (B, 3) raw logits
        risk_inj_logit = self.risk_inj_head(last_h)        # (B, 1) raw logit
        # Inference-friendly probabilities. Softmax over the 3 OT classes
        # ensures over+under+neither = 1 (mutually exclusive); sigmoid for
        # injury is independent.
        risk_ot_probs = torch.softmax(risk_ot_logits, dim=-1)
        risk_inj_prob = torch.sigmoid(risk_inj_logit)

        return {
            "workout_logits": self.workout_head(last_h),        # (B, num_types)
            "intensity":      self.intensity_head(last_h) * 2,  # (B, 1) → [0, 2] IF
            "duration":       self.duration_head(last_h) * 6,   # (B, 1) → [0, 6] hours
            "ftp_delta":      self.ftp_delta_head(last_h),      # (B, 1) watts
            "ctl_peak":       self.ctl_forecast_head(last_h),   # (B, 1)
            # Training outputs (raw logits — use *_with_logits losses).
            "risk_ot_logits": risk_ot_logits,                   # (B, 3)
            "risk_inj_logit": risk_inj_logit,                   # (B, 1)
            # Inference output: [overtraining_p, undertraining_p, injury_p]
            "risks":          torch.cat([risk_ot_probs[:, :2], risk_inj_prob], dim=-1),  # (B, 3)
        }
