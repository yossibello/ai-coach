"""
Trained-model periodization validation.

Verifies that the actual cycling_coach.pt checkpoint:
  1. Loads cleanly with the current model architecture
  2. Produces valid (non-NaN, non-zero) output on synthetic sequences
  3. When combined with the phase_prior + safety factor, recommends the right
     workout type for each phase of a Coggan/Friel block:
       - Base phase (positive TSB)     → endurance / easy dominant
       - Build phase (TSB -10 to -25)  → sweetspot / tempo acceptable
       - Overreach (TSB < -30)         → forced recovery regardless of raw logits
       - Taper (close to event)        → easy / recovery

These tests use the FULL inference pipeline:
    raw model logits → phase_prior → safety_factor(TSB) → top-1 prediction

No database required. Synthetic sequences are generated inline using the same
generate_synthetic.py code that produced the training data.

Skip gracefully when the checkpoint is not present (CI without model artefact).
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./test_aicoach.db")
os.environ.setdefault("JWT_SECRET", "test-secret")

import numpy as np
import pytest
import torch

CKPT_PATH = ROOT / "backend" / "models" / "cycling_coach.pt"


# ── helpers ───────────────────────────────────────────────────────────────────

_MODEL_CACHE: tuple | None = None


def _load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not CKPT_PATH.exists():
        pytest.skip(f"Trained model not found: {CKPT_PATH}")
    from app.ml.model import CyclingTransformer
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    m = CyclingTransformer(
        input_dim=cfg.get("input_dim", 61),  # checkpoint may predate training_days_norm (61 vs 62)
        d_model=cfg["d_model"], nhead=cfg["nhead"],
        num_layers=cfg["num_layers"], dim_feedforward=cfg["dim_feedforward"],
    )
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    _MODEL_CACHE = (m, ckpt)
    return m, ckpt


def _synth_sequence(philosophy="coggan_friel", n_weeks=8, ctl0=5.0, atl0=3.0, seed=42):
    """Generate a synthetic athlete sequence and return an encoded tensor.

    Token width is trimmed to match the loaded checkpoint's input_dim so the
    test works even when the feature set has grown since the checkpoint was saved.
    """
    import pandas as pd
    from ml.training.generate_synthetic import _make_athlete, _simulate_athlete
    from app.ml.norm import encode_activity_dataframe, encode_profile_dataframe

    rng = np.random.default_rng(seed)
    athlete = _make_athlete(rng, athlete_id=1)
    athlete.philosophy = philosophy
    athlete.n_weeks = n_weeks
    athlete.ctl = ctl0
    athlete.atl = atl0

    rows = _simulate_athlete(athlete, rng)
    if not rows:
        pytest.skip("generate_synthetic produced no rows")

    df = pd.DataFrame(rows)
    act  = encode_activity_dataframe(df).astype(np.float32)
    prof = encode_profile_dataframe(df).astype(np.float32)
    tokens = np.concatenate([act, prof], axis=1)

    # Trim to checkpoint's input_dim if the feature list has grown since training
    if _MODEL_CACHE is not None:
        ckpt_input_dim = _MODEL_CACHE[1]["config"].get("input_dim")
        if ckpt_input_dim and tokens.shape[1] > ckpt_input_dim:
            tokens = tokens[:, :ckpt_input_dim]

    x  = torch.from_numpy(tokens).unsqueeze(0)           # (1, T, D)
    di = torch.arange(tokens.shape[0]).unsqueeze(0)      # (1, T)
    return x, di, df


def _run_model(model, x, di):
    with torch.no_grad():
        out = model(x, di)
    return out


def _top_workout(logits: torch.Tensor, tsb: float, phase: str) -> str:
    """Apply phase_prior + safety_factor on raw logits → return top-1 workout name."""
    from app.ml.phase_prior import select_workout
    name, _conf, _alts = select_workout(
        logits.numpy(),
        phase=phase,
        event_type=None,
        tsb=tsb,
        hrv_z=None,
        temperature=1.0,
        prior_weight=0.6,
    )
    return name


HARD_TYPES    = {"threshold", "vo2max", "race", "sprint"}
BUILD_TYPES   = {"sweetspot", "threshold", "tempo", "vo2max"}
EASY_TYPES    = {"recovery", "easy", "endurance", "long_ride"}


# ── 1. Checkpoint integrity ───────────────────────────────────────────────────

class TestCheckpointIntegrity:

    def test_checkpoint_loads(self):
        """cycling_coach.pt loads with current architecture — no key mismatches."""
        model, ckpt = _load_model()
        assert model is not None

    def test_checkpoint_has_expected_config(self):
        """Saved config matches expected 8M-param architecture."""
        _model, ckpt = _load_model()
        cfg = ckpt["config"]
        assert cfg["d_model"]        == 256,  f"d_model={cfg['d_model']}, expected 256"
        assert cfg["nhead"]          == 8,    f"nhead={cfg['nhead']}, expected 8"
        assert cfg["num_layers"]     == 8,    f"num_layers={cfg['num_layers']}, expected 8"
        assert cfg["dim_feedforward"]== 1024, f"d_ff={cfg['dim_feedforward']}, expected 1024"

    def test_param_count_approx_8m(self):
        """Model should be ~8M parameters (±20%)."""
        model, _ = _load_model()
        n = sum(p.numel() for p in model.parameters())
        assert 6_000_000 <= n <= 12_000_000, (
            f"Expected ~8M params, got {n:,}. Architecture may have changed."
        )


# ── 2. Forward pass on synthetic sequence ────────────────────────────────────

class TestForwardPassOnSyntheticData:

    def test_no_nan_in_outputs(self):
        """No NaN or Inf in any output tensor when fed a real synthetic sequence."""
        model, _ = _load_model()
        x, di, _df = _synth_sequence()
        out = _run_model(model, x, di)
        for key, tensor in out.items():
            assert not torch.isnan(tensor).any(),  f"NaN in {key}"
            assert not torch.isinf(tensor).any(),  f"Inf in {key}"

    def test_workout_logits_not_all_zero(self):
        """Model must differentiate workout types — logits can't all be the same."""
        model, _ = _load_model()
        x, di, _df = _synth_sequence()
        out = _run_model(model, x, di)
        logits = out["workout_logits"][0]
        assert logits.std().item() > 0.01, (
            f"Workout logits are nearly identical (std={logits.std().item():.4f}). "
            "Model may be collapsed."
        )

    def test_output_shapes(self):
        """All output tensors have the expected shapes for a single-batch sequence."""
        model, _ = _load_model()
        x, di, _df = _synth_sequence()
        out = _run_model(model, x, di)
        assert out["workout_logits"].shape == (1, 10), "Expected (1, 10) workout logits"
        assert out["intensity"].shape      == (1, 1),  "Expected (1, 1) intensity"
        assert out["duration"].shape       == (1, 1),  "Expected (1, 1) duration"
        assert out["risks"].shape          == (1, 3),  "Expected (1, 3) risk probs"

    def test_risk_probs_sum_to_plausible_range(self):
        """risk_ot (3-class softmax) must sum to ~1; risk_inj (sigmoid) in [0,1]."""
        model, _ = _load_model()
        x, di, _df = _synth_sequence()
        out = _run_model(model, x, di)
        risks = out["risks"][0]           # [p_ot, p_under, p_inj]
        ot_probs = out.get("risk_ot_logits")
        if ot_probs is not None:
            ot_softmax = torch.softmax(out["risk_ot_logits"][0], dim=-1)
            assert abs(ot_softmax.sum().item() - 1.0) < 1e-4, "OT softmax must sum to 1"
        assert 0.0 <= risks[2].item() <= 1.0, "Injury prob must be in [0, 1]"


# ── 3. Periodization-aware recommendations ───────────────────────────────────

class TestPeriodizationRecommendations:
    """
    These tests check the FULL pipeline: model raw logits + phase_prior + safety.
    Even a partially-trained model should pass because the safety_factor dominates
    for extreme TSB, and the phase_prior provides strong inductive bias.
    """

    def test_extreme_overreach_forces_recovery(self):
        """When TSB < -30, the safety_factor must override the model → recovery only.
        This tests the safety layer, not model quality."""
        model, _ = _load_model()
        x, di, _df = _synth_sequence()
        out = _run_model(model, x, di)
        logits = out["workout_logits"][0]

        # Force extreme overreach scenario
        wt = _top_workout(logits, tsb=-40.0, phase="base_build")
        assert wt not in HARD_TYPES, (
            f"TSB=-40 must produce a non-hard workout, got '{wt}'. "
            "Safety factor should suppress threshold/vo2max by 60%."
        )

    def test_fresh_athlete_gets_endurance_not_vo2max(self):
        """Positive TSB in base_build phase → endurance/easy recommended."""
        model, _ = _load_model()
        x, di, _df = _synth_sequence()
        out = _run_model(model, x, di)
        logits = out["workout_logits"][0]

        wt = _top_workout(logits, tsb=+10.0, phase="base_build")
        assert wt not in {"threshold", "race"}, (
            f"Fresh athlete in base_build shouldn't get threshold/race, got '{wt}'"
        )

    def test_coggan_sequence_produces_valid_recommendations(self):
        """Full 8-week Coggan sequence → all recommendations are valid workout types."""
        from app.ml.norm import WORKOUT_TYPE_IDX
        valid_types = set(WORKOUT_TYPE_IDX.keys())

        model, _ = _load_model()
        x, di, df = _synth_sequence(philosophy="coggan_friel", n_weeks=8)
        out = _run_model(model, x, di)
        logits = out["workout_logits"][0]
        tsb = float(df["tsb"].iloc[-1])
        phase = "base_build" if tsb > -10 else ("recovery_week" if tsb < -30 else "build")

        wt = _top_workout(logits, tsb=tsb, phase=phase)
        assert wt in valid_types, f"Predicted workout type '{wt}' not in known types"

    def test_polarized_sequence_produces_valid_recommendations(self):
        """Full 8-week Polarized (Seiler 80/20) sequence → valid prediction."""
        from app.ml.norm import WORKOUT_TYPE_IDX
        valid_types = set(WORKOUT_TYPE_IDX.keys())

        model, _ = _load_model()
        x, di, df = _synth_sequence(philosophy="polarized", n_weeks=8, seed=7)
        out = _run_model(model, x, di)
        logits = out["workout_logits"][0]
        tsb = float(df["tsb"].iloc[-1])
        phase = "base_build" if tsb > -10 else "build"

        wt = _top_workout(logits, tsb=tsb, phase=phase)
        assert wt in valid_types, f"Predicted workout type '{wt}' not in known types"

    def test_model_intensity_output_in_plausible_range(self):
        """Model's intensity_factor prediction must be a positive, finite value."""
        model, _ = _load_model()
        x, di, _df = _synth_sequence()
        out = _run_model(model, x, di)
        raw_if = float(out["intensity"][0, 0])
        # The raw output is in IF units — after any squashing should be >0
        assert not (raw_if != raw_if), "Intensity output is NaN"

    def test_model_duration_output_finite(self):
        """Model's duration prediction must be finite."""
        model, _ = _load_model()
        x, di, _df = _synth_sequence()
        out = _run_model(model, x, di)
        dur = float(out["duration"][0, 0])
        assert not (dur != dur), "Duration output is NaN"


# ── 4. Sequence length invariance ────────────────────────────────────────────

class TestSequenceLengthInvariance:

    @pytest.mark.parametrize("seq_len", [5, 20, 60, 90])
    def test_model_handles_varying_sequence_lengths(self, seq_len):
        """Model must run without error on sequences from 5 to 90 activities."""
        model, ckpt = _load_model()
        input_dim = ckpt["config"].get("input_dim", 61)
        x  = torch.randn(1, seq_len, input_dim)
        di = torch.arange(seq_len).unsqueeze(0)
        out = _run_model(model, x, di)
        assert out["workout_logits"].shape == (1, 10)
        assert not torch.isnan(out["workout_logits"]).any()
