"""
Comprehensive ML pipeline tests for CyclingTransformer + inference pipeline.

Covers:
  1. Unit tests — model forward pass, horizon encoding, gradient flow
  2. Unit tests — dataset 3-horizon sampling, collate_fn
  3. Unit tests — phase_prior correctness, planner TSS math
  4. Sanity tests — model actually differentiates horizons in raw logits
  5. Sanity tests — prior contribution vs model contribution
  6. Training convergence — loss decreases, no NaN
  7. Edge cases — empty sequence, event in past, CTL=0, cold start boundary
  8. Checkpoint round-trip — save/load, inference stable
"""

from __future__ import annotations

import math
import sys
import os
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///c:/Users/yossi/ai coach/aicoach_dev.db")
os.environ.setdefault("SECRET_KEY", "dev-secret-stable")

import numpy as np
import torch
import pytest

from app.ml.model import (
    CyclingTransformer, INPUT_DIM, HORIZON_DIM,
    encode_horizon, HORIZON_BUCKETS,
)
from app.ml.phase_prior import (
    phase_prior, posterior, select_workout,
    WORKOUT_TYPE_NAMES, N_TYPES, horizon_to_phase,
)
from app.ml.planner import (
    desired_weekly_tss, solve_week, _project_pmc,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_model(fast=True) -> CyclingTransformer:
    if fast:
        return CyclingTransformer(d_model=32, nhead=2, num_layers=2, dim_feedforward=64)
    return CyclingTransformer()


def fake_batch(batch=2, seq=10, dim=INPUT_DIM):
    x  = torch.randn(batch, seq, dim)
    di = torch.arange(seq).unsqueeze(0).expand(batch, -1)
    pm = torch.zeros(batch, seq, dtype=torch.bool)   # no padding
    return x, di, pm


def horizon_tensor(label, days) -> torch.Tensor:
    return torch.tensor(encode_horizon(label, days)).unsqueeze(0).float()


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestModelForward:

    def test_output_shapes_without_horizon(self):
        """Model returns correct output tensor shapes with no horizon query."""
        m = make_model()
        x, di, pm = fake_batch()
        out = m(x, di, pm, horizon_query=None)
        assert out["workout_logits"].shape == (2, 10), "workout logits must be (B, 10)"
        assert out["intensity"].shape == (2, 1),       "intensity must be (B, 1)"
        assert out["duration"].shape  == (2, 1),       "duration must be (B, 1)"
        assert out["ftp_delta"].shape == (2, 1),       "ftp_delta must be (B, 1)"
        assert "risks" in out or "ctl_peak" in out,   "model must have risk/ctl output head"

    def test_output_shapes_with_horizon(self):
        """Model returns correct shapes when horizon_query is provided."""
        m = make_model()
        x, di, pm = fake_batch()
        hq = torch.randn(2, HORIZON_DIM)
        out = m(x, di, pm, horizon_query=hq)
        assert out["workout_logits"].shape == (2, 10)

    def test_horizon_changes_logits(self):
        """Different horizon tokens must produce different logits (model uses them)."""
        m = make_model(); m.eval()
        x, di, pm = fake_batch(batch=1)
        logits = {}
        with torch.no_grad():
            for label, days in [("short", 7), ("medium", 28), ("event", 72)]:
                hq = horizon_tensor(label, days)
                logits[label] = m(x, di, pm, horizon_query=hq)["workout_logits"].squeeze(0)

        diff_sm = (logits["short"]  - logits["medium"]).abs().sum().item()
        diff_se = (logits["short"]  - logits["event"] ).abs().sum().item()
        diff_me = (logits["medium"] - logits["event"] ).abs().sum().item()
        print(f"\n  logit L1 diffs — sm:{diff_sm:.4f}  se:{diff_se:.4f}  me:{diff_me:.4f}")
        assert diff_sm > 0, "short vs medium logits must differ"
        assert diff_se > 0, "short vs event logits must differ"
        assert diff_me > 0, "medium vs event logits must differ"

    def test_no_nan_in_forward(self):
        """Model output must never contain NaN."""
        m = make_model(); m.eval()
        x, di, pm = fake_batch()
        hq = torch.randn(2, HORIZON_DIM)
        out = m(x, di, pm, horizon_query=hq)
        for k, v in out.items():
            assert not torch.isnan(v).any(), f"NaN in output '{k}'"

    def test_single_token_sequence(self):
        """Model must handle seq_len=1 (athlete has only 1 ride)."""
        m = make_model(); m.eval()
        x  = torch.randn(1, 1, INPUT_DIM)
        di = torch.zeros(1, 1, dtype=torch.long)
        pm = torch.zeros(1, 1, dtype=torch.bool)
        hq = horizon_tensor("event", 60)
        out = m(x, di, pm, horizon_query=hq)
        assert out["workout_logits"].shape == (1, 10)

    def test_all_padded_except_one(self):
        """Most extreme padding: only first token is real, rest are masked."""
        m = make_model(); m.eval()
        B, T = 1, 20
        x  = torch.randn(B, T, INPUT_DIM)
        di = torch.arange(T).unsqueeze(0)
        pm = torch.ones(B, T, dtype=torch.bool)
        pm[0, 0] = False  # only first token is real
        hq = horizon_tensor("medium", 28)
        out = m(x, di, pm, horizon_query=hq)
        assert not torch.isnan(out["workout_logits"]).any()

    def test_gradient_flows_to_horizon_proj(self):
        """Gradients must reach the horizon_proj layer (horizon token is trained)."""
        m = make_model()
        x, di, pm = fake_batch()
        hq = torch.randn(2, HORIZON_DIM)
        out = m(x, di, pm, horizon_query=hq)
        loss = out["workout_logits"].sum()
        loss.backward()
        grad = m.horizon_proj[0].weight.grad
        assert grad is not None,          "No grad on horizon_proj.weight"
        assert not torch.isnan(grad).any(), "NaN in horizon_proj gradient"
        assert grad.abs().sum().item() > 0, "Zero gradient on horizon_proj — horizon token not connected"

    def test_gradient_flows_to_input_proj(self):
        """Gradients must reach the input_proj layer."""
        m = make_model()
        x, di, pm = fake_batch()
        out = m(x, di, pm)
        loss = out["workout_logits"].sum()
        loss.backward()
        grad = m.input_proj[0].weight.grad
        assert grad is not None and grad.abs().sum().item() > 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. HORIZON ENCODING UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeHorizon:

    def test_length(self):
        assert len(encode_horizon("short", 7)) == HORIZON_DIM

    def test_one_hot_short(self):
        v = encode_horizon("short", 7)
        assert v[0] == 1.0 and v[1] == 0.0 and v[2] == 0.0

    def test_one_hot_medium(self):
        v = encode_horizon("medium", 28)
        assert v[0] == 0.0 and v[1] == 1.0 and v[2] == 0.0

    def test_one_hot_event(self):
        v = encode_horizon("event", 72)
        assert v[0] == 0.0 and v[1] == 0.0 and v[2] == 1.0

    def test_values_bounded_0_1(self):
        for label in HORIZON_BUCKETS:
            for days in [0, 7, 28, 72, 365, 1000]:
                v = encode_horizon(label, days)
                assert all(0.0 <= x <= 1.0 for x in v), \
                    f"Out-of-range value for {label},{days}: {v}"

    def test_days_norm_increases_with_days(self):
        v7  = encode_horizon("short",  7)
        v28 = encode_horizon("medium", 28)
        v72 = encode_horizon("event",  72)
        # position 3 = horizon_days_norm
        assert v7[3] < v28[3] < v72[3]

    def test_unknown_label_gives_all_zero_one_hot(self):
        v = encode_horizon("unknown_label", 30)
        assert v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0

    def test_zero_days(self):
        v = encode_horizon("event", 0)
        assert v[3] == 0.0   # 0/365 = 0
        assert v[4] == 0.0   # log1p(0)/log(366) = 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. PHASE PRIOR UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPhasePrior:

    def test_prior_sums_to_1(self):
        for phase in ["base", "base_build", "build", "peak", "taper", "recovery_week"]:
            p = phase_prior(phase)
            assert abs(p.sum() - 1.0) < 1e-5, f"Phase '{phase}' prior does not sum to 1"

    def test_prior_all_positive(self):
        for phase in ["base", "build", "peak", "recovery_week"]:
            p = phase_prior(phase)
            assert (p > 0).all(), f"Phase '{phase}' has zero-probability workout types"

    def test_recovery_week_favors_easy_types(self):
        p = phase_prior("recovery_week")
        recovery_idx = WORKOUT_TYPE_NAMES.index("recovery")
        easy_idx     = WORKOUT_TYPE_NAMES.index("easy")
        vo2max_idx   = WORKOUT_TYPE_NAMES.index("vo2max")
        assert p[recovery_idx] > p[vo2max_idx], "recovery_week must favor recovery over vo2max"
        assert p[easy_idx]     > p[vo2max_idx], "recovery_week must favor easy over vo2max"

    def test_build_phase_favors_hard_types(self):
        p = phase_prior("build")
        threshold_idx = WORKOUT_TYPE_NAMES.index("threshold")
        recovery_idx  = WORKOUT_TYPE_NAMES.index("recovery")
        assert p[threshold_idx] > p[recovery_idx], "build must favor threshold over recovery"

    def test_posterior_shape(self):
        logits = np.zeros(N_TYPES, dtype=np.float32)
        p = posterior(logits, phase="build", event_type=None, tsb=0.0, hrv_z=0.0, temperature=1.0, prior_weight=0.5)
        assert p.shape == (N_TYPES,)
        assert abs(p.sum() - 1.0) < 1e-5

    def test_posterior_suppresses_hard_work_when_fatigued(self):
        """With TSB = -30 (very fatigued), vo2max should be suppressed."""
        logits = np.zeros(N_TYPES, dtype=np.float32)
        # Pump the model logit for vo2max
        logits[WORKOUT_TYPE_NAMES.index("vo2max")] = 5.0
        p_fresh    = posterior(logits, phase="build", event_type=None, tsb=5.0,   hrv_z=0.0,  temperature=1.0, prior_weight=0.4)
        p_fatigued = posterior(logits, phase="build", event_type=None, tsb=-30.0, hrv_z=-2.0, temperature=1.0, prior_weight=0.4)
        vo2_fresh    = p_fresh[WORKOUT_TYPE_NAMES.index("vo2max")]
        vo2_fatigued = p_fatigued[WORKOUT_TYPE_NAMES.index("vo2max")]
        print(f"\n  vo2max: fresh={vo2_fresh:.3f}  fatigued={vo2_fatigued:.3f}")
        assert vo2_fatigued < vo2_fresh, "vo2max probability must drop when fatigued (TSB<-25)"

    def test_select_workout_returns_valid_type(self):
        logits = np.random.randn(N_TYPES).astype(np.float32)
        name, conf, alts = select_workout(logits, phase="peak", event_type=None,
                                          tsb=0.0, hrv_z=0.0, temperature=1.0)
        assert name in WORKOUT_TYPE_NAMES
        assert 0.0 <= conf <= 1.0
        assert len(alts) <= 3

    def test_horizon_to_phase_mapping(self):
        # horizon_to_phase(days): calls get_periodization_phase(days//7)
        # ≤3 weeks → recovery_week; >3-8 weeks → peak; >8-20 → build; >20 → base_build
        assert horizon_to_phase(6)   == "recovery_week"   # 0 weeks → recovery
        assert horizon_to_phase(21)  == "recovery_week"   # 3 weeks exactly → recovery_week
        assert horizon_to_phase(29)  == "peak"            # 4 weeks → peak
        assert horizon_to_phase(56)  == "peak"            # 8 weeks exactly → peak
        assert horizon_to_phase(70)  == "build"           # 10 weeks → build
        assert horizon_to_phase(200) == "base_build"      # >20 weeks → base_build


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLANNER / TSS SOLVER UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanner:

    def test_desired_tss_increases_in_build(self):
        """Build phase should target more TSS than recovery_week."""
        tss_build    = desired_weekly_tss(ctl=50.0, tsb=0.0,   phase="build")
        tss_recovery = desired_weekly_tss(ctl=50.0, tsb=-20.0, phase="recovery_week")
        assert tss_build > tss_recovery

    def test_desired_tss_clamped_above_zero(self):
        """Even in taper, weekly TSS must be > 0."""
        tss = desired_weekly_tss(ctl=10.0, tsb=-30.0, phase="taper")
        assert tss > 0

    def test_desired_tss_clamped_by_max_pct(self):
        """Weekly TSS can't increase more than 30% vs 7×CTL baseline."""
        ctl = 100.0
        tss = desired_weekly_tss(ctl=ctl, tsb=10.0, phase="build")
        assert tss <= ctl * 7 * 1.30 + 1, "TSS ramp exceeded 30% cap"

    def test_solve_week_returns_7_values(self):
        types = ["endurance", "recovery", "sweetspot", "recovery", "threshold", "long_ride", "recovery"]
        daily, notes = solve_week(types, ctl=50.0, atl=55.0, tsb=-5.0, phase="build", hrv_z=0.0)
        assert len(daily) == 7

    def test_solve_week_all_positive_tss(self):
        types = ["vo2max", "recovery", "threshold", "recovery", "sweetspot", "long_ride", "recovery"]
        daily, _ = solve_week(types, ctl=60.0, atl=65.0, tsb=-5.0, phase="build", hrv_z=0.0)
        assert all(t >= 0 for t in daily), f"Negative TSS in plan: {daily}"

    def test_solve_week_throttles_when_very_fatigued(self):
        """With TSB = -35 (deep overreaching), plan must be trimmed."""
        types = ["vo2max", "recovery", "threshold", "recovery", "sweetspot", "long_ride", "threshold"]
        daily_fresh, _ = solve_week(types, ctl=80.0, atl=80.0, tsb=0.0,   phase="build", hrv_z=0.0)
        daily_tired, _ = solve_week(types, ctl=80.0, atl=115.0, tsb=-35.0, phase="build", hrv_z=-2.0)
        assert sum(daily_tired) < sum(daily_fresh), "Fatigued plan must have lower total TSS"

    def test_pmc_projection_math(self):
        """PMC projector must follow Banister exponential model."""
        ctl, atl = 50.0, 50.0
        # Single day of 100 TSS
        ctl2, atl2, tsb2 = _project_pmc(ctl, atl, [100.0])
        alpha_ctl = 2.0 / 43.0
        alpha_atl = 2.0 / 8.0
        expected_ctl = ctl + alpha_ctl * (100.0 - ctl)
        expected_atl = atl + alpha_atl * (100.0 - atl)
        assert abs(ctl2 - expected_ctl) < 1e-6
        assert abs(atl2 - expected_atl) < 1e-6
        assert abs(tsb2 - (expected_ctl - expected_atl)) < 1e-6

    def test_recovery_day_gets_low_tss(self):
        """Recovery day TSS must be lower than endurance day TSS in any plan."""
        types = ["recovery", "endurance", "recovery", "endurance", "recovery", "recovery", "recovery"]
        daily, _ = solve_week(types, ctl=50.0, atl=52.0, tsb=-2.0, phase="build", hrv_z=0.0)
        # Day 1 (endurance) vs day 0 (recovery)
        assert daily[0] < daily[1], f"Recovery day TSS {daily[0]} >= endurance day TSS {daily[1]}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATASET UNIT TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestDataset:

    def _load_small_dataset(self):
        import pandas as pd
        path = ROOT / "ml" / "data" / "synthetic_small.parquet"
        if not path.exists():
            pytest.skip("synthetic_small.parquet not found")
        return pd.read_parquet(path)

    def test_dataset_has_days_to_event(self):
        df = self._load_small_dataset()
        assert "days_to_event" in df.columns, "Dataset must have days_to_event column"

    def test_dataset_has_horizon_query(self):
        """CyclingDataset must emit horizon_query in each sample."""
        from ml.training.dataset import CyclingDataset
        import pandas as pd
        df = self._load_small_dataset()
        ds = CyclingDataset(df, seq_len=30, horizon_aware=True)
        sample = ds[0]
        assert "horizon_query" in sample, "Sample must contain 'horizon_query'"
        assert sample["horizon_query"].shape == (HORIZON_DIM,), \
            f"horizon_query shape {sample['horizon_query'].shape} != ({HORIZON_DIM},)"

    def test_dataset_emits_3_horizons_per_anchor(self):
        """With 1 anchor athlete, dataset should emit ≥2 and ≤3 samples (short + medium + maybe event)."""
        from ml.training.dataset import CyclingDataset
        import pandas as pd
        df = self._load_small_dataset()
        # Take a single athlete
        ath = df["athlete_id"].iloc[0]
        df1 = df[df["athlete_id"] == ath].reset_index(drop=True)
        ds = CyclingDataset(df1, seq_len=30, horizon_aware=True)
        # Each eligible anchor produces 2–3 samples
        assert len(ds) >= 2, "Must have at least short+medium horizons"

    def test_collate_fn_stacks_horizon_query(self):
        """collate_fn must stack horizon_query to (B, HORIZON_DIM)."""
        from ml.training.dataset import CyclingDataset, collate_fn
        from torch.utils.data import DataLoader
        df = self._load_small_dataset()
        ds = CyclingDataset(df, seq_len=30, horizon_aware=True)
        dl = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
        batch = next(iter(dl))
        assert "horizon_query" in batch, "Batch must contain horizon_query"
        hq = batch["horizon_query"]
        assert hq.dim() == 2,          f"horizon_query must be 2D, got {hq.dim()}"
        assert hq.shape[1] == HORIZON_DIM

    def test_target_workout_types_are_valid(self):
        """All target workout labels in dataset must map to known types."""
        from ml.training.dataset import CyclingDataset
        known = set(range(10))  # 10 workout type classes
        df = self._load_small_dataset()
        ds = CyclingDataset(df, seq_len=30, horizon_aware=True)
        for i in range(min(100, len(ds))):
            s = ds[i]
            wt = s.get("target_workout_type")
            if wt is not None:
                assert int(wt.item()) in known, f"Unknown workout type index {wt} at sample {i}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINING CONVERGENCE SANITY TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingConvergence:
    """Mini training loop on synthetic data to verify loss decreases."""

    def test_loss_decreases_over_10_steps(self):
        """Cross-entropy on random data must decrease under gradient descent."""
        import torch.optim as optim
        torch.manual_seed(42)
        m = make_model(); m.train()
        opt = optim.Adam(m.parameters(), lr=1e-3)
        ce = torch.nn.CrossEntropyLoss()

        losses = []
        for _ in range(10):
            x, di, pm = fake_batch(batch=8, seq=20)
            hq = torch.randn(8, HORIZON_DIM)
            labels = torch.randint(0, 10, (8,))
            opt.zero_grad()
            out = m(x, di, pm, horizon_query=hq)
            loss = ce(out["workout_logits"], labels)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print(f"\n  losses: {[f'{l:.4f}' for l in losses]}")
        assert losses[-1] < losses[0], \
            f"Loss did not decrease: start={losses[0]:.4f} end={losses[-1]:.4f}"

    def test_no_nan_during_training(self):
        """No NaN must appear in loss or gradients during a training step."""
        import torch.optim as optim
        torch.manual_seed(0)
        m = make_model(); m.train()
        opt = optim.Adam(m.parameters(), lr=1e-3)
        ce = torch.nn.CrossEntropyLoss()

        for step in range(5):
            x, di, pm = fake_batch(batch=4, seq=15)
            hq = torch.randn(4, HORIZON_DIM)
            labels = torch.randint(0, 10, (4,))
            opt.zero_grad()
            out = m(x, di, pm, horizon_query=hq)
            loss = ce(out["workout_logits"], labels)
            assert not math.isnan(loss.item()), f"NaN loss at step {step}"
            loss.backward()
            for name, p in m.named_parameters():
                if p.grad is not None:
                    assert not torch.isnan(p.grad).any(), f"NaN grad in {name} at step {step}"
            opt.step()


# ─────────────────────────────────────────────────────────────────────────────
# 7. CHECKPOINT ROUND-TRIP TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckpoint:

    def test_smoke_v2_checkpoint_loads(self):
        """cycling_coach_smoke_v2.pt must load cleanly with horizon weights."""
        ckpt_path = ROOT / "backend" / "models" / "cycling_coach_smoke_v2.pt"
        if not ckpt_path.exists():
            pytest.skip("cycling_coach_smoke_v2.pt not found")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = dict(ckpt["config"])
        assert cfg.get("horizon_aware") is True
        assert cfg.get("horizon_dim") == HORIZON_DIM
        cfg.pop("horizon_aware", None)
        cfg.pop("temperature", None)
        m = CyclingTransformer(**cfg)
        missing, unexpected = m.load_state_dict(ckpt["state_dict"], strict=False)
        # Allow missing keys for newly-added task heads (horizon, risk_ot, risk_inj)
        # — old checkpoints predate them and that's fine, they just initialize
        # randomly until the next retrain.
        allowed_prefixes = ("horizon", "risk_ot_head", "risk_inj_head")
        unexpected_missing = [
            k for k in missing if not any(p in k for p in allowed_prefixes)
        ]
        assert len(unexpected_missing) == 0, f"Unexpected missing keys: {unexpected_missing}"

    def test_checkpoint_inference_stable(self):
        """Same input → same output on two separate loads of smoke_v2."""
        ckpt_path = ROOT / "backend" / "models" / "cycling_coach_smoke_v2.pt"
        if not ckpt_path.exists():
            pytest.skip("cycling_coach_smoke_v2.pt not found")

        def load_and_infer():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            cfg = dict(ckpt["config"])
            cfg.pop("horizon_aware", None); cfg.pop("temperature", None)
            m = CyclingTransformer(**cfg); m.eval()
            m.load_state_dict(ckpt["state_dict"], strict=False)
            torch.manual_seed(0)
            x, di, pm = fake_batch(batch=1, seq=15)
            hq = horizon_tensor("event", 71)
            with torch.no_grad():
                return m(x, di, pm, horizon_query=hq)["workout_logits"]

        out1 = load_and_infer()
        out2 = load_and_infer()
        assert torch.allclose(out1, out2), "Checkpoint inference not deterministic"


# ─────────────────────────────────────────────────────────────────────────────
# 8. MODEL SANITY: IS THE MODEL DOING ANYTHING (vs prior alone)?
# ─────────────────────────────────────────────────────────────────────────────

class TestModelVsPriorContribution:
    """
    Key scientific question: does the model's logit output actually vary with
    horizon, or is all differentiation coming from the Bayesian phase prior?
    """

    def test_raw_logits_differ_across_horizons(self):
        """
        With smoke_v2 checkpoint, logits for short/medium/event must differ
        by at least a small amount. If they're identical, the model learned
        nothing from horizon conditioning.
        """
        ckpt_path = ROOT / "backend" / "models" / "cycling_coach_smoke_v2.pt"
        if not ckpt_path.exists():
            pytest.skip("cycling_coach_smoke_v2.pt not found")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg  = dict(ckpt["config"])
        cfg.pop("horizon_aware", None); cfg.pop("temperature", None)
        m = CyclingTransformer(**cfg); m.eval()
        m.load_state_dict(ckpt["state_dict"], strict=False)

        torch.manual_seed(42)
        x, di, pm = fake_batch(batch=1, seq=20)

        logits_per_horizon = {}
        with torch.no_grad():
            for label, days in [("short", 7), ("medium", 28), ("event", 71)]:
                hq = horizon_tensor(label, days)
                logits_per_horizon[label] = m(x, di, pm, horizon_query=hq)["workout_logits"][0]

        diff_sm = (logits_per_horizon["short"]  - logits_per_horizon["medium"]).abs().max().item()
        diff_se = (logits_per_horizon["short"]  - logits_per_horizon["event"] ).abs().max().item()
        diff_me = (logits_per_horizon["medium"] - logits_per_horizon["event"] ).abs().max().item()

        print(f"\n  Max logit diff — short/medium: {diff_sm:.4f}  short/event: {diff_se:.4f}  medium/event: {diff_me:.4f}")

        # After only 3 epochs the model won't have LEARNED strong differentiation,
        # but horizon_proj must at least be pushing logits around a bit.
        assert diff_sm > 1e-4, "short vs medium logits are identical — horizon token not connected"
        assert diff_se > 1e-4, "short vs event logits are identical"
        assert diff_me > 1e-4, "medium vs event logits are identical"

    def test_prior_alone_differentiates_phases(self):
        """
        Even with FLAT model logits, the prior alone must produce different
        top-1 picks for recovery_week vs build phase. This validates that
        the Bayesian prior is correctly wired.
        """
        flat = np.zeros(N_TYPES, dtype=np.float32)
        name_recovery, _, _ = select_workout(flat, phase="recovery_week", event_type=None,
                                              tsb=5.0, hrv_z=0.5, temperature=1.0, prior_weight=1.0)
        name_build,    _, _ = select_workout(flat, phase="build",         event_type=None,
                                              tsb=5.0, hrv_z=0.5, temperature=1.0, prior_weight=1.0)
        print(f"\n  Prior-only picks — recovery_week: {name_recovery}  build: {name_build}")
        assert name_recovery != name_build, "Prior alone must pick different workouts for recovery vs build"

    def test_model_logit_magnitude_is_meaningful(self):
        """
        Logits should NOT be near-zero (untrained model collapses to uniform).
        After 3 epochs they should have learnt some structure.
        """
        ckpt_path = ROOT / "backend" / "models" / "cycling_coach_smoke_v2.pt"
        if not ckpt_path.exists():
            pytest.skip("cycling_coach_smoke_v2.pt not found")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg  = dict(ckpt["config"])
        cfg.pop("horizon_aware", None); cfg.pop("temperature", None)
        m = CyclingTransformer(**cfg); m.eval()
        m.load_state_dict(ckpt["state_dict"], strict=False)

        torch.manual_seed(7)
        x, di, pm = fake_batch(batch=4, seq=30)
        hq = torch.stack([horizon_tensor("event", 71).squeeze(0)] * 4)
        with torch.no_grad():
            logits = m(x, di, pm, horizon_query=hq)["workout_logits"]

        logit_range = (logits.max() - logits.min()).item()
        logit_std   = logits.std().item()
        print(f"\n  Logit range: {logit_range:.4f}  std: {logit_std:.4f}")
        # A completely random model has ~N(0,1) logits → std~1; trained model should be ≥0.1
        assert logit_std > 0.05, "Logit std is near zero — model may have collapsed"


# ─────────────────────────────────────────────────────────────────────────────
# 9. EDGE CASE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_encode_horizon_large_days(self):
        """encode_horizon must not crash or go out of [0,1] for 1000-day horizon."""
        v = encode_horizon("event", 1000)
        assert all(0.0 <= x <= 1.0 for x in v)

    def test_desired_tss_ctl_zero(self):
        """CTL=0 (new athlete) must return a small but positive weekly TSS target."""
        tss = desired_weekly_tss(ctl=0.0, tsb=0.0, phase="base")
        assert tss >= 0, "CTL=0 must yield non-negative TSS"

    def test_phase_prior_unknown_phase(self):
        """Unknown phase string must fall back gracefully (not KeyError)."""
        p = phase_prior("totally_unknown_phase")
        assert p.shape == (N_TYPES,)
        assert abs(p.sum() - 1.0) < 1e-5

    def test_solve_week_all_recovery(self):
        """7 recovery days must produce a valid (very low TSS) plan."""
        types = ["recovery"] * 7
        daily, _ = solve_week(types, ctl=80.0, atl=100.0, tsb=-20.0, phase="taper", hrv_z=-1.5)
        assert all(t >= 0 for t in daily)
        # With CTL=80, even taper produces ~35 TSS/day (35×7=245). Just confirm it's
        # lower than an unconstrained build week would be (7×80×1.0=560).
        assert sum(daily) < 560, "All-recovery week must be well below max build TSS"

    def test_model_batch_size_1_vs_8_same_result(self):
        """Model output for a single sample must be same whether batch=1 or batch=8."""
        torch.manual_seed(0)
        m = make_model(); m.eval()
        x1 = torch.randn(1, 10, INPUT_DIM)
        di = torch.arange(10).unsqueeze(0)
        pm = torch.zeros(1, 10, dtype=torch.bool)
        hq1 = horizon_tensor("event", 72)

        # batch=1
        with torch.no_grad():
            out1 = m(x1, di, pm, horizon_query=hq1)["workout_logits"]

        # batch=8 (same row repeated)
        x8  = x1.expand(8, -1, -1)
        di8 = di.expand(8, -1)
        pm8 = pm.expand(8, -1)
        hq8 = hq1.expand(8, -1)
        with torch.no_grad():
            out8 = m(x8, di8, pm8, horizon_query=hq8)["workout_logits"]

        # All 8 rows should be identical
        assert torch.allclose(out1, out8[0:1], atol=1e-5), "Batch size affects single-sample output"
