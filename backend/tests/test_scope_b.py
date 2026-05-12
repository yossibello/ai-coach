"""
Tests for Scope-B additions:
  * synthetic data risk labels exist and are well-formed
  * dataset.collate_fn carries risk targets through
  * train.py's risk losses run a forward+backward step without NaN
  * Outcome / RecommendationFeedback ORM tables are registered
  * /recommendation/{id}/feedback endpoint is auth-protected
  * outcomes.backfill_user_outcomes is callable and returns 0 cleanly
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Synthetic data carries risk labels ───────────────────────────────────────

def test_synthetic_smoke_parquet_has_risk_labels():
    p = ROOT / "ml" / "data" / "synthetic_smoke.parquet"
    if not p.exists():
        pytest.skip("synthetic_smoke.parquet not generated yet")
    df = pd.read_parquet(p)
    assert "risk_ot_class" in df.columns
    assert "risk_inj_target" in df.columns
    # Class space sanity.
    classes = set(df["risk_ot_class"].unique())
    assert classes.issubset({0, 1, 2}), classes
    inj_vals = set(df["risk_inj_target"].unique())
    assert inj_vals.issubset({0, 1}), inj_vals
    # Label distribution must not be degenerate (all-one-class is a bug).
    assert len(classes) >= 2, "risk_ot_class is constant — generator broken"


# ── Dataset surfaces risk targets ────────────────────────────────────────────

def test_dataset_collate_includes_risk_targets():
    p = ROOT / "ml" / "data" / "synthetic_smoke.parquet"
    if not p.exists():
        pytest.skip("synthetic_smoke.parquet not generated yet")
    from ml.training.dataset import CyclingDataset, collate_fn

    df = pd.read_parquet(p)
    ds = CyclingDataset(df, seq_len=30)
    if len(ds) < 4:
        pytest.skip("dataset too small")
    batch = collate_fn([ds[i] for i in range(4)])
    assert "target_risk_ot" in batch
    assert "target_risk_inj" in batch
    assert batch["target_risk_ot"].dtype == torch.long
    assert batch["target_risk_inj"].dtype == torch.float32
    assert batch["target_risk_ot"].shape == (4,)
    assert batch["target_risk_inj"].shape == (4,)


# ── Model exposes raw risk logits, forward+backward is finite ────────────────

def test_model_risk_logits_finite_and_trainable():
    from app.ml.model import CyclingTransformer, INPUT_DIM, HORIZON_DIM

    m = CyclingTransformer(
        input_dim=INPUT_DIM, d_model=32, nhead=4, num_layers=2,
        horizon_dim=HORIZON_DIM,
    )
    B, T = 2, 8
    x = torch.randn(B, T, INPUT_DIM)
    day_indices = torch.arange(T).unsqueeze(0).expand(B, T).contiguous()
    h = torch.zeros(B, HORIZON_DIM); h[:, 0] = 1.0
    out = m(x, day_indices, horizon_query=h)

    # Both heads expose logits + the combined `risks` probability vector.
    assert out["risk_ot_logits"].shape == (B, 3)
    assert out["risk_inj_logit"].shape == (B, 1)
    assert torch.isfinite(out["risk_ot_logits"]).all()
    assert torch.isfinite(out["risk_inj_logit"]).all()
    assert out["risks"].shape == (B, 3)
    # Combined probability vector is bounded.
    assert ((out["risks"] >= 0) & (out["risks"] <= 1)).all()

    # CE+BCE risk losses produce a finite gradient.
    tgt_ot  = torch.tensor([0, 2], dtype=torch.long)
    tgt_inj = torch.tensor([1.0, 0.0], dtype=torch.float32)
    ce  = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss()
    loss = ce(out["risk_ot_logits"], tgt_ot) + bce(out["risk_inj_logit"].squeeze(-1), tgt_inj)
    assert torch.isfinite(loss)
    loss.backward()
    # At least one head parameter received a non-zero grad.
    grads = [p.grad for p in m.risk_ot_head.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


# ── Outcome ORM is registered ────────────────────────────────────────────────

def test_outcome_tables_registered_on_metadata():
    from app.core.database import Base
    from app.models import outcome  # noqa: F401  — triggers registration

    names = set(Base.metadata.tables.keys())
    assert "recommendation_feedback" in names
    assert "prediction_outcomes" in names


# ── Feedback endpoint is auth-protected ──────────────────────────────────────

@pytest.mark.asyncio
async def test_feedback_endpoint_requires_auth():
    from httpx import AsyncClient, ASGITransport
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post(
            "/api/v1/coach/recommendation/00000000-0000-0000-0000-000000000000/feedback",
            json={"action": "accepted"},
        )
    assert r.status_code in (401, 403)


# ── Outcome backfill no-op for unknown user is harmless ──────────────────────

@pytest.mark.asyncio
async def test_backfill_user_outcomes_handles_no_recs():
    from app.core.database import AsyncSessionLocal, init_db
    from app.ml.outcomes import backfill_user_outcomes

    await init_db()
    async with AsyncSessionLocal() as db:
        n = await backfill_user_outcomes("nonexistent-user-id", db)
    assert n == 0
