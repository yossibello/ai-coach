"""
Model registry — single point of access for all model artifacts.

The system has a 3-tier model architecture (see docs/architecture.md):

  Layer 1: SYNTHETIC BASE      ./backend/models/cycling_coach_v1_synthetic.pt
                               Trained once on synthetic data. Frozen forever.
                               Used as cold-start fallback and as warm-start init
                               for community retrains.

  Layer 2: COMMUNITY MODEL     ./backend/models/community/cycling_coach_v{N}_community.pt
                               Periodically retrained: synthetic base + real high-quality
                               rides (only from users with allow_for_training=True).
                               This is the model new users get on signup.

  Layer 3: USER ADAPTER        ./backend/models/adapters/{user_id}/v{N}.pt
                               Tiny LoRA-style adapter on top of the active community model.
                               Trained per user once they have ~50+ rides.
                               Falls back to community model if missing.

This module hides all of that behind a clean API:

    registry = ModelRegistry()
    model    = registry.get_model_for_user(user_id, db)
    version  = registry.active_version()

It also exposes the metadata needed by the future retrain worker:

    registry.list_versions()
    registry.promote("community_v3")
    registry.rollback_user_adapter(user_id)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from app.core.config import settings
from app.ml.model import CyclingTransformer

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


# ── Paths ────────────────────────────────────────────────────────────────────
def _models_root() -> Path:
    # ML_MODEL_PATH points at the active production model file.
    # Its parent dir is the model registry root.
    p = Path(settings.ML_MODEL_PATH).parent
    p.mkdir(parents=True, exist_ok=True)
    (p / "community").mkdir(exist_ok=True)
    (p / "adapters").mkdir(exist_ok=True)
    return p


def _registry_index_path() -> Path:
    return _models_root() / "registry.json"


# ── Default registry index ───────────────────────────────────────────────────
# Created on first access; later updated by the retrain worker / ops scripts.
_DEFAULT_INDEX = {
    "active_base":      "synthetic_v1",
    "active_community": None,            # set after first community retrain
    "versions": {
        "synthetic_v1": {
            "path":       "cycling_coach.pt",   # the file pretrain.ps1 produces
            "kind":       "base",
            "trained_on": "synthetic_50k",
            "val_loss":   None,
        }
    },
}


@dataclass
class _CachedModel:
    version: str
    model: CyclingTransformer
    mtime: float


class ModelRegistry:
    """Process-singleton model registry. Loaded lazily, cached in-memory."""

    _instance: "ModelRegistry | None" = None

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}    # type: ignore[attr-defined]
        return cls._instance

    # ── Index management ─────────────────────────────────────────────────────
    def _read_index(self) -> dict:
        path = _registry_index_path()
        if not path.exists():
            self._write_index(_DEFAULT_INDEX)
            return dict(_DEFAULT_INDEX)
        with path.open() as f:
            return json.load(f)

    def _write_index(self, idx: dict) -> None:
        with _registry_index_path().open("w") as f:
            json.dump(idx, f, indent=2)

    # ── Public read API ──────────────────────────────────────────────────────
    def active_version(self) -> str:
        """Return the version key currently serving (community if set, else base)."""
        idx = self._read_index()
        return idx.get("active_community") or idx["active_base"]

    def list_versions(self) -> dict:
        return self._read_index()["versions"]

    # ── Loading ──────────────────────────────────────────────────────────────
    def get_base_model(self, version: str | None = None) -> CyclingTransformer:
        """Load a model by version key. Falls back to active version if None."""
        version = version or self.active_version()
        cache = self._cache  # type: ignore[attr-defined]

        idx = self._read_index()
        meta = idx["versions"].get(version)
        if not meta:
            raise FileNotFoundError(f"Unknown model version: {version}")

        full_path = _models_root() / meta["path"]
        if not full_path.exists():
            raise FileNotFoundError(f"Model file missing: {full_path}")

        mtime = full_path.stat().st_mtime
        cached = cache.get(version)
        if cached and cached.mtime == mtime:
            return cached.model

        model = CyclingTransformer()
        state = torch.load(full_path, map_location="cpu")
        # Allow either raw state_dict or a checkpoint dict
        state_dict = state.get("model", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict)
        model.eval()
        cache[version] = _CachedModel(version=version, model=model, mtime=mtime)
        return model

    def get_model_for_user(
        self, user_id: str, db: "Session | None" = None
    ) -> tuple[CyclingTransformer, str]:
        """
        Return (model, version_label) for a given user.

        Phase 1: just returns the active base/community model.
        Phase 3: will compose community model + user adapter (LoRA merge).
        """
        # Phase 3 hook — placeholder, no-op for now
        adapter_path = _models_root() / "adapters" / user_id / "active.pt"
        version = self.active_version()
        model = self.get_base_model(version)

        if adapter_path.exists():
            # TODO(phase-3): load LoRA weights and merge / wrap model
            version = f"{version}+adapter"

        return model, version

    # ── Phase-2/3 write API (used by retrain worker, ops scripts) ────────────
    def register_version(
        self,
        version: str,
        path: str,
        kind: str,
        trained_on: str,
        val_loss: float | None,
    ) -> None:
        idx = self._read_index()
        idx["versions"][version] = {
            "path": path, "kind": kind, "trained_on": trained_on, "val_loss": val_loss,
        }
        self._write_index(idx)

    def promote_community(self, version: str) -> None:
        """Mark a community version as the active production model."""
        idx = self._read_index()
        if version not in idx["versions"]:
            raise ValueError(f"Unknown version: {version}")
        idx["active_community"] = version
        self._write_index(idx)
        # Bust cache so next request reloads
        self._cache.clear()  # type: ignore[attr-defined]

    def rollback_community(self) -> None:
        """Drop the active community model — fall back to the synthetic base."""
        idx = self._read_index()
        idx["active_community"] = None
        self._write_index(idx)
        self._cache.clear()  # type: ignore[attr-defined]
