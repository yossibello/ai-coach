"""
Idempotent schema upgrade for Phase-1 ML hooks.

Adds:
  activities.quality_score, .quality_reasons, .review_status, .is_outlier
  users.allow_for_training, .model_version

Run:
    docker compose exec backend python -m app.scripts.upgrade_schema

Safe to run multiple times — each ALTER uses IF NOT EXISTS.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.core.database import engine


STATEMENTS = [
    # Activities ──────────────────────────────────────────────────────────────
    "ALTER TABLE activities ADD COLUMN IF NOT EXISTS quality_score   VARCHAR(20)",
    "ALTER TABLE activities ADD COLUMN IF NOT EXISTS quality_reasons JSON",
    "ALTER TABLE activities ADD COLUMN IF NOT EXISTS review_status   VARCHAR(20) NOT NULL DEFAULT 'confirmed'",
    "ALTER TABLE activities ADD COLUMN IF NOT EXISTS is_outlier      BOOLEAN     NOT NULL DEFAULT FALSE",
    "CREATE INDEX IF NOT EXISTS ix_activities_quality_score ON activities(quality_score)",
    "CREATE INDEX IF NOT EXISTS ix_activities_review_status ON activities(review_status)",
    # Users ───────────────────────────────────────────────────────────────────
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS allow_for_training BOOLEAN     NOT NULL DEFAULT TRUE",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS model_version      VARCHAR(50)",
]


async def main() -> None:
    async with engine.begin() as conn:
        for stmt in STATEMENTS:
            print(f"  → {stmt}")
            await conn.execute(text(stmt))
    print("Schema upgrade complete.")


if __name__ == "__main__":
    asyncio.run(main())
