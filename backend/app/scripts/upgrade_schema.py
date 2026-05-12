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
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS model_version      VARCHAR(50)",    # Athlete profile nutrition context ──────────────────────────────────────────────
    "ALTER TABLE athlete_profiles ADD COLUMN IF NOT EXISTS diet                    VARCHAR(20)",
    "ALTER TABLE athlete_profiles ADD COLUMN IF NOT EXISTS climate                 VARCHAR(30)",
    "ALTER TABLE athlete_profiles ADD COLUMN IF NOT EXISTS event_type              VARCHAR(30)",
    "ALTER TABLE athlete_profiles ADD COLUMN IF NOT EXISTS recent_illness_count_3m INTEGER",    # Nutrition ───────────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS blood_tests (
        id                UUID PRIMARY KEY,
        user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        test_date         TIMESTAMPTZ NOT NULL,
        lab_name          VARCHAR(200),
        source            VARCHAR(20) NOT NULL DEFAULT 'pdf',
        raw_filename      VARCHAR(500),
        parser_version    VARCHAR(20),
        parser_confidence FLOAT,
        markers           JSON NOT NULL DEFAULT '{}',
        notes             TEXT,
        created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_blood_tests_user_id   ON blood_tests(user_id)",
    "CREATE INDEX IF NOT EXISTS ix_blood_tests_test_date ON blood_tests(test_date)",
    """
    CREATE TABLE IF NOT EXISTS blood_markers (
        id            UUID PRIMARY KEY,
        user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        blood_test_id UUID NOT NULL REFERENCES blood_tests(id) ON DELETE CASCADE,
        test_date     TIMESTAMPTZ NOT NULL,
        marker_key    VARCHAR(50) NOT NULL,
        value         FLOAT NOT NULL,
        unit          VARCHAR(30),
        ref_low       FLOAT,
        ref_high      FLOAT,
        status        VARCHAR(20) NOT NULL DEFAULT 'unknown'
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_blood_markers_user_id    ON blood_markers(user_id)",
    "CREATE INDEX IF NOT EXISTS ix_blood_markers_marker_key ON blood_markers(marker_key)",
    "CREATE INDEX IF NOT EXISTS ix_blood_markers_status     ON blood_markers(status)",
    """
    CREATE TABLE IF NOT EXISTS supplement_recommendations (
        id                     UUID PRIMARY KEY,
        user_id                UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        generated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        engine_version         VARCHAR(50) NOT NULL DEFAULT 'rule_v1',
        is_cold_start          BOOLEAN NOT NULL DEFAULT TRUE,
        based_on_blood_test_id UUID REFERENCES blood_tests(id) ON DELETE SET NULL,
        payload                JSON NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_supp_recs_user_id      ON supplement_recommendations(user_id)",
    "CREATE INDEX IF NOT EXISTS ix_supp_recs_generated_at ON supplement_recommendations(generated_at)",

    # Tracking: supplement intake + performance tests ─────────────────────────
    """
    CREATE TABLE IF NOT EXISTS supplement_intakes (
        id             UUID PRIMARY KEY,
        user_id        UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        supplement_key VARCHAR(80) NOT NULL,
        label          VARCHAR(200) NOT NULL,
        dose           FLOAT,
        dose_unit      VARCHAR(40),
        frequency      VARCHAR(80),
        timing         VARCHAR(120),
        started_at     TIMESTAMPTZ NOT NULL,
        stopped_at     TIMESTAMPTZ,
        adherence_pct  INTEGER,
        source         VARCHAR(20) NOT NULL DEFAULT 'manual',
        notes          TEXT,
        created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_supp_intakes_user_id        ON supplement_intakes(user_id)",
    "CREATE INDEX IF NOT EXISTS ix_supp_intakes_supplement_key ON supplement_intakes(supplement_key)",
    "CREATE INDEX IF NOT EXISTS ix_supp_intakes_started_at     ON supplement_intakes(started_at)",
    "CREATE INDEX IF NOT EXISTS ix_supp_intakes_stopped_at     ON supplement_intakes(stopped_at)",
    # Supplement dose logs (actual daily doses taken — feeds rule engine + future ML) ────────────
    """
    CREATE TABLE IF NOT EXISTS supplement_dose_logs (
        id             UUID PRIMARY KEY,
        user_id        UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        supplement_key VARCHAR(80) NOT NULL,
        label          VARCHAR(200) NOT NULL,
        dose_taken     FLOAT NOT NULL,
        dose_unit      VARCHAR(40),
        taken_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        notes          TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_dose_logs_user_id        ON supplement_dose_logs(user_id)",
    "CREATE INDEX IF NOT EXISTS ix_dose_logs_supplement_key ON supplement_dose_logs(supplement_key)",
    "CREATE INDEX IF NOT EXISTS ix_dose_logs_taken_at       ON supplement_dose_logs(taken_at)",

    """
    CREATE TABLE IF NOT EXISTS performance_tests (
        id         UUID PRIMARY KEY,
        user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        test_date  TIMESTAMPTZ NOT NULL,
        test_type  VARCHAR(40) NOT NULL,
        value      FLOAT NOT NULL,
        unit       VARCHAR(20) NOT NULL,
        source     VARCHAR(20) NOT NULL DEFAULT 'manual',
        notes      TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_perf_tests_user_id   ON performance_tests(user_id)",
    "CREATE INDEX IF NOT EXISTS ix_perf_tests_test_date ON performance_tests(test_date)",
    "CREATE INDEX IF NOT EXISTS ix_perf_tests_test_type ON performance_tests(test_type)",
]


async def main() -> None:
    async with engine.begin() as conn:
        for stmt in STATEMENTS:
            print(f"  → {stmt}")
            await conn.execute(text(stmt))
    print("Schema upgrade complete.")


if __name__ == "__main__":
    asyncio.run(main())
