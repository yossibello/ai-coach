from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

# SQLite uses NullPool and doesn't accept pool_size/max_overflow.
# Apply pooling args only for true client/server DBs (Postgres / MySQL).
_engine_kwargs: dict = {
    "echo": settings.ENVIRONMENT == "development",
    "pool_pre_ping": True,
}
if settings.DATABASE_URL.startswith("sqlite"):
    # SQLite: use WAL mode (allows concurrent reads while writing) and a
    # generous busy timeout so parallel background tasks queue up instead
    # of immediately failing with "database is locked".
    _engine_kwargs["connect_args"] = {"timeout": 30, "check_same_thread": False}
else:
    _engine_kwargs.update(pool_size=10, max_overflow=20)

engine = create_async_engine(settings.DATABASE_URL, **_engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def init_db():
    """Create all tables on startup (dev convenience).

    For SQLite dev DBs this also runs a tiny idempotent column-add pass
    so newly-introduced fields don't require manual migrations.
    """
    async with engine.begin() as conn:
        # Import all models so Base knows about them
        from app.models import user, activity, recommendation, health, outcome  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)

        # Enable WAL mode for SQLite so concurrent background tasks (Strava +
        # Garmin syncs running in parallel) can read freely and queue writes
        # rather than immediately failing with "database is locked".
        if settings.DATABASE_URL.startswith("sqlite"):
            from sqlalchemy import text
            await conn.execute(text("PRAGMA journal_mode=WAL"))
            await conn.execute(text("PRAGMA busy_timeout=30000"))
            await _ensure_columns(conn, "users", {
                "garmin_username":           "VARCHAR(200)",
                "garmin_password_enc":       "VARCHAR",
                "garmin_user_id":            "VARCHAR",
                "garmin_token_store":        "VARCHAR",
                "oura_access_token_enc":     "VARCHAR",
                "fitbit_access_token_enc":   "VARCHAR",
                "fitbit_refresh_token_enc":  "VARCHAR",
                "fitbit_user_id":            "VARCHAR",
                "fitbit_token_expires_at":   "DATETIME",
            })
            await _ensure_columns(conn, "activities", {
                "pc_5s_wkg":    "REAL",
                "pc_1min_wkg":  "REAL",
                "pc_5min_wkg":  "REAL",
                "pc_20min_wkg": "REAL",
                "device_watts": "BOOLEAN",
            })
            await _ensure_columns(conn, "prediction_outcomes", {
                "avg_hrv_recovery": "REAL",
                "outcome_weight":   "REAL",
            })
            await _ensure_columns(conn, "athlete_profiles", {
                "strength_approach": "VARCHAR(30)",
            })
        else:
            # Postgres: idempotently add any new columns not created by CREATE TABLE
            await _ensure_columns_pg(conn, "activities", {
                "device_watts": "BOOLEAN",
            })
            await _ensure_columns_pg(conn, "prediction_outcomes", {
                "avg_hrv_recovery": "REAL",
                "outcome_weight":   "REAL",
            })
            await _ensure_columns_pg(conn, "athlete_profiles", {
                "strength_approach": "VARCHAR(30)",
            })
            await _ensure_columns_pg(conn, "users", {
                "garmin_token_store": "TEXT",
                "oura_access_token_enc": "TEXT",
                "fitbit_access_token_enc": "TEXT",
                "fitbit_refresh_token_enc": "TEXT",
                "fitbit_user_id": "TEXT",
                "fitbit_token_expires_at": "TIMESTAMPTZ",
            })


async def _ensure_columns(conn, table: str, columns: dict[str, str]) -> None:
    """Add any missing columns to *table* (SQLite).  No-op if column already exists."""
    from sqlalchemy import text
    res = await conn.execute(text(f"PRAGMA table_info({table})"))
    existing = {row[1] for row in res.fetchall()}
    for col, ddl in columns.items():
        if col not in existing:
            await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"))


async def _ensure_columns_pg(conn, table: str, columns: dict[str, str]) -> None:
    """Add any missing columns to *table* (Postgres).  No-op if column already exists."""
    from sqlalchemy import text
    for col, ddl in columns.items():
        res = await conn.execute(text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :t AND column_name = :c"
        ), {"t": table, "c": col})
        if not res.scalar():
            await conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{col}" {ddl}'))


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
