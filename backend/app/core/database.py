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
                "garmin_username":      "VARCHAR(200)",
                "garmin_password_enc":  "VARCHAR",
                "garmin_user_id":       "VARCHAR",
            })


async def _ensure_columns(conn, table: str, columns: dict[str, str]) -> None:
    """Add any missing columns to *table*.  No-op if column already exists."""
    from sqlalchemy import text
    res = await conn.execute(text(f"PRAGMA table_info({table})"))
    existing = {row[1] for row in res.fetchall()}
    for col, ddl in columns.items():
        if col not in existing:
            await conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}"))


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
