"""
Tests for `estimate_ftp_from_activities`. Uses an in-memory SQLite DB.
Verifies the estimator never returns implausibly low or stale values.
Now returns a dict: {estimated_ftp, confidence, method, best_ride_age_days, tsb_correction}
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import asyncio
import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.core.database import Base
from app.models.activity import Activity
from app.services.metrics_service import estimate_ftp_from_activities


@pytest_asyncio.fixture
async def db():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as s:
        yield s
    await engine.dispose()


USER_ID = uuid.uuid4().hex


def _act(user_id, days_ago, duration_s, np=None, ap=None):
    return Activity(
        id=uuid.uuid4().hex,
        user_id=user_id,
        date=datetime.now(timezone.utc) - timedelta(days=days_ago),
        duration_seconds=duration_s,
        normalized_power=np,
        avg_power=ap,
        source="test",
    )


@pytest.mark.asyncio
async def test_returns_none_with_no_activities(db):
    result = await estimate_ftp_from_activities(USER_ID, db)
    assert result["estimated_ftp"] is None
    assert result["method"] == "no_data"


@pytest.mark.asyncio
async def test_long_ride_np_used_directly(db):
    db.add(_act(USER_ID, days_ago=10, duration_s=3700, np=240))
    await db.flush()
    result = await estimate_ftp_from_activities(USER_ID, db)
    assert result["estimated_ftp"] == 240


@pytest.mark.asyncio
async def test_short_ride_np_discounted(db):
    """45–60 min NP gets ×0.97."""
    db.add(_act(USER_ID, days_ago=5, duration_s=3000, np=300))
    await db.flush()
    result = await estimate_ftp_from_activities(USER_ID, db)
    assert result["estimated_ftp"] == int(round(300 * 0.97))


@pytest.mark.asyncio
async def test_avg_power_alone_ignored(db):
    """avg_power without NP is not used (real rides under-report threshold via AP)."""
    db.add(_act(USER_ID, days_ago=5, duration_s=3700, ap=250))
    await db.flush()
    result = await estimate_ftp_from_activities(USER_ID, db)
    assert result["estimated_ftp"] is None


@pytest.mark.asyncio
async def test_short_ride_no_np_ignored(db):
    """A 50-min ride with no NP is too short for AP estimation."""
    db.add(_act(USER_ID, days_ago=5, duration_s=3000, ap=300))
    await db.flush()
    result = await estimate_ftp_from_activities(USER_ID, db)
    assert result["estimated_ftp"] is None


@pytest.mark.asyncio
async def test_old_rides_ignored(db):
    """Rides older than 3 years should not contribute to current FTP."""
    db.add(_act(USER_ID, days_ago=365 * 4, duration_s=3700, np=400))
    await db.flush()
    result = await estimate_ftp_from_activities(USER_ID, db)
    assert result["estimated_ftp"] is None


@pytest.mark.asyncio
async def test_implausibly_low_returns_none(db):
    """A 60W ride shouldn't be presented as the user's FTP."""
    db.add(_act(USER_ID, days_ago=5, duration_s=3700, np=60))
    await db.flush()
    result = await estimate_ftp_from_activities(USER_ID, db)
    assert result["estimated_ftp"] is None


@pytest.mark.asyncio
async def test_recent_ride_dominates_old_ride(db):
    """Recency weighting: a recent lower-power ride should pull the estimate
    noticeably below the old high-power ride (not just pick the max)."""
    db.add(_act(USER_ID, days_ago=5,   duration_s=3700, np=220))  # recent, lower
    await db.flush()
    db.add(_act(USER_ID, days_ago=500, duration_s=3700, np=290))  # old, higher
    await db.flush()
    result = await estimate_ftp_from_activities(USER_ID, db)
    ftp = result["estimated_ftp"]
    # Weighted avg must be well below the old max (290) and close to the recent (220)
    assert ftp is not None
    assert ftp < 270, f"Expected < 270 (recent ride should dominate), got {ftp}"
    assert ftp >= 220, f"Expected >= 220, got {ftp}"
