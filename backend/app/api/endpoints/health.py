"""
Daily health & wellness API. Sources data from the HealthMetric table
(populated by Garmin sync) and computes a Readiness score.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.health import HealthMetric
from app.models.user import User
from app.services.readiness import compute_readiness

router = APIRouter()


class HealthDayOut(BaseModel):
    date: str
    sleep_total_seconds: Optional[int] = None
    sleep_score: Optional[int] = None
    hrv_overnight_avg_ms: Optional[float] = None
    hrv_7d_avg_ms: Optional[float] = None
    hrv_status: Optional[str] = None
    resting_hr: Optional[int] = None
    body_battery_high: Optional[int] = None
    body_battery_low: Optional[int] = None
    stress_avg: Optional[int] = None


class ReadinessOut(BaseModel):
    score: float
    status: str
    hrv_z: Optional[float] = None
    rhr_delta: Optional[float] = None
    sleep_score: Optional[int] = None
    body_battery: Optional[int] = None
    hrv_score: Optional[float] = None
    rhr_score: Optional[float] = None
    drivers: list[str]
    advice: str


class HealthRecentOut(BaseModel):
    days: list[HealthDayOut]
    readiness: ReadinessOut


def _to_out(m: HealthMetric) -> HealthDayOut:
    return HealthDayOut(
        date=m.date.isoformat(),
        sleep_total_seconds=m.sleep_total_seconds,
        sleep_score=m.sleep_score,
        hrv_overnight_avg_ms=m.hrv_overnight_avg_ms,
        hrv_7d_avg_ms=m.hrv_7d_avg_ms,
        hrv_status=m.hrv_status,
        resting_hr=m.resting_hr,
        body_battery_high=m.body_battery_high,
        body_battery_low=m.body_battery_low,
        stress_avg=m.stress_avg,
    )


@router.get("/recent", response_model=HealthRecentOut)
async def get_recent_health(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    result = await db.execute(
        select(HealthMetric)
        .where(
            HealthMetric.user_id == current_user.id,
            HealthMetric.date >= cutoff,
        )
        .order_by(HealthMetric.date)
    )
    metrics = list(result.scalars().all())
    snap = compute_readiness(metrics)
    return HealthRecentOut(
        days=[_to_out(m) for m in metrics],
        readiness=ReadinessOut(**snap.__dict__),
    )


@router.get("/readiness", response_model=ReadinessOut)
async def get_readiness(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    cutoff = datetime.now(timezone.utc) - timedelta(days=45)
    result = await db.execute(
        select(HealthMetric)
        .where(
            HealthMetric.user_id == current_user.id,
            HealthMetric.date >= cutoff,
        )
        .order_by(HealthMetric.date)
    )
    metrics = list(result.scalars().all())
    snap = compute_readiness(metrics)
    return ReadinessOut(**snap.__dict__)
