"""
Daily health & wellness API. Sources data from the HealthMetric table
(populated by Garmin sync or manual entry) and computes a Readiness score.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.activity import Activity
from app.models.health import HealthMetric
from app.models.user import User
from app.services.hr_drift import get_drift_assessment
from app.services.readiness import compute_readiness

router = APIRouter()


class HealthDayOut(BaseModel):
    date: str
    source: str = "garmin"
    sleep_total_seconds: Optional[int] = None
    sleep_score: Optional[int] = None
    hrv_overnight_avg_ms: Optional[float] = None
    hrv_7d_avg_ms: Optional[float] = None
    hrv_status: Optional[str] = None
    resting_hr: Optional[int] = None
    body_battery_high: Optional[int] = None
    body_battery_low: Optional[int] = None
    stress_avg: Optional[int] = None


class DriftOut(BaseModel):
    state: str           # stable / decoupled / stressed / unknown
    drift_pct: Optional[float] = None
    trend: str           # improving / worsening / stable / unknown
    overtraining_risk: bool
    action: str
    note: str


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
    drift: Optional[DriftOut] = None
    has_manual_today: bool = False


class ManualHealthLog(BaseModel):
    hrv_ms: Optional[float] = Field(None, ge=10, le=250, description="HRV overnight avg in ms")
    resting_hr: Optional[int] = Field(None, ge=25, le=130)
    sleep_hours: Optional[float] = Field(None, ge=0, le=24)
    sleep_score: Optional[int] = Field(None, ge=0, le=100)
    energy_level: Optional[int] = Field(None, ge=0, le=100, description="0=exhausted 100=fully charged")
    stress_level: Optional[int] = Field(None, ge=0, le=100)


def _to_out(m: HealthMetric) -> HealthDayOut:
    return HealthDayOut(
        date=m.date.isoformat(),
        source=m.source,
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


async def _get_drift(user_id: str, db: AsyncSession) -> DriftOut:
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    r = await db.execute(
        select(Activity)
        .where(Activity.user_id == user_id, Activity.date >= cutoff)
        .order_by(Activity.date.desc())
        .limit(10)
    )
    acts = list(r.scalars().all())
    d = get_drift_assessment(acts)
    return DriftOut(
        state=d.state,
        drift_pct=d.drift_pct,
        trend=d.trend,
        overtraining_risk=d.overtraining_risk,
        action=d.action,
        note=d.note,
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
    drift = await _get_drift(current_user.id, db)
    snap = compute_readiness(
        metrics,
        drift_state=drift.state if drift.state != "unknown" else None,
        drift_pct=drift.drift_pct,
    )

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    has_manual = any(
        m.source == "manual" and m.date >= today_start for m in metrics
    )

    return HealthRecentOut(
        days=[_to_out(m) for m in metrics],
        readiness=ReadinessOut(**snap.__dict__),
        drift=drift,
        has_manual_today=has_manual,
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
    drift = await _get_drift(current_user.id, db)
    snap = compute_readiness(
        metrics,
        drift_state=drift.state if drift.state != "unknown" else None,
        drift_pct=drift.drift_pct,
    )
    return ReadinessOut(**snap.__dict__)


@router.post("/log", response_model=HealthDayOut)
async def log_health_manual(
    body: ManualHealthLog,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upsert today's health metrics from manual entry.
    If Garmin already populated a field today, manual entry fills only NULL fields.
    If the row is already manual, values are overwritten.
    """
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)

    r = await db.execute(
        select(HealthMetric).where(
            HealthMetric.user_id == current_user.id,
            HealthMetric.date >= today_start,
            HealthMetric.date < today_end,
        )
    )
    existing = r.scalar_one_or_none()
    is_manual = existing is None or existing.source == "manual"

    if existing is None:
        existing = HealthMetric(
            user_id=current_user.id,
            date=datetime.now(timezone.utc),
            source="manual",
        )
        db.add(existing)

    def _set(attr: str, value):
        if value is None:
            return
        if is_manual or getattr(existing, attr) is None:
            setattr(existing, attr, value)

    _set("hrv_overnight_avg_ms", body.hrv_ms)
    _set("resting_hr", body.resting_hr)
    if body.sleep_hours is not None:
        _set("sleep_total_seconds", int(body.sleep_hours * 3600))
    _set("sleep_score", body.sleep_score)
    _set("body_battery_high", body.energy_level)
    _set("stress_avg", body.stress_level)

    await db.flush()
    return _to_out(existing)
