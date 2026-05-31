from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta, timezone

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User, AthleteProfile
from app.models.recommendation import FitnessMetric
from app.services.metrics_service import compute_pmc_for_user

router = APIRouter()


class FitnessSnapshotOut(BaseModel):
    date: str
    ctl: float
    atl: float
    tsb: float
    tss: float
    ftp: float
    ftp_method: Optional[str] = None
    ftp_meta: Optional[dict] = None


class FitnessProgressionOut(BaseModel):
    current: Optional[FitnessSnapshotOut]
    history: list[FitnessSnapshotOut]
    ftp_history: list[dict]
    personal_records: list[dict]


@router.get("/progression", response_model=FitnessProgressionOut)
async def get_fitness_progression(
    weeks: int = Query(16, ge=4, le=104),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    cutoff = datetime.now(timezone.utc) - timedelta(weeks=weeks)
    result = await db.execute(
        select(FitnessMetric)
        .where(
            FitnessMetric.user_id == current_user.id,
            FitnessMetric.date >= cutoff,
        )
        .order_by(FitnessMetric.date)
    )
    metrics = result.scalars().all()

    history = [
        FitnessSnapshotOut(
            date=m.date.isoformat(),
            ctl=round(m.ctl, 1),
            atl=round(m.atl, 1),
            tsb=round(m.tsb, 1),
            tss=round(m.tss, 1),
            ftp=round(m.ftp, 1),
        )
        for m in metrics
    ]

    current_snapshot = history[-1] if history else None

    # Attach the stored ftp_method from the athlete profile to the current snapshot
    if current_snapshot is not None:
        profile_result = await db.execute(
            select(AthleteProfile).where(AthleteProfile.user_id == current_user.id)
        )
        profile = profile_result.scalar_one_or_none()
        if profile and (profile.ftp_method or profile.ftp_meta):
            current_snapshot = current_snapshot.model_copy(update={
                "ftp_method": profile.ftp_method,
                "ftp_meta": profile.ftp_meta,
            })

    # FTP history (unique FTP values over time)
    ftp_history = []
    last_ftp = None
    for snap in history:
        if snap.ftp != last_ftp:
            ftp_history.append({"date": snap.date, "ftp": snap.ftp})
            last_ftp = snap.ftp

    return FitnessProgressionOut(
        current=current_snapshot,
        history=history,
        ftp_history=ftp_history,
        personal_records=[],  # Populated by dedicated endpoint in future
    )


@router.post("/recalculate")
async def recalculate_fitness(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await compute_pmc_for_user(current_user.id, db)
    return {"status": "recalculated"}


@router.get("/capabilities")
async def get_capabilities(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return event capability tiers for the current athlete."""
    from app.ml.capabilities import evaluate
    from sqlalchemy import desc

    # Load profile for FTP and weight
    profile_res = await db.execute(
        select(AthleteProfile).where(AthleteProfile.user_id == current_user.id)
    )
    profile = profile_res.scalar_one_or_none()

    ftp_w      = (profile.ftp if profile and profile.ftp else 200) or 200
    weight_kg  = (profile.weight_kg if profile and getattr(profile, "weight_kg", None) else 70) or 70
    pc_5min    = getattr(profile, "pc5min_capacity_wkg", None)
    pc_1min    = getattr(profile, "pc1min_capacity_wkg", None)

    # Load latest CTL
    metric_res = await db.execute(
        select(FitnessMetric)
        .where(FitnessMetric.user_id == current_user.id)
        .order_by(desc(FitnessMetric.date)).limit(1)
    )
    metric = metric_res.scalar_one_or_none()
    ctl = float(metric.ctl) if metric else 0.0

    return {
        "athlete": {
            "ftp_w":     ftp_w,
            "weight_kg": weight_kg,
            "wkg":       round(ftp_w / max(weight_kg, 1), 2),
            "ctl":       round(ctl, 1),
        },
        "events": evaluate(ftp_w, weight_kg, ctl, pc_5min_wkg=pc_5min, pc_1min_wkg=pc_1min),
    }
