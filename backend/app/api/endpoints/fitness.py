from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta, timezone

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User
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
