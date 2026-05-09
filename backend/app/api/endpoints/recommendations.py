from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel
from typing import Any

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User
from app.models.recommendation import Recommendation
from app.ml.inference import generate_recommendation, generate_multi_horizon_recommendation

router = APIRouter()


class RecommendationOut(BaseModel):
    id: str
    generated_at: str
    confidence: float
    model_version: str
    is_cold_start: bool
    next_workout: Any
    weekly_plan: list[Any]
    insights: list[Any]
    forecast: Any
    risks: list[Any]


@router.get("/recommendation", response_model=RecommendationOut)
async def get_recommendation(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Return latest cached recommendation if fresh (< 6 hours)
    result = await db.execute(
        select(Recommendation)
        .where(Recommendation.user_id == current_user.id)
        .order_by(desc(Recommendation.generated_at))
        .limit(1)
    )
    rec = result.scalar_one_or_none()

    from datetime import datetime, timedelta, timezone
    if rec:
        age = datetime.now(timezone.utc) - rec.generated_at.replace(tzinfo=timezone.utc)
        if age < timedelta(hours=6):
            return _rec_out(rec)

    # Generate fresh recommendation
    rec = await generate_recommendation(current_user, db)
    db.add(rec)
    await db.flush()
    return _rec_out(rec)


@router.post("/recommendation/refresh", response_model=RecommendationOut)
async def refresh_recommendation(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rec = await generate_recommendation(current_user, db)
    db.add(rec)
    await db.flush()
    return _rec_out(rec)


# ── Multi-horizon: 3 alternative recommendations side-by-side ────────────────
class MultiHorizonOut(BaseModel):
    """
    Three alternative "next workout" suggestions, one per planning horizon:
      • short  — best for 7-day FTP gain
      • medium — best for 28-day build
      • event  — best path to peak on the user's goal-event date
    Each horizon's payload mirrors the standard RecommendationOut shape.
    Also includes a rule-based supplement stack derived from training load
    and (if available) the user's latest blood test.
    """
    is_cold_start:  bool
    model_version:  str
    active_horizon: str
    horizons:       dict[str, Any]
    supplements:    dict[str, Any] | None = None


@router.get("/recommendation/multi-horizon", response_model=MultiHorizonOut)
async def get_multi_horizon_recommendation(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns 3 alternative recommendations side-by-side so the user can pick
    based on their current priority (immediate gain vs long-term peak).
    Always fresh — does not use the 6h cache.
    """
    return await generate_multi_horizon_recommendation(current_user, db)


@router.get("/analyze/{activity_id}")
async def analyze_activity(
    activity_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from app.models.activity import Activity
    result = await db.execute(
        select(Activity).where(
            Activity.id == activity_id,
            Activity.user_id == current_user.id,
        )
    )
    activity = result.scalar_one_or_none()
    if not activity:
        from fastapi import HTTPException
        raise HTTPException(404, "Activity not found")

    from app.ml.inference import analyze_single_activity
    analysis = await analyze_single_activity(activity, current_user, db)
    return analysis


def _rec_out(rec: Recommendation) -> RecommendationOut:
    p = rec.payload
    return RecommendationOut(
        id=rec.id,
        generated_at=rec.generated_at.isoformat(),
        confidence=rec.confidence,
        model_version=rec.model_version,
        is_cold_start=rec.is_cold_start,
        next_workout=p.get("next_workout"),
        weekly_plan=p.get("weekly_plan", []),
        insights=p.get("insights", []),
        forecast=p.get("forecast"),
        risks=p.get("risks", []),
    )
