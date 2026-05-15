from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User, AthleteProfile

router = APIRouter()


class ProfileOut(BaseModel):
    id: str
    user_id: str
    age: Optional[int]
    weight_kg: Optional[float]
    height_cm: Optional[float]
    sex: Optional[str]
    ftp: Optional[int]
    max_hr: Optional[int]
    resting_hr: Optional[int]
    vo2max_estimate: Optional[float]
    cycling_experience_years: Optional[int]
    primary_goal: Optional[str]
    goal_event_date: Optional[str]
    goal_event_name: Optional[str]
    training_days_per_week: Optional[int]
    diet: Optional[str] = None
    climate: Optional[str] = None
    event_type: Optional[str] = None
    recent_illness_count_3m: Optional[int] = None
    strength_approach: Optional[str] = None


class ProfileUpdate(BaseModel):
    age: Optional[int] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    sex: Optional[str] = None
    ftp: Optional[int] = None
    max_hr: Optional[int] = None
    resting_hr: Optional[int] = None
    vo2max_estimate: Optional[float] = None
    cycling_experience_years: Optional[int] = None
    primary_goal: Optional[str] = None
    goal_event_date: Optional[str] = None
    goal_event_name: Optional[str] = None
    training_days_per_week: Optional[int] = None
    diet: Optional[str] = None
    climate: Optional[str] = None
    event_type: Optional[str] = None
    recent_illness_count_3m: Optional[int] = None
    strength_approach: Optional[str] = None


@router.get("", response_model=ProfileOut)
async def get_profile(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AthleteProfile).where(AthleteProfile.user_id == current_user.id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        # Create empty profile
        profile = AthleteProfile(user_id=current_user.id)
        db.add(profile)
        await db.flush()

    return _profile_out(profile)


@router.put("", response_model=ProfileOut)
async def update_profile(
    body: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AthleteProfile).where(AthleteProfile.user_id == current_user.id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        profile = AthleteProfile(user_id=current_user.id)
        db.add(profile)

    fields = body.model_dump(exclude_none=True)
    if "ftp" in fields:
        profile.ftp_source = "manual"
    for field, value in fields.items():
        if field == "goal_event_date" and value:
            setattr(profile, field, datetime.fromisoformat(value))
        else:
            setattr(profile, field, value)

    await db.flush()
    return _profile_out(profile)


def _profile_out(p: AthleteProfile) -> ProfileOut:
    return ProfileOut(
        id=p.id,
        user_id=p.user_id,
        age=p.age,
        weight_kg=p.weight_kg,
        height_cm=p.height_cm,
        sex=p.sex,
        ftp=p.ftp,
        max_hr=p.max_hr,
        resting_hr=p.resting_hr,
        vo2max_estimate=p.vo2max_estimate,
        cycling_experience_years=p.cycling_experience_years,
        primary_goal=p.primary_goal,
        goal_event_date=p.goal_event_date.isoformat() if p.goal_event_date else None,
        goal_event_name=p.goal_event_name,
        training_days_per_week=p.training_days_per_week,
        diet=p.diet,
        climate=p.climate,
        event_type=p.event_type,
        recent_illness_count_3m=p.recent_illness_count_3m,
        strength_approach=p.strength_approach or "friel",
    )
