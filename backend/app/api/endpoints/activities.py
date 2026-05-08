from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from typing import Optional
import os, uuid, pathlib

from app.core.database import get_db
from app.core.deps import get_current_user
from app.core.config import settings
from app.models.user import User
from app.models.activity import Activity
from app.services.file_parser import parse_activity_file
from app.services.metrics_service import compute_activity_metrics

router = APIRouter()

ALLOWED_EXTENSIONS = {".gpx", ".fit", ".tcx"}
MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB


class ActivityOut(BaseModel):
    id: str
    source: str
    name: str
    date: str
    duration_seconds: int
    distance_meters: float
    elevation_gain_meters: Optional[float]
    avg_power: Optional[float]
    normalized_power: Optional[float]
    tss: Optional[float]
    avg_hr: Optional[int]
    hr_drift: Optional[float]
    workout_type: Optional[str]
    temperature_c: Optional[float]
    model_config = {"from_attributes": True}


class PaginatedActivities(BaseModel):
    items: list[ActivityOut]
    total: int
    page: int
    size: int
    pages: int


class UploadResult(BaseModel):
    activity_id: Optional[str]
    status: str   # success | duplicate | error
    message: str


@router.get("", response_model=PaginatedActivities)
async def list_activities(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * size
    total_result = await db.execute(
        select(func.count()).select_from(Activity).where(Activity.user_id == current_user.id)
    )
    total = total_result.scalar_one()

    result = await db.execute(
        select(Activity)
        .where(Activity.user_id == current_user.id)
        .order_by(Activity.date.desc())
        .offset(offset)
        .limit(size)
    )
    activities = result.scalars().all()

    return PaginatedActivities(
        items=[_activity_out(a) for a in activities],
        total=total,
        page=page,
        size=size,
        pages=max(1, -(-total // size)),
    )


@router.post("/upload", response_model=UploadResult)
async def upload_activity(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Use .gpx, .fit, or .tcx")

    content = await file.read()
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(413, "File too large (max 50 MB)")

    # Save to disk
    upload_dir = pathlib.Path(settings.UPLOAD_DIR) / current_user.id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / f"{uuid.uuid4()}{ext}"
    file_path.write_bytes(content)

    try:
        parsed = parse_activity_file(str(file_path), ext)
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(422, f"Could not parse file: {exc}")

    # Check duplicate by external_id or date+duration
    if parsed.get("external_id"):
        dup = await db.execute(
            select(Activity).where(
                Activity.user_id == current_user.id,
                Activity.external_id == parsed["external_id"],
            )
        )
        if dup.scalar_one_or_none():
            return UploadResult(activity_id=None, status="duplicate", message="Activity already exists")

    activity = Activity(user_id=current_user.id, source=ext.lstrip("."), **parsed)
    compute_activity_metrics(activity, current_user)
    db.add(activity)
    await db.flush()

    return UploadResult(activity_id=activity.id, status="success", message="Uploaded successfully")


@router.get("/{activity_id}", response_model=ActivityOut)
async def get_activity(
    activity_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Activity).where(
            Activity.id == activity_id,
            Activity.user_id == current_user.id,
        )
    )
    activity = result.scalar_one_or_none()
    if not activity:
        raise HTTPException(404, "Activity not found")
    return _activity_out(activity)


@router.delete("/{activity_id}", status_code=204)
async def delete_activity(
    activity_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Activity).where(
            Activity.id == activity_id,
            Activity.user_id == current_user.id,
        )
    )
    activity = result.scalar_one_or_none()
    if not activity:
        raise HTTPException(404, "Activity not found")
    await db.delete(activity)


def _activity_out(a: Activity) -> ActivityOut:
    return ActivityOut(
        id=a.id,
        source=a.source,
        name=a.name,
        date=a.date.isoformat(),
        duration_seconds=a.duration_seconds,
        distance_meters=a.distance_meters,
        elevation_gain_meters=a.elevation_gain_meters,
        avg_power=a.avg_power,
        normalized_power=a.normalized_power,
        tss=a.tss,
        avg_hr=a.avg_hr,
        hr_drift=a.hr_drift,
        workout_type=a.workout_type,
        temperature_c=a.temperature_c,
    )
