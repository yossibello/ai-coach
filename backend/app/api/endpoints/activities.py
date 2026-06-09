from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from typing import Optional
import os, uuid, pathlib
from datetime import timedelta

from app.core.database import get_db
from app.core.deps import get_current_user
from app.core.config import settings
from app.models.user import User
from app.models.activity import Activity
from app.models.tracking import PerformanceTest
from app.services.file_parser import parse_activity_file
from app.services.metrics_service import compute_activity_metrics, _score_and_tag, detect_ftp_test
from app.services.recommendation_linking import latest_recommendation_id

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
    avg_hr: Optional[float]
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
        select(func.count()).select_from(Activity).where(
            Activity.user_id == current_user.id,
            Activity.review_status != "deleted",
        )
    )
    total = total_result.scalar_one()

    result = await db.execute(
        select(Activity)
        .where(
            Activity.user_id == current_user.id,
            Activity.review_status != "deleted",
        )
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

    parsed.setdefault("source", ext.lstrip("."))

    # ── Try to find an existing activity this file matches ──────────────────
    # 1. Exact match by external_id (rare for GPX, but possible for FIT exports)
    existing: Activity | None = None
    if parsed.get("external_id"):
        r = await db.execute(
            select(Activity).where(
                Activity.user_id == current_user.id,
                Activity.external_id == parsed["external_id"],
            )
        )
        existing = r.scalar_one_or_none()

    # 2. Fuzzy match — two passes to handle timezone-naive GPX files.
    #    Pass A: tight ±5 min window (exact match for well-formed files).
    #    Pass B: same UTC calendar day + duration within 10% (catches GPX files
    #            that store local time as if it were UTC, e.g. MyWhoosh exports).
    if existing is None and parsed.get("date"):
        dur = parsed.get("duration_seconds", 0)

        for window_hours in (0, 12):   # pass A: 5 min, pass B: whole day
            if window_hours == 0:
                delta = timedelta(minutes=5)
                start = parsed["date"] - delta
                end   = parsed["date"] + delta
            else:
                # Search the entire UTC calendar day of the parsed date
                day_start = parsed["date"].replace(hour=0, minute=0, second=0, microsecond=0)
                start = day_start - timedelta(hours=12)   # cover ±12 h for any TZ offset
                end   = day_start + timedelta(hours=36)

            r = await db.execute(
                select(Activity).where(
                    Activity.user_id == current_user.id,
                    Activity.date >= start,
                    Activity.date <= end,
                    Activity.review_status != "deleted",
                )
            )
            candidates = r.scalars().all()
            tol = 0.15 if window_hours == 0 else 0.10   # tighter on day-level match
            best = None
            best_dur_diff = float("inf")
            for c in candidates:
                if dur > 0 and c.duration_seconds > 0:
                    ratio = dur / c.duration_seconds
                    if (1 - tol) <= ratio <= (1 + tol):
                        diff = abs(dur - c.duration_seconds)
                        if diff < best_dur_diff:
                            best = c
                            best_dur_diff = diff
            if best:
                existing = best
                break

    # ── Merge into existing activity ─────────────────────────────────────────
    if existing is not None:
        # Fields that the file parser can provide but Strava/Garmin API often cannot.
        # Always overwrite hr_drift (the main reason to upload a GPX after the fact).
        # For other fields, only fill in if the existing value is NULL.
        always_overwrite = {"hr_drift", "avg_cadence", "temperature_c"}
        fill_if_null = {
            "avg_power", "max_power", "normalized_power",
            "avg_hr", "max_hr", "elevation_gain_meters",
        }
        enriched_fields: list[str] = []
        for field in always_overwrite | fill_if_null:
            new_val = parsed.get(field)
            if new_val is None:
                continue
            if field in always_overwrite or getattr(existing, field) is None:
                setattr(existing, field, new_val)
                enriched_fields.append(field)

        # Re-derive computed metrics from the new raw values
        compute_activity_metrics(existing, current_user)

        await db.flush()
        fields_str = ", ".join(enriched_fields) if enriched_fields else "nothing new"
        return UploadResult(
            activity_id=existing.id,
            status="enriched",
            message=f"Matched existing activity — enriched: {fields_str}",
        )

    # ── New activity ──────────────────────────────────────────────────────────
    activity = Activity(user_id=current_user.id, **parsed)
    compute_activity_metrics(activity, current_user)
    _score_and_tag(activity, current_user)
    activity.recommendation_id = await latest_recommendation_id(
        current_user.id, activity.date, db
    )
    db.add(activity)
    await db.flush()

    # ── Auto-detect FTP test and log to PerformanceTest ───────────────────────
    msg = "Uploaded successfully"
    test_type = detect_ftp_test(activity.name, activity.workout_type)
    if test_type and activity.normalized_power:
        profile = getattr(current_user, "profile", None)
        weight_kg = getattr(profile, "weight_kg", None)
        if activity.pc_20min_wkg and weight_kg:
            ftp_value = round(activity.pc_20min_wkg * weight_kg * 0.95, 1)
        else:
            ftp_value = round(float(activity.normalized_power) * 0.95, 1)
        pt = PerformanceTest(
            user_id=current_user.id,
            test_date=activity.date,
            test_type=test_type,
            value=ftp_value,
            unit="W",
            source="auto_detected",
            notes=f"Auto-detected from: {activity.name}",
        )
        db.add(pt)
        await db.flush()
        msg = f"FTP test detected — estimated FTP: {int(round(ftp_value))} W"

    return UploadResult(activity_id=activity.id, status="success", message=msg)


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
