from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
import uuid
import traceback
import logging

from app.core.database import get_db, AsyncSessionLocal
from app.core.deps import get_current_user
from app.core.config import settings
from app.models.user import User
from app.services.strava_service import (
    get_strava_auth_url,
    exchange_strava_code,
    sync_strava_history,
)

log = logging.getLogger(__name__)

router = APIRouter()

# In-memory task status store (use Redis in production)
_sync_tasks: dict[str, dict] = {}


class AuthURLResponse(BaseModel):
    url: str


class SyncResponse(BaseModel):
    task_id: str


class SyncStatus(BaseModel):
    status: str
    progress: int
    total: int


@router.get("/status")
async def strava_status(
    current_user: User = Depends(get_current_user),
):
    return {
        "connected": current_user.strava_connected,
        "athlete_id": current_user.strava_athlete_id,
    }


@router.get("/auth-url", response_model=AuthURLResponse)
async def get_auth_url():
    url = get_strava_auth_url()
    return AuthURLResponse(url=url)


@router.post("/exchange")
async def exchange_code(
    body: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    code = body.get("code")
    if not code:
        raise HTTPException(400, "Missing code")
    try:
        await exchange_strava_code(current_user, code, db)
    except Exception as exc:
        raise HTTPException(400, f"Strava auth failed: {exc}")
    return {"status": "connected"}


@router.post("/sync-history", response_model=SyncResponse)
async def sync_history(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not current_user.strava_connected:
        raise HTTPException(400, "Strava not connected")

    task_id = str(uuid.uuid4())
    _sync_tasks[task_id] = {"status": "running", "progress": 0, "total": 0}
    user_id = current_user.id  # capture scalar — don't hold session ref

    def update_progress(progress: int, total: int):
        _sync_tasks[task_id].update({"progress": progress, "total": total})

    async def run():
        """Background task: opens its own DB session so the request session
        lifecycle doesn't affect us."""
        try:
            async with AsyncSessionLocal() as bg_db:
                from sqlalchemy import select as sa_select
                from sqlalchemy.orm import selectinload
                result = await bg_db.execute(
                    sa_select(User).where(User.id == user_id).options(
                        selectinload(User.profile)
                    )
                )
                bg_user = result.scalar_one()
                await sync_strava_history(bg_user, bg_db, update_progress)
                await bg_db.commit()
            _sync_tasks[task_id]["status"] = "completed"
        except Exception:
            err = traceback.format_exc()
            log.error("Strava sync failed for user %s:\n%s", user_id, err)
            print(f"[SYNC ERROR] user={user_id}\n{err}", flush=True)
            _sync_tasks[task_id]["status"] = "failed"

    background_tasks.add_task(run)
    return SyncResponse(task_id=task_id)


@router.get("/sync-status/{task_id}", response_model=SyncStatus)
async def sync_status(
    task_id: str,
    current_user: User = Depends(get_current_user),
):
    task = _sync_tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return SyncStatus(**task)


@router.delete("/disconnect")
async def disconnect_strava(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    current_user.strava_athlete_id = None
    current_user.strava_access_token = None
    current_user.strava_refresh_token = None
    current_user.strava_token_expires_at = None
    db.add(current_user)
    return {"status": "disconnected"}


class FTPEstimateResponse(BaseModel):
    estimated_ftp: int | None
    previous_ftp: int | None
    updated: bool
    message: str
    confidence: float = 0.0
    confidence_low: int | None = None
    confidence_high: int | None = None
    method: str = "unknown"
    best_ride_age_days: int | None = None
    tsb_correction: float = 1.0
    sample_count: int = 0
    trend: str = "stable"


@router.post("/estimate-ftp", response_model=FTPEstimateResponse)
async def estimate_ftp(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Estimate FTP from power rides (recency-weighted, fatigue-corrected) and update the athlete profile."""
    from app.services.metrics_service import estimate_ftp_from_activities, compute_pmc_for_user
    from app.models.user import AthleteProfile

    # Fetch current profile to pass manual FTP as fallback
    result = await db.execute(
        select(AthleteProfile).where(AthleteProfile.user_id == current_user.id)
    )
    profile = result.scalar_one_or_none()
    prev = profile.ftp if profile else None

    ftp_result = await estimate_ftp_from_activities(
        current_user.id, db, manual_ftp=prev
    )
    estimated = ftp_result["estimated_ftp"]

    if estimated is None:
        return FTPEstimateResponse(
            estimated_ftp=None,
            previous_ftp=prev,
            updated=False,
            confidence=0.0,
            method="no_data",
            message="Not enough long power rides to estimate FTP. Do a 45+ min hard effort.",
        )

    if profile is None:
        profile = AthleteProfile(user_id=current_user.id, ftp=estimated)
        db.add(profile)
    else:
        profile.ftp = estimated
        db.add(profile)

    # Rebuild PMC with the new FTP so TSB/CTL reflect the updated value
    await db.flush()
    await compute_pmc_for_user(current_user.id, db)

    method = ftp_result.get("method", "power_weighted")
    confidence = ftp_result.get("confidence", 0.0)
    age_days = ftp_result.get("best_ride_age_days")
    tsb_corr = ftp_result.get("tsb_correction", 1.0)
    conf_low = ftp_result.get("confidence_low")
    conf_high = ftp_result.get("confidence_high")
    sample_count = ftp_result.get("sample_count", 0)
    trend = ftp_result.get("trend", "stable")

    method_labels = {
        "power_weighted": "from recent power data",
        "blended": "blended with manual FTP (low training load)",
        "manual_fallback": "from manual entry (power data too old)",
    }
    method_note = method_labels.get(method, method)
    band = f", range {conf_low}-{conf_high}W" if conf_low and conf_high else ""
    trend_note = "" if trend == "stable" else f", trend: {trend}"

    if prev != estimated:
        msg = (f"FTP updated from {prev}W → {estimated}W "
               f"({method_note}{band}, {sample_count} rides, "
               f"confidence {int(confidence*100)}%{trend_note})")
    else:
        msg = (f"FTP confirmed at {estimated}W "
               f"({method_note}{band}, {sample_count} rides{trend_note})")

    return FTPEstimateResponse(
        estimated_ftp=estimated,
        previous_ftp=prev,
        updated=(prev != estimated),
        confidence=confidence,
        confidence_low=conf_low,
        confidence_high=conf_high,
        method=method,
        best_ride_age_days=age_days,
        tsb_correction=tsb_corr,
        sample_count=sample_count,
        trend=trend,
        message=msg,
    )


@router.post("/rebuild-pmc")
async def rebuild_pmc(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Rebuild CTL/ATL/TSB from all stored activities (no Strava API call needed)."""
    from app.services.metrics_service import compute_pmc_for_user
    await compute_pmc_for_user(current_user.id, db)
    return {"status": "ok", "message": "PMC rebuilt from existing activities"}
