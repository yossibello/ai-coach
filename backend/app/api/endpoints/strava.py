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
    backfill_power_curves,
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

            # After sync, run the power-curve backfill then the outcome backfill
            # in the same background task (create_task is unreliable here).
            await _run_backfill(user_id)
            await _run_outcome_backfill(user_id)

        except Exception:
            err = traceback.format_exc()
            log.error("Strava sync failed for user %s:\n%s", user_id, err)
            print(f"[SYNC ERROR] user={user_id}\n{err}", flush=True)
            _sync_tasks[task_id]["status"] = "failed"

    background_tasks.add_task(run)
    return SyncResponse(task_id=task_id)


async def _run_backfill(user_id: str) -> None:
    """Standalone backfill task — runs its own DB session."""
    try:
        async with AsyncSessionLocal() as db:
            from sqlalchemy import select as sa_select
            from sqlalchemy.orm import selectinload
            result = await db.execute(
                sa_select(User).where(User.id == user_id).options(
                    selectinload(User.profile)
                )
            )
            user = result.scalar_one()
            updated = await backfill_power_curves(user, db)
            await db.commit()
            log.info("Power-curve backfill done for user %s: %d activities updated", user_id, updated)
    except Exception:
        log.error("Power-curve backfill failed for user %s:\n%s", user_id, traceback.format_exc())


async def _run_outcome_backfill(user_id: str) -> None:
    """Score mature recommendations (≥28d old) for this user after each sync.

    Runs with a short horizon (3d) first to capture the day-1 actual workout,
    then the full 28d horizon for FTP/HRV outcome weighting. Using the
    28d default here covers both — the job is idempotent so it's safe to run
    every sync even though most recs won't be mature yet.
    """
    try:
        from app.ml.outcomes import backfill_user_outcomes
        async with AsyncSessionLocal() as db:
            written = await backfill_user_outcomes(user_id, db)
            await db.commit()
            if written:
                log.info("Outcome backfill: %d new rows for user %s", written, user_id)
    except Exception:
        log.error("Outcome backfill failed for user %s:\n%s", user_id, traceback.format_exc())


@router.post("/backfill-power-curves", response_model=SyncResponse)
async def trigger_backfill(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """Manually trigger power-curve backfill for activities missing pc_* data."""
    if not current_user.strava_connected:
        raise HTTPException(400, "Strava not connected")

    task_id = str(uuid.uuid4())
    _sync_tasks[task_id] = {"status": "running", "progress": 0, "total": 0}
    user_id = current_user.id

    def update_progress(progress: int, total: int):
        _sync_tasks[task_id].update({"progress": progress, "total": total})

    async def run():
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
                updated = await backfill_power_curves(bg_user, bg_db, update_progress)
                await bg_db.commit()
            _sync_tasks[task_id].update({"status": "completed", "updated": updated})
        except Exception:
            log.error("Backfill failed for user %s:\n%s", user_id, traceback.format_exc())
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

    method = ftp_result.get("method", "power_weighted")
    confidence = ftp_result.get("confidence", 0.0)
    age_days = ftp_result.get("best_ride_age_days")
    tsb_corr = ftp_result.get("tsb_correction", 1.0)
    conf_low = ftp_result.get("confidence_low")
    conf_high = ftp_result.get("confidence_high")
    sample_count = ftp_result.get("sample_count", 0)
    last_test_age = ftp_result.get("last_test_age_days")

    ftp_meta = {
        "method": method,
        "confidence": confidence,
        "sample_count": sample_count,
        "trend": ftp_result.get("trend", "stable"),
        "best_ride_age_days": age_days,
        "last_test_age_days": last_test_age,
        "confidence_low": conf_low,
        "confidence_high": conf_high,
        "tsb_correction": tsb_corr,
    }

    is_manual = profile is not None and profile.ftp_source == "manual"
    is_test_result = method.startswith("verified_test") or method == "test_blend" or method == "test_anchored"
    # Manual FTP is protected only against regular ride estimates, not against real test results
    manual_wins = is_manual and prev is not None and estimated < prev and not is_test_result

    if manual_wins:
        # Don't overwrite a manually-set FTP with a lower estimate — just persist meta
        profile.ftp_method = method
        profile.ftp_meta = ftp_meta
        db.add(profile)
    elif profile is None:
        profile = AthleteProfile(user_id=current_user.id, ftp=estimated, ftp_source="estimated", ftp_method=method, ftp_meta=ftp_meta)
        db.add(profile)
    else:
        profile.ftp = estimated
        profile.ftp_source = "estimated"
        profile.ftp_method = method
        profile.ftp_meta = ftp_meta
        db.add(profile)

    # Rebuild PMC with the new FTP so TSB/CTL reflect the updated value
    await db.flush()
    await compute_pmc_for_user(current_user.id, db)
    trend = ftp_result.get("trend", "stable")

    method_labels = {
        "power_weighted":   "power curve · multi-ride",
        "blended":          "blended with profile FTP",
        "manual_fallback":  "profile FTP (no recent power data)",
        "test_blend":       f"test ({ftp_result.get('last_test_age_days', '?')}d ago) + rides",
        "test_anchored":    f"rides · anchored by test ({ftp_result.get('last_test_age_days', '?')}d ago)",
    }
    for key in list(method_labels.keys()):
        if key.startswith("verified_test"):
            method_labels[key] = f"verified test ({ftp_result.get('last_test_age_days', '?')}d ago)"
    if method.startswith("verified_test"):
        method_labels[method] = f"verified test ({ftp_result.get('last_test_age_days', '?')}d ago)"
    method_note = method_labels.get(method, method)
    band = f", range {conf_low}-{conf_high}W" if conf_low and conf_high else ""
    trend_note = "" if trend == "stable" else f", trend: {trend}"

    if manual_wins:
        msg = (f"Manual FTP {prev}W kept — ride estimate: {estimated}W "
               f"({method_note}{band}, {sample_count} rides, "
               f"confidence {int(confidence*100)}%{trend_note}). "
               f"Clear manual FTP in Profile to let estimation take over.")
    elif prev != estimated:
        msg = (f"FTP updated from {prev}W → {estimated}W "
               f"({method_note}{band}, {sample_count} rides, "
               f"confidence {int(confidence*100)}%{trend_note})")
    else:
        msg = (f"FTP confirmed at {estimated}W "
               f"({method_note}{band}, {sample_count} rides{trend_note})")

    return FTPEstimateResponse(
        estimated_ftp=estimated,
        previous_ftp=prev,
        updated=(not manual_wins and prev != estimated),
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
