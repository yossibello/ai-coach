"""
Garmin Connect endpoints.

POST /garmin/connect       — login with username/password (dev path)
POST /garmin/sync          — kick off background sync (rides + wellness)
GET  /garmin/sync-status/{task_id}
GET  /garmin/status        — connection status + last sync stats
DELETE /garmin/disconnect
GET  /garmin/auth-url      — placeholder for future OAuth flow
"""
from __future__ import annotations

import logging
import traceback
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select as sa_select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import AsyncSessionLocal, get_db
from app.core.deps import get_current_user
from app.models.user import User
from app.services import garmin_service

log = logging.getLogger(__name__)
router = APIRouter()

_sync_tasks: dict[str, dict[str, Any]] = {}


class ConnectRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=200)
    password: str = Field(..., min_length=4, max_length=200)


class StatusResponse(BaseModel):
    connected: bool
    username: str | None
    method: str  # "credentials" | "oauth" | "none"


class SyncResponse(BaseModel):
    task_id: str


class SyncStatus(BaseModel):
    status: str
    progress: int
    total: int
    stats: dict[str, int] | None = None
    error: str | None = None


@router.get("/status", response_model=StatusResponse)
async def status(current_user: User = Depends(get_current_user)):
    if current_user.garmin_access_token:
        method = "oauth"
    elif current_user.garmin_username:
        method = "credentials"
    else:
        method = "none"
    return StatusResponse(
        connected=current_user.garmin_connected,
        username=current_user.garmin_username,
        method=method,
    )


@router.post("/connect")
async def connect(
    body: ConnectRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        await garmin_service.connect_with_credentials(
            current_user, body.username, body.password, db
        )
    except Exception as exc:
        log.warning("garmin connect failed for %s: %s", current_user.email, exc)
        raise HTTPException(status_code=400, detail=f"Garmin login failed: {exc}")
    return {"status": "connected", "username": body.username}


@router.delete("/disconnect")
async def disconnect(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await garmin_service.disconnect(current_user, db)
    return {"status": "disconnected"}


@router.get("/auth-url")
async def auth_url():
    """Placeholder for the official Garmin OAuth flow."""
    try:
        return {"url": garmin_service.get_oauth_authorize_url()}
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))


@router.post("/sync", response_model=SyncResponse)
async def sync(
    background_tasks: BackgroundTasks,
    days: int = 30,
    current_user: User = Depends(get_current_user),
):
    if not current_user.garmin_connected:
        raise HTTPException(400, "Garmin not connected")
    days = max(1, min(days, 180))

    task_id = str(uuid.uuid4())
    _sync_tasks[task_id] = {"status": "running", "progress": 0, "total": 0, "stats": None}
    user_id = current_user.id

    def update_progress(progress: int, total: int):
        _sync_tasks[task_id].update({"progress": progress, "total": total})

    async def run():
        try:
            async with AsyncSessionLocal() as bg_db:
                res = await bg_db.execute(
                    sa_select(User).where(User.id == user_id).options(selectinload(User.profile))
                )
                bg_user = res.scalar_one()
                stats = await garmin_service.sync_garmin(
                    bg_user, bg_db, days=days, progress_callback=update_progress
                )
                # After importing rides, recompute PMC so CTL/ATL/TSB include the new TSS
                from app.services.metrics_service import compute_pmc_for_user
                await compute_pmc_for_user(user_id, bg_db)
                await bg_db.commit()
            _sync_tasks[task_id].update({"status": "completed", "stats": stats})
        except Exception:
            err = traceback.format_exc()
            log.error("Garmin sync failed for user %s:\n%s", user_id, err)
            _sync_tasks[task_id].update({"status": "failed", "error": err.splitlines()[-1]})

    background_tasks.add_task(run)
    return SyncResponse(task_id=task_id)


@router.get("/sync-status/{task_id}", response_model=SyncStatus)
async def sync_status(task_id: str, current_user: User = Depends(get_current_user)):
    task = _sync_tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return SyncStatus(**task)
