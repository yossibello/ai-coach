from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional
import uuid

from app.core.database import get_db
from app.core.deps import get_current_user
from app.core.config import settings
from app.models.user import User
from app.services.strava_service import (
    get_strava_auth_url,
    exchange_strava_code,
    sync_strava_history,
)

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

    def update_progress(progress: int, total: int):
        _sync_tasks[task_id].update({"progress": progress, "total": total})

    async def run():
        try:
            await sync_strava_history(current_user, db, update_progress)
            _sync_tasks[task_id]["status"] = "completed"
        except Exception:
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
