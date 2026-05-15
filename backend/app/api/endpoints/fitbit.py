"""
Fitbit endpoints.

GET    /fitbit/auth-url        — OAuth redirect URL
GET    /fitbit/callback        — OAuth callback (no auth header)
GET    /fitbit/status          — connection status
DELETE /fitbit/disconnect
POST   /fitbit/sync            — background sync
GET    /fitbit/sync-status/{id}
"""
from __future__ import annotations

import logging
import traceback
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy import select as sa_select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.core.database import AsyncSessionLocal, get_db
from app.core.deps import get_current_user
from app.models.user import User
from app.services import fitbit_service

log = logging.getLogger(__name__)
router = APIRouter()

_sync_tasks: dict[str, dict[str, Any]] = {}


class SyncResponse:
    def __init__(self, task_id: str):
        self.task_id = task_id


@router.get("/auth-url")
async def auth_url(current_user: User = Depends(get_current_user)):
    url = fitbit_service.get_auth_url(current_user.id)
    return {"url": url}


@router.get("/callback")
async def callback(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """OAuth callback — no Authorization header (called by Fitbit redirect)."""
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    error = request.query_params.get("error")

    if error or not code or not state:
        return RedirectResponse(f"{settings.FRONTEND_URL}/profile?fitbit=error")

    try:
        await fitbit_service.exchange_code(code, state, db)
        await db.commit()
    except Exception as exc:
        log.error("Fitbit callback failed: %s", exc)
        return RedirectResponse(f"{settings.FRONTEND_URL}/profile?fitbit=error")

    return RedirectResponse(f"{settings.FRONTEND_URL}/profile?fitbit=connected")


@router.get("/status")
async def status(current_user: User = Depends(get_current_user)):
    return {
        "connected": bool(current_user.fitbit_access_token_enc),
        "user_id": current_user.fitbit_user_id,
    }


@router.delete("/disconnect")
async def disconnect(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await fitbit_service.disconnect(current_user, db)
    return {"status": "disconnected"}


from pydantic import BaseModel


class SyncResponseModel(BaseModel):
    task_id: str


class SyncStatusModel(BaseModel):
    status: str
    stats: dict[str, int] | None = None
    error: str | None = None


@router.post("/sync", response_model=SyncResponseModel)
async def sync(
    background_tasks: BackgroundTasks,
    days: int = 30,
    current_user: User = Depends(get_current_user),
):
    if not current_user.fitbit_access_token_enc:
        raise HTTPException(400, "Fitbit not connected")
    days = max(1, min(days, 180))

    task_id = str(uuid.uuid4())
    _sync_tasks[task_id] = {"status": "running", "stats": None, "error": None}
    user_id = current_user.id

    async def run():
        try:
            async with AsyncSessionLocal() as bg_db:
                res = await bg_db.execute(
                    sa_select(User).where(User.id == user_id).options(selectinload(User.profile))
                )
                bg_user = res.scalar_one()
                stats = await fitbit_service.sync_fitbit(bg_user, bg_db, days=days)
                await bg_db.commit()
            _sync_tasks[task_id].update({"status": "completed", "stats": stats})
        except Exception:
            err = traceback.format_exc()
            log.error("Fitbit sync failed for user %s:\n%s", user_id, err)
            _sync_tasks[task_id].update({"status": "failed", "error": err.splitlines()[-1]})

    background_tasks.add_task(run)
    return SyncResponseModel(task_id=task_id)


@router.get("/sync-status/{task_id}", response_model=SyncStatusModel)
async def sync_status(task_id: str, current_user: User = Depends(get_current_user)):
    task = _sync_tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return SyncStatusModel(**task)
