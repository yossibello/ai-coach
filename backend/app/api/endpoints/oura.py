"""
Oura Ring endpoints.

POST   /oura/connect           — save personal access token
GET    /oura/status            — connection status
DELETE /oura/disconnect
POST   /oura/sync              — background sync
GET    /oura/sync-status/{id}
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
from app.services import oura_service

log = logging.getLogger(__name__)
router = APIRouter()

_sync_tasks: dict[str, dict[str, Any]] = {}


class ConnectRequest(BaseModel):
    token: str = Field(..., min_length=10)


class SyncResponse(BaseModel):
    task_id: str


class SyncStatus(BaseModel):
    status: str
    stats: dict[str, int] | None = None
    error: str | None = None


@router.get("/status")
async def status(current_user: User = Depends(get_current_user)):
    return {"connected": bool(current_user.oura_access_token_enc)}


@router.post("/connect")
async def connect(
    body: ConnectRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        await oura_service.validate_and_save_token(current_user, body.token, db)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Oura token validation failed: {exc}")
    return {"status": "connected"}


@router.delete("/disconnect")
async def disconnect(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await oura_service.disconnect(current_user, db)
    return {"status": "disconnected"}


@router.post("/sync", response_model=SyncResponse)
async def sync(
    background_tasks: BackgroundTasks,
    days: int = 30,
    current_user: User = Depends(get_current_user),
):
    if not current_user.oura_access_token_enc:
        raise HTTPException(400, "Oura not connected")
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
                stats = await oura_service.sync_oura(bg_user, bg_db, days=days)
                await bg_db.commit()
            _sync_tasks[task_id].update({"status": "completed", "stats": stats})
        except Exception:
            err = traceback.format_exc()
            log.error("Oura sync failed for user %s:\n%s", user_id, err)
            _sync_tasks[task_id].update({"status": "failed", "error": err.splitlines()[-1]})

    background_tasks.add_task(run)
    return SyncResponse(task_id=task_id)


@router.get("/sync-status/{task_id}", response_model=SyncStatus)
async def sync_status(task_id: str, current_user: User = Depends(get_current_user)):
    task = _sync_tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return SyncStatus(**task)
