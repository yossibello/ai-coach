import os
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, UploadFile, File, Form, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User
from app.models.bug_report import BugReport

router = APIRouter()

UPLOAD_DIR = "/app/uploads/bug_reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/bugs")
async def submit_bug_report(
    request: Request,
    page:        str        = Form(...),
    description: str        = Form(...),
    severity:    str        = Form("medium"),
    screenshot:  UploadFile | None = File(None),
    current_user: User      = Depends(get_current_user),
    db: AsyncSession        = Depends(get_db),
):
    screenshot_path = None
    if screenshot and screenshot.filename:
        ext = os.path.splitext(screenshot.filename)[-1].lower() or ".png"
        fname = f"{uuid.uuid4()}{ext}"
        fpath = os.path.join(UPLOAD_DIR, fname)
        with open(fpath, "wb") as f:
            f.write(await screenshot.read())
        screenshot_path = fpath

    report = BugReport(
        user_id         = current_user.id,
        page            = page[:255],
        description     = description,
        severity        = severity if severity in ("low", "medium", "high") else "medium",
        screenshot_path = screenshot_path,
        user_agent      = request.headers.get("user-agent", "")[:512],
    )
    db.add(report)
    await db.commit()
    return {"ok": True, "id": str(report.id)}


@router.get("/bugs")
async def list_bug_reports(
    current_user: User       = Depends(get_current_user),
    db: AsyncSession         = Depends(get_db),
):
    result = await db.execute(
        select(BugReport).order_by(desc(BugReport.created_at)).limit(200)
    )
    reports = result.scalars().all()
    return [
        {
            "id":          str(r.id),
            "page":        r.page,
            "description": r.description,
            "severity":    r.severity,
            "status":      r.status,
            "has_screenshot": r.screenshot_path is not None,
            "created_at":  r.created_at.isoformat() if r.created_at else None,
        }
        for r in reports
    ]
