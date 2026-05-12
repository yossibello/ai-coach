"""
Tracking endpoints — supplement intake & performance test logging.

Routes:
  POST   /tracking/intake                     — log a supplement I'm taking
  POST   /tracking/intake/from-recommendation — one-click from current stack
  GET    /tracking/intake                     — list intakes (?active=true to filter)
  PATCH  /tracking/intake/{id}                — update (e.g. stop, set adherence)
  DELETE /tracking/intake/{id}                — remove

  POST   /tracking/performance-tests          — log an FTP/VO2max/etc test
  GET    /tracking/performance-tests          — list (?test_type= to filter)
  DELETE /tracking/performance-tests/{id}     — remove
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User
from app.models.tracking import SupplementIntake, SupplementDoseLog, PerformanceTest
from app.models.nutrition import SupplementRecommendation
from app.nutrition.supplements import SUPPLEMENTS


router = APIRouter()


ALLOWED_TEST_TYPES = {
    "ftp_20min", "ftp_ramp", "ftp_8min",
    "vo2max", "threshold_hr", "weight", "resting_hr",
}


# ─── Schemas ──────────────────────────────────────────────────────────────────
class IntakeCreate(BaseModel):
    supplement_key: str
    label: Optional[str] = None
    dose: Optional[float] = None
    dose_unit: Optional[str] = None
    frequency: Optional[str] = None
    timing: Optional[str] = None
    started_at: Optional[datetime] = None
    notes: Optional[str] = None


class IntakeUpdate(BaseModel):
    dose: Optional[float] = None
    dose_unit: Optional[str] = None
    frequency: Optional[str] = None
    timing: Optional[str] = None
    stopped_at: Optional[datetime] = None
    adherence_pct: Optional[int] = Field(default=None, ge=0, le=100)
    notes: Optional[str] = None


class IntakeOut(BaseModel):
    id: str
    supplement_key: str
    label: str
    dose: Optional[float]
    dose_unit: Optional[str]
    frequency: Optional[str]
    timing: Optional[str]
    started_at: str
    stopped_at: Optional[str]
    adherence_pct: Optional[int]
    source: str
    notes: Optional[str]
    is_active: bool


class IntakeFromRecommendation(BaseModel):
    supplement_key: str
    started_at: Optional[datetime] = None
    notes: Optional[str] = None


class PerformanceTestCreate(BaseModel):
    test_date: datetime
    test_type: str
    value: float
    unit: str
    notes: Optional[str] = None


class PerformanceTestOut(BaseModel):
    id: str
    test_date: str
    test_type: str
    value: float
    unit: str
    source: str
    notes: Optional[str]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _intake_out(i: SupplementIntake) -> IntakeOut:
    return IntakeOut(
        id=str(i.id),
        supplement_key=i.supplement_key,
        label=i.label,
        dose=i.dose,
        dose_unit=i.dose_unit,
        frequency=i.frequency,
        timing=i.timing,
        started_at=i.started_at.isoformat(),
        stopped_at=i.stopped_at.isoformat() if i.stopped_at else None,
        adherence_pct=i.adherence_pct,
        source=i.source,
        notes=i.notes,
        is_active=i.stopped_at is None,
    )


def _perf_out(p: PerformanceTest) -> PerformanceTestOut:
    return PerformanceTestOut(
        id=str(p.id),
        test_date=p.test_date.isoformat(),
        test_type=p.test_type,
        value=p.value,
        unit=p.unit,
        source=p.source,
        notes=p.notes,
    )


# ─── Supplement intake ────────────────────────────────────────────────────────
@router.post("/intake", response_model=IntakeOut)
async def create_intake(
    payload: IntakeCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Guard: prevent duplicate active entries for the same supplement
    existing = await db.execute(
        select(SupplementIntake)
        .where(
            SupplementIntake.user_id == current_user.id,
            SupplementIntake.supplement_key == payload.supplement_key,
            SupplementIntake.stopped_at.is_(None),
        )
        .limit(1)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, f"'{payload.supplement_key}' is already in your active log. Stop it first before logging again.")

    catalog = SUPPLEMENTS.get(payload.supplement_key)
    label = payload.label or (catalog["label"] if catalog else payload.supplement_key)

    intake = SupplementIntake(
        user_id=current_user.id,
        supplement_key=payload.supplement_key,
        label=label,
        dose=payload.dose if payload.dose is not None else (catalog["default_dose"] if catalog else None),
        dose_unit=payload.dose_unit or (catalog["dose_unit"] if catalog else None),
        frequency=payload.frequency or (catalog["frequency"] if catalog else None),
        timing=payload.timing or (catalog["timing"] if catalog else None),
        started_at=payload.started_at or datetime.now(timezone.utc),
        source="manual",
        notes=payload.notes,
    )
    db.add(intake)
    await db.commit()
    await db.refresh(intake)
    return _intake_out(intake)


@router.post("/intake/from-recommendation", response_model=IntakeOut)
async def create_intake_from_recommendation(
    payload: IntakeFromRecommendation,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """One-click 'I'm taking this' from the latest recommended stack."""
    res = await db.execute(
        select(SupplementRecommendation)
        .where(SupplementRecommendation.user_id == current_user.id)
        .order_by(desc(SupplementRecommendation.generated_at))
        .limit(1)
    )
    latest = res.scalar_one_or_none()
    if not latest:
        raise HTTPException(404, "No recommendation stack on file. Refresh /nutrition/supplements first.")

    stack = (latest.payload or {}).get("stack", [])
    item = next((s for s in stack if s.get("supplement_key") == payload.supplement_key), None)
    if not item:
        raise HTTPException(404, f"'{payload.supplement_key}' is not in your latest recommended stack.")

    # Guard: prevent duplicate active entries
    existing = await db.execute(
        select(SupplementIntake)
        .where(
            SupplementIntake.user_id == current_user.id,
            SupplementIntake.supplement_key == payload.supplement_key,
            SupplementIntake.stopped_at.is_(None),
        )
        .limit(1)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(409, f"'{payload.supplement_key}' is already in your active log.")

    intake = SupplementIntake(
        user_id=current_user.id,
        supplement_key=item["supplement_key"],
        label=item["label"],
        dose=item.get("dose"),
        dose_unit=item.get("dose_unit"),
        frequency=item.get("frequency"),
        timing=item.get("timing"),
        started_at=payload.started_at or datetime.now(timezone.utc),
        source="recommended",
        notes=payload.notes,
    )
    db.add(intake)
    await db.commit()
    await db.refresh(intake)
    return _intake_out(intake)


@router.get("/intake", response_model=list[IntakeOut])
async def list_intakes(
    active: Optional[bool] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    q = (
        select(SupplementIntake)
        .where(SupplementIntake.user_id == current_user.id)
        .order_by(desc(SupplementIntake.started_at))
    )
    if active is True:
        q = q.where(SupplementIntake.stopped_at.is_(None))
    elif active is False:
        q = q.where(SupplementIntake.stopped_at.is_not(None))

    rows = (await db.execute(q)).scalars().all()
    return [_intake_out(r) for r in rows]


@router.patch("/intake/{intake_id}", response_model=IntakeOut)
async def update_intake(
    intake_id: str,
    payload: IntakeUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(SupplementIntake).where(
            SupplementIntake.id == intake_id,
            SupplementIntake.user_id == current_user.id,
        )
    )
    intake = res.scalar_one_or_none()
    if not intake:
        raise HTTPException(404, "Intake not found.")

    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(intake, k, v)

    await db.commit()
    await db.refresh(intake)
    return _intake_out(intake)


@router.delete("/intake/{intake_id}")
async def delete_intake(
    intake_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(SupplementIntake).where(
            SupplementIntake.id == intake_id,
            SupplementIntake.user_id == current_user.id,
        )
    )
    intake = res.scalar_one_or_none()
    if not intake:
        raise HTTPException(404, "Intake not found.")
    await db.delete(intake)
    await db.commit()
    return {"ok": True}


# ─── Performance tests ────────────────────────────────────────────────────────
@router.post("/performance-tests", response_model=PerformanceTestOut)
async def create_performance_test(
    payload: PerformanceTestCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if payload.test_type not in ALLOWED_TEST_TYPES:
        raise HTTPException(
            400,
            f"test_type must be one of: {sorted(ALLOWED_TEST_TYPES)}",
        )
    if payload.value <= 0:
        raise HTTPException(400, "value must be positive.")

    pt = PerformanceTest(
        user_id=current_user.id,
        test_date=payload.test_date,
        test_type=payload.test_type,
        value=payload.value,
        unit=payload.unit,
        source="manual",
        notes=payload.notes,
    )
    db.add(pt)
    await db.commit()
    await db.refresh(pt)
    return _perf_out(pt)


@router.get("/performance-tests", response_model=list[PerformanceTestOut])
async def list_performance_tests(
    test_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    q = (
        select(PerformanceTest)
        .where(PerformanceTest.user_id == current_user.id)
        .order_by(desc(PerformanceTest.test_date))
    )
    if test_type:
        q = q.where(PerformanceTest.test_type == test_type)
    rows = (await db.execute(q)).scalars().all()
    return [_perf_out(r) for r in rows]


@router.delete("/performance-tests/{test_id}")
async def delete_performance_test(
    test_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(PerformanceTest).where(
            PerformanceTest.id == test_id,
            PerformanceTest.user_id == current_user.id,
        )
    )
    pt = res.scalar_one_or_none()
    if not pt:
        raise HTTPException(404, "Test not found.")
    await db.delete(pt)
    await db.commit()
    return {"ok": True}


# ─── Supplement dose log ──────────────────────────────────────────────────────
# Separate from intake (enrollment record). One row = one actual dose taken.
# Multiple entries per day are valid for split-dose protocols.
# Feeds: rule engine (taken_today / dose_exceeded) + future ML adherence features.

class DoseLogCreate(BaseModel):
    supplement_key: str
    label: str
    dose_taken: float = Field(gt=0, description="Actual amount taken")
    dose_unit: Optional[str] = None
    taken_at: Optional[datetime] = None
    notes: Optional[str] = None


class DoseLogUpdate(BaseModel):
    dose_taken: Optional[float] = Field(default=None, gt=0)
    notes: Optional[str] = None


class DoseLogOut(BaseModel):
    id: str
    supplement_key: str
    label: str
    dose_taken: float
    dose_unit: Optional[str]
    taken_at: str
    notes: Optional[str]


def _dose_log_out(d: SupplementDoseLog) -> DoseLogOut:
    return DoseLogOut(
        id=str(d.id),
        supplement_key=d.supplement_key,
        label=d.label,
        dose_taken=d.dose_taken,
        dose_unit=d.dose_unit,
        taken_at=d.taken_at.isoformat(),
        notes=d.notes,
    )


@router.post("/dose-log", response_model=DoseLogOut, status_code=201)
async def create_dose_log(
    payload: DoseLogCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Log one actual dose taken. Multiple per day are fine for split-dose protocols."""
    log = SupplementDoseLog(
        user_id=current_user.id,
        supplement_key=payload.supplement_key,
        label=payload.label,
        dose_taken=payload.dose_taken,
        dose_unit=payload.dose_unit,
        taken_at=payload.taken_at or datetime.now(timezone.utc),
        notes=payload.notes,
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return _dose_log_out(log)


@router.get("/dose-log", response_model=list[DoseLogOut])
async def list_dose_logs(
    since: Optional[str] = None,          # ISO date string; default last 30 days
    supplement_key: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        cutoff = (
            # Python 3.10 fromisoformat() doesn't handle trailing "Z"; normalise first
            datetime.fromisoformat(since.replace("Z", "+00:00")).astimezone(timezone.utc)
            if since
            else datetime.now(timezone.utc) - timedelta(days=30)
        )
        # Strip tzinfo so the comparison against SQLite's naive TEXT timestamps works
        cutoff = cutoff.replace(tzinfo=None)
    except ValueError:
        raise HTTPException(400, "Invalid 'since' date format. Use ISO 8601.")

    q = (
        select(SupplementDoseLog)
        .where(
            SupplementDoseLog.user_id == current_user.id,
            SupplementDoseLog.taken_at >= cutoff,
        )
        .order_by(desc(SupplementDoseLog.taken_at))
    )
    if supplement_key:
        q = q.where(SupplementDoseLog.supplement_key == supplement_key)

    rows = (await db.execute(q)).scalars().all()
    return [_dose_log_out(r) for r in rows]


@router.patch("/dose-log/{log_id}", response_model=DoseLogOut)
async def update_dose_log(
    log_id: str,
    payload: DoseLogUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(SupplementDoseLog).where(
            SupplementDoseLog.id == log_id,
            SupplementDoseLog.user_id == current_user.id,
        )
    )
    log = res.scalar_one_or_none()
    if not log:
        raise HTTPException(404, "Dose log entry not found.")
    if payload.dose_taken is not None:
        log.dose_taken = payload.dose_taken
    if payload.notes is not None:
        log.notes = payload.notes
    await db.commit()
    await db.refresh(log)
    return _dose_log_out(log)


@router.delete("/dose-log/{log_id}", status_code=204)
async def delete_dose_log(
    log_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(SupplementDoseLog).where(
            SupplementDoseLog.id == log_id,
            SupplementDoseLog.user_id == current_user.id,
        )
    )
    log = res.scalar_one_or_none()
    if not log:
        raise HTTPException(404, "Dose log entry not found.")
    await db.delete(log)
    await db.commit()

