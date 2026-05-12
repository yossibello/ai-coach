"""
Nutrition / blood-test / supplement endpoints.

Routes:
  POST   /nutrition/blood-tests          — upload PDF, parse, store
  POST   /nutrition/blood-tests/manual   — manual marker entry
  GET    /nutrition/blood-tests          — list user's tests
  GET    /nutrition/blood-tests/{id}     — single test detail
  DELETE /nutrition/blood-tests/{id}     — remove a test
  GET    /nutrition/markers/{key}        — time-series for one marker
  GET    /nutrition/supplements          — current recommended stack (cached 6h)
  POST   /nutrition/supplements/refresh  — force refresh
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from sqlalchemy import select, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User, AthleteProfile
from app.models.activity import Activity
from app.models.recommendation import FitnessMetric
from app.models.nutrition import BloodTest, BloodMarker, SupplementRecommendation
from app.models.tracking import SupplementDoseLog, SupplementIntake
from app.nutrition.engine import recommend_supplements, ENGINE_VERSION
from app.nutrition.pdf_parser import parse_blood_test_pdf
from app.nutrition.markers import MARKERS, status_for_value, normalize_unit


router = APIRouter()


# ─── Schemas ──────────────────────────────────────────────────────────────────
class MarkerEntry(BaseModel):
    marker_key: str
    value:      float
    unit:       Optional[str] = None


class ManualBloodTestIn(BaseModel):
    test_date: datetime
    lab_name:  Optional[str] = None
    notes:     Optional[str] = None
    markers:   list[MarkerEntry] = Field(default_factory=list)


class BloodTestOut(BaseModel):
    id: str
    test_date: str
    lab_name: Optional[str]
    source: str
    parser_version: Optional[str]
    parser_confidence: Optional[float]
    markers: dict[str, Any]
    notes: Optional[str]


class SupplementStackOut(BaseModel):
    id: str
    generated_at: str
    engine_version: str
    is_cold_start: bool
    based_on_blood_test_id: Optional[str]
    payload: dict[str, Any]


# ─── Blood test upload (PDF) ──────────────────────────────────────────────────
@router.post("/blood-tests", response_model=BloodTestOut)
async def upload_blood_test_pdf(
    file: UploadFile = File(...),
    notes: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a lab-report PDF; parser extracts markers and stores them."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    pdf_bytes = await file.read()
    if len(pdf_bytes) > 20 * 1024 * 1024:
        raise HTTPException(400, "PDF too large (>20 MB).")

    sex = await _user_sex(current_user, db)
    parsed = parse_blood_test_pdf(pdf_bytes, sex=sex)

    if not parsed["markers"]:
        raise HTTPException(
            422,
            f"Could not extract any markers from this PDF. "
            f"Warnings: {'; '.join(parsed.get('warnings', []))}. "
            f"Please use the manual-entry endpoint instead.",
        )

    test_date = (
        datetime.fromisoformat(parsed["test_date"]).replace(tzinfo=timezone.utc)
        if parsed.get("test_date")
        else datetime.now(timezone.utc)
    )

    bt = BloodTest(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        test_date=test_date,
        lab_name=parsed.get("lab_name"),
        source="pdf",
        raw_filename=file.filename,
        parser_version=parsed["parser_version"],
        parser_confidence=parsed["parser_confidence"],
        markers=parsed["markers"],
        notes=notes,
    )
    db.add(bt)
    await db.flush()

    _persist_markers_denormalized(db, bt, parsed["markers"])
    await db.commit()
    return _bt_out(bt)


@router.post("/blood-tests/manual", response_model=BloodTestOut)
async def upload_blood_test_manual(
    body: ManualBloodTestIn,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Manual marker entry — for users without a PDF or when parser fails."""
    sex = await _user_sex(current_user, db)
    markers_dict: dict[str, dict] = {}
    for m in body.markers:
        if m.marker_key not in MARKERS:
            continue
        v_norm, u_norm = normalize_unit(m.marker_key, m.value, m.unit)
        m_def = MARKERS[m.marker_key]
        ref_low, ref_high = m_def["ref_low"], m_def["ref_high"]
        if sex and "sex_specific" in m_def and sex.lower() in m_def["sex_specific"]:
            ref_low, ref_high = m_def["sex_specific"][sex.lower()]
        markers_dict[m.marker_key] = {
            "value":    round(v_norm, 4),
            "unit":     u_norm,
            "ref_low":  ref_low,
            "ref_high": ref_high,
            "status":   status_for_value(m.marker_key, v_norm, sex),
            "label":    m_def["label"],
            "category": m_def["category"],
        }

    if not markers_dict:
        raise HTTPException(400, "No recognized markers in submission.")

    bt = BloodTest(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        test_date=body.test_date,
        lab_name=body.lab_name,
        source="manual",
        markers=markers_dict,
        notes=body.notes,
    )
    db.add(bt)
    await db.flush()
    _persist_markers_denormalized(db, bt, markers_dict)
    await db.commit()
    return _bt_out(bt)


@router.get("/blood-tests", response_model=list[BloodTestOut])
async def list_blood_tests(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(BloodTest)
        .where(BloodTest.user_id == current_user.id)
        .order_by(desc(BloodTest.test_date))
    )
    return [_bt_out(bt) for bt in res.scalars().all()]


@router.get("/blood-tests/{test_id}", response_model=BloodTestOut)
async def get_blood_test(
    test_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    bt = await _get_user_blood_test(db, current_user.id, test_id)
    return _bt_out(bt)


@router.delete("/blood-tests/{test_id}")
async def delete_blood_test(
    test_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    bt = await _get_user_blood_test(db, current_user.id, test_id)
    await db.delete(bt)
    await db.commit()
    return {"deleted": test_id}


@router.get("/markers/{marker_key}")
async def marker_time_series(
    marker_key: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if marker_key not in MARKERS:
        raise HTTPException(404, f"Unknown marker: {marker_key}")
    res = await db.execute(
        select(BloodMarker)
        .where(
            BloodMarker.user_id == current_user.id,
            BloodMarker.marker_key == marker_key,
        )
        .order_by(asc(BloodMarker.test_date))
    )
    rows = res.scalars().all()
    return {
        "marker_key": marker_key,
        "label":      MARKERS[marker_key]["label"],
        "unit":       MARKERS[marker_key]["unit"],
        "points":     [
            {
                "test_date": r.test_date.isoformat(),
                "value":     r.value,
                "status":    r.status,
                "ref_low":   r.ref_low,
                "ref_high":  r.ref_high,
            }
            for r in rows
        ],
    }


# ─── Supplement stack ─────────────────────────────────────────────────────────
@router.get("/supplements", response_model=SupplementStackOut)
async def get_supplement_stack(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return latest supplement stack (cached 6h) enriched with today's dose data."""
    res = await db.execute(
        select(SupplementRecommendation)
        .where(SupplementRecommendation.user_id == current_user.id)
        .order_by(desc(SupplementRecommendation.generated_at))
        .limit(1)
    )
    sr = res.scalar_one_or_none()
    if sr:
        age = datetime.now(timezone.utc) - sr.generated_at.replace(tzinfo=timezone.utc)
        if age < timedelta(hours=6):
            payload = dict(sr.payload or {})
            payload["stack"] = await _enrich_stack(
                payload.get("stack", []), current_user.id, db
            )
            return SupplementStackOut(
                id=sr.id,
                generated_at=sr.generated_at.isoformat(),
                engine_version=sr.engine_version,
                is_cold_start=sr.is_cold_start,
                based_on_blood_test_id=sr.based_on_blood_test_id,
                payload=payload,
            )

    sr = await _generate_and_store_stack(current_user, db)
    payload = dict(sr.payload or {})
    payload["stack"] = await _enrich_stack(
        payload.get("stack", []), current_user.id, db
    )
    return SupplementStackOut(
        id=sr.id,
        generated_at=sr.generated_at.isoformat(),
        engine_version=sr.engine_version,
        is_cold_start=sr.is_cold_start,
        based_on_blood_test_id=sr.based_on_blood_test_id,
        payload=payload,
    )


@router.post("/supplements/refresh", response_model=SupplementStackOut)
async def refresh_supplement_stack(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    sr = await _generate_and_store_stack(current_user, db)
    payload = dict(sr.payload or {})
    payload["stack"] = await _enrich_stack(
        payload.get("stack", []), current_user.id, db
    )
    return SupplementStackOut(
        id=sr.id,
        generated_at=sr.generated_at.isoformat(),
        engine_version=sr.engine_version,
        is_cold_start=sr.is_cold_start,
        based_on_blood_test_id=sr.based_on_blood_test_id,
        payload=payload,
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────
async def _user_sex(user: User, db: AsyncSession) -> Optional[str]:
    res = await db.execute(
        select(AthleteProfile).where(AthleteProfile.user_id == user.id)
    )
    p = res.scalar_one_or_none()
    return p.sex if p else None


async def _get_user_blood_test(db: AsyncSession, user_id: str, test_id: str) -> BloodTest:
    res = await db.execute(
        select(BloodTest).where(
            BloodTest.id == test_id, BloodTest.user_id == user_id
        )
    )
    bt = res.scalar_one_or_none()
    if not bt:
        raise HTTPException(404, "Blood test not found.")
    return bt


def _persist_markers_denormalized(
    db: AsyncSession, bt: BloodTest, markers: dict[str, dict]
) -> None:
    for key, m in markers.items():
        db.add(BloodMarker(
            id=str(uuid.uuid4()),
            user_id=bt.user_id,
            blood_test_id=bt.id,
            test_date=bt.test_date,
            marker_key=key,
            value=m["value"],
            unit=m.get("unit"),
            ref_low=m.get("ref_low"),
            ref_high=m.get("ref_high"),
            status=m.get("status", "unknown"),
        ))


def _bt_out(bt: BloodTest) -> BloodTestOut:
    return BloodTestOut(
        id=bt.id,
        test_date=bt.test_date.isoformat(),
        lab_name=bt.lab_name,
        source=bt.source,
        parser_version=bt.parser_version,
        parser_confidence=bt.parser_confidence,
        markers=bt.markers or {},
        notes=bt.notes,
    )


def _supp_out(sr: SupplementRecommendation) -> SupplementStackOut:
    return SupplementStackOut(
        id=sr.id,
        generated_at=sr.generated_at.isoformat(),
        engine_version=sr.engine_version,
        is_cold_start=sr.is_cold_start,
        based_on_blood_test_id=sr.based_on_blood_test_id,
        payload=sr.payload or {},
    )


async def _enrich_stack(stack: list[dict], user_id: str, db: AsyncSession) -> list[dict]:
    """Overlay real-time dose-log + enrollment status onto stored stack items.

    Returns a new list with four extra fields per item:
      taken_today      — bool: any dose logged since midnight UTC
      today_total_dose — float | None: sum of all doses today
      dose_exceeded    — bool: today_total > 1.5× recommended (safety flag)
      already_enrolled — bool: user has an active SupplementIntake record
      dose_warning     — str | None: human-readable over-dose message
    """
    # Use naive UTC for comparison — SQLite stores datetimes as naive TEXT
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
    )

    dose_res = await db.execute(
        select(SupplementDoseLog).where(
            SupplementDoseLog.user_id == user_id,
            SupplementDoseLog.taken_at >= today_start,
        )
    )
    today_summary: dict[str, dict] = {}
    for log in dose_res.scalars().all():
        entry = today_summary.setdefault(
            log.supplement_key, {"total": 0.0, "unit": log.dose_unit}
        )
        entry["total"] += log.dose_taken

    intake_res = await db.execute(
        select(SupplementIntake.supplement_key).where(
            SupplementIntake.user_id == user_id,
            SupplementIntake.stopped_at.is_(None),
        )
    )
    active_keys = {row[0] for row in intake_res.all()}

    # Deduplicate by supplement_key — keep first occurrence (highest-priority)
    seen: set[str] = set()
    unique_stack: list[dict] = []
    for item in stack:
        k = item.get("supplement_key", "")
        if k and k not in seen:
            seen.add(k)
            unique_stack.append(item)
    stack = unique_stack

    enriched = []
    for item in stack:
        key = item["supplement_key"]
        taken = key in today_summary
        total = today_summary.get(key, {}).get("total", 0.0)
        rec_dose = item.get("dose") or 0.0
        exceeded = taken and rec_dose > 0 and total > rec_dose * 1.5
        enriched.append({
            **item,
            "taken_today":      taken,
            "today_total_dose": round(total, 2) if taken else None,
            "dose_exceeded":    exceeded,
            "already_enrolled": key in active_keys,
            "dose_warning": (
                f"You've taken {total:.1f} {item.get('dose_unit', '')} today — "
                f"recommended is {rec_dose} {item.get('dose_unit', '')}."
            ) if exceeded else None,
        })
    return enriched


async def _generate_and_store_stack(
    user: User, db: AsyncSession
) -> SupplementRecommendation:
    # Profile
    pres = await db.execute(
        select(AthleteProfile).where(AthleteProfile.user_id == user.id)
    )
    profile = pres.scalar_one_or_none()
    profile_dict = {
        "sex":     profile.sex if profile else None,
        "age":     profile.age if profile else None,
        "diet":    profile.diet if profile else None,
        "climate": profile.climate if profile else None,
        "training_days_per_week": profile.training_days_per_week if profile else None,
        "recent_illness_count_3m": profile.recent_illness_count_3m if profile else 0,
    }

    # Latest fitness metric
    fres = await db.execute(
        select(FitnessMetric)
        .where(FitnessMetric.user_id == user.id)
        .order_by(desc(FitnessMetric.date)).limit(1)
    )
    fm = fres.scalar_one_or_none()
    ctl = fm.ctl if fm else 0.0

    # Last 28 days of activities for weekly hours, weekly TSS, avg temp
    cutoff = datetime.now(timezone.utc) - timedelta(days=28)
    ares = await db.execute(
        select(Activity).where(
            Activity.user_id == user.id,
            Activity.date >= cutoff,
            Activity.review_status == "confirmed",
        )
    )
    acts = ares.scalars().all()
    total_seconds = sum((a.duration_seconds or 0) for a in acts)
    weekly_hours = (total_seconds / 3600.0) / 4.0
    weekly_tss = sum((a.tss or 0) for a in acts) / 4.0
    temps = [a.temperature_c for a in acts if a.temperature_c is not None]
    recent_avg_temp = (sum(temps) / len(temps)) if temps else None

    # Latest blood test
    btres = await db.execute(
        select(BloodTest)
        .where(BloodTest.user_id == user.id)
        .order_by(desc(BloodTest.test_date)).limit(1)
    )
    bt = btres.scalar_one_or_none()
    blood_test = (
        {"id": bt.id, "markers": bt.markers}
        if bt else None
    )

    # Upcoming event context
    days_to_event = None
    if profile and profile.goal_event_date:
        days_to_event = max(0, (
            profile.goal_event_date.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)
        ).days)

    payload = recommend_supplements(
        profile=profile_dict,
        weekly_hours=weekly_hours,
        ctl=ctl,
        weekly_tss=weekly_tss,
        recent_avg_temp_c=recent_avg_temp,
        upcoming_event_type=(profile.event_type if profile else None),
        days_to_event=days_to_event,
        workout_focus=None,                   # could pull from latest recommendation
        blood_test=blood_test,
    )

    sr = SupplementRecommendation(
        id=str(uuid.uuid4()),
        user_id=user.id,
        engine_version=ENGINE_VERSION,
        is_cold_start=payload["is_cold_start"],
        based_on_blood_test_id=payload.get("based_on_blood_test_id"),
        payload=payload,
    )
    db.add(sr)
    await db.commit()
    await db.refresh(sr)
    return sr
