"""
Compute cycling metrics for activities and the Performance Management Chart.
Implements Coggan's TSS, CTL/ATL/TSB (exponential moving averages).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select as sa_select, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.activity import Activity
from app.models.user import User, AthleteProfile
from app.models.recommendation import FitnessMetric
from app.ml.quality import score_activity

# EMA constants
CTL_DAYS = 42  # Chronic Training Load (fitness)
ATL_DAYS = 7   # Acute Training Load (fatigue)
CTL_ALPHA = 2 / (CTL_DAYS + 1)
ATL_ALPHA = 2 / (ATL_DAYS + 1)


def compute_activity_metrics(activity: Activity, user: User | None) -> None:
    """
    Fill derived metrics: IF, TSS, NP (if missing), HR drift, aerobic efficiency.
    Modifies activity in place.
    """
    ftp = _get_ftp(user)

    # Intensity Factor = NP / FTP
    if activity.normalized_power and ftp:
        activity.intensity_factor = round(activity.normalized_power / ftp, 3)

    # TSS = (duration_s × NP × IF) / (FTP × 3600) × 100
    if activity.normalized_power and activity.intensity_factor and ftp and activity.duration_seconds:
        raw_tss = (
            activity.duration_seconds
            * activity.normalized_power
            * activity.intensity_factor
            / (ftp * 3600)
            * 100
        )
        activity.tss = round(raw_tss, 1)

    # TSS from HR if no power (estimate using HR reserve)
    if not activity.tss and activity.avg_hr and user:
        profile = getattr(user, "profile", None)
        if profile and profile.max_hr and profile.resting_hr:
            hrr_frac = (activity.avg_hr - profile.resting_hr) / (
                profile.max_hr - profile.resting_hr
            )
            # Approximate IF from HR fraction (rough)
            if_hr = hrr_frac ** 0.91
            activity.intensity_factor = round(if_hr, 3)
            if activity.duration_seconds:
                activity.tss = round(
                    (activity.duration_seconds / 3600) * (if_hr ** 2) * 100, 1
                )

    # Variability Index = NP / AP
    if activity.normalized_power and activity.avg_power and activity.avg_power > 0:
        activity.variability_index = round(activity.normalized_power / activity.avg_power, 3)

    # Aerobic Efficiency = avg_power / avg_hr  (w/bpm)
    if activity.avg_power and activity.avg_hr and activity.avg_hr > 0:
        activity.aerobic_efficiency = round(activity.avg_power / activity.avg_hr, 3)

    # Classify workout type from IF/TSS if not already set
    if not activity.workout_type and activity.intensity_factor:
        activity.workout_type = _classify_workout(activity.intensity_factor, activity.duration_seconds)


def _get_ftp(user: User | None) -> int | None:
    if user is None:
        return None
    profile = getattr(user, "profile", None)
    if profile and profile.ftp:
        return profile.ftp
    return 200  # fallback


def _score_and_tag(activity: Activity, user: User | None) -> None:
    """
    Apply rule-based quality scoring to a freshly-ingested activity.

    Sets activity.quality_score, activity.quality_reasons, activity.review_status.
    Auto-quarantines obviously bad rides; everything else stays 'confirmed'.
    Phase-2 outlier detection (vs the user's own history) is added separately.
    """
    profile = getattr(user, "profile", None) if user else None
    score, reasons = score_activity(
        {
            "duration_seconds":  activity.duration_seconds,
            "distance_meters":   activity.distance_meters,
            "avg_power":         activity.avg_power,
            "max_power":         activity.max_power,
            "normalized_power":  activity.normalized_power,
            "intensity_factor":  activity.intensity_factor,
            "tss":               activity.tss,
            "avg_hr":            activity.avg_hr,
            "max_hr":            activity.max_hr,
            "avg_cadence":       activity.avg_cadence,
            "trainer":           activity.trainer,
            "time_in_zones":     activity.time_in_zones,
        },
        profile_max_hr=getattr(profile, "max_hr", None),
        profile_ftp=getattr(profile, "ftp", None),
    )
    activity.quality_score   = score
    activity.quality_reasons = reasons or None
    activity.review_status   = "quarantined" if score == "rejected" else "confirmed"


def _classify_workout(if_: float, duration_s: int) -> str:
    """Classify workout type from Intensity Factor."""
    if if_ < 0.55:
        return "recovery"
    if if_ < 0.75:
        return "easy" if duration_s < 5400 else "endurance"
    if if_ < 0.88:
        return "tempo"
    if if_ < 0.95:
        return "sweetspot"
    if if_ < 1.05:
        return "threshold"
    if if_ < 1.20:
        return "vo2max"
    return "sprint"


async def compute_pmc_for_user(user_id: str, db: AsyncSession) -> None:
    """
    Recompute the full Performance Management Chart for a user.
    Deletes existing FitnessMetric rows and rebuilds from scratch.
    """
    # Fetch all activities ordered by date
    result = await db.execute(
        sa_select(Activity, AthleteProfile)
        .join(AthleteProfile, AthleteProfile.user_id == Activity.user_id, isouter=True)
        .where(Activity.user_id == user_id)
        .order_by(Activity.date)
    )
    rows = result.all()

    if not rows:
        return

    # Build daily TSS map
    daily_tss: dict[str, float] = {}
    daily_ftp: dict[str, float] = {}
    default_ftp = 200.0

    for act, profile in rows:
        day = act.date.date().isoformat()
        tss = act.tss or 0
        daily_tss[day] = daily_tss.get(day, 0) + tss
        ftp = (profile.ftp if profile and profile.ftp else default_ftp)
        daily_ftp[day] = ftp

    # Generate continuous date range
    start_date = min(datetime.fromisoformat(d) for d in daily_tss)
    end_date = datetime.now(timezone.utc)

    # Delete existing metrics
    await db.execute(sa_delete(FitnessMetric).where(FitnessMetric.user_id == user_id))

    ctl = 0.0
    atl = 0.0
    new_metrics = []
    current = start_date

    while current <= end_date:
        day = current.date().isoformat()
        tss = daily_tss.get(day, 0)

        ctl = ctl + CTL_ALPHA * (tss - ctl)
        atl = atl + ATL_ALPHA * (tss - atl)
        tsb = ctl - atl

        ftp = daily_ftp.get(day, default_ftp)

        metric = FitnessMetric(
            user_id=user_id,
            date=current.replace(tzinfo=timezone.utc) if current.tzinfo is None else current,
            ctl=round(ctl, 2),
            atl=round(atl, 2),
            tsb=round(tsb, 2),
            tss=round(tss, 2),
            ftp=ftp,
        )
        new_metrics.append(metric)
        current += timedelta(days=1)

    db.add_all(new_metrics)
    await db.flush()
