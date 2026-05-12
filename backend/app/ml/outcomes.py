"""
Outcome backfill — computes prediction-vs-actual diffs for each
recommendation that is now mature enough to evaluate.

A recommendation generated at T₀ is evaluable once T₀ + horizon_days
have elapsed AND the user has logged at least one ride after T₀. We
write one PredictionOutcome row per recommendation (idempotent — the
unique constraint on `recommendation_id` prevents duplicates).

Metrics computed:
  • workout_type_match  — did the next ride after T₀ match the
                          predicted workout_type? (boolean)
  • duration_abs_err_min — |actual - predicted| minutes for that ride
  • ftp_delta_abs_err_w  — |Δftp_actual - Δftp_predicted| over the
                           horizon window (default 28 days)

The job is intentionally simple and deterministic so it can be re-run
safely. It does NOT update or train anything — it only records data
that the Phase-3 LoRA fine-tune loop will consume.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.activity import Activity
from app.models.recommendation import Recommendation
from app.models.outcome import PredictionOutcome


# How long after a recommendation is generated before we score it.
DEFAULT_HORIZON_DAYS = 28
# Window AROUND the recommendation date in which we look for the
# "actual" next ride. ±2 days handles small timezone / scheduling drift.
NEXT_RIDE_WINDOW_DAYS = 3


def _ensure_aware(dt: datetime) -> datetime:
    """SQLite returns DateTime(timezone=True) as naive — coerce to UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _extract_predictions(rec: Recommendation) -> dict:
    """Pull predicted fields out of the JSON payload defensively."""
    p = rec.payload or {}
    nw = p.get("next_workout") or {}
    forecast = p.get("forecast") or {}
    return {
        "workout_type": nw.get("workout_type"),
        "duration_min": nw.get("duration_minutes"),
        "intensity":    nw.get("target_intensity_factor")
                        or nw.get("intensity_factor"),
        "ftp_delta_w":  forecast.get("predicted_ftp_change_watts"),
    }


def _ftp_at(activities: list[Activity], at: datetime, profile_ftp: float | None) -> float | None:
    """Best-effort FTP lookup: average IF×duration is not used here — we
    rely on the rider's CURRENT profile FTP at the boundary. If we want
    more precision later we can read FitnessMetric.ftp closest to `at`.
    """
    return profile_ftp


async def _backfill_one(
    rec: Recommendation,
    activities_after: list[Activity],
    horizon_end: datetime,
    db: AsyncSession,
) -> PredictionOutcome | None:
    """Compute and insert a single PredictionOutcome row.

    Returns the new row, or None if there isn't enough data yet.
    """
    pred = _extract_predictions(rec)
    if not activities_after:
        return None

    rec_at = _ensure_aware(rec.generated_at)

    # First ride after the recommendation date that falls within the
    # ±NEXT_RIDE_WINDOW_DAYS window. If none, fall back to the very next
    # ride so we still capture "rider did something different on day N+5".
    window_end = rec_at + timedelta(days=NEXT_RIDE_WINDOW_DAYS)
    next_ride = next(
        (a for a in activities_after if _ensure_aware(a.date) <= window_end),
        activities_after[0],
    )

    actual_workout_type = next_ride.workout_type
    actual_duration_min = (
        int(next_ride.duration_seconds / 60) if next_ride.duration_seconds else None
    )
    actual_intensity = next_ride.intensity_factor

    # FTP delta over the horizon: simplest stable signal is
    # (most-recent-ride IF × ftp) trend — but we'd rather pull from a
    # FitnessMetric snapshot. For now compare the LAST ride within the
    # horizon window vs the FIRST one to estimate fitness drift.
    rides_in_horizon = [
        a for a in activities_after
        if _ensure_aware(a.date) <= horizon_end
    ]
    actual_ftp_delta_w: float | None = None
    if len(rides_in_horizon) >= 2:
        # Use NP-derived effective FTP proxy: median of the top-3 NPs.
        nps = sorted(
            [float(a.normalized_power) for a in rides_in_horizon if a.normalized_power],
            reverse=True,
        )
        if len(nps) >= 3:
            recent_proxy = sum(nps[:3]) / 3.0
            # Compare to the same metric over the FIRST week of the horizon.
            week1 = [
                a for a in rides_in_horizon
                if _ensure_aware(a.date) <= rec_at + timedelta(days=7)
                and a.normalized_power
            ]
            week1_nps = sorted([float(a.normalized_power) for a in week1], reverse=True)
            if len(week1_nps) >= 2:
                old_proxy = sum(week1_nps[: min(3, len(week1_nps))]) / min(3, len(week1_nps))
                actual_ftp_delta_w = recent_proxy - old_proxy

    outcome = PredictionOutcome(
        recommendation_id=rec.id,
        user_id=rec.user_id,
        predicted_workout_type=pred["workout_type"],
        predicted_duration_min=pred["duration_min"],
        predicted_intensity=pred["intensity"],
        predicted_ftp_delta_w=pred["ftp_delta_w"],
        actual_workout_type=actual_workout_type,
        actual_duration_min=actual_duration_min,
        actual_intensity=actual_intensity,
        actual_ftp_delta_w=actual_ftp_delta_w,
        horizon_days=DEFAULT_HORIZON_DAYS,
        workout_type_match=(
            actual_workout_type is not None
            and pred["workout_type"] is not None
            and actual_workout_type == pred["workout_type"]
        ),
        duration_abs_err_min=(
            abs(actual_duration_min - pred["duration_min"])
            if actual_duration_min is not None and pred["duration_min"] is not None
            else None
        ),
        ftp_delta_abs_err_w=(
            abs(actual_ftp_delta_w - pred["ftp_delta_w"])
            if actual_ftp_delta_w is not None and pred["ftp_delta_w"] is not None
            else None
        ),
    )
    db.add(outcome)
    return outcome


async def backfill_user_outcomes(
    user_id: str,
    db: AsyncSession,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> int:
    """Score every mature recommendation for a single user.

    A recommendation is "mature" when:
      - generated_at is older than `horizon_days` ago
      - no PredictionOutcome row exists for it yet

    Returns the number of new outcome rows written.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=horizon_days)

    # All mature recommendations for this user.
    rec_rows = (await db.execute(
        select(Recommendation).where(
            and_(
                Recommendation.user_id == user_id,
                Recommendation.generated_at <= cutoff,
            )
        )
    )).scalars().all()
    if not rec_rows:
        return 0

    # Already-scored set (subquery is overkill at this scale).
    done = set((await db.execute(
        select(PredictionOutcome.recommendation_id).where(
            PredictionOutcome.user_id == user_id
        )
    )).scalars().all())

    pending = [r for r in rec_rows if r.id not in done]
    if not pending:
        return 0

    # Pull this user's confirmed activities once.
    acts = (await db.execute(
        select(Activity)
        .where(
            Activity.user_id == user_id,
            Activity.review_status == "confirmed",
        )
        .order_by(Activity.date)
    )).scalars().all()

    written = 0
    for rec in pending:
        rec_at = _ensure_aware(rec.generated_at)
        horizon_end = rec_at + timedelta(days=horizon_days)
        after = [a for a in acts if _ensure_aware(a.date) > rec_at]
        outcome = await _backfill_one(rec, after, horizon_end, db)
        if outcome is not None:
            written += 1

    if written:
        await db.flush()
    return written


async def backfill_all_outcomes(db: AsyncSession) -> dict[str, int]:
    """Worker entrypoint: score every user's mature recommendations.

    Returns {user_id: rows_written}. Designed for a daily celery beat job.
    """
    user_ids = (await db.execute(
        select(Recommendation.user_id).distinct()
    )).scalars().all()
    out: dict[str, int] = {}
    for uid in user_ids:
        out[uid] = await backfill_user_outcomes(uid, db)
    return out
