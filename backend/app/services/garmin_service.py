"""
Garmin integration.

Two paths are supported:

* **Dev path (`garminconnect` library)** — username/password.  Works today,
  no Garmin partner agreement required.  Unsupported by Garmin, may break.
* **Prod path (Garmin Health/Activity OAuth)** — stub functions raise
  NotImplementedError until partner credentials are wired in.

Public surface:
    connect_with_credentials(user, username, password, db)
    disconnect(user, db)
    sync_garmin(user, db, days=30)             — activities + health + dedup
    get_oauth_authorize_url() / exchange_oauth_code()  — placeholders

The sync flow:
    1. Pull activity list for the last *days* days.
    2. For each ride, compute fields, save (skip if external_id already known).
    3. Run dedup against Strava activities (start_time ± 5 min, duration ± 2 min).
       Garmin wins → matching Strava activity is soft-deleted (review_status='deleted').
    4. Pull daily wellness (sleep, HRV, RHR, body battery, stress) → upsert HealthMetric.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from sqlalchemy import select as sa_select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.crypto import encrypt_str, decrypt_str
from app.models.user import User
from app.models.activity import Activity
from app.models.health import HealthMetric
from app.services.metrics_service import compute_activity_metrics
from app.services.recommendation_linking import latest_recommendation_id

log = logging.getLogger(__name__)

# Minimum ride length to import from Garmin (filters warmup segments, auto-pauses, etc.)
MIN_RIDE_DURATION_S = 120            # 2 minutes
# Duration tolerance for same-day dedup: absolute floor OR relative %, whichever is larger
DEDUP_DURATION_ABS_S = 10 * 60      # ±10 min absolute
DEDUP_DURATION_REL = 0.25           # ±25% relative


# ─── Dev path: garminconnect ──────────────────────────────────────────────────

def _build_garmin_client(username: str, password: str, tokenstore: str | None = None):
    """Lazy import + login. Pass tokenstore to skip username/password re-auth."""
    try:
        from garminconnect import Garmin
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "garminconnect not installed. Add `garminconnect` to requirements.txt."
        ) from exc
    client = Garmin(username, password)
    try:
        client.login(tokenstore)  # None → fresh login; string → restore session
    except Exception as exc:
        if tokenstore:
            raise  # let caller handle token expiry and retry with password
        msg = str(exc)
        if "TOO_MANY_REQUESTS" in msg or "429" in msg or "TooManyRequests" in type(exc).__name__:
            raise RuntimeError(
                "Garmin Connect rate-limited this account. "
                "This can last several hours — try again later. "
                "If this keeps happening, log out of Garmin Connect on all devices and wait 1 hour."
            ) from exc
        if any(k in msg.lower() for k in ("auth", "invalid", "credentials", "password", "401", "403")):
            raise RuntimeError(
                "Garmin Connect login failed — check your username and password in Profile → Integrations."
            ) from exc
        if any(k in msg.lower() for k in ("timeout", "connection", "network", "ssl")):
            raise RuntimeError(
                "Could not reach Garmin Connect servers. Check your internet connection and retry."
            ) from exc
        raise RuntimeError(f"Garmin Connect error: {msg}") from exc
    return client


async def connect_with_credentials(
    user: User, username: str, password: str, db: AsyncSession
) -> None:
    """Validate credentials with Garmin Connect and persist (encrypted).

    Tries the existing cached token first to avoid Garmin's aggressive
    login rate-limiting. Falls back to password login only if needed.
    """
    client = None

    # Reuse cached session if the username matches — avoids rate limit on /connect
    if user.garmin_token_store and user.garmin_username == username:
        tokenstore = decrypt_str(user.garmin_token_store)
        if tokenstore:
            try:
                client = await asyncio.to_thread(_build_garmin_client, username, password, tokenstore)
                log.info("Garmin re-connect: reused cached token for %s", username)
            except Exception:
                log.info("Garmin cached token invalid for %s, doing fresh login", username)
                client = None

    if client is None:
        client = await asyncio.to_thread(_build_garmin_client, username, password)

    user.garmin_username = username
    user.garmin_password_enc = encrypt_str(password)
    user.garmin_user_id = str(getattr(client, "display_name", "") or username)
    user.garmin_token_store = encrypt_str(client.garth.dumps())
    db.add(user)
    await db.flush()


async def disconnect(user: User, db: AsyncSession) -> None:
    user.garmin_username = None
    user.garmin_password_enc = None
    user.garmin_user_id = None
    user.garmin_access_token = None
    user.garmin_refresh_token = None
    user.garmin_token_store = None
    db.add(user)
    await db.flush()


async def _client_for(user: User, db: AsyncSession):
    """Return authenticated Garmin client, preferring cached tokens over password re-auth."""
    if not user.garmin_username or not user.garmin_password_enc:
        raise RuntimeError("Garmin Connect credentials not configured for user")
    username = user.garmin_username
    pwd = decrypt_str(user.garmin_password_enc)
    if pwd is None:
        raise RuntimeError("Cannot decrypt Garmin credentials (SECRET_KEY changed?)")

    # Try cached token store first — avoids triggering rate limits
    if user.garmin_token_store:
        tokenstore = decrypt_str(user.garmin_token_store)
        try:
            client = await asyncio.to_thread(_build_garmin_client, username, pwd, tokenstore)
            return client
        except Exception:
            log.info("Garmin cached token expired for %s, falling back to password login", username)

    # Password login + save fresh tokens
    client = await asyncio.to_thread(_build_garmin_client, username, pwd)
    user.garmin_token_store = encrypt_str(client.garth.dumps())
    db.add(user)
    await db.flush()
    return client


# ─── Prod path: OAuth shell (placeholders) ────────────────────────────────────

def get_oauth_authorize_url() -> str:
    raise NotImplementedError(
        "Garmin Health API OAuth requires a partner agreement. "
        "Configure GARMIN_OAUTH_CLIENT_ID / SECRET to enable."
    )


async def exchange_oauth_code(user: User, code: str, db: AsyncSession) -> None:
    raise NotImplementedError("Garmin OAuth not yet provisioned.")


# ─── Sync activities ──────────────────────────────────────────────────────────

async def _fetch_activities(client, days: int) -> list[dict]:
    """Wrapper for the blocking garminconnect call."""
    return await asyncio.to_thread(client.get_activities_by_date,
                                   (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d"),
                                   datetime.utcnow().strftime("%Y-%m-%d"),
                                   "cycling")


async def _fetch_daily_wellness(client, day: datetime) -> dict[str, Any]:
    """Pull sleep + HRV + RHR + body battery + stress for one calendar day."""
    iso = day.strftime("%Y-%m-%d")
    out: dict[str, Any] = {}

    def _safe(fn, *args):
        try:
            return fn(*args)
        except Exception as exc:  # pragma: no cover
            log.warning("garmin %s failed for %s: %s", fn.__name__, iso, exc)
            return None

    sleep = await asyncio.to_thread(_safe, client.get_sleep_data, iso)
    hrv = await asyncio.to_thread(_safe, client.get_hrv_data, iso)
    rhr = await asyncio.to_thread(_safe, client.get_rhr_day, iso)
    bb = await asyncio.to_thread(_safe, client.get_body_battery, iso)
    stress = await asyncio.to_thread(_safe, client.get_stress_data, iso)

    if sleep:
        dto = sleep.get("dailySleepDTO", {}) if isinstance(sleep, dict) else {}
        out["sleep_total_seconds"] = dto.get("sleepTimeSeconds")
        out["sleep_deep_seconds"] = dto.get("deepSleepSeconds")
        out["sleep_light_seconds"] = dto.get("lightSleepSeconds")
        out["sleep_rem_seconds"] = dto.get("remSleepSeconds")
        out["sleep_awake_seconds"] = dto.get("awakeSleepSeconds")
        out["sleep_score"] = (dto.get("sleepScores") or {}).get("overall", {}).get("value") \
            if isinstance(dto.get("sleepScores"), dict) else None

    if hrv and isinstance(hrv, dict):
        summary = hrv.get("hrvSummary") or {}
        out["hrv_overnight_avg_ms"] = summary.get("lastNightAvg")
        out["hrv_7d_avg_ms"] = summary.get("weeklyAvg")
        out["hrv_status"] = (summary.get("status") or "").lower() or None

    if rhr and isinstance(rhr, dict):
        # garminconnect returns either {'allMetrics': {...}} or direct
        metrics = rhr.get("allMetrics") or rhr
        if isinstance(metrics, dict):
            vals = metrics.get("metricsMap", {}).get("WELLNESS_RESTING_HEART_RATE")
            if isinstance(vals, list) and vals:
                out["resting_hr"] = vals[0].get("value")

    if bb and isinstance(bb, list) and bb:
        first = bb[0]
        if isinstance(first, dict):
            out["body_battery_high"] = first.get("charged")
            out["body_battery_low"] = first.get("drained")

    if stress and isinstance(stress, dict):
        out["stress_avg"] = stress.get("avgStressLevel")
        out["stress_max"] = stress.get("maxStressLevel")

    return out


def _activity_payload_to_kwargs(g: dict) -> dict[str, Any]:
    """Convert a Garmin activity JSON blob to Activity-model kwargs."""
    start_str = g.get("startTimeGMT") or g.get("startTimeLocal")
    start: datetime | None = None
    if start_str:
        try:
            start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    return {
        "external_id": str(g.get("activityId")) if g.get("activityId") is not None else None,
        "source": "garmin",
        "name": g.get("activityName") or "Garmin Ride",
        "date": start,
        "duration_seconds": int(g.get("duration") or 0),
        "distance_meters": float(g.get("distance") or 0.0),
        "elevation_gain_meters": float(g.get("elevationGain") or 0.0),
        "avg_power": g.get("avgPower"),
        "max_power": g.get("maxPower"),
        "normalized_power": g.get("normPower"),
        "avg_hr": g.get("averageHR"),
        "max_hr": g.get("maxHR"),
        "avg_cadence": g.get("averageBikingCadenceInRevPerMinute"),
        "temperature_c": g.get("avgTemperature"),
        "trainer": bool(g.get("trainer")),
    }


def _durations_match(dur_a: int, dur_b: int) -> bool:
    """True when two durations are close enough to be the same ride.

    Uses the larger of an absolute floor (10 min) and a relative tolerance
    (25%) so short rides and very long rides are both handled correctly.
    Garmin reports elapsed time; Strava/GPX report moving time — they can
    differ by 10-15 % on rides with traffic stops.
    """
    diff = abs(dur_a - dur_b)
    rel_tol = max(dur_a, dur_b) * DEDUP_DURATION_REL
    return diff <= max(DEDUP_DURATION_ABS_S, rel_tol)


async def _find_duplicate_any_source(
    user_id: str, garmin_date: datetime, garmin_dur: int, db: AsyncSession
) -> Activity | None:
    """Find any non-Garmin activity on the same calendar day with similar duration.

    Matching is intentionally loose on start-time because:
    * Garmin may store local time, Strava stores UTC → can be hours apart.
    * Only calendar-day granularity is reliable across all sources.
    """
    if garmin_date is None or garmin_dur < MIN_RIDE_DURATION_S:
        return None

    # Use UTC date of the Garmin activity as the anchor day
    day_str = garmin_date.strftime("%Y-%m-%d")
    # Search the day before/after as well to handle UTC offset edge cases
    day_before = (garmin_date - timedelta(days=1)).strftime("%Y-%m-%d")
    day_after  = (garmin_date + timedelta(days=1)).strftime("%Y-%m-%d")

    from sqlalchemy import func as sa_func
    res = await db.execute(
        sa_select(Activity).where(
            Activity.user_id == user_id,
            Activity.source != "garmin",
            Activity.review_status != "deleted",
            sa_func.date(Activity.date).in_([day_before, day_str, day_after]),
        )
    )
    candidates = res.scalars().all()

    best: Activity | None = None
    best_diff = float("inf")
    for cand in candidates:
        cand_dur = cand.duration_seconds or 0
        if _durations_match(garmin_dur, cand_dur):
            diff = abs(cand_dur - garmin_dur)
            if diff < best_diff:
                best_diff = diff
                best = cand
    return best


async def _retroactive_dedup(user_id: str, db: AsyncSession) -> int:
    """Resolve duplicates that pre-date the current sync.

    For every Garmin activity find any non-deleted, non-Garmin activity on
    the same calendar day with a matching duration and mark it deleted.
    Returns the number of activities newly soft-deleted.
    """
    from sqlalchemy import func as sa_func

    garmin_res = await db.execute(
        sa_select(Activity).where(
            Activity.user_id == user_id,
            Activity.source == "garmin",
            Activity.review_status != "deleted",
        )
    )
    garmin_acts = garmin_res.scalars().all()

    resolved = 0
    for gact in garmin_acts:
        if not gact.date or not gact.duration_seconds:
            continue
        day_str = gact.date.strftime("%Y-%m-%d")
        day_before = (gact.date - timedelta(days=1)).strftime("%Y-%m-%d")
        day_after  = (gact.date + timedelta(days=1)).strftime("%Y-%m-%d")

        others_res = await db.execute(
            sa_select(Activity).where(
                Activity.user_id == user_id,
                Activity.source != "garmin",
                Activity.review_status != "deleted",
                sa_func.date(Activity.date).in_([day_before, day_str, day_after]),
            )
        )
        for cand in others_res.scalars().all():
            if _durations_match(gact.duration_seconds, cand.duration_seconds or 0):
                cand.review_status = "deleted"
                cand.notes = (cand.notes or "") + f"\n[deduped: replaced by Garmin activity {gact.external_id}]"
                db.add(cand)
                resolved += 1
                break  # one duplicate per Garmin activity

    return resolved


async def sync_garmin(
    user: User,
    db: AsyncSession,
    days: int = 30,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """
    Sync Garmin activities + daily wellness, deduping with Strava.

    Returns a stats dict: {activities_added, activities_updated, duplicates_resolved,
                           health_days_added, health_days_updated, errors}.
    """
    stats = {
        "activities_added": 0,
        "activities_updated": 0,
        "duplicates_resolved": 0,
        "health_days_added": 0,
        "health_days_updated": 0,
        "errors": 0,
    }
    client = await _client_for(user, db)

    # ── activities ────────────────────────────────────────────────────────────
    activities = await _fetch_activities(client, days)
    activities = activities or []
    total = len(activities)
    if progress_callback:
        progress_callback(0, total)

    for i, g in enumerate(activities, 1):
        try:
            kw = _activity_payload_to_kwargs(g)
            if not kw.get("date") or not kw.get("external_id"):
                continue

            # Skip tiny non-rides (warmup laps, auto-saved segments, etc.)
            if (kw.get("duration_seconds") or 0) < MIN_RIDE_DURATION_S:
                continue

            # Idempotency: skip if we already have this Garmin activity
            existing = await db.execute(
                sa_select(Activity).where(
                    Activity.user_id == user.id,
                    Activity.source == "garmin",
                    Activity.external_id == kw["external_id"],
                )
            )
            if existing.scalar_one_or_none():
                continue

            act = Activity(user_id=user.id, **kw)
            compute_activity_metrics(act, user)
            act.recommendation_id = await latest_recommendation_id(
                user.id, act.date, db
            )
            db.add(act)
            stats["activities_added"] += 1

            # Dedup: any matching ride from another source on the same day?
            dup = await _find_duplicate_any_source(user.id, kw["date"], kw["duration_seconds"], db)
            if dup is not None:
                dup.review_status = "deleted"
                dup.notes = (dup.notes or "") + f"\n[deduped: replaced by Garmin activity {kw['external_id']}]"
                db.add(dup)
                stats["duplicates_resolved"] += 1
        except Exception as exc:  # pragma: no cover
            log.exception("garmin activity import failed: %s", exc)
            stats["errors"] += 1

        if progress_callback and (i % 5 == 0 or i == total):
            progress_callback(i, total)

    await db.flush()

    # ── Retroactive dedup: clean up pairs that existed before this sync ───────
    retroactive = await _retroactive_dedup(user.id, db)
    stats["duplicates_resolved"] += retroactive

    # ── daily wellness for the same window ────────────────────────────────────
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    window_start = today - timedelta(days=days)

    # Load all existing health rows for the window in ONE query (avoids N+1).
    existing_result = await db.execute(
        sa_select(HealthMetric).where(
            HealthMetric.user_id == user.id,
            HealthMetric.date >= window_start,
        )
    )
    existing_map: dict = {row.date: row for row in existing_result.scalars().all()}

    for d in range(days):
        day = today - timedelta(days=d)
        try:
            payload = await _fetch_daily_wellness(client, day)
            if not payload:
                continue

            row = existing_map.get(day)
            if row is None:
                row = HealthMetric(user_id=user.id, date=day, source="garmin", **payload)
                db.add(row)
                existing_map[day] = row
                stats["health_days_added"] += 1
            else:
                for k, v in payload.items():
                    if v is not None:
                        setattr(row, k, v)
                db.add(row)
                stats["health_days_updated"] += 1
        except Exception as exc:  # pragma: no cover
            log.exception("garmin wellness import failed for %s: %s", day, exc)
            stats["errors"] += 1

    # Persist any token refresh that garth performed during the session
    try:
        user.garmin_token_store = encrypt_str(client.garth.dumps())
        db.add(user)
    except Exception:
        pass  # non-fatal: worst case next sync re-authenticates with password

    await db.flush()
    return stats
