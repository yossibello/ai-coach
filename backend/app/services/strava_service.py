"""
Strava OAuth + API integration.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
import aiohttp

from app.core.config import settings
from app.models.user import User
from app.models.activity import Activity

STRAVA_AUTH_URL  = "https://www.strava.com/oauth/authorize"
STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_API_BASE  = "https://www.strava.com/api/v3"

SCOPE = "read,activity:read_all"


def _extract_power_curve(watts: list, weight_kg: float) -> dict[str, float]:
    """Compute best mean power over standard windows (W/kg) from a power stream."""
    import numpy as np
    arr = np.array(watts, dtype=np.float32)
    n = len(arr)
    result: dict[str, float] = {}
    for window_s, key in [
        (5,    "pc_5s_wkg"),
        (60,   "pc_1min_wkg"),
        (300,  "pc_5min_wkg"),
        (1200, "pc_20min_wkg"),
    ]:
        if n < window_s:
            result[key] = 0.0
            continue
        cs = np.concatenate([[0.0], np.cumsum(arr)])
        sums = cs[window_s:] - cs[:-window_s]
        best_w = float(np.max(sums)) / window_s
        result[key] = round(best_w / weight_kg, 4)
    return result


async def _fetch_power_curve(
    session: "aiohttp.ClientSession",
    activity_id: int,
    weight_kg: float,
) -> dict[str, float]:
    """Fetch Strava power stream for one activity and return W/kg power curve.

    Returns empty dict on any error (404 = no power data, 429 = rate limited, etc.)
    so the caller can safely proceed with pc values defaulting to 0.
    """
    url = f"{STRAVA_API_BASE}/activities/{activity_id}/streams"
    try:
        async with session.get(
            url, params={"keys": "watts", "key_by_type": "true"}
        ) as resp:
            if resp.status != 200:
                return {}
            data = await resp.json()
            # Response is {watts: {data: [...], ...}} with key_by_type=true
            watts = data.get("watts", {}).get("data", [])
            if not watts:
                return {}
            return _extract_power_curve(watts, weight_kg)
    except Exception:
        return {}


def get_strava_auth_url() -> str:
    params = (
        f"client_id={settings.STRAVA_CLIENT_ID}"
        f"&redirect_uri={settings.STRAVA_REDIRECT_URI}"
        f"&response_type=code"
        f"&approval_prompt=auto"
        f"&scope={SCOPE}"
    )
    return f"{STRAVA_AUTH_URL}?{params}"


async def exchange_strava_code(user: User, code: str, db) -> None:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            STRAVA_TOKEN_URL,
            data={
                "client_id": settings.STRAVA_CLIENT_ID,
                "client_secret": settings.STRAVA_CLIENT_SECRET,
                "code": code,
                "grant_type": "authorization_code",
            },
        ) as resp:
            data = await resp.json()
            if "errors" in data or "error" in data:
                raise ValueError(data.get("message", "Strava auth failed"))

    user.strava_athlete_id = str(data["athlete"]["id"])
    user.strava_access_token = data["access_token"]
    user.strava_refresh_token = data["refresh_token"]
    user.strava_token_expires_at = datetime.fromtimestamp(
        data["expires_at"], tz=timezone.utc
    )
    db.add(user)
    await db.flush()


async def _refresh_token_if_needed(user: User, db) -> str:
    """Return a valid access token, refreshing if expired."""
    expires = user.strava_token_expires_at
    if expires is not None:
        # SQLite returns naive datetimes; make them UTC-aware for comparison
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires > datetime.now(timezone.utc):
            return user.strava_access_token  # type: ignore

    async with aiohttp.ClientSession() as session:
        async with session.post(
            STRAVA_TOKEN_URL,
            data={
                "client_id": settings.STRAVA_CLIENT_ID,
                "client_secret": settings.STRAVA_CLIENT_SECRET,
                "refresh_token": user.strava_refresh_token,
                "grant_type": "refresh_token",
            },
        ) as resp:
            data = await resp.json()

    user.strava_access_token = data["access_token"]
    user.strava_refresh_token = data["refresh_token"]
    user.strava_token_expires_at = datetime.fromtimestamp(data["expires_at"], tz=timezone.utc)
    db.add(user)
    await db.flush()
    return user.strava_access_token  # type: ignore


async def sync_strava_history(
    user: User,
    db,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Fetch Strava cycling activities and upsert into DB.

    Incremental: if the user already has Strava activities in the DB, only
    activities newer than the most-recently-synced one are fetched (using
    Strava's ``after`` timestamp filter).  A full backfill is performed on
    first sync (no existing activities).
    """
    from sqlalchemy import select as sa_select, func as sa_func
    from app.services.metrics_service import compute_activity_metrics, _score_and_tag

    token = await _refresh_token_if_needed(user, db)

    # Re-fetch user+profile after token refresh flushes/expires ORM attributes.
    from sqlalchemy import select as sa_select_user
    from sqlalchemy.orm import selectinload as _selectinload
    _result = await db.execute(
        sa_select_user(User).where(User.id == user.id).options(
            _selectinload(User.profile)
        )
    )
    user = _result.scalar_one()

    # ── Determine incremental cutoff ──────────────────────────────────────
    # Find the most recent Strava activity already in the DB so we can pass
    # ``after=<unix_ts>`` to the API and skip already-synced rides.
    latest_result = await db.execute(
        sa_select(sa_func.max(Activity.date)).where(
            Activity.user_id == user.id,
            Activity.source == "strava",
        )
    )
    latest_date = latest_result.scalar_one_or_none()

    import calendar as _cal
    # Use the timestamp of the most-recent activity (minus 1 day overlap to
    # catch same-day edits), or None for a full backfill on first sync.
    after_ts: int | None = None
    if latest_date is not None:
        from datetime import timedelta
        cutoff = latest_date - timedelta(days=1)
        after_ts = int(_cal.timegm(cutoff.timetuple()))

    headers = {"Authorization": f"Bearer {token}"}

    page = 1
    per_page = 200
    all_activities: list[dict] = []

    timeout = aiohttp.ClientTimeout(total=60)
    import asyncio

    is_full_sync = after_ts is None

    # Weight needed to convert peak watts → W/kg for the power curve
    weight_kg: float | None = None
    if user.profile and user.profile.weight_kg and user.profile.weight_kg > 0:
        weight_kg = float(user.profile.weight_kg)

    # ── Paginate Strava API + process activities within the same session ──
    ride_types = {"Ride", "VirtualRide", "MountainBikeRide", "GravelRide", "EBikeRide"}

    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        # Pagination
        while True:
            params: dict = {"page": page, "per_page": per_page}
            if after_ts is not None:
                params["after"] = after_ts

            # Retry up to 3 times on transient 5xx errors
            batch = None
            for attempt in range(3):
                async with session.get(
                    f"{STRAVA_API_BASE}/athlete/activities",
                    params=params,
                ) as resp:
                    if resp.status == 200:
                        batch = await resp.json()
                        break
                    if resp.status in (429, 500, 502, 503, 504) and attempt < 2:
                        await asyncio.sleep(5 * (attempt + 1))
                        continue
                    text = await resp.text()
                    raise RuntimeError(f"Strava API error {resp.status}: {text[:200]}")

            # Strava returns an error dict on auth failure — must be a list
            if not isinstance(batch, list):
                raise RuntimeError(f"Unexpected Strava response: {batch}")

            if not batch:
                break

            all_activities.extend(batch)

            # During pagination, total is unknown — show fetched count only
            if progress_callback:
                progress_callback(len(all_activities), 0)

            if len(batch) < per_page:
                break
            page += 1

        # ── Filter to cycling types ───────────────────────────────────────
        cycling = [
            a for a in all_activities
            if a.get("sport_type") in ride_types or a.get("type") in ride_types
        ]
        total_cycling = len(cycling)

        if progress_callback:
            progress_callback(0, total_cycling)

        inserted = 0
        for i, sa in enumerate(cycling):
            ext_id = str(sa["id"])
            result = await db.execute(
                sa_select(Activity).where(
                    Activity.user_id == user.id,
                    Activity.external_id == ext_id,
                )
            )
            existing = result.scalar_one_or_none()
            if not existing:
                # Fetch power stream only on incremental syncs (≤ 50 new rides).
                # Full backfills skip this to avoid hitting Strava's 200 req/15min
                # rate limit — a separate /backfill-power-curves endpoint handles it.
                pc_curve: dict[str, float] = {}
                if not is_full_sync and weight_kg and sa.get("device_watts"):
                    pc_curve = await _fetch_power_curve(session, sa["id"], weight_kg)

                activity = _strava_to_activity(sa, user.id, pc_curve)
                compute_activity_metrics(activity, user)
                _score_and_tag(activity, user)
                db.add(activity)
                inserted += 1

                # Auto-detect FTP tests (e.g. Zwift "FTP Bike Test")
                from app.services.metrics_service import detect_ftp_test
                from app.models.tracking import PerformanceTest
                test_type = detect_ftp_test(activity.name, activity.workout_type)
                if test_type and activity.normalized_power:
                    ftp_val: float
                    if activity.pc_20min_wkg and weight_kg:
                        ftp_val = activity.pc_20min_wkg * weight_kg * 0.95
                    else:
                        ftp_val = float(activity.normalized_power) * 0.95
                    db.add(PerformanceTest(
                        user_id=user.id,
                        test_date=activity.date,
                        test_type=test_type,
                        value=round(ftp_val, 1),
                        unit="W",
                        source="auto_detected",
                        notes=f"Auto-detected from Strava: {activity.name}",
                    ))

            if progress_callback:
                progress_callback(i + 1, total_cycling)

    await db.flush()

    # Rebuild Performance Management Chart (CTL/ATL/TSB) from all activities
    from app.services.metrics_service import compute_pmc_for_user
    await compute_pmc_for_user(user.id, db)

    # Force-refresh the coach recommendation so it uses the new activity data
    from app.ml.inference import generate_recommendation
    from app.models.recommendation import Recommendation
    try:
        rec = await generate_recommendation(user, db)
        db.add(rec)
        await db.flush()
    except Exception:
        pass  # recommendation refresh is best-effort; don't fail the sync


async def backfill_power_curves(
    user: User,
    db,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    """Fetch Strava power streams for existing activities missing pc_* values.

    Rate-limited to 1 request per 6 seconds (~10 req/min, well under Strava's
    200/15min limit). Returns the number of activities updated.
    """
    import asyncio
    from sqlalchemy import select as sa_select

    weight_kg: float | None = None
    if user.profile and user.profile.weight_kg and user.profile.weight_kg > 0:
        weight_kg = float(user.profile.weight_kg)
    if not weight_kg:
        return 0

    # Only activities with a real power meter (device_watts=True) that are missing curves.
    # Strava can show avg_power for rides with estimated power but won't have stored streams
    # for those — device_watts=True is the reliable signal that streams exist.
    result = await db.execute(
        sa_select(Activity)
        .where(
            Activity.user_id == user.id,
            Activity.device_watts == True,  # noqa: E712
            Activity.pc_5min_wkg.is_(None),
            Activity.external_id.isnot(None),
            Activity.source == "strava",
        )
        .order_by(Activity.date.desc())
    )
    activities = list(result.scalars().all())
    total = len(activities)

    if not activities:
        return 0

    token = await _refresh_token_if_needed(user, db)
    headers = {"Authorization": f"Bearer {token}"}
    timeout = aiohttp.ClientTimeout(total=30)

    updated = 0
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        for i, act in enumerate(activities):
            if progress_callback:
                progress_callback(i, total)

            pc = await _fetch_power_curve(session, int(act.external_id), weight_kg)
            if pc and any(v for v in pc.values()):
                act.pc_5s_wkg    = pc.get("pc_5s_wkg") or None
                act.pc_1min_wkg  = pc.get("pc_1min_wkg") or None
                act.pc_5min_wkg  = pc.get("pc_5min_wkg") or None
                act.pc_20min_wkg = pc.get("pc_20min_wkg") or None
                db.add(act)
                updated += 1

            # Flush every 20 to avoid huge pending transactions
            if (i + 1) % 20 == 0:
                await db.flush()

            # 6s between requests → ~10 req/min → safe under Strava's 200/15min
            await asyncio.sleep(6.0)

    await db.flush()

    if progress_callback:
        progress_callback(total, total)

    return updated


def _strava_to_activity(
    sa: dict[str, Any],
    user_id: str,
    pc_curve: dict[str, float] | None = None,
) -> Activity:
    from dateutil import parser as dp  # type: ignore

    # Use start_date_local (athlete's local time) so calendar dates match what
    # the rider sees in Strava. Strava formats it as ISO 8601 with a spurious Z
    # suffix that doesn't mean UTC — strip tzinfo so it's stored as local naive.
    raw_date = sa.get("start_date_local") or sa["start_date"]
    date = dp.parse(raw_date).replace(tzinfo=None)
    pc = pc_curve or {}

    return Activity(
        user_id=user_id,
        external_id=str(sa["id"]),
        source="strava",
        name=sa.get("name", "Strava Ride"),
        date=date,
        duration_seconds=sa.get("moving_time", 0),
        distance_meters=sa.get("distance", 0),
        elevation_gain_meters=sa.get("total_elevation_gain"),
        avg_power=sa.get("average_watts"),
        max_power=sa.get("max_watts"),
        normalized_power=sa.get("weighted_average_watts"),
        avg_hr=sa.get("average_heartrate"),
        max_hr=sa.get("max_heartrate"),
        avg_cadence=sa.get("average_cadence"),
        trainer=sa.get("trainer", False),
        kudos_count=sa.get("kudos_count"),
        device_watts=sa.get("device_watts", False) or False,
        pc_5s_wkg=pc.get("pc_5s_wkg") or None,
        pc_1min_wkg=pc.get("pc_1min_wkg") or None,
        pc_5min_wkg=pc.get("pc_5min_wkg") or None,
        pc_20min_wkg=pc.get("pc_20min_wkg") or None,
    )
