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
    """Fetch all Strava cycling activities and upsert into DB."""
    from sqlalchemy import select as sa_select
    from app.services.metrics_service import compute_activity_metrics, _score_and_tag

    token = await _refresh_token_if_needed(user, db)

    # _refresh_token_if_needed flushes the user row, which expires ORM attributes.
    # Re-fetch user+profile so compute_activity_metrics can access user.profile
    # synchronously (without triggering a lazy load that would fail outside greenlet).
    from sqlalchemy import select as sa_select_user
    from sqlalchemy.orm import selectinload as _selectinload
    _result = await db.execute(
        sa_select_user(User).where(User.id == user.id).options(
            _selectinload(User.profile)
        )
    )
    user = _result.scalar_one()

    headers = {"Authorization": f"Bearer {token}"}

    page = 1
    per_page = 200
    total_fetched = 0
    all_activities: list[dict] = []

    timeout = aiohttp.ClientTimeout(total=60)

    # Paginate through all activities
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        while True:
            async with session.get(
                f"{STRAVA_API_BASE}/athlete/activities",
                params={"page": page, "per_page": per_page},
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"Strava API error {resp.status}: {text[:200]}")
                batch = await resp.json()

            # Strava returns an error dict on auth failure — must be a list
            if not isinstance(batch, list):
                raise RuntimeError(f"Unexpected Strava response: {batch}")

            if not batch:
                break

            all_activities.extend(batch)
            total_fetched = len(all_activities)

            if progress_callback:
                progress_callback(total_fetched, total_fetched)  # total unknown until done

            if len(batch) < per_page:
                break
            page += 1

    # Upsert — only process Ride/VirtualRide sport types
    ride_types = {"Ride", "VirtualRide", "MountainBikeRide", "GravelRide", "EBikeRide"}
    cycling = [a for a in all_activities if a.get("sport_type") in ride_types or a.get("type") in ride_types]
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
            activity = _strava_to_activity(sa, user.id)
            compute_activity_metrics(activity, user)
            _score_and_tag(activity, user)
            db.add(activity)
            inserted += 1

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


def _strava_to_activity(sa: dict[str, Any], user_id: str) -> Activity:
    from dateutil import parser as dp  # type: ignore

    date = dp.parse(sa["start_date"])

    avg_power = sa.get("average_watts")
    np_ = sa.get("weighted_average_watts")
    ftp_guess = 200  # fallback; real FTP loaded from profile during metrics computation

    tss = None
    if avg_power and np_:
        # TSS = (duration_s × NP × IF) / (FTP × 3600) × 100
        # IF = NP / FTP  (rough, profile FTP used in metrics_service)
        pass

    return Activity(
        user_id=user_id,
        external_id=str(sa["id"]),
        source="strava",
        name=sa.get("name", "Strava Ride"),
        date=date,
        duration_seconds=sa.get("moving_time", 0),
        distance_meters=sa.get("distance", 0),
        elevation_gain_meters=sa.get("total_elevation_gain"),
        avg_power=avg_power,
        max_power=sa.get("max_watts"),
        normalized_power=np_,
        avg_hr=sa.get("average_heartrate"),
        max_hr=sa.get("max_heartrate"),
        avg_cadence=sa.get("average_cadence"),
        trainer=sa.get("trainer", False),
        kudos_count=sa.get("kudos_count"),
    )
