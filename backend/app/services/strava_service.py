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
    if user.strava_token_expires_at and user.strava_token_expires_at > datetime.now(timezone.utc):
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
    headers = {"Authorization": f"Bearer {token}"}

    page = 1
    per_page = 200
    total_fetched = 0
    all_activities: list[dict] = []

    # Paginate through all activities
    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            async with session.get(
                f"{STRAVA_API_BASE}/athlete/activities",
                params={"page": page, "per_page": per_page, "type": "Ride"},
            ) as resp:
                batch = await resp.json()

            if not batch:
                break

            all_activities.extend(batch)
            total_fetched = len(all_activities)

            if progress_callback:
                progress_callback(total_fetched, total_fetched)  # total unknown until done

            if len(batch) < per_page:
                break
            page += 1

    # Upsert
    for i, sa in enumerate(all_activities):
        if progress_callback:
            progress_callback(i, total_fetched)

        ext_id = str(sa["id"])
        result = await db.execute(
            sa_select(Activity).where(
                Activity.user_id == user.id,
                Activity.external_id == ext_id,
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            continue  # Skip duplicates

        activity = _strava_to_activity(sa, user.id)
        compute_activity_metrics(activity, user)
        _score_and_tag(activity, user)
        db.add(activity)

    await db.flush()


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
