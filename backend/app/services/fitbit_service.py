from __future__ import annotations

import base64
import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Callable

import httpx
from sqlalchemy import select as sa_select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.crypto import encrypt_str, decrypt_str
from app.models.user import User
from app.models.health import HealthMetric

log = logging.getLogger(__name__)

_AUTH_URL = "https://www.fitbit.com/oauth2/authorize"
_TOKEN_URL = "https://api.fitbit.com/oauth2/token"
_BASE = "https://api.fitbit.com"
_SCOPES = "sleep heartrate"


def get_auth_url(user_id: str) -> str:
    params = (
        f"response_type=code"
        f"&client_id={settings.FITBIT_CLIENT_ID}"
        f"&redirect_uri={settings.FITBIT_REDIRECT_URI}"
        f"&scope={_SCOPES.replace(' ', '%20')}"
        f"&state={user_id}"
    )
    return f"{_AUTH_URL}?{params}"


async def _token_request(payload: dict) -> dict:
    credentials = base64.b64encode(
        f"{settings.FITBIT_CLIENT_ID}:{settings.FITBIT_CLIENT_SECRET}".encode()
    ).decode()
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            _TOKEN_URL,
            data=payload,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        r.raise_for_status()
        return r.json()


async def exchange_code(code: str, state: str, db: AsyncSession) -> str:
    """Exchange OAuth code → tokens, persist to user identified by state (user_id). Returns user_id."""
    token_data = await _token_request({
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": settings.FITBIT_REDIRECT_URI,
    })

    res = await db.execute(sa_select(User).where(User.id == state))
    user = res.scalar_one_or_none()
    if not user:
        raise RuntimeError(f"No user found for state={state}")

    expires_at = datetime.now(timezone.utc) + timedelta(seconds=token_data.get("expires_in", 28800))
    user.fitbit_access_token_enc = encrypt_str(token_data["access_token"])
    user.fitbit_refresh_token_enc = encrypt_str(token_data["refresh_token"])
    user.fitbit_user_id = token_data.get("user_id")
    user.fitbit_token_expires_at = expires_at
    db.add(user)
    await db.flush()
    return state


async def disconnect(user: User, db: AsyncSession) -> None:
    user.fitbit_access_token_enc = None
    user.fitbit_refresh_token_enc = None
    user.fitbit_user_id = None
    user.fitbit_token_expires_at = None
    db.add(user)
    await db.flush()


async def _ensure_fresh_token(user: User, db: AsyncSession) -> str:
    """Return a valid access token, refreshing if it expires within 5 minutes."""
    if not user.fitbit_access_token_enc:
        raise RuntimeError("Fitbit not connected")

    now = datetime.now(timezone.utc)
    expires_at = user.fitbit_token_expires_at
    # Refresh if missing expiry or expiring soon
    if expires_at is None or (expires_at - now).total_seconds() < 300:
        if not user.fitbit_refresh_token_enc:
            raise RuntimeError("Fitbit refresh token missing")
        refresh_token = decrypt_str(user.fitbit_refresh_token_enc)
        if not refresh_token:
            raise RuntimeError("Cannot decrypt Fitbit refresh token")

        token_data = await _token_request({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        })
        new_expires = now + timedelta(seconds=token_data.get("expires_in", 28800))
        user.fitbit_access_token_enc = encrypt_str(token_data["access_token"])
        user.fitbit_refresh_token_enc = encrypt_str(token_data["refresh_token"])
        user.fitbit_token_expires_at = new_expires
        db.add(user)
        await db.flush()

    token = decrypt_str(user.fitbit_access_token_enc)
    if not token:
        raise RuntimeError("Cannot decrypt Fitbit access token")
    return token


async def _get(token: str, path: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{_BASE}{path}",
            headers={"Authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()


async def sync_fitbit(
    user: User,
    db: AsyncSession,
    days: int = 30,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    stats = {"health_days_added": 0, "health_days_updated": 0, "errors": 0}

    token = await _ensure_fresh_token(user, db)

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    window_start = today - timedelta(days=days)

    existing_result = await db.execute(
        sa_select(HealthMetric).where(
            HealthMetric.user_id == user.id,
            HealthMetric.date >= window_start,
        )
    )
    existing_map: dict[datetime, HealthMetric] = {
        row.date: row for row in existing_result.scalars().all()
    }

    total = days
    if progress_callback:
        progress_callback(0, total)

    for d in range(days):
        day = today - timedelta(days=d)
        day_str = day.strftime("%Y-%m-%d")
        try:
            payload: dict = {}

            try:
                sleep_data = await _get(token, f"/1/user/-/sleep/date/{day_str}.json")
                summary = sleep_data.get("summary", {})
                total_min = summary.get("totalMinutesAsleep")
                if total_min is not None:
                    payload["sleep_total_seconds"] = total_min * 60
                stages = summary.get("stages", {})
                if stages.get("deep") is not None:
                    payload["sleep_deep_seconds"] = stages["deep"] * 60
                if stages.get("light") is not None:
                    payload["sleep_light_seconds"] = stages["light"] * 60
                if stages.get("rem") is not None:
                    payload["sleep_rem_seconds"] = stages["rem"] * 60
                if stages.get("wake") is not None:
                    payload["sleep_awake_seconds"] = stages["wake"] * 60
                efficiency = summary.get("efficiency")
                if efficiency is not None:
                    payload["sleep_score"] = efficiency
            except Exception as exc:
                log.debug("fitbit sleep %s failed: %s", day_str, exc)

            try:
                hr_data = await _get(token, f"/1/user/-/activities/heart/date/{day_str}/1d.json")
                heart_entries = hr_data.get("activities-heart", [])
                if heart_entries:
                    rhr = heart_entries[0].get("value", {}).get("restingHeartRate")
                    if rhr is not None:
                        payload["resting_hr"] = rhr
            except Exception as exc:
                log.debug("fitbit hr %s failed: %s", day_str, exc)

            try:
                hrv_data = await _get(token, f"/1/user/-/hrv/date/{day_str}.json")
                hrv_entries = hrv_data.get("hrv", [])
                if hrv_entries:
                    rmssd = hrv_entries[0].get("value", {}).get("dailyRmssd")
                    if rmssd is not None:
                        payload["hrv_overnight_avg_ms"] = rmssd
            except Exception as exc:
                # HRV requires Fitbit Premium; skip silently
                log.debug("fitbit hrv %s skipped: %s", day_str, exc)

            if not payload:
                continue

            row = existing_map.get(day)
            if row is None:
                row = HealthMetric(user_id=user.id, date=day, source="fitbit", **payload)
                db.add(row)
                existing_map[day] = row
                stats["health_days_added"] += 1
            else:
                for k, v in payload.items():
                    if v is not None:
                        setattr(row, k, v)
                db.add(row)
                stats["health_days_updated"] += 1
        except Exception as exc:
            log.exception("fitbit day %s failed: %s", day_str, exc)
            stats["errors"] += 1

        if progress_callback and (d % 5 == 0 or d == total - 1):
            progress_callback(d + 1, total)

    await db.flush()
    return stats
