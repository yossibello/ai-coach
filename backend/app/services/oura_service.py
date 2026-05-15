from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Callable

import httpx
from sqlalchemy import select as sa_select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.crypto import encrypt_str, decrypt_str
from app.models.user import User
from app.models.health import HealthMetric

log = logging.getLogger(__name__)

_BASE = "https://api.ouraring.com/v2"


async def _get(token: str, path: str, params: dict) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{_BASE}{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params,
        )
        r.raise_for_status()
        return r.json()


async def validate_and_save_token(user: User, token: str, db: AsyncSession) -> None:
    """Verify the token works by fetching one day of sleep, then persist encrypted."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    await _get(token, "/usercollection/daily_sleep", {"start_date": today, "end_date": today})
    user.oura_access_token_enc = encrypt_str(token)
    db.add(user)
    await db.flush()


async def disconnect(user: User, db: AsyncSession) -> None:
    user.oura_access_token_enc = None
    db.add(user)
    await db.flush()


async def sync_oura(
    user: User,
    db: AsyncSession,
    days: int = 30,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    stats = {"health_days_added": 0, "health_days_updated": 0, "errors": 0}

    if not user.oura_access_token_enc:
        raise RuntimeError("Oura not connected")
    token = decrypt_str(user.oura_access_token_enc)
    if not token:
        raise RuntimeError("Cannot decrypt Oura token (SECRET_KEY changed?)")

    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    window_start = today - timedelta(days=days)
    start_str = window_start.strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")
    params = {"start_date": start_str, "end_date": end_str}

    sleep_map: dict[str, dict] = {}
    readiness_map: dict[str, dict] = {}
    hrv_map: dict[str, dict] = {}

    try:
        data = await _get(token, "/usercollection/daily_sleep", params)
        for item in data.get("data", []):
            sleep_map[item["day"]] = item
    except Exception as exc:
        log.warning("oura sleep fetch failed: %s", exc)

    try:
        data = await _get(token, "/usercollection/daily_readiness", params)
        for item in data.get("data", []):
            readiness_map[item["day"]] = item
    except Exception as exc:
        log.warning("oura readiness fetch failed: %s", exc)

    try:
        data = await _get(token, "/usercollection/daily_hrv", params)
        for item in data.get("data", []):
            hrv_map[item["day"]] = item
    except Exception as exc:
        log.warning("oura hrv fetch failed: %s", exc)

    existing_result = await db.execute(
        sa_select(HealthMetric).where(
            HealthMetric.user_id == user.id,
            HealthMetric.date >= window_start,
        )
    )
    existing_map: dict[datetime, HealthMetric] = {
        row.date: row for row in existing_result.scalars().all()
    }

    all_days = set(sleep_map) | set(readiness_map) | set(hrv_map)
    total = len(all_days)
    if progress_callback:
        progress_callback(0, total)

    for i, day_str in enumerate(sorted(all_days), 1):
        try:
            payload: dict = {}

            sleep = sleep_map.get(day_str, {})
            if sleep:
                payload["sleep_score"] = sleep.get("score")
                payload["sleep_total_seconds"] = sleep.get("total_sleep_duration")
                payload["sleep_deep_seconds"] = sleep.get("deep_sleep_duration")
                payload["sleep_light_seconds"] = sleep.get("light_sleep_duration")
                payload["sleep_rem_seconds"] = sleep.get("rem_sleep_duration")
                payload["sleep_awake_seconds"] = sleep.get("awake_time")

            readiness = readiness_map.get(day_str, {})
            if readiness:
                payload["body_battery_high"] = readiness.get("score")
                rhr = (readiness.get("contributors") or {}).get("resting_heart_rate")
                if rhr is not None:
                    payload["resting_hr"] = rhr

            hrv = hrv_map.get(day_str, {})
            if hrv:
                rmssd = (hrv.get("summary") or {}).get("rmssd")
                if rmssd is not None:
                    payload["hrv_overnight_avg_ms"] = rmssd

            if not payload:
                continue

            day_dt = datetime.strptime(day_str, "%Y-%m-%d")
            row = existing_map.get(day_dt)
            if row is None:
                row = HealthMetric(user_id=user.id, date=day_dt, source="oura", **payload)
                db.add(row)
                existing_map[day_dt] = row
                stats["health_days_added"] += 1
            else:
                for k, v in payload.items():
                    if v is not None:
                        setattr(row, k, v)
                db.add(row)
                stats["health_days_updated"] += 1
        except Exception as exc:
            log.exception("oura day %s failed: %s", day_str, exc)
            stats["errors"] += 1

        if progress_callback and (i % 5 == 0 or i == total):
            progress_callback(i, total)

    await db.flush()
    return stats
