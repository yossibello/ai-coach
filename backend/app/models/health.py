"""
Daily health & wellness metrics, sourced primarily from Garmin Connect
(sleep, HRV, resting HR, body battery, stress).  One row per user per day.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class HealthMetric(Base):
    __tablename__ = "health_metrics"
    __table_args__ = (
        UniqueConstraint("user_id", "date", name="uq_health_metrics_user_date"),
    )

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="garmin")

    # Sleep
    sleep_total_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sleep_deep_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sleep_light_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sleep_rem_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sleep_awake_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sleep_score: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # HRV (overnight, in ms)
    hrv_overnight_avg_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    hrv_7d_avg_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    hrv_status: Mapped[str | None] = mapped_column(String(20), nullable=True)  # balanced/unbalanced/low/poor

    # Resting HR
    resting_hr: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Body Battery (Garmin's recovery score, 0-100)
    body_battery_high: Mapped[int | None] = mapped_column(Integer, nullable=True)
    body_battery_low: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Stress (0-100, Garmin's all-day stress)
    stress_avg: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stress_max: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # SpO2
    spo2_avg: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
