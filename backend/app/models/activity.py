import uuid
from datetime import datetime

from sqlalchemy import String, Float, Integer, DateTime, Text, func, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class Activity(Base):
    __tablename__ = "activities"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    external_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False)  # strava/garmin/gpx/fit/manual

    name: Mapped[str] = mapped_column(String(500), nullable=False, default="Ride")
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    distance_meters: Mapped[float] = mapped_column(Float, nullable=False, default=0)
    elevation_gain_meters: Mapped[float] = mapped_column(Float, nullable=True)

    # Power
    avg_power: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_power: Mapped[float | None] = mapped_column(Float, nullable=True)
    normalized_power: Mapped[float | None] = mapped_column(Float, nullable=True)
    intensity_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    tss: Mapped[float | None] = mapped_column(Float, nullable=True)
    variability_index: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Heart rate
    avg_hr: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_hr: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hr_drift: Mapped[float | None] = mapped_column(Float, nullable=True)  # aerobic decoupling %
    aerobic_efficiency: Mapped[float | None] = mapped_column(Float, nullable=True)  # pw:hr

    # Cadence
    avg_cadence: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Environment
    temperature_c: Mapped[float | None] = mapped_column(Float, nullable=True)
    humidity_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    wind_speed_kmh: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Zones (JSON: {z1: pct, z2: pct, ...})
    time_in_zones: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    workout_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    perceived_exertion: Mapped[int | None] = mapped_column(Integer, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Strava-specific extras
    kudos_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trainer: Mapped[bool | None] = mapped_column(nullable=True)

    # ── Data quality (Phase 1: rule-based; Phase 2+: feeds community retraining) ──
    # quality_score: 'high' | 'medium' | 'low' | 'rejected'
    quality_score: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    # JSON list of reason codes, e.g. ["missing_power", "hr_too_high"]
    quality_reasons: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # 'pending' | 'confirmed' | 'quarantined' | 'deleted'
    review_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="confirmed", server_default="confirmed", index=True
    )
    # Set by outlier detector (z-score vs user history) once user has enough rides
    is_outlier: Mapped[bool] = mapped_column(
        nullable=False, default=False, server_default="false"
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship("User", back_populates="activities")
