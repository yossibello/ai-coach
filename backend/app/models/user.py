import uuid
from datetime import datetime, timezone

from sqlalchemy import String, Boolean, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)

    # Strava OAuth
    strava_athlete_id: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    strava_access_token: Mapped[str | None] = mapped_column(String, nullable=True)
    strava_refresh_token: Mapped[str | None] = mapped_column(String, nullable=True)
    strava_token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Garmin OAuth
    garmin_access_token: Mapped[str | None] = mapped_column(String, nullable=True)
    garmin_refresh_token: Mapped[str | None] = mapped_column(String, nullable=True)

    # ── ML / training participation ───────────────────────────────────────────
    # User consent: may we use their (anonymized) rides to retrain the community model?
    allow_for_training: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    # Which model version is currently serving this user's recommendations.
    # 'synthetic_v1', 'community_v2', etc. NULL = use latest active.
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    profile: Mapped["AthleteProfile | None"] = relationship(
        "AthleteProfile", back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    activities: Mapped[list["Activity"]] = relationship(
        "Activity", back_populates="user", cascade="all, delete-orphan"
    )

    @property
    def strava_connected(self) -> bool:
        return self.strava_athlete_id is not None


class AthleteProfile(Base):
    __tablename__ = "athlete_profiles"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
    )

    age: Mapped[int | None] = mapped_column(nullable=True)
    weight_kg: Mapped[float | None] = mapped_column(nullable=True)
    height_cm: Mapped[float | None] = mapped_column(nullable=True)
    sex: Mapped[str | None] = mapped_column(String(10), nullable=True)

    ftp: Mapped[int | None] = mapped_column(nullable=True)        # watts
    max_hr: Mapped[int | None] = mapped_column(nullable=True)     # bpm
    resting_hr: Mapped[int | None] = mapped_column(nullable=True)
    vo2max_estimate: Mapped[float | None] = mapped_column(nullable=True)
    cycling_experience_years: Mapped[int | None] = mapped_column(nullable=True)

    primary_goal: Mapped[str | None] = mapped_column(String(50), nullable=True)
    goal_event_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    goal_event_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    training_days_per_week: Mapped[int | None] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="profile")
