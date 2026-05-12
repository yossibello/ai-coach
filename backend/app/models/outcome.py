"""
Outcome tracking — closes the prediction → reality feedback loop.

Two complementary tables:

  RecommendationFeedback
      What did the user DO with the suggested workout?
      Captured immediately when the user reacts in the UI:
        - "accepted" / "modified" / "rejected" / "skipped"
        - optional post-ride RPE (1–10) and free-text comment
      One row per recommendation. Used to weight personalization
      (LoRA fine-tuning Phase 3): rejected suggestions get negative
      reinforcement, accepted+high-RPE-match get positive.

  PredictionOutcome
      Did the model's PREDICTIONS come true?
      Backfilled by a scheduled job ~28 days after each recommendation:
        - predicted_ftp_delta_w  vs  actual_ftp_delta_w
        - predicted_workout_type vs  actual_workout_type (next ride)
        - predicted_duration_min vs  actual_duration_min
      Used as a regression target for LoRA fine-tuning and as a
      monitoring signal for community-model retrains.

Both tables are append-only. Neither is on the inference hot path.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, Float, Integer, DateTime, Text, ForeignKey, func, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class RecommendationFeedback(Base):
    __tablename__ = "recommendation_feedback"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    recommendation_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("recommendations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # 'accepted' | 'modified' | 'rejected' | 'skipped'
    action: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    # Post-ride RPE 1–10 (nullable — captured later than action).
    post_ride_rpe: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # If user modified the suggestion, what did they actually do?
    modified_workout_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    modified_duration_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Free-text reason ("legs felt dead", "weather", "moved to long ride").
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PredictionOutcome(Base):
    """
    Backfilled outcome row — written by a scheduled job ~28 days
    after the recommendation was generated, once enough new rides
    are available to compute actual FTP delta and ride match.
    """
    __tablename__ = "prediction_outcomes"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    recommendation_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("recommendations.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # Predictions snapshot (denormalized so we don't have to re-parse payload).
    predicted_workout_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    predicted_duration_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    predicted_intensity:    Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_ftp_delta_w:  Mapped[float | None] = mapped_column(Float, nullable=True)

    # Observed reality (filled by the backfill job).
    actual_workout_type: Mapped[str | None] = mapped_column(String(30), nullable=True)
    actual_duration_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actual_intensity:    Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_ftp_delta_w:  Mapped[float | None] = mapped_column(Float, nullable=True)
    horizon_days:        Mapped[int]   = mapped_column(Integer, nullable=False, default=28)

    # Derived metrics (cached for cheap querying / monitoring dashboards).
    workout_type_match: Mapped[bool | None] = mapped_column(nullable=True)
    duration_abs_err_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    ftp_delta_abs_err_w:  Mapped[float | None] = mapped_column(Float, nullable=True)

    measured_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# Composite index for backfill job: "give me all recs with no outcome yet
# whose generation time is older than 28 days".
Index(
    "ix_prediction_outcomes_recid_user",
    PredictionOutcome.recommendation_id,
    PredictionOutcome.user_id,
)
