"""
Nutrition / blood-marker / supplement models.

BloodTest             — one row per uploaded lab report. Markers stored as JSON
                        so we can extend the panel without migrations.
BloodMarker           — denormalized one-row-per-marker for time-series queries
                        (e.g. "show ferritin trend over 12 months").
SupplementRecommendation — generated stack snapshot, mirrors Recommendation table.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    String, Float, DateTime, JSON, ForeignKey, Boolean, Text, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class BloodTest(Base):
    __tablename__ = "blood_tests"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    test_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    lab_name:    Mapped[str | None] = mapped_column(String(200), nullable=True)
    source:      Mapped[str]        = mapped_column(String(20), default="pdf")  # pdf|manual|api
    raw_filename: Mapped[str | None] = mapped_column(String(500), nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(20), nullable=True)
    parser_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Full extracted marker dict: {"ferritin": {"value": 35, "unit": "ng/mL", "ref_low": 30, "ref_high": 400}, ...}
    markers: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship("User")


class BloodMarker(Base):
    """One row per (user, marker, test_date) for fast time-series queries."""
    __tablename__ = "blood_markers"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    blood_test_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("blood_tests.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    test_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    marker_key: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit:  Mapped[str | None] = mapped_column(String(30), nullable=True)
    ref_low:  Mapped[float | None] = mapped_column(Float, nullable=True)
    ref_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 'low' | 'optimal' | 'high' | 'critical_low' | 'critical_high' | 'unknown'
    status: Mapped[str] = mapped_column(String(20), default="unknown", index=True)


class SupplementRecommendation(Base):
    __tablename__ = "supplement_recommendations"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    engine_version: Mapped[str] = mapped_column(String(50), default="rule_v1")
    is_cold_start: Mapped[bool] = mapped_column(Boolean, default=True)

    # Inputs that produced this stack (for auditability)
    based_on_blood_test_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False), ForeignKey("blood_tests.id", ondelete="SET NULL"),
        nullable=True,
    )

    # payload = {
    #   "stack": [ {supplement_key, name, dose, unit, timing, frequency,
    #              rationale, evidence_grade, citations, contraindications}, ... ],
    #   "depletion_scores": { "iron": 0.6, "magnesium": 0.4, ... },
    #   "warnings": [ ... ],
    # }
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    user: Mapped["User"] = relationship("User")
