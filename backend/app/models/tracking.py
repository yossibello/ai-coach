"""
Phase-1 supplement-intake & performance-test logging.

These tables capture *what users actually do* (took supplement X for N weeks)
and *what changed* (FTP test, VO2max, HR threshold). Together they form the
ground truth needed for future causal modelling of supplement → performance
effects (Stage 3 in the architecture plan).
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    String, Float, Integer, DateTime, Text, ForeignKey, func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class SupplementIntake(Base):
    """A single 'I am taking / I took X' log entry."""
    __tablename__ = "supplement_intakes"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    # Stable key from supplements catalog (e.g. "iron", "vitamin_d3"). May also
    # be a free-text custom supplement not in the catalog.
    supplement_key: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    label:          Mapped[str] = mapped_column(String(200), nullable=False)

    dose:      Mapped[float | None] = mapped_column(Float, nullable=True)
    dose_unit: Mapped[str | None]   = mapped_column(String(40), nullable=True)
    frequency: Mapped[str | None]   = mapped_column(String(80), nullable=True)
    timing:    Mapped[str | None]   = mapped_column(String(120), nullable=True)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    stopped_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    # Self-reported adherence 0–100 %
    adherence_pct: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Where the entry came from: 'recommended' (one-click from stack),
    # 'manual' (user typed it), 'imported' (future: from a tracker app).
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="manual")

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship("User")


class PerformanceTest(Base):
    """A logged FTP / VO2max / threshold test result."""
    __tablename__ = "performance_tests"

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

    # Allowed: ftp_20min | ftp_ramp | ftp_8min | vo2max | threshold_hr | weight | resting_hr
    test_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)

    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit:  Mapped[str]   = mapped_column(String(20), nullable=False)

    # 'manual' | 'auto_detected' | 'imported'
    source: Mapped[str] = mapped_column(String(20), nullable=False, default="manual")

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship("User")
