import uuid
from datetime import datetime

from sqlalchemy import String, Float, Integer, DateTime, Text, JSON, func, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    model_version: Mapped[str] = mapped_column(String(50), default="cold_start_v1")
    is_cold_start: Mapped[bool] = mapped_column(Boolean, default=True)

    # Full JSON payload of the recommendation
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    user: Mapped["User"] = relationship("User")


class FitnessMetric(Base):
    """Daily PMC metrics per user (CTL/ATL/TSB)."""
    __tablename__ = "fitness_metrics"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    ctl: Mapped[float] = mapped_column(Float, default=0)   # 42-day EMA of TSS
    atl: Mapped[float] = mapped_column(Float, default=0)   # 7-day EMA of TSS
    tsb: Mapped[float] = mapped_column(Float, default=0)   # CTL - ATL
    tss: Mapped[float] = mapped_column(Float, default=0)   # Daily TSS
    ftp: Mapped[float] = mapped_column(Float, default=0)   # FTP on that date

    user: Mapped["User"] = relationship("User")
