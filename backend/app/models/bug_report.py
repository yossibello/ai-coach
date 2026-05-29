from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import String, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class BugReport(Base):
    __tablename__ = "bug_reports"

    id:          Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id:     Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    page:        Mapped[str]  = mapped_column(String(255), nullable=False)
    description: Mapped[str]  = mapped_column(Text, nullable=False)
    severity:    Mapped[str]  = mapped_column(String(20), default="medium")   # low / medium / high
    screenshot_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    user_agent:  Mapped[str | None] = mapped_column(String(512), nullable=True)
    status:      Mapped[str]  = mapped_column(String(20), default="open")     # open / in_progress / fixed
    created_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
