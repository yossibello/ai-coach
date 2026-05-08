"""
Celery application definition.

Start with:
  celery -A app.worker worker --loglevel=info

Tasks are defined inline here for simplicity; add @celery_app.task decorators
as needed for background work (e.g., CSV exports, model retraining).
"""
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "aicoach",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=3600,
)
