from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.recommendation import Recommendation


async def latest_recommendation_id(
    user_id: str,
    activity_date,
    db: AsyncSession,
    window_hours: int = 48,
) -> str | None:
    """Return the most recent recommendation for user within window_hours before activity_date."""
    result = await db.execute(
        select(Recommendation.id)
        .where(Recommendation.user_id == user_id)
        .where(Recommendation.generated_at <= activity_date)
        .where(Recommendation.generated_at >= activity_date - timedelta(hours=window_hours))
        .order_by(Recommendation.generated_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()
