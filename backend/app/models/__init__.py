# backend/app/models/__init__.py
from app.models.user import User, AthleteProfile  # noqa: F401
from app.models.activity import Activity  # noqa: F401
from app.models.recommendation import Recommendation, FitnessMetric  # noqa: F401
from app.models.nutrition import (  # noqa: F401
    BloodTest,
    BloodMarker,
    SupplementRecommendation,
)
