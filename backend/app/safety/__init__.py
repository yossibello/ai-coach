"""Safety guards: hard limits applied to coach + supplement output before
the user ever sees them. These exist to prevent the system from suggesting
something that could cause physical harm (overtraining, supplement overdose,
dangerous combos).
"""
from app.safety.guards import (
    apply_workout_safety,
    apply_supplement_safety,
    MAX_DAILY_TSS,
    MAX_WEEKLY_TSS_RAMP_PCT,
    MAX_SINGLE_SESSION_MINUTES,
    HARD_RECOVERY_TSB,
    SUPPLEMENT_HARD_LIMITS,
)

__all__ = [
    "apply_workout_safety",
    "apply_supplement_safety",
    "MAX_DAILY_TSS",
    "MAX_WEEKLY_TSS_RAMP_PCT",
    "MAX_SINGLE_SESSION_MINUTES",
    "HARD_RECOVERY_TSB",
    "SUPPLEMENT_HARD_LIMITS",
]
