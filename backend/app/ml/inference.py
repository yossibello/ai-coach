"""
Inference engine: decides whether to use the trained transformer or cold-start rules,
builds the full recommendation, and analyzes individual activities.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import torch
from sqlalchemy import select as sa_select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User, AthleteProfile
from app.models.activity import Activity
from app.models.recommendation import Recommendation, FitnessMetric
from app.ml.cold_start import build_cold_start_recommendation
from app.ml.features import encode_activity, encode_profile
from app.ml.model import CyclingTransformer, INPUT_DIM, ACTIVITY_DIM, PROFILE_DIM
from app.core.config import settings

COLD_START_THRESHOLD = 50   # activities required before using transformer
MODEL_SEQ_LEN = 90          # last N activities fed to transformer

WORKOUT_TYPE_NAMES = [
    "recovery", "easy", "endurance", "tempo", "sweetspot",
    "threshold", "vo2max", "sprint", "race", "long_ride",
]

# Global model singleton (loaded once)
_model: CyclingTransformer | None = None


def _load_model() -> CyclingTransformer | None:
    """Load trained model from disk if available."""
    global _model
    if _model is not None:
        return _model

    path = settings.ML_MODEL_PATH
    if not os.path.exists(path):
        return None

    try:
        model = CyclingTransformer()
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _model = model
        return _model
    except Exception:
        return None


async def generate_recommendation(user: User, db: AsyncSession) -> Recommendation:
    """Main entry: decide cold-start vs transformer, return Recommendation ORM object."""
    # Load profile
    profile_result = await db.execute(
        sa_select(AthleteProfile).where(AthleteProfile.user_id == user.id)
    )
    profile = profile_result.scalar_one_or_none()

    # Load recent activities
    act_result = await db.execute(
        sa_select(Activity)
        .where(Activity.user_id == user.id)
        .order_by(desc(Activity.date))
        .limit(MODEL_SEQ_LEN)
    )
    activities: list[Activity] = list(reversed(act_result.scalars().all()))

    # Load current PMC snapshot
    metric_result = await db.execute(
        sa_select(FitnessMetric)
        .where(FitnessMetric.user_id == user.id)
        .order_by(desc(FitnessMetric.date))
        .limit(1)
    )
    latest_metric = metric_result.scalar_one_or_none()

    ctl = latest_metric.ctl if latest_metric else 0.0
    atl = latest_metric.atl if latest_metric else 0.0
    tsb = latest_metric.tsb if latest_metric else 0.0

    recent_types = [a.workout_type for a in activities[-7:] if a.workout_type]

    # Count total activities
    count_result = await db.execute(
        sa_select(func.count()).select_from(Activity).where(Activity.user_id == user.id)
    )
    total_activities = count_result.scalar_one()

    model = _load_model()

    if total_activities >= COLD_START_THRESHOLD and model is not None:
        payload = _transformer_recommendation(model, activities, profile, ctl, atl, tsb)
    else:
        payload = build_cold_start_recommendation(
            profile, ctl, atl, tsb, recent_types, total_activities
        )

    return Recommendation(
        user_id=user.id,
        confidence=payload["confidence"],
        model_version=payload["model_version"],
        is_cold_start=payload["is_cold_start"],
        payload=payload,
    )


def _transformer_recommendation(
    model: CyclingTransformer,
    activities: list[Activity],
    profile: AthleteProfile | None,
    ctl: float,
    atl: float,
    tsb: float,
) -> dict:
    """Run the transformer and convert outputs to recommendation payload."""
    # Encode activity sequence
    profile_vec = encode_profile(profile, profile.ftp if profile and profile.ftp else 200)

    seq_features = []
    day_indices = []
    first_date = activities[0].date if activities else datetime.now(timezone.utc)

    # Running CTL/ATL — pass PRE-update state into the encoder to match the
    # synthetic training distribution (which stores ctl/atl observed AT the
    # moment of the ride, before the ride's own TSS is added to the EMA).
    _ctl, _atl = 0.0, 0.0
    prev_date = None

    for act in activities:
        days_since = (act.date - first_date).days if act.date > first_date else 0
        days_since_last = (act.date - prev_date).days if prev_date else 3
        prev_date = act.date

        _tsb = _ctl - _atl
        act_vec = encode_activity(act, profile, _ctl, _atl, _tsb, days_since_last)
        token = np.concatenate([act_vec, profile_vec])
        seq_features.append(token)
        day_indices.append(min(days_since, 1499))

        # Update EMA AFTER encoding (so next iteration sees post-ride state)
        tss = act.tss or 0
        _ctl = _ctl + (2 / 43) * (tss - _ctl)
        _atl = _atl + (2 / 8) * (tss - _atl)

    # Pad or truncate to MODEL_SEQ_LEN
    seq_len = len(seq_features)
    if seq_len == 0:
        return build_cold_start_recommendation(profile, ctl, atl, tsb, [], 0)

    x = torch.tensor(np.stack(seq_features), dtype=torch.float32).unsqueeze(0)  # (1, T, D)
    di = torch.tensor(day_indices, dtype=torch.long).unsqueeze(0)                # (1, T)

    with torch.no_grad():
        out = model(x, di)

    workout_probs = torch.softmax(out["workout_logits"][0], dim=-1).numpy()
    workout_idx = int(workout_probs.argmax())
    workout_type = WORKOUT_TYPE_NAMES[workout_idx]
    intensity_if = float(out["intensity"][0, 0])
    duration_h = float(out["duration"][0, 0])
    ftp_delta = float(out["ftp_delta"][0, 0])
    ctl_peak = float(out["ctl_peak"][0, 0])
    risks_scores = out["risks"][0].numpy()  # [overtraining, undertraining, injury]

    from app.ml.cold_start import WORKOUT_LIBRARY, _build_risks
    tmpl = WORKOUT_LIBRARY.get(workout_type, WORKOUT_LIBRARY["endurance"])

    ftp = profile.ftp if profile and profile.ftp else 200
    duration_minutes = max(20, min(300, int(duration_h * 60)))

    next_workout = {
        "day_offset": 0,
        "workout_type": workout_type,
        "duration_minutes": duration_minutes,
        "description": tmpl["description"],
        "structure": [dict(s) for s in tmpl["structures"]],
        "target_tss": tmpl["target_tss"],
        "rationale": f"Transformer model selected {workout_type} based on your {seq_len}-ride training history.",
        "key_metric": tmpl["key_metric"].format(
            z2_lo=int(ftp * 0.56), z2_hi=int(ftp * 0.75)
        ),
    }

    # Build simple 7-day plan from recommended type + recoveries
    from app.ml.cold_start import WORKOUT_LIBRARY as WL
    weekly_types = [workout_type, "recovery", "sweetspot", "recovery", "endurance", "long_ride", "recovery"]
    weekly_plan = []
    for i, wt in enumerate(weekly_types):
        t = WL.get(wt, WL["endurance"])
        weekly_plan.append({
            "day_offset": i,
            "workout_type": wt,
            "duration_minutes": t["duration_minutes"],
            "description": t["description"],
            "structure": [dict(s) for s in t["structures"]],
            "target_tss": t["target_tss"],
            "rationale": t["rationale"],
            "key_metric": t["key_metric"].format(z2_lo=int(ftp * 0.56), z2_hi=int(ftp * 0.75)),
        })
    weekly_plan[0] = next_workout

    # Risks from transformer outputs
    risks = []
    if risks_scores[0] > 0.6:
        risks.append({"type": "overtraining", "severity": "high" if risks_scores[0] > 0.8 else "medium",
                      "message": "Model detected overtraining patterns. Consider a recovery day."})
    if risks_scores[1] > 0.6:
        risks.append({"type": "undertraining", "severity": "low",
                      "message": "Training load is below your potential. Room to add volume safely."})
    if risks_scores[2] > 0.6:
        risks.append({"type": "injury", "severity": "medium",
                      "message": "Training pattern resembles overuse sequences. Monitor for soreness."})

    confidence = float(workout_probs.max())
    return {
        "next_workout": next_workout,
        "weekly_plan": weekly_plan,
        "insights": [],
        "forecast": {
            "weeks": 8,
            "predicted_ftp_change_watts": round(ftp_delta, 1),
            "predicted_ctl_peak": round(max(ctl, ctl_peak), 1),
            "event_readiness_pct": None,
            "confidence_interval": [round(ftp_delta - 5, 1), round(ftp_delta + 8, 1)],
        },
        "risks": risks,
        "confidence": round(confidence, 3),
        "model_version": "cycling_transformer_v1",
        "is_cold_start": False,
    }


async def analyze_single_activity(
    activity: Activity,
    user: User,
    db: AsyncSession,
) -> dict:
    """Generate a text analysis + insights for a specific completed activity."""
    profile_result = await db.execute(
        sa_select(AthleteProfile).where(AthleteProfile.user_id == user.id)
    )
    profile = profile_result.scalar_one_or_none()
    ftp = (profile.ftp if profile and profile.ftp else 200) or 200

    insights = []

    if activity.tss:
        if activity.tss > 150:
            insights.append("Very high TSS — make sure you schedule 48h of easy riding afterward.")
        elif activity.tss < 20:
            insights.append("Low TSS — this was a light recovery session, as intended.")

    if activity.hr_drift and activity.hr_drift > 5:
        insights.append(
            f"HR drift of {activity.hr_drift:.1f}% indicates aerobic decoupling — "
            "your cardiovascular system was working harder at the end relative to power output. "
            "This is normal in hot conditions or when fatigued."
        )

    if activity.normalized_power and activity.avg_power:
        vi = activity.normalized_power / activity.avg_power
        if vi > 1.15:
            insights.append(
                f"High Variability Index ({vi:.2f}) suggests a stop-start or surgy ride. "
                "For aerobic sessions, aim for VI below 1.10."
            )

    if activity.avg_power and ftp:
        pct = activity.avg_power / ftp * 100
        insights.append(
            f"Average power was {activity.avg_power:.0f}W ({pct:.0f}% of FTP). "
            + _zone_comment(pct)
        )

    analysis = f"Analysis of '{activity.name}': "
    if activity.tss:
        analysis += f"{activity.tss:.0f} TSS, "
    if activity.workout_type:
        analysis += f"classified as {activity.workout_type.replace('_', ' ')} effort. "
    if insights:
        analysis += insights[0]

    return {"analysis": analysis, "insights": insights}


def _zone_comment(pct_ftp: float) -> str:
    if pct_ftp < 55: return "Zone 1 — active recovery."
    if pct_ftp < 75: return "Zone 2 — aerobic endurance."
    if pct_ftp < 90: return "Zone 3 — tempo."
    if pct_ftp < 105: return "Zone 4 — threshold."
    if pct_ftp < 120: return "Zone 5 — VO2max territory."
    return "Zone 6+ — anaerobic/neuromuscular."
