"""
Cold-start coaching: rule-based periodization for users with < 50 activities.

Based on:
 - Coggan/Allen power zones
 - Friel periodization (base, build, peak, recovery)
 - HRV-guided intensity (simplified)
 - Known scientific training patterns
"""
from __future__ import annotations

import random
from datetime import datetime, timezone, timedelta
from typing import Any

from app.models.user import AthleteProfile

# Workout descriptions for each type
WORKOUT_LIBRARY: dict[str, dict[str, Any]] = {
    "recovery": {
        "description": "Very easy spin — keep HR in Zone 1, legs only, no effort.",
        "key_metric": "HR below Zone 2 (<{z2_lo} bpm). Spin freely at 90+ rpm.",
        "structures": [
            {"phase": "warmup", "duration_minutes": 5, "power_target_pct_ftp": 40, "description": "Easy warm-up"},
            {"phase": "main",   "duration_minutes": 25, "power_target_pct_ftp": 45, "description": "Zone 1 easy spin"},
            {"phase": "cooldown","duration_minutes": 5, "power_target_pct_ftp": 35, "description": "Cool down"},
        ],
        "duration_minutes": 35,
        "target_tss": 25,
        "rationale": "Active recovery accelerates adaptation by promoting blood flow without adding stress.",
    },
    "easy": {
        "description": "Easy aerobic ride — chatty pace, low Z2, no fatigue.",
        "key_metric": "HR low Zone 2 ({z2_lo}–{z2_hi} bpm). Conversational throughout.",
        "structures": [
            {"phase": "warmup", "duration_minutes": 10, "power_target_pct_ftp": 50, "description": "Easy warm-up"},
            {"phase": "main",   "duration_minutes": 50, "power_target_pct_ftp": 60, "description": "Easy aerobic spin"},
            {"phase": "cooldown","duration_minutes": 10, "power_target_pct_ftp": 50, "description": "Cool down"},
        ],
        "duration_minutes": 70,
        "target_tss": 45,
        "rationale": "Easy aerobic riding builds the foundation: low-stress aerobic stimulus that supports recovery from harder sessions.",
    },
    "endurance": {
        "description": "Zone 2 aerobic base building. Talk-test pace throughout.",
        "key_metric": "Stay in Zone 2 ({z2_lo}–{z2_hi} bpm). Resist the urge to push.",
        "structures": [
            {"phase": "warmup", "duration_minutes": 15, "power_target_pct_ftp": 55, "description": "Gradual warm-up"},
            {"phase": "main",   "duration_minutes": 75, "power_target_pct_ftp": 65, "description": "Steady Zone 2"},
            {"phase": "cooldown","duration_minutes": 10, "power_target_pct_ftp": 50, "description": "Easy cool-down"},
        ],
        "duration_minutes": 100,
        "target_tss": 65,
        "rationale": "Zone 2 training builds mitochondrial density and fat oxidation capacity — your aerobic engine.",
    },
    "tempo": {
        "description": "Tempo — comfortably hard. Steady-state effort at 76–87% FTP.",
        "key_metric": "Power 76–87% FTP. Heart rate drifts up slowly — that's normal.",
        "structures": [
            {"phase": "warmup", "duration_minutes": 15, "power_target_pct_ftp": 60, "description": "Warm up"},
            {"phase": "main",   "duration_minutes": 40, "power_target_pct_ftp": 82, "description": "Tempo effort"},
            {"phase": "cooldown","duration_minutes": 10, "power_target_pct_ftp": 50, "description": "Cool down"},
        ],
        "duration_minutes": 65,
        "target_tss": 72,
        "rationale": "Tempo develops lactate clearance and raises the power output at aerobic threshold.",
    },
    "sweetspot": {
        "description": "Sweet Spot — 88-94% FTP. Maximum aerobic bang for your buck.",
        "key_metric": "Power 88–94% FTP. Sustainable but demanding. Breathe steadily.",
        "structures": [
            {"phase": "warmup", "duration_minutes": 15, "power_target_pct_ftp": 60, "description": "Warm up"},
            {"phase": "main",   "duration_minutes": 20, "power_target_pct_ftp": 91, "description": "Sweet Spot block 1", "cadence_rpm": 90},
            {"phase": "rest",   "duration_minutes": 5,  "power_target_pct_ftp": 50, "description": "Recovery"},
            {"phase": "main",   "duration_minutes": 20, "power_target_pct_ftp": 91, "description": "Sweet Spot block 2", "cadence_rpm": 90},
            {"phase": "cooldown","duration_minutes": 10, "power_target_pct_ftp": 50, "description": "Cool down"},
        ],
        "duration_minutes": 70,
        "target_tss": 80,
        "rationale": "Sweet Spot maximizes the training stimulus relative to recovery cost — ideal for raising FTP.",
    },
    "threshold": {
        "description": "Threshold work — riding at or just below FTP. Your ceiling effort.",
        "key_metric": "Power 95–105% FTP. Hold form, keep cadence 85-95 rpm.",
        "structures": [
            {"phase": "warmup",  "duration_minutes": 15, "power_target_pct_ftp": 60, "description": "Warm up"},
            {"phase": "main",    "duration_minutes": 12, "power_target_pct_ftp": 100, "description": "Threshold interval 1"},
            {"phase": "rest",    "duration_minutes": 6,  "power_target_pct_ftp": 50,  "description": "Recovery"},
            {"phase": "main",    "duration_minutes": 12, "power_target_pct_ftp": 100, "description": "Threshold interval 2"},
            {"phase": "rest",    "duration_minutes": 6,  "power_target_pct_ftp": 50,  "description": "Recovery"},
            {"phase": "main",    "duration_minutes": 12, "power_target_pct_ftp": 100, "description": "Threshold interval 3"},
            {"phase": "cooldown","duration_minutes": 10, "power_target_pct_ftp": 50,  "description": "Cool down"},
        ],
        "duration_minutes": 73,
        "target_tss": 85,
        "rationale": "Threshold intervals push your FTP ceiling by stressing the phosphocreatine and lactate systems.",
    },
    "vo2max": {
        "description": "VO2max intervals — 106-120% FTP. Short, intense, highly effective.",
        "key_metric": "Power 110–120% FTP. Breathing should be maximal. Complete all reps.",
        "structures": [
            {"phase": "warmup",  "duration_minutes": 20, "power_target_pct_ftp": 60, "description": "Thorough warm-up"},
            {"phase": "main",    "duration_minutes": 5,  "power_target_pct_ftp": 115, "description": "VO2 rep 1"},
            {"phase": "rest",    "duration_minutes": 5,  "power_target_pct_ftp": 45,  "description": "Full recovery"},
            {"phase": "main",    "duration_minutes": 5,  "power_target_pct_ftp": 115, "description": "VO2 rep 2"},
            {"phase": "rest",    "duration_minutes": 5,  "power_target_pct_ftp": 45,  "description": "Full recovery"},
            {"phase": "main",    "duration_minutes": 5,  "power_target_pct_ftp": 115, "description": "VO2 rep 3"},
            {"phase": "cooldown","duration_minutes": 15, "power_target_pct_ftp": 50,  "description": "Easy cool-down"},
        ],
        "duration_minutes": 60,
        "target_tss": 75,
        "rationale": "VO2max intervals improve cardiac output and maximal oxygen uptake — the ceiling of aerobic performance.",
    },
    "sprint": {
        "description": "Neuromuscular sprints — 5-10 second maximal efforts. Power and snap.",
        "key_metric": "Go all-out for each sprint. 10+ min recovery between. Stay seated or out of the saddle.",
        "structures": [
            {"phase": "warmup",  "duration_minutes": 20, "power_target_pct_ftp": 60, "description": "Warm up"},
            {"phase": "main",    "duration_minutes": 1,  "power_target_pct_ftp": 200, "description": "Sprint × 6", "cadence_rpm": 110},
            {"phase": "cooldown","duration_minutes": 20, "power_target_pct_ftp": 50,  "description": "Easy cool-down"},
        ],
        "duration_minutes": 60,
        "target_tss": 45,
        "rationale": "Neuromuscular work trains fast-twitch fiber recruitment and raises your sprint ceiling without significant aerobic stress.",
    },
    "long_ride": {
        "description": "Long aerobic ride — base building at comfortable pace.",
        "key_metric": "Stay Zone 2 the whole time. Nutrition every 45 min. Hydrate well.",
        "structures": [
            {"phase": "warmup",  "duration_minutes": 20, "power_target_pct_ftp": 55, "description": "Warm up"},
            {"phase": "main",    "duration_minutes": 130, "power_target_pct_ftp": 65, "description": "Zone 2 endurance"},
            {"phase": "cooldown","duration_minutes": 10, "power_target_pct_ftp": 50,  "description": "Cool down"},
        ],
        "duration_minutes": 160,
        "target_tss": 120,
        "rationale": "The long ride builds deep aerobic infrastructure, fat adaptation, and mental durability.",
    },
}

# Periodization: 3-week loading + 1-week recovery macro-cycle
WEEKLY_PATTERNS: dict[str, list[str]] = {
    "base_build": [
        "endurance", "recovery", "sweetspot", "recovery", "endurance", "long_ride", "recovery"
    ],
    "build": [
        "tempo", "recovery", "threshold", "recovery", "sweetspot", "long_ride", "recovery"
    ],
    "peak": [
        "threshold", "recovery", "vo2max", "recovery", "sweetspot", "long_ride", "recovery"
    ],
    "recovery_week": [
        "recovery", "recovery", "endurance", "recovery", "recovery", "recovery", "recovery"
    ],
}


def get_periodization_phase(weeks_to_event: int | None) -> str:
    """Simple periodization phase selector."""
    if weeks_to_event is None or weeks_to_event > 20:
        return "base_build"
    if weeks_to_event > 8:
        return "build"
    if weeks_to_event > 3:
        return "peak"
    return "recovery_week"


# Event-specific schedule biases. Each maps a workout type to a preferred
# replacement during the build/peak phase. Only applied for cold-start users;
# the trained model picks these patterns up automatically from the synthetic
# data (see ml/training/generate_synthetic.py).
_EVENT_BIAS: dict[str, dict[str, str]] = {
    # Long climbing days dominate Alps-style camps → swap intervals for
    # sweetspot + long Z2.
    "climbing_camp":   {"vo2max": "sweetspot", "threshold": "sweetspot", "tempo": "long_ride"},
    "gran_fondo":      {"vo2max": "sweetspot", "threshold": "tempo"},
    "ultra_endurance": {"vo2max": "endurance", "threshold": "tempo", "sweetspot": "endurance"},
    "mtb_marathon":    {"threshold": "sweetspot", "tempo": "sweetspot"},
    "stage_race":      {"vo2max": "threshold"},
    # Short max-effort events keep VO2/sprint emphasis.
    "crit":            {"sweetspot": "vo2max", "tempo": "vo2max"},
    "tt":              {"vo2max": "threshold", "sweetspot": "threshold"},
    "long_road":       {"tempo": "sweetspot"},
    "triathlon_70_3":  {"vo2max": "threshold"},
    "triathlon_140_6": {"vo2max": "tempo", "threshold": "sweetspot"},
}


def _bias_pattern_for_event(pattern: list[str], event_type: str) -> list[str]:
    """Apply event-specific replacements to a weekly workout pattern."""
    bias = _EVENT_BIAS.get(event_type)
    if not bias:
        return pattern
    return [bias.get(w, w) for w in pattern]


def build_cold_start_recommendation(
    profile: AthleteProfile | None,
    ctl: float,
    atl: float,
    tsb: float,
    recent_types: list[str],
    activity_count: int,
) -> dict[str, Any]:
    """
    Builds a full recommendation dict using periodization rules.
    Used when activity count < 50 or no trained model available.
    """
    ftp = (profile.ftp if profile and profile.ftp else 200) or 200
    max_hr = (profile.max_hr if profile and profile.max_hr else 190) or 190
    resting_hr = (profile.resting_hr if profile and profile.resting_hr else 55) or 55
    z2_lo = int(resting_hr + 0.6 * (max_hr - resting_hr))
    z2_hi = int(resting_hr + 0.7 * (max_hr - resting_hr))

    # Weeks to event
    weeks_to_event = None
    if profile and profile.goal_event_date:
        dte = (profile.goal_event_date.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).days
        weeks_to_event = max(0, dte // 7)

    phase = get_periodization_phase(weeks_to_event)
    pattern = WEEKLY_PATTERNS[phase]

    # Event-type bias (overlay on top of the base periodization pattern).
    # Only applied during build/peak when the user has a real event.
    event_type = getattr(profile, "event_type", None) if profile else None
    if event_type and phase in ("build", "peak") and weeks_to_event and weeks_to_event <= 12:
        pattern = _bias_pattern_for_event(pattern, event_type)

    # Fatigue overrides
    if tsb < -30:
        pattern = WEEKLY_PATTERNS["recovery_week"]
    elif tsb < -15 and phase not in ("recovery_week",):
        pattern = [p if p not in ("threshold", "vo2max", "sprint") else "sweetspot" for p in pattern]

    # Build 7-day plan
    weekly_plan = []
    for day_offset, wt in enumerate(pattern):
        tmpl = WORKOUT_LIBRARY.get(wt, WORKOUT_LIBRARY["endurance"])
        workout = {
            "day_offset": day_offset,
            "workout_type": wt,
            "duration_minutes": tmpl["duration_minutes"],
            "description": tmpl["description"],
            "structure": [dict(s) for s in tmpl["structures"]],
            "target_tss": tmpl["target_tss"],
            "rationale": tmpl["rationale"],
            "key_metric": tmpl["key_metric"].format(z2_lo=z2_lo, z2_hi=z2_hi),
        }
        weekly_plan.append(workout)

    next_workout = weekly_plan[0]

    # Insights
    insights = _build_insights(ctl, atl, tsb, ftp, activity_count, phase)

    # Risks
    risks = _build_risks(tsb, ctl, atl)

    # Forecast (conservative rule-based)
    if activity_count < 10:
        ftp_delta = 15  # typical starter gains
        confidence = 0.40
    elif activity_count < 30:
        ftp_delta = 10
        confidence = 0.55
    else:
        ftp_delta = 8
        confidence = 0.65

    # Inject strength sessions based on the athlete's chosen approach
    approach = (profile.strength_approach or "friel") if profile else "friel"
    if approach != "none":
        from app.strength.scheduler import add_strength_to_plan
        weekly_plan = add_strength_to_plan(weekly_plan, phase=phase, approach_key=approach)
        next_workout = weekly_plan[0]

    return {
        "next_workout": next_workout,
        "weekly_plan": weekly_plan,
        "insights": insights,
        "forecast": {
            "weeks": 8,
            "predicted_ftp_change_watts": ftp_delta,
            "predicted_ctl_peak": round(min(ctl * 1.3, ctl + 20), 1),
            "event_readiness_pct": _event_readiness(ctl, weeks_to_event),
            "confidence_interval": [ftp_delta - 5, ftp_delta + 10],
        },
        "risks": risks,
        "confidence": confidence,
        "model_version": "cold_start_v1",
        "is_cold_start": True,
    }


def _build_insights(
    ctl: float, atl: float, tsb: float, ftp: float, activity_count: int, phase: str
) -> list[dict[str, Any]]:
    insights = []

    if activity_count < 10:
        insights.append({
            "type": "tip",
            "title": "Getting started",
            "body": "Upload more rides to let the AI personalize your plan. "
                    "For now, we're using scientifically proven periodization patterns.",
        })

    if ctl < 20:
        insights.append({
            "type": "tip",
            "title": "Build your base",
            "body": "Your fitness (CTL) is in the early building phase. "
                    "Focus on consistency — ride 3-4× per week and avoid skipping rides.",
            "metric": "CTL", "value": round(ctl, 1), "unit": "TSS/day",
        })
    elif ctl > 80:
        insights.append({
            "type": "progress",
            "title": "High fitness level",
            "body": "Strong fitness base (CTL > 80). You're ready for race-level training stimulus.",
            "metric": "CTL", "value": round(ctl, 1), "unit": "TSS/day",
        })

    if tsb > 10:
        insights.append({
            "type": "progress",
            "title": "Well rested — time to train",
            "body": f"Your form (TSB +{round(tsb)}) shows you're fresh. "
                    "This is the perfect time for a quality threshold or VO2max session.",
            "metric": "TSB", "value": round(tsb, 1),
        })

    if phase == "peak":
        insights.append({
            "type": "tip",
            "title": "Peak phase — sharpen",
            "body": "You're in the peaking phase. Reduce volume but maintain intensity. "
                    "Trust your training — the fitness is built.",
        })

    return insights


def _build_risks(tsb: float, ctl: float, atl: float) -> list[dict[str, Any]]:
    risks = []
    ramp_rate = atl - ctl  # approx weekly CTL increase proxy

    if tsb < -30:
        risks.append({
            "type": "overtraining",
            "severity": "high",
            "message": f"TSB is {round(tsb)} — very deep fatigue. "
                       "Take 2-3 easy days before any quality work.",
        })
    elif tsb < -15:
        risks.append({
            "type": "overtraining",
            "severity": "medium",
            "message": "Moderate fatigue (TSB < -15). Schedule a recovery day soon.",
        })

    if ramp_rate > 10:
        risks.append({
            "type": "injury",
            "severity": "medium",
            "message": "Training load is ramping quickly. "
                       "Keep weekly TSS increases below 10% to reduce injury risk.",
        })

    if ctl > 0 and atl / ctl < 0.6:
        risks.append({
            "type": "undertraining",
            "severity": "low",
            "message": "Your acute load is much lower than your fitness. "
                       "Consider adding a session this week.",
        })

    return risks


def _event_readiness(ctl: float, weeks_to_event: int | None) -> float | None:
    if weeks_to_event is None:
        return None
    # Very rough heuristic: good readiness = high CTL + positive TSB close to event
    score = min(100, ctl * 0.8 + 20)
    if weeks_to_event < 1:
        score *= 1.1  # Taper effect
    return round(min(100, score), 1)
