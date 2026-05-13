"""
Inference engine: decides whether to use the trained transformer or cold-start rules,
builds the full recommendation, and analyzes individual activities.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
from sqlalchemy import select as sa_select, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User, AthleteProfile
from app.models.activity import Activity
from app.models.recommendation import Recommendation, FitnessMetric
from app.models.health import HealthMetric
from app.ml.cold_start import build_cold_start_recommendation
from app.ml.features import encode_activity, encode_profile
from app.ml.model import CyclingTransformer, INPUT_DIM, ACTIVITY_DIM, PROFILE_DIM
from app.ml.norm import encode_profile_row
from app.services.readiness import compute_readiness, per_day_health_features
from app.core.config import settings

COLD_START_THRESHOLD = 50   # activities required before using transformer
MODEL_SEQ_LEN = 90          # last N activities fed to transformer

# Phase 1 multi-horizon: same model, called 3 times with different days_to_event
# inputs. The transformer was trained on synthetic athletes with varying event
# timelines, so this elicits (weakly) different recommendations per horizon.
# Phase 1.5 / B will add explicit horizon-conditioning + dedicated heads.
HORIZON_DAYS = {
    "short":  7,    # "what gives me the biggest 1-week gain?"
    "medium": 28,   # "what's optimal for a 4-week build?"
    "event":  None, # uses the user's actual goal_event_date (or 90 if none set)
}

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
        state = torch.load(path, map_location="cpu")
        # New checkpoint format: {"state_dict": ..., "config": {...}}
        # Old format: raw state_dict (kwargs default to current model defaults)
        if isinstance(state, dict) and "state_dict" in state:
            cfg = dict(state.get("config", {}) or {})
            sd = state["state_dict"]
        else:
            cfg = {}
            sd = state
        # Detect input_dim from the input projection layer so old checkpoints
        # (trained before health features were added) still load. Slicing in
        # _transformer_recommendation handles the feature-count mismatch.
        proj_w = sd.get("input_proj.0.weight")
        if proj_w is not None:
            cfg["input_dim"] = int(proj_w.shape[1])
        # Strip non-init keys before forwarding to the constructor.
        horizon_aware = bool(cfg.pop("horizon_aware", False))
        # Per-checkpoint softmax temperature from temperature scaling
        # (Guo et al. 2017). Defaults to 1.0 (no calibration).
        temperature = float(cfg.pop("temperature", 1.0))
        # Detect horizon-aware checkpoints by presence of horizon_proj weights
        # (in case `horizon_aware` flag is missing from older horizon-aware
        # checkpoints).
        if "horizon_proj.0.weight" in sd:
            horizon_aware = True
            cfg.setdefault("horizon_dim", int(sd["horizon_proj.0.weight"].shape[1]))
        model = CyclingTransformer(**cfg)
        # strict=False so a checkpoint without horizon_proj/horizon_query_bias
        # still loads cleanly into the new architecture (those weights stay at
        # their initialized values, and we won't pass horizon_query at infer time).
        model.load_state_dict(sd, strict=False)
        model.eval()
        # Stash for downstream slicing & dispatch.
        model._input_dim = cfg.get("input_dim", INPUT_DIM)
        model._horizon_aware = horizon_aware
        model._temperature = temperature
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

    # Load recent activities — exclude quarantined/rejected junk so the model
    # context is clean. (See app.ml.quality / metrics_service._score_and_tag.)
    act_result = await db.execute(
        sa_select(Activity)
        .where(
            Activity.user_id == user.id,
            Activity.review_status == "confirmed",
        )
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

    # Load health metrics (HRV / RHR / sleep / body battery) for the same window
    # as activities + a 30-day prefix so HRV z-scores have a baseline.
    health_cutoff = datetime.utcnow() - timedelta(days=120)
    health_result = await db.execute(
        sa_select(HealthMetric)
        .where(
            HealthMetric.user_id == user.id,
            HealthMetric.date >= health_cutoff,
        )
        .order_by(HealthMetric.date)
    )
    health_metrics: list[HealthMetric] = list(health_result.scalars().all())
    health_by_date = per_day_health_features(health_metrics)
    readiness_snapshot = compute_readiness(health_metrics)

    recent_types = [a.workout_type for a in activities[-7:] if a.workout_type]

    # Count total activities
    count_result = await db.execute(
        sa_select(func.count()).select_from(Activity).where(Activity.user_id == user.id)
    )
    total_activities = count_result.scalar_one()

    model = _load_model()

    if total_activities >= COLD_START_THRESHOLD and model is not None:
        payload = _transformer_recommendation(model, activities, profile, ctl, atl, tsb,
                                              health_by_date=health_by_date)
    else:
        payload = build_cold_start_recommendation(
            profile, ctl, atl, tsb, recent_types, total_activities
        )

    # Attach readiness so the UI / safety guard can use it.
    payload["readiness"] = {
        "score": readiness_snapshot.score,
        "status": readiness_snapshot.status,
        "hrv_z": readiness_snapshot.hrv_z,
        "rhr_delta": readiness_snapshot.rhr_delta,
        "sleep_score": readiness_snapshot.sleep_score,
        "body_battery": readiness_snapshot.body_battery,
        "drivers": readiness_snapshot.drivers,
        "advice": readiness_snapshot.advice,
    }

    # ── Hard safety pass: clamp duration / TSS / intensity to safe ranges ──
    from app.safety.guards import apply_workout_safety, apply_weekly_plan_safety, apply_health_safety

    # Estimate last-week TSS from recent activities for ramp guard.
    # Activity dates from SQLite are naive; compare against a naive cutoff.
    week_ago = datetime.utcnow() - timedelta(days=7)
    def _naive(d):
        return d.replace(tzinfo=None) if d and d.tzinfo else d
    last_week_tss = sum(
        (a.tss or 0) for a in activities if a.date and _naive(a.date) >= week_ago
    )

    safe_next, next_notes = apply_workout_safety(
        payload.get("next_workout"), ctl=ctl, tsb=tsb
    )
    # Layer health-readiness override on top of TSB-based safety.
    safe_next, health_notes = apply_health_safety(
        safe_next, readiness=payload.get("readiness")
    )
    next_notes = next_notes + health_notes
    payload["next_workout"] = safe_next

    # ── Already-rode-today guard ────────────────────────────────────────────
    # If the rider has a confirmed activity dated today, don't push another
    # full workout for today — shift the suggestion to tomorrow.
    today = datetime.utcnow().date()
    today_acts = [a for a in activities if a.date and _naive(a.date).date() == today]
    today_tss = sum((a.tss or 0.0) for a in today_acts)
    today_dur_min = sum((a.duration_seconds or 0) for a in today_acts) / 60.0

    if today_acts and safe_next is not None:
        # Significant ride already done → shift to tomorrow as a rest/easy day
        if today_tss >= 30 or today_dur_min >= 45:
            safe_next["day_offset"] = 1
            safe_next["already_rode_today"] = True
            safe_next["today_tss"] = round(today_tss, 1)
            safe_next["today_duration_minutes"] = int(today_dur_min)
            existing_rationale = safe_next.get("rationale", "")
            safe_next["rationale"] = (
                f"You've already ridden today ({int(today_dur_min)} min, "
                f"{int(today_tss)} TSS). Suggested workout moved to tomorrow. "
                f"{existing_rationale}"
            ).strip()
            payload.setdefault("safety_notes", []).append(
                f"Today's ride detected ({int(today_tss)} TSS) — next workout shifted to tomorrow."
            )
        else:
            # Small spin already done — keep today but subtract its TSS from target
            tgt = safe_next.get("target_tss")
            if tgt:
                safe_next["target_tss"] = max(15, int(tgt - today_tss))
                safe_next["already_rode_today"] = True
                safe_next["today_tss"] = round(today_tss, 1)
                payload.setdefault("safety_notes", []).append(
                    f"Light ride already done today ({int(today_tss)} TSS) — "
                    f"target reduced accordingly."
                )

    safe_plan, plan_notes = apply_weekly_plan_safety(
        payload.get("weekly_plan", []),
        last_week_tss=last_week_tss,
        ctl=ctl,
        tsb=tsb,
    )
    # Keep weekly_plan[0] in sync with next_workout (safety guard may have
    # returned a copy, so day_offset and other fields can diverge).
    # Then renumber the rest of the plan sequentially after next_workout's day.
    if safe_next is not None and safe_plan:
        safe_plan[0] = safe_next
        start = safe_next.get("day_offset", 0)
        for i, w in enumerate(safe_plan[1:], start=1):
            w["day_offset"] = start + i
    payload["weekly_plan"] = safe_plan
    if next_notes or plan_notes:
        payload.setdefault("safety_notes", []).extend(next_notes + plan_notes)

    return Recommendation(
        user_id=user.id,
        confidence=payload["confidence"],
        model_version=payload["model_version"],
        is_cold_start=payload["is_cold_start"],
        payload=payload,
    )


def _encode_profile_with_horizon(
    profile: AthleteProfile | None, horizon_override_days: int | None
) -> np.ndarray:
    """Encode profile, optionally overriding `days_to_event` for multi-horizon probing."""
    if horizon_override_days is None:
        return encode_profile(profile, profile.ftp if profile and profile.ftp else 200)

    p = profile
    raw = {
        "age":                       p.age if p else None,
        "weight_kg":                 p.weight_kg if p else None,
        "height_cm":                 p.height_cm if p else None,
        "sex":                       p.sex if p else None,
        "ftp":                       (p.ftp if p and p.ftp else 200),
        "athlete_max_hr":            p.max_hr if p else None,
        "resting_hr":                p.resting_hr if p else None,
        "cycling_experience_years":  p.cycling_experience_years if p else None,
        "primary_goal":              p.primary_goal if p else None,
        "days_to_event":             max(0, int(horizon_override_days)),
    }
    return encode_profile_row(raw)


def _transformer_recommendation(
    model: CyclingTransformer,
    activities: list[Activity],
    profile: AthleteProfile | None,
    ctl: float,
    atl: float,
    tsb: float,
    horizon_override_days: int | None = None,
    *,
    health_by_date: dict | None = None,
) -> dict:
    """Run the transformer and convert outputs to recommendation payload.

    If `horizon_override_days` is provided, it replaces `days_to_event` in the
    profile vector — letting us probe the same model under different planning
    horizons without retraining (Phase-1 multi-horizon trick).
    """
    # Encode activity sequence
    profile_vec = _encode_profile_with_horizon(profile, horizon_override_days)

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

        # Pull health context for this ride's date if available.
        h_kwargs = {}
        if health_by_date and act.date is not None:
            tup = health_by_date.get(act.date.date())
            if tup is not None:
                h_kwargs = {
                    "hrv_z": tup[0],
                    "rhr_delta": tup[1],
                    "sleep_score": tup[2],
                    "body_battery": tup[3],
                }

        act_vec = encode_activity(act, profile, _ctl, _atl, _tsb, days_since_last, **h_kwargs)
        # Backwards-compat: an older checkpoint trained without health features
        # expects ACTIVITY_DIM-4 columns. Slice off the trailing health block
        # so the input projection shape matches.
        expected_input = getattr(model, "_input_dim", INPUT_DIM)
        expected_act = expected_input - PROFILE_DIM
        if act_vec.shape[0] > expected_act:
            act_vec = act_vec[:expected_act]
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

    # ── Horizon query token (only if model was trained horizon-aware) ────
    # Map the override (days) into the same {short, medium, event} bucket
    # used during training. Without an override, default to "short".
    hq_tensor = None
    if getattr(model, "_horizon_aware", False):
        from app.ml.model import encode_horizon
        h_days = horizon_override_days if horizon_override_days is not None else 7
        if h_days <= 14:
            label = "short"
        elif h_days <= 42:
            label = "medium"
        else:
            label = "event"
        hq_vec = np.asarray(encode_horizon(label, float(h_days)), dtype=np.float32)
        hq_tensor = torch.from_numpy(hq_vec).unsqueeze(0)  # (1, HORIZON_DIM)

    with torch.no_grad():
        out = model(x, di, horizon_query=hq_tensor)

    # ── Bayesian posterior selection ──────────────────────────────────────
    # Combine the model's likelihood with a periodization phase prior, an
    # event-type bias, and a fatigue/HRV safety factor. This makes the
    # selection robust to the model's argmax wobbling between similar
    # workouts (e.g. tempo vs sweetspot) and refuses to recommend
    # vo2max/threshold when the athlete is clearly fatigued.
    from app.ml.phase_prior import (
        select_workout, horizon_to_phase, WORKOUT_TYPE_NAMES as _PP_NAMES,
    )
    raw_logits = out["workout_logits"][0].numpy()
    h_for_phase = horizon_override_days if horizon_override_days is not None else (
        max(0, int((profile.goal_event_date.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).days))
        if profile and profile.goal_event_date else 0
    )
    phase_for_prior = horizon_to_phase(h_for_phase)
    # Pull most-recent HRV z-score from the PMC-tagged sequence if available.
    hrv_z_now: float | None = None
    try:
        # `tail_health` may be defined upstream in this function via the same
        # path used to build `profile_vec`. We use a soft lookup to avoid
        # coupling.
        hrv_z_now = float(locals().get("_hrv_z_latest")) if locals().get("_hrv_z_latest") is not None else None
    except Exception:
        hrv_z_now = None
    workout_type, workout_conf, top_alts = select_workout(
        raw_logits,
        phase=phase_for_prior,
        event_type=getattr(profile, "event_type", None) if profile else None,
        tsb=tsb,
        hrv_z=hrv_z_now,
        temperature=float(getattr(model, "_temperature", 1.0)),
        prior_weight=0.6,  # moderate: trust model but respect periodization
        top_k=3,
    )
    workout_probs = np.asarray(
        [next((p for n, p in top_alts if n == name), 0.0) for name in _PP_NAMES],
        dtype=np.float32,
    )
    workout_idx = _PP_NAMES.index(workout_type)
    intensity_if    = float(out["intensity"][0, 0])
    duration_h      = float(out["duration"][0, 0])
    ftp_delta_pct   = float(out["ftp_delta"][0, 0])
    pc5min_delta_pct = float(out["pc5min_delta"][0, 0]) if "pc5min_delta" in out else 0.0
    pc1min_delta_pct = float(out["pc1min_delta"][0, 0]) if "pc1min_delta" in out else 0.0
    ctl_peak        = float(out["ctl_peak"][0, 0])
    risks_scores    = out["risks"][0].numpy()  # [overtraining, undertraining, injury]

    from app.ml.cold_start import WORKOUT_LIBRARY, _build_risks
    tmpl = WORKOUT_LIBRARY.get(workout_type, WORKOUT_LIBRARY["endurance"])

    ftp    = profile.ftp if profile and profile.ftp else 200
    weight = profile.weight_kg if profile and getattr(profile, "weight_kg", None) else 70.0
    ftp_delta = ftp_delta_pct * ftp  # fraction → watts for this rider
    # Capacity estimates for 1-min and 5-min (W/kg); use profile FTP/weight as proxy
    # if individual capacity fields are not stored on the profile yet.
    ftp_wkg = ftp / max(weight, 1.0)
    pc5min_now_wkg = getattr(profile, "pc5min_capacity_wkg", None) or ftp_wkg * 1.18
    pc1min_now_wkg = getattr(profile, "pc1min_capacity_wkg", None) or ftp_wkg * 1.65
    pc5min_delta_wkg = pc5min_delta_pct * pc5min_now_wkg
    pc1min_delta_wkg = pc1min_delta_pct * pc1min_now_wkg
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

    # Build 7-day plan — use the PHASE-APPROPRIATE schedule for this horizon
    # so that short/medium/event horizons look genuinely different.
    # (Without this, all 3 horizons get the same hardcoded pattern and the
    # user sees no change when switching tabs.)
    from app.ml.cold_start import WORKOUT_LIBRARY as WL, WEEKLY_PATTERNS, get_periodization_phase, _bias_pattern_for_event
    dte_for_plan = horizon_override_days if horizon_override_days is not None else (
        (profile.goal_event_date.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).days
        if profile and profile.goal_event_date else 90
    )
    weeks_for_plan = max(0, dte_for_plan // 7)
    phase_for_plan = get_periodization_phase(weeks_for_plan)
    phase_pattern = list(WEEKLY_PATTERNS.get(phase_for_plan, WEEKLY_PATTERNS["build"]))

    # Override Day 0 with what the transformer actually predicted.
    phase_pattern[0] = workout_type

    # Apply event-type bias on the horizon-specific pattern.
    event_type_val = getattr(profile, "event_type", None) if profile else None
    if event_type_val and phase_for_plan in ("build", "peak"):
        phase_pattern = _bias_pattern_for_event(phase_pattern, event_type_val)

    weekly_plan = []
    for i, wt in enumerate(phase_pattern):
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

    # ── Phase 3b: Weekly TSS constraint solver (Friel safe ramp + TSB cap) ─
    # Re-balance per-day TSS so the weekly total respects: maintain CTL +
    # phase-appropriate ramp, never push end-of-week TSB below -25, throttle
    # hard sessions when HRV is suppressed.
    try:
        from app.ml.planner import solve_week, tss_to_duration_minutes
        plan_types = [w["workout_type"] for w in weekly_plan]
        solved_tss, solver_notes = solve_week(
            plan_types,
            ctl=float(ctl),
            atl=float(atl),
            tsb=float(tsb),
            phase=phase_for_plan,
            hrv_z=hrv_z_now,
        )
        for i, (w, new_tss) in enumerate(zip(weekly_plan, solved_tss)):
            # Skip Day 0 — its duration was set by the transformer's own
            # duration head and is more athlete-specific.
            if i == 0:
                continue
            new_tss_int = int(round(new_tss))
            w["target_tss"] = new_tss_int
            # Update duration only when the workout has a typical IF we can
            # invert (we use the WL key_metric implied IF).
            wl_entry = WL.get(w["workout_type"])
            if wl_entry:
                # Approximate IF from workout type (matches WL targets):
                if_map = {
                    "recovery": 0.45, "easy": 0.60, "endurance": 0.68,
                    "long_ride": 0.65, "tempo": 0.82, "sweetspot": 0.91,
                    "threshold": 1.00, "vo2max": 1.10, "sprint": 1.05,
                    "race": 1.00,
                }
                est_if = if_map.get(w["workout_type"], 0.70)
                w["duration_minutes"] = tss_to_duration_minutes(new_tss, est_if)
        # Surface solver notes downstream via payload.safety_notes.
        _planner_notes = solver_notes
    except Exception:
        _planner_notes = []

    # Risks from transformer outputs.
    # Overtraining and undertraining are mutually exclusive: if overtraining
    # fires more strongly, suppress undertraining entirely (and vice versa).
    # This corrects for the independent-sigmoid risk head used in v2 — the
    # next retrain will replace it with a softmax head.
    risks = []
    over_score  = float(risks_scores[0])
    under_score = float(risks_scores[1])
    inj_score   = float(risks_scores[2])

    # Only fire the stronger of over/under; suppress the weaker one.
    if over_score > 0.6 and over_score >= under_score:
        risks.append({"type": "overtraining",
                      "severity": "high" if over_score > 0.8 else "medium",
                      "message": "Model detected overtraining patterns. Consider a recovery day."})
    elif under_score > 0.6 and under_score > over_score:
        risks.append({"type": "undertraining", "severity": "low",
                      "message": "Training load is below your potential. Room to add volume safely."})

    if inj_score > 0.6:
        risks.append({"type": "injury", "severity": "medium",
                      "message": "Training pattern resembles overuse sequences. Monitor for soreness."})

    confidence = float(workout_probs.max())
    return {
        "next_workout": next_workout,
        "weekly_plan": weekly_plan,
        "insights": [],
        "forecast": {
            "weeks": 8,
            "predicted_ftp_change_watts":   round(ftp_delta, 1),
            "predicted_pc5min_change_wkg":  round(pc5min_delta_wkg, 3),
            "predicted_pc1min_change_wkg":  round(pc1min_delta_wkg, 3),
            "predicted_ctl_peak":           round(max(ctl, ctl_peak), 1),
            "event_readiness_pct": None,
            "confidence_interval": [round(ftp_delta - 5, 1), round(ftp_delta + 8, 1)],
        },
        "risks": risks,
        "confidence": round(confidence, 3),
        "model_version": (
            "cycling_transformer_v2_horizon"
            if getattr(model, "_horizon_aware", False)
            else "cycling_transformer_v1"
        ),
        "is_cold_start": False,
        "safety_notes": list(_planner_notes) if _planner_notes else [],
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


def _apply_today_ride_guard(payload: dict, activities: list) -> dict:
    """
    Shift next_workout and weekly_plan[0] to tomorrow if the user already did a
    significant ride today. Mirrors the guard in generate_recommendation so that
    multi-horizon payloads stay in sync with the standard recommendation.
    """
    def _naive(d):
        return d.replace(tzinfo=None) if d and d.tzinfo else d

    today = datetime.utcnow().date()
    today_acts = [a for a in activities if a.date and _naive(a.date).date() == today]
    today_tss = sum((a.tss or 0.0) for a in today_acts)
    today_dur_min = sum((a.duration_seconds or 0) for a in today_acts) / 60.0

    nw = payload.get("next_workout")
    if today_acts and nw is not None and (today_tss >= 30 or today_dur_min >= 45):
        nw = dict(nw)
        nw["day_offset"] = 1
        nw["already_rode_today"] = True
        nw["today_tss"] = round(today_tss, 1)
        nw["today_duration_minutes"] = int(today_dur_min)
        existing_rationale = nw.get("rationale", "")
        nw["rationale"] = (
            f"You've already ridden today ({int(today_dur_min)} min, "
            f"{int(today_tss)} TSS). Suggested workout moved to tomorrow. "
            f"{existing_rationale}"
        ).strip()
        payload["next_workout"] = nw

        weekly_plan = [dict(w) for w in payload.get("weekly_plan", [])]
        if weekly_plan:
            weekly_plan[0] = nw
            for i, w in enumerate(weekly_plan[1:], start=1):
                w["day_offset"] = 1 + i
            payload["weekly_plan"] = weekly_plan

    return payload


# ── Multi-horizon recommendation (Phase-1 / Option A) ─────────────────────────
async def generate_multi_horizon_recommendation(
    user: User, db: AsyncSession
) -> dict:
    """
    Returns 3 alternative "next workout" recommendations, one per planning horizon:

      • short  — 7-day FTP-gain horizon  ("what gives me the biggest bump this week?")
      • medium — 28-day build horizon    ("what's the optimal 4-week move?")
      • event  — actual goal-event date  ("what should I do today to peak on race day?")

    Phase 1 implementation: probes the SAME model 3 times with different
    `days_to_event` inputs. The synthetic training data covered varying event
    timelines, so this elicits (weakly) different answers per horizon.
    Phase 1.5/B will add explicit horizon-conditioning + dedicated heads.
    """
    # Load profile, recent activities, fitness — same context as the standard rec.
    profile_result = await db.execute(
        sa_select(AthleteProfile).where(AthleteProfile.user_id == user.id)
    )
    profile = profile_result.scalar_one_or_none()

    act_result = await db.execute(
        sa_select(Activity)
        .where(
            Activity.user_id == user.id,
            Activity.review_status == "confirmed",
        )
        .order_by(desc(Activity.date))
        .limit(MODEL_SEQ_LEN)
    )
    activities: list[Activity] = list(reversed(act_result.scalars().all()))

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

    # Load health history for per-day feature lookup (same as standard rec).
    health_cutoff = datetime.utcnow() - timedelta(days=120)
    health_result = await db.execute(
        sa_select(HealthMetric)
        .where(
            HealthMetric.user_id == user.id,
            HealthMetric.date >= health_cutoff,
        )
        .order_by(HealthMetric.date)
    )
    health_metrics: list[HealthMetric] = list(health_result.scalars().all())
    health_by_date = per_day_health_features(health_metrics)

    count_result = await db.execute(
        sa_select(func.count()).select_from(Activity).where(Activity.user_id == user.id)
    )
    total_activities = count_result.scalar_one()

    model = _load_model()

    # If we don't have a model or enough history, generate cold-start recs for
    # all 3 horizons using the phase-appropriate schedule for each.
    if total_activities < COLD_START_THRESHOLD or model is None:
        recent_types = [a.workout_type for a in activities[-7:] if a.workout_type]
        from app.ml.cold_start import WEEKLY_PATTERNS, get_periodization_phase, _bias_pattern_for_event, WORKOUT_LIBRARY as WL

        cs_horizons: dict[str, dict] = {}
        for h_label, h_days in [("short", HORIZON_DAYS["short"]), ("medium", HORIZON_DAYS["medium"]), ("event", 90)]:
            wte = h_days // 7
            phase = get_periodization_phase(wte)
            cs = build_cold_start_recommendation(
                profile, ctl, atl, tsb, recent_types, total_activities
            )
            # Override the weekly plan with the horizon-specific phase pattern.
            phase_pattern = list(WEEKLY_PATTERNS.get(phase, WEEKLY_PATTERNS["build"]))
            event_type_val = getattr(profile, "event_type", None) if profile else None
            if event_type_val and phase in ("build", "peak"):
                phase_pattern = _bias_pattern_for_event(phase_pattern, event_type_val)
            ftp_val = (profile.ftp if profile and profile.ftp else 200) or 200
            weekly_plan = []
            for i, wt in enumerate(phase_pattern):
                t = WL.get(wt, WL["endurance"])
                weekly_plan.append({
                    "day_offset": i,
                    "workout_type": wt,
                    "duration_minutes": t["duration_minutes"],
                    "description": t["description"],
                    "structure": [dict(s) for s in t["structures"]],
                    "target_tss": t["target_tss"],
                    "rationale": t["rationale"],
                    "key_metric": t["key_metric"].format(
                        z2_lo=int(ftp_val * 0.56), z2_hi=int(ftp_val * 0.75)
                    ),
                })
            cs["weekly_plan"] = weekly_plan
            cs["next_workout"] = weekly_plan[0]
            cs["horizon"] = h_label
            cs["horizon_days"] = h_days
            cs["horizon_label"] = _horizon_label(h_label, h_days)
            cs_horizons[h_label] = cs

        return {
            "horizons":       cs_horizons,
            "is_cold_start":  True,
            "model_version":  cs_horizons["medium"].get("model_version", "cold_start_v1"),
            "active_horizon": "event" if (profile and profile.goal_event_date) else "medium",
        }

    # Compute the user's true days-to-event (used for the 'event' horizon).
    if profile and profile.goal_event_date:
        true_dte = max(
            0,
            (profile.goal_event_date.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).days,
        )
    else:
        true_dte = 90  # sensible default if user has no event set

    horizons_to_run = {
        "short":  HORIZON_DAYS["short"],
        "medium": HORIZON_DAYS["medium"],
        "event":  true_dte,
    }

    out: dict[str, dict] = {}
    for label, dte in horizons_to_run.items():
        payload = _transformer_recommendation(
            model, activities, profile, ctl, atl, tsb,
            horizon_override_days=dte,
            health_by_date=health_by_date,
        )
        # Apply "already rode today" shift so the coach page weekly_plan
        # stays in sync with the standard recommendation's next_workout.
        _apply_today_ride_guard(payload, activities)
        payload["horizon"]              = label
        payload["horizon_days"]         = dte
        payload["horizon_label"]        = _horizon_label(label, dte)
        out[label] = payload

    # Pick the active horizon: prefer 'event' if user has one set, else 'medium'.
    active = "event" if (profile and profile.goal_event_date) else "medium"

    # Attach supplement stack (rule-based, fast — runs in same request).
    supplements = await _supplement_stack_for_user(
        user, db, profile=profile, ctl=ctl,
        activities=activities,
        days_to_event=true_dte if (profile and profile.goal_event_date) else None,
        workout_focus=out["medium"]["next_workout"].get("workout_type") if out.get("medium") else None,
    )

    return {
        "horizons":       out,
        "is_cold_start":  False,
        "model_version":  out["medium"]["model_version"],
        "active_horizon": active,
        "supplements":    supplements,
    }


def _horizon_label(label: str, days: int) -> str:
    if label == "short":
        return f"Max FTP gain in next {days} days"
    if label == "medium":
        return f"Best for {days}-day build"
    if label == "event":
        if days <= 0:
            return "Event today — race-day prep"
        if days <= 14:
            return f"Peak for event in {days} days (taper window)"
        return f"Best path to peak on event day ({days} days out)"
    return label


async def _supplement_stack_for_user(
    user: User,
    db: AsyncSession,
    *,
    profile: AthleteProfile | None,
    ctl: float,
    activities: list[Activity],
    days_to_event: int | None,
    workout_focus: str | None,
) -> dict:
    """
    Compute the rule-based supplement stack for a user. Pulls latest blood test
    if present. Lightweight enough to run in the same request as the recommendation.
    """
    from app.models.nutrition import BloodTest
    from app.nutrition.engine import recommend_supplements

    # Last 28 d windows
    cutoff = datetime.now(timezone.utc) - timedelta(days=28)
    recent = [a for a in activities if a.date and a.date >= cutoff.replace(tzinfo=None)] \
        if activities else []
    total_seconds = sum((a.duration_seconds or 0) for a in recent)
    weekly_hours = (total_seconds / 3600.0) / 4.0
    weekly_tss = sum((a.tss or 0) for a in recent) / 4.0
    temps = [a.temperature_c for a in recent if a.temperature_c is not None]
    recent_avg_temp = (sum(temps) / len(temps)) if temps else None

    bt_res = await db.execute(
        sa_select(BloodTest)
        .where(BloodTest.user_id == user.id)
        .order_by(desc(BloodTest.test_date))
        .limit(1)
    )
    bt = bt_res.scalar_one_or_none()
    blood_test = ({"id": bt.id, "markers": bt.markers} if bt else None)

    profile_dict = {
        "sex":     profile.sex if profile else None,
        "age":     profile.age if profile else None,
        "diet":    profile.diet if profile else None,
        "climate": profile.climate if profile else None,
        "training_days_per_week": profile.training_days_per_week if profile else None,
        "recent_illness_count_3m": profile.recent_illness_count_3m if profile else 0,
    }

    return recommend_supplements(
        profile=profile_dict,
        weekly_hours=weekly_hours,
        ctl=ctl,
        weekly_tss=weekly_tss,
        recent_avg_temp_c=recent_avg_temp,
        upcoming_event_type=(profile.event_type if profile else None),
        days_to_event=days_to_event,
        workout_focus=workout_focus,
        blood_test=blood_test,
    )

