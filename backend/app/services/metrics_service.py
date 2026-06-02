"""
Compute cycling metrics for activities and the Performance Management Chart.
Implements Coggan's TSS, CTL/ATL/TSB (exponential moving averages).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select as sa_select, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.activity import Activity
from app.models.user import User, AthleteProfile
from app.models.recommendation import FitnessMetric
from app.ml.quality import score_activity

# EMA constants
CTL_DAYS = 42  # Chronic Training Load (fitness)
ATL_DAYS = 7   # Acute Training Load (fatigue)
CTL_ALPHA = 2 / (CTL_DAYS + 1)
ATL_ALPHA = 2 / (ATL_DAYS + 1)


def compute_activity_metrics(activity: Activity, user: User | None) -> None:
    """
    Fill derived metrics: IF, TSS, NP (if missing), HR drift, aerobic efficiency.
    Modifies activity in place.
    """
    ftp = _get_ftp(user)

    # Intensity Factor = NP / FTP
    if activity.normalized_power and ftp:
        activity.intensity_factor = round(activity.normalized_power / ftp, 3)

    # TSS = (duration_s × NP × IF) / (FTP × 3600) × 100
    if activity.normalized_power and activity.intensity_factor and ftp and activity.duration_seconds:
        raw_tss = (
            activity.duration_seconds
            * activity.normalized_power
            * activity.intensity_factor
            / (ftp * 3600)
            * 100
        )
        activity.tss = round(raw_tss, 1)

    # Fallback: use avg_power as proxy for NP when NP is not available
    if not activity.tss and activity.avg_power and ftp and activity.duration_seconds:
        if_ = activity.avg_power / ftp
        activity.intensity_factor = round(if_, 3)
        activity.tss = round((activity.duration_seconds / 3600) * (if_ ** 2) * 100, 1)

    # TSS from HR if no power (estimate using HR reserve)
    if not activity.tss and activity.avg_hr and user:
        profile = getattr(user, "profile", None)
        if profile and profile.max_hr and profile.resting_hr:
            hrr_frac = (activity.avg_hr - profile.resting_hr) / (
                profile.max_hr - profile.resting_hr
            )
            # Approximate IF from HR fraction (rough)
            if_hr = hrr_frac ** 0.91
            activity.intensity_factor = round(if_hr, 3)
            if activity.duration_seconds:
                activity.tss = round(
                    (activity.duration_seconds / 3600) * (if_hr ** 2) * 100, 1
                )

    # Variability Index = NP / AP
    if activity.normalized_power and activity.avg_power and activity.avg_power > 0:
        activity.variability_index = round(activity.normalized_power / activity.avg_power, 3)

    # Aerobic Efficiency = avg_power / avg_hr  (w/bpm)
    if activity.avg_power and activity.avg_hr and activity.avg_hr > 0:
        activity.aerobic_efficiency = round(activity.avg_power / activity.avg_hr, 3)

    # Classify workout type from IF/TSS if not already set
    if not activity.workout_type and activity.intensity_factor:
        activity.workout_type = _classify_workout(activity.intensity_factor, activity.duration_seconds)


def _get_ftp(user: User | None) -> int | None:
    if user is None:
        return None
    profile = getattr(user, "profile", None)
    if profile and profile.ftp:
        return profile.ftp
    return 200  # fallback


def _score_and_tag(activity: Activity, user: User | None) -> None:
    """
    Apply rule-based quality scoring to a freshly-ingested activity.

    Sets activity.quality_score, activity.quality_reasons, activity.review_status.
    Auto-quarantines obviously bad rides; everything else stays 'confirmed'.
    Phase-2 outlier detection (vs the user's own history) is added separately.
    """
    profile = getattr(user, "profile", None) if user else None
    score, reasons = score_activity(
        {
            "duration_seconds":  activity.duration_seconds,
            "distance_meters":   activity.distance_meters,
            "avg_power":         activity.avg_power,
            "max_power":         activity.max_power,
            "normalized_power":  activity.normalized_power,
            "intensity_factor":  activity.intensity_factor,
            "tss":               activity.tss,
            "avg_hr":            activity.avg_hr,
            "max_hr":            activity.max_hr,
            "avg_cadence":       activity.avg_cadence,
            "trainer":           activity.trainer,
            "time_in_zones":     activity.time_in_zones,
        },
        profile_max_hr=getattr(profile, "max_hr", None),
        profile_ftp=getattr(profile, "ftp", None),
    )
    activity.quality_score   = score
    activity.quality_reasons = reasons or None
    activity.review_status   = "quarantined" if score == "rejected" else "confirmed"


def _classify_workout(if_: float, duration_s: int) -> str:
    """Classify workout type from Intensity Factor and duration.

    Duration matters at low intensity: a 10-min spin is recovery;
    a 4-hour ride at the same IF is endurance or long_ride.
    Recovery rides are short by definition (Friel: ≤45 min).
    """
    if if_ < 0.55:
        if duration_s < 2700:    # < 45 min → genuine recovery spin
            return "recovery"
        if duration_s < 7200:    # 45 min – 2 h → easy aerobic
            return "easy"
        if duration_s < 14400:   # 2 h – 4 h → endurance
            return "endurance"
        return "long_ride"       # > 4 h at low IF → long endurance ride
    if if_ < 0.75:
        return "easy" if duration_s < 5400 else "endurance"
    if if_ < 0.88:
        return "tempo"
    if if_ < 0.95:
        return "sweetspot"
    if if_ < 1.05:
        return "threshold"
    if if_ < 1.20:
        return "vo2max"
    return "sprint"


_FTP_TEST_PATTERNS = [
    "ftp bike test", "ftp test", "zwift ftp", "ramp test",
    "20 min ftp", "20 minute ftp", "20min ftp", "threshold test",
    "ftp ramp", "short power build",
]


def detect_ftp_test(activity_name: str | None, workout_type: str | None = None) -> str | None:
    """Return test type string if the activity looks like an FTP test, else None."""
    name_lower = (activity_name or "").lower()
    for pat in _FTP_TEST_PATTERNS:
        if pat in name_lower:
            if "ramp" in name_lower:
                return "ftp_ramp"
            if "8 min" in name_lower or "8min" in name_lower:
                return "ftp_8min"
            return "ftp_20min"
    if workout_type and "threshold" in workout_type.lower():
        return "ftp_20min"
    return None


async def estimate_ftp_from_activities(
    user_id: str,
    db: AsyncSession,
    manual_ftp: int | None = None,
) -> dict:
    """
    Multi-source, recency-weighted, robust FTP estimator.

    Why this beats ZwiftPower's "best 20 min × 0.95"
    ------------------------------------------------
    ZwiftPower picks ONE peak effort and applies a fixed 0.95 multiplier.
    That throws away most of the signal in your training history and is
    sensitive to a single anomalous ride (a tailwind, a downhill segment,
    a power-meter spike).

    This estimator instead:
      1. Derives a **per-ride FTP estimate** from many rides using
         duration-aware physiology coefficients (Coggan / WKO-style).
      2. Weights each estimate by (recency × duration confidence).
      3. Uses a **weighted 75th-percentile** (not max, not mean) — robust
         to outliers but representative of true threshold capability.
      4. Applies a **TSB fatigue correction** (rides done tired
         under-represent FTP).
      5. Splits into "recent (30d)" and "older (30-180d)" buckets to
         detect trend and slightly bias toward the more recent estimate
         when the athlete is improving.
      6. Reports a **confidence band** (low/high watts) from the spread
         of contributing estimates.

    Per-ride estimate coefficients (NP × coefficient → FTP):
      duration ≥ 60 min  → NP × 1.00          (long hard effort, NP ≈ FTP)
      duration ≥ 45 min  → NP × 0.97          (sub-hour, slight discount)
      duration ≥ 20 min  → NP × 0.95          (classic 20-min test)
      duration ≥  8 min  → NP × 0.90          (Carmichael 8-min)
      otherwise → skip

    No IF-based gating: any ride of sufficient duration with NP ≥ 80W
    contributes (with appropriate confidence weight).  The robust 75th-
    percentile then picks the rider's true threshold capability while
    discarding casual recovery rides.

    Rides without NP are skipped (no avg-power fallback, since avg power
    on real rides under-estimates threshold capability badly).

    Returns
    -------
    dict with:
      estimated_ftp        – int (or None)
      confidence           – 0-1 (effective sample size + recency)
      confidence_low       – low end of 80% band
      confidence_high      – high end of 80% band
      best_ride_age_days   – age of newest contributing ride
      tsb_correction       – multiplier applied for fatigue
      method               – "power_weighted" | "blended" | "manual_fallback" | "no_data"
      sample_count         – number of rides that contributed
      trend                – "improving" | "stable" | "declining"
    """
    from app.models.tracking import PerformanceTest
    from app.models.user import AthleteProfile as _AP

    RECENCY_TAU_DAYS = 60.0
    STALENESS_DAYS = 365
    FRESH_TEST_DAYS  = 28   # < 4 weeks → full trust
    BLEND_TEST_DAYS  = 84   # 4-12 weeks → blend test + rides
    # > 12 weeks → test becomes a soft anchor, rides take over
    now = datetime.utcnow()
    cutoff = now - timedelta(days=365 * 3)

    # ── Fetch most recent FTP test (any age — we'll apply smart aging below) ──
    test_r = await db.execute(
        sa_select(PerformanceTest)
        .where(
            PerformanceTest.user_id == user_id,
            PerformanceTest.test_type.in_(["ftp_20min", "ftp_ramp", "ftp_8min"]),
        )
        .order_by(PerformanceTest.test_date.desc())
        .limit(1)
    )
    latest_test = test_r.scalar_one_or_none()

    # ── Fetch current CTL + CTL at test date (for fitness-retention aging) ────
    fm_r = await db.execute(
        sa_select(FitnessMetric)
        .where(FitnessMetric.user_id == user_id)
        .order_by(FitnessMetric.date.desc())
        .limit(1)
    )
    today_fm = fm_r.scalar_one_or_none()
    current_ctl = today_fm.ctl if today_fm else 0.0
    current_tsb = today_fm.tsb if today_fm else 0.0

    test_ctl: float | None = None
    test_age_days: int = 9999
    if latest_test:
        test_d = latest_test.test_date.replace(tzinfo=None)
        test_age_days = max((now - test_d).days, 0)
        # CTL at test date — find the closest fitness metric row
        fm_test_r = await db.execute(
            sa_select(FitnessMetric)
            .where(
                FitnessMetric.user_id == user_id,
                FitnessMetric.date <= latest_test.test_date,
            )
            .order_by(FitnessMetric.date.desc())
            .limit(1)
        )
        fm_at_test = fm_test_r.scalar_one_or_none()
        test_ctl = fm_at_test.ctl if fm_at_test else None

    # ── Case 1: test < 4 weeks → full trust, return immediately ──────────────
    if latest_test and test_age_days <= FRESH_TEST_DAYS:
        ftp_v = int(round(latest_test.value))
        return {
            "estimated_ftp": ftp_v,
            "confidence": 0.97,
            "confidence_low": int(round(ftp_v * 0.97)),
            "confidence_high": int(round(ftp_v * 1.03)),
            "best_ride_age_days": test_age_days,
            "tsb_correction": 1.0,
            "method": f"verified_test:{latest_test.test_type}",
            "sample_count": 1,
            "trend": "stable",
        }

    # ── Compute age-adjusted test FTP (for blend or anchor) ──────────────────
    aged_test_ftp: float | None = None
    if latest_test and test_age_days <= STALENESS_DAYS:
        raw = float(latest_test.value)
        # CTL retention: if you maintained training, FTP decays minimally.
        # If CTL dropped a lot → the test result is stale, apply fitness decay.
        if test_ctl and test_ctl > 5 and current_ctl > 0:
            ctl_ratio = min(current_ctl / test_ctl, 1.0)
            # Aerobic fitness decays ~1%/week of detraining beyond week 2.
            # We scale by CTL ratio so maintained training = no decay.
            weeks_exposed = max(0, test_age_days - 14) / 7
            max_decay = min(weeks_exposed * 0.01 * (1.0 - ctl_ratio), 0.15)
            aged_test_ftp = raw * (1.0 - max_decay)
        else:
            # No CTL history — apply mild calendar decay: 0.5%/week after 4 weeks
            excess_weeks = max(0, test_age_days - FRESH_TEST_DAYS) / 7
            aged_test_ftp = raw * (1.0 - min(excess_weeks * 0.005, 0.10))

    # ── Fetch user weight for power-curve watts conversion ────────────────────
    prof_r = await db.execute(
        sa_select(_AP).where(_AP.user_id == user_id).limit(1)
    )
    profile = prof_r.scalar_one_or_none()
    weight_kg: float | None = getattr(profile, "weight_kg", None)

    result = await db.execute(
        sa_select(Activity)
        .where(
            Activity.user_id == user_id,
            Activity.duration_seconds >= 480,
            Activity.date >= cutoff,
        )
        .order_by(Activity.date)
    )
    activities = result.scalars().all()

    estimates: list[dict] = []

    for act in activities:
        dur = act.duration_seconds or 0

        # ── Power-curve estimate (gold standard when available) ───────────────
        # pc_20min_wkg = best 20-min average W/kg from the raw power stream.
        # This is more accurate than whole-ride NP because it captures the
        # actual peak sustainable effort, not a ride average.
        is_ftp_test = bool(detect_ftp_test(act.name, act.workout_type))
        if act.pc_20min_wkg and weight_kg:
            best_20min_w = act.pc_20min_wkg * weight_kg
            est = best_20min_w * 0.95   # standard 20-min → FTP conversion
            # FTP tests get confidence 1.0; regular rides are slightly lower
            dur_conf = 1.0 if is_ftp_test else 0.90
        else:
            # ── Whole-ride NP estimate (fallback for Strava API imports) ──────
            np_ = act.normalized_power
            if not np_ or np_ < 80:
                continue
            power = float(np_)
            if dur >= 3600:
                coef, dur_conf = 1.00, 1.00
            elif dur >= 2700:
                coef, dur_conf = 0.97, 0.95
            elif dur >= 1200:
                coef, dur_conf = 0.95, 0.70
            elif dur >= 480:
                coef, dur_conf = 0.90, 0.40
            else:
                continue
            if is_ftp_test:
                dur_conf = min(dur_conf * 1.5, 1.0)  # boost confidence for detected tests
            est = power * coef

        if est < 80 or est > 600:
            continue

        act_date = act.date.replace(tzinfo=None) if act.date and act.date.tzinfo else act.date
        age_days = max((now - act_date).days, 0) if act_date else 9999
        recency_w = math.exp(-age_days / RECENCY_TAU_DAYS)
        weight = recency_w * dur_conf

        estimates.append({
            "est": est,
            "weight": weight,
            "age_days": age_days,
            "dur_conf": dur_conf,
        })

    n = len(estimates)
    has_power_data = n > 0
    best_ride_age_days = min((e["age_days"] for e in estimates), default=None)

    # ── Fallback: no usable rides ─────────────────────────────────────────────
    if not has_power_data:
        if manual_ftp and manual_ftp >= 80:
            return {
                "estimated_ftp": manual_ftp,
                "confidence": 0.2,
                "confidence_low": int(round(manual_ftp * 0.92)),
                "confidence_high": int(round(manual_ftp * 1.08)),
                "best_ride_age_days": None,
                "tsb_correction": 1.0,
                "method": "manual_fallback",
                "sample_count": 0,
                "trend": "stable",
            }
        return {
            "estimated_ftp": None,
            "confidence": 0.0,
            "confidence_low": None,
            "confidence_high": None,
            "best_ride_age_days": None,
            "tsb_correction": 1.0,
            "method": "no_data",
            "sample_count": 0,
            "trend": "stable",
        }

    # ── Stale fallback ────────────────────────────────────────────────────────
    if best_ride_age_days is not None and best_ride_age_days > STALENESS_DAYS:
        if manual_ftp and manual_ftp >= 80:
            return {
                "estimated_ftp": manual_ftp,
                "confidence": 0.3,
                "confidence_low": int(round(manual_ftp * 0.90)),
                "confidence_high": int(round(manual_ftp * 1.10)),
                "best_ride_age_days": int(best_ride_age_days),
                "tsb_correction": 1.0,
                "method": "manual_fallback",
                "sample_count": n,
                "trend": "stable",
            }
        # No manual to fall back on — use stale data with low confidence
        sorted_ests = sorted(e["est"] for e in estimates)
        stale_est = int(round(sorted_ests[len(sorted_ests) // 2]))
        return {
            "estimated_ftp": stale_est,
            "confidence": 0.3,
            "confidence_low": int(round(min(sorted_ests))),
            "confidence_high": int(round(max(sorted_ests))),
            "best_ride_age_days": int(best_ride_age_days),
            "tsb_correction": 1.0,
            "method": "power_weighted",
            "sample_count": n,
            "trend": "stable",
        }

    # ── Robust weighted percentile (75th) ─────────────────────────────────────
    estimates.sort(key=lambda e: e["est"])
    total_w = sum(e["weight"] for e in estimates)
    cum = 0.0
    p75 = estimates[-1]["est"]
    for e in estimates:
        cum += e["weight"]
        if cum / total_w >= 0.75:
            p75 = e["est"]
            break

    # Confidence band: weighted 10th and 90th percentiles
    cum = 0.0
    p10 = estimates[0]["est"]
    for e in estimates:
        cum += e["weight"]
        if cum / total_w >= 0.10:
            p10 = e["est"]; break
    cum = 0.0
    p90 = estimates[-1]["est"]
    for e in estimates:
        cum += e["weight"]
        if cum / total_w >= 0.90:
            p90 = e["est"]; break

    weighted_ftp = p75

    # ── Trend: recent (≤ 30d) vs older (30-180d) ──────────────────────────────
    recent = [e["est"] for e in estimates if e["age_days"] <= 30]
    older = [e["est"] for e in estimates if 30 < e["age_days"] <= 180]
    trend = "stable"
    if recent and older:
        r_mean = sum(recent) / len(recent)
        o_mean = sum(older) / len(older)
        delta = (r_mean - o_mean) / o_mean if o_mean else 0
        if delta > 0.03:
            trend = "improving"
            weighted_ftp = 0.7 * r_mean + 0.3 * weighted_ftp  # bias toward recent
        elif delta < -0.03:
            trend = "declining"
            weighted_ftp = 0.7 * r_mean + 0.3 * weighted_ftp

    # ── Fatigue correction (TSB) ──────────────────────────────────────────────
    tsb = current_tsb
    ctl = current_ctl
    fatigue_depth = max(-tsb, 0.0)
    tsb_correction = 1.0 + min(fatigue_depth, 30.0) * 0.003  # max +9%
    corrected_ftp = weighted_ftp * tsb_correction

    # ── Effective sample-size confidence ──────────────────────────────────────
    sum_w2 = sum(e["weight"] ** 2 for e in estimates)
    ess = (total_w ** 2) / sum_w2 if sum_w2 > 0 else 0
    ride_confidence = min(ess / 5.0, 1.0)
    ctl_confidence = min(ctl / 20.0, 1.0)
    confidence = min(ride_confidence, ctl_confidence)

    # ── Blend: rides + aged test + manual FTP ────────────────────────────────
    # Priority hierarchy (highest → lowest):
    #   1. aged_test_ftp  (test 4-12 weeks old, fitness-retention adjusted)
    #   2. corrected_ftp  (multi-ride weighted estimate)
    #   3. manual_ftp     (profile value, used when confidence is low)
    #
    # Rule: if recent rides show HIGHER power than the aged test → rides win
    # (you improved). If rides show lower → aged test wins (likely just fatigue).
    method = "power_weighted"

    if aged_test_ftp is not None and test_age_days <= BLEND_TEST_DAYS:
        # 4-12 week range: blend test weight from 1.0→0.0 linearly
        test_w = 1.0 - (test_age_days - FRESH_TEST_DAYS) / (BLEND_TEST_DAYS - FRESH_TEST_DAYS)
        ride_w = 1.0 - test_w
        blended_ftp = test_w * aged_test_ftp + ride_w * corrected_ftp
        # If recent rides exceed the aged test, let them override (you improved)
        blended_ftp = max(blended_ftp, corrected_ftp)
        method = "test_blend"
    elif aged_test_ftp is not None:
        # > 12 weeks: test is a soft floor — don't go below it by more than 15%
        floor = aged_test_ftp * 0.85
        blended_ftp = max(corrected_ftp, floor)
        method = "test_anchored" if blended_ftp > corrected_ftp else "power_weighted"
    elif manual_ftp and manual_ftp >= 80 and confidence < 1.0:
        blended_ftp = corrected_ftp * confidence + manual_ftp * (1.0 - confidence)
        method = "blended"
    else:
        blended_ftp = corrected_ftp

    final_ftp = int(round(blended_ftp))
    if final_ftp < 80:
        return {
            "estimated_ftp": None,
            "confidence": 0.0,
            "confidence_low": None,
            "confidence_high": None,
            "best_ride_age_days": int(best_ride_age_days),
            "tsb_correction": round(tsb_correction, 3),
            "method": method,
            "sample_count": n,
            "trend": trend,
        }

    return {
        "estimated_ftp": final_ftp,
        "confidence": round(confidence, 2),
        "confidence_low": int(round(p10 * tsb_correction)),
        "confidence_high": int(round(p90 * tsb_correction)),
        "best_ride_age_days": int(best_ride_age_days),
        "tsb_correction": round(tsb_correction, 3),
        "method": method,
        "sample_count": n,
        "trend": trend,
        "last_test_age_days": test_age_days if test_age_days < 9999 else None,
        "aged_test_ftp": int(round(aged_test_ftp)) if aged_test_ftp else None,
    }


async def compute_pmc_for_user(user_id: str, db: AsyncSession) -> None:
    """
    Recompute the full Performance Management Chart for a user.
    Deletes existing FitnessMetric rows and rebuilds from scratch.

    Note: only `confirmed` (non-quarantined, non-deleted) activities count.
    The user's current profile FTP is used for every day in the series so
    the FTP column doesn't oscillate between profile_ftp and a default.
    """
    # Fetch profile once — used as the canonical FTP for every day
    prof_result = await db.execute(
        sa_select(AthleteProfile).where(AthleteProfile.user_id == user_id)
    )
    profile = prof_result.scalar_one_or_none()
    user_ftp = float(profile.ftp) if profile and profile.ftp else 200.0

    # Fetch all *confirmed* activities ordered by date
    result = await db.execute(
        sa_select(Activity)
        .where(
            Activity.user_id == user_id,
            Activity.review_status == "confirmed",
        )
        .order_by(Activity.date)
    )
    activities = result.scalars().all()

    if not activities:
        # Still wipe any stale metrics so the UI doesn't show old numbers
        await db.execute(sa_delete(FitnessMetric).where(FitnessMetric.user_id == user_id))
        await db.flush()
        return

    # Build daily TSS map
    daily_tss: dict[str, float] = {}
    for act in activities:
        day = act.date.date().isoformat()
        daily_tss[day] = daily_tss.get(day, 0.0) + (act.tss or 0.0)

    # Generate continuous date range
    start_date = min(datetime.fromisoformat(d).replace(tzinfo=timezone.utc) for d in daily_tss)
    end_date = datetime.now(timezone.utc)

    # Delete existing metrics
    await db.execute(sa_delete(FitnessMetric).where(FitnessMetric.user_id == user_id))

    ctl = 0.0
    atl = 0.0
    new_metrics = []
    current = start_date

    while current <= end_date:
        day = current.date().isoformat()
        tss = daily_tss.get(day, 0.0)

        ctl = ctl + CTL_ALPHA * (tss - ctl)
        atl = atl + ATL_ALPHA * (tss - atl)
        tsb = ctl - atl

        metric = FitnessMetric(
            user_id=user_id,
            date=current.replace(tzinfo=timezone.utc) if current.tzinfo is None else current,
            ctl=round(ctl, 2),
            atl=round(atl, 2),
            tsb=round(tsb, 2),
            tss=round(tss, 2),
            ftp=user_ftp,
        )
        new_metrics.append(metric)
        current += timedelta(days=1)

    db.add_all(new_metrics)
    await db.flush()
