"""
Readiness score: combines HRV, resting HR, sleep, and body battery into a single
0–100 score that gates the next workout's intensity.

Scientific basis:
  - Plews & Buchheit (2013) — HRV-guided endurance training. Day-to-day Ln(RMSSD)
    deviation from a 7-day rolling baseline is the most actionable HRV signal.
    > +0.5 SD = good adaptation; < -0.5 SD = parasympathetic withdrawal.
  - Buchheit (2014) — Monitoring training status with HR + HRV. RHR drifting
    upward (+5 bpm vs 30-day baseline) signals accumulated fatigue or illness.
  - Stanley, Peake & Buchheit (2013) — Time course of HR recovery: Z1/Z2 < 24h,
    threshold 24–48h, VO2max 48–72h.
  - Vesterinen et al. (2016) — HRV-guided training improves V̇O2max more than
    pre-planned blocks; weekly average is more stable than daily values.

The score is intentionally conservative: a single bad night should reduce
intensity, not cancel training. Persistent multi-day red zones trigger forced
recovery via the safety guard.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

from app.models.health import HealthMetric


@dataclass
class ReadinessSnapshot:
    score: float                     # 0–100
    status: str                      # "green" | "amber" | "red"
    hrv_z: float | None              # z-score of last-night HRV vs 7d baseline
    rhr_delta: float | None          # bpm above/below 30d baseline
    sleep_score: int | None
    body_battery: int | None
    hrv_score: float | None
    rhr_score: float | None
    drivers: list[str]               # human-readable bullets
    advice: str                      # one-line coaching cue


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _hrv_baseline(history: Sequence[HealthMetric], days: int = 7) -> tuple[float | None, float | None]:
    """Return (mean, std) of overnight HRV over the last `days` (excluding today)."""
    vals = [
        h.hrv_overnight_avg_ms for h in history[-(days + 1):-1]
        if h.hrv_overnight_avg_ms is not None
    ]
    if len(vals) < 3:
        return None, None
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
    std = max(var ** 0.5, 1.0)  # floor SD at 1 ms to avoid divide-by-zero blow-ups
    return mean, std


def _rhr_baseline(history: Sequence[HealthMetric], days: int = 30) -> float | None:
    vals = [
        h.resting_hr for h in history[-(days + 1):-1]
        if h.resting_hr is not None
    ]
    if len(vals) < 5:
        return None
    return sum(vals) / len(vals)


def per_day_health_features(
    history: Sequence[HealthMetric],
) -> dict:
    """
    Build a dict mapping `date.date()` → (hrv_z, rhr_delta, sleep_score, body_battery)
    using only past data at each point — i.e. no future leakage. Suitable for
    enriching ML features on historical activities.
    """
    out: dict = {}
    sorted_h = sorted(history, key=lambda m: m.date)
    for i, today in enumerate(sorted_h):
        prefix = sorted_h[: i + 1]
        baseline_mean, baseline_std = _hrv_baseline(prefix)
        rhr_baseline = _rhr_baseline(prefix)
        hrv_z = 0.0
        if today.hrv_overnight_avg_ms is not None and baseline_mean and baseline_std:
            hrv_z = (today.hrv_overnight_avg_ms - baseline_mean) / baseline_std
        rhr_delta = 0.0
        if today.resting_hr is not None and rhr_baseline:
            rhr_delta = float(today.resting_hr) - rhr_baseline
        sleep_score = float(today.sleep_score) if today.sleep_score is not None else 50.0
        body_battery = float(today.body_battery_high) if today.body_battery_high is not None else 50.0
        out[today.date.date()] = (
            float(hrv_z),
            float(rhr_delta),
            sleep_score,
            body_battery,
        )
    return out


def compute_readiness(
    history: Sequence[HealthMetric],
    *,
    drift_state: str | None = None,
    drift_pct: float | None = None,
) -> ReadinessSnapshot:
    """
    Compute today's readiness from a chronologically sorted list of HealthMetric.
    The LAST entry is treated as 'today'. Missing data is handled gracefully —
    each contributor only adds to the score if its source signal is available.
    """
    if not history:
        return ReadinessSnapshot(
            score=50.0, status="amber",
            hrv_z=None, rhr_delta=None, sleep_score=None, body_battery=None,
            hrv_score=None, rhr_score=None,
            drivers=["No health data yet — connect Garmin to enable readiness."],
            advice="Train as planned; readiness will activate after a few days of Garmin data.",
        )

    today = history[-1]
    drivers: list[str] = []

    # ── HRV ────────────────────────────────────────────────────────────────
    hrv_z = None
    hrv_score = None
    if today.hrv_overnight_avg_ms is not None:
        baseline_mean, baseline_std = _hrv_baseline(history)
        if baseline_mean and baseline_std:
            hrv_z = (today.hrv_overnight_avg_ms - baseline_mean) / baseline_std
            # Plews & Buchheit: +1 SD = strong recovery, -1 SD = parasympathetic withdrawal
            # Map z ∈ [-2, +2] → score ∈ [0, 100] via tanh squash.
            import math
            hrv_score = _clamp(50.0 + 50.0 * math.tanh(hrv_z / 1.5), 0.0, 100.0)
            if hrv_z <= -1.0:
                drivers.append(f"HRV {today.hrv_overnight_avg_ms:.0f} ms is "
                               f"{abs(hrv_z):.1f} SD below your 7-day baseline.")
            elif hrv_z >= 0.5:
                drivers.append(f"HRV {today.hrv_overnight_avg_ms:.0f} ms is above baseline — well recovered.")
        else:
            # Not enough baseline yet — neutral score from raw value vs Garmin status
            status_map = {"balanced": 70.0, "unbalanced": 45.0, "low": 30.0, "poor": 20.0}
            hrv_score = status_map.get((today.hrv_status or "").lower(), 60.0)

    # ── Resting HR ─────────────────────────────────────────────────────────
    rhr_delta = None
    rhr_score = None
    if today.resting_hr is not None:
        baseline = _rhr_baseline(history)
        if baseline:
            rhr_delta = today.resting_hr - baseline
            # +5 bpm above baseline → -25, +10 bpm → -50 (Buchheit 2014)
            rhr_score = _clamp(100.0 - 5.0 * rhr_delta, 0.0, 100.0)
            if rhr_delta >= 5:
                drivers.append(f"Resting HR is {rhr_delta:+.0f} bpm vs 30-day baseline — sympathetic load.")
            elif rhr_delta <= -3:
                drivers.append(f"Resting HR {rhr_delta:+.0f} bpm — strong recovery signal.")
        else:
            rhr_score = 60.0  # neutral until baseline is built

    # ── Sleep ──────────────────────────────────────────────────────────────
    sleep_score = today.sleep_score
    if sleep_score is not None:
        if sleep_score < 50:
            drivers.append(f"Poor sleep ({sleep_score}/100).")
        elif sleep_score >= 85:
            drivers.append(f"Excellent sleep ({sleep_score}/100).")

    # ── Body Battery ───────────────────────────────────────────────────────
    body_battery = today.body_battery_high
    if body_battery is not None and body_battery < 40:
        drivers.append(f"Body Battery only {body_battery}% — limited reserves.")

    # ── HR Drift (aerobic decoupling trend) ────────────────────────────────
    # Drift integrates cumulative physiological stress across recent rides,
    # complementing the acute HRV/RHR signals (which reflect only last night).
    drift_score: float | None = None
    if drift_state is not None and drift_pct is not None:
        if drift_state == "stable":
            drift_score = 85.0
        elif drift_state == "decoupled":
            drift_score = 50.0
            drivers.append(
                f"HR drift {drift_pct:.1f}% — aerobic consolidation phase. "
                "Cap intensity at sweetspot until drift normalises."
            )
        else:  # stressed
            drift_score = 20.0
            drivers.append(
                f"HR drift {drift_pct:.1f}% — high metabolic stress detected. "
                "Reduce session duration and verify other recovery signals."
            )

    # ── Weighted blend ─────────────────────────────────────────────────────
    parts: list[tuple[float, float]] = []  # (weight, score)
    if hrv_score is not None:
        parts.append((0.35, hrv_score))
    if rhr_score is not None:
        parts.append((0.20, rhr_score))
    if sleep_score is not None:
        parts.append((0.20, float(sleep_score)))
    if body_battery is not None:
        parts.append((0.15, float(body_battery)))
    if drift_score is not None:
        parts.append((0.10, drift_score))

    if not parts:
        score = 50.0
    else:
        total_w = sum(w for w, _ in parts)
        score = sum(w * s for w, s in parts) / total_w

    # ── Status + advice ────────────────────────────────────────────────────
    if score >= 70:
        status = "green"
        advice = "Green light — full session as planned. Hard intervals are productive today."
    elif score >= 45:
        status = "amber"
        advice = "Yellow — keep volume but cap intensity at sweetspot. Skip VO2max if you feel flat."
    else:
        status = "red"
        advice = "Red — Z1/Z2 spin only or rest. Pushing through low HRV/elevated RHR risks illness."

    if not drivers:
        drivers.append("All recovery signals within normal range.")

    return ReadinessSnapshot(
        score=round(score, 1),
        status=status,
        hrv_z=round(hrv_z, 2) if hrv_z is not None else None,
        rhr_delta=round(rhr_delta, 1) if rhr_delta is not None else None,
        sleep_score=sleep_score,
        body_battery=body_battery,
        hrv_score=round(hrv_score, 1) if hrv_score is not None else None,
        rhr_score=round(rhr_score, 1) if rhr_score is not None else None,
        drivers=drivers,
        advice=advice,
    )
