"""
HR drift (aerobic decoupling) analysis.

Scientific basis:
  EF = HR / Power  (bpm per watt — internal cost per unit of work)
  Drift = (EF₂ / EF₁ − 1) × 100

  Positive drift = HR rose relative to power in second half → cardiac drift.
  Thresholds (Coggan / Friel):
    Drift < 5%     Stable     — cardiovascular system matched to mechanical output
    Drift 5–10%    Decoupled  — aerobic base building needed; cap intensity
    Drift > 10%    Stressed   — regressive load; check HRV before next session
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from typing import Sequence

STABLE_THRESHOLD    = 5.0   # %
DECOUPLED_THRESHOLD = 10.0  # %


@dataclass
class DriftAssessment:
    state: str               # "stable" | "decoupled" | "stressed"
    drift_pct: float | None  # latest qualified drift value (%)
    trend: str               # "improving" | "stable" | "worsening"
    overtraining_risk: bool  # Drift/HRV matrix fires when both are bad
    action: str              # "progression" | "consolidation" | "regressive_load"
    note: str                # human-readable coaching note


def compute_hr_drift_from_streams(
    power_values: list[float],
    hr_values: list[float],
    *,
    min_points: int = 60,   # ~1 min of data at 1 Hz — below this we return None
    max_cv: float = 0.15,   # power CV above this = not steady state
) -> tuple[float | None, bool]:
    """Compute aerobic decoupling from parallel power + HR streams.

    Returns (drift_pct, is_steady_state).
    drift_pct is None when data is too sparse or degenerate.
    is_steady_state is True when power CV ≤ max_cv (ride was evenly paced).

    Callers should still store the drift even when is_steady_state=False — it
    carries signal for interval sessions and hilly rides, just with lower
    confidence.
    """
    n = min(len(power_values), len(hr_values))
    pairs = [
        (float(p), float(h))
        for p, h in zip(power_values[:n], hr_values[:n])
        if p is not None and h is not None and p > 0 and h > 0
    ]

    if len(pairs) < min_points:
        return None, False

    powers = [p for p, _ in pairs]
    avg_p = mean(powers)
    if avg_p <= 0:
        return None, False

    sd_p = stdev(powers) if len(powers) > 1 else 0.0
    cv = sd_p / avg_p
    is_steady = cv <= max_cv

    mid = len(pairs) // 2
    first, second = pairs[:mid], pairs[mid:]

    avg_p1 = mean(p for p, _ in first)
    avg_h1 = mean(h for _, h in first)
    avg_p2 = mean(p for p, _ in second)
    avg_h2 = mean(h for _, h in second)

    if avg_p1 <= 0 or avg_p2 <= 0:
        return None, is_steady

    ef1 = avg_h1 / avg_p1  # bpm / watt
    ef2 = avg_h2 / avg_p2
    if ef1 <= 0:
        return None, is_steady

    drift = (ef2 / ef1 - 1) * 100
    return round(drift, 2), is_steady


def classify_drift_state(drift_pct: float | None) -> str:
    if drift_pct is None:
        return "stable"
    if drift_pct < STABLE_THRESHOLD:
        return "stable"
    if drift_pct < DECOUPLED_THRESHOLD:
        return "decoupled"
    return "stressed"


def get_drift_assessment(
    activities: Sequence,
    *,
    hrv_z: float | None = None,
    lookback: int = 5,
) -> DriftAssessment:
    """Analyse recent hr_drift values and combine with HRV for overtraining risk.

    `activities` should be sorted oldest-first (same order as the inference
    sequence). Works with both ORM Activity objects and plain dicts.
    """
    recent_drifts: list[float] = []
    for act in reversed(list(activities)):
        d = getattr(act, "hr_drift", None) if not isinstance(act, dict) else act.get("hr_drift")
        if d is not None and isinstance(d, (int, float)):
            recent_drifts.append(float(d))
            if len(recent_drifts) >= lookback:
                break

    latest_drift = recent_drifts[0] if recent_drifts else None
    state = classify_drift_state(latest_drift)

    # Trend: more-recent half vs older half of the window
    trend = "stable"
    if len(recent_drifts) >= 3:
        half = len(recent_drifts) // 2
        newer_avg = mean(recent_drifts[:half])
        older_avg = mean(recent_drifts[half:])
        if newer_avg < older_avg - 1.5:
            trend = "improving"
        elif newer_avg > older_avg + 1.5:
            trend = "worsening"

    # Drift / HRV Matrix — Predictive Fatigue Analysis
    # Overtraining risk fires 3–5 days BEFORE the athlete feels exhausted
    overtraining_risk = False
    if state == "stressed" and hrv_z is not None and hrv_z < -0.5:
        overtraining_risk = True
    elif state == "decoupled" and trend == "worsening" and hrv_z is not None and hrv_z < -0.3:
        overtraining_risk = True

    # Action + coaching note
    if state == "stable":
        action = "progression"
        if latest_drift is not None:
            note = (
                f"HR drift {latest_drift:.1f}% — aerobic system matched to output. "
                "Nudge FTP target up 2–3% or introduce Zone 4/5 intervals."
            )
        else:
            note = "No steady-state drift data yet — upload a 60+ min even-pace ride to enable this signal."
    elif state == "decoupled":
        action = "consolidation"
        note = (
            f"HR drift {latest_drift:.1f}% — internal cost rising significantly. "
            "Hold current volume, cap intensity at sweetspot. Do not increase FTP targets yet."
        )
    else:  # stressed
        action = "regressive_load"
        note = (
            f"HR drift {latest_drift:.1f}% — high metabolic cost detected. "
            "Reduce ride duration and verify recovery metrics before next hard session."
        )
        if overtraining_risk:
            note += (
                " ⚠ Drift/HRV matrix: rising drift + suppressed HRV — "
                "overtraining event predicted 3–5 days out. Prioritise rest now."
            )

    return DriftAssessment(
        state=state,
        drift_pct=latest_drift,
        trend=trend,
        overtraining_risk=overtraining_risk,
        action=action,
        note=note,
    )
