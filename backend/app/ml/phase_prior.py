"""
Bayesian phase prior over workout types.

Combines the transformer's learned softmax with a periodization prior so the
final recommendation respects:
  - Training phase (Friel: base / build / peak / taper / recovery)
  - Event-type bias (Coggan/Allen event specificity)
  - Fatigue safety (TSB / HRV)

Posterior: ``p(workout | history, horizon, phase) ∝ softmax(logits / T) ·
prior(workout | phase, event_type) · safety(workout | tsb, hrv_z)``

References:
  - Friel J., *The Cyclist's Training Bible* (5th ed.), Chap. 7-9.
  - Coggan & Allen, *Training and Racing with a Power Meter* (3rd ed.).
  - Seiler S. (2010). What is best practice for training intensity and
    duration distribution in endurance athletes? IJSPP 5(3): 276-291.
  - Plews & Buchheit (2013). Training adaptation and HRV in elite endurance
    athletes. Eur J Appl Physiol 113(7): 1605-1620.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from app.ml.cold_start import _EVENT_BIAS

# Workout types in the order the model emits logits.
WORKOUT_TYPE_NAMES = [
    "recovery", "easy", "endurance", "tempo", "sweetspot",
    "threshold", "vo2max", "sprint", "race", "long_ride",
]
N_TYPES = len(WORKOUT_TYPE_NAMES)
_NAME_TO_IDX = {n: i for i, n in enumerate(WORKOUT_TYPE_NAMES)}

# ── Phase priors ─────────────────────────────────────────────────────────────
# Empirical workout-type frequencies derived from the synthetic generator's
# weekly schedules across the three philosophies. Values are unnormalized
# pseudo-counts; they will be L1-normalized to a probability distribution
# before use. A small floor (0.02) is added everywhere so the model can
# still pick an "unusual" workout if its likelihood is overwhelmingly strong.
PHASE_PRIORS_RAW: dict[str, dict[str, float]] = {
    "base": {
        "endurance": 5.0, "easy": 3.0, "long_ride": 2.0, "recovery": 1.0,
        "tempo": 1.0, "sweetspot": 0.7, "threshold": 0.5, "vo2max": 0.3,
        "sprint": 0.2, "race": 0.05,
    },
    "base_build": {  # alias used by cold_start.WEEKLY_PATTERNS
        "endurance": 4.5, "easy": 2.5, "long_ride": 2.0, "recovery": 1.0,
        "tempo": 1.5, "sweetspot": 1.2, "threshold": 0.7, "vo2max": 0.4,
        "sprint": 0.3, "race": 0.05,
    },
    "build": {
        "endurance": 3.5, "easy": 1.5, "long_ride": 1.8, "recovery": 1.0,
        "tempo": 1.2, "sweetspot": 1.5, "threshold": 1.5, "vo2max": 1.5,
        "sprint": 0.5, "race": 0.1,
    },
    "peak": {
        "endurance": 2.0, "easy": 1.0, "long_ride": 1.0, "recovery": 1.0,
        "tempo": 0.8, "sweetspot": 1.2, "threshold": 1.8, "vo2max": 2.0,
        "sprint": 0.7, "race": 0.3,
    },
    "taper": {
        "endurance": 2.0, "easy": 1.5, "long_ride": 0.3, "recovery": 1.5,
        "tempo": 0.5, "sweetspot": 0.6, "threshold": 0.7, "vo2max": 0.8,
        "sprint": 0.6, "race": 0.5,
    },
    "recovery_week": {
        "endurance": 1.5, "easy": 3.0, "long_ride": 0.3, "recovery": 4.0,
        "tempo": 0.3, "sweetspot": 0.2, "threshold": 0.05, "vo2max": 0.05,
        "sprint": 0.05, "race": 0.0,
    },
}


def _to_vec(d: dict[str, float], floor: float = 0.02) -> np.ndarray:
    v = np.full(N_TYPES, floor, dtype=np.float32)
    for k, val in d.items():
        idx = _NAME_TO_IDX.get(k)
        if idx is not None:
            v[idx] = max(val, floor)
    v /= v.sum()
    return v


_PHASE_VECS: dict[str, np.ndarray] = {
    name: _to_vec(d) for name, d in PHASE_PRIORS_RAW.items()
}


def phase_prior(phase: str) -> np.ndarray:
    """Return the (N_TYPES,) prior probability vector for a periodization phase."""
    return _PHASE_VECS.get(phase, _PHASE_VECS["base"])


def event_bias_factor(event_type: str | None, phase: str) -> np.ndarray:
    """Return a multiplicative bias vector encoding event-specificity.

    Re-uses the same {original → replacement} substitutions used in cold-start
    and synthetic data generation so behaviour stays consistent end-to-end.
    Substitutions only apply during build/peak phases (Coggan & Allen, ch. 8).

    Returns a (N_TYPES,) vector in [0, 2] that downweights "wrong" workouts
    and upweights "right" ones for the event.
    """
    factors = np.ones(N_TYPES, dtype=np.float32)
    if not event_type or phase not in ("build", "peak"):
        return factors
    bias = _EVENT_BIAS.get(event_type)
    if not bias:
        return factors
    for src, dst in bias.items():
        si = _NAME_TO_IDX.get(src)
        di = _NAME_TO_IDX.get(dst)
        if si is not None:
            factors[si] *= 0.4   # de-emphasize the "wrong" workout
        if di is not None:
            factors[di] *= 1.8   # boost the event-specific replacement
    return factors


def safety_factor(tsb: float | None, hrv_z: float | None) -> np.ndarray:
    """Return a (N_TYPES,) multiplicative factor that softly bans hard work
    when fatigue is high or HRV is suppressed.

    Plews & Buchheit (2013): when 7-day rolling RMSSD drops > 1 SD below
    baseline, high-intensity work has reduced training effect AND increased
    risk. We translate that to a 0.3× multiplier on threshold/VO2/sprint/race.
    """
    factors = np.ones(N_TYPES, dtype=np.float32)
    risky = ("threshold", "vo2max", "sprint", "race", "sweetspot")
    if tsb is not None and tsb < -25:
        for w in risky:
            factors[_NAME_TO_IDX[w]] *= 0.4
        factors[_NAME_TO_IDX["recovery"]] *= 2.0
        factors[_NAME_TO_IDX["easy"]] *= 1.5
    if hrv_z is not None and hrv_z < -1.0:
        for w in risky:
            factors[_NAME_TO_IDX[w]] *= 0.5
        factors[_NAME_TO_IDX["recovery"]] *= 1.5
    return factors


def posterior(
    logits: np.ndarray,
    *,
    phase: str,
    event_type: str | None = None,
    tsb: float | None = None,
    hrv_z: float | None = None,
    temperature: float = 1.0,
    prior_weight: float = 1.0,
) -> np.ndarray:
    """Compute a calibrated, prior-corrected posterior over workout types.

    Args:
        logits: raw model logits, shape (N_TYPES,).
        phase: periodization phase ("base"/"build"/"peak"/"taper"/"recovery_week").
        event_type: optional event for event-specific bias.
        tsb: current Training Stress Balance.
        hrv_z: current HRV z-score vs baseline.
        temperature: softmax temperature; >1 softens, <1 sharpens.
        prior_weight: blend between pure likelihood (0) and prior-only (∞).
            Posterior ∝ likelihood^(1) · prior^(prior_weight).

    Returns:
        (N_TYPES,) posterior probability vector that sums to 1.
    """
    z = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-3)
    z -= z.max()
    likelihood = np.exp(z)
    likelihood /= likelihood.sum()

    prior = phase_prior(phase) ** float(prior_weight)
    bias = event_bias_factor(event_type, phase)
    safety = safety_factor(tsb, hrv_z)

    post = likelihood * prior * bias * safety
    s = post.sum()
    if s <= 0 or not np.isfinite(s):
        return likelihood.astype(np.float32)
    return (post / s).astype(np.float32)


def select_workout(
    logits: np.ndarray,
    *,
    phase: str,
    event_type: str | None = None,
    tsb: float | None = None,
    hrv_z: float | None = None,
    temperature: float = 1.0,
    prior_weight: float = 1.0,
    top_k: int = 3,
) -> tuple[str, float, list[tuple[str, float]]]:
    """Pick the highest-posterior workout and return alternatives.

    Returns:
        (workout_name, confidence, top_k_alternatives_with_probs)
    """
    post = posterior(
        logits,
        phase=phase,
        event_type=event_type,
        tsb=tsb,
        hrv_z=hrv_z,
        temperature=temperature,
        prior_weight=prior_weight,
    )
    order = np.argsort(post)[::-1]
    top = [(WORKOUT_TYPE_NAMES[i], float(post[i])) for i in order[:top_k]]
    return WORKOUT_TYPE_NAMES[int(order[0])], float(post[int(order[0])]), top


def horizon_to_phase(days_to_event: int | float | None) -> str:
    """Map an integer horizon (days) to the phase used in cold_start.WEEKLY_PATTERNS.

    This is a thin wrapper around `cold_start.get_periodization_phase` that
    accepts horizon **days** instead of weeks for ergonomic call-sites.
    """
    from app.ml.cold_start import get_periodization_phase
    if days_to_event is None or days_to_event <= 0:
        weeks = None
    else:
        weeks = int(days_to_event) // 7
    return get_periodization_phase(weeks)
