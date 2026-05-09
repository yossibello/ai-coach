"""
Activity data-quality scoring.

Phase 1 (current): rule-based scorer. Returns ('high'|'medium'|'low'|'rejected', [reasons]).
Phase 2 (later):  ML-based outlier detection vs the user's own history.
Phase 3 (later):  Model-confidence-driven labeling for active learning.

The scorer is the gate that decides whether a ride:
  - serves user-facing recommendations  (medium+ -> yes; low -> with caveat; rejected -> no)
  - is eligible for community model retraining  (high only, plus user.allow_for_training)
  - is eligible for the user's own LoRA adapter (medium+)

Keep the rules CONSERVATIVE — false positives (junk slipping in) hurt the model far
more than false negatives (good rides flagged for review).
"""
from __future__ import annotations

from typing import Any


# ── Hard physiological / sensor sanity bounds ─────────────────────────────────
# Anything outside these is almost certainly a sensor glitch or wrong-sport upload.
HARD_BOUNDS = {
    "duration_seconds":     (300, 24 * 3600),    # 5 min .. 24 h
    "distance_meters":      (500, 1_000_000),    # 0.5 km .. 1000 km
    "avg_power":            (30, 500),           # W
    "max_power":            (50, 2500),          # W (sprint spikes)
    "normalized_power":     (30, 600),
    "intensity_factor":     (0.2, 1.6),
    "tss":                  (1, 1000),
    "avg_hr":               (40, 220),
    "max_hr":               (60, 230),
    "avg_cadence":          (30, 130),
}


def score_activity(
    act: dict[str, Any],
    *,
    profile_max_hr: int | None = None,
    profile_ftp: int | None = None,
) -> tuple[str, list[str]]:
    """
    Returns (quality_score, reasons).

    quality_score ∈ {'high', 'medium', 'low', 'rejected'}.
    reasons is a list of short codes for the UI / debugging.
    """
    reasons: list[str] = []

    # ── Hard rejects ──────────────────────────────────────────────────────────
    for field, (lo, hi) in HARD_BOUNDS.items():
        v = act.get(field)
        if v is None:
            continue
        if not (lo <= float(v) <= hi):
            reasons.append(f"{field}_out_of_range")

    # Profile-relative checks (only if we have the profile values)
    if profile_max_hr and act.get("max_hr") and act["max_hr"] > profile_max_hr + 15:
        reasons.append("hr_above_profile_max")
    if profile_ftp and act.get("normalized_power") and act["normalized_power"] > profile_ftp * 1.6:
        reasons.append("np_far_above_ftp")

    # Trainer rides with no power are nearly useless for ML
    if act.get("trainer") and not act.get("avg_power"):
        reasons.append("trainer_no_power")

    if any(r.endswith("_out_of_range") for r in reasons):
        return "rejected", reasons

    # ── Determine completeness tier ───────────────────────────────────────────
    has_power = act.get("avg_power") is not None and act.get("normalized_power") is not None
    has_tss   = act.get("tss") is not None and act.get("intensity_factor") is not None
    has_hr    = act.get("avg_hr") is not None
    has_zones = bool(act.get("time_in_zones"))
    has_dur   = (act.get("duration_seconds") or 0) >= 600  # >= 10 min

    # Soft warnings
    if not has_dur:
        reasons.append("very_short_ride")
    if not has_hr and not has_power:
        reasons.append("no_hr_no_power")

    # Tier assignment
    if has_power and has_tss and has_hr and has_zones and has_dur:
        return "high", reasons          # full power+HR ride: gold standard
    if (has_power and has_tss and has_dur) or (has_hr and has_zones and has_dur):
        return "medium", reasons        # usable for inference, OK for adapters
    return "low", reasons               # store, but don't train on it


# ── Phase-2 hook: outlier detection vs the user's own history ─────────────────
def is_outlier_vs_history(
    act: dict[str, Any],
    user_history_stats: dict[str, dict[str, float]] | None,
    *,
    z_threshold: float = 4.0,
) -> bool:
    """
    Returns True if the ride is more than `z_threshold` standard deviations from
    the user's rolling history on key metrics. Caller is responsible for
    computing `user_history_stats` (e.g. {'tss': {'mean': 80, 'std': 25}, ...}).

    Returns False if we don't have enough history yet (<20 rides).
    """
    if not user_history_stats:
        return False
    for field in ("tss", "normalized_power", "avg_hr", "duration_seconds"):
        stats = user_history_stats.get(field)
        v = act.get(field)
        if not stats or v is None or stats.get("std", 0) <= 0:
            continue
        z = abs((float(v) - stats["mean"]) / stats["std"])
        if z > z_threshold:
            return True
    return False
