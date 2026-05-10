"""
Supplement recommendation engine (rule-based, Phase 1).

Inputs:
  • User profile (sex, age, diet, climate)
  • Training context (weekly_hours, CTL, weekly_tss, recent_temp,
                      upcoming event, workout_focus, days_to_event)
  • Latest blood test (optional)

Output:
  {
    "stack": [ {supplement_key, label, dose, unit, frequency, timing,
                rationale, evidence_grade, citations, contraindications,
                warnings, score, sources_triggered}, ... ],
    "depletion_signals": {...},
    "warnings": [ ... ],
    "based_on_blood_test_id": "...",
    "engine_version": "rule_v1",
  }

Phase 2: replace `_score_supplements` with a learned head sharing the
existing transformer encoder.
"""
from __future__ import annotations

from typing import Optional

from app.nutrition.depletion_rules import compute_signals
from app.nutrition.supplements import SUPPLEMENTS, ANTI_SUPPLEMENTS
from app.safety.guards import apply_supplement_safety


ENGINE_VERSION = "rule_v1"

# Supplements reaching this score are included in the stack.
INCLUSION_THRESHOLD = 0.5

# Standard disclaimer appended to every stack.
DISCLAIMER = (
    "Educational guidance only — not medical advice. Always consult a "
    "physician or sports dietitian before starting any new supplement, "
    "especially if you have underlying health conditions or take medications."
)


def recommend_supplements(
    *,
    profile: dict,
    weekly_hours: float,
    ctl: float,
    weekly_tss: float,
    recent_avg_temp_c: Optional[float],
    upcoming_event_type: Optional[str],
    days_to_event: Optional[int],
    workout_focus: Optional[str],
    blood_test: Optional[dict] = None,           # {"id": str, "markers": {...}}
    user_contraindications: Optional[list[str]] = None,
) -> dict:
    user_contras = set((user_contraindications or []))

    # Extract markers + status from blood test if provided
    blood_markers: dict = {}
    based_on_id: Optional[str] = None
    if blood_test:
        blood_markers = blood_test.get("markers", {}) or {}
        based_on_id = blood_test.get("id")

    has_blood = bool(blood_markers)

    # If ferritin is high, flag that as a hard contraindication for iron supps.
    fer = blood_markers.get("ferritin")
    if fer and fer.get("status") in ("high", "critical_high"):
        user_contras.add("ferritin_high")

    signals = compute_signals(
        blood_markers=blood_markers,
        profile=profile,
        weekly_hours=weekly_hours,
        ctl=ctl,
        weekly_tss=weekly_tss,
        recent_avg_temp_c=recent_avg_temp_c,
        upcoming_event_type=upcoming_event_type,
        days_to_event=days_to_event,
        workout_focus=workout_focus,
        has_recent_blood_test=has_blood,
    )

    stack: list[dict] = []
    for key, supp in SUPPLEMENTS.items():
        # Skip if the user has any contraindication for this supplement.
        contras = set(supp.get("contraindications", []))
        if contras & user_contras:
            continue

        triggers = supp.get("triggers", [])
        if not triggers:
            continue

        # Score = max strength of any trigger that fired
        triggered = [(t, signals.get(t, 0.0)) for t in triggers if signals.get(t, 0.0) > 0]
        if not triggered:
            continue
        score = max(strength for _, strength in triggered)
        if score < INCLUSION_THRESHOLD:
            continue

        stack.append({
            "supplement_key":   key,
            "label":            supp["label"],
            "category":         supp["category"],
            "evidence_grade":   supp["evidence_grade"],
            "dose":             supp["default_dose"],
            "dose_unit":        supp["dose_unit"],
            "frequency":        supp["frequency"],
            "timing":           supp["timing"],
            "duration":         supp["duration"],
            "rationale":        supp["rationale"],
            "citations":        supp["citations"],
            "warnings":         supp.get("warnings", []),
            "contraindications": supp.get("contraindications", []),
            "score":            round(score, 3),
            "triggered_by":     [t for t, _ in triggered],
        })

    # Sort: evidence grade A first, then by score desc
    grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    stack.sort(key=lambda x: (grade_order.get(x["evidence_grade"], 9), -x["score"]))

    # ── Anti-supplement warnings ─────────────────────────────────────────────
    warnings: list[dict] = []
    for key, anti in ANTI_SUPPLEMENTS.items():
        trig = anti.get("trigger")
        if trig == "always":
            warnings.append({
                "warning_key": key,
                "applies_to":  anti["applies_to"],
                "message":     anti["message"],
                "citations":   anti.get("citations", []),
            })
        elif trig and signals.get(trig, 0.0) > 0:
            warnings.append({
                "warning_key": key,
                "applies_to":  anti["applies_to"],
                "message":     anti["message"],
                "citations":   anti.get("citations", []),
            })

    # ── Hard safety pass: cap doses, drop blocked items, dangerous combos ──
    safe_stack, safety_warnings = apply_supplement_safety(
        stack, user_conditions=list(user_contras)
    )
    warnings.extend(safety_warnings)

    return {
        "stack":                  safe_stack,
        "depletion_signals":      {k: round(v, 3) for k, v in signals.items()},
        "warnings":                warnings,
        "based_on_blood_test_id": based_on_id,
        "has_blood_test":         has_blood,
        "engine_version":         ENGINE_VERSION,
        "is_cold_start":          not has_blood,
        "disclaimer":             DISCLAIMER,
    }
