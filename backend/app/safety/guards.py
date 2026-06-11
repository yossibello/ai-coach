"""
Safety guards for coach recommendations and supplement stacks.

These are HARD LIMITS — they override any model output. The principle is:
  "Never recommend something that could plausibly hurt the user."

Two guard surfaces:
  • apply_workout_safety(...)    — clamps duration / target_tss / intensity
                                   given current CTL & TSB
  • apply_supplement_safety(...) — clamps doses, blocks dangerous combos

Both functions are pure and side-effect free; they return new dicts.
"""
from __future__ import annotations

from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Workout safety constants
# ─────────────────────────────────────────────────────────────────────────────

# Absolute caps regardless of fitness — even pros rarely exceed these
MAX_DAILY_TSS = 350                 # 5h+ at threshold = 350 TSS
MAX_SINGLE_SESSION_MINUTES = 360    # 6h hard cap
MAX_WEEKLY_TSS_RAMP_PCT = 0.10      # +10% per week max increase

# TSB thresholds (Form / freshness)
HARD_RECOVERY_TSB = -30             # below this = mandatory recovery only
MEDIUM_FATIGUE_TSB = -15            # below this = no high-intensity

# Workout types we treat as "high intensity" for fatigue gating
HIGH_INTENSITY_TYPES = {"threshold", "vo2max", "sprint", "race"}

# Per-workout sane upper bounds for TSS given workout type, used as a
# secondary cap (model could output an absurd 500 TSS recovery ride).
TYPE_TSS_CEILING = {
    "recovery":  40,
    "easy":      80,
    "endurance": 200,
    "tempo":     180,
    "sweetspot": 200,
    "threshold": 220,
    "vo2max":    180,
    "sprint":    150,
    "long_ride": 350,
    "race":      350,
}

# CTL-relative cap: a single session shouldn't exceed ~2× CTL or you risk
# acute overload (rule of thumb from Friel / TrainingPeaks coaches).
SESSION_TSS_AS_CTL_MULTIPLE = 2.5


# ─────────────────────────────────────────────────────────────────────────────
# Health-readiness thresholds (HRV / RHR — Plews & Buchheit 2013, Buchheit 2014)
# ─────────────────────────────────────────────────────────────────────────────
# Z-score below which today's HRV is considered "crashed" → force recovery.
HRV_CRASH_Z = -1.5
# Z-score below which intensity should be downgraded.
HRV_LOW_Z   = -0.7
# Resting HR delta (bpm) above 30-day baseline that signals systemic fatigue.
RHR_HIGH_DELTA = 7.0
# Combined readiness score thresholds.
READINESS_RED   = 40.0
READINESS_AMBER = 60.0


# ─────────────────────────────────────────────────────────────────────────────
# Supplement hard limits
# ─────────────────────────────────────────────────────────────────────────────

# Per-day MAXIMUMS in the unit shown. Safety margin under tolerable upper
# intake levels (UL) from EFSA / IOM. Engine doses must not exceed these.
SUPPLEMENT_HARD_LIMITS: dict[str, dict[str, Any]] = {
    "caffeine":              {"max_dose": 6.0,  "unit": "mg/kg"},   # >9 mg/kg → diminishing + side effects
    "creatine_monohydrate":  {"max_dose": 5.0,  "unit": "g"},       # daily maintenance; loading not auto-prescribed
    "iron":                  {"max_dose": 100,  "unit": "mg"},      # UL 45 mg/d, but split-dose protocols up to 100
    "vitamin_d3":            {"max_dose": 4000, "unit": "IU"},      # EFSA UL = 4000 IU/d
    "magnesium_glycinate":   {"max_dose": 350,  "unit": "mg"},      # IOM UL for supplemental Mg = 350 mg
    "omega3":                {"max_dose": 3000, "unit": "mg"},      # EFSA: up to 5g safe; we cap conservatively
    "beta_alanine":          {"max_dose": 6.5,  "unit": "g"},
    "beetroot_nitrate":      {"max_dose": 12.0, "unit": "mmol NO₃⁻"},
    "sodium_bicarbonate":    {"max_dose": 0.3,  "unit": "g/kg"},    # >0.4 g/kg → severe GI distress
    "electrolytes":          {"max_dose": 2000, "unit": "mg"},      # sodium per session
    "carbohydrate_drink":    {"max_dose": 120,  "unit": "g/h"},
    "whey_protein":          {"max_dose": 60,   "unit": "g"},
}

# Hard contraindications keyed by user condition.
# If the user's profile has any of these flags, the supplement is BLOCKED
# regardless of any score / trigger from the engine.
CONDITION_BLOCKLIST: dict[str, set[str]] = {
    "pregnancy":               {"caffeine", "beta_alanine", "beetroot_nitrate", "sodium_bicarbonate"},
    "uncontrolled_hypertension": {"caffeine", "sodium_bicarbonate"},
    "chronic_kidney_disease":  {"creatine_monohydrate", "sodium_bicarbonate", "whey_protein"},
    "ferritin_high":           {"iron"},
    "anxiety_disorder":        {"caffeine"},
    "anticoagulant_use":       {"omega3"},      # high-dose fish oil + warfarin = bleeding risk
}

# Pairs of supplements that should NEVER appear together in the same stack.
DANGEROUS_PAIRS: list[tuple[str, str, str]] = [
    # (supp_a, supp_b, reason)
    ("caffeine", "sodium_bicarbonate",
     "Caffeine + bicarb significantly increases GI distress risk during exercise."),
]


# ═════════════════════════════════════════════════════════════════════════════
# Workout safety
# ═════════════════════════════════════════════════════════════════════════════

def apply_workout_safety(
    next_workout: dict[str, Any] | None,
    *,
    ctl: float,
    tsb: float,
) -> tuple[dict[str, Any] | None, list[str]]:
    """
    Clamp a single workout to safe ranges. Returns (clamped_workout, notes).

    Notes is a list of human-readable adjustments made — surface these to
    the user so they understand why the recommendation changed.
    """
    if next_workout is None:
        return None, []

    notes: list[str] = []
    w = dict(next_workout)  # copy
    wtype = (w.get("workout_type") or "endurance").lower()
    duration = float(w.get("duration_minutes") or 0)
    target_tss = float(w.get("target_tss") or 0)

    # 1. Mandatory recovery if TSB is dangerously negative
    if tsb < HARD_RECOVERY_TSB:
        if wtype != "recovery":
            notes.append(
                f"Forced recovery: TSB {round(tsb)} is below safe threshold "
                f"({HARD_RECOVERY_TSB}). High-intensity work blocked."
            )
            wtype = "recovery"
            duration = min(duration, 45)
            target_tss = min(target_tss, 30)

    # 2. Block high-intensity if moderately fatigued
    elif tsb < MEDIUM_FATIGUE_TSB and wtype in HIGH_INTENSITY_TYPES:
        notes.append(
            f"Downgraded {wtype} → sweetspot: TSB {round(tsb)} indicates fatigue."
        )
        wtype = "sweetspot"
        target_tss = min(target_tss, 120)

    # 2b. Cap endurance/long_ride duration if moderately fatigued.
    # Heavy aerobic days after accumulated load still carry physiological cost —
    # 145min z2 when TSB=-20 adds stress without providing recovery.
    if HARD_RECOVERY_TSB < tsb < MEDIUM_FATIGUE_TSB and wtype in ("endurance", "long_ride", "easy"):
        if duration > 90:
            notes.append(
                f"Shortened {wtype}: TSB {round(tsb)} — kept aerobic session short to allow recovery."
            )
            duration = 90
            target_tss = min(target_tss, 75)

    # 3. Per-type duration ceiling (model can output absurd durations for easy types)
    TYPE_DURATION_CEILING = {
        "recovery": 60,
        "easy":     120,
    }
    type_dur_ceiling = TYPE_DURATION_CEILING.get(wtype)
    if type_dur_ceiling and duration > type_dur_ceiling:
        notes.append(f"Capped {wtype} duration: {int(duration)} min → {type_dur_ceiling} min.")
        duration = type_dur_ceiling

    # 4. Hard duration cap
    if duration > MAX_SINGLE_SESSION_MINUTES:
        notes.append(
            f"Capped duration: {int(duration)} min → {MAX_SINGLE_SESSION_MINUTES} min."
        )
        duration = MAX_SINGLE_SESSION_MINUTES

    # 4. Per-type TSS ceiling
    type_ceiling = TYPE_TSS_CEILING.get(wtype, 250)
    if target_tss > type_ceiling:
        notes.append(
            f"Capped {wtype} TSS: {int(target_tss)} → {type_ceiling}."
        )
        target_tss = type_ceiling

    # 5. CTL-relative cap (don't suggest 2.5× current fitness in one ride)
    if ctl > 5:  # only meaningful once user has any base
        ctl_cap = ctl * SESSION_TSS_AS_CTL_MULTIPLE
        if target_tss > ctl_cap:
            notes.append(
                f"Capped TSS to {SESSION_TSS_AS_CTL_MULTIPLE}× CTL "
                f"({int(target_tss)} → {int(ctl_cap)})."
            )
            target_tss = ctl_cap

    # 6. Absolute hard cap
    if target_tss > MAX_DAILY_TSS:
        notes.append(f"Capped TSS at absolute max {MAX_DAILY_TSS}.")
        target_tss = MAX_DAILY_TSS

    w["workout_type"] = wtype
    w["duration_minutes"] = int(duration)
    w["target_tss"] = round(target_tss, 1)
    if notes:
        w.setdefault("safety_notes", []).extend(notes)
    return w, notes


def apply_health_safety(
    next_workout: dict[str, Any] | None,
    *,
    readiness: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """
    Override workout intensity based on today's HRV / RHR / readiness.

    Hierarchy (most-restrictive wins):
      1. HRV crashed (z ≤ -1.5) OR readiness red  → force recovery
      2. HRV low    (z ≤ -0.7) OR RHR up ≥ 7 bpm OR readiness amber
                                                    → downgrade high-intensity
      3. otherwise: unchanged
    """
    if next_workout is None or readiness is None:
        return next_workout, []

    notes: list[str] = []
    w = dict(next_workout)
    wtype = (w.get("workout_type") or "endurance").lower()
    duration = float(w.get("duration_minutes") or 0)
    target_tss = float(w.get("target_tss") or 0)

    score = float(readiness.get("score") or 50.0)
    status = str(readiness.get("status") or "amber")
    hrv_z = readiness.get("hrv_z")
    rhr_delta = readiness.get("rhr_delta")

    crashed = (
        (hrv_z is not None and hrv_z <= HRV_CRASH_Z)
        or score <= READINESS_RED
        or status == "red"
    )
    low = (
        (hrv_z is not None and hrv_z <= HRV_LOW_Z)
        or (rhr_delta is not None and rhr_delta >= RHR_HIGH_DELTA)
        or score <= READINESS_AMBER
        or status == "amber"
    )

    if crashed and wtype != "recovery":
        why = []
        if hrv_z is not None and hrv_z <= HRV_CRASH_Z:
            why.append(f"HRV {hrv_z:+.1f} SD vs baseline")
        if rhr_delta is not None and rhr_delta >= RHR_HIGH_DELTA:
            why.append(f"RHR {rhr_delta:+.0f} bpm vs baseline")
        why.append(f"readiness {score:.0f}/100")
        notes.append(
            "Forced recovery on health signals: " + ", ".join(why) +
            ". Pushing intensity today increases illness/overtraining risk "
            "(Plews & Buchheit 2013; Buchheit 2014)."
        )
        wtype = "recovery"
        duration = min(duration, 45)
        target_tss = min(target_tss, 30)
    elif low and wtype in HIGH_INTENSITY_TYPES:
        notes.append(
            f"Downgraded {wtype} → sweetspot on low readiness "
            f"({score:.0f}/100). HRV and RHR suggest incomplete recovery."
        )
        wtype = "sweetspot"
        target_tss = min(target_tss, 120)

    w["workout_type"] = wtype
    w["duration_minutes"] = int(duration)
    w["target_tss"] = round(target_tss, 1)
    if notes:
        w.setdefault("safety_notes", []).extend(notes)
    return w, notes


def apply_drift_safety(
    next_workout: dict[str, Any] | None,
    *,
    drift_state: str | None,
    drift_pct: float | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Gate intensity based on aerobic decoupling state.

    Stable  (<5%):  no change — caller may trigger progression event.
    Decoupled (5–10%): block threshold/VO2max; cap at sweetspot.
    Stressed (>10%): force easy/recovery and reduce duration.
    """
    if next_workout is None or drift_state is None or drift_pct is None:
        return next_workout, []

    notes: list[str] = []
    w = dict(next_workout)
    wtype = (w.get("workout_type") or "endurance").lower()

    if drift_state == "stressed" and wtype not in {"recovery", "easy"}:
        notes.append(
            f"HR drift {drift_pct:.1f}% (stressed) — downgraded {wtype} → easy. "
            "Reduce ride duration and verify HRV before next hard session."
        )
        w["workout_type"] = "easy"
        w["duration_minutes"] = min(int(w.get("duration_minutes") or 60), 60)
        w["target_tss"] = min(float(w.get("target_tss") or 50), 50)

    elif drift_state == "decoupled" and wtype in HIGH_INTENSITY_TYPES:
        notes.append(
            f"HR drift {drift_pct:.1f}% (decoupled) — downgraded {wtype} → sweetspot. "
            "Consolidate aerobic base before adding intensity."
        )
        w["workout_type"] = "sweetspot"
        w["target_tss"] = min(float(w.get("target_tss") or 100), 120)

    if notes:
        w.setdefault("safety_notes", []).extend(notes)
    return w, notes


def check_weekly_ramp_safe(
    last_week_tss: float,
    proposed_week_tss: float,
) -> tuple[bool, str | None]:
    """
    Returns (is_safe, message_if_not). True ramp guard is enforced at the
    weekly_plan level by `apply_weekly_plan_safety`.
    """
    if last_week_tss <= 0:
        return True, None
    delta_pct = (proposed_week_tss - last_week_tss) / last_week_tss
    if delta_pct > MAX_WEEKLY_TSS_RAMP_PCT:
        return False, (
            f"Proposed week is {delta_pct * 100:.0f}% above last week "
            f"(max safe ramp = {MAX_WEEKLY_TSS_RAMP_PCT * 100:.0f}%)."
        )
    return True, None


def apply_weekly_plan_safety(
    weekly_plan: list[dict[str, Any]],
    *,
    last_week_tss: float,
    ctl: float,
    tsb: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Clamp every workout in the plan and enforce the weekly ramp limit.
    Returns (safe_plan, top_level_notes).
    """
    notes: list[str] = []
    safe_plan: list[dict[str, Any]] = []
    for w in weekly_plan:
        sw, _ = apply_workout_safety(w, ctl=ctl, tsb=tsb)
        if sw is not None:
            safe_plan.append(sw)

    proposed_total = sum(float(w.get("target_tss") or 0) for w in safe_plan)
    is_safe, msg = check_weekly_ramp_safe(last_week_tss, proposed_total)
    if not is_safe and msg:
        # Scale every workout down proportionally to fit the ramp cap.
        cap = last_week_tss * (1 + MAX_WEEKLY_TSS_RAMP_PCT)
        scale = cap / proposed_total if proposed_total > 0 else 1.0
        for w in safe_plan:
            w["target_tss"] = round(float(w.get("target_tss") or 0) * scale, 1)
            w["duration_minutes"] = max(20, int(float(w.get("duration_minutes") or 0) * scale))
        notes.append(
            f"Weekly plan scaled down ({scale:.0%}) to keep ramp ≤ "
            f"{MAX_WEEKLY_TSS_RAMP_PCT * 100:.0f}%."
        )

    return safe_plan, notes


# ═════════════════════════════════════════════════════════════════════════════
# Supplement safety
# ═════════════════════════════════════════════════════════════════════════════

def apply_supplement_safety(
    stack: list[dict[str, Any]],
    *,
    user_conditions: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns (safe_stack, extra_warnings).

    1. Drops items blocked by a user condition (pregnancy, kidney disease, ...).
    2. Clamps doses to the per-supplement hard upper limit.
    3. Removes the lower-priority half of any dangerous pair, adds a warning.
    """
    conditions = set(user_conditions or [])
    safe: list[dict[str, Any]] = []
    extra_warnings: list[dict[str, Any]] = []

    # 1+2: drop blocked, clamp doses
    for item in stack:
        key = item.get("supplement_key")
        if not key:
            continue

        blocking = [c for c in conditions if key in CONDITION_BLOCKLIST.get(c, set())]
        if blocking:
            extra_warnings.append({
                "warning_key": f"blocked_{key}",
                "applies_to":  [key],
                "message":     (
                    f"{item.get('label', key)} blocked due to user condition(s): "
                    f"{', '.join(blocking)}."
                ),
                "citations":   [],
            })
            continue

        clamped = dict(item)
        limit = SUPPLEMENT_HARD_LIMITS.get(key)
        if limit and isinstance(clamped.get("dose"), (int, float)):
            if clamped["dose"] > limit["max_dose"]:
                extra_warnings.append({
                    "warning_key": f"clamped_{key}",
                    "applies_to":  [key],
                    "message":     (
                        f"{item.get('label', key)} dose clamped from "
                        f"{clamped['dose']} → {limit['max_dose']} "
                        f"{limit['unit']} (safety cap)."
                    ),
                    "citations":   [],
                })
                clamped["dose"] = limit["max_dose"]
                clamped["dose_unit"] = limit["unit"]
        safe.append(clamped)

    # 3: remove the lower-scored half of dangerous pairs
    keys_in_stack = {item["supplement_key"]: item for item in safe}
    to_remove: set[str] = set()
    for a, b, reason in DANGEROUS_PAIRS:
        if a in keys_in_stack and b in keys_in_stack and a not in to_remove and b not in to_remove:
            score_a = keys_in_stack[a].get("score", 0)
            score_b = keys_in_stack[b].get("score", 0)
            drop = b if score_a >= score_b else a
            to_remove.add(drop)
            extra_warnings.append({
                "warning_key": f"pair_{a}_{b}",
                "applies_to":  [a, b],
                "message":     f"Removed {drop}: {reason}",
                "citations":   [],
            })

    safe = [s for s in safe if s["supplement_key"] not in to_remove]
    return safe, extra_warnings
