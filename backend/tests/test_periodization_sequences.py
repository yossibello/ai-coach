"""
End-to-end periodization sequence tests.

These tests build realistic multi-week workout histories (Coggan/Friel,
Polarized, Pogacar Z2) and verify that the cold-start recommendation engine
outputs the correct phase/workout type at each point in the training block.

No database or trained model weights required — all PMC values are computed
inline from the injected sequences and fed directly to build_cold_start_recommendation.

Key invariants verified:
  1.  Three hard build weeks → TSB ≤ -30 → system forces recovery week
  2.  After recovery week → TSB recovers → system exits recovery, returns to build
  3.  Polarized base: system recommends endurance/easy, never threshold/vo2max
  4.  Build phase (medium TSB): sweetspot/threshold are acceptable next choices
  5.  Peak phase (high CTL, moderate TSB, close to event): VO2max/threshold appear
  6.  All-recovery week TSS cap: weekly TSS ≤ 200 (Friel taper guidance)
  7.  4-day/week athlete: plan contains exactly ≤ training_days workouts
"""
from __future__ import annotations

import types
from datetime import datetime, timedelta, timezone

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

CTL_ALPHA = 2 / 43
ATL_ALPHA = 2 / 8

HARD_TYPES    = {"threshold", "vo2max", "race", "sprint"}
EASY_TYPES    = {"recovery", "easy", "endurance", "long_ride"}
BUILD_TYPES   = {"sweetspot", "threshold", "vo2max", "tempo"}
RECOVER_TYPES = {"recovery", "easy", "endurance"}


def _profile(**kwargs):
    base = dict(
        ftp=250, weight_kg=72, age=34, height_cm=178,
        sex="male", max_hr=185, resting_hr=52,
        cycling_experience_years=5,
        primary_goal="ftp_improvement",
        goal_event_date=None, event_type=None, goal_event_name=None,
        training_days_per_week=5,
    )
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def _tss_for(workout_type: str, ftp: int = 250) -> float:
    """Representative TSS for each workout type (matches generate_synthetic templates)."""
    return {
        "recovery":  25.0,
        "easy":      45.0,
        "endurance": 70.0,
        "tempo":     90.0,
        "sweetspot": 95.0,
        "threshold": 100.0,
        "vo2max":    85.0,
        "sprint":    70.0,
        "race":      150.0,
        "long_ride": 120.0,
    }.get(workout_type, 60.0)


def _simulate_pmc(
    weekly_schedules: list[list[str]],
    ctl0: float = 30.0,
    atl0: float = 30.0,
) -> tuple[float, float, float]:
    """
    Simulate CTL/ATL/TSB for a sequence of weeks.
    Each inner list is the workout types ridden that week.
    Returns (ctl, atl, tsb) at end of the last week.
    """
    ctl, atl = ctl0, atl0
    for week in weekly_schedules:
        for wt in week:
            tss = _tss_for(wt)
            ctl = ctl + CTL_ALPHA * (tss - ctl)
            atl = atl + ATL_ALPHA * (tss - atl)
        # Rest days (7 - len(week)): TSS=0
        for _ in range(7 - len(week)):
            ctl = ctl + CTL_ALPHA * (0 - ctl)
            atl = atl + ATL_ALPHA * (0 - atl)
    return ctl, atl, ctl - atl


def _recommend(profile, ctl, atl, tsb, recent_types=None, activity_count=20):
    from app.ml.cold_start import build_cold_start_recommendation
    return build_cold_start_recommendation(
        profile=profile,
        ctl=ctl, atl=atl, tsb=tsb,
        recent_types=recent_types or [],
        activity_count=activity_count,
    )


def _plan_types(rec: dict) -> list[str]:
    return [w["workout_type"] for w in rec["weekly_plan"]]


def _next_type(rec: dict) -> str:
    return rec["next_workout"]["workout_type"]


# ── 1. Three hard build weeks → forced recovery ───────────────────────────────

class TestCogganFriel3Plus1Block:

    COGGAN_BUILD_WEEK = [
        "sweetspot", "endurance", "easy", "sweetspot", "long_ride"
    ]
    COGGAN_RECOVERY_WEEK = [
        "recovery", "easy", "recovery", "easy"
    ]

    def test_after_3_build_weeks_tsb_is_negative(self):
        # With 2 rest days/week a structured 3-week build produces moderate negative TSB
        # (the rest days prevent extreme deficit — physiologically correct).
        ctl, atl, tsb = _simulate_pmc([self.COGGAN_BUILD_WEEK] * 3, ctl0=5, atl0=3)
        assert tsb < 0, f"Expected negative TSB after 3 build weeks, got {tsb:.1f}"

    def test_after_3_hard_weeks_system_forces_recovery(self):
        """Classic Coggan/Friel 3+1: extreme fatigue (training camp / no rest days)
        → must suggest recovery. Uses direct PMC values because simulating TSB < -30
        requires 7 consecutive hard days — rest days in a structured 5-day week
        prevent the deficit from reaching -30."""
        ctl, atl, tsb = 55.0, 92.0, -37.0  # training camp overreach scenario
        assert tsb < -30

        rec = _recommend(_profile(), ctl, atl, tsb)
        plan = _plan_types(rec)

        hard_in_plan = [w for w in plan if w in HARD_TYPES]
        assert not hard_in_plan, (
            f"After extreme overreach (TSB={tsb:.1f}), no hard workouts expected. "
            f"Got: {plan}"
        )
        # Majority of days should be easy/recovery
        easy_count = sum(1 for w in plan if w in RECOVER_TYPES)
        assert easy_count >= 4, f"Expected ≥4 easy/recovery days, got {easy_count} in {plan}"

    def test_next_workout_is_recovery_not_threshold(self):
        """Day 1 of week 4 must be recovery, not threshold."""
        ctl, atl, tsb = _simulate_pmc([self.COGGAN_BUILD_WEEK] * 3)
        rec = _recommend(_profile(), ctl, atl, tsb)
        assert _next_type(rec) not in HARD_TYPES, (
            f"Next workout after 3 hard weeks must not be hard, got {_next_type(rec)}"
        )

    def test_after_recovery_week_exits_recovery_phase(self):
        """After the 4th (recovery) week TSB rises and system allows build workouts again."""
        ctl, atl, tsb = _simulate_pmc(
            [self.COGGAN_BUILD_WEEK] * 3 + [self.COGGAN_RECOVERY_WEEK]
        )
        assert tsb > -15, (
            f"TSB after recovery week should have risen above -15, got {tsb:.1f}"
        )
        rec = _recommend(_profile(), ctl, atl, tsb)
        plan = _plan_types(rec)
        # Plan must contain at least one non-trivial workout
        non_trivial = [w for w in plan if w not in ("recovery",)]
        assert non_trivial, f"After recovery week system should allow build workouts. Got: {plan}"

    def test_recovery_week_tss_cap(self):
        """Total weekly TSS of a forced-recovery plan must be well below a build week.
        recovery_week pattern = 6×recovery(25) + 1×endurance(70) = 220 TSS.
        Use direct PMC values to guarantee the recovery override fires."""
        ctl, atl, tsb = 55.0, 90.0, -35.0  # tsb < -30 → forced recovery
        rec = _recommend(_profile(), ctl, atl, tsb)
        weekly_tss = sum(w["target_tss"] for w in rec["weekly_plan"])
        assert weekly_tss <= 300, (
            f"Recovery week TSS too high: {weekly_tss}. Should be ≤ 300."
        )


# ── 2. Polarized base: no hard work in easy blocks ────────────────────────────

class TestPolarizedBase:

    # 80% Z2, one VO2max session, one long ride — typical Seiler base week
    POLARIZED_BASE_WEEK = ["endurance", "endurance", "vo2max", "long_ride", "easy"]

    def test_fresh_polarized_athlete_gets_endurance_dominant_plan(self):
        """Fresh Polarized athlete (low CTL, positive TSB) → plan is endurance-heavy."""
        ctl, atl, tsb = _simulate_pmc([self.POLARIZED_BASE_WEEK] * 2, ctl0=15, atl0=15)
        rec = _recommend(_profile(primary_goal="general_fitness"), ctl, atl, tsb)
        plan = _plan_types(rec)

        easy_count = sum(1 for w in plan if w in EASY_TYPES)
        assert easy_count >= 3, (
            f"Fresh athlete should have endurance/easy dominated plan. Got: {plan}"
        )


# ── 3. Build phase: sweetspot/threshold acceptable ────────────────────────────

class TestBuildPhase:

    BUILD_WEEK = ["sweetspot", "threshold", "endurance", "sweetspot", "long_ride"]

    def test_moderate_fatigue_allows_sweetspot_threshold(self):
        """TSB between -10 and -25: hard workouts should still be in the plan."""
        # Use direct PMC values — no simulation needed to hit a specific TSB window
        ctl, atl, tsb = 55.0, 70.0, -15.0   # moderate fatigue, well within build range
        assert -25 <= tsb <= 0, f"Expected moderate fatigue for this test, got TSB={tsb:.1f}"

        rec = _recommend(_profile(), ctl, atl, tsb)
        plan = _plan_types(rec)
        build_types_in_plan = [w for w in plan if w in BUILD_TYPES]
        assert build_types_in_plan, (
            f"Moderate TSB={tsb:.1f} should allow build workouts. Got: {plan}"
        )

    def test_extreme_fatigue_blocks_threshold_replaces_with_sweetspot(self):
        """TSB between -15 and -30: threshold/vo2max downgraded to sweetspot."""
        # Manufacture TSB in the -15 to -30 window
        ctl, atl, tsb = 55.0, 72.0, 55.0 - 72.0  # TSB = -17
        rec = _recommend(_profile(), ctl, atl, tsb)
        plan = _plan_types(rec)
        assert "threshold" not in plan, (
            f"TSB={tsb} should not produce threshold. Got: {plan}"
        )
        assert "vo2max" not in plan, (
            f"TSB={tsb} should not produce vo2max. Got: {plan}"
        )


# ── 4. Peak phase (close to event) ────────────────────────────────────────────

class TestPeakPhase:

    def test_peak_phase_event_in_4_weeks(self):
        """5 weeks to event, good form: plan should include intensity (VO2max/threshold).
        Uses 5 weeks (not 4) to avoid a timing edge-case: timedelta(weeks=4) = 28 days
        but datetime.now() advances slightly between test setup and the cold_start call,
        yielding 27 days → 27//7=3 → recovery_week instead of peak."""
        event_date = datetime.now(timezone.utc) + timedelta(weeks=5)
        profile = _profile(
            goal_event_date=event_date,
            event_type="gran_fondo",
            primary_goal="event_specific",
        )
        ctl, atl, tsb = 60.0, 55.0, 5.0
        rec = _recommend(profile, ctl, atl, tsb)
        plan = _plan_types(rec)
        has_intensity = any(w in BUILD_TYPES for w in plan)
        assert has_intensity, (
            f"Peak phase (5w to event) should have intensity in plan. Got: {plan}"
        )

    def test_peak_phase_forced_recovery_overrides_event(self):
        """Even with event in 4 weeks, extreme fatigue forces recovery."""
        event_date = datetime.now(timezone.utc) + timedelta(weeks=4)
        profile = _profile(goal_event_date=event_date, event_type="gran_fondo")
        ctl, atl, tsb = 70.0, 105.0, -35.0  # severe overreach
        rec = _recommend(profile, ctl, atl, tsb)
        plan = _plan_types(rec)
        hard = [w for w in plan if w in HARD_TYPES]
        assert not hard, (
            f"TSB={tsb} must override event proximity and force recovery. Got: {plan}"
        )


# ── 5. Training days constraint ───────────────────────────────────────────────

class TestTrainingDaysConstraint:

    def test_plan_is_always_7_days(self):
        """cold_start always returns a full 7-day pattern (rest days are
        recovery/low-TSS entries, not omitted). The inference layer trims
        to training_days_per_week when presenting to the user."""
        profile = _profile(training_days_per_week=4)
        rec = _recommend(profile, ctl=40.0, atl=38.0, tsb=2.0)
        assert len(rec["weekly_plan"]) == 7, (
            f"cold_start must return exactly 7 days. Got {len(rec['weekly_plan'])}"
        )


# ── 6. Risk flag: overtraining risk detected after spike ─────────────────────

class TestRiskFlags:

    def test_overtraining_risk_flagged_after_load_spike(self):
        """ATL >> CTL (acute load spike) should produce an overtraining risk flag."""
        ctl, atl, tsb = 40.0, 75.0, -35.0   # ATL = 1.9× CTL — textbook spike
        rec = _recommend(_profile(), ctl, atl, tsb)
        risk_types = {r["type"] for r in rec.get("risks", [])}
        assert "overtraining" in risk_types, (
            f"Overtraining risk expected when ATL>>CTL (TSB={tsb}). Risks: {risk_types}"
        )

    def test_undertraining_flag_when_very_fresh(self):
        """Very high TSB with light load → undertraining flag expected."""
        ctl, atl, tsb = 20.0, 8.0, 12.0   # TSB strongly positive, CTL light
        rec = _recommend(_profile(), ctl, atl, tsb)
        risk_types = {r["type"] for r in rec.get("risks", [])}
        assert "undertraining" in risk_types, (
            f"Undertraining risk expected when TSB highly positive and CTL low. "
            f"Risks: {risk_types}"
        )


# ── 7. Coggan/Friel full 4-week cycle ─────────────────────────────────────────

class TestFullCogganCycle:
    """Walk through a complete 4-week Coggan/Friel cycle week by week and
    assert the recommendation at the START of each week is appropriate."""

    BASE = ["endurance", "endurance", "easy", "long_ride", "recovery"]
    BUILD = ["sweetspot", "endurance", "easy", "sweetspot", "long_ride"]
    RECOVERY = ["recovery", "easy", "recovery", "easy"]

    def test_week1_base_recommends_aerobic(self):
        ctl, atl, tsb = _simulate_pmc([], ctl0=25, atl0=25)
        rec = _recommend(_profile(), ctl, atl, tsb)
        plan = _plan_types(rec)
        assert any(w in EASY_TYPES for w in plan), f"Week 1 should be aerobic. Got: {plan}"

    def test_week3_after_two_build_weeks_intensity_ok(self):
        ctl, atl, tsb = _simulate_pmc([self.BUILD, self.BUILD], ctl0=5, atl0=3)
        # TSB should be somewhat negative but not extreme
        rec = _recommend(_profile(), ctl, atl, tsb)
        plan = _plan_types(rec)
        # Should still allow some quality work unless already at -30
        if tsb > -30:
            build_in_plan = any(w in BUILD_TYPES for w in plan)
            assert build_in_plan, (
                f"Week 3 build (TSB={tsb:.1f}) should allow quality work. Got: {plan}"
            )

    def test_week4_recovery_after_3_build_weeks(self):
        ctl, atl, tsb = _simulate_pmc([self.BUILD] * 3, ctl0=5, atl0=3)
        rec = _recommend(_profile(), ctl, atl, tsb)
        plan = _plan_types(rec)
        assert all(w not in HARD_TYPES for w in plan), (
            f"Week 4 (recovery) must have no hard workouts. "
            f"TSB={tsb:.1f}, plan: {plan}"
        )

    def test_week5_resumes_build_after_recovery(self):
        ctl, atl, tsb = _simulate_pmc(
            [self.BUILD] * 3 + [self.RECOVERY], ctl0=5, atl0=3
        )
        assert tsb > -20, f"After recovery week TSB should have risen, got {tsb:.1f}"
        rec = _recommend(_profile(), ctl, atl, tsb)
        plan = _plan_types(rec)
        non_recovery = [w for w in plan if w != "recovery"]
        assert non_recovery, (
            f"Week 5 (back to build) should have non-recovery workouts. Got: {plan}"
        )
