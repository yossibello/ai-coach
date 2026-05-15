"""
Unit tests for the weekly plan scheduling logic.

Covers:
  - _trim_pattern_to_days: correct workout selection + Friel-based day spacing
  - _rollout_week: budget pruning, output length, day-0 preservation
  - planner.solve_week: TSS budget scales correctly for fewer training days
  - planner._project_pmc: rest-day padding changes end-of-week TSB

No database, no model weights — pure logic tests using SimpleNamespace stubs.
"""
from __future__ import annotations

import types
from datetime import datetime, timezone, timedelta

import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

def _profile(training_days=5, ftp=250, weight=70):
    return types.SimpleNamespace(
        training_days_per_week=training_days,
        ftp=ftp, weight_kg=weight,
        age=35, height_cm=178, sex="male",
        max_hr=185, resting_hr=50,
        cycling_experience_years=5,
        primary_goal="build_fitness",
        goal_event_date=None,
        event_type=None,
        goal_event_name=None,
    )


def _fake_workout(workout_type="endurance", tss=60, duration_min=60, day_offset=0):
    return {
        "workout_type": workout_type,
        "target_tss": tss,
        "duration_minutes": duration_min,
        "day_offset": day_offset,
        "description": "",
        "structure": [],
        "rationale": "",
        "key_metric": "",
    }


# ── _trim_pattern_to_days ─────────────────────────────────────────────────────

class TestTrimPatternToDays:
    from app.ml.inference import _trim_pattern_to_days

    def test_full_7_day_unchanged(self):
        from app.ml.inference import _trim_pattern_to_days
        pattern = ["tempo", "recovery", "threshold", "recovery", "sweetspot", "long_ride", "recovery"]
        result = _trim_pattern_to_days(pattern, 7)
        assert [wt for _, wt in result] == pattern
        assert [off for off, _ in result] == list(range(7))

    def test_4_days_drops_recovery_filler(self):
        from app.ml.inference import _trim_pattern_to_days
        pattern = ["tempo", "recovery", "threshold", "recovery", "sweetspot", "long_ride", "recovery"]
        result = _trim_pattern_to_days(pattern, 4)
        workout_types = [wt for _, wt in result]
        assert len(result) == 4
        # Should prefer non-recovery workouts
        assert "tempo" in workout_types
        assert "threshold" in workout_types
        assert "sweetspot" in workout_types or "long_ride" in workout_types

    def test_3_days_no_consecutive_hard(self):
        from app.ml.inference import _trim_pattern_to_days
        from app.ml.inference import _RECOVERY_DAYS_NEEDED
        pattern = ["threshold", "recovery", "vo2max", "recovery", "sweetspot", "long_ride", "recovery"]
        result = _trim_pattern_to_days(pattern, 3)
        offsets = [off for off, _ in result]
        workouts = [wt for _, wt in result]
        assert len(result) == 3
        # threshold needs 2 rest days → next session at offset >= 3
        threshold_off = offsets[workouts.index("threshold")]
        next_off = offsets[1]
        gap = _RECOVERY_DAYS_NEEDED.get("threshold", 1)
        assert next_off >= threshold_off + gap + 1

    def test_day_0_always_first(self):
        from app.ml.inference import _trim_pattern_to_days
        pattern = ["vo2max", "recovery", "threshold", "recovery", "sweetspot", "long_ride", "recovery"]
        for n in [3, 4, 5]:
            result = _trim_pattern_to_days(pattern, n)
            assert result[0][0] == 0, f"day_offset[0] must be 0 for n={n}"
            assert result[0][1] == "vo2max", f"day-0 workout must be preserved for n={n}"

    def test_offsets_strictly_increasing(self):
        from app.ml.inference import _trim_pattern_to_days
        pattern = ["threshold", "recovery", "sweetspot", "recovery", "endurance", "long_ride", "recovery"]
        for n in [3, 4, 5, 6]:
            result = _trim_pattern_to_days(pattern, n)
            offsets = [off for off, _ in result]
            assert offsets == sorted(set(offsets)), f"Offsets not strictly increasing for n={n}: {offsets}"

    def test_offsets_within_week(self):
        from app.ml.inference import _trim_pattern_to_days
        pattern = ["tempo", "recovery", "threshold", "recovery", "sweetspot", "long_ride", "recovery"]
        for n in range(1, 8):
            result = _trim_pattern_to_days(pattern, n)
            for off, _ in result:
                assert 0 <= off <= 6, f"Offset {off} out of [0,6] for n={n}"


# ── _rollout_week budget pruning ──────────────────────────────────────────────

class TestRolloutWeekBudget:
    """
    Tests the budget-pruning logic of _rollout_week without a real model.
    We monkey-patch _transformer_recommendation to return fixed predictions.
    """

    def _make_activities(self, n=10):
        base = datetime.now(timezone.utc) - timedelta(days=n)
        acts = []
        for i in range(n):
            acts.append(types.SimpleNamespace(
                date=base + timedelta(days=i),
                tss=60.0, duration_seconds=3600,
                avg_power=180.0, normalized_power=190.0, max_power=350.0,
                avg_hr=145, max_hr=175, avg_cadence=88,
                distance_meters=40000.0, elevation_gain_meters=300.0,
                workout_type="endurance", device_watts=True, trainer=False,
                temperature_c=18.0,
                pc_5s_wkg=None, pc_1min_wkg=None,
                pc_5min_wkg=None, pc_20min_wkg=None,
            ))
        return acts

    def test_output_length_matches_budget(self, monkeypatch):
        from app.ml import inference as inf

        call_count = {"n": 0}
        TYPES = ["threshold", "recovery", "sweetspot", "recovery", "endurance", "recovery", "long_ride"]

        def fake_transformer(model, activities, profile, ctl, atl, tsb, **kwargs):
            wt = TYPES[call_count["n"] % len(TYPES)]
            call_count["n"] += 1
            return {"next_workout": _fake_workout(wt, tss=70, day_offset=0)}

        monkeypatch.setattr(inf, "_transformer_recommendation", fake_transformer)

        for budget in [3, 4, 5]:
            call_count["n"] = 0
            result = inf._rollout_week(
                model=object(),
                activities=self._make_activities(),
                profile=_profile(training_days=budget),
                ctl=40.0, atl=35.0, tsb=5.0,
                day0_workout=_fake_workout("tempo", tss=80),
                health_by_date=None,
                training_days=budget,
                horizon_override_days=30,
            )
            assert len(result) <= budget, (
                f"rollout returned {len(result)} sessions for budget={budget}"
            )

    def test_day0_preserved(self, monkeypatch):
        from app.ml import inference as inf

        monkeypatch.setattr(inf, "_transformer_recommendation",
                            lambda *a, **kw: {"next_workout": _fake_workout("recovery")})

        day0 = _fake_workout("vo2max", tss=100)
        result = inf._rollout_week(
            model=object(),
            activities=self._make_activities(),
            profile=_profile(training_days=4),
            ctl=50.0, atl=45.0, tsb=5.0,
            day0_workout=day0,
            health_by_date=None,
            training_days=4,
            horizon_override_days=30,
        )
        assert result[0]["workout_type"] == "vo2max"
        assert result[0]["day_offset"] == 0

    def test_hard_sessions_kept_over_easy(self, monkeypatch):
        """When over budget, recovery/easy sessions are dropped before hard ones."""
        from app.ml import inference as inf

        predictions = [
            _fake_workout("threshold", tss=90),
            _fake_workout("recovery",  tss=20),
            _fake_workout("sweetspot", tss=75),
            _fake_workout("recovery",  tss=20),
            _fake_workout("long_ride", tss=85),
            _fake_workout("recovery",  tss=20),
        ]
        idx = {"i": 0}

        def fake_pred(*a, **kw):
            p = predictions[idx["i"] % len(predictions)]
            idx["i"] += 1
            return {"next_workout": p}

        monkeypatch.setattr(inf, "_transformer_recommendation", fake_pred)

        result = inf._rollout_week(
            model=object(),
            activities=self._make_activities(),
            profile=_profile(training_days=3),
            ctl=50.0, atl=45.0, tsb=5.0,
            day0_workout=_fake_workout("vo2max", tss=95),
            health_by_date=None,
            training_days=3,
            horizon_override_days=30,
        )
        types_out = [w["workout_type"] for w in result]
        assert len(result) == 3
        # Hard sessions must be in the output
        assert "vo2max" in types_out or "threshold" in types_out


# ── planner: TSS budget with fewer training days ──────────────────────────────

class TestPlannerTrainingDays:

    def test_fewer_days_higher_per_session_tss(self):
        """Same CTL + phase but fewer training days → higher TSS per session."""
        from app.ml.planner import solve_week

        workouts_7 = ["tempo", "recovery", "threshold", "recovery", "sweetspot", "long_ride", "recovery"]
        workouts_4 = ["tempo", "threshold", "sweetspot", "long_ride"]

        tss_7, _ = solve_week(workouts_7, ctl=50, atl=45, tsb=5, phase="build")
        tss_4, _ = solve_week(workouts_4, ctl=50, atl=45, tsb=5, phase="build")

        # Weekly total should be similar (same CTL target)
        assert abs(sum(tss_7) - sum(tss_4)) < sum(tss_7) * 0.15, (
            "Weekly TSS totals should be within 15% of each other"
        )
        # But average per session must be higher for 4-day athlete
        assert sum(tss_4) / 4 > sum(tss_7) / 7

    def test_rest_day_padding_lowers_tsb(self):
        """PMC projection with rest days must show lower ATL than without."""
        from app.ml.planner import _project_pmc

        daily = [80.0, 60.0, 90.0, 50.0]  # 4 training days
        # Without rest padding (old behaviour)
        _, atl_no_rest, _ = _project_pmc(50, 45, daily)
        # With 3 rest days padded (correct 7-day projection)
        _, atl_with_rest, _ = _project_pmc(50, 45, daily + [0.0, 0.0, 0.0])

        # Rest days decay ATL — projection WITH rest must show lower ATL
        assert atl_with_rest < atl_no_rest

    def test_solve_week_safety_note_mentions_training_days(self):
        """solve_week should mention actual training days in its scaling note."""
        from app.ml.planner import solve_week

        _, notes = solve_week(
            ["tempo", "threshold", "sweetspot", "long_ride"],
            ctl=50, atl=45, tsb=5, phase="build"
        )
        # If scaling note emitted, it should reference 4 training days
        scaling_notes = [n for n in notes if "scaled" in n.lower()]
        if scaling_notes:
            assert "4" in scaling_notes[0]
