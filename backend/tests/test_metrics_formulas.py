"""
Pure-formula unit tests for metrics_service.

These do NOT touch the database — they exercise the math directly using
lightweight stub objects that mimic the SQLAlchemy ORM attributes used by
`compute_activity_metrics`.

Why this matters:
    Wrong TSS / IF / classification means wrong CTL/ATL/TSB → bad coach
    recommendations → real-world overtraining or undertraining for the user.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from app.services.metrics_service import (
    compute_activity_metrics,
    _classify_workout,
    CTL_ALPHA,
    ATL_ALPHA,
)


def _user(ftp=200, max_hr=190, resting_hr=50):
    profile = SimpleNamespace(ftp=ftp, max_hr=max_hr, resting_hr=resting_hr)
    return SimpleNamespace(profile=profile)


def _activity(**kwargs):
    defaults = dict(
        normalized_power=None,
        avg_power=None,
        avg_hr=None,
        max_power=None,
        max_hr=None,
        duration_seconds=None,
        intensity_factor=None,
        tss=None,
        variability_index=None,
        aerobic_efficiency=None,
        workout_type=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ─── TSS via Normalized Power (canonical formula) ────────────────────────────
def test_tss_from_np_one_hour_at_threshold_equals_100():
    """1h at NP=FTP → IF=1.0 → TSS=100. Coggan's golden test case."""
    user = _user(ftp=250)
    act = _activity(normalized_power=250, duration_seconds=3600)
    compute_activity_metrics(act, user)
    assert act.intensity_factor == pytest.approx(1.0, abs=0.001)
    assert act.tss == pytest.approx(100.0, abs=0.1)


def test_tss_from_np_two_hours_easy():
    """2h at IF=0.7 → TSS = 2 * 0.7^2 * 100 = 98."""
    user = _user(ftp=200)
    act = _activity(normalized_power=140, duration_seconds=7200)
    compute_activity_metrics(act, user)
    assert act.intensity_factor == pytest.approx(0.7, abs=0.001)
    assert act.tss == pytest.approx(98.0, abs=0.5)


# ─── TSS via avg_power fallback (no NP) ──────────────────────────────────────
def test_tss_avg_power_fallback():
    """No NP → IF = AP/FTP, TSS = h * IF² * 100."""
    user = _user(ftp=250)
    act = _activity(avg_power=200, duration_seconds=3600)
    compute_activity_metrics(act, user)
    expected_if = 200 / 250  # 0.8
    expected_tss = 1 * expected_if ** 2 * 100  # 64
    assert act.intensity_factor == pytest.approx(expected_if, abs=0.01)
    assert act.tss == pytest.approx(expected_tss, abs=0.5)


# ─── TSS via heart rate (no power at all) ────────────────────────────────────
def test_tss_hr_only_uses_hrr_fraction():
    """HR-based TSS for a power-less ride."""
    user = _user(ftp=200, max_hr=190, resting_hr=50)
    act = _activity(avg_hr=140, duration_seconds=3600)
    compute_activity_metrics(act, user)
    # HRR fraction = (140-50)/(190-50) = 0.6428...
    hrr = (140 - 50) / (190 - 50)
    expected_if = hrr ** 0.91
    assert act.intensity_factor == pytest.approx(expected_if, abs=0.005)
    # Should produce a sane sub-100 TSS for a ~1h tempo ride
    assert 40 < act.tss < 80


def test_tss_hr_only_skipped_without_hr_zones():
    """No HR + no power → no TSS computed (we don't fabricate it)."""
    user = _user()
    act = _activity(duration_seconds=3600)
    compute_activity_metrics(act, user)
    assert act.tss is None


def test_tss_hr_only_skipped_when_profile_missing_hr_zones():
    """User with no max_hr/resting_hr → HR fallback shouldn't run."""
    user = SimpleNamespace(profile=SimpleNamespace(ftp=200, max_hr=None, resting_hr=None))
    act = _activity(avg_hr=140, duration_seconds=3600)
    compute_activity_metrics(act, user)
    assert act.tss is None


# ─── Variability Index + Aerobic Efficiency ─────────────────────────────────
def test_variability_index():
    user = _user(ftp=200)
    act = _activity(normalized_power=220, avg_power=200, duration_seconds=3600)
    compute_activity_metrics(act, user)
    assert act.variability_index == pytest.approx(1.10, abs=0.001)


def test_aerobic_efficiency():
    user = _user(ftp=200)
    act = _activity(normalized_power=200, avg_power=200, avg_hr=150, duration_seconds=3600)
    compute_activity_metrics(act, user)
    assert act.aerobic_efficiency == pytest.approx(200 / 150, abs=0.001)


# ─── Workout classifier boundaries ──────────────────────────────────────────
@pytest.mark.parametrize(
    "if_, duration_s, expected",
    [
        (0.40, 1800, "recovery"),
        (0.54, 1800, "recovery"),
        (0.55, 1800, "easy"),       # boundary: 0.55 → no longer recovery
        (0.70, 3000, "easy"),       # short = easy
        (0.70, 7200, "endurance"),  # long = endurance
        (0.74, 7200, "endurance"),
        (0.75, 3600, "tempo"),      # boundary: tempo starts at 0.75
        (0.87, 3600, "tempo"),
        (0.88, 3600, "sweetspot"),  # boundary
        (0.94, 3600, "sweetspot"),
        (0.95, 3600, "threshold"),  # boundary
        (1.04, 3600, "threshold"),
        (1.05, 1200, "vo2max"),     # boundary
        (1.19, 600,  "vo2max"),
        (1.20, 300,  "sprint"),     # boundary
        (1.50, 60,   "sprint"),
    ],
)
def test_classify_workout_boundaries(if_, duration_s, expected):
    assert _classify_workout(if_, duration_s) == expected


# ─── EMA constants are correct ───────────────────────────────────────────────
def test_ema_constants():
    """CTL uses 42-day EMA, ATL uses 7-day EMA — Coggan standard."""
    assert CTL_ALPHA == pytest.approx(2 / 43, abs=1e-9)
    assert ATL_ALPHA == pytest.approx(2 / 8, abs=1e-9)


# ─── EMA convergence (CTL/ATL math) ──────────────────────────────────────────
def test_ctl_converges_to_constant_tss():
    """Apply TSS=100 every day for ~6 months. CTL must converge near 100."""
    ctl = 0.0
    for _ in range(180):
        ctl = ctl + CTL_ALPHA * (100 - ctl)
    assert ctl == pytest.approx(100.0, abs=1.0)


def test_atl_responds_faster_than_ctl():
    """A single 200 TSS pulse should move ATL much more than CTL."""
    ctl = atl = 0.0
    ctl = ctl + CTL_ALPHA * (200 - ctl)
    atl = atl + ATL_ALPHA * (200 - atl)
    assert atl > ctl * 4  # ATL alpha is ~5× CTL alpha
