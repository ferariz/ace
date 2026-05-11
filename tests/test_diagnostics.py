"""
tests/test_diagnostics.py

Tests for the physical diagnostics module.
All numpy — no GPU, no API, no LLM.
"""
from __future__ import annotations

import numpy as np
import pytest

from diagnostics.checks import (
    check_non_negativity,
    check_spatial_smoothness,
    check_value_ranges,
    check_wind_divergence,
)
from diagnostics.report import run_diagnostics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _good_fields(seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "precipitation": rng.lognormal(0.5, 0.8, (192, 384)).astype("f"),
        "tmp2m":         rng.normal(290, 8, (192, 384)).astype("f"),
        "u10m":          rng.normal(0, 5, (192, 384)).astype("f"),
        "v10m":          rng.normal(0, 5, (192, 384)).astype("f"),
    }


def _bad_fields(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {k: rng.standard_normal((192, 384)).astype("f")
            for k in ["precipitation", "tmp2m", "u10m", "v10m"]}


# ---------------------------------------------------------------------------
# check_non_negativity
# ---------------------------------------------------------------------------

def test_non_negativity_passes_on_good_precip():
    result = check_non_negativity(_good_fields())
    assert result.severity == "pass"


def test_non_negativity_fails_on_negative_precip():
    fields = _good_fields()
    fields["precipitation"][0, 0] = -1.0
    result = check_non_negativity(fields)
    assert result.severity == "fail"
    assert "precipitation" in result.message


def test_non_negativity_ignores_negative_winds():
    """Wind components can be negative — should not trigger this check."""
    fields = _good_fields()
    fields["u10m"][:] = -10.0
    result = check_non_negativity(fields)
    assert result.severity == "pass"


# ---------------------------------------------------------------------------
# check_value_ranges
# ---------------------------------------------------------------------------

def test_value_ranges_passes_on_good_fields():
    result = check_value_ranges(_good_fields())
    assert result.severity == "pass"


def test_value_ranges_fails_on_extreme_temperature():
    fields = _good_fields()
    fields["tmp2m"][:] = 999.0   # 726°C — clearly wrong
    result = check_value_ranges(fields)
    assert result.severity in ("warn", "fail")
    assert "tmp2m" in result.message


def test_value_ranges_warn_vs_fail_threshold():
    """Less than 1% out of range → warn. More than 1% → fail."""
    fields = _good_fields()
    # Set exactly 0.5% of cells out of range
    n = int(192 * 384 * 0.005)
    fields["precipitation"].flat[:n] = 999.0
    result = check_value_ranges(fields)
    assert result.severity == "warn"

    # Set 2% out of range → fail
    n2 = int(192 * 384 * 0.02)
    fields["precipitation"].flat[:n2] = 999.0
    result2 = check_value_ranges(fields)
    assert result2.severity == "fail"


# ---------------------------------------------------------------------------
# check_spatial_smoothness
# ---------------------------------------------------------------------------

def test_smoothness_passes_on_good_fields():
    result = check_spatial_smoothness(_good_fields())
    assert result.severity == "pass"


def test_smoothness_passes_on_tiny_field():
    fields = {"precipitation": np.ones((2, 2), dtype="f")}
    result = check_spatial_smoothness(fields)
    assert result.severity == "pass"
    # tiny field — either skips or passes cleanly


# ---------------------------------------------------------------------------
# check_wind_divergence
# ---------------------------------------------------------------------------

def test_wind_divergence_passes_on_good_fields():
    result = check_wind_divergence(_good_fields())
    assert result.severity == "pass"


def test_wind_divergence_skips_without_wind_fields():
    fields = {"precipitation": _good_fields()["precipitation"]}
    result = check_wind_divergence(fields)
    assert result.severity == "pass"
    # tiny field — either skips or passes cleanly


def test_wind_divergence_fails_on_extreme_winds():
    """Test by lowering the threshold, not by creating extreme fields."""
    fields = _good_fields()
    # good_fields has mean divergence ~4.0 — use threshold below that
    result = check_wind_divergence(fields, max_mean_divergence=0.1)
    assert result.severity in ("warn", "fail")


# ---------------------------------------------------------------------------
# run_diagnostics (report aggregator)
# ---------------------------------------------------------------------------

def test_run_diagnostics_good_output():
    report = run_diagnostics(_good_fields())
    assert report.overall == "pass"
    assert report.passed
    assert len(report.checks) == 4


def test_run_diagnostics_bad_output():
    report = run_diagnostics(_bad_fields())
    assert report.overall == "fail"
    assert report.failed


def test_run_diagnostics_to_dict():
    report = run_diagnostics(_good_fields())
    d = report.to_dict()
    assert "overall" in d
    assert "checks" in d
    assert isinstance(d["checks"], list)
    assert all("name" in c and "severity" in c for c in d["checks"])


def test_run_diagnostics_strict_mode_raises():
    with pytest.raises(ValueError, match="Physical diagnostics failed"):
        run_diagnostics(_bad_fields(), strict=True)


def test_run_diagnostics_strict_mode_passes_silently():
    report = run_diagnostics(_good_fields(), strict=True)
    assert report.passed
