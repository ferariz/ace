"""
diagnostics/report.py

Aggregates individual CheckResults into a DiagnosticsReport.
Designed to attach cleanly to the API response layer.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from diagnostics.checks import (
    CheckResult,
    Severity,
    check_non_negativity,
    check_spatial_smoothness,
    check_value_ranges,
    check_wind_divergence,
)

_SEVERITY_RANK = {"pass": 0, "warn": 1, "fail": 2}


@dataclasses.dataclass
class DiagnosticsReport:
    """Aggregated physical plausibility report for one inference output."""
    checks: list[CheckResult]
    overall: Severity

    @property
    def passed(self) -> bool:
        return self.overall == "pass"

    @property
    def has_warnings(self) -> bool:
        return self.overall == "warn"

    @property
    def failed(self) -> bool:
        return self.overall == "fail"

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "passed": self.passed,
            "checks": [
                {
                    "name": c.name,
                    "severity": c.severity,
                    "message": c.message,
                    "measured": c.measured,
                    "threshold": c.threshold,
                }
                for c in self.checks
            ],
        }

    def __str__(self) -> str:
        lines = [f"DiagnosticsReport — overall: {self.overall.upper()}"]
        for c in self.checks:
            lines.append(f"  {c}")
        return "\n".join(lines)


def run_diagnostics(
    fields: dict[str, np.ndarray],
    strict: bool = False,
) -> DiagnosticsReport:
    """
    Run all physical checks on a dict of variable fields.

    Args:
        fields:  dict mapping variable name → 2D numpy array (lat x lon).
                 Pass the mean field from generate_numpy output.
        strict:  if True, raises ValueError on any failed check.

    Returns:
        DiagnosticsReport with per-check results and overall severity.
    """
    checks = [
        check_non_negativity(fields),
        check_value_ranges(fields),
        check_spatial_smoothness(fields),
        check_wind_divergence(fields),
    ]

    overall_rank = max(_SEVERITY_RANK[c.severity] for c in checks)
    overall: Severity = ["pass", "warn", "fail"][overall_rank]

    report = DiagnosticsReport(checks=checks, overall=overall)

    if strict and report.failed:
        raise ValueError(
            f"Physical diagnostics failed:\n{report}"
        )

    return report
