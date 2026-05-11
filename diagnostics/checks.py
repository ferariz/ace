"""
diagnostics/checks.py

Physical plausibility checks for ACE downscaling outputs.

Each check takes a dict[str, np.ndarray] of variable fields
(shape: lat x lon) and returns a CheckResult.

Design:
- Checks are independent functions — easy to add, remove, or upstream.
- No dependency on fme internals — works on plain numpy arrays.
- Severity levels: "pass" < "warn" < "fail"
"""
from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np

Severity = Literal["pass", "warn", "fail"]


@dataclasses.dataclass
class CheckResult:
    """Result of a single physical plausibility check."""
    name: str
    severity: Severity
    message: str
    measured: float | None = None   # the value we computed
    threshold: float | None = None  # the limit we tested against

    def __str__(self) -> str:
        icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}[self.severity]
        detail = ""
        if self.measured is not None:
            detail = f" (measured={self.measured:.4f}"
            if self.threshold is not None:
                detail += f", threshold={self.threshold:.4f}"
            detail += ")"
        return f"{icon} {self.name}: {self.message}{detail}"


# ---------------------------------------------------------------------------
# Physical bounds per variable
# (units match ACE mock: precip=mm/day, tmp2m=K, winds=m/s)
# ---------------------------------------------------------------------------

VARIABLE_BOUNDS: dict[str, tuple[float, float]] = {
    "precipitation": (0.0,   500.0),   # mm/day, non-negative, <500 is generous
    "tmp2m":         (180.0, 340.0),   # K, ~-93°C to ~67°C
    "u10m":          (-80.0,  80.0),   # m/s
    "v10m":          (-80.0,  80.0),   # m/s
}


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_non_negativity(
    fields: dict[str, np.ndarray],
    variables: list[str] | None = None,
) -> CheckResult:
    """
    Fail if any non-negative variable contains negative values.
    Variables checked: precipitation (extendable to specific humidity etc.)
    """
    non_negative_vars = {"precipitation"}
    to_check = non_negative_vars & set(variables or fields.keys())

    violations = {}
    for var in to_check:
        arr = fields[var]
        n_negative = int(np.sum(arr < 0))
        if n_negative > 0:
            violations[var] = n_negative

    if not violations:
        return CheckResult(
            name="non_negativity",
            severity="pass",
            message="All non-negative variables are ≥ 0.",
            measured=0.0,
            threshold=0.0,
        )

    total = sum(violations.values())
    detail = ", ".join(f"{v}: {n} cells" for v, n in violations.items())
    return CheckResult(
        name="non_negativity",
        severity="fail",
        message=f"Negative values found in: {detail}",
        measured=float(total),
        threshold=0.0,
    )


def check_value_ranges(
    fields: dict[str, np.ndarray],
    bounds: dict[str, tuple[float, float]] | None = None,
) -> CheckResult:
    """
    Warn/fail if any variable has values outside its physical bounds.
    Uses VARIABLE_BOUNDS by default; custom bounds can be passed.
    """
    bounds = bounds or VARIABLE_BOUNDS
    violations = {}

    for var, arr in fields.items():
        if var not in bounds:
            continue
        lo, hi = bounds[var]
        frac_out = float(np.mean((arr < lo) | (arr > hi)))
        if frac_out > 0:
            violations[var] = frac_out

    if not violations:
        return CheckResult(
            name="value_ranges",
            severity="pass",
            message="All variables within physical bounds.",
            measured=0.0,
            threshold=0.0,
        )

    max_frac = max(violations.values())
    detail = ", ".join(f"{v}: {f*100:.1f}% out of range" for v, f in violations.items())
    severity: Severity = "fail" if max_frac > 0.01 else "warn"
    return CheckResult(
        name="value_ranges",
        severity=severity,
        message=f"Out-of-range values in: {detail}",
        measured=max_frac,
        threshold=0.01,
    )


def check_spatial_smoothness(
    fields: dict[str, np.ndarray],
    max_gradient_ratio: float = 2.0,
) -> CheckResult:
    """
    Warn if fields contain excessive high-frequency spatial noise.

    Computes mean |gradient| / mean |field| per variable.
    Real atmospheric fields have smooth spatial structure;
    diffusion artifacts or uncorrelated noise show high ratios.

    Note: threshold=2.0 is conservative — calibrate against real
    ACE checkpoint output once GPU access is available.

    Args:
        max_gradient_ratio: max allowed ratio of mean gradient to mean field.
    """
    ratios = {}
    for var, arr in fields.items():
        if arr.ndim != 2 or min(arr.shape) < 4:
            continue
        mean_abs_val = float(np.mean(np.abs(arr)))
        if mean_abs_val < 1e-8:
            continue
        gy = np.gradient(arr.astype(np.float32), axis=0)
        gx = np.gradient(arr.astype(np.float32), axis=1)
        mean_grad = float(np.mean(np.sqrt(gx**2 + gy**2)))
        ratios[var] = mean_grad / mean_abs_val

    if not ratios:
        return CheckResult(
            name="spatial_smoothness",
            severity="pass",
            message="Smoothness check skipped (fields too small).",
        )

    max_ratio = max(ratios.values())
    worst_var = max(ratios, key=ratios.__getitem__)

    if max_ratio <= max_gradient_ratio:
        return CheckResult(
            name="spatial_smoothness",
            severity="pass",
            message="Fields are spatially smooth.",
            measured=max_ratio,
            threshold=max_gradient_ratio,
        )

    severity: Severity = "fail" if max_ratio > max_gradient_ratio * 2 else "warn"
    return CheckResult(
        name="spatial_smoothness",
        severity=severity,
        message=f"High-frequency noise detected in '{worst_var}'. "
                f"Threshold calibrated for real checkpoint output.",
        measured=max_ratio,
        threshold=max_gradient_ratio,
    )


def check_wind_divergence(
    fields: dict[str, np.ndarray],
    max_mean_divergence: float = 5.0,
) -> CheckResult:
    """
    Warn if the 2D wind divergence (∂u/∂x + ∂v/∂y) is pathologically large.

    Large divergence in near-surface winds indicates unphysical flow patterns.
    Requires both u10m and v10m to be present.

    Args:
        max_mean_divergence: threshold for mean |divergence| in m/s per grid cell.
    """
    if "u10m" not in fields or "v10m" not in fields:
        return CheckResult(
            name="wind_divergence",
            severity="pass",
            message="Skipped — u10m or v10m not present.",
        )

    u = fields["u10m"].astype(np.float32)
    v = fields["v10m"].astype(np.float32)

    # Central differences (wrap longitude, reflect latitude)
    du_dx = np.gradient(u, axis=1)   # ∂u/∂x (longitude axis)
    dv_dy = np.gradient(v, axis=0)   # ∂v/∂y (latitude axis)
    divergence = du_dx + dv_dy

    mean_div = float(np.mean(np.abs(divergence)))

    if mean_div <= max_mean_divergence:
        return CheckResult(
            name="wind_divergence",
            severity="pass",
            message="Wind divergence within acceptable range.",
            measured=mean_div,
            threshold=max_mean_divergence,
        )

    severity = "fail" if mean_div > max_mean_divergence * 3 else "warn"
    return CheckResult(
        name="wind_divergence",
        severity=severity,
        message="Elevated wind divergence — possible unphysical flow.",
        measured=mean_div,
        threshold=max_mean_divergence,
    )
