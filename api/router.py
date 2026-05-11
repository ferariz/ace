"""
api/router.py

FastAPI router exposing ACE downscaling inference via HTTP.

Endpoints:
  POST /downscale        — Run the emulator for a region + date.
  GET  /health           — Liveness check (also shows model mode).
  GET  /variables        — List variables the loaded model produces.

The router never imports from fme directly — all model access goes through
api/model.py, which means the mock path works identically to the real one.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
from fastapi import APIRouter, HTTPException

from api.model import get_model
from api.schemas import (
    BoundingBox,
    DownscaleRequest,
    DownscaleResponse,
    VariableResult,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coarse_inputs(
    request: DownscaleRequest,
    model_var_names: list[str],
    coarse_shape: tuple[int, int] = (48, 96),
) -> dict[str, np.ndarray]:
    """
    Build a minimal coarse input dict for the model.

    In production this would load real ERA5 / FV3GFS coarse data for
    request.date and request.region. For now we return synthetic zeros
    (the mock model ignores them; the real model will produce non-trivial
    output because of the diffusion noise draw).
    """
    lat, lon = coarse_shape
    rng = np.random.default_rng(seed=int(
        datetime.fromisoformat(request.date).timestamp()
    ))
    return {
        var: rng.standard_normal((lat, lon)).astype(np.float32)
        for var in model_var_names
    }


def _tensor_to_grid(arr: np.ndarray) -> list[list[float]]:
    """Convert a 2-D numpy array to a nested Python list (JSON-serialisable)."""
    return arr.tolist()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/health")
def health() -> dict:
    """Liveness + model status."""
    model = get_model()
    return {
        "status": "ok",
        "model_type": type(model).__name__,
        "variables": model.out_names,
        "downscale_factor": model.downscale_factor,
    }


@router.get("/variables")
def list_variables() -> dict:
    """Return the variable names the loaded model can produce."""
    model = get_model()
    return {"variables": model.out_names}


@router.post("/downscale", response_model=DownscaleResponse)
def downscale(request: DownscaleRequest) -> DownscaleResponse:
    """
    Run ACE downscaling for a given region, date, and variable set.

    The response includes per-variable mean and std across `n_samples`
    diffusion draws — giving you both a best-estimate and an uncertainty map.
    """
    model = get_model()

    # Validate requested variables against what the model supports
    unsupported = set(request.variables) - set(model.out_names)
    if unsupported:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported variables: {sorted(unsupported)}. "
                f"Available: {sorted(model.out_names)}"
            ),
        )

    # Build coarse inputs (synthetic for now — real data loader goes here)
    coarse_data = _make_coarse_inputs(request, model.out_names)

    logger.info(
        "Running inference: region=%s date=%s n_samples=%d",
        request.region,
        request.date,
        request.n_samples,
    )

    try:
        generated = model.generate_numpy(coarse_data, n_samples=request.n_samples)
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc

    # Build response — compute mean + std across sample dimension
    variable_results = []
    for var in request.variables:
        samples = generated[var]          # shape: (n_samples, lat_fine, lon_fine)
        mean_grid = _tensor_to_grid(np.mean(samples, axis=0))
        std_grid  = _tensor_to_grid(np.std(samples,  axis=0))
        variable_results.append(
            VariableResult(name=var, mean=mean_grid, std=std_grid)
        )

    return DownscaleResponse(
        region=request.region,
        date=request.date,
        downscale_factor=model.downscale_factor,
        variables=variable_results,
    )
