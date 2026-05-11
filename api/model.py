"""
api/model.py

Singleton loader for the ACE DiffusionModel (CheckpointModelConfig).

Design decisions:
- Model loads once at startup, stays in memory for the API lifetime.
- If no checkpoint path is configured, falls back to MockACEModel so the
  entire API is runnable and testable without a GPU or downloaded weights.
- Keeps fme internals behind this module — router.py never imports from fme directly.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol — what the router needs from any model (real or mock)
# ---------------------------------------------------------------------------

@runtime_checkable
class ACEModelProtocol(Protocol):
    """Minimal interface the router depends on."""

    @property
    def out_names(self) -> list[str]:
        """Variable names produced by the model (e.g. ['precipitation', 'tmp2m'])."""
        ...

    @property
    def downscale_factor(self) -> int:
        """Integer ratio coarse → fine resolution."""
        ...

    def generate_numpy(
        self,
        coarse_data: dict[str, np.ndarray],
        n_samples: int = 4,
    ) -> dict[str, np.ndarray]:
        """
        Run inference and return outputs as plain numpy arrays.

        Args:
            coarse_data: dict mapping variable name → ndarray of shape (lat, lon).
            n_samples:   Number of diffusion samples to draw (uncertainty estimate).

        Returns:
            dict mapping variable name → ndarray of shape (n_samples, lat_fine, lon_fine).
        """
        ...


# ---------------------------------------------------------------------------
# Mock model — CPU-only, zero dependencies on weights
# ---------------------------------------------------------------------------

class MockACEModel:
    """
    Fake model for local development and CI.

    Returns synthetic Gaussian noise shaped like real ACE output.
    Useful for:
      - Testing the API layer end-to-end without a checkpoint.
      - CI pipelines that don't have GPU access.
      - Interview demos where you want to show the plumbing, not the weights.
    """

    _DEFAULT_VARIABLES = ["precipitation", "tmp2m", "u10m", "v10m"]
    _COARSE_SHAPE = (48, 96)   # typical ACE coarse lat/lon grid
    _DOWNSCALE_FACTOR = 4

    @property
    def out_names(self) -> list[str]:
        return self._DEFAULT_VARIABLES

    @property
    def downscale_factor(self) -> int:
        return self._DOWNSCALE_FACTOR

    def generate_numpy(
        self,
        coarse_data: dict[str, np.ndarray],
        n_samples: int = 4,
    ) -> dict[str, np.ndarray]:
        lat_c, lon_c = self._COARSE_SHAPE
        lat_f = lat_c * self._DOWNSCALE_FACTOR
        lon_f = lon_c * self._DOWNSCALE_FACTOR

        rng = np.random.default_rng(seed=42)
        return {
            var: rng.standard_normal((n_samples, lat_f, lon_f)).astype(np.float32)
            for var in self.out_names
        }


# ---------------------------------------------------------------------------
# Real model wrapper — thin adapter over fme's DiffusionModel
# ---------------------------------------------------------------------------

class RealACEModel:
    """
    Wraps fme.downscaling.models.DiffusionModel behind ACEModelProtocol.

    Loaded from a local checkpoint file (.pt) via CheckpointModelConfig.
    Only instantiated when CHECKPOINT_PATH env var is set.
    """

    def __init__(self, checkpoint_path: str) -> None:
        # Import fme lazily so mock path never pulls in torch
        from fme.downscaling.models import CheckpointModelConfig

        logger.info("Loading ACE checkpoint from %s ...", checkpoint_path)
        cfg = CheckpointModelConfig(checkpoint_path=checkpoint_path)
        self._model = cfg.build()
        self._model.module.eval()
        logger.info(
            "Checkpoint loaded. out_names=%s  downscale_factor=%d",
            self._model.config.out_names,
            self._model.downscale_factor,
        )

    @property
    def out_names(self) -> list[str]:
        return list(self._model.config.out_names)

    @property
    def downscale_factor(self) -> int:
        return self._model.downscale_factor

    def generate_numpy(
        self,
        coarse_data: dict[str, np.ndarray],
        n_samples: int = 4,
    ) -> dict[str, np.ndarray]:
        import torch

        # Convert numpy inputs → torch tensors with batch dim
        torch_data = {
            k: torch.from_numpy(v).unsqueeze(0)   # (1, lat, lon)
            for k, v in coarse_data.items()
        }

        with torch.no_grad():
            generated = self._model.generate(
                coarse_data=torch_data,
                static_inputs=None,
                n_samples=n_samples,
            )[0]  # generate() returns (TensorDict, norm_tensor, latent_steps)

        # Convert back to numpy: (n_samples, lat_fine, lon_fine)
        return {
            var: tensor.squeeze(0).cpu().numpy()
            for var, tensor in generated.items()
        }


# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------

_model_instance: ACEModelProtocol | None = None


def get_model() -> ACEModelProtocol:
    """
    Return the loaded model, initialising on first call.

    Resolution order:
      1. CHECKPOINT_PATH env var set → RealACEModel
      2. Otherwise                   → MockACEModel (no GPU needed)
    """
    global _model_instance

    if _model_instance is None:
        checkpoint_path = os.getenv("CHECKPOINT_PATH", "")
        if checkpoint_path:
            _model_instance = RealACEModel(checkpoint_path)
        else:
            logger.warning(
                "CHECKPOINT_PATH not set — using MockACEModel. "
                "Outputs are synthetic noise, not real climate predictions."
            )
            _model_instance = MockACEModel()

    return _model_instance
