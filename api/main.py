"""
api/main.py

FastAPI application entrypoint.

Run locally:
    uvicorn api.main:app --reload --port 8000

With a real checkpoint:
    CHECKPOINT_PATH=/path/to/checkpoint.pt uvicorn api.main:app --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from api.model import get_model
from api.router import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

app = FastAPI(
    title="ACE Downscaling API",
    description=(
        "HTTP interface to the Ai2 Climate Emulator (ACE) downscaling model. "
        "Wraps fme.downscaling.models.DiffusionModel behind a clean REST API."
    ),
    version="0.1.0",
)

app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
def _preload_model() -> None:
    """Warm up the model at startup so the first request isn't slow."""
    get_model()
