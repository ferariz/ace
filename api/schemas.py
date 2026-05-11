"""
Pydantic schemas for the ACE API layer.

These define the contract between the agent/client and the ACE emulator.
Derived from fme.downscaling.models.DiffusionModel.generate() signature.
"""
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    lat_min: float = Field(..., ge=-90, le=90)
    lat_max: float = Field(..., ge=-90, le=90)
    lon_min: float = Field(..., ge=-180, le=360)
    lon_max: float = Field(..., ge=-180, le=360)


class DownscaleRequest(BaseModel):
    """Input to a single downscaling inference call."""
    region: BoundingBox
    date: str = Field(..., description="ISO datetime, e.g. 2020-01-15T00:00")
    variables: list[str] = Field(
        default=["precipitation", "temperature"],
        description="Output variables to return"
    )
    n_samples: int = Field(default=4, ge=1, le=64)


class VariableResult(BaseModel):
    name: str
    mean: list[list[float]]   # [lat, lon] grid
    std: list[list[float]]    # uncertainty across samples


class DownscaleResponse(BaseModel):
    """Output of a single downscaling inference call."""
    region: BoundingBox
    date: str
    downscale_factor: int
    variables: list[VariableResult]
