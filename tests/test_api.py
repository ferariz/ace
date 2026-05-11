"""
tests/test_api.py

Fast tests for the ACE API layer using FastAPI's TestClient.
No GPU, no checkpoint, no network — runs entirely against MockACEModel.
These should always pass in CI.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_returns_ok():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_type"] == "MockACEModel"
    assert isinstance(body["variables"], list)
    assert body["downscale_factor"] == 4


def test_variables_endpoint():
    r = client.get("/api/v1/variables")
    assert r.status_code == 200
    assert "variables" in r.json()
    assert "precipitation" in r.json()["variables"]


def test_downscale_response_structure():
    payload = {
        "region": {"lat_min": -40, "lat_max": -20, "lon_min": -70, "lon_max": -45},
        "date": "2020-01-15T00:00",
        "variables": ["precipitation", "tmp2m"],
        "n_samples": 2,
    }
    r = client.post("/api/v1/downscale", json=payload)
    assert r.status_code == 200
    body = r.json()

    assert body["downscale_factor"] == 4
    assert len(body["variables"]) == 2

    for var_result in body["variables"]:
        assert "name" in var_result
        assert "mean" in var_result
        assert "std" in var_result
        # mean and std should be 2D grids (list of lists)
        assert isinstance(var_result["mean"], list)
        assert isinstance(var_result["mean"][0], list)


def test_downscale_grid_shape():
    """Fine grid should be coarse * downscale_factor (48*4=192 x 96*4=384)."""
    payload = {
        "region": {"lat_min": -90, "lat_max": 90, "lon_min": -180, "lon_max": 180},
        "date": "2020-06-01T00:00",
        "variables": ["precipitation"],
        "n_samples": 2,
    }
    r = client.post("/api/v1/downscale", json=payload)
    assert r.status_code == 200
    grid = r.json()["variables"][0]["mean"]
    assert len(grid) == 192        # lat_fine
    assert len(grid[0]) == 384    # lon_fine


def test_unsupported_variable_returns_422():
    payload = {
        "region": {"lat_min": -40, "lat_max": -20, "lon_min": -70, "lon_max": -45},
        "date": "2020-01-15T00:00",
        "variables": ["not_a_real_variable"],
        "n_samples": 1,
    }
    r = client.post("/api/v1/downscale", json=payload)
    assert r.status_code == 422


def test_invalid_region_rejected():
    """lat_min > 90 should fail Pydantic validation."""
    payload = {
        "region": {"lat_min": -999, "lat_max": -20, "lon_min": -70, "lon_max": -45},
        "date": "2020-01-15T00:00",
        "variables": ["precipitation"],
        "n_samples": 1,
    }
    r = client.post("/api/v1/downscale", json=payload)
    assert r.status_code == 422
