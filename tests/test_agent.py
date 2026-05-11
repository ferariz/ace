"""
tests/test_agent.py

Tests for the LangGraph agent — all LLM and HTTP calls are mocked.
No OPENAI_API_KEY, no running server, no GPU required.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent.graph import build_graph
from agent.state import AgentState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VALID_REQUEST = {
    "region": {"lat_min": -35, "lat_max": -20, "lon_min": -60, "lon_max": -45},
    "date": "2020-01-15T00:00",
    "variables": ["precipitation", "tmp2m"],
    "n_samples": 4,
}

VALID_API_RESPONSE = {
    "region": VALID_REQUEST["region"],
    "date": "2020-01-15T00:00",
    "downscale_factor": 4,
    "variables": [
        {
            "name": "precipitation",
            "mean": [[2.1] * 384] * 192,
            "std":  [[0.4] * 384] * 192,
        },
        {
            "name": "tmp2m",
            "mean": [[292.0] * 384] * 192,
            "std":  [[1.2] * 384] * 192,
        },
    ],
}


def _mock_llm_response(content: str) -> MagicMock:
    """Return a mock that looks like a LangChain AIMessage."""
    msg = MagicMock()
    msg.content = content
    return msg


# ---------------------------------------------------------------------------
# Unit tests — individual nodes
# ---------------------------------------------------------------------------

@patch("agent.nodes._get_llm")
def test_parse_query_returns_valid_request(mock_get_llm):
    """parse_query should return a dict with region, date, variables."""
    mock_get_llm.return_value.invoke.return_value = _mock_llm_response(
        json.dumps(VALID_REQUEST)
    )

    from agent.nodes import parse_query
    result = parse_query({"query": "Precipitation over southern Brazil?"})

    assert result["error"] is None
    assert "region" in result["request"]
    assert "date" in result["request"]
    assert "variables" in result["request"]


@patch("agent.nodes._get_llm")
def test_parse_query_handles_bad_json(mock_get_llm):
    """parse_query should set error if LLM returns non-JSON."""
    mock_get_llm.return_value.invoke.return_value = _mock_llm_response(
        "Sorry, I cannot parse that."
    )

    from agent.nodes import parse_query
    result = parse_query({"query": "gibberish query"})

    assert result["error"] is not None
    assert result.get("request") is None


@patch("agent.nodes.httpx.Client")
def test_call_api_returns_summary(mock_client_cls):
    """call_api should summarise grid stats, not return raw arrays."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = VALID_API_RESPONSE
    mock_resp.raise_for_status = MagicMock()
    mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp

    from agent.nodes import call_api
    result = call_api({"query": "test", "request": VALID_REQUEST, "error": None})

    assert result["error"] is None
    summary = result["api_response"]
    assert "variables" in summary
    var = summary["variables"][0]
    assert "mean_stats" in var
    assert "uncertainty" in var
    # Raw grids should NOT be in the summary (too large for LLM context)
    assert "mean" not in var
    assert "std" not in var


def test_call_api_skips_on_error():
    """call_api should do nothing if a previous node set an error."""
    from agent.nodes import call_api
    result = call_api({"query": "test", "error": "parse failed"})
    assert result == {}


@patch("agent.nodes._get_llm")
def test_interpret_result_returns_answer(mock_get_llm):
    """interpret_result should return a non-empty answer string."""
    mock_get_llm.return_value.invoke.return_value = _mock_llm_response(
        "Precipitation over southern Brazil averages 2.1 mm/day in January 2020."
    )

    from agent.nodes import interpret_result
    result = interpret_result({
        "query": "Precipitation over southern Brazil?",
        "api_response": {"variables": []},
        "error": None,
    })

    assert "answer" in result
    assert len(result["answer"]) > 10


def test_interpret_result_handles_error_state():
    """interpret_result should surface errors gracefully as an answer."""
    from agent.nodes import interpret_result
    result = interpret_result({
        "query": "test",
        "error": "API was unreachable",
    })
    assert "answer" in result
    assert "API was unreachable" in result["answer"]


# ---------------------------------------------------------------------------
# Integration test — full graph run
# ---------------------------------------------------------------------------

@patch("agent.nodes.httpx.Client")
@patch("agent.nodes._get_llm")
def test_full_graph_happy_path(mock_get_llm, mock_client_cls):
    """Full graph should produce a non-empty answer for a valid query."""
    # Mock LLM: first call returns parsed JSON, second returns interpretation
    mock_get_llm.return_value.invoke.side_effect = [
        _mock_llm_response(json.dumps(VALID_REQUEST)),
        _mock_llm_response("Precipitation averages 2.1 mm/day with low uncertainty."),
    ]

    mock_resp = MagicMock()
    mock_resp.json.return_value = VALID_API_RESPONSE
    mock_resp.raise_for_status = MagicMock()
    mock_client_cls.return_value.__enter__.return_value.post.return_value = mock_resp

    graph = build_graph()
    result = graph.invoke({"query": "Precipitation over southern Brazil?"})

    assert result.get("error") is None
    assert result.get("answer") is not None
    assert len(result["answer"]) > 10


@patch("agent.nodes._get_llm")
def test_full_graph_parse_failure(mock_get_llm):
    """Graph should reach END gracefully when parse_query fails."""
    mock_get_llm.return_value.invoke.side_effect = [
        _mock_llm_response("not json at all"),           # parse fails
        _mock_llm_response("Sorry, could not process."), # interpret error
    ]

    graph = build_graph()
    result = graph.invoke({"query": "completely unparseable query ##!!"})

    # Should always have an answer, even on failure
    assert result.get("answer") is not None
