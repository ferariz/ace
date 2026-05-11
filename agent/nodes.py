"""
agent/nodes.py

The three node functions for the ACE climate query agent.

Each function receives the full AgentState and returns a dict
containing only the keys it updates — that's the LangGraph contract.
"""
from __future__ import annotations

import json
import logging
import os
import statistics

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared LLM instance (lazy init)
# ---------------------------------------------------------------------------

_llm: ChatOpenAI | None = None


def _get_llm() -> ChatAnthropic:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"],
            max_tokens=1024,
        )
    return _llm


# ---------------------------------------------------------------------------
# Node 1 — parse_query
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = """\
You are a climate science assistant. Extract structured parameters from a
natural language climate query.

Return ONLY a JSON object with this exact schema:
{
  "region": {
    "lat_min": float,   // degrees, -90 to 90
    "lat_max": float,
    "lon_min": float,   // degrees, -180 to 360
    "lon_max": float
  },
  "date": "YYYY-MM-DDTHH:MM",   // ISO format, use 00:00 if unspecified
  "variables": ["precipitation", "tmp2m"],  // subset of: precipitation, tmp2m, u10m, v10m
  "n_samples": 4
}

Rules:
- If the query mentions a named region (e.g. "southern Brazil", "Río de la Plata"),
  infer approximate bounding box coordinates.
- If no date is mentioned, use "2020-01-15T00:00".
- Always include "precipitation" and "tmp2m" unless the user asks for specific variables.
- Return ONLY the JSON. No explanation, no markdown fences.
"""


def parse_query(state: AgentState) -> dict:
    """Node 1: use Claude to extract a DownscaleRequest from the user's question."""
    query = state["query"]
    logger.info("parse_query: %s", query)

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=_PARSE_SYSTEM),
            HumanMessage(content=query),
        ])
        raw = response.content.strip()
        request = json.loads(raw)
        logger.info("Parsed request: %s", request)
        return {"request": request, "error": None}

    except json.JSONDecodeError as e:
        logger.warning("parse_query: LLM returned non-JSON: %s", e)
        return {"error": f"Could not parse query into a climate request: {e}"}
    except Exception as e:
        logger.exception("parse_query failed")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Node 2 — call_api
# ---------------------------------------------------------------------------

_API_BASE = os.getenv("ACE_API_BASE", "http://localhost:8000")


def call_api(state: AgentState) -> dict:
    """Node 2: POST the parsed request to the ACE downscaling API."""
    if state.get("error"):
        return {}   # skip — let error propagate to END

    request = state["request"]
    logger.info("call_api: posting to %s/api/v1/downscale", _API_BASE)

    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(f"{_API_BASE}/api/v1/downscale", json=request)
            r.raise_for_status()

        response = r.json()

        # Summarise grids so the interpreter LLM doesn't get 200k floats
        summary = {
            "region": response["region"],
            "date": response["date"],
            "downscale_factor": response["downscale_factor"],
            "variables": [],
        }
        for var in response["variables"]:
            flat_mean = [v for row in var["mean"] for v in row]
            flat_std  = [v for row in var["std"]  for v in row]
            summary["variables"].append({
                "name": var["name"],
                "mean_stats": {
                    "min":    round(min(flat_mean), 4),
                    "max":    round(max(flat_mean), 4),
                    "mean":   round(statistics.mean(flat_mean), 4),
                    "stdev":  round(statistics.stdev(flat_mean), 4),
                },
                "uncertainty": {
                    "mean_std": round(statistics.mean(flat_std), 4),
                    "max_std":  round(max(flat_std), 4),
                },
            })

        logger.info("call_api: summary built for %d variables", len(summary["variables"]))
        return {"api_response": summary, "error": None}

    except httpx.HTTPStatusError as e:
        return {"error": f"API returned {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        logger.exception("call_api failed")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Node 3 — interpret_result
# ---------------------------------------------------------------------------

_INTERPRET_SYSTEM = """\
You are a climate scientist explaining model output to a non-expert.
You will receive a summary of ACE (AI Climate Emulator) downscaling output.

Variable units and physical constraints:
- precipitation: mm/day, MUST be >= 0. Flag if negative values appear.
- tmp2m: Kelvin (subtract 273.15 for Celsius). Typical range 240-320 K.
- u10m, v10m: wind components in m/s, can be negative (direction).

Write 2-3 sentences that:
  1. Directly answer the user's original question.
  2. State key values converted to intuitive units (mm/day, °C, m/s).
  3. Note uncertainty level in plain language.
  4. If any value violates physical constraints (e.g. negative precipitation),
     explicitly flag it as a model quality issue rather than narrating it.

Be specific but accessible. Do not use jargon.
"""


def interpret_result(state: AgentState) -> dict:
    """Node 3: use Claude to turn grid statistics into a plain-language answer."""
    if state.get("error"):
        return {"answer": f"Sorry, I could not complete the query: {state['error']}"}

    context = json.dumps({
        "original_question": state["query"],
        "model_output_summary": state["api_response"],
    }, indent=2)

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=_INTERPRET_SYSTEM),
            HumanMessage(content=context),
        ])
        return {"answer": response.content.strip()}

    except Exception as e:
        logger.exception("interpret_result failed")
        return {"answer": f"Model ran successfully but interpretation failed: {e}"}
