"""
agent/state.py

LangGraph state for the ACE climate query agent.
Every node receives this state and returns a partial update.
"""
from __future__ import annotations
from typing import TypedDict


class AgentState(TypedDict, total=False):
    query: str                    # original natural language question
    request: dict | None          # parsed DownscaleRequest as dict
    api_response: dict | None     # raw response from /api/v1/downscale
    answer: str | None            # final plain-language answer
    error: str | None             # set if any node fails gracefully
