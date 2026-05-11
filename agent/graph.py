"""
agent/graph.py

Builds and compiles the ACE LangGraph agent.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent.nodes import call_api, interpret_result, parse_query
from agent.state import AgentState


def _route_after_parse(state: AgentState) -> str:
    """Conditional edge: skip to END if parsing failed."""
    return "call_api" if not state.get("error") else "interpret_result"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("parse_query",      parse_query)
    g.add_node("call_api",         call_api)
    g.add_node("interpret_result", interpret_result)

    g.add_edge(START, "parse_query")
    g.add_conditional_edges("parse_query", _route_after_parse)
    g.add_edge("call_api",         "interpret_result")
    g.add_edge("interpret_result", END)

    return g.compile()


# Singleton — import this to run queries
ace_agent = build_graph()
