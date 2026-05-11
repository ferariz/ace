"""
agent/run.py

Quick CLI to run the ACE agent.
Usage:
    python -m agent.run "What would precipitation look like over southern Brazil?"
"""
from __future__ import annotations

import logging
import sys

logging.basicConfig(level=logging.WARNING)  # quiet for CLI use

from agent.graph import ace_agent


def run(query: str) -> str:
    result = ace_agent.invoke({"query": query})
    return result.get("answer") or result.get("error") or "No response."


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or \
        "What would precipitation look like over southern Brazil in January 2020?"
    print("\n" + "─" * 60)
    print(f"Query: {query}")
    print("─" * 60)
    print(run(query))
    print("─" * 60 + "\n")
