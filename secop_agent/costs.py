"""USD cost accounting for model calls, independent of LangSmith's own pricing table.

LangSmith only reports cost for models it has a price for. This module reads the token
usage LangChain attaches to every AI message and applies Anthropic's public rates::

    tracker = CostTracker()
    agent.invoke(inputs, config={"callbacks": [tracker]})
    print(tracker.summary())
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


@dataclass(frozen=True)
class Rate:
    """USD per million tokens."""

    input: float
    output: float
    cache_write: float
    cache_read: float


# Anthropic API list prices, October 2026. Haiku 5.5 rates apply to prompts up to 100K tokens.
RATES: dict[str, Rate] = {
    "claude-haiku-5-5": Rate(0.10, 0.50, 0.125, 0.01),
    "claude-sonnet-5-5": Rate(2.00, 10.00, 2.50, 0.20),
    "claude-opus-5-5": Rate(4.00, 20.00, 5.00, 0.20),
    "claude-sonnet-5": Rate(2.00, 10.00, 2.50, 0.20),
    "claude-opus-5": Rate(5.00, 25.00, 6.25, 0.50),
    "claude-sonnet-4-6": Rate(3.00, 15.00, 3.75, 0.30),
    "claude-haiku-4-5": Rate(1.00, 5.00, 1.25, 0.10),
}


@dataclass(frozen=True)
class Usage:
    """Token usage of one model call. `input_tokens` includes cached tokens, as in LangChain."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0


def model_name(model: str | None) -> str:
    """Strip the provider prefix: 'anthropic:claude-haiku-5-5' -> 'claude-haiku-5-5'."""
    return str(model or "").split(":")[-1]


def cost_usd(model: str | None, usage: Usage) -> float | None:
    """Cost of one call, or None when the model has no known rate."""
    rate = RATES.get(model_name(model))
    if rate is None:
        return None
    uncached = max(0, usage.input_tokens - usage.cache_write_tokens - usage.cache_read_tokens)
    total = (
        uncached * rate.input
        + usage.output_tokens * rate.output
        + usage.cache_write_tokens * rate.cache_write
        + usage.cache_read_tokens * rate.cache_read
    )
    return total / 1_000_000


def usage_from_message(message: Any) -> Usage:
    """Read token usage from a LangChain AIMessage (zeros when it carries none)."""
    metadata = getattr(message, "usage_metadata", None) or {}
    details = metadata.get("input_token_details") or {}
    return Usage(
        input_tokens=int(metadata.get("input_tokens") or 0),
        output_tokens=int(metadata.get("output_tokens") or 0),
        cache_write_tokens=int(details.get("cache_creation") or 0),
        cache_read_tokens=int(details.get("cache_read") or 0),
    )


@dataclass(frozen=True)
class CallRecord:
    model: str
    usage: Usage
    cost_usd: float | None


class CostTracker(BaseCallbackHandler):
    """Callback that accumulates tokens and cost for every model call in a run,
    including calls made inside tools (for example, structured extraction)."""

    def __init__(self, default_model: str = "") -> None:
        super().__init__()
        self.default_model = default_model
        self._calls: list[CallRecord] = []
        self._lock = threading.Lock()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                if message is None:
                    continue
                metadata = getattr(message, "response_metadata", None) or {}
                model = metadata.get("model_name") or metadata.get("model") or self.default_model
                usage = usage_from_message(message)
                record = CallRecord(model_name(model), usage, cost_usd(model, usage))
                with self._lock:
                    self._calls.append(record)

    @property
    def calls(self) -> list[CallRecord]:
        with self._lock:
            return list(self._calls)

    def summary(self) -> dict[str, Any]:
        calls = self.calls
        known_costs = [c.cost_usd for c in calls if c.cost_usd is not None]
        return {
            "model_calls": len(calls),
            "input_tokens": sum(c.usage.input_tokens for c in calls),
            "output_tokens": sum(c.usage.output_tokens for c in calls),
            "cache_read_tokens": sum(c.usage.cache_read_tokens for c in calls),
            "cost_usd": round(sum(known_costs), 6),
            # False when a model without a known rate was called, so cost_usd is a lower bound.
            "cost_complete": len(known_costs) == len(calls),
        }
