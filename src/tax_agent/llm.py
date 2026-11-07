"""The one place the LLM is constructed. The model name comes from config so it can
be swapped for the model comparison without touching the agent."""
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

DEFAULT_MODEL = "claude-haiku-5-5"  # small model; see docs/decisions.md 2026-11-05
MAX_OUTPUT_TOKENS = 2048  # explicit cap: answers are short, and it bounds the cost per call


# USD per 1M tokens (input, output) for prompts up to 100K tokens. Source: the Claude API
# pricing table cached 2026-10-06. Re-check on the console before trusting a cost number.
PRICE_PER_MTOK_USD = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimated cost of a run, or None if the model is not in the price table."""
    price = PRICE_PER_MTOK_USD.get(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def get_model(name: str | None = None) -> ChatAnthropic:
    """Build the chat model. Priority: argument, TAX_AGENT_MODEL env var, DEFAULT_MODEL.

    No temperature/top_p/top_k is passed: Haiku 5.5 rejects non-default sampling values.
    """
    load_dotenv()  # reads .env (ANTHROPIC_API_KEY, TAX_AGENT_MODEL); never overrides real env vars
    return ChatAnthropic(
        model=name or os.getenv("TAX_AGENT_MODEL") or DEFAULT_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
