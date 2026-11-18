"""The one place the LLM is constructed. The model name comes from config so it can
be swapped for the model comparison without touching the agent.

A name is "<model id>" (Anthropic) or "<provider>:<model id>", for example "openai:<model id>"."""
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

DEFAULT_MODEL = "claude-haiku-5-5"  # small model; see docs/decisions.md 2026-11-05
MAX_OUTPUT_TOKENS = 2048  # explicit cap: answers are short, and it bounds the cost per call


# USD per 1M tokens (input, output), standard tier, short prompts. Anthropic: the Claude API pricing
# table cached 2026-10-06 (up to 100K tokens). OpenAI: developers.openai.com/api/docs/pricing read on
# 2026-11-18 (up to 272K input tokens). Re-check the console before trusting a cost number.
PRICE_PER_MTOK_USD = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "gpt-6.2-sol": (2.00, 10.00),  # OpenAI, newest generation (6.2), middle tier between astra and luna
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimated cost of a run, or None if the model is not in the price table."""
    price = PRICE_PER_MTOK_USD.get(model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def get_model(name: str | None = None):
    """Build the chat model. Priority: argument, TAX_AGENT_MODEL env var, DEFAULT_MODEL.

    A bare id is an Anthropic model; "openai:<id>" is an OpenAI model (needs OPENAI_API_KEY).
    The returned object's `.model` is the bare id, which is what results and the price table use.
    No temperature/top_p/top_k is passed: Haiku 5.5 rejects non-default sampling values.
    """
    load_dotenv()  # reads .env (API keys, TAX_AGENT_MODEL); never overrides real env vars
    spec = name or os.getenv("TAX_AGENT_MODEL") or DEFAULT_MODEL
    provider, _, model_id = spec.rpartition(":")
    if provider == "openai":
        from langchain_openai import ChatOpenAI  # imported here so Anthropic-only runs do not load it

        return ChatOpenAI(model=model_id, max_completion_tokens=MAX_OUTPUT_TOKENS)
    if provider not in ("", "anthropic"):
        raise ValueError(f"unknown provider {provider!r} in model name {spec!r}; use a bare id or 'openai:<id>'")
    return ChatAnthropic(model=model_id, max_tokens=MAX_OUTPUT_TOKENS)
