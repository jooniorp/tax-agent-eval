"""Command line entry: ask the agent one question about a fictional taxpayer.

    uv run tax-agent "올해 카드 소득공제 얼마나 받을 수 있어?" --user user_a
"""
import argparse
import json
import sys
import time

from langchain_core.messages import AIMessage, BaseMessage

from tax_agent.agent import ask, build_agent
from tax_agent.llm import estimate_cost_usd, get_model
from tax_agent.mock_api.source import MockDataSource


def token_totals(messages: list[BaseMessage]) -> tuple[int, int]:
    """Total (input, output) tokens over every model call in the run."""
    usages = [m.usage_metadata for m in messages if isinstance(m, AIMessage) and m.usage_metadata]
    return sum(u["input_tokens"] for u in usages), sum(u["output_tokens"] for u in usages)


def tool_trace(messages: list[BaseMessage]) -> list[str]:
    """One line per tool call the model made, in order."""
    return [
        f"{call['name']} {json.dumps(call['args'], ensure_ascii=False)}"
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="tax-agent", description="Ask the mock tax assistant a question.")
    parser.add_argument("question")
    parser.add_argument("--user", default="user_a", help="fictional taxpayer id (default: user_a)")
    parser.add_argument("--model", help="model id; default: TAX_AGENT_MODEL or the built-in default")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")  # a piped Windows console may not encode every character

    model = get_model(args.model)
    agent = build_agent(MockDataSource(), args.user, model=model)
    started = time.perf_counter()
    messages = ask(agent, args.question)
    elapsed = time.perf_counter() - started

    in_tokens, out_tokens = token_totals(messages)
    cost = estimate_cost_usd(model.model, in_tokens, out_tokens)
    cost_text = f"~${cost:.4f} (estimate)" if cost is not None else "n/a (model not in price table)"
    print("[tool calls]")
    for line in tool_trace(messages):
        print(f"  {line}")
    print(f"\n[answer]\n{messages[-1].text}")
    print(f"\n[stats] model={model.model} time={elapsed:.1f}s tokens in/out={in_tokens}/{out_tokens} cost={cost_text}")
