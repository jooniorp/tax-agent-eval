"""Run the eval set: ask the agent every question in the dataset, score the answers, print a report.

    uv run eval                       # all items, the default agent (one compute_deduction tool)
    uv run eval --variant calculator  # the step-by-step agent: profile, data, then calc_deduction
    uv run eval --no-calculator       # the control: the model calculates itself from the rules as text
    uv run eval --limit 2        # a cheap smoke run
    uv run eval --only card-01,rent-01
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage

from tax_agent import tracing
from tax_agent.agent import DEFAULT_VARIANT, VARIANTS, ask, build_agent, reasoning_tokens, token_totals
from tax_agent.evaluation.scoring import Run, score_item
from tax_agent.evaluation.timing import CallTimer
from tax_agent.llm import estimate_cost_usd, get_model
from tax_agent.mock_api.source import MockDataSource

# Tokens (in, out) assumed per question for the cost estimate printed before a run, per agent variant.
# Measured on the three-variant batch of 2026-11-14 (34 questions per pass of the 32 items), rounded up:
# calculator 6.9k / 0.86k, no_calculator 6.2k / 1.12k, combined 2.4k / 0.67k. An item's input grows with
# every tool round, so the step-by-step agents cost more input. Re-measure when the prompt or the tools change.
EST_TOKENS_PER_TURN = {"calculator": (7_000, 900), "no_calculator": (6_500, 1_150), "combined": (2_500, 700)}
RESULTS_DIR = Path("evals/results")


def load_dataset(
    path: str, only: list[str] | None = None, limit: int | None = None, group: str | None = None
) -> list[dict]:
    """Items without a "group" belong to "basic"; the harder amounts live in "exact"."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    if only:
        items = [i for i in items if i["id"] in only]
    if group:
        items = [i for i in items if i.get("group", "basic") == group]
    return items[:limit] if limit else items


def run_item(item: dict, model, source=None, variant: str = DEFAULT_VARIANT) -> dict:
    """Ask the agent every turn of one item and score it. A crash counts as a failed item."""
    agent = build_agent(source or MockDataSource(), item["user"], model=model, variant=variant)
    messages, turn_ends, error = [], [], None
    timer = CallTimer()
    started = time.perf_counter()
    try:
        with tracing.trace(f"eval-{item['id']}", user_id=item["user"]) as callbacks:
            for question in item["turns"]:
                messages = ask(agent, question, callbacks=[*callbacks, timer], history=messages)
                turn_ends.append(len(messages))
    except Exception as e:  # e.g. the step limit; the eval goes on with the next item
        error = f"{type(e).__name__}: {e}"
    seconds = time.perf_counter() - started

    run = Run(messages, turn_ends)
    tokens_in, tokens_out = token_totals(messages)
    result = {
        "id": item["id"], "group": item.get("group", "basic"), "kind": item["kind"], "topic": item["topic"],
        "seconds": round(seconds, 2),
        "variant": variant,
        "tokens_in": tokens_in, "tokens_out": tokens_out, "error": error,
        "reasoning_tokens": reasoning_tokens(messages),  # part of tokens_out spent thinking
        "model_calls": sum(isinstance(m, AIMessage) for m in messages),
        "model_seconds": round(sum(timer.model_seconds), 2),  # of `seconds`: the model calls ...
        "tool_seconds": round(timer.tool_seconds, 3),  # ... and the tools; the rest is graph overhead
        "tool_calls": [c.name for c in run.calls()], "answer": run.answer(),
        "turn_answers": [run.answer(t) for t in range(1, len(turn_ends) + 1)],  # every turn, for debugging
    }
    if error is not None:
        return {**result, "answer_ok": False, "rules_ok": False, "passed": False, "answer_detail": error, "checks": []}
    score = score_item(item, run, variant)
    return {
        **result, "answer_ok": score.answer_ok, "rules_ok": score.rules_ok, "passed": score.passed,
        "answer_detail": score.answer_detail, "checks": [asdict(c) for c in score.checks],
    }


def summarize(results: list[dict]) -> dict:
    rated = [r for r in results if r["checks"] or r["error"]]  # items that have at least one rule check
    return {
        "items": len(results),
        "answer_ok": sum(r["answer_ok"] for r in results),
        "rules_n": len(rated),
        "rules_ok": sum(r["rules_ok"] for r in rated),
        "passed": sum(r["passed"] for r in results),
        "tokens_in": sum(r["tokens_in"] for r in results),
        "tokens_out": sum(r["tokens_out"] for r in results),
        "reasoning_tokens": sum(r.get("reasoning_tokens", 0) for r in results),
        "model_calls": sum(r.get("model_calls", 0) for r in results),
        "model_seconds": round(sum(r.get("model_seconds", 0) for r in results), 1),
        "seconds": round(sum(r["seconds"] for r in results), 1),
    }


def format_report(results: list[dict], model_name: str) -> str:
    mark = lambda ok: "O" if ok else "X"  # noqa: E731
    lines = [f"{'id':<10} {'정답':^4} {'규칙':^4} {'통과':^4} {'시간':>6}  툴 호출"]
    for r in results:
        rules = mark(r["rules_ok"]) if (r["checks"] or r["error"]) else "-"  # "-": no rule check applies
        lines.append(
            f"{r['id']:<10} {mark(r['answer_ok']):^4} {rules:^4} {mark(r['passed']):^4} "
            f"{r['seconds']:>5.1f}s  {' > '.join(r['tool_calls']) or '-'}"
        )
    failed = [r for r in results if not r["passed"]]
    if failed:
        lines.append("\n[실패 상세]")
    for r in failed:
        lines.append(f"- {r['id']}")
        if not r["answer_ok"]:
            lines.append(f"    정답 불일치: {r['answer_detail']}")
        lines += [f"    규칙 실패 {c['name']}: {c['detail']}" for c in r["checks"] if not c["ok"]]
        lines.append(f"    답변: {r['answer'][:200]!r}")
    s = summarize(results)
    cost = estimate_cost_usd(model_name, s["tokens_in"], s["tokens_out"])
    cost_text = f"~${cost:.4f}" if cost is not None else "n/a"
    variant = results[0]["variant"] if results else "calculator"
    label = {"no_calculator": "대조군(계산기 없음)", "combined": "에이전트(통합 툴 1개)"}.get(variant, "에이전트(계산기 툴)")
    lines.append(
        f"\n[요약 · {label}] 정답 일치 {s['answer_ok']}/{s['items']} · 규칙 {s['rules_ok']}/{s['rules_n']} · "
        f"**전체 통과 {s['passed']}/{s['items']}**"
    )
    thinking = ""
    if s["model_calls"]:  # runs saved before 2026-11-14 do not have these fields
        share = s["reasoning_tokens"] / s["tokens_out"] if s["tokens_out"] else 0
        thinking = (f" · 추론 토큰 {s['reasoning_tokens']}(출력의 {share:.0%}) · 모델 호출 {s['model_calls']}회"
                    f" · 모델 시간 {s['model_seconds']}s")
    lines.append(
        f"[비용·지연] 토큰 in/out {s['tokens_in']}/{s['tokens_out']} · 합계 {s['seconds']}s · 비용 {cost_text} (추정)" + thinking
    )
    groups = sorted({r.get("group", "basic") for r in results})
    if len(groups) > 1:  # report each group on its own: the harder "exact" items should not hide in an average
        parts = []
        for g in groups:
            members = [r for r in results if r.get("group", "basic") == g]
            parts.append(f"{g} 정답 {sum(r['answer_ok'] for r in members)}/{len(members)} · "
                         f"통과 {sum(r['passed'] for r in members)}/{len(members)}")
        lines.append("[그룹별] " + " | ".join(parts))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eval", description="Run the eval set against the agent.")
    parser.add_argument("--dataset", default="evals/dataset.jsonl")
    parser.add_argument("--model", help="model id; default: TAX_AGENT_MODEL or the built-in default")
    parser.add_argument("--only", help="comma-separated item ids")
    parser.add_argument(
        "--group", help='run only one group: "basic" (default items), "exact" (odd amounts) or "sweep" (1-6 card categories)'
    )
    parser.add_argument("--limit", type=int, help="run only the first N items")
    parser.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT, help=f"which agent to run (default: {DEFAULT_VARIANT})")
    parser.add_argument("--no-calculator", action="store_true", help="shorthand for --variant no_calculator")
    parser.add_argument("--no-save", action="store_true", help="do not write evals/results/<time>.json")
    parser.add_argument("--min-pass-rate", type=float, metavar="RATE",
                        help="regression gate: exit with status 1 when the share of passed items is below RATE (0-1)")
    parser.add_argument("--max-estimated-cost", type=float, metavar="USD",
                        help="cost guard: exit with status 2 before any model call when the estimate is above USD or unknown")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")  # a piped Windows console may not encode every character

    items = load_dataset(args.dataset, args.only.split(",") if args.only else None, args.limit, args.group)
    model = get_model(args.model)
    questions = sum(len(i["turns"]) for i in items)
    variant = "no_calculator" if args.no_calculator else args.variant
    tokens_in, tokens_out = EST_TOKENS_PER_TURN[variant]
    estimate = estimate_cost_usd(model.model, questions * tokens_in, questions * tokens_out)
    print(f"[eval] {len(items)} items, {questions} questions | model={model.model} | variant={variant} | dataset={args.dataset}")
    print("[eval] estimated cost of this run: " + (f"~${estimate:.4f}" if estimate is not None else "n/a (model not in price table)"))
    if args.max_estimated_cost is not None and (estimate is None or estimate > args.max_estimated_cost):
        print(f"[eval] refused: the estimate is above the limit of ${args.max_estimated_cost:.4f} or unknown; no model was called")
        sys.exit(2)

    results = []
    for item in items:
        result = run_item(item, model, variant=variant)
        results.append(result)
        print(f"  {item['id']:<10} {'PASS' if result['passed'] else 'FAIL'}  {result['seconds']}s", flush=True)
    tracing.flush()  # eval traces are sent to Langfuse when tracing is on

    print("\n" + format_report(results, model.model))
    if not args.no_save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        path = RESULTS_DIR / f"{now:%Y%m%d-%H%M%S}.json"
        record = {
            "created": now.isoformat(), "argv": ["eval", *(argv if argv is not None else sys.argv[1:])],
            "model": model.model, "variant": variant, "dataset": args.dataset,
            "summary": summarize(results), "results": results,
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] saved {path}")

    passed = sum(r["passed"] for r in results)
    if args.min_pass_rate is not None and results and passed / len(results) < args.min_pass_rate:
        print(f"[eval] regression: {passed}/{len(results)} passed, below the required rate {args.min_pass_rate:.0%}")
        sys.exit(1)
