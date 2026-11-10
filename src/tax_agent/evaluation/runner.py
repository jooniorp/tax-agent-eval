"""Run the eval set: ask the agent every question in the dataset, score the answers, print a report.

    uv run eval              # all items
    uv run eval --limit 2    # a cheap smoke run
    uv run eval --only card-01,rent-01
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from tax_agent import tracing
from tax_agent.agent import ask, build_agent, token_totals
from tax_agent.evaluation.scoring import Run, score_item
from tax_agent.llm import estimate_cost_usd, get_model
from tax_agent.mock_api.source import MockDataSource

# Tokens assumed per question for the cost estimate printed before a run. Measured on eval
# item 1 on 2026-11-07 (4,131-4,157 in, 911-955 out) and rounded up: later prompts are longer.
EST_TOKENS_PER_TURN = (4_500, 1_000)
RESULTS_DIR = Path("evals/results")


def load_dataset(path: str, only: list[str] | None = None, limit: int | None = None) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    if only:
        items = [i for i in items if i["id"] in only]
    return items[:limit] if limit else items


def run_item(item: dict, model, source=None) -> dict:
    """Ask the agent every turn of one item and score it. A crash counts as a failed item."""
    agent = build_agent(source or MockDataSource(), item["user"], model=model)
    messages, turn_ends, error = [], [], None
    started = time.perf_counter()
    try:
        with tracing.trace(f"eval-{item['id']}", user_id=item["user"]) as callbacks:
            for question in item["turns"]:
                messages = ask(agent, question, callbacks=callbacks, history=messages)
                turn_ends.append(len(messages))
    except Exception as e:  # e.g. the step limit; the eval goes on with the next item
        error = f"{type(e).__name__}: {e}"
    seconds = time.perf_counter() - started

    run = Run(messages, turn_ends)
    tokens_in, tokens_out = token_totals(messages)
    result = {
        "id": item["id"], "kind": item["kind"], "topic": item["topic"], "seconds": round(seconds, 2),
        "tokens_in": tokens_in, "tokens_out": tokens_out, "error": error,
        "tool_calls": [c.name for c in run.calls()], "answer": run.answer(),
    }
    if error is not None:
        return {**result, "answer_ok": False, "rules_ok": False, "passed": False, "answer_detail": error, "checks": []}
    score = score_item(item, run)
    return {
        **result, "answer_ok": score.answer_ok, "rules_ok": score.rules_ok, "passed": score.passed,
        "answer_detail": score.answer_detail, "checks": [asdict(c) for c in score.checks],
    }


def summarize(results: list[dict]) -> dict:
    return {
        "items": len(results),
        "answer_ok": sum(r["answer_ok"] for r in results),
        "rules_ok": sum(r["rules_ok"] for r in results),
        "passed": sum(r["passed"] for r in results),
        "tokens_in": sum(r["tokens_in"] for r in results),
        "tokens_out": sum(r["tokens_out"] for r in results),
        "seconds": round(sum(r["seconds"] for r in results), 1),
    }


def format_report(results: list[dict], model_name: str) -> str:
    mark = lambda ok: "O" if ok else "X"  # noqa: E731
    lines = [f"{'id':<10} {'정답':^4} {'규칙':^4} {'통과':^4} {'시간':>6}  툴 호출"]
    for r in results:
        lines.append(
            f"{r['id']:<10} {mark(r['answer_ok']):^4} {mark(r['rules_ok']):^4} {mark(r['passed']):^4} "
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
    lines.append(
        f"\n[요약] 정답 일치 {s['answer_ok']}/{s['items']} · 규칙 {s['rules_ok']}/{s['items']} · "
        f"**전체 통과 {s['passed']}/{s['items']}**"
    )
    lines.append(f"[비용·지연] 토큰 in/out {s['tokens_in']}/{s['tokens_out']} · 합계 {s['seconds']}s · 비용 {cost_text} (추정)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eval", description="Run the eval set against the agent.")
    parser.add_argument("--dataset", default="evals/dataset.jsonl")
    parser.add_argument("--model", help="model id; default: TAX_AGENT_MODEL or the built-in default")
    parser.add_argument("--only", help="comma-separated item ids")
    parser.add_argument("--limit", type=int, help="run only the first N items")
    parser.add_argument("--no-save", action="store_true", help="do not write evals/results/<time>.json")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")  # a piped Windows console may not encode every character

    items = load_dataset(args.dataset, args.only.split(",") if args.only else None, args.limit)
    model = get_model(args.model)
    questions = sum(len(i["turns"]) for i in items)
    estimate = estimate_cost_usd(model.model, questions * EST_TOKENS_PER_TURN[0], questions * EST_TOKENS_PER_TURN[1])
    print(f"[eval] {len(items)} items, {questions} questions | model={model.model} | dataset={args.dataset}")
    print("[eval] estimated cost of this run: " + (f"~${estimate:.4f}" if estimate is not None else "n/a (model not in price table)"))

    results = []
    for item in items:
        result = run_item(item, model)
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
            "model": model.model, "dataset": args.dataset, "summary": summarize(results), "results": results,
        }
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] saved {path}")
