"""Compare saved eval runs: the three agent variants against each other, and one variant across models.

    uv run eval-compare                    # every file in evals/results/
    uv run eval-compare --prefix 20261114  # only the runs saved on that day
    uv run eval-compare --group sweep      # one group

For each model and group it averages every item over its runs, then adds the items up, so the
numbers are "one run of the group". When the runs cover several models, a second part puts the models
side by side for each variant (the default model is the baseline). Runs saved before 2026-11-14 have no reasoning-token or
model-call fields; those columns show "-" for them.
"""
import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from langchain_core.messages import AIMessage

from tax_agent.evaluation.scoring import Run, amounts, answer_match, rejudge_texts
from tax_agent.llm import DEFAULT_MODEL, estimate_cost_usd

METRICS = ("tokens_in", "tokens_out", "reasoning_tokens", "model_calls", "seconds", "model_seconds")
GROUPS = ("basic", "exact", "sweep")


def load_rows(directory: str, prefix: str = "", model: str | None = None, since: str = "") -> list[dict]:
    """Every item result of every saved run, tagged with the run's model and variant.

    Files are named <date>-<time>.json, so `since="20261114-143000"` keeps the runs saved from then on.
    """
    rows = []
    for path in sorted(Path(directory).glob(f"{prefix}*.json")):
        if path.name < since:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if model and record.get("model") != model:
            continue
        for r in record["results"]:
            rows.append({**r, "model": record["model"], "variant": r.get("variant", record.get("variant", "calculator"))})
    return rows


def _tool_amount_again(check: dict, turn_answers: list[str]) -> dict:
    """`answer_uses_tool_amount` needs the calculator's result, which the file does not keep, but its stored
    detail lists the amounts ("tool said [375000]; answer has [...]"), so the answer can be checked again."""
    if not check["name"].startswith("answer_uses_tool_amount"):
        return check
    listed = re.match(r"tool said \[([\d, ]*)\]", check["detail"])
    if not listed:
        return check  # "not eligible", "0" or "no result": nothing to compare amounts with
    wanted = [int(x) for x in listed.group(1).split(",") if x.strip()]
    turn = re.search(r"@turn(\d+)", check["name"])
    stated = amounts(turn_answers[int(turn.group(1)) - 1] if turn else turn_answers[-1])
    missing = [d for d in wanted if d not in stated]
    return {**check, "ok": not missing, "detail": f"tool said {wanted}; answer has {sorted(stated)}"}


def rescore(rows: list[dict], dataset: str) -> list[dict]:
    """Judge every saved answer again with the current scoring rules.

    Result files keep the verdict of the day they were written. When a wording rule was too narrow
    (a correct answer scored as wrong), fixing the rule changes the verdicts of old runs as well, so the
    comparison re-judges the stored answers: the answer match and the checks that read only the answer
    text. The checks on the tool calls are not re-run (the trail is not saved), their stored result is
    kept. A row whose verdict changed gets `rescored=True`.
    """
    items = {}
    for line in Path(dataset).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            items[item["id"]] = item
    out = []
    for r in rows:
        item = items.get(r["id"])
        if item is None or r.get("error") or not r.get("answer"):
            out.append(r)
            continue
        stored = r.get("checks") or []
        if r.get("turn_answers"):  # runs saved from 2026-11-14 on keep the answer of every turn
            ok, new = rejudge_texts(item, r["variant"], r["turn_answers"])
            if len(new) == len(stored):
                stored = [{**s, "ok": n.ok, "detail": n.detail} if n is not None else s for s, n in zip(stored, new)]
            stored = [_tool_amount_again(c, r["turn_answers"]) for c in stored]
        else:
            ok, _ = answer_match(Run([AIMessage(r["answer"])], [1]), item["expected"])
        rules_ok = all(c["ok"] for c in stored) if stored else r.get("rules_ok", True)
        passed = ok and rules_ok
        out.append({**r, "answer_ok": ok, "rules_ok": rules_ok, "checks": stored, "passed": passed,
                    "rescored": passed != r["passed"]})
    return out


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def item_means(rows: list[dict], model: str, variant: str, group: str) -> dict:
    """Per item id: how many runs, how many passed, and the mean of every metric over the runs."""
    by_id = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["variant"] == variant and r.get("group", "basic") == group:
            by_id[r["id"]].append(r)
    means = {}
    for item_id, runs in by_id.items():
        entry = {"runs": len(runs), "passed": sum(r["passed"] for r in runs), "topic": runs[0]["topic"]}
        entry.update({m: _mean([r.get(m) for r in runs]) for m in METRICS})
        means[item_id] = entry
    return means


def total(items: dict, ids, metric: str):
    values = [items[i][metric] for i in ids]
    return None if any(v is None for v in values) else sum(values)


def change(a, b) -> str:
    """How much larger b is than a, as a signed percentage; "-" when either is missing."""
    return "-" if a in (None, 0) or b is None else f"{b / a - 1:+.0%}"


def _num(value, spec: str = ",.0f") -> str:
    return "-" if value is None else format(value, spec)


VARIANTS = ("calculator", "no_calculator", "combined")
LABELS = {"no_calculator": "control", "combined": "combined"}  # names used in the report lines
CHANGE_METRICS = (("in tokens", "tokens_in"), ("out tokens", "tokens_out"), ("reasoning", "reasoning_tokens"),
                  ("model calls", "model_calls"), ("seconds", "seconds"))


def _header(first: str) -> str:
    return (f"{first:<17}{'runs':>5}{'pass':>9}{'in tok':>9}{'out tok':>9}{'reasoning':>10}{'calls':>7}"
            f"{'seconds':>9}{'model s':>9}{'cost':>9}")


def _table_row(label: str, model: str, items: dict) -> str:
    ids = sorted(items)
    runs = [items[i]["runs"] for i in ids]
    passed, attempts = sum(items[i]["passed"] for i in ids), sum(runs)
    t = {m: total(items, ids, m) for m in METRICS}
    cost = estimate_cost_usd(model, t["tokens_in"], t["tokens_out"])
    return (
        f"{label:<17}{min(runs):>3}-{max(runs):<1}{passed:>5}/{attempts:<3}{_num(t['tokens_in']):>9}{_num(t['tokens_out']):>9}"
        f"{_num(t['reasoning_tokens']):>10}{_num(t['model_calls'], '.1f'):>7}{_num(t['seconds'], '.1f'):>9}"
        f"{_num(t['model_seconds'], '.1f'):>9}{('$' + format(cost, '.4f')) if cost is not None else '-':>9}"
    )


def _versus(label: str, base_label: str, base_model: str, base: dict, model: str, items: dict) -> list[str]:
    """How one entry differs from the baseline over the items both have: one summary line, then per topic."""
    ids = sorted(set(base) & set(items))
    t_base = {m: total(base, ids, m) for m in METRICS}
    t_other = {m: total(items, ids, m) for m in METRICS}
    cost = [estimate_cost_usd(m, t["tokens_in"], t["tokens_out"]) for m, t in ((base_model, t_base), (model, t_other))]
    lines = [f"{label} vs {base_label}: " + " · ".join(f"{text} {change(t_base[m], t_other[m])}" for text, m in CHANGE_METRICS)
             + f" · cost {change(*cost)}"]
    by_topic = defaultdict(list)
    for i in ids:
        by_topic[base[i]["topic"]].append(i)
    if len(by_topic) > 1:
        lines.append(f"per topic, {label} vs {base_label} (out tokens / seconds):")
        for topic, topic_ids in sorted(by_topic.items()):
            lines.append(
                f"  {topic:<8}({len(topic_ids):>2} items)  "
                f"{change(total(base, topic_ids, 'tokens_out'), total(items, topic_ids, 'tokens_out')):>6} / "
                f"{change(total(base, topic_ids, 'seconds'), total(items, topic_ids, 'seconds')):>6}"
            )
    return lines


def _per_item(base_label: str, base: dict, others: list[dict]) -> list[str]:
    """Small groups show every item: out tokens / seconds of the baseline, then each other entry with its change."""
    ids = sorted(set(base).intersection(*[set(items) for items in others]))
    lines = [f"per item (out tokens / seconds; % = change against {base_label}):"]
    for i in ids:
        cells = [f"{_num(base[i]['tokens_out']):>6} / {_num(base[i]['seconds'], '.1f'):>5}s"]
        for items in others:
            cells.append(f"{_num(items[i]['tokens_out']):>6} ({change(base[i]['tokens_out'], items[i]['tokens_out'])}) / "
                         f"{_num(items[i]['seconds'], '.1f'):>5}s ({change(base[i]['seconds'], items[i]['seconds'])})")
        lines.append(f"  {i:<9}" + " | ".join(cells))
    return lines


def group_report(model: str, group: str, by_variant: dict[str, dict]) -> list[str]:
    """One model and group: a table of every variant present, then how each other variant differs from "calculator"."""
    variants = [(name, by_variant[name]) for name in VARIANTS if by_variant.get(name)]
    ids_all = sorted(set().union(*[set(items) for _, items in variants]))
    lines = [f"\n== {model} · group {group} ({len(ids_all)} items)", _header("variant")]
    lines += [_table_row(name, model, items) for name, items in variants]
    base = by_variant.get("calculator")
    others = [(name, items) for name, items in variants if name != "calculator"]
    if not (base and others):
        return lines
    for name, items in others:
        lines += _versus(LABELS[name], "calculator", model, base, model, items)
    if group in ("exact", "sweep"):
        lines += _per_item("calculator", base, [items for _, items in others])
    return lines


def model_report(rows: list[dict], group: str | None = None) -> list[str]:
    """The same agent variant under different models, side by side. The first model is the baseline:
    the default model when it is there, otherwise the first by name."""
    models = sorted({r["model"] for r in rows}, key=lambda m: (m != DEFAULT_MODEL, m))
    lines = []
    for g in GROUPS:
        if group and g != group:
            continue
        for variant in VARIANTS:
            present = [(m, item_means(rows, m, variant, g)) for m in models]
            present = [(m, items) for m, items in present if items]
            if len(present) < 2:
                continue
            ids_all = sorted(set().union(*[set(items) for _, items in present]))
            lines += [f"\n== models · group {g} · variant {variant} ({len(ids_all)} items)", _header("model")]
            lines += [_table_row(m, m, items) for m, items in present]
            (base_model, base), rest = present[0], present[1:]
            for model, items in rest:
                lines += _versus(model, base_model, base_model, base, model, items)
            if g in ("exact", "sweep"):
                lines += _per_item(base_model, base, [items for _, items in rest])
    return lines


def report(rows: list[dict], group: str | None = None) -> str:
    lines = []
    for model in sorted({r["model"] for r in rows}):
        for g in GROUPS:
            if group and g != group:
                continue
            by_variant = {v: item_means(rows, model, v, g) for v in VARIANTS}
            if any(by_variant.values()):
                lines += group_report(model, g, by_variant)
    lines += model_report(rows, group)
    return "\n".join(lines) if lines else "no saved runs found"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="eval-compare", description="Compare saved eval runs.")
    parser.add_argument("--dir", default="evals/results")
    parser.add_argument("--prefix", default="", help="only result files whose name starts with this, e.g. 20261114")
    parser.add_argument("--since", default="", help="only result files from this name on, e.g. 20261114-143000")
    parser.add_argument("--model")
    parser.add_argument("--group", choices=GROUPS)
    parser.add_argument("--dataset", default="evals/dataset.jsonl", help="expected answers used to re-judge saved answers")
    parser.add_argument("--raw", action="store_true", help="keep the verdicts stored in the files (no re-judging)")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")
    rows = load_rows(args.dir, args.prefix, args.model, args.since)
    if not args.raw:
        rows = rescore(rows, args.dataset)
        changed = [r for r in rows if r.get("rescored")]
        if changed:
            print(f"re-judged with the current answer rules: {len(changed)} of {len(rows)} item runs changed verdict "
                  f"({', '.join(sorted({r['id'] for r in changed}))}); --raw shows the stored verdicts")
    print(report(rows, args.group))
