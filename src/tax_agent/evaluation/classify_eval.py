"""Measure a transaction classifier on synthetic users: how often the category is right, and what a wrong
category does to the deduction.

    uv run sim-eval                         # 50 users, rule-based classifier
    uv run sim-eval --users 200 --obscure 0.3
    uv run sim-eval --classifier llm        # a chat model labels each distinct merchant once (costs a few cents at most)

For every simulated user the rows are stripped of their category, the classifier labels them, and both
the true and the predicted rows go through the same calculators. No model is called and nothing costs
money. The error in won is the difference of the deduction (card), or tax credit (rent, medical).
"""
import argparse
import re
import statistics
import sys
from collections import Counter, defaultdict

from tax_agent.classify import Classifier, LLMClassifier, classify_rules
from tax_agent.llm import estimate_cost_usd, get_model
from tax_agent.mock_api.deduction import calc_card_deduction
from tax_agent.mock_api.medical import calc_medical_credit
from tax_agent.mock_api.rent import calc_rent_credit
from tax_agent.simulation import SimUser, simulate_user
from tax_agent.transactions import Transaction, card_spending, medical_expenses, rent_payments

YEAR = 2025
TOPICS = ("card", "rent", "medical")


def deductions(user: SimUser, rows: list[Transaction], year: int = YEAR) -> dict[str, int]:
    """Card deduction, rent credit and medical credit of the user computed from these rows."""
    salary, homeless = user.profile.gross_salary, user.profile.homeless_household_head
    return {
        "card": calc_card_deduction(salary, card_spending(rows, year, salary)).deduction,
        "rent": calc_rent_credit(salary, homeless, rent_payments(rows, year)).deduction,
        "medical": calc_medical_credit(salary, medical_expenses(rows, year)).deduction,
    }


def evaluate(classify: Classifier, users: int = 50, obscure_rate: float = 0.15) -> dict:
    """Run the classifier over `users` synthetic users (seeds 0..users-1) and collect the numbers."""
    rows_total = right = 0
    recall = defaultdict(lambda: [0, 0])  # true category -> [right, all]
    confused = Counter()  # (true, predicted) for the wrong ones
    errors = {t: [] for t in TOPICS}
    simulated = [simulate_user(seed, YEAR, obscure_rate) for seed in range(users)]
    everything = classify([t for user in simulated for t in user.unlabeled()])  # one call: a model-based classifier asks once
    assert len(everything) == sum(len(user.transactions) for user in simulated)
    start = 0
    for user in simulated:
        predicted = everything[start:start + len(user.transactions)]
        start += len(user.transactions)
        for truth, guess in zip(user.transactions, predicted):
            rows_total += 1
            recall[truth.category][1] += 1
            if guess.category == truth.category:
                right += 1
                recall[truth.category][0] += 1
            else:
                confused[(truth.category, guess.category)] += 1
        true_d, guess_d = deductions(user, user.transactions), deductions(user, predicted)
        for topic in TOPICS:
            errors[topic].append(guess_d[topic] - true_d[topic])
    return {"users": users, "obscure_rate": obscure_rate, "rows": rows_total, "right": right, "recall": dict(recall),
            "confused": confused, "errors": errors}


def format_report(result: dict, name: str) -> str:
    lines = [f"[분류기: {name}] 합성 사용자 {result['users']}명 · 거래 {result['rows']}건 · 이름이 불분명한 가맹점 비율 {result['obscure_rate']:.0%}",
             f"분류 정확도: {result['right'] / result['rows']:.1%} ({result['right']}/{result['rows']})",
             "분류별 재현율 (실제 그 분류인 거래 중 맞힌 비율):"]
    for category, (ok, n) in sorted(result["recall"].items()):
        lines.append(f"  {category:<9}{ok / n:>7.1%}  ({ok}/{n})")
    if result["confused"]:
        worst = ", ".join(f"{t}→{p} {c}건" for (t, p), c in result["confused"].most_common(5))
        lines.append(f"가장 많이 틀린 쌍: {worst}")
    lines.append("공제액 오차 (예측 - 정답, 원) · 사용자별:")
    for topic in TOPICS:
        errs = result["errors"][topic]
        exact = sum(e == 0 for e in errs) / len(errs)
        lines.append(f"  {topic:<8} 정확히 맞은 사용자 {exact:>5.0%} · 평균 절대 오차 {statistics.mean(abs(e) for e in errs):>10,.0f} · "
                     f"평균 {statistics.mean(errs):>+11,.0f} · 최악 {max(errs, key=abs):>+12,}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sim-eval", description="Measure a transaction classifier on synthetic users.")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--obscure", type=float, default=0.15, help="share of special-category purchases whose merchant name says nothing")
    parser.add_argument("--classifier", choices=("rules", "llm"), default="rules")
    parser.add_argument("--model", help="model id for --classifier llm; default: TAX_AGENT_MODEL or the built-in default")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")
    if args.classifier == "rules":
        print(format_report(evaluate(classify_rules, args.users, args.obscure), "rules"))
        return
    model = get_model(args.model)
    kinds = len({(re.sub(r"\d+", "", t.merchant).strip(), t.method) for seed in range(args.users)
                 for t in simulate_user(seed, YEAR, args.obscure).transactions})
    estimate = estimate_cost_usd(model.model, 400 + 30 * kinds, 12 * kinds)  # prompt text plus a short answer per distinct merchant
    print(f"[sim-eval] {args.users} users, {kinds} distinct merchants | model={model.model} | estimated cost: "
          + (f"~${estimate:.4f}" if estimate is not None else "n/a (model not in price table)"))
    classifier = LLMClassifier(model)
    result = evaluate(classifier, args.users, args.obscure)
    cost = estimate_cost_usd(model.model, classifier.tokens_in, classifier.tokens_out)
    print(format_report(result, f"llm:{model.model}"))
    print(f"토큰 in/out {classifier.tokens_in}/{classifier.tokens_out}" + (f" · 비용 ~${cost:.4f} (추정)" if cost is not None else ""))
