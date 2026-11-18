"""End to end on synthetic users: transactions -> classifier -> data source -> agent -> answer, against the truth.

    uv run sim-agent                      # 20 users x 3 questions, keyword classifier, the default agent
    uv run sim-agent --classifier llm --users 50

Every question has three numbers. T is the deduction the user truly has (computed from the true
categories). C is what the calculator returns for the rows the classifier labeled: the number the agent's
tool will hand over. A is the amount in the agent's answer. So the errors separate by layer:
  - classification error: C differs from T (the tool is right about wrong data);
  - agent error: A differs from C (the agent mis-relays what its tool said);
  - end to end: A differs from T, whatever the cause.
The agent is asked about a user and nothing else: salary and household status are in the profile, so it
should never need to ask a question back.
"""
import argparse
import sys
import time
from dataclasses import dataclass

from tax_agent.agent import DEFAULT_VARIANT, ask, build_agent, token_totals
from tax_agent.classify import Classifier, LLMClassifier, classify_rules
from tax_agent.evaluation.classify_eval import TOPICS, YEAR, deductions
from tax_agent.evaluation.runner import EST_TOKENS_PER_TURN
from tax_agent.evaluation.scoring import amounts, says_ineligible, says_zero
from tax_agent.llm import estimate_cost_usd, get_model
from tax_agent.simulation import SimUser, simulate_user
from tax_agent.transactions import TransactionDataSource

QUESTIONS = {
    "card": "올해(2025년) 카드 소득공제 얼마나 받을 수 있어?",
    "rent": "올해 월세 세액공제는 얼마야?",
    "medical": "올해 의료비 세액공제는 얼마나 받아?",
}


def states(answer: str, amount: int) -> bool:
    """Does the answer say this amount (or, for 0, say that there is no deduction)?"""
    if amount > 0:
        return amount in amounts(answer)
    return says_zero(answer) or says_ineligible(answer)


@dataclass
class Case:
    seed: int
    topic: str
    truth: int  # T
    tool: int  # C
    answer: str
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    error: str | None = None

    @property
    def end_to_end(self) -> bool:  # A matches T
        return self.error is None and states(self.answer, self.truth)

    @property
    def faithful(self) -> bool:  # A matches C
        return self.error is None and states(self.answer, self.tool)


def run_cases(model, users: list[SimUser], labeled: list[list], topics=TOPICS) -> list[Case]:
    """Ask the default agent every topic of every user. `labeled[i]` are the classified rows of users[i]."""
    cases = []
    for user, rows in zip(users, labeled):
        truth, tool = deductions(user, user.transactions), deductions(user, rows)
        agent = build_agent(TransactionDataSource(user.profile, rows), user.profile.user_id, model=model)
        for topic in topics:
            started = time.perf_counter()
            try:
                messages = ask(agent, QUESTIONS[topic])
                answer, error = messages[-1].text, None
                tokens_in, tokens_out = token_totals(messages)
            except Exception as e:  # e.g. the step limit; the run goes on with the next question
                answer, error, tokens_in, tokens_out = "", f"{type(e).__name__}: {e}", 0, 0
            cases.append(Case(user.seed, topic, truth[topic], tool[topic], answer, tokens_in, tokens_out,
                              time.perf_counter() - started, error))
    return cases


def format_report(cases: list[Case], model_name: str, classifier_name: str, obscure_rate: float) -> str:
    n = len(cases)
    wrong_data = [c for c in cases if c.tool != c.truth]
    follows_tool = [c for c in wrong_data if c.faithful]
    cost = estimate_cost_usd(model_name, sum(c.tokens_in for c in cases), sum(c.tokens_out for c in cases))
    lines = [
        f"[sim-agent] 사용자 {len({c.seed for c in cases})}명 · 질문 {n}개 · 에이전트 {model_name}({DEFAULT_VARIANT}) · 분류기 {classifier_name} · 이름 불분명 {obscure_rate:.0%}",
        f"종단 정확(답한 금액 = 정답 공제액):      {sum(c.end_to_end for c in cases):>3}/{n}",
        f"툴 결과에 충실(답한 금액 = 툴이 준 값):   {sum(c.faithful for c in cases):>3}/{n}",
        f"분류 오류로 툴 값이 정답과 다른 질문:     {len(wrong_data):>3}/{n}  (그중 에이전트가 툴 값을 그대로 따름 {len(follows_tool)})",
        "주제별 종단 정확 / 툴에 충실 / 툴 값이 틀린 질문:",
    ]
    for topic in dict.fromkeys(c.topic for c in cases):
        part = [c for c in cases if c.topic == topic]
        lines.append(f"  {topic:<8}{sum(c.end_to_end for c in part):>3}/{len(part)}  {sum(c.faithful for c in part):>3}/{len(part)}  {sum(c.tool != c.truth for c in part):>3}/{len(part)}")
    unfaithful = [c for c in cases if not c.faithful][:5]
    if unfaithful:
        lines.append("툴 값을 못 따른 질문 (최대 5개):")
        for c in unfaithful:
            lines.append(f"  user {c.seed} {c.topic}: 툴 {c.tool:,} · 정답 {c.truth:,} · 답변 {c.error or c.answer[:90].replace(chr(10), ' ')!r}")
    cost_text = f" · 비용 ~${cost:.4f} (추정)" if cost is not None else ""
    lines.append(f"[비용·지연] 토큰 in/out {sum(c.tokens_in for c in cases)}/{sum(c.tokens_out for c in cases)} · 합계 {sum(c.seconds for c in cases):.1f}s{cost_text}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sim-agent", description="End to end on synthetic users.")
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--obscure", type=float, default=0.15)
    parser.add_argument("--classifier", choices=("rules", "llm"), default="rules")
    parser.add_argument("--topics", default=",".join(TOPICS), help="comma-separated: card,rent,medical")
    parser.add_argument("--model", help="agent model id; default: TAX_AGENT_MODEL or the built-in default")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")

    topics = tuple(args.topics.split(","))
    model = get_model(args.model)
    questions = args.users * len(topics)
    tokens_in, tokens_out = EST_TOKENS_PER_TURN[DEFAULT_VARIANT]
    estimate = estimate_cost_usd(model.model, questions * tokens_in, questions * tokens_out)
    print(f"[sim-agent] {args.users} users, {questions} questions | model={model.model} | classifier={args.classifier} | estimated cost: "
          + (f"~${estimate:.4f}" if estimate is not None else "n/a (model not in price table)"), flush=True)

    users = [simulate_user(seed, YEAR, args.obscure) for seed in range(args.users)]
    classifier: Classifier = classify_rules if args.classifier == "rules" else LLMClassifier(get_model(args.model))
    everything = classifier([t for u in users for t in u.unlabeled()])
    labeled, start = [], 0
    for u in users:
        labeled.append(everything[start:start + len(u.transactions)])
        start += len(u.transactions)
    cases = run_cases(model, users, labeled, topics)
    print(format_report(cases, model.model, args.classifier, args.obscure))
