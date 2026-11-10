import json
from dataclasses import asdict
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tax_agent.evaluation.scoring import Run, amounts, says_ineligible, says_zero, score_item
from tax_agent.mock_api.deduction import CardSpending, calc_card_deduction
from tax_agent.mock_api.source import MockDataSource

DATASET = Path(__file__).resolve().parent.parent / "evals" / "dataset.jsonl"
ITEMS = {json.loads(line)["id"]: json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line}


def test_amounts_reads_both_korean_styles():
    assert amounts("공제액은 5,300,000원입니다") == {5_300_000}
    assert amounts("공제액은 530만 원입니다") == {5_300_000}
    assert amounts("공제액은 530만원입니다") == {5_300_000}
    assert amounts("총급여 6,800만 원, 최저사용금액 1,700만원, 한도 25%") == {68_000_000, 17_000_000}
    assert amounts("공제가 없습니다") == set()


def test_says_zero_and_says_ineligible():
    for text in ("공제액은 0원입니다.", "공제를 받을 수 없습니다.", "공제 대상이 아닙니다.", "공제액이 없습니다."):
        assert says_zero(text), text
    for text in ("5,300,000원입니다.", "100원 정도입니다.", "공제를 받을 수 있습니다."):
        assert not says_zero(text), text
    for text in ("월세 세액공제 대상이 아닙니다.", "공제를 받을 수 없습니다.", "요건에 해당하지 않습니다."):
        assert says_ineligible(text), text
    for text in ("공제 대상입니다.", "1,020,000원을 받을 수 있습니다."):
        assert not says_ineligible(text), text


# --- a trail shaped like the real one: AI messages with tool calls, JSON tool results ---

def tool_result(name: str, payload: dict, call_id: str) -> ToolMessage:
    return ToolMessage(content=json.dumps(payload), name=name, tool_call_id=call_id)


def card_trail(answer: str, passed_debit: int = 2_000_000) -> Run:
    """The expected trail for item card-01 (user_a), with the debit figure the model passed on."""
    spending = asdict(MockDataSource().get_card_spending("user_a", 2025))
    data = {**spending, "debit": passed_debit}
    calc = asdict(calc_card_deduction(68_000_000, CardSpending(**data)))
    messages = [
        HumanMessage("올해 카드 소득공제 얼마나 받을 수 있어?"),
        AIMessage("", tool_calls=[
            {"name": "get_user_profile", "args": {"user_id": "user_a"}, "id": "1"},
            {"name": "get_card_spending", "args": {"user_id": "user_a", "year": 2025}, "id": "2"},
        ]),
        tool_result("get_user_profile", {"user_id": "user_a", "gross_salary": 68_000_000, "homeless_household_head": None}, "1"),
        tool_result("get_card_spending", spending, "2"),
        AIMessage("", tool_calls=[{"name": "calc_deduction", "args": {"item": "card", "gross_salary": 68_000_000, "data": data}, "id": "3"}]),
        tool_result("calc_deduction", calc, "3"),
        AIMessage(answer),
    ]
    return Run(messages, [len(messages)])


def test_correct_trail_passes_whichever_way_the_amount_is_written():
    for answer in ("카드 소득공제는 5,300,000원입니다.", "카드 소득공제는 530만 원입니다."):
        score = score_item(ITEMS["card-01"], card_trail(answer))
        assert score.passed, [c for c in score.checks if not c.ok]


def test_a_number_changed_on_the_way_to_the_calculator_fails_the_rules():
    score = score_item(ITEMS["card-01"], card_trail("카드 소득공제는 5,300,000원입니다.", passed_debit=2_100_000))
    failed = [c.name for c in score.checks if not c.ok]
    assert "args_relayed" in failed and not score.rules_ok


def test_an_answer_that_differs_from_the_tool_result_fails_both_methods():
    score = score_item(ITEMS["card-01"], card_trail("카드 소득공제는 6,350,000원입니다."))  # the pre-cap figure
    assert not score.answer_ok
    assert "answer_uses_tool_amount" in [c.name for c in score.checks if not c.ok]


def test_no_calculator_call_fails():
    run = card_trail("카드 소득공제는 530만 원입니다.")
    only_lookups = Run(run.messages[:4] + [AIMessage("카드 소득공제는 530만 원입니다.")], [5])
    failed = [c.name for c in score_item(ITEMS["card-01"], only_lookups).checks if not c.ok]
    assert "tool_called" in failed


def ask_back_trail(first_reply: AIMessage, *, calc_in_turn_1: bool) -> Run:
    spending = asdict(MockDataSource().get_card_spending("user_d", 2025))
    calc = asdict(calc_card_deduction(68_000_000, CardSpending(**spending)))
    calc_call = AIMessage("", tool_calls=[{"name": "calc_deduction", "args": {"item": "card", "gross_salary": 68_000_000, "data": spending}, "id": "3"}])
    messages = [
        HumanMessage("카드 소득공제 얼마나 받아?"),
        AIMessage("", tool_calls=[
            {"name": "get_user_profile", "args": {"user_id": "user_d"}, "id": "1"},
            {"name": "get_card_spending", "args": {"user_id": "user_d", "year": 2025}, "id": "2"},
        ]),
        tool_result("get_user_profile", {"user_id": "user_d", "gross_salary": None, "homeless_household_head": None}, "1"),
        tool_result("get_card_spending", spending, "2"),
    ]
    if calc_in_turn_1:
        messages += [calc_call, tool_result("calc_deduction", calc, "3")]
    messages.append(first_reply)
    turn_1_end = len(messages)
    messages += [HumanMessage("총급여는 6,800만원이야.")]
    if not calc_in_turn_1:
        messages += [calc_call, tool_result("calc_deduction", calc, "3")]
    messages.append(AIMessage("카드 소득공제는 530만 원입니다."))
    return Run(messages, [turn_1_end, len(messages)])


def test_asking_for_the_missing_salary_before_calculating_passes():
    run = ask_back_trail(AIMessage("총급여가 얼마인지 알려주세요."), calc_in_turn_1=False)
    score = score_item(ITEMS["ask-01"], run)
    assert score.passed, [c for c in score.checks if not c.ok]


def test_guessing_the_salary_instead_of_asking_fails():
    run = ask_back_trail(AIMessage("카드 소득공제는 530만 원입니다."), calc_in_turn_1=True)
    failed = [c.name for c in score_item(ITEMS["ask-01"], run).checks if not c.ok]
    assert "tool_not_called@turn1" in failed and "no_amount@turn1" in failed and "asks_question@turn1" in failed
