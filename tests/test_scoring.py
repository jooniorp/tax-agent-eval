import json
from dataclasses import asdict
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tax_agent.evaluation.scoring import (
    Run, amounts, answer_match, args_relayed, needs_calculator, says_ineligible, says_not_found,
    says_unsupported, says_zero, score_item, specs_for,
)
from tax_agent.combined_tools import build_combined_tools
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
    assert amounts("공제액은 167만 8,011원입니다") == {1_678_011}  # the mixed style a model uses for exact figures
    assert amounts("공제 대상은 1억 2,000만 원, 한도는 300만원") == {120_000_000, 3_000_000}
    assert amounts("2025년에 25%를 적용해 3개월 동안") == set()  # numbers without a won unit are not amounts


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
    assert "tool_not_called@turn1" in failed and "asks_question@turn1" in failed
    assert "answer_not_contains_amount@turn1" in failed  # it stated the deduction before it had the salary


def test_restating_known_data_while_asking_is_not_stating_the_deduction():
    # Seen in a real run: the agent asks for the missing fact and recaps what the tools already told it.
    reply = AIMessage("세대주이면서 무주택자이신가요? 확인된 정보는 총급여 5,000만 원, 월세 700만 원입니다.")
    run = ask_back_trail(reply, calc_in_turn_1=False)
    failed = [c.name for c in score_item(ITEMS["ask-01"], run).checks if not c.ok]
    assert "answer_not_contains_amount@turn1" not in failed and "no_amount@turn1" not in failed


def test_says_unsupported_and_says_not_found():
    for text in ("교육비 공제는 아직 지원하지 않습니다.", "교육비는 계산할 수 없습니다.", "해당 도구가 제공되지 않아 확인이 어렵습니다. 아직 지원 전입니다."):
        assert says_unsupported(text), text
    for text in ("교육비 공제는 530만 원입니다.", "월세 공제를 받을 수 있습니다."):
        assert not says_unsupported(text), text
    for text in ("사용자 정보를 찾을 수 없습니다.", "해당 사용자는 등록되어 있지 않습니다.", "조회되지 않았습니다."):
        assert says_not_found(text), text
    assert not says_not_found("카드 소득공제는 530만 원입니다.")


def test_wordings_the_first_scoring_rules_missed_are_read_correctly():
    # Real answers from the 2026-11-14 batch that were scored as failures although they were right.
    no_tool = "저는 이 질문에 답할 수 없습니다. 교육비 세액공제에 필요한 데이터 조회와 계산 기능이 없습니다. 그래서 금액을 알려드릴 수 없습니다."
    unknown_user = '"알 수 없는 사용자"라는 오류가 두 번 반환되어 카드 사용 내역을 가져오지 못했습니다.'
    assert says_unsupported(no_tool) and not says_not_found(no_tool)
    assert says_not_found(unknown_user)
    # The combined agent's wording in the second batch: the tool cannot look the topic up.
    can_not_look_up = "죄송하지만 이 도구로는 **교육비 세액공제 금액을 조회할 수 없습니다.** 현재 조회할 수 있는 항목은 다음 세 가지입니다."
    assert says_unsupported(can_not_look_up)
    assert not says_unsupported("월세 세액공제는 조회할 수 있고 금액은 102만 원입니다.")
    assert not says_unsupported("카드 소득공제는 530만 원이고, 월세 공제도 받을 수 있습니다.")


def run_with_answer(answer: str) -> Run:
    messages = [HumanMessage("q"), AIMessage(answer)]
    return Run(messages, [len(messages)])


def test_answer_match_for_every_expected_form():
    assert answer_match(run_with_answer("공제는 530만 원입니다."), {"amount": 5_300_000})[0]
    assert not answer_match(run_with_answer("공제는 630만 원입니다."), {"amount": 5_300_000})[0]
    assert answer_match(run_with_answer("카드 225만 원, 월세 1,020,000원입니다."), {"amounts": [2_250_000, 1_020_000]})[0]
    assert not answer_match(run_with_answer("카드 225만 원입니다."), {"amounts": [2_250_000, 1_020_000]})[0]  # one missing
    assert answer_match(run_with_answer("지원하지 않는 항목입니다."), {"unsupported": True})[0]
    assert answer_match(run_with_answer("사용자를 찾을 수 없습니다."), {"not_found": True})[0]


def test_a_null_in_the_profile_means_the_user_may_supply_the_value():
    spending = {"annual_rent": 7_000_000}
    messages = [
        HumanMessage("월세 공제 얼마야?"),
        AIMessage("", tool_calls=[{"name": "get_user_profile", "args": {"user_id": "user_p"}, "id": "1"}]),
        tool_result("get_user_profile", {"user_id": "user_p", "gross_salary": 50_000_000, "homeless_household_head": None}, "1"),
        AIMessage("", tool_calls=[{"name": "get_rent_payments", "args": {"user_id": "user_p", "year": 2025}, "id": "2"}]),
        tool_result("get_rent_payments", spending, "2"),
        AIMessage("", tool_calls=[{"name": "calc_deduction", "args": {
            "item": "rent", "gross_salary": 50_000_000, "data": {**spending, "homeless_household_head": True}}, "id": "3"}]),
        tool_result("calc_deduction", {"eligible": True, "reason": "ok", "deduction": 1_190_000}, "3"),
        AIMessage("월세 세액공제는 119만 원입니다."),
    ]
    run = Run(messages, [len(messages)])
    assert args_relayed(run, {})[0]  # True was given by the user, the profile said null
    wrong = Run(messages[:5] + [AIMessage("", tool_calls=[{"name": "calc_deduction", "args": {
        "item": "rent", "gross_salary": 50_000_000, "data": {"annual_rent": 7_700_000, "homeless_household_head": True}}, "id": "4"}])], [6])
    assert not args_relayed(wrong, {})[0]  # a changed rent figure is still caught


def test_every_calculator_result_must_appear_in_the_answer_of_a_two_topic_question():
    spending = asdict(MockDataSource().get_card_spending("user_s", 2025))
    card = asdict(calc_card_deduction(50_000_000, CardSpending(**spending)))
    messages = [
        HumanMessage("카드 공제랑 월세 공제 각각 얼마야?"),
        AIMessage("", tool_calls=[{"name": "get_user_profile", "args": {"user_id": "user_s"}, "id": "1"}]),
        tool_result("get_user_profile", {"user_id": "user_s", "gross_salary": 50_000_000, "homeless_household_head": True}, "1"),
        AIMessage("", tool_calls=[{"name": "get_card_spending", "args": {"user_id": "user_s", "year": 2025}, "id": "2"}]),
        tool_result("get_card_spending", spending, "2"),
        AIMessage("", tool_calls=[{"name": "calc_deduction", "args": {"item": "card", "gross_salary": 50_000_000, "data": spending}, "id": "3"}]),
        tool_result("calc_deduction", card, "3"),
        AIMessage("", tool_calls=[{"name": "get_rent_payments", "args": {"user_id": "user_s", "year": 2025}, "id": "4"}]),
        tool_result("get_rent_payments", {"annual_rent": 6_000_000}, "4"),
        AIMessage("", tool_calls=[{"name": "calc_deduction", "args": {
            "item": "rent", "gross_salary": 50_000_000, "data": {"annual_rent": 6_000_000, "homeless_household_head": True}}, "id": "5"}]),
        tool_result("calc_deduction", {"eligible": True, "reason": "ok", "deduction": 1_020_000}, "5"),
    ]

    def score(answer: str):
        full = messages + [AIMessage(answer)]
        return score_item(ITEMS["multi-01"], Run(full, [len(full)]))

    assert score("카드 소득공제는 225만 원, 월세 세액공제는 102만 원입니다.").passed
    only_one = score("카드 소득공제는 225만 원입니다.")
    assert not only_one.answer_ok and "answer_uses_tool_amount" in [c.name for c in only_one.checks if not c.ok]


def test_the_control_variant_skips_the_checks_that_need_the_calculator():
    assert needs_calculator({"type": "args_relayed"}) and needs_calculator({"type": "tool_called", "name": "calc_deduction"})
    assert not needs_calculator({"type": "tool_called", "name": "get_user_profile"})
    assert not needs_calculator({"type": "asks_question", "turn": 1})
    no_calc = Run([HumanMessage("q"), AIMessage("카드 소득공제는 530만 원입니다.")], [2])
    score = score_item(ITEMS["card-01"], no_calc, variant="no_calculator")
    assert score.checks == [] and score.passed  # the answer is right and no check applies
    assert not score_item(ITEMS["card-01"], no_calc, variant="calculator").rules_ok  # with the calculator, no tool call fails


# --- the combined agent: one compute_deduction tool instead of profile + data + calculator ---

COMBINED = build_combined_tools(MockDataSource())["compute_deduction"]


def combined_ask_back_trail(first_reply: AIMessage, *, compute_in_turn_1: bool) -> Run:
    """Item ask-01 (user_d, salary not on file) as the combined agent would play it."""
    missing_args = {"user_id": "user_d", "item": "card"}
    with_salary = {**missing_args, "gross_salary": 68_000_000}
    messages = [HumanMessage("카드 소득공제 얼마나 받아?")]
    if compute_in_turn_1:  # a guess: the salary is made up, the tool computes
        messages += [AIMessage("", tool_calls=[{"name": "compute_deduction", "args": with_salary, "id": "1"}]),
                     tool_result("compute_deduction", COMBINED(**with_salary), "1")]
    else:
        messages += [AIMessage("", tool_calls=[{"name": "compute_deduction", "args": missing_args, "id": "1"}]),
                     tool_result("compute_deduction", COMBINED(**missing_args), "1")]
    messages.append(first_reply)
    turn_1_end = len(messages)
    messages += [
        HumanMessage("총급여는 6,800만원이야."),
        AIMessage("", tool_calls=[{"name": "compute_deduction", "args": with_salary, "id": "2"}]),
        tool_result("compute_deduction", COMBINED(**with_salary), "2"),
        AIMessage("카드 소득공제는 530만 원입니다."),
    ]
    return Run(messages, [turn_1_end, len(messages)])


def test_specs_for_the_combined_agent_are_rewritten_for_the_single_tool():
    specs = specs_for(ITEMS["ask-01"]["checks"], "combined")
    kinds = [(s["type"], s.get("name"), s.get("turn")) for s in specs]
    assert ("tool_called", "compute_deduction", 1) in kinds  # looking the user up is part of that call
    assert ("no_deduction_result", None, 1) in kinds and ("tool_not_called", "calc_deduction", 1) not in kinds
    assert specs_for([{"type": "tool_order"}, {"type": "args_relayed"}], "combined") == []  # no steps to order or pass on


def test_the_combined_agent_passes_when_it_asks_first_and_computes_after_the_answer():
    run = combined_ask_back_trail(AIMessage("총급여가 얼마인지 알려주세요."), compute_in_turn_1=False)
    score = score_item(ITEMS["ask-01"], run, variant="combined")
    assert score.passed, [c for c in score.checks if not c.ok]


def test_the_combined_agent_fails_when_it_computes_with_a_guessed_salary():
    run = combined_ask_back_trail(AIMessage("카드 소득공제는 530만 원입니다."), compute_in_turn_1=True)
    failed = [c.name for c in score_item(ITEMS["ask-01"], run, variant="combined").checks if not c.ok]
    assert "no_deduction_result@turn1" in failed and "asks_question@turn1" in failed


def test_the_thousand_unit_and_the_wordings_the_openai_model_used_are_read():
    # Real answers from the 2026-11-18 model comparison that were scored as failures although they were right.
    assert amounts("의료비 세액공제액은 37만 5천 원입니다") == {375_000}
    assert amounts("총급여 5천만 원, 공제 3천 원") == {50_000_000, 3_000}
    assert amounts("1억 2천만 원과 5,300,000원") == {120_000_000, 5_300_000}
    assert amounts("2025년에 3개월 동안 천천히") == set()
    refusals = ("교육비 세액공제는 현재 조회할 수 없어 공제액을 확인해 드릴 수 없어요.",
                "교육비 세액공제 금액을 조회할 수 없어, 얼마를 공제받는지 확인해 드릴 수 없어요.",
                "교육비 세액공제액은 확인해 드릴 수 없어요.")
    assert all(says_unsupported(text) for text in refusals)
    assert not says_unsupported("카드 소득공제는 530만 원이고 월세 공제도 계산해 드릴 수 있어요.")


def test_an_explanation_that_says_the_money_was_given_back_counts_as_mentioning_the_insurance_payout():
    answer = "병원비 120만 원에서 돌려받은 30만 원을 빼면 실제 부담액은 90만 원이고, 총급여의 3%인 150만 원을 넘지 않아 0원이에요."
    run = Run([HumanMessage("q"), AIMessage(answer)], [2])
    failed = [c.name for c in score_item(ITEMS["med-01"], run, variant="no_calculator").checks if not c.ok]
    assert failed == []


def test_zero_and_ineligible_wordings_seen_in_the_simulated_users_run():
    # Real answer of 2026-11-18 (sim-agent): right, but the first rules did not read it as "no deduction".
    answer = "올해 월세 세액공제는 해당이 없어서 받을 수 있는 금액이 없습니다. 이유는 무주택 세대주가 아니기 때문입니다."
    assert says_zero(answer) and says_ineligible(answer)
    assert not says_zero("월세 세액공제는 1,020,000원입니다.") and not says_ineligible("월세 세액공제는 1,020,000원입니다.")
