import pytest

from tax_agent.mock_api.deduction import CardSpending
from tax_agent.mock_api.source import MockDataSource
from tax_agent.datasource import UserProfile
from tax_agent.tools import build_tools


def test_card_question_for_user_a_end_to_end():
    # The call sequence the agent is expected to make for eval item 1.
    tools = build_tools(MockDataSource())
    profile = tools["get_user_profile"]("user_a")
    spending = tools["get_card_spending"]("user_a", 2025)
    result = tools["calc_deduction"]("card", profile["gross_salary"], spending)
    assert result["deduction"] == 5_300_000  # NTS guide official example, 530만원


def test_tools_only_depend_on_the_datasource_interface():
    class FakeSource:
        def get_user_profile(self, user_id):
            return UserProfile(user_id=user_id, gross_salary=50_000_000)

        def get_card_spending(self, user_id, year):
            return CardSpending(debit=50_000_000)

    tools = build_tools(FakeSource())
    spending = tools["get_card_spending"]("anyone", 2025)
    assert tools["calc_deduction"]("card", 50_000_000, spending)["deduction"] == 3_000_000


def test_unknown_user_and_unsupported_item_raise():
    tools = build_tools(MockDataSource())
    with pytest.raises(KeyError):
        tools["get_user_profile"]("nobody")
    with pytest.raises(KeyError):
        tools["get_card_spending"]("user_a", 2024)
    with pytest.raises(ValueError):
        tools["calc_deduction"]("education", 68_000_000, {})


def test_rent_and_medical_questions_through_the_tools():
    tools = build_tools(MockDataSource())
    # rent, eligible: user_j (salary 50M, homeless household head, rent 6M) -> 6M x 17%
    profile = tools["get_user_profile"]("user_j")
    rent = tools["get_rent_payments"]("user_j", 2025)
    data = {**rent, "homeless_household_head": profile["homeless_household_head"]}
    assert tools["calc_deduction"]("rent", profile["gross_salary"], data)["deduction"] == 1_020_000
    # rent, not eligible: user_b earns 85M, over the 80M limit
    profile = tools["get_user_profile"]("user_b")
    data = {**tools["get_rent_payments"]("user_b", 2025), "homeless_household_head": True}
    result = tools["calc_deduction"]("rent", profile["gross_salary"], data)
    assert result == {"eligible": False, "reason": "total_salary_over_80m", "deduction": 0}
    # medical: user_k (salary 40M, 5M paid for self) -> (5M - 3% of 40M) x 15%
    profile = tools["get_user_profile"]("user_k")
    expenses = tools["get_medical_expenses"]("user_k", 2025)
    assert tools["calc_deduction"]("medical", profile["gross_salary"], expenses)["deduction"] == 570_000


def test_missing_salary_is_reported_as_null_not_invented():
    assert build_tools(MockDataSource())["get_user_profile"]("user_d")["gross_salary"] is None
