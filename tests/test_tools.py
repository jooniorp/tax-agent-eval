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
        tools["calc_deduction"]("medical", 68_000_000, {})
