"""Agent tools: plain functions over a `DataSource` and the rule-based calculator.

They return JSON-friendly dicts so any agent framework can wrap them. The
framework binding (LangGraph) comes later; docstrings here become the tool
descriptions the model sees. All amounts are KRW integers.
"""
from collections.abc import Callable
from dataclasses import asdict

from tax_agent.datasource import DataSource
from tax_agent.mock_api.deduction import CardSpending, calc_card_deduction


def build_tools(source: DataSource) -> dict[str, Callable[..., dict]]:
    def get_user_profile(user_id: str) -> dict:
        """Get a taxpayer's profile: user_id and gross_salary (total annual salary, KRW)."""
        return asdict(source.get_user_profile(user_id))

    def get_card_spending(user_id: str, year: int) -> dict:
        """Get a taxpayer's eligible card / cash-receipt usage for a tax year, in KRW.

        Fields: credit, debit, cash_receipt (regular usage), and market, transit,
        culture (special categories). The categories never overlap, so pass the
        dict unchanged as `spending` to calc_deduction.
        """
        return asdict(source.get_card_spending(user_id, year))

    def calc_deduction(item: str, gross_salary: int, spending: dict[str, int]) -> dict:
        """Calculate a deduction by the tax rules. Never compute amounts yourself.

        item: only "card" (credit/debit card and cash-receipt income deduction) is supported.
        Returns minimum_usage (25% of gross salary), before_limit and deduction (final, KRW).
        """
        if item != "card":
            raise ValueError(f"unsupported item {item!r}; supported: 'card'")
        return asdict(calc_card_deduction(gross_salary, CardSpending(**spending)))

    return {f.__name__: f for f in (get_user_profile, get_card_spending, calc_deduction)}
