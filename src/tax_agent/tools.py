"""Agent tools: plain functions over a `DataSource` and the rule-based calculators.

They return JSON-friendly dicts so any agent framework can wrap them. The
framework binding (LangGraph) happens in agent.py; the docstrings here become
the tool descriptions the model sees. All amounts are KRW integers.
"""
from collections.abc import Callable
from dataclasses import asdict

from tax_agent.datasource import DataSource
from tax_agent.mock_api.deduction import CardSpending, calc_card_deduction
from tax_agent.mock_api.medical import MedicalExpenses, calc_medical_credit
from tax_agent.mock_api.rent import RentPayments, calc_rent_credit


def build_tools(source: DataSource) -> dict[str, Callable[..., dict]]:
    def get_user_profile(user_id: str) -> dict:
        """Get a taxpayer's profile: gross_salary (total annual salary, KRW) and
        homeless_household_head (head of a household that owns no home). A null value
        means the information is not known."""
        return asdict(source.get_user_profile(user_id))

    def get_card_spending(user_id: str, year: int) -> dict:
        """Get a taxpayer's eligible card / cash-receipt usage for a tax year, in KRW.

        Fields: credit, debit, cash_receipt (regular usage), and market, transit,
        culture (special categories). The categories never overlap, so pass the
        dict unchanged as `data` to calc_deduction with item "card".
        """
        return asdict(source.get_card_spending(user_id, year))

    def get_rent_payments(user_id: str, year: int) -> dict:
        """Get the total monthly rent a taxpayer paid in a tax year, as annual_rent in KRW."""
        return asdict(source.get_rent_payments(user_id, year))

    def get_medical_expenses(user_id: str, year: int) -> dict:
        """Get a taxpayer's medical expenses for a tax year, in KRW.

        self_senior: paid for the taxpayer, people aged 65+ or 6 or under, disabled people.
        other: paid for other dependents. reimbursed_self_senior / reimbursed_other: insurance
        payouts already received for those expenses. Pass the dict unchanged as `data` to
        calc_deduction with item "medical".
        """
        return asdict(source.get_medical_expenses(user_id, year))

    def calc_deduction(item: str, gross_salary: int, data: dict) -> dict:
        """Calculate a deduction or tax credit by the tax rules. Never compute amounts yourself.

        item "card": credit/debit card and cash-receipt income deduction.
            data = the get_card_spending result.
        item "rent": monthly rent tax credit.
            data = {"annual_rent": <from get_rent_payments>,
                    "homeless_household_head": <from get_user_profile>}.
        item "medical": medical expense tax credit. data = the get_medical_expenses result.

        The amount is in "deduction" (KRW); the other fields are the values the rules used.
        For "rent", eligible=false means there is no credit at all; explain "reason".
        """
        if item == "card":
            return asdict(calc_card_deduction(gross_salary, CardSpending(**data)))
        if item == "rent":
            rent = RentPayments(annual_rent=data["annual_rent"])
            return asdict(calc_rent_credit(gross_salary, data.get("homeless_household_head"), rent))
        if item == "medical":
            return asdict(calc_medical_credit(gross_salary, MedicalExpenses(**data)))
        raise ValueError(f"unsupported item {item!r}; supported: 'card', 'rent', 'medical'")

    tools = (get_user_profile, get_card_spending, get_rent_payments, get_medical_expenses, calc_deduction)
    return {f.__name__: f for f in tools}
