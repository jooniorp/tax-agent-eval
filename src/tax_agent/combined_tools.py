"""One tool that does the whole job: look the taxpayer up, fetch the data, apply the rules.

The step-by-step tools in tools.py make the model fetch the profile, fetch the data and then
call the calculator: three tool steps, so at least three model calls, and every call sends the whole
conversation again. `compute_deduction` does all of it in code, so the agent needs two model calls
(ask for the tool, explain the result). The calculation is still done by the rule-based calculators.

What the model gives up: it no longer copies numbers from one tool to the next, so the
"number changed on the way" failure cannot happen (and the eval cannot measure it).
"""
from collections.abc import Callable
from dataclasses import asdict

from tax_agent.datasource import DataSource
from tax_agent.mock_api.deduction import calc_card_deduction
from tax_agent.mock_api.medical import calc_medical_credit
from tax_agent.mock_api.rent import calc_rent_credit

TAX_YEAR = 2025
ITEMS = ("card", "rent", "medical")


def build_combined_tools(source: DataSource) -> dict[str, Callable[..., dict]]:
    def compute_deduction(
        user_id: str, item: str, gross_salary: int | None = None, homeless_household_head: bool | None = None
    ) -> dict:
        """Compute a taxpayer's deduction or tax credit for 2025. It looks up the profile and the
        data itself and applies the tax rules, so never calculate amounts yourself.

        item: "card" (credit/debit card and cash-receipt income deduction), "rent" (monthly rent
        tax credit) or "medical" (medical expense tax credit). Call it once per item.
        gross_salary / homeless_household_head: leave them out; fill them in only with a value the
        user gave you after a result said it was missing.

        Returns the amount in "deduction" (KRW), the values the rules used, and "inputs" (the data
        it used). For "rent", eligible=false means there is no credit at all; explain "reason".
        If the result has "missing", ask the user for those values, then call again with them.
        """
        if item not in ITEMS:
            raise ValueError(f"unsupported item {item!r}; supported: {', '.join(repr(i) for i in ITEMS)}")
        profile = source.get_user_profile(user_id)
        salary = gross_salary if gross_salary is not None else profile.gross_salary
        homeless = homeless_household_head if homeless_household_head is not None else profile.homeless_household_head
        missing = (["gross_salary"] if salary is None else []) + (
            ["homeless_household_head"] if item == "rent" and homeless is None else []
        )
        if missing:
            return {"item": item, "missing": missing,
                    "message": "Ask the user for these values, then call compute_deduction again with them."}
        if item == "card":
            data = source.get_card_spending(user_id, TAX_YEAR)
            result = asdict(calc_card_deduction(salary, data))
        elif item == "rent":
            data = source.get_rent_payments(user_id, TAX_YEAR)
            result = asdict(calc_rent_credit(salary, homeless, data))
        else:
            data = source.get_medical_expenses(user_id, TAX_YEAR)
            result = asdict(calc_medical_credit(salary, data))
        inputs = {**asdict(data), **({"homeless_household_head": homeless} if item == "rent" else {})}
        return {"item": item, "gross_salary": salary, **result, "inputs": inputs}

    return {"compute_deduction": compute_deduction}
