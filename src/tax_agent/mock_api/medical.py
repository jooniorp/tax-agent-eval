"""Rule-based medical expense tax credit (ground truth for evals).

Rules: 2025 tax-year NTS year-end settlement guide, printed pp. 173 and 176
(Income Tax Act Art. 59-4 (2)). All amounts are KRW integers.

Modelled: the two common groups (taxpayer / age 65+ / age 6 or under / disabled with no
cap, and other dependents with a 7M cap), the 3% of total salary threshold taken from the
"other dependents" group first, and insurance payouts that are not eligible. Not modelled:
infertility treatment (30%) and premature / congenital-anomaly infants (20%).
"""
from dataclasses import dataclass

RATE_PERCENT = 15
THRESHOLD_PERCENT = 3
OTHER_DEPENDENTS_CAP = 7_000_000


@dataclass(frozen=True)
class MedicalExpenses:
    self_senior: int = 0  # paid for the taxpayer, age 65+, age 6 or under, disabled: no cap
    other: int = 0  # paid for other basic-deduction dependents: capped
    reimbursed_self_senior: int = 0  # insurance payouts received for the expenses above
    reimbursed_other: int = 0  # insurance payouts received for the "other" expenses


@dataclass(frozen=True)
class MedicalCredit:
    threshold: int  # 3% of total salary
    net_expenses: int  # all expenses after subtracting insurance payouts
    eligible_expenses: int  # the part the 15% applies to
    deduction: int  # tax credit


def calc_medical_credit(gross_salary: int, expenses: MedicalExpenses) -> MedicalCredit:
    amounts = (expenses.self_senior, expenses.other, expenses.reimbursed_self_senior, expenses.reimbursed_other)
    if gross_salary <= 0 or any(a < 0 for a in amounts):
        raise ValueError("gross_salary must be positive and amounts non-negative")
    net_self = max(expenses.self_senior - expenses.reimbursed_self_senior, 0)
    net_other = max(expenses.other - expenses.reimbursed_other, 0)
    threshold = gross_salary * THRESHOLD_PERCENT // 100
    # The threshold is taken from the capped group first; what is left over comes off the uncapped group.
    eligible_other = min(max(net_other - threshold, 0), OTHER_DEPENDENTS_CAP)
    eligible_self = max(net_self - max(threshold - net_other, 0), 0)
    eligible = eligible_self + eligible_other
    return MedicalCredit(threshold, net_self + net_other, eligible, eligible * RATE_PERCENT // 100)
