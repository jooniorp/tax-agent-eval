"""Rule-based card / cash-receipt income deduction: the ground truth for evals.

Rules: 2025 tax-year NTS year-end settlement guide, printed pp. 132-138
(Special Tax Treatment Control Act Art. 126-2). All amounts are KRW integers.

The caller passes only *eligible* usage. Filtering out excluded spending
(utilities, insurance, tuition, ...) and merging family members' usage is the
data source's job, not this module's.
"""
from dataclasses import dataclass

HIGH_INCOME_LINE = 70_000_000  # total salary above this gets lower limits
MIN_USAGE_PERCENT = 25  # usage below 25% of total salary earns no deduction


@dataclass(frozen=True)
class CardSpending:
    """Eligible usage by category.

    The three special categories (market, transit, culture) are NOT included
    in credit / debit / cash_receipt: the guide subtracts them from those
    totals before applying the regular rates.
    """

    credit: int = 0  # credit card, 15%
    debit: int = 0  # debit / prepaid (real-name verified), 30%
    cash_receipt: int = 0  # cash receipt, 30%
    market: int = 0  # traditional market, any payment method, 40%
    transit: int = 0  # public transit, any payment method, 40%
    culture: int = 0  # books / shows / sports, any method, 30%, salary <= 70M only


@dataclass(frozen=True)
class CardDeduction:
    minimum_usage: int  # 25% of total salary
    before_limit: int  # deductible amount before the cap
    base_limit: int  # basic cap (3M KRW, or 2.5M above 70M total salary)
    extra_deduction: int  # amount allowed above the basic cap for market / transit / culture usage
    deduction: int  # final deduction after the cap (the basic cap part + extra_deduction)


def calc_card_deduction(gross_salary: int, spending: CardSpending) -> CardDeduction:
    amounts = (spending.credit, spending.debit, spending.cash_receipt,
               spending.market, spending.transit, spending.culture)
    if gross_salary <= 0 or any(a < 0 for a in amounts):
        raise ValueError("gross_salary must be positive and amounts non-negative")
    high_income = gross_salary > HIGH_INCOME_LINE
    if spending.culture and high_income:
        # Above 70M the culture usage is plain card usage; the caller must
        # fold it into credit/debit/cash_receipt by payment method instead.
        raise ValueError("culture usage only applies to total salary <= 70M")

    # (rate %, amount), lowest rate first: the guide subtracts the minimum
    # usage from the lowest-rate spending first (credit -> debit/cash -> market/transit).
    tiers = (
        (15, spending.credit),
        (30, spending.debit + spending.cash_receipt + spending.culture),
        (40, spending.market + spending.transit),
    )
    minimum_usage = gross_salary * MIN_USAGE_PERCENT // 100
    remaining = minimum_usage
    weighted = 0  # amount * percent, divided by 100 once at the end
    for rate, amount in tiers:
        absorbed = min(amount, remaining)
        remaining -= absorbed
        weighted += (amount - absorbed) * rate
    before_limit = weighted // 100

    base_limit = 2_500_000 if high_income else 3_000_000
    extra_limit = 2_000_000 if high_income else 3_000_000
    special = (spending.market * 40 + spending.transit * 40 + spending.culture * 30) // 100
    extra = min(max(before_limit - base_limit, 0), special, extra_limit)
    return CardDeduction(minimum_usage, before_limit, base_limit, extra, min(before_limit, base_limit) + extra)
