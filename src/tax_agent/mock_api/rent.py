"""Rule-based monthly rent tax credit (ground truth for evals).

Rules: 2025 tax-year NTS year-end settlement guide, printed pp. 202-203
(Special Tax Treatment Control Act Art. 95-2). All amounts are KRW integers.

Not modelled: the house size / price limit, the address match and the total-income
test. The data source only reports rent that qualifies on those points.
"""
from dataclasses import dataclass

MAX_SALARY = 80_000_000  # total salary above this gets no credit
HIGH_RATE_SALARY = 55_000_000  # up to this the rate is 17%, above it 15%
ANNUAL_RENT_LIMIT = 10_000_000  # rent counted at most


@dataclass(frozen=True)
class RentPayments:
    annual_rent: int  # rent paid in the tax year for the qualifying home


@dataclass(frozen=True)
class RentCredit:
    eligible: bool
    reason: str  # "ok", "total_salary_over_80m" or "not_homeless_household_head"
    deduction: int  # tax credit; 0 when not eligible


def calc_rent_credit(gross_salary: int, homeless_household_head: bool | None, rent: RentPayments) -> RentCredit:
    if gross_salary <= 0 or rent.annual_rent < 0:
        raise ValueError("gross_salary must be positive and annual_rent non-negative")
    if homeless_household_head is None:
        raise ValueError("homeless_household_head is unknown: ask the user")
    if gross_salary > MAX_SALARY:
        return RentCredit(False, "total_salary_over_80m", 0)
    if not homeless_household_head:
        return RentCredit(False, "not_homeless_household_head", 0)
    rate = 17 if gross_salary <= HIGH_RATE_SALARY else 15
    return RentCredit(True, "ok", min(rent.annual_rent, ANNUAL_RENT_LIMIT) * rate // 100)
