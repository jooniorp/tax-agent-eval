"""Fictional taxpayers. Not real people; no real names or identifiers.

Each user exists for one eval case (see evals/dataset.jsonl); amounts are KRW.
"""
from tax_agent.datasource import UserProfile
from tax_agent.mock_api.deduction import CardSpending
from tax_agent.mock_api.medical import MedicalExpenses
from tax_agent.mock_api.rent import RentPayments

# Eval item 1: the NTS guide's official card-deduction example (pp. 136-138).
# Guide totals: credit 32M (incl. transit 2M), cash receipt 8M (incl. market 3M), debit 3M
# (incl. culture 1M). Stored the way CardSpending expects: special categories split out.
_OFFICIAL_CARD_EXAMPLE = CardSpending(
    credit=30_000_000, debit=2_000_000, cash_receipt=5_000_000, market=3_000_000, transit=2_000_000, culture=1_000_000
)

_PROFILES = {
    "user_a": UserProfile("user_a", 68_000_000),
    "user_b": UserProfile("user_b", 85_000_000, homeless_household_head=True),  # rent: salary over the limit
    "user_c": UserProfile("user_c", 50_000_000),  # medical: expenses below the 3% threshold
    "user_d": UserProfile("user_d", None),  # salary unknown: the agent has to ask
    "user_e": UserProfile("user_e", 50_000_000),  # card: threshold taken from a single tier
    "user_f": UserProfile("user_f", 30_000_000),  # card: usage below the 25% minimum
    "user_g": UserProfile("user_g", 80_000_000),  # card: over 70M, lower cap
    "user_h": UserProfile("user_h", 40_000_000),  # card: threshold spills over two tiers
    "user_j": UserProfile("user_j", 50_000_000, homeless_household_head=True),  # rent: eligible, 17%
    "user_k": UserProfile("user_k", 40_000_000),  # medical: eligible
}

_CARD_SPENDING = {
    ("user_a", 2025): _OFFICIAL_CARD_EXAMPLE,
    ("user_d", 2025): _OFFICIAL_CARD_EXAMPLE,  # same spending as user_a, only the salary is missing
    ("user_e", 2025): CardSpending(debit=20_000_000),
    ("user_f", 2025): CardSpending(credit=6_000_000),
    ("user_g", 2025): CardSpending(debit=60_000_000),
    ("user_h", 2025): CardSpending(credit=4_000_000, debit=10_000_000),
}

_RENT = {
    ("user_b", 2025): RentPayments(annual_rent=8_400_000),  # 700k x 12 months
    ("user_j", 2025): RentPayments(annual_rent=6_000_000),
}

_MEDICAL = {
    ("user_c", 2025): MedicalExpenses(self_senior=1_200_000, reimbursed_self_senior=300_000),
    ("user_k", 2025): MedicalExpenses(self_senior=5_000_000),
}


def _lookup(table: dict, user_id: str, year: int, what: str):
    if (user_id, year) not in table:
        raise KeyError(f"no {what} for user {user_id!r}, year {year}")
    return table[(user_id, year)]


class MockDataSource:
    def get_user_profile(self, user_id: str) -> UserProfile:
        if user_id not in _PROFILES:
            raise KeyError(f"unknown user: {user_id!r}")
        return _PROFILES[user_id]

    def get_card_spending(self, user_id: str, year: int) -> CardSpending:
        return _lookup(_CARD_SPENDING, user_id, year, "card spending")

    def get_rent_payments(self, user_id: str, year: int) -> RentPayments:
        return _lookup(_RENT, user_id, year, "rent payments")

    def get_medical_expenses(self, user_id: str, year: int) -> MedicalExpenses:
        return _lookup(_MEDICAL, user_id, year, "medical expenses")
