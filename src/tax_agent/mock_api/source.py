"""Fictional taxpayers. Not real people; no real names or identifiers."""
from tax_agent.datasource import UserProfile
from tax_agent.mock_api.deduction import CardSpending

_PROFILES = {
    # Eval item 1: the NTS guide's official card-deduction example (pp. 136-138).
    "user_a": UserProfile(user_id="user_a", gross_salary=68_000_000),
}

_CARD_SPENDING = {
    # Guide totals: credit 32M (incl. transit 2M), cash receipt 8M (incl. market 3M),
    # debit 3M (incl. culture 1M). Stored the way CardSpending expects: the special
    # categories are split out, so the categories never overlap.
    ("user_a", 2025): CardSpending(
        credit=30_000_000,
        debit=2_000_000,
        cash_receipt=5_000_000,
        market=3_000_000,
        transit=2_000_000,
        culture=1_000_000,
    ),
}


class MockDataSource:
    def get_user_profile(self, user_id: str) -> UserProfile:
        if user_id not in _PROFILES:
            raise KeyError(f"unknown user: {user_id!r}")
        return _PROFILES[user_id]

    def get_card_spending(self, user_id: str, year: int) -> CardSpending:
        if (user_id, year) not in _CARD_SPENDING:
            raise KeyError(f"no card spending for user {user_id!r}, year {year}")
        return _CARD_SPENDING[(user_id, year)]
