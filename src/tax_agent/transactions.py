"""A taxpayer's own transactions as a data source: spending in, deduction inputs out.

Whatever channel the transactions come from (a card company's export, a bank export, an open banking
or MyData API later), an adapter turns them into `Transaction` rows and everything below stays the
same. `TransactionDataSource` adds them up per tax year into the inputs the calculators take, so the
agent tools see it exactly like `MockDataSource` (see docs/decisions.md, 2026-11-18).

Rules this module applies (first version, to be checked against real exports):
- one transaction counts in one bucket only;
- a cancellation or refund is a negative amount and reduces its bucket, which never goes below 0;
- "excluded" and "unclassified" transactions count nowhere, and `unclassified()` lists them so a person
  can label them: guessing a category would silently change a deduction;
- the calculators want only *eligible* usage, so what counts as eligible is decided by the category an
  adapter or a person gave the row, not here.

Not handled yet (open questions, not silently assumed): whether a medical payment made by credit card
also counts as card usage, donations (no calculator yet), insurance payouts (passed in by hand).
"""
from dataclasses import dataclass
from datetime import date

from tax_agent.datasource import UserProfile
from tax_agent.mock_api.deduction import HIGH_INCOME_LINE, CardSpending
from tax_agent.mock_api.medical import MedicalExpenses
from tax_agent.mock_api.rent import RentPayments

METHODS = ("credit", "debit", "cash_receipt", "other")
SPECIAL = ("market", "transit", "culture")  # card deduction buckets that ignore the payment method
CATEGORIES = ("general", *SPECIAL, "medical", "rent", "excluded", "unclassified")


@dataclass(frozen=True)
class Transaction:
    date: str  # "YYYY-MM-DD"
    amount: int  # KRW; negative for a cancellation or refund
    merchant: str
    method: str  # one of METHODS
    category: str = "unclassified"  # one of CATEGORIES
    for_whom: str = "self"  # medical only: "self" (taxpayer, 65+, 6 or under, disabled) or "other" (capped group)

    def __post_init__(self) -> None:
        try:
            date.fromisoformat(self.date)
        except ValueError:
            raise ValueError(f"date must be YYYY-MM-DD, got {self.date!r}") from None
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}, got {self.category!r}")
        if self.for_whom not in ("self", "other"):
            raise ValueError(f"for_whom must be 'self' or 'other', got {self.for_whom!r}")

    @property
    def year(self) -> int:
        return int(self.date[:4])


def _bucket_sums(rows: list[Transaction], keys: tuple[str, ...], key_of) -> dict[str, int]:
    sums = dict.fromkeys(keys, 0)
    for t in rows:
        key = key_of(t)
        if key in sums:
            sums[key] += t.amount
    return {k: max(v, 0) for k, v in sums.items()}  # refunds can empty a bucket, never make it negative


def card_spending(transactions: list[Transaction], year: int, gross_salary: int | None = None) -> CardSpending:
    """Eligible card and cash-receipt usage of the year, by calculator bucket.

    Above a total salary of 70M the culture usage is plain card usage (the calculator refuses it
    otherwise), so with `gross_salary` given it is counted by payment method instead.
    """
    rows = [t for t in transactions if t.year == year]
    culture_applies = gross_salary is None or gross_salary <= HIGH_INCOME_LINE

    def bucket(t: Transaction) -> str | None:
        if t.category == "culture" and not culture_applies:
            return t.method if t.method in ("credit", "debit", "cash_receipt") else None
        if t.category in SPECIAL:
            return t.category
        if t.category == "general" and t.method in ("credit", "debit", "cash_receipt"):
            return t.method
        return None

    return CardSpending(**_bucket_sums(rows, ("credit", "debit", "cash_receipt", *SPECIAL), bucket))


def rent_payments(transactions: list[Transaction], year: int) -> RentPayments:
    rows = [t for t in transactions if t.year == year]
    return RentPayments(annual_rent=_bucket_sums(rows, ("rent",), lambda t: t.category)["rent"])


def medical_expenses(
    transactions: list[Transaction], year: int, reimbursed_self_senior: int = 0, reimbursed_other: int = 0
) -> MedicalExpenses:
    """Insurance payouts are not in a card or bank export in a form this can match, so they are passed in."""
    rows = [t for t in transactions if t.year == year]
    sums = _bucket_sums(rows, ("self", "other"), lambda t: t.for_whom if t.category == "medical" else None)
    return MedicalExpenses(
        self_senior=sums["self"], other=sums["other"],
        reimbursed_self_senior=reimbursed_self_senior, reimbursed_other=reimbursed_other,
    )


def unclassified(transactions: list[Transaction], year: int) -> list[Transaction]:
    """What still needs a category, largest first. They count nowhere until someone labels them."""
    rows = [t for t in transactions if t.year == year and t.category == "unclassified"]
    return sorted(rows, key=lambda t: -abs(t.amount))


class TransactionDataSource:
    """`DataSource` for one real taxpayer. The profile (total salary, household status) is given by the
    taxpayer: a card or bank export does not contain it."""

    def __init__(
        self, profile: UserProfile, transactions: list[Transaction],
        reimbursed_self_senior: int = 0, reimbursed_other: int = 0,
    ):
        self._profile = profile
        self._transactions = list(transactions)
        self._reimbursed = (reimbursed_self_senior, reimbursed_other)

    def _check(self, user_id: str) -> None:
        if user_id != self._profile.user_id:
            raise KeyError(f"unknown user {user_id!r}")

    def get_user_profile(self, user_id: str) -> UserProfile:
        self._check(user_id)
        return self._profile

    def get_card_spending(self, user_id: str, year: int) -> CardSpending:
        self._check(user_id)
        return card_spending(self._transactions, year, self._profile.gross_salary)

    def get_rent_payments(self, user_id: str, year: int) -> RentPayments:
        self._check(user_id)
        return rent_payments(self._transactions, year)

    def get_medical_expenses(self, user_id: str, year: int) -> MedicalExpenses:
        self._check(user_id)
        return medical_expenses(self._transactions, year, *self._reimbursed)
