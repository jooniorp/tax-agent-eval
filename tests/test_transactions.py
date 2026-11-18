import pytest

from tax_agent.combined_tools import build_combined_tools
from tax_agent.datasource import UserProfile
from tax_agent.mock_api.deduction import CardSpending
from tax_agent.mock_api.medical import MedicalExpenses
from tax_agent.mock_api.rent import RentPayments
from tax_agent.mock_api.source import MockDataSource
from tax_agent.transactions import (
    Transaction, TransactionDataSource, card_spending, medical_expenses, rent_payments, unclassified,
)

# Fictional rows only: the repository is public, real transactions never go into it.


def tx(amount, method="credit", category="general", day="2025-03-10", merchant="shop", **extra):
    return Transaction(day, amount, merchant, method, category, **extra)


# user_a of the mock data (the NTS guide example, 5,300,000 KRW) written as individual purchases
USER_A_AS_TRANSACTIONS = [
    tx(20_000_000, "credit"), tx(10_000_000, "credit", day="2025-09-02"),
    tx(2_000_000, "debit"), tx(5_000_000, "cash_receipt"),
    tx(3_000_000, "credit", "market"), tx(2_000_000, "debit", "transit"), tx(1_000_000, "cash_receipt", "culture"),
]


def test_transactions_add_up_to_the_inputs_of_the_mock_user_and_give_the_same_deduction():
    assert card_spending(USER_A_AS_TRANSACTIONS, 2025) == MockDataSource().get_card_spending("user_a", 2025)
    source = TransactionDataSource(UserProfile("me", 68_000_000), USER_A_AS_TRANSACTIONS)
    result = build_combined_tools(source)["compute_deduction"]("me", "card")
    assert result["deduction"] == 5_300_000  # through the same tool and calculator the agent uses


def test_special_categories_ignore_the_payment_method_and_are_not_counted_twice():
    spending = card_spending([tx(100, "credit", "market"), tx(200, "debit", "transit"), tx(300, "credit")], 2025)
    assert spending == CardSpending(credit=300, market=100, transit=200)


def test_a_cancellation_reduces_its_bucket_and_never_makes_it_negative():
    assert card_spending([tx(500_000), tx(-200_000, day="2025-04-01")], 2025).credit == 300_000
    assert card_spending([tx(100_000), tx(-300_000)], 2025).credit == 0


def test_only_the_asked_year_counts():
    rows = [tx(1_000, day="2024-12-31"), tx(2_000, day="2025-01-01"), tx(4_000, day="2025-12-31"), tx(8_000, day="2026-01-01")]
    assert card_spending(rows, 2025).credit == 6_000


def test_excluded_and_unclassified_count_nowhere_and_the_unclassified_are_listed_largest_first():
    rows = [tx(10_000), tx(900_000, category="excluded", merchant="insurance"),
            tx(30_000, category="unclassified", merchant="a"), tx(-70_000, category="unclassified", merchant="b")]
    assert card_spending(rows, 2025).credit == 10_000
    assert [t.merchant for t in unclassified(rows, 2025)] == ["b", "a"]  # by size, a refund counts by its absolute value
    assert unclassified(rows, 2024) == []


def test_rent_and_medical_buckets():
    rows = [tx(700_000, "other", "rent", merchant="landlord"), tx(700_000, "other", "rent", day="2025-04-10"),
            tx(300_000, "credit", "medical"), tx(80_000, "debit", "medical", for_whom="other"),
            tx(50_000, "credit", "medical", day="2024-05-05")]
    assert rent_payments(rows, 2025) == RentPayments(annual_rent=1_400_000)
    assert medical_expenses(rows, 2025, reimbursed_self_senior=100_000) == MedicalExpenses(
        self_senior=300_000, other=80_000, reimbursed_self_senior=100_000, reimbursed_other=0
    )


def test_the_data_source_knows_one_taxpayer_and_gives_the_profile_the_taxpayer_supplied():
    source = TransactionDataSource(UserProfile("me", None, homeless_household_head=True), [])
    assert source.get_user_profile("me").gross_salary is None  # the agent will ask for it
    for call in (source.get_user_profile, lambda u: source.get_card_spending(u, 2025)):
        with pytest.raises(KeyError):
            call("someone-else")


def test_malformed_rows_are_rejected_when_an_adapter_builds_them():
    for bad in ({"day": "2025/03/10"}, {"method": "bitcoin"}, {"category": "food"}, {"for_whom": "cat"}):
        with pytest.raises(ValueError):
            tx(1_000, **bad)


def test_culture_usage_of_a_high_earner_is_counted_as_regular_usage_by_payment_method():
    rows = [tx(100_000, "debit", "culture"), tx(50_000, "credit", "culture"), tx(70_000, "cash_receipt", "culture")]
    assert card_spending(rows, 2025) == CardSpending(culture=220_000)  # salary unknown: the culture bucket
    assert card_spending(rows, 2025, gross_salary=70_000_000) == CardSpending(culture=220_000)  # the line itself is still <= 70M
    assert card_spending(rows, 2025, gross_salary=70_000_001) == CardSpending(credit=50_000, debit=100_000, cash_receipt=70_000)
