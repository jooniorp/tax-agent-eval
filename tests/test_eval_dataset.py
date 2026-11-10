import json
from pathlib import Path

from tax_agent.evaluation.scoring import BUNDLES, CHECKS
from tax_agent.mock_api.deduction import calc_card_deduction
from tax_agent.mock_api.medical import calc_medical_credit
from tax_agent.mock_api.rent import calc_rent_credit
from tax_agent.mock_api.source import MockDataSource

DATASET = Path(__file__).resolve().parent.parent / "evals" / "dataset.jsonl"
ITEMS = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
SOURCE = MockDataSource()


def ground_truth(item: dict) -> dict:
    """What the rule-based calculators say for the item's user: the real gold answer."""
    user, year = item["user"], 2025
    profile = SOURCE.get_user_profile(user)
    gross = profile.gross_salary or item.get("given_gross_salary")
    if item["topic"] == "card":
        return {"amount": calc_card_deduction(gross, SOURCE.get_card_spending(user, year)).deduction}
    if item["topic"] == "rent":
        credit = calc_rent_credit(gross, profile.homeless_household_head, SOURCE.get_rent_payments(user, year))
        return {"amount": credit.deduction} if credit.eligible else {"eligible": False}
    return {"amount": calc_medical_credit(gross, SOURCE.get_medical_expenses(user, year)).deduction}


def test_dataset_has_ten_unique_items_covering_every_planned_kind():
    assert len(ITEMS) == 10
    assert len({i["id"] for i in ITEMS}) == 10
    topics = [i["topic"] for i in ITEMS]
    assert (topics.count("card"), topics.count("rent"), topics.count("medical")) == (6, 2, 2)  # ask-01 is a card item
    assert {i["kind"] for i in ITEMS} == {"amount", "zero", "ineligible", "explain", "ask_back"}


def test_every_expected_answer_equals_what_the_calculators_say():
    for item in ITEMS:
        assert item["expected"] == ground_truth(item), item["id"]


def test_checks_only_use_known_types_and_bundles():
    for item in ITEMS:
        for entry in item["checks"]:
            if isinstance(entry, str):
                assert entry in BUNDLES, (item["id"], entry)
            else:
                assert entry["type"] in CHECKS, (item["id"], entry)


def test_the_ask_back_user_really_has_no_salary_on_file():
    item = next(i for i in ITEMS if i["kind"] == "ask_back")
    assert SOURCE.get_user_profile(item["user"]).gross_salary is None
    assert item["given_gross_salary"] == 68_000_000 and len(item["turns"]) == 2


def test_forbidden_rent_amounts_are_the_credits_a_careless_answer_would_state():
    item = next(i for i in ITEMS if i["id"] == "rent-01")
    rent = SOURCE.get_rent_payments(item["user"], 2025).annual_rent
    forbidden = next(c for c in item["checks"] if isinstance(c, dict) and c["type"] == "answer_not_contains_amount")
    assert set(forbidden["amounts"]) == {rent * 15 // 100, rent * 17 // 100}
