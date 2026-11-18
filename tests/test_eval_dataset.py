import json
from pathlib import Path

import pytest

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
    if item["kind"] == "unsupported":
        return {"unsupported": True}
    if item["kind"] == "not_found":
        with pytest.raises(KeyError):
            SOURCE.get_user_profile(user)  # the user really is missing from the data source
        return {"not_found": True}
    profile = SOURCE.get_user_profile(user)
    gross = profile.gross_salary or item.get("given_gross_salary")
    homeless = profile.homeless_household_head
    if homeless is None:
        homeless = item.get("given_homeless")
    if item["topic"] == "multi":
        card = calc_card_deduction(gross, SOURCE.get_card_spending(user, year)).deduction
        rent = calc_rent_credit(gross, homeless, SOURCE.get_rent_payments(user, year)).deduction
        return {"amounts": [card, rent]}
    if item["topic"] == "card":
        return {"amount": calc_card_deduction(gross, SOURCE.get_card_spending(user, year)).deduction}
    if item["topic"] == "rent":
        credit = calc_rent_credit(gross, homeless, SOURCE.get_rent_payments(user, year))
        return {"amount": credit.deduction} if credit.eligible else {"eligible": False}
    return {"amount": calc_medical_credit(gross, SOURCE.get_medical_expenses(user, year)).deduction}


def test_dataset_has_two_groups_of_unique_items_covering_every_planned_kind():
    assert len(ITEMS) == 32
    assert len({i["id"] for i in ITEMS}) == 32
    groups = [i.get("group", "basic") for i in ITEMS]
    assert (groups.count("basic"), groups.count("exact"), groups.count("sweep")) == (20, 6, 6)
    topics = [i["topic"] for i in ITEMS]
    assert {t: topics.count(t) for t in set(topics)} == {"card": 16, "rent": 7, "medical": 6, "edge": 2, "multi": 1}
    assert {i["kind"] for i in ITEMS} == {
        "amount", "zero", "ineligible", "explain", "ask_back", "false_premise", "unsupported", "not_found", "multi",
    }


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


def test_the_ask_back_users_really_have_the_information_missing():
    by_id = {i["id"]: i for i in ITEMS}
    salary_case, rent_case = by_id["ask-01"], by_id["rent-04"]
    assert SOURCE.get_user_profile(salary_case["user"]).gross_salary is None
    assert salary_case["given_gross_salary"] == 68_000_000 and len(salary_case["turns"]) == 2
    assert SOURCE.get_user_profile(rent_case["user"]).homeless_household_head is None
    assert rent_case["given_homeless"] is True and len(rent_case["turns"]) == 2


def test_the_length_sweep_changes_only_the_number_of_categories():
    sweep = [i for i in ITEMS if i.get("group") == "sweep"]
    assert [i["id"] for i in sweep] == [f"sweep-{k}" for k in range(1, 7)]
    assert len({i["turns"][0] for i in sweep}) == 1  # the same question every time
    for k, item in enumerate(sweep, start=1):
        spending = SOURCE.get_card_spending(item["user"], 2025)
        used = [v for v in vars(spending).values() if v]
        assert len(used) == k and sum(used) == 15_271_830, item["id"]  # k categories, one fixed total
        assert SOURCE.get_user_profile(item["user"]).gross_salary == 40_000_000
        assert item["expected"]["amount"] < 3_000_000, item["id"]  # under the cap, so every category matters
        assert item["expected"]["amount"] % 1000 != 0, item["id"]  # and the answer is not round


def test_an_ask_back_item_forbids_exactly_the_final_amount_in_its_first_reply():
    for item in (i for i in ITEMS if i["kind"] == "ask_back"):
        forbidden = next(c for c in item["checks"] if isinstance(c, dict) and c["type"] == "answer_not_contains_amount")
        assert forbidden["turn"] == 1 and forbidden["amounts"] == [item["expected"]["amount"]], item["id"]


def test_forbidden_rent_amounts_are_the_credits_a_careless_answer_would_state():
    item = next(i for i in ITEMS if i["id"] == "rent-01")
    rent = SOURCE.get_rent_payments(item["user"], 2025).annual_rent
    forbidden = next(c for c in item["checks"] if isinstance(c, dict) and c["type"] == "answer_not_contains_amount")
    assert set(forbidden["amounts"]) == {rent * 15 // 100, rent * 17 // 100}


def test_the_boundary_and_cap_items_really_sit_on_the_boundary():
    by_id = {i["id"]: i for i in ITEMS}
    assert SOURCE.get_user_profile(by_id["rent-03"]["user"]).gross_salary == 80_000_000  # "80M or less" is eligible
    assert SOURCE.get_rent_payments(by_id["rent-03"]["user"], 2025).annual_rent > 10_000_000  # the rent cap bites
    card = SOURCE.get_card_spending(by_id["card-07"]["user"], 2025)
    result = calc_card_deduction(SOURCE.get_user_profile(by_id["card-07"]["user"]).gross_salary, card)
    assert result.extra_deduction == 3_000_000 and result.before_limit - result.base_limit > result.extra_deduction  # extra cap bites
    medical = SOURCE.get_medical_expenses(by_id["med-03"]["user"], 2025)
    assert medical.other - medical.reimbursed_other - 1_200_000 > 7_000_000  # the 7M cap on "other" bites
    # the exact-amount group: the rate boundary is one won wide, and the figures are not round
    assert SOURCE.get_user_profile(by_id["exact-03"]["user"]).gross_salary == 55_000_000  # 17% (55M or less)
    assert SOURCE.get_user_profile(by_id["exact-04"]["user"]).gross_salary == 55_000_001  # 15%
    for item in (i for i in ITEMS if i.get("group") == "exact"):
        assert item["expected"]["amount"] % 1000 != 0, item["id"]  # not a round gold answer
