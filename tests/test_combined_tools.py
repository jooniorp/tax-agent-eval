import pytest

from tax_agent.combined_tools import build_combined_tools
from tax_agent.mock_api.source import MockDataSource

compute = build_combined_tools(MockDataSource())["compute_deduction"]


def test_one_call_gives_the_same_amounts_as_the_step_by_step_tools():
    assert compute("user_a", "card")["deduction"] == 5_300_000  # the NTS official example
    assert compute("user_j", "rent")["deduction"] == 1_020_000
    assert compute("user_k", "medical")["deduction"] == 570_000
    two_topics = (compute("user_s", "card")["deduction"], compute("user_s", "rent")["deduction"])
    assert two_topics == (2_250_000, 1_020_000)


def test_the_result_carries_what_the_model_needs_to_explain_it():
    result = compute("user_a", "card")
    assert result["gross_salary"] == 68_000_000 and result["base_limit"] == 3_000_000
    assert result["extra_deduction"] == 2_300_000 and result["inputs"]["transit"] == 2_000_000
    rent = compute("user_b", "rent")
    assert (rent["eligible"], rent["reason"], rent["deduction"]) == (False, "total_salary_over_80m", 0)
    assert rent["inputs"] == {"annual_rent": 8_400_000, "homeless_household_head": True}


def test_missing_information_is_reported_so_the_agent_can_ask_and_call_again():
    first = compute("user_d", "card")  # the salary is not on file
    assert first["missing"] == ["gross_salary"] and "deduction" not in first
    assert compute("user_d", "card", gross_salary=68_000_000)["deduction"] == 5_300_000
    rent = compute("user_p", "rent")  # the home-ownership flag is not on file
    assert rent["missing"] == ["homeless_household_head"] and "deduction" not in rent
    assert compute("user_p", "rent", homeless_household_head=True)["deduction"] == 1_190_000


def test_a_value_the_user_gave_overrides_the_profile_and_errors_are_plain_exceptions():
    assert compute("user_a", "card", gross_salary=50_000_000)["gross_salary"] == 50_000_000
    with pytest.raises(KeyError):
        compute("user_z", "card")  # unknown user
    with pytest.raises(ValueError):
        compute("user_a", "education")  # not supported
