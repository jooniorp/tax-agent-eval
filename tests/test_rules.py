from tax_agent.mock_api.deduction import HIGH_INCOME_LINE, MIN_USAGE_PERCENT
from tax_agent.mock_api.medical import OTHER_DEPENDENTS_CAP, RATE_PERCENT, THRESHOLD_PERCENT
from tax_agent.mock_api.rent import ANNUAL_RENT_LIMIT, HIGH_RATE_SALARY, MAX_SALARY
from tax_agent.rules import RULES_TEXT


def test_the_rules_text_uses_the_same_numbers_as_the_calculators():
    # The no-calculator control is only a fair test if it is given the rules the calculators implement.
    for number in (MAX_SALARY, HIGH_RATE_SALARY, ANNUAL_RENT_LIMIT, OTHER_DEPENDENTS_CAP, HIGH_INCOME_LINE,
                   3_000_000, 2_500_000, 2_000_000):  # the last three are the card caps written in the function
        assert f"{number:,}" in RULES_TEXT, number
    for percent in (MIN_USAGE_PERCENT, THRESHOLD_PERCENT, RATE_PERCENT, 17, 30, 40):
        assert f"{percent}%" in RULES_TEXT, percent
