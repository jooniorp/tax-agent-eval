import pytest

from tax_agent.mock_api.deduction import CardSpending, calc_card_deduction


def man(n: int) -> int:
    """10,000 KRW units, the unit the NTS guide examples use."""
    return n * 10_000


def test_nts_official_example():
    # NTS guide pp. 136-138: salary 68M. Credit 32M (incl. transit 2M),
    # cash receipt 8M (incl. market 3M), debit 3M (incl. culture 1M).
    spending = CardSpending(
        credit=man(3200 - 200),
        debit=man(300 - 100),
        cash_receipt=man(800 - 300),
        market=man(300),
        transit=man(200),
        culture=man(100),
    )
    result = calc_card_deduction(man(6800), spending)
    assert result.minimum_usage == man(1700)
    assert result.before_limit == man(635)  # forgetting the 25% threshold gives 890
    assert result.deduction == man(530)  # 3M base cap + 2.3M extra for special usage


def test_usage_below_minimum_gets_nothing():
    result = calc_card_deduction(man(5000), CardSpending(credit=man(1000)))  # threshold 12.5M
    assert result.deduction == 0


def test_threshold_spills_over_to_higher_rate_tier():
    # Threshold 10M: credit 4M absorbs 4M, debit absorbs the other 6M -> 4M left at 30%.
    result = calc_card_deduction(man(4000), CardSpending(credit=man(400), debit=man(1000)))
    assert result.deduction == man(120)


def test_base_limit_without_special_usage():
    result = calc_card_deduction(man(5000), CardSpending(debit=man(5000)))
    assert result.before_limit == man(1125)
    assert result.deduction == man(300)  # no extra limit without market/transit/culture


def test_lower_limit_above_70m():
    result = calc_card_deduction(man(8000), CardSpending(debit=man(6000)))
    assert result.deduction == man(250)


def test_culture_above_70m_is_rejected():
    with pytest.raises(ValueError):
        calc_card_deduction(man(8000), CardSpending(culture=man(100)))
