import pytest

from tax_agent.mock_api.medical import MedicalExpenses, calc_medical_credit
from tax_agent.mock_api.rent import RentPayments, calc_rent_credit


def man(n: int) -> int:
    return n * 10_000


# --- rent (NTS guide pp. 202-203) ---

def test_rent_rate_is_17_percent_up_to_55m_and_15_percent_above():
    assert calc_rent_credit(man(5500), True, RentPayments(man(600))).deduction == man(102)  # 17%
    assert calc_rent_credit(man(5501), True, RentPayments(man(600))).deduction == man(90)  # 15%


def test_rent_is_capped_at_10m_a_year():
    assert calc_rent_credit(man(4000), True, RentPayments(man(1500))).deduction == man(170)  # 10M x 17%


def test_rent_not_eligible_over_80m_but_80m_itself_is_fine():
    assert calc_rent_credit(man(8000), True, RentPayments(man(840))).deduction == man(126)  # 8.4M x 15%
    over = calc_rent_credit(man(8500), True, RentPayments(man(840)))
    assert (over.eligible, over.reason, over.deduction) == (False, "total_salary_over_80m", 0)


def test_rent_requires_a_homeless_household_head_and_asks_when_unknown():
    result = calc_rent_credit(man(5000), False, RentPayments(man(600)))
    assert (result.eligible, result.reason) == (False, "not_homeless_household_head")
    with pytest.raises(ValueError):
        calc_rent_credit(man(5000), None, RentPayments(man(600)))


# --- medical (NTS guide pp. 173, 176) ---

def test_medical_below_threshold_after_insurance_payout_is_zero():
    # eval item 3: 1.2M paid for self, 0.3M paid back by insurance -> 0.9M, under 3% of 50M = 1.5M
    result = calc_medical_credit(man(5000), MedicalExpenses(self_senior=man(120), reimbursed_self_senior=man(30)))
    assert (result.threshold, result.net_expenses, result.deduction) == (man(150), man(90), 0)


def test_medical_self_expenses_above_threshold():
    result = calc_medical_credit(man(4000), MedicalExpenses(self_senior=man(500)))
    assert result.eligible_expenses == man(380)  # 5M - 3% of 40M
    assert result.deduction == man(57)  # 15%


def test_medical_threshold_comes_off_the_capped_group_first():
    # guide case "ga": other >= threshold -> self + min(other - threshold, 7M)
    result = calc_medical_credit(man(5000), MedicalExpenses(self_senior=man(100), other=man(200)))
    assert result.eligible_expenses == man(100) + man(50)  # other 2M - 1.5M = 0.5M
    # guide case "na": other < threshold -> the rest of the threshold comes off self
    result = calc_medical_credit(man(5000), MedicalExpenses(self_senior=man(100), other=man(100)))
    assert result.eligible_expenses == man(100) - man(50)  # threshold 1.5M - other 1M = 0.5M off self


def test_medical_other_dependents_are_capped_at_7m():
    result = calc_medical_credit(man(2000), MedicalExpenses(other=man(2000)))
    assert result.eligible_expenses == man(700)  # (20M - 0.6M) would be 19.4M, capped


def test_medical_rejects_negative_amounts():
    with pytest.raises(ValueError):
        calc_medical_credit(man(5000), MedicalExpenses(self_senior=-1))
