"""The tax rules as plain text, for the no-calculator control (docs/decisions.md, 2026-11-14).

The same rules are implemented in mock_api/deduction.py, rent.py and medical.py. The control agent
gets this text instead of the calc_deduction tool, so any mistake it makes is a mistake in applying
known rules (arithmetic and bookkeeping), not in knowing them. tests/test_rules.py checks that the
numbers in this text match the constants in the code.
"""

RULES_TEXT = """\
[Card / cash-receipt income deduction]
- Usage rates: credit card 15%; debit or prepaid card 30%; cash receipt 30%; traditional market 40%; public transit 40%; culture 30% (culture counts only when total salary is 70,000,000 or less).
- get_card_spending reports the six categories separately, without overlap.
- Minimum usage = 25% of total salary. Subtract the minimum usage from the usage amounts in this order: (1) credit card, (2) debit + cash receipt + culture, (3) market + transit. Only the amount left after the subtraction earns a deduction.
- before_limit = the sum of (remaining amount x rate) over the six categories.
- Basic cap: 3,000,000 (2,500,000 if total salary is above 70,000,000).
- Extra cap for market / transit / culture: at most 3,000,000 (2,000,000 if total salary is above 70,000,000). Extra = min(before_limit - basic cap (0 if negative), market x 40% + transit x 40% + culture x 30%, extra cap).
- Deduction = min(before_limit, basic cap) + extra.

[Monthly rent tax credit]
- Eligible only if total salary is 80,000,000 or less AND the person is the head of a household that owns no home (homeless_household_head). Otherwise there is no credit at all.
- Rent counted: at most 10,000,000 a year. Rate: 17% if total salary is 55,000,000 or less, otherwise 15%. Credit = counted rent x rate.

[Medical expense tax credit]
- Two groups: self_senior (the taxpayer, age 65+, age 6 or under, disabled: no cap) and other (other dependents: capped at 7,000,000).
- Net amount of each group = paid - insurance payouts received (reimbursed_*), at least 0.
- Threshold = 3% of total salary. Take the threshold from the "other" group first: eligible_other = min(max(net_other - threshold, 0), 7,000,000). The part of the threshold still left (threshold - net_other, at least 0) is taken from self_senior: eligible_self = max(net_self - left, 0).
- Credit = (eligible_self + eligible_other) x 15%.

Rounding: round down to whole won every time a percentage of an amount is taken: the minimum usage (25% of salary), the medical threshold (3% of salary), before_limit and the extra-cap amount (add up amount x percent over all the categories first, then divide by 100 and round down once), the rent credit and the medical credit.
"""
