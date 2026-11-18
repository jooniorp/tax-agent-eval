from collections import Counter

from tax_agent.classify import classify_one, classify_rules
from tax_agent.combined_tools import build_combined_tools
from tax_agent.evaluation import classify_eval
from tax_agent.simulation import MERCHANTS, simulate_user
from tax_agent.transactions import CATEGORIES, TransactionDataSource, card_spending


def test_the_same_seed_gives_the_same_user_and_other_seeds_give_other_users():
    a, b, c = simulate_user(7), simulate_user(7), simulate_user(8)
    assert a == b and a.transactions != c.transactions and a.profile.user_id == "sim_0007"


def test_the_truth_is_complete_and_refunds_follow_a_purchase():
    user = simulate_user(3)
    assert all(t.category in CATEGORIES and t.category != "unclassified" for t in user.transactions)
    assert all(t.year == 2025 and t.merchant for t in user.transactions)
    assert user.profile.gross_salary and 30_000_000 <= user.profile.gross_salary < 90_000_000
    refunds = [t for t in user.transactions if t.amount < 0]
    purchases = {(t.merchant, t.category, t.method) for t in user.transactions if t.amount > 0}
    assert refunds and all((t.merchant, t.category, t.method) in purchases for t in refunds)


def test_unlabeled_rows_keep_everything_but_the_category():
    user = simulate_user(5)
    rows = user.unlabeled()
    assert {t.category for t in rows} == {"unclassified"}
    assert [(t.date, t.amount, t.merchant, t.method) for t in rows] == [(t.date, t.amount, t.merchant, t.method) for t in user.transactions]


def test_when_every_merchant_name_is_clear_the_keyword_rules_are_always_right():
    result = classify_eval.evaluate(classify_rules, users=15, obscure_rate=0.0)
    assert result["right"] == result["rows"] and not result["confused"]
    assert all(e == 0 for errs in result["errors"].values() for e in errs)  # so every deduction is exact too


def test_a_name_that_says_nothing_makes_the_rules_wrong_and_they_err_toward_general_or_unclassified():
    result = classify_eval.evaluate(classify_rules, users=15, obscure_rate=0.3)
    assert result["right"] < result["rows"]
    assert {guess for (_, guess) in result["confused"]} <= {"general", "unclassified"}  # an unknown name is never invented into a special kind
    assert any(e != 0 for e in result["errors"]["card"])  # and that moves the card deduction


def test_classify_one_decides_by_the_first_keyword_and_leaves_an_unknown_transfer_unclassified():
    clear_bus = next(t for t in simulate_user(1, obscure_rate=0).transactions if t.merchant.endswith("버스 요금"))
    assert classify_one(clear_bus).category == "transit"  # "요금" would also say excluded; transit is checked first
    mystery_transfer = next(t for t in simulate_user(2, obscure_rate=1.0).transactions if t.merchant == "집주인 송금")
    assert classify_one(mystery_transfer).category == "unclassified"
    assert set(MERCHANTS) == {"general", "medical", "transit", "market", "culture", "excluded", "rent"}


def test_a_high_earner_with_culture_spending_still_gets_a_deduction():
    users = [simulate_user(s) for s in range(40)]
    high = [u for u in users if u.profile.gross_salary > 70_000_000 and any(t.category == "culture" for t in u.transactions)]
    assert high  # the seeds do contain such users
    for user in high[:5]:
        source = TransactionDataSource(user.profile, classify_rules(user.unlabeled()))
        result = build_combined_tools(source)["compute_deduction"](user.profile.user_id, "card")
        assert result["deduction"] >= 0 and result["inputs"]["culture"] == 0  # folded into the regular buckets
        assert card_spending(user.transactions, 2025).culture > 0  # while without the salary the culture bucket stays


def test_the_report_names_the_numbers_it_measured():
    text = classify_eval.format_report(classify_eval.evaluate(classify_rules, users=3), "rules")
    assert "분류기: rules" in text and "분류 정확도" in text and "공제액 오차" in text
    assert Counter(line.split()[0] for line in text.splitlines() if line.startswith("  ")).get("card") == 1


# --- the model as classifier (a scripted model stands in: no API call) ---

def scripted(*answers):
    from langchain_core.messages import AIMessage

    from test_agent import ScriptedModel  # tests/ is on sys.path

    usage = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
    return ScriptedModel(messages=iter([AIMessage(a, usage_metadata=usage) for a in answers]))


def rows_for_llm():
    from tax_agent.transactions import Transaction

    return [Transaction("2025-01-02", 1000, "한솔마트 3호점", "credit"), Transaction("2025-02-02", 2000, "한솔마트 17호점", "credit"),
            Transaction("2025-03-02", 3000, "봄날 건강센터", "debit")]


def test_the_llm_classifier_asks_once_per_distinct_merchant_and_remembers_the_answers():
    from tax_agent.classify import LLMClassifier

    model = scripted('Sure: {"1": "medical", "2": "general"}')  # one reply only: a second request would fail
    classifier = LLMClassifier(model)
    labeled = classifier(rows_for_llm())
    # the numbers are dropped from the names, so the three rows are two distinct lines, asked in sorted order
    assert [t.category for t in labeled] == ["general", "general", "medical"]
    prompt = model.seen[0][1].content
    assert len(prompt.splitlines()) == 2 and "한솔마트 호점 | credit" in prompt and "17" not in prompt
    assert [t.category for t in classifier(rows_for_llm())] == ["general", "general", "medical"]  # from memory
    assert (classifier.tokens_in, classifier.tokens_out) == (100, 10)


def test_a_broken_or_short_or_unknown_reply_leaves_the_rows_unclassified_instead_of_guessing():
    from tax_agent.classify import LLMClassifier

    for reply in ("I cannot do that", "{}", "[1, 2]", '{"1": "food", "2": "food"}', '{"3": "general"}'):
        labeled = LLMClassifier(scripted(reply))(rows_for_llm())
        assert {t.category for t in labeled} == {"unclassified"}, reply


def test_an_answer_belongs_to_its_line_number_so_a_skipped_line_does_not_shift_the_rest():
    from tax_agent.classify import LLMClassifier

    # lines in sorted order: 1 = 봄날 건강센터 | debit, 2 = 한솔마트 호점 | credit. The model answers line 2 only.
    labeled = LLMClassifier(scripted('{"2": "general"}'))(rows_for_llm())
    assert [t.category for t in labeled] == ["general", "general", "unclassified"]


def test_sim_eval_with_the_llm_classifier_prints_the_estimate_first(monkeypatch, capsys):
    from tax_agent.evaluation import classify_eval

    answers = ['{' + ", ".join(f'"{i}": "general"' for i in range(1, 51)) + '}'] * 4  # one reply per chunk of 50 lines, extra lines are ignored
    monkeypatch.setattr(classify_eval, "get_model", lambda name=None: scripted(*answers))
    classify_eval.main(["--classifier", "llm", "--users", "2"])
    out = capsys.readouterr().out
    assert out.index("estimated cost") < out.index("분류 정확도") and "llm:claude-haiku-5-5" in out and "토큰 in/out 200/20" in out  # 100 distinct merchants = two requests of 50 lines
