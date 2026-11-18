from langchain_core.messages import AIMessage

from tax_agent.classify import classify_rules
from tax_agent.evaluation import sim_agent
from tax_agent.evaluation.classify_eval import deductions
from tax_agent.simulation import simulate_user
from test_agent import ScriptedModel, call  # tests/ is on sys.path


def scripted_for(user, rows, *, say):
    """What a faithful agent does for every topic: one tool call, then the amount the tool will return."""
    tool = deductions(user, rows)
    script = []
    for i, topic in enumerate(sim_agent.TOPICS):
        script += [call("compute_deduction", {"user_id": user.profile.user_id, "item": topic}, str(i)),
                   AIMessage(say(topic, tool[topic]))]
    return ScriptedModel(messages=iter(script)), tool


def test_states_reads_an_amount_or_a_zero():
    assert sim_agent.states("카드 소득공제는 2,250,000원입니다", 2_250_000)
    assert not sim_agent.states("카드 소득공제는 2,250,000원입니다", 2_150_000)
    assert sim_agent.states("공제 대상이 아닙니다", 0) and sim_agent.states("공제액은 0원입니다", 0)
    assert not sim_agent.states("공제액은 120만 원입니다", 0)


def test_with_clear_names_a_faithful_agent_is_right_end_to_end_and_the_report_says_so():
    user = simulate_user(4, obscure_rate=0.0)
    rows = classify_rules(user.unlabeled())
    model, tool = scripted_for(user, rows, say=lambda topic, amount: f"{topic} 공제는 {amount:,}원입니다." if amount else "공제액은 0원입니다.")
    cases = sim_agent.run_cases(model, [user], [rows])
    assert [c.topic for c in cases] == list(sim_agent.TOPICS)
    assert all(c.faithful and c.end_to_end and c.tool == c.truth for c in cases)
    text = sim_agent.format_report(cases, "claude-haiku-5-5", "rules", 0.0)
    assert "종단 정확(답한 금액 = 정답 공제액):        3/3" in text and "분류 오류로 툴 값이 정답과 다른 질문:       0/3" in text


def test_a_wrong_classification_shows_up_as_a_tool_value_that_differs_from_the_truth_while_the_agent_stays_faithful():
    user = next(u for u in (simulate_user(s, obscure_rate=0.5) for s in range(30))
                if deductions(u, classify_rules(u.unlabeled()))["card"] != deductions(u, u.transactions)["card"])
    rows = classify_rules(user.unlabeled())
    model, tool = scripted_for(user, rows, say=lambda topic, amount: f"{topic} 공제는 {amount:,}원입니다." if amount else "공제액은 0원입니다.")
    cases = sim_agent.run_cases(model, [user], [rows], topics=sim_agent.TOPICS)
    card = next(c for c in cases if c.topic == "card")
    assert card.faithful and not card.end_to_end and card.tool != card.truth  # the layer to blame is the classifier
    assert "(그중 에이전트가 툴 값을 그대로 따름 " in sim_agent.format_report(cases, "claude-haiku-5-5", "rules", 0.5)


def test_an_agent_that_misreports_what_its_tool_said_is_unfaithful_and_listed():
    user = simulate_user(4, obscure_rate=0.0)
    rows = classify_rules(user.unlabeled())
    model, tool = scripted_for(user, rows, say=lambda topic, amount: "공제는 1,234원입니다.")
    cases = sim_agent.run_cases(model, [user], [rows])
    assert not any(c.faithful for c in cases if c.tool != 1234)
    assert "툴 값을 못 따른 질문" in sim_agent.format_report(cases, "claude-haiku-5-5", "rules", 0.0)


def test_main_prints_the_estimate_first(monkeypatch, capsys):
    user = simulate_user(0)
    model, _ = scripted_for(user, classify_rules(user.unlabeled()), say=lambda topic, amount: f"{amount:,}원")
    monkeypatch.setattr(sim_agent, "get_model", lambda name=None: model)
    sim_agent.main(["--users", "1"])
    out = capsys.readouterr().out
    assert out.index("estimated cost") < out.index("종단 정확") and "질문 3개" in out
