import json
from dataclasses import asdict
from pathlib import Path

from langchain_core.messages import AIMessage

from tax_agent.evaluation import runner
from tax_agent.mock_api.source import MockDataSource
from test_agent import ScriptedModel, call  # tests/ is on sys.path

DATASET = Path(__file__).resolve().parent.parent / "evals" / "dataset.jsonl"
CARD_01 = next(i for i in runner.load_dataset(str(DATASET)) if i["id"] == "card-01")


def card_01_script(answer: str) -> list:
    spending = asdict(MockDataSource().get_card_spending("user_a", 2025))
    return [
        AIMessage("", tool_calls=[
            {"name": "get_user_profile", "args": {"user_id": "user_a"}, "id": "1"},
            {"name": "get_card_spending", "args": {"user_id": "user_a", "year": 2025}, "id": "2"},
        ]),
        call("calc_deduction", {"item": "card", "gross_salary": 68_000_000, "data": spending}, "3"),
        AIMessage(answer),
    ]


def test_run_item_scores_a_good_run_as_passed():
    result = runner.run_item(CARD_01, ScriptedModel(messages=iter(card_01_script("카드 소득공제는 530만 원입니다."))), variant="calculator")
    assert result["passed"] and result["answer_ok"] and result["rules_ok"]
    assert result["tool_calls"] == ["get_user_profile", "get_card_spending", "calc_deduction"]


def test_run_item_fails_a_wrong_answer_and_keeps_the_reason():
    result = runner.run_item(CARD_01, ScriptedModel(messages=iter(card_01_script("카드 소득공제는 630만 원입니다."))), variant="calculator")
    assert not result["passed"] and not result["answer_ok"]
    assert any(c["name"] == "answer_uses_tool_amount" and not c["ok"] for c in result["checks"])


def test_a_crashing_item_is_a_failed_item_not_a_crashed_eval():
    result = runner.run_item(CARD_01, ScriptedModel(messages=iter([])))  # the model has nothing to say
    assert result["passed"] is False and result["error"]


def test_report_and_summary_numbers():
    ok = {"id": "a", "kind": "amount", "topic": "card", "seconds": 1.0, "tokens_in": 100, "tokens_out": 10, "error": None,
          "variant": "calculator", "tool_calls": ["calc_deduction"], "answer": "정답", "answer_ok": True,
          "rules_ok": True, "passed": True, "answer_detail": "", "checks": [{"name": "tool_called", "ok": True, "detail": "ok"}]}
    bad = {**ok, "id": "b", "answer_ok": False, "passed": False, "answer_detail": "expected 1; answer has [2]"}
    no_rules = {**ok, "id": "c", "checks": []}  # an item with no rule check: rules do not apply to it
    summary = runner.summarize([ok, bad, no_rules])
    assert (summary["items"], summary["answer_ok"], summary["passed"]) == (3, 2, 2)
    assert (summary["rules_ok"], summary["rules_n"]) == (2, 2)  # the item without checks is not counted
    report = runner.format_report([ok, bad, no_rules], "claude-haiku-5-5")
    assert "전체 통과 2/3" in report and "규칙 2/2" in report and "정답 불일치: expected 1" in report
    assert "에이전트(계산기 툴)" in report
    assert "[그룹별]" not in report  # one group only: no per-group line
    two_groups = runner.format_report([ok, {**bad, "group": "exact"}], "claude-haiku-5-5")
    assert "basic 정답 1/1 · 통과 1/1 | exact 정답 0/1 · 통과 0/1" in two_groups


def test_load_dataset_can_pick_one_group():
    exact = runner.load_dataset(str(DATASET), group="exact")
    assert len(exact) == 6 and all(i["id"].startswith("exact-") for i in exact)
    assert len(runner.load_dataset(str(DATASET), group="basic")) == 20
    assert len(runner.load_dataset(str(DATASET), group="sweep")) == 6


def test_run_item_records_reasoning_tokens_model_calls_and_the_split_of_the_time():
    def usage(out, reasoning):
        return {"input_tokens": 100, "output_tokens": out, "total_tokens": 100 + out, "output_token_details": {"reasoning": reasoning}}

    script = [AIMessage(m.content, tool_calls=m.tool_calls, usage_metadata=usage(60, 20))
              for m in card_01_script("카드 소득공제는 530만 원입니다.")]
    result = runner.run_item(CARD_01, ScriptedModel(messages=iter(script)), variant="calculator")
    assert (result["model_calls"], result["tokens_out"], result["reasoning_tokens"]) == (3, 180, 60)
    assert result["model_seconds"] >= 0 and result["tool_seconds"] >= 0
    summary = runner.summarize([result])
    assert (summary["reasoning_tokens"], summary["model_calls"]) == (60, 3)
    assert "추론 토큰 60(출력의 33%) · 모델 호출 3회" in runner.format_report([result], "claude-haiku-5-5")


def test_the_control_variant_is_labelled_and_skips_the_calculator_checks():
    first_step, _final = card_01_script("카드 소득공제는 530만 원입니다.")[0], None
    model = ScriptedModel(messages=iter([first_step, AIMessage("카드 소득공제는 530만 원입니다.")]))
    result = runner.run_item(CARD_01, model, variant="no_calculator")
    assert result["variant"] == "no_calculator" and result["checks"] == []
    assert result["answer_ok"] and result["passed"]
    assert "calc_deduction" not in model.bound
    assert "대조군(계산기 없음)" in runner.format_report([result], "claude-haiku-5-5")


def test_main_prints_the_cost_estimate_first_and_saves_the_result(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)  # evals/results/ is relative to the working directory
    monkeypatch.setattr(runner, "get_model", lambda name=None: ScriptedModel(messages=iter(card_01_script("카드 소득공제는 5,300,000원입니다."))))
    runner.main(["--dataset", str(DATASET), "--only", "card-01", "--variant", "calculator"])
    out = capsys.readouterr().out
    assert out.index("estimated cost") < out.index("PASS")  # the estimate comes before the first question
    assert "전체 통과 1/1" in out
    saved = json.loads(next((tmp_path / "evals" / "results").glob("*.json")).read_text(encoding="utf-8"))
    assert saved["summary"]["passed"] == 1 and saved["model"] == "claude-haiku-5-5"


def test_the_combined_variant_runs_with_one_tool_and_is_scored_for_it():
    answer = "카드 소득공제는 530만 원입니다."
    model = ScriptedModel(messages=iter([call("compute_deduction", {"user_id": "user_a", "item": "card"}, "1"), AIMessage(answer)]))
    result = runner.run_item(CARD_01, model, variant="combined")
    assert result["variant"] == "combined" and result["tool_calls"] == ["compute_deduction"]
    assert result["passed"] and [c["name"] for c in result["checks"]] == ["tool_called", "answer_uses_tool_amount"]
    assert model.bound == ["compute_deduction"] and result["model_calls"] == 2
    assert "통합 툴" in runner.format_report([result], "claude-haiku-5-5")


def test_the_eval_runs_the_default_agent_unless_told_otherwise(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    model = ScriptedModel(messages=iter([call("compute_deduction", {"user_id": "user_a", "item": "card"}, "1"), AIMessage("카드 소득공제는 5,300,000원입니다.")]))
    monkeypatch.setattr(runner, "get_model", lambda name=None: model)
    runner.main(["--dataset", str(DATASET), "--only", "card-01"])
    assert "variant=combined" in capsys.readouterr().out and model.bound == ["compute_deduction"]
    saved = json.loads(next((tmp_path / "evals" / "results").glob("*.json")).read_text(encoding="utf-8"))
    assert saved["variant"] == "combined" and saved["summary"]["passed"] == 1


def test_every_variant_has_a_token_estimate_and_the_default_one_is_the_cheapest():
    from tax_agent.agent import DEFAULT_VARIANT, VARIANTS

    assert set(runner.EST_TOKENS_PER_TURN) == set(VARIANTS)
    cost = {v: runner.estimate_cost_usd("claude-haiku-5-5", *runner.EST_TOKENS_PER_TURN[v]) for v in VARIANTS}
    assert cost[DEFAULT_VARIANT] == min(cost.values())
