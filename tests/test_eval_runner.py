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
    result = runner.run_item(CARD_01, ScriptedModel(messages=iter(card_01_script("카드 소득공제는 530만 원입니다."))))
    assert result["passed"] and result["answer_ok"] and result["rules_ok"]
    assert result["tool_calls"] == ["get_user_profile", "get_card_spending", "calc_deduction"]


def test_run_item_fails_a_wrong_answer_and_keeps_the_reason():
    result = runner.run_item(CARD_01, ScriptedModel(messages=iter(card_01_script("카드 소득공제는 630만 원입니다."))))
    assert not result["passed"] and not result["answer_ok"]
    assert any(c["name"] == "answer_uses_tool_amount" and not c["ok"] for c in result["checks"])


def test_a_crashing_item_is_a_failed_item_not_a_crashed_eval():
    result = runner.run_item(CARD_01, ScriptedModel(messages=iter([])))  # the model has nothing to say
    assert result["passed"] is False and result["error"]


def test_report_and_summary_numbers():
    ok = {"id": "a", "kind": "amount", "topic": "card", "seconds": 1.0, "tokens_in": 100, "tokens_out": 10, "error": None,
          "tool_calls": ["calc_deduction"], "answer": "정답", "answer_ok": True, "rules_ok": True, "passed": True,
          "answer_detail": "", "checks": []}
    bad = {**ok, "id": "b", "answer_ok": False, "passed": False, "answer_detail": "expected 1; answer has [2]"}
    summary = runner.summarize([ok, bad])
    assert (summary["items"], summary["answer_ok"], summary["rules_ok"], summary["passed"]) == (2, 1, 2, 1)
    report = runner.format_report([ok, bad], "claude-haiku-5-5")
    assert "전체 통과 1/2" in report and "정답 불일치: expected 1" in report


def test_main_prints_the_cost_estimate_first_and_saves_the_result(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)  # evals/results/ is relative to the working directory
    monkeypatch.setattr(runner, "get_model", lambda name=None: ScriptedModel(messages=iter(card_01_script("카드 소득공제는 5,300,000원입니다."))))
    runner.main(["--dataset", str(DATASET), "--only", "card-01"])
    out = capsys.readouterr().out
    assert out.index("estimated cost") < out.index("PASS")  # the estimate comes before the first question
    assert "전체 통과 1/1" in out
    saved = json.loads(next((tmp_path / "evals" / "results").glob("*.json")).read_text(encoding="utf-8"))
    assert saved["summary"]["passed"] == 1 and saved["model"] == "claude-haiku-5-5"
