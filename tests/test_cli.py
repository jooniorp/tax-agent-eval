from dataclasses import asdict

from langchain_core.messages import AIMessage

from tax_agent import cli
from tax_agent.llm import estimate_cost_usd
from tax_agent.mock_api.source import MockDataSource
from test_agent import ScriptedModel  # tests/ is on sys.path


def ai(content="", tool_calls=(), tokens=(0, 0)) -> AIMessage:
    usage = {"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)}
    return AIMessage(content=content, tool_calls=list(tool_calls), usage_metadata=usage)


def test_estimate_cost_usd():
    assert estimate_cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == 0.60  # $0.10 + $0.50
    assert estimate_cost_usd("some-unknown-model", 1000, 1000) is None


def test_main_prints_tool_calls_answer_and_stats(monkeypatch, capsys):
    spending = asdict(MockDataSource().get_card_spending("user_a", 2025))
    script = [
        ai(tool_calls=[{"name": "get_user_profile", "args": {"user_id": "user_a"}, "id": "1"}], tokens=(100, 10)),
        ai(tool_calls=[{"name": "get_card_spending", "args": {"user_id": "user_a", "year": 2025}, "id": "2"}], tokens=(150, 10)),
        ai(tool_calls=[{"name": "calc_deduction", "args": {"item": "card", "gross_salary": 68_000_000, "spending": spending}, "id": "3"}], tokens=(300, 40)),
        ai("카드 소득공제는 530만원입니다.", tokens=(400, 30)),
    ]
    monkeypatch.setattr(cli, "get_model", lambda name=None: ScriptedModel(messages=iter(script)))

    cli.main(["카드 공제 얼마야?"])

    out = capsys.readouterr().out
    assert out.index("get_user_profile") < out.index("get_card_spending") < out.index("calc_deduction")
    assert "[answer]\n카드 소득공제는 530만원입니다." in out
    assert "tokens in/out=950/90" in out  # summed over all four model calls
    assert "cost=~$0.0001 (estimate)" in out  # (950 * 0.10 + 90 * 0.50) / 1M
