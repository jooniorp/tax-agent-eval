from dataclasses import asdict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import Field

from tax_agent.agent import ask, build_agent
from tax_agent.llm import DEFAULT_MODEL, get_model
from tax_agent.mock_api.source import MockDataSource


class ScriptedModel(GenericFakeChatModel):
    """Replays scripted AI messages (no API call) and records what it was shown."""

    seen: list = Field(default_factory=list)
    model: str = "claude-haiku-5-5"  # the CLI reads the model name from the model object

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, *args, **kwargs)


def call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def run(script: list, question: str = "카드 공제 얼마야?"):
    model = ScriptedModel(messages=iter(script))
    agent = build_agent(MockDataSource(), "user_a", model=model)
    return model, ask(agent, question)


def test_agent_follows_the_tool_sequence_and_returns_the_answer():
    spending = asdict(MockDataSource().get_card_spending("user_a", 2025))
    model, messages = run([
        call("get_user_profile", {"user_id": "user_a"}, "1"),
        call("get_card_spending", {"user_id": "user_a", "year": 2025}, "2"),
        call("calc_deduction", {"item": "card", "gross_salary": 68_000_000, "spending": spending}, "3"),
        AIMessage(content="카드 소득공제는 530만원입니다."),
    ])
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    assert [m.name for m in tool_messages] == ["get_user_profile", "get_card_spending", "calc_deduction"]
    assert "5300000" in tool_messages[-1].content  # the calculator's result reached the model
    assert messages[-1].text == "카드 소득공제는 530만원입니다."
    assert "user_a" in model.seen[0][0].content  # system prompt carries the user id


def test_tool_errors_go_back_to_the_model_instead_of_crashing():
    _, messages = run([
        call("calc_deduction", {"item": "medical", "gross_salary": 68_000_000, "spending": {}}, "1"),
        AIMessage(content="의료비는 아직 지원하지 않습니다."),
    ])
    tool_message = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_message.status == "error" and "unsupported item" in tool_message.content
    assert messages[-1].text == "의료비는 아직 지원하지 않습니다."


def test_model_name_priority(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")  # constructing the client makes no request
    monkeypatch.delenv("TAX_AGENT_MODEL", raising=False)
    assert get_model().model == DEFAULT_MODEL
    monkeypatch.setenv("TAX_AGENT_MODEL", "from-env")
    assert get_model().model == "from-env"
    assert get_model("from-arg").model == "from-arg"
