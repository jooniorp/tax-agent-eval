from langchain_core.messages import AIMessage

from tax_agent.agent import ask, build_agent, reasoning_tokens
from tax_agent.evaluation.timing import CallTimer
from tax_agent.mock_api.source import MockDataSource
from test_agent import ScriptedModel, call  # tests/ is on sys.path


def test_the_timer_counts_every_model_call_and_adds_up_the_tool_time():
    timer = CallTimer()
    model = ScriptedModel(messages=iter([
        call("compute_deduction", {"user_id": "user_a", "item": "card"}, "1"),
        AIMessage(content="끝"),
    ]))
    ask(build_agent(MockDataSource(), "user_a", model=model), "안녕", callbacks=[timer])
    assert len(timer.model_seconds) == 2  # the call that asked for the tool and the final answer
    assert all(s >= 0 for s in timer.model_seconds) and timer.tool_seconds >= 0
    assert timer._started == {}  # nothing is left open


def test_reasoning_tokens_are_read_from_the_usage_details_and_missing_ones_count_as_zero():
    usage = {"input_tokens": 100, "output_tokens": 80, "total_tokens": 180, "output_token_details": {"reasoning": 30}}
    messages = [
        AIMessage("a", usage_metadata=usage),
        AIMessage("b", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}),  # no details
        AIMessage("c"),  # no usage at all
    ]
    assert reasoning_tokens(messages) == 30
