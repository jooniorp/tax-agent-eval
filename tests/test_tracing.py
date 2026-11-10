from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langfuse.span_filter import is_default_export_span
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tax_agent import tracing
from tax_agent.agent import ask, build_agent
from tax_agent.mock_api.source import MockDataSource
from test_agent import ScriptedModel, call  # tests/ is on sys.path


class RecordingHandler(BaseCallbackHandler):
    def __init__(self):
        self.tool_names = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.tool_names.append(kwargs.get("name") or (serialized or {}).get("name"))


def test_tracing_is_off_without_keys():
    # conftest.py blanks the Langfuse keys for every test
    assert tracing.enabled() is False
    with tracing.trace("tax-agent", user_id="user_a") as callbacks:
        assert callbacks == []
    tracing.flush()  # must be a harmless no-op


def test_langfuse_handler_can_be_imported():
    # Guards the dependency: the handler refuses to import without the `langchain` package,
    # and tracing is lazy-imported, so nothing else would notice until a real run.
    from langfuse.langchain import CallbackHandler

    assert CallbackHandler is not None


def make_span(scope_name: str, **attributes):
    """A real finished OpenTelemetry span emitted under the given instrumentation scope."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer(scope_name).start_as_current_span("span", attributes=attributes):
        pass
    return exporter.get_finished_spans()[0]


def test_anthropic_sdk_spans_are_not_exported_but_other_llm_spans_are():
    llm_attributes = {"gen_ai.operation.name": "chat"}
    sdk_span = make_span(tracing.ANTHROPIC_SDK_SCOPE, **llm_attributes)
    assert is_default_export_span(sdk_span) is True  # why the filter is needed: Langfuse exports it by default
    assert tracing._should_export(sdk_span) is False  # it duplicates the LangChain handler's generation
    assert tracing._should_export(make_span("some.other.instrumentation", **llm_attributes)) is True
    assert tracing._should_export(make_span(tracing.ANTHROPIC_SDK_SCOPE)) is False


def test_tracing_turns_on_only_when_both_keys_are_set(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    assert tracing.enabled() is False
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    assert tracing.enabled() is True


def test_ask_passes_callbacks_through_the_graph():
    recorder = RecordingHandler()
    model = ScriptedModel(messages=iter([
        call("get_user_profile", {"user_id": "user_a"}, "1"),
        AIMessage(content="done"),
    ]))
    agent = build_agent(MockDataSource(), "user_a", model=model)
    ask(agent, "hi", callbacks=[recorder])
    assert recorder.tool_names == ["get_user_profile"]  # the handler Langfuse relies on sees tool runs
