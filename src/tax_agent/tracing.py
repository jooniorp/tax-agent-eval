"""Optional Langfuse tracing: a no-op unless both Langfuse keys are set.

Traces contain the question, the tool inputs/outputs and the answer, so only
fictional data may go through here (see CLAUDE.md). Langfuse is imported lazily:
with tracing off nothing from it is loaded or contacted.
"""
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache

from dotenv import load_dotenv

ANTHROPIC_SDK_SCOPE = "com.anthropic.sdk.python"  # OpenTelemetry scope of the Anthropic SDK's own spans


def enabled() -> bool:
    load_dotenv()
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _should_export(span) -> bool:
    """Langfuse's default span filter, minus the Anthropic SDK's own spans.

    The LangChain handler already records every model call as a generation, with reasoning
    tokens counted as output. The SDK emits a second span for the same call, and Langfuse's
    default filter exports it too (it carries gen_ai.* attributes). Exporting both would
    double the token and cost totals of every trace.
    """
    from langfuse.span_filter import is_default_export_span

    scope = span.instrumentation_scope
    return is_default_export_span(span) and not (scope and scope.name == ANTHROPIC_SDK_SCOPE)


@cache
def _client():
    """The Langfuse client, created once per process. It must exist before the first
    get_client() call (made by the callback handler) so that it carries our span filter."""
    from langfuse import Langfuse

    return Langfuse(should_export_span=_should_export)


@contextmanager
def trace(name: str, user_id: str) -> Iterator[list]:
    """Yield the LangChain callbacks to pass to `ask` (an empty list when tracing is off).

    Attributes set by propagate_attributes reach every span of the run, so the
    trace shows up in Langfuse under `name` and `user_id`.
    """
    if not enabled():
        yield []
        return
    from langfuse import propagate_attributes
    from langfuse.langchain import CallbackHandler

    _client()
    with propagate_attributes(trace_name=name, user_id=user_id):
        yield [CallbackHandler()]


def flush() -> None:
    """Send pending traces. A short-lived process (the CLI) must call this before it exits."""
    if enabled():
        _client().flush()
