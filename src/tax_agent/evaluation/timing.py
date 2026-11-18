"""How long each model call and each tool call of an eval run took.

The wall-clock time of an item mixes two things: the model thinking and writing, and the extra
round trips a tool needs (the model asks for the tool, the tool answers, the model is called again).
CallTimer is a LangChain callback handler that times the two separately.
"""
import time

from langchain_core.callbacks import BaseCallbackHandler


class CallTimer(BaseCallbackHandler):
    def __init__(self):
        self._started: dict = {}
        self.model_seconds: list[float] = []  # one entry per finished model call
        self.tool_seconds: float = 0.0  # all tool calls together

    def _finish(self, run_id) -> float | None:
        started = self._started.pop(run_id, None)
        return None if started is None else time.perf_counter() - started

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._started[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs):
        if (seconds := self._finish(run_id)) is not None:
            self.model_seconds.append(seconds)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._finish(run_id)  # a failed call is not counted as a model call

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._started[run_id] = time.perf_counter()

    def on_tool_end(self, output, *, run_id, **kwargs):
        if (seconds := self._finish(run_id)) is not None:
            self.tool_seconds += seconds

    def on_tool_error(self, error, *, run_id, **kwargs):
        if (seconds := self._finish(run_id)) is not None:
            self.tool_seconds += seconds
