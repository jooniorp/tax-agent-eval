"""Scoring for the eval set: what the agent said (answer match) and how it got there (rule checks).

Pure functions over the message trail, so they are unit-tested without any model call.
Two methods, as in docs/research.md section 2:
  - answer match: the final answer states the expected amount (or "zero" / "not eligible");
  - rules: checks on the tool calls and the wording (right tools, numbers passed on
    unchanged, no amount before the needed information is known, ...).
"""
import json
import re
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

_MAN = re.compile(r"(\d[\d,]*)\s*만\s*원?")  # 530만원, 530만 원, 6,800만
_WON = re.compile(r"(\d[\d,]*)\s*원")  # 5,300,000원
_ZERO = re.compile(
    r"(?<![\d,])0\s*원|공제(?:액)?(?:이|가|를|은|는|도)?\s*(?:전혀\s*)?(?:없|받을 수 없|불가)"
    r"|받을 수 없|공제\s*대상(?:이|은)?\s*(?:아닙|아니|아님|없)"
)
_INELIGIBLE = re.compile(
    r"대상(?:이|은|자가)?\s*(?:아닙|아니|아님|없|되지 않)|받을 수 없|공제(?:가|를|는)?\s*(?:불가|안\s*됩|되지 않)"
    r"|해당(?:하지|되지)\s*않|적용(?:되지|받지)\s*(?:않|못)"
)
_ASKS = re.compile(r"(?:알려|말씀|입력|확인)\s*(?:해\s*)?(?:주세요|주시|주실|부탁)")


def amounts(text: str) -> set[int]:
    """Won amounts written in the text, in either style: "5,300,000원" or "530만 원"."""
    found = {int(m.group(1).replace(",", "")) * 10_000 for m in _MAN.finditer(text)}
    return found | {int(m.group(1).replace(",", "")) for m in _WON.finditer(text)}


def says_zero(text: str) -> bool:
    return bool(_ZERO.search(text))


def says_ineligible(text: str) -> bool:
    return bool(_INELIGIBLE.search(text))


@dataclass
class ToolCall:
    index: int  # position of the AI message in the whole trail
    name: str
    args: dict
    call_id: str


@dataclass
class Run:
    """The message trail of one eval item. `turn_ends[i]` is where turn i+1 ended (exclusive)."""

    messages: list[BaseMessage]
    turn_ends: list[int]

    def _bounds(self, turn: int | None) -> tuple[int, int]:
        if turn is None:
            return 0, len(self.messages)
        return (self.turn_ends[turn - 2] if turn > 1 else 0), self.turn_ends[turn - 1]

    def calls(self, turn: int | None = None) -> list[ToolCall]:
        start, end = self._bounds(turn)
        return [
            ToolCall(i, c["name"], c["args"], c["id"])
            for i, m in enumerate(self.messages[start:end], start)
            if isinstance(m, AIMessage)
            for c in m.tool_calls
        ]

    def answer(self, turn: int | None = None) -> str:
        """Text of the last AI message in the turn (the whole run when turn is None)."""
        start, end = self._bounds(turn)
        texts = [m.text for m in self.messages[start:end] if isinstance(m, AIMessage)]
        return texts[-1] if texts else ""

    def facts_before(self, index: int) -> dict:
        """Everything the data tools reported before `index`, merged into one dict."""
        facts: dict = {}
        for m in self.messages[:index]:
            if isinstance(m, ToolMessage) and m.name != "calc_deduction":
                facts.update(_as_dict(m.content))
        return facts

    def result_of(self, call: ToolCall) -> dict:
        for m in self.messages[call.index + 1:]:
            if isinstance(m, ToolMessage) and m.tool_call_id == call.call_id:
                return _as_dict(m.content)
        return {}


def _as_dict(content) -> dict:
    try:
        value = json.loads(content) if isinstance(content, str) else None
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


# --- rule checks: (run, spec) -> (ok, detail) -------------------------------------------------

def _names(run: Run, spec: dict) -> list[str]:
    return [c.name for c in run.calls(spec.get("turn"))]


def _last_calc(run: Run, spec: dict) -> ToolCall | None:
    calcs = [c for c in run.calls(spec.get("turn")) if c.name == "calc_deduction"]
    return calcs[-1] if calcs else None


def tool_called(run: Run, spec: dict):
    names = _names(run, spec)
    return spec["name"] in names, f"calls: {names}"


def tool_not_called(run: Run, spec: dict):
    names = _names(run, spec)
    return spec["name"] not in names, f"calls: {names}"


def tool_order(run: Run, spec: dict):
    names = _names(run, spec)
    ok = spec["before"] in names and spec["after"] in names and names.index(spec["before"]) < names.index(spec["after"])
    return ok, f"calls: {names}"


def args_relayed(run: Run, spec: dict):
    """The numbers given to calc_deduction are the ones the data tools reported."""
    call = _last_calc(run, spec)
    if call is None:
        return False, "no calc_deduction call"
    facts, problems = run.facts_before(call.index), []
    if facts.get("gross_salary") is not None and call.args.get("gross_salary") != facts["gross_salary"]:
        problems.append(f"gross_salary: tool said {facts['gross_salary']}, passed {call.args.get('gross_salary')}")
    for key, value in (call.args.get("data") or {}).items():
        if key not in facts:
            problems.append(f"{key}: no tool reported it")
        elif facts[key] != value:
            problems.append(f"{key}: tool said {facts[key]}, passed {value}")
    return not problems, "; ".join(problems) or "ok"


def gross_salary_equals(run: Run, spec: dict):
    call = _last_calc(run, spec)
    passed = call.args.get("gross_salary") if call else None
    return passed == spec["value"], f"passed {passed}, expected {spec['value']}"


def answer_uses_tool_amount(run: Run, spec: dict):
    """The answer states what calc_deduction returned (the tool, not the model, makes the number)."""
    call = _last_calc(run, spec)
    result = run.result_of(call) if call else {}
    if "deduction" not in result:
        return False, "no calc_deduction result"
    text = run.answer(spec.get("turn"))
    if result.get("eligible") is False:
        return says_ineligible(text), "tool said not eligible"
    if result["deduction"] == 0:
        return says_zero(text), "tool said 0"
    return result["deduction"] in amounts(text), f"tool said {result['deduction']}; answer has {sorted(amounts(text))}"


def answer_contains(run: Run, spec: dict):
    text = run.answer(spec.get("turn"))
    missing = [p for p in spec["all"] if not re.search(p, text)]
    return not missing, f"missing: {missing}" if missing else "ok"


def answer_not_contains_amount(run: Run, spec: dict):
    wrong = sorted(set(spec["amounts"]) & amounts(run.answer(spec.get("turn"))))
    return not wrong, f"stated forbidden amounts: {wrong}" if wrong else "ok"


def asks_question(run: Run, spec: dict):
    text = run.answer(spec.get("turn"))
    return "?" in text or bool(_ASKS.search(text)), "answer does not ask for anything"


def no_amount(run: Run, spec: dict):
    stated = sorted(a for a in amounts(run.answer(spec.get("turn"))) if a > 0)
    return not stated, f"stated amounts: {stated}" if stated else "ok"


CHECKS = {f.__name__: f for f in (
    tool_called, tool_not_called, tool_order, args_relayed, gross_salary_equals,
    answer_uses_tool_amount, answer_contains, answer_not_contains_amount, asks_question, no_amount,
)}

# A dataset entry can name a bundle instead of repeating its checks.
BUNDLES = {
    "standard_calc": [
        {"type": "tool_called", "name": "calc_deduction"},
        {"type": "tool_order", "before": "get_user_profile", "after": "calc_deduction"},
        {"type": "args_relayed"},
        {"type": "answer_uses_tool_amount"},
    ],
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass
class ItemScore:
    answer_ok: bool
    answer_detail: str
    checks: list[CheckResult]

    @property
    def rules_ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def passed(self) -> bool:
        return self.answer_ok and self.rules_ok


def answer_match(run: Run, expected: dict) -> tuple[bool, str]:
    text = run.answer()
    if expected.get("eligible") is False:
        return says_ineligible(text), "expected: not eligible"
    if expected["amount"] == 0:
        return says_zero(text), "expected: 0 / no deduction"
    return expected["amount"] in amounts(text), f"expected {expected['amount']}; answer has {sorted(amounts(text))}"


def score_item(item: dict, run: Run) -> ItemScore:
    answer_ok, answer_detail = answer_match(run, item["expected"])
    specs = [s for entry in item["checks"] for s in (BUNDLES[entry] if isinstance(entry, str) else [entry])]
    results = []
    for spec in specs:
        label = spec["type"] + (f"@turn{spec['turn']}" if "turn" in spec else "")
        try:
            ok, detail = CHECKS[spec["type"]](run, spec)
        except Exception as e:  # a malformed trail must fail the check, not crash the eval
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append(CheckResult(label, ok, detail))
    return ItemScore(answer_ok, answer_detail, results)
