"""Scoring for the eval set: what the agent said (answer match) and how it got there (rule checks).

Pure functions over the message trail, so they are unit-tested without any model call.
Two methods, as in docs/research.md section 2:
  - answer match: the final answer states the expected amount(s), or "zero" / "not eligible" /
    "not supported" / "data not found";
  - rules: checks on the tool calls and the wording (right tools, numbers passed on
    unchanged, no amount before the needed information is known, ...).
The dataset is written for the step-by-step agent. For the other two agents `specs_for` leaves out the checks
that need the calculator (no-calculator control) or rewrites them for the single compute_deduction tool (combined).
"""
import json
import re
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

# A Korean amount: 5,300,000원 / 530만 원 / 167만 8,011원 / 1억 2,000만 원 / 37만 5천 원 / 5천만 원
# (each part optional, one needed)
_AMOUNT = re.compile(
    r"(?:(?P<eok>\d[\d,]*)\s*억)?\s*(?:(?P<cman>\d[\d,]*)\s*천\s*만)?\s*(?:(?P<man>\d[\d,]*)\s*만)?"
    r"\s*(?:(?P<cheon>\d[\d,]*)\s*천)?\s*(?:(?P<won>\d[\d,]*)\s*원)?"
)
_ZERO = re.compile(
    r"(?<![\d,])0\s*원|공제(?:액)?(?:이|가|를|은|는|도)?\s*(?:전혀\s*)?(?:없|받을 수 없|불가)"
    r"|받을 수 없|공제\s*대상(?:이|은)?\s*(?:아닙|아니|아님|없)"
    r"|금액(?:이|은)?\s*(?:전혀\s*)?없|받을 수 있는 (?:공제|금액)(?:이|은)?\s*없"
)
_INELIGIBLE = re.compile(
    r"대상(?:이|은|자가)?\s*(?:아닙|아니|아님|없|되지 않)|받을 수 없|공제(?:가|를|는)?\s*(?:불가|안\s*됩|되지 않)"
    r"|해당(?:하지|되지)\s*않|적용(?:되지|받지)\s*(?:않|못)|해당(?:이|은)?\s*없"
)
_UNSUPPORTED = re.compile(
    r"지원하지\s*(?:않|못)|지원되지\s*않|지원\s*대상이\s*아|계산(?:해\s*드릴\s*수\s*없|할\s*수\s*없|이\s*어렵)"
    r"|제공하지\s*(?:않|못)|도구(?:가|는)\s*(?:없|제공)|아직\s*(?:지원|구현|제공)"
    r"|(?:기능|도구)(?:이|가|은|는)?\s*없|(?:알려|안내해?|답변해?|답해?)\s*(?:드릴|할)\s*수\s*없"
    r"|(?:이\s*도구(?:로|는)|도구로)\s*(?:는\s*)?[^.\n]{0,40}(?:조회|확인|계산)(?:해\s*드릴|할)\s*수\s*없"
    r"|(?:조회|확인|계산|알려|안내|답변|답)\s*(?:해\s*)?(?:드릴|할)\s*수\s*없"
)
_NOT_FOUND = re.compile(
    r"찾을\s*수\s*없|조회(?:되지|할\s*수)\s*(?:않|없)|등록되(?:어\s*있)?지\s*않|존재하지\s*않"
    r"|확인할\s*수\s*없|확인되지\s*않|정보(?:가|를)\s*(?:없|찾)|알\s*수\s*없는\s*사용자|(?:가져오|불러오|조회하)지\s*못"
)
_ASKS = re.compile(r"(?:알려|말씀|입력|확인)\s*(?:해\s*)?(?:주세요|주시|주실|부탁)")


def amounts(text: str) -> set[int]:
    """Won amounts written in the text, in any of these styles: "5,300,000원", "530만 원",
    "167만 8,011원", "1억 2,000만 원", "37만 5천 원", "5천만 원". Decimals such as "167.8만" are not read (not exact)."""
    to_int = lambda s: int(s.replace(",", ""))  # noqa: E731
    unit = {"eok": 100_000_000, "cman": 10_000_000, "man": 10_000, "cheon": 1_000, "won": 1}
    found = set()
    for m in _AMOUNT.finditer(text):
        parts = {k: m.group(k) for k in unit}
        if any(parts.values()):
            found.add(sum(to_int(v) * unit[k] for k, v in parts.items() if v))
    return found


def says_zero(text: str) -> bool:
    return bool(_ZERO.search(text))


def says_ineligible(text: str) -> bool:
    return bool(_INELIGIBLE.search(text))


def says_unsupported(text: str) -> bool:
    return bool(_UNSUPPORTED.search(text))


def says_not_found(text: str) -> bool:
    return bool(_NOT_FOUND.search(text))


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


CALC_TOOLS = ("calc_deduction", "compute_deduction")  # the step-by-step agent's calculator, the combined agent's tool


def _calc_calls(run: Run, spec: dict) -> list[ToolCall]:
    return [c for c in run.calls(spec.get("turn")) if c.name in CALC_TOOLS]


def _last_calc(run: Run, spec: dict) -> ToolCall | None:
    calls = _calc_calls(run, spec)
    return calls[-1] if calls else None


def tool_called(run: Run, spec: dict):
    names = _names(run, spec)
    return names.count(spec["name"]) >= spec.get("min", 1), f"calls: {names}"


def tool_not_called(run: Run, spec: dict):
    names = _names(run, spec)
    return spec["name"] not in names, f"calls: {names}"


def tool_order(run: Run, spec: dict):
    names = _names(run, spec)
    ok = spec["before"] in names and spec["after"] in names and names.index(spec["before"]) < names.index(spec["after"])
    return ok, f"calls: {names}"


def args_relayed(run: Run, spec: dict):
    """Every number given to calc_deduction is one the data tools reported (a null there means
    "unknown", so the user may have supplied the value)."""
    calls = _calc_calls(run, spec)
    if not calls:
        return False, "no calc_deduction call"
    problems = []
    for call in calls:
        facts = run.facts_before(call.index)
        if facts.get("gross_salary") is not None and call.args.get("gross_salary") != facts["gross_salary"]:
            problems.append(f"gross_salary: tool said {facts['gross_salary']}, passed {call.args.get('gross_salary')}")
        for key, value in (call.args.get("data") or {}).items():
            if key not in facts:
                problems.append(f"{key}: no tool reported it")
            elif facts[key] is not None and facts[key] != value:
                problems.append(f"{key}: tool said {facts[key]}, passed {value}")
    return not problems, "; ".join(problems) or "ok"


def gross_salary_equals(run: Run, spec: dict):
    call = _last_calc(run, spec)
    passed = call.args.get("gross_salary") if call else None
    return passed == spec["value"], f"passed {passed}, expected {spec['value']}"


def calc_data_equals(run: Run, spec: dict):
    call = _last_calc(run, spec)
    passed = (call.args.get("data") or call.args).get(spec["key"]) if call else None  # combined: a top-level argument
    return passed == spec["value"], f"{spec['key']}: passed {passed!r}, expected {spec['value']!r}"


def answer_uses_tool_amount(run: Run, spec: dict):
    """The answer states what calc_deduction returned (the tool, not the model, makes the number)."""
    results = [r for r in (run.result_of(c) for c in _calc_calls(run, spec)) if "deduction" in r]
    if not results:
        return False, "no calc_deduction result"
    text, stated = run.answer(spec.get("turn")), amounts(run.answer(spec.get("turn")))
    positives = sorted({r["deduction"] for r in results if r["deduction"] > 0})
    if positives:
        missing = [d for d in positives if d not in stated]
        return not missing, f"tool said {positives}; answer has {sorted(stated)}"
    if any(r.get("eligible") is False for r in results):
        return says_ineligible(text), "tool said not eligible"
    return says_zero(text), "tool said 0"


def no_deduction_result(run: Run, spec: dict):
    """No calculation result with an amount yet (the combined agent's tool also answers "missing")."""
    results = [run.result_of(c) for c in _calc_calls(run, spec)]
    done = [r for r in results if "deduction" in r]
    return not done, f"{len(done)} calculation result(s) in this turn" if done else "ok"


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
    tool_called, tool_not_called, tool_order, args_relayed, gross_salary_equals, calc_data_equals,
    answer_uses_tool_amount, answer_contains, answer_not_contains_amount, asks_question, no_amount,
    no_deduction_result,
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

_CALCULATOR_ONLY = {"args_relayed", "gross_salary_equals", "calc_data_equals", "answer_uses_tool_amount"}


def needs_calculator(spec: dict) -> bool:
    """True for a check that only makes sense when the agent has the calc_deduction tool."""
    if spec["type"] in _CALCULATOR_ONLY:
        return True
    return spec["type"] in {"tool_called", "tool_not_called", "tool_order"} and "calc_deduction" in (
        spec.get("name"), spec.get("before"), spec.get("after")
    )


def specs_for(specs: list[dict], variant: str) -> list[dict]:
    """The checks that apply to an agent variant (the dataset is written for the step-by-step one)."""
    if variant == "calculator":
        return specs
    if variant == "no_calculator":
        return [s for s in specs if not needs_calculator(s)]
    out = []  # combined: one tool does lookup, data and calculation
    for s in specs:
        if s["type"] in ("args_relayed", "tool_order"):
            continue  # there are no steps to pass numbers between
        if s["type"] == "tool_called" and s["name"] in ("get_user_profile", "calc_deduction"):
            s = {**s, "name": "compute_deduction"}  # looking the user up is now part of that call
        elif s["type"] == "tool_not_called" and s["name"] == "calc_deduction":
            s = {"type": "no_deduction_result", **{k: v for k, v in s.items() if k not in ("type", "name")}}
        out.append(s)
    return out


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
    if expected.get("unsupported"):
        return says_unsupported(text), "expected: says the topic is not supported"
    if expected.get("not_found"):
        return says_not_found(text), "expected: says the user's data was not found"
    stated = amounts(text)
    if "amounts" in expected:
        missing = [a for a in expected["amounts"] if a not in stated]
        return not missing, f"expected all of {expected['amounts']}; answer has {sorted(stated)}"
    if expected["amount"] == 0:
        return says_zero(text), "expected: 0 / no deduction"
    return expected["amount"] in stated, f"expected {expected['amount']}; answer has {sorted(stated)}"


def item_specs(item: dict, variant: str) -> list[dict]:
    """The rule checks of an item for an agent variant, bundles expanded."""
    specs = [s for entry in item["checks"] for s in (BUNDLES[entry] if isinstance(entry, str) else [entry])]
    return specs_for(specs, variant)


# The checks that read nothing but the answer text, so they can be judged again from saved answers.
ANSWER_ONLY = {"answer_contains", "answer_not_contains_amount", "no_amount", "asks_question"}


def rejudge_texts(item: dict, variant: str, turn_answers: list[str]) -> tuple[bool, list[CheckResult | None]]:
    """Judge again from the saved answer of every turn: the answer match, and the checks that need only the
    text. A check that needs the tool calls comes back as None (the trail is not saved)."""
    messages = [AIMessage(a) for a in turn_answers]
    run = Run(messages, list(range(1, len(messages) + 1)))
    answer_ok, _ = answer_match(run, item["expected"])
    checks: list[CheckResult | None] = []
    for spec in item_specs(item, variant):
        if spec["type"] not in ANSWER_ONLY:
            checks.append(None)
            continue
        label = spec["type"] + (f"@turn{spec['turn']}" if "turn" in spec else "")
        try:
            ok, detail = CHECKS[spec["type"]](run, spec)
        except Exception as e:  # same rule as score_item: a malformed trail fails the check
            ok, detail = False, f"{type(e).__name__}: {e}"
        checks.append(CheckResult(label, ok, detail))
    return answer_ok, checks


def score_item(item: dict, run: Run, variant: str = "calculator") -> ItemScore:
    """Score one item for the given agent variant (see `specs_for`)."""
    answer_ok, answer_detail = answer_match(run, item["expected"])
    results = []
    for spec in item_specs(item, variant):
        label = spec["type"] + (f"@turn{spec['turn']}" if "turn" in spec else "")
        try:
            ok, detail = CHECKS[spec["type"]](run, spec)
        except Exception as e:  # a malformed trail must fail the check, not crash the eval
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append(CheckResult(label, ok, detail))
    return ItemScore(answer_ok, answer_detail, results)
