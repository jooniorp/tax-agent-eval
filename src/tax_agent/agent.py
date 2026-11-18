"""LangGraph tool-calling agent: a model node and a tool node in a loop.

    START -> model --(tool calls?)--> tools -> model ... -> END (final answer)

`build_agent(..., variant=...)` builds the three agents the eval compares:
  - "combined" (default, owner decision 2026-11-18): one compute_deduction tool that looks the user up,
    fetches the data and applies the rules, so a question takes two model calls;
  - "calculator": step-by-step tools, the model passes the data to calc_deduction (the eval baseline);
  - "no_calculator": the control, no calc_deduction and the tax rules as text, the model calculates itself.
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tax_agent.combined_tools import build_combined_tools
from tax_agent.datasource import DataSource
from tax_agent.llm import get_model
from tax_agent.rules import RULES_TEXT
from tax_agent.tools import build_tools

MAX_STEPS = 12  # graph steps per question (model + tool calls); caps a runaway loop's cost
VARIANTS = ("calculator", "no_calculator", "combined")
DEFAULT_VARIANT = "combined"

SYSTEM_PROMPT = """You are a year-end tax settlement assistant for Korean salaried workers (2025 tax year).
Answer in Korean.

Rules:
- Never calculate deduction amounts yourself. Get them from the calc_deduction tool.
- To answer a deduction question: get_user_profile first, then the data tool for the topic (get_card_spending, get_rent_payments or get_medical_expenses), then calc_deduction, passing the tool results through unchanged.
- If something the rules need is missing (a null in the profile, for example gross_salary, or homeless_household_head for rent), ask the user for it and wait for the answer. Do not call calc_deduction until you have it, and never guess it.
- Use only numbers returned by tools or given by the user. Never invent amounts.
- If calc_deduction says the user is not eligible, say so with the reason and do not state any deduction amount.
- Briefly explain how the result came about, in plain Korean (for example the minimum usage threshold, the 3% threshold or the cap). Do not repeat internal field names such as before_limit.
- This is a mock demo with fictional data, not tax advice.

The current user's id is {user_id}."""

# The control for the eval: same behaviour rules, but the model calculates with the rules below.
NO_CALCULATOR_PROMPT = """You are a year-end tax settlement assistant for Korean salaried workers (2025 tax year).
Answer in Korean.

Rules:
- There is no calculator tool. Work out the deduction amount yourself with the tax rules below, step by step, and double-check the arithmetic.
- To answer a deduction question: get_user_profile first, then the data tool for the topic (get_card_spending, get_rent_payments or get_medical_expenses), then calculate.
- If something the rules need is missing (a null in the profile, for example gross_salary, or homeless_household_head for rent), ask the user for it and wait for the answer. Do not calculate until you have it, and never guess it.
- Use only numbers returned by tools or given by the user. Never invent amounts.
- If the user is not eligible, say so with the reason and do not state any deduction amount.
- Briefly explain how the result came about, in plain Korean (for example the minimum usage threshold, the 3% threshold or the cap). Do not repeat internal field names such as before_limit.
- This is a mock demo with fictional data, not tax advice.

Tax rules (2025 tax year, amounts in KRW):
{rules}
The current user's id is {user_id}."""

# The combined agent: same behaviour rules, but one tool does lookup, data and calculation.
COMBINED_PROMPT = """You are a year-end tax settlement assistant for Korean salaried workers (2025 tax year).
Answer in Korean.

Rules:
- Never calculate deduction amounts yourself. Get them from the compute_deduction tool. It looks up the user's profile and data itself, so call it directly, once per topic (card, rent or medical), and do not call anything else first.
- If the result lists missing values (gross_salary, or homeless_household_head for rent), ask the user for them and wait for the answer. Then call compute_deduction again with the value the user gave. Never guess it.
- Use only numbers returned by the tool or given by the user. Never invent amounts.
- If the result says eligible is false, say so with the reason and do not state any deduction amount.
- Briefly explain how the result came about, in plain Korean (for example the minimum usage threshold, the 3% threshold or the cap). Do not repeat internal field names such as before_limit.
- This is a mock demo with fictional data, not tax advice.

The current user's id is {user_id}."""


def build_agent(source: DataSource, user_id: str, model: BaseChatModel | None = None, variant: str = DEFAULT_VARIANT):
    """Compile the agent graph for one user. `source` is where the data comes from.

    `variant` is one of VARIANTS (see the module docstring).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; use one of {VARIANTS}")
    if variant == "combined":
        available = build_combined_tools(source)
    else:
        available = build_tools(source)
        if variant == "no_calculator":
            available.pop("calc_deduction")
    tools = [StructuredTool.from_function(f) for f in available.values()]
    llm = (model or get_model()).bind_tools(tools)
    template = {"calculator": SYSTEM_PROMPT, "no_calculator": NO_CALCULATOR_PROMPT, "combined": COMBINED_PROMPT}[variant]
    system = SystemMessage(template.format(user_id=user_id, rules=RULES_TEXT))

    def call_model(state: MessagesState) -> dict:
        return {"messages": [llm.invoke([system, *state["messages"]])]}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    # handle_tool_errors=True hands tool exceptions back to the model as error messages;
    # the default would re-raise them and crash the run.
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)  # tool calls -> "tools", else END
    graph.add_edge("tools", "model")
    return graph.compile()


def ask(
    agent, question: str, callbacks: list | None = None, history: list[BaseMessage] | None = None
) -> list[BaseMessage]:
    """Run one question. Returns the whole message trail; the last message is the answer.

    `history` is the trail returned by an earlier call, for a follow-up turn.
    `callbacks` are LangChain callback handlers (for example the tracing one).
    """
    config = {"recursion_limit": MAX_STEPS, "callbacks": callbacks or []}
    result = agent.invoke({"messages": [*(history or []), HumanMessage(question)]}, config)
    return result["messages"]


def token_totals(messages: list[BaseMessage]) -> tuple[int, int]:
    """Total (input, output) tokens over every model call in the trail."""
    usages = [m.usage_metadata for m in messages if isinstance(m, AIMessage) and m.usage_metadata]
    return sum(u["input_tokens"] for u in usages), sum(u["output_tokens"] for u in usages)


def reasoning_tokens(messages: list[BaseMessage]) -> int:
    """The part of the output tokens the model spent thinking (it is billed as output)."""
    usages = [m.usage_metadata for m in messages if isinstance(m, AIMessage) and m.usage_metadata]
    return sum((u.get("output_token_details") or {}).get("reasoning", 0) for u in usages)
