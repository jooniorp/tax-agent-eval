"""LangGraph tool-calling agent: a model node and a tool node in a loop.

    START -> model --(tool calls?)--> tools -> model ... -> END (final answer)
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tax_agent.datasource import DataSource
from tax_agent.llm import get_model
from tax_agent.tools import build_tools

MAX_STEPS = 12  # graph steps per question (model + tool calls); caps a runaway loop's cost

SYSTEM_PROMPT = """You are a year-end tax settlement assistant for Korean salaried workers (2025 tax year).
Answer in Korean.

Rules:
- Never calculate deduction amounts yourself. Get them from the calc_deduction tool.
- To answer a deduction question: get_user_profile (gross salary), then get_card_spending, then calc_deduction, passing the tool results through unchanged.
- Use only numbers returned by tools. Never invent amounts. If something you need is missing, say so instead of guessing.
- Briefly explain how the result came about (for example the 25% minimum usage threshold and the cap).
- This is a mock demo with fictional data, not tax advice.

The current user's id is {user_id}."""


def build_agent(source: DataSource, user_id: str, model: BaseChatModel | None = None):
    """Compile the agent graph for one user. `source` is where the data comes from."""
    tools = [StructuredTool.from_function(f) for f in build_tools(source).values()]
    llm = (model or get_model()).bind_tools(tools)
    system = SystemMessage(SYSTEM_PROMPT.format(user_id=user_id))

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


def ask(agent, question: str) -> list[BaseMessage]:
    """Run one question. Returns the whole message trail; the last message is the answer."""
    result = agent.invoke({"messages": [HumanMessage(question)]}, {"recursion_limit": MAX_STEPS})
    return result["messages"]
