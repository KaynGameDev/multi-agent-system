from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.workers.general_chat_agent import GeneralChatAgentNode
from agents.workers.project_task_agent import ProjectTaskAgentNode
from core.gateway import GatewayNode
from core.state import AgentState
from tools.google_sheets import get_project_sheet_overview, read_project_tasks


def build_graph(llm, checkpointer=None):
    tools = [
        read_project_tasks,
        get_project_sheet_overview,
    ]

    gateway_node = GatewayNode(llm)
    general_chat_agent_node = GeneralChatAgentNode(llm)
    project_task_agent_node = ProjectTaskAgentNode(llm, tools)
    project_tools_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    graph.add_node("gateway", gateway_node)
    graph.add_node("general_chat_agent", general_chat_agent_node)
    graph.add_node("project_task_agent", project_task_agent_node)
    graph.add_node("project_tools", project_tools_node)

    graph.add_edge(START, "gateway")

    graph.add_conditional_edges(
        "gateway",
        route_after_gateway,
        {
            "general_chat_agent": "general_chat_agent",
            "project_task_agent": "project_task_agent",
        },
    )

    graph.add_edge("general_chat_agent", END)

    graph.add_conditional_edges(
        "project_task_agent",
        tools_condition,
        {
            "tools": "project_tools",
            "__end__": END,
        },
    )

    graph.add_edge("project_tools", "project_task_agent")

    return graph.compile(checkpointer=checkpointer)


def build_agent_graph(llm, checkpointer=None):
    return build_graph(llm, checkpointer=checkpointer)


def route_after_gateway(state: AgentState) -> str:
    route = state.get("route", "general_chat_agent")
    if route == "project_task_agent":
        return "project_task_agent"
    return "general_chat_agent"
