from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.workers.general_chat_agent import GeneralChatAgentNode
from agents.workers.project_task_agent import ProjectTaskAgentNode
from core.gateway import GatewayNode
from core.state import AgentState
from tools.google_sheets import get_project_sheet_overview, read_project_tasks


PROJECT_TOOLS = [read_project_tasks, get_project_sheet_overview]


def build_agent_graph(llm, checkpointer=None):
    workflow = StateGraph(AgentState)

    workflow.add_node("gateway", GatewayNode(llm))
    workflow.add_node("general_chat_agent", GeneralChatAgentNode(llm))
    workflow.add_node("project_task_agent", ProjectTaskAgentNode(llm, PROJECT_TOOLS))
    workflow.add_node(
        "project_tools",
        ToolNode(PROJECT_TOOLS, handle_tool_errors=True),
    )

    workflow.add_edge(START, "gateway")
    workflow.add_conditional_edges(
        "gateway",
        route_from_gateway,
        {
            "project_task_agent": "project_task_agent",
            "general_chat_agent": "general_chat_agent",
        },
    )

    workflow.add_conditional_edges(
        "project_task_agent",
        tools_condition,
        {
            "tools": "project_tools",
            END: END,
        },
    )
    workflow.add_edge("project_tools", "project_task_agent")
    workflow.add_edge("general_chat_agent", END)

    return workflow.compile(checkpointer=checkpointer)


def route_from_gateway(state: AgentState) -> str:
    route = state.get("route", "general_chat_agent")
    if route not in {"project_task_agent", "general_chat_agent"}:
        return "general_chat_agent"
    return route
