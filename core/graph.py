from __future__ import annotations

from collections.abc import Iterable, Sequence

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.workers.document_conversion_agent import DocumentConversionAgentNode
from agents.workers.general_chat_agent import GeneralChatAgentNode
from agents.workers.knowledge_agent import KnowledgeAgentNode
from agents.workers.project_task_agent import ProjectTaskAgentNode
from core.agent_registry import AgentRegistration
from core.config import load_settings
from core.gateway import GatewayNode
from core.state import AgentState
from tools.knowledge_base import list_knowledge_documents, read_knowledge_document, search_knowledge_documents
from tools.project_tracker_google_sheets import get_project_sheet_overview, read_project_tasks


def build_default_agent_registrations(settings=None) -> tuple[AgentRegistration, ...]:
    resolved_settings = settings or load_settings()
    knowledge_tools = (
        list_knowledge_documents,
        search_knowledge_documents,
        read_knowledge_document,
    )
    project_tools = (read_project_tasks, get_project_sheet_overview)

    return (
        AgentRegistration(
            name="general_chat_agent",
            description="Use for greetings, casual chat, and general questions that do not require project data or internal documentation.",
            build_node=lambda llm: GeneralChatAgentNode(llm),
        ),
        AgentRegistration(
            name="knowledge_agent",
            description=(
                "Use for internal documentation, system architecture, setup instructions, repository guidance, "
                "and documented company workflows."
            ),
            build_node=lambda llm, tools=knowledge_tools: KnowledgeAgentNode(llm, list(tools)),
            tools=knowledge_tools,
        ),
        AgentRegistration(
            name="project_task_agent",
            description=(
                "Use for project tracker questions, assignees, deadlines, schedules, priorities, iterations, "
                "project status, or anything that likely requires Google Sheets data."
            ),
            build_node=lambda llm, tools=project_tools: ProjectTaskAgentNode(llm, list(tools)),
            tools=project_tools,
        ),
        AgentRegistration(
            name="document_conversion_agent",
            description=(
                "Use for Slack-driven design document conversion, canonical knowledge package staging, "
                "follow-up questions about missing conversion fields, and approval-gated publishing."
            ),
            build_node=lambda llm, settings=resolved_settings: DocumentConversionAgentNode(llm, settings=settings),
        ),
    )


def build_web_agent_registrations(settings=None) -> tuple[AgentRegistration, ...]:
    allowed_agent_names = {
        "general_chat_agent",
        "knowledge_agent",
        "project_task_agent",
    }
    return tuple(
        registration
        for registration in build_default_agent_registrations(settings=settings)
        if registration.name in allowed_agent_names
    )


def normalize_agent_registrations(agent_registrations: Sequence[AgentRegistration] | None, *, settings=None) -> tuple[AgentRegistration, ...]:
    registrations = tuple(agent_registrations or build_default_agent_registrations(settings=settings))
    if not registrations:
        raise ValueError("At least one agent registration is required.")

    seen_names: set[str] = set()
    for registration in registrations:
        if not registration.name.strip():
            raise ValueError("Agent registration names cannot be empty.")
        if registration.name in seen_names:
            raise ValueError(f"Duplicate agent registration name: {registration.name}")
        if registration.name == "gateway":
            raise ValueError("The agent name 'gateway' is reserved.")
        seen_names.add(registration.name)

    return registrations


def resolve_default_route(
    agent_registrations: Sequence[AgentRegistration],
    default_route: str | None = None,
) -> str:
    registrations = normalize_agent_registrations(agent_registrations)
    if default_route is None:
        return registrations[0].name

    if default_route not in {registration.name for registration in registrations}:
        raise ValueError(f"Unknown default route: {default_route}")
    return default_route


def resolve_gateway_route(
    state: AgentState,
    *,
    valid_routes: Iterable[str],
    default_route: str,
) -> str:
    route = state.get("route")
    valid_route_names = set(valid_routes)
    if isinstance(route, str) and route in valid_route_names:
        return route
    return default_route


def build_graph(
    llm,
    checkpointer=None,
    *,
    agent_registrations: Sequence[AgentRegistration] | None = None,
    default_route: str | None = None,
    settings=None,
):
    registrations = normalize_agent_registrations(agent_registrations, settings=settings)
    resolved_default_route = resolve_default_route(registrations, default_route=default_route)

    gateway_node = GatewayNode(
        llm,
        agent_registrations=registrations,
        default_route=resolved_default_route,
    )

    graph = StateGraph(AgentState)
    graph.add_node("gateway", gateway_node)
    graph.add_edge(START, "gateway")

    route_map: dict[str, str] = {}
    for registration in registrations:
        graph.add_node(registration.name, registration.build_node(llm))
        route_map[registration.name] = registration.name

    graph.add_conditional_edges(
        "gateway",
        lambda state: resolve_gateway_route(
            state,
            valid_routes=route_map.keys(),
            default_route=resolved_default_route,
        ),
        route_map,
    )

    for registration in registrations:
        if registration.tools:
            tool_node_name = f"{registration.name}_tools"
            graph.add_node(tool_node_name, ToolNode(list(registration.tools)))
            graph.add_conditional_edges(
                registration.name,
                tools_condition,
                {
                    "tools": tool_node_name,
                    "__end__": END,
                },
            )
            graph.add_edge(tool_node_name, registration.name)
            continue

        graph.add_edge(registration.name, END)

    return graph.compile(checkpointer=checkpointer)


def build_agent_graph(
    llm,
    checkpointer=None,
    *,
    agent_registrations: Sequence[AgentRegistration] | None = None,
    default_route: str | None = None,
    settings=None,
):
    return build_graph(
        llm,
        checkpointer=checkpointer,
        agent_registrations=agent_registrations,
        default_route=default_route,
        settings=settings,
    )
