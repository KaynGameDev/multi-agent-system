from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agents.document_conversion.agent import DocumentConversionAgentNode
from agents.general_chat.agent import GeneralChatAgentNode
from agents.knowledge.agent import KnowledgeAgentNode
from agents.knowledge_base_builder.agent import KnowledgeBaseBuilderAgentNode
from agents.project_task.agent import ProjectTaskAgentNode
from app.agent_registry import AgentRegistration
from app.config import load_settings
from app.pending_action_parser import PendingActionReplyInterpreter
from gateway.agent import GatewayNode
from gateway.agent import (
    document_conversion_matcher,
    general_chat_matcher,
    knowledge_base_builder_matcher,
    knowledge_matcher,
    project_task_matcher_factory,
)
from app.skills import SkillRegistry
from app.state import AgentState
from tools.knowledge_base import (
    list_knowledge_documents,
    read_knowledge_document,
    resolve_knowledge_markdown_path,
    search_knowledge_documents,
    write_knowledge_markdown_document,
)
from tools.project_tracker_google_sheets import get_project_sheet_overview, read_project_tasks
from app.tool_registry import KNOWLEDGE_BUILDER_TOOL_IDS, KNOWLEDGE_TOOL_IDS, PROJECT_TOOL_IDS


def build_default_agent_registrations(
    settings=None,
    *,
    pending_action_interpreter: PendingActionReplyInterpreter | None = None,
) -> tuple[AgentRegistration, ...]:
    resolved_settings = settings or load_settings()
    knowledge_read_tools = (
        list_knowledge_documents,
        search_knowledge_documents,
        read_knowledge_document,
    )
    knowledge_read_tool_ids = KNOWLEDGE_TOOL_IDS
    knowledge_builder_tools = (
        *knowledge_read_tools,
        resolve_knowledge_markdown_path,
        write_knowledge_markdown_document,
    )
    knowledge_builder_tool_ids = KNOWLEDGE_BUILDER_TOOL_IDS
    project_tools = (read_project_tasks, get_project_sheet_overview)
    project_tool_ids = PROJECT_TOOL_IDS

    return (
        AgentRegistration(
            name="general_chat_agent",
            description="Use for greetings, casual chat, and general questions that do not require project data or internal documentation.",
            build_node=lambda llm, skill_registry=None, pending_action_interpreter=None: GeneralChatAgentNode(
                llm,
                skill_registry=skill_registry,
                agent_name="general_chat_agent",
            ),
            selection_order=40,
            is_general_assistant=True,
            skill_namespace="general_chat",
            matcher=general_chat_matcher,
        ),
        AgentRegistration(
            name="knowledge_agent",
            description=(
                "Use for internal documentation, system architecture, setup instructions, repository guidance, "
                "and documented company workflows."
            ),
            build_node=lambda llm, tools=knowledge_read_tools, skill_registry=None, pending_action_interpreter=pending_action_interpreter: KnowledgeAgentNode(
                llm,
                list(tools),
                skill_registry=skill_registry,
                pending_action_interpreter=pending_action_interpreter,
                agent_name="knowledge_agent",
                tool_ids=knowledge_read_tool_ids,
            ),
            tools=knowledge_read_tools,
            tool_ids=knowledge_read_tool_ids,
            selection_order=30,
            skill_namespace="knowledge",
            matcher=knowledge_matcher,
        ),
        AgentRegistration(
            name="knowledge_base_builder_agent",
            description=(
                "Use for knowledge elicitation, KB document review, layer placement decisions, "
                "feature-spec skeleton building, and KB V1 execution tracking."
            ),
            build_node=lambda llm, tools=knowledge_builder_tools, skill_registry=None, pending_action_interpreter=pending_action_interpreter: KnowledgeBaseBuilderAgentNode(
                llm,
                list(tools),
                skill_registry=skill_registry,
                pending_action_interpreter=pending_action_interpreter,
                agent_name="knowledge_base_builder_agent",
                tool_ids=knowledge_builder_tool_ids,
            ),
            tools=knowledge_builder_tools,
            tool_ids=knowledge_builder_tool_ids,
            selection_order=35,
            skill_namespace="knowledge_base_builder",
            matcher=knowledge_base_builder_matcher,
        ),
        AgentRegistration(
            name="project_task_agent",
            description=(
                "Use for project tracker questions, assignees, deadlines, schedules, priorities, iterations, "
                "project status, or anything that likely requires Google Sheets data."
            ),
            build_node=lambda llm, tools=project_tools, skill_registry=None, pending_action_interpreter=pending_action_interpreter: ProjectTaskAgentNode(
                llm,
                list(tools),
                skill_registry=skill_registry,
                pending_action_interpreter=pending_action_interpreter,
                agent_name="project_task_agent",
                tool_ids=project_tool_ids,
            ),
            tools=project_tools,
            tool_ids=project_tool_ids,
            selection_order=20,
            skill_namespace="project_task",
            matcher=project_task_matcher_factory(resolved_settings.project_lookup_keywords),
        ),
        AgentRegistration(
            name="document_conversion_agent",
            description=(
                "Use for Slack-driven design document conversion, canonical knowledge package staging, "
                "follow-up questions about missing conversion fields, and approval-gated publishing."
            ),
            build_node=lambda llm, settings=resolved_settings, skill_registry=None, pending_action_interpreter=pending_action_interpreter: DocumentConversionAgentNode(
                llm,
                settings=settings,
                skill_registry=skill_registry,
                pending_action_interpreter=pending_action_interpreter,
                agent_name="document_conversion_agent",
            ),
            selection_order=10,
            skill_namespace="document_conversion",
            matcher=document_conversion_matcher,
        ),
    )


def build_web_agent_registrations(
    settings=None,
    *,
    pending_action_interpreter: PendingActionReplyInterpreter | None = None,
) -> tuple[AgentRegistration, ...]:
    allowed_agent_names = {
        "general_chat_agent",
        "knowledge_agent",
        "knowledge_base_builder_agent",
        "project_task_agent",
        "document_conversion_agent",
    }
    return tuple(
        registration
        for registration in build_default_agent_registrations(
            settings=settings,
            pending_action_interpreter=pending_action_interpreter,
        )
        if registration.name in allowed_agent_names
    )


def normalize_agent_registrations(
    agent_registrations: Sequence[AgentRegistration] | None,
    *,
    settings=None,
    pending_action_interpreter: PendingActionReplyInterpreter | None = None,
) -> tuple[AgentRegistration, ...]:
    registrations = tuple(
        agent_registrations
        or build_default_agent_registrations(
            settings=settings,
            pending_action_interpreter=pending_action_interpreter,
        )
    )
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
    skill_registry: SkillRegistry | None = None,
    pending_action_interpreter: PendingActionReplyInterpreter | None = None,
):
    registrations = normalize_agent_registrations(
        agent_registrations,
        settings=settings,
        pending_action_interpreter=pending_action_interpreter,
    )
    resolved_default_route = resolve_default_route(registrations, default_route=default_route)
    resolved_settings = settings or load_settings()
    resolved_skill_registry = skill_registry or SkillRegistry(
        registrations,
        project_skills_dir=resolved_settings.jade_project_skills_dir,
    )

    gateway_node = GatewayNode(
        llm,
        agent_registrations=registrations,
        default_route=resolved_default_route,
        skill_registry=resolved_skill_registry,
    )

    graph = StateGraph(AgentState)
    graph.add_node("gateway", gateway_node)
    graph.add_edge(START, "gateway")

    route_map: dict[str, str] = {}
    for registration in registrations:
        graph.add_node(
            registration.name,
            build_registered_agent_node(
                registration,
                llm,
                skill_registry=resolved_skill_registry,
                pending_action_interpreter=pending_action_interpreter,
            ),
        )
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
    skill_registry: SkillRegistry | None = None,
    pending_action_interpreter: PendingActionReplyInterpreter | None = None,
):
    return build_graph(
        llm,
        checkpointer=checkpointer,
        agent_registrations=agent_registrations,
        default_route=default_route,
        settings=settings,
        skill_registry=skill_registry,
        pending_action_interpreter=pending_action_interpreter,
    )


def build_registered_agent_node(
    registration: AgentRegistration,
    llm,
    *,
    skill_registry: SkillRegistry | None = None,
    pending_action_interpreter: PendingActionReplyInterpreter | None = None,
):
    kwargs = build_supported_agent_node_kwargs(
        registration.build_node,
        skill_registry=skill_registry,
        pending_action_interpreter=pending_action_interpreter,
    )
    return registration.build_node(llm, **kwargs)


def build_supported_agent_node_kwargs(build_node, **candidate_kwargs):
    try:
        signature = inspect.signature(build_node)
    except (TypeError, ValueError):
        return {}

    parameters = signature.parameters.values()
    supports_var_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters)
    if supports_var_kwargs:
        return {name: value for name, value in candidate_kwargs.items() if value is not None}

    supported_names = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return {
        name: value
        for name, value in candidate_kwargs.items()
        if value is not None and name in supported_names
    }
