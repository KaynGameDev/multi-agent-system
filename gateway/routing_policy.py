from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent_registry import AgentRegistration
from app.pending_actions import get_pending_action, is_pending_action_active
from app.skills import SkillDefinition
from app.state import AgentState
from gateway.model_router import ModelRouter

GENERAL_ASSISTANT_ALIAS = "GeneralAssistant"
ROUTING_POLICY_REQUESTED_AGENT = "requested_agent"
ROUTING_POLICY_FORKED_SKILL_DELEGATE = "forked_skill_delegate"
ROUTING_POLICY_FORKED_SKILL_FALLBACK = "forked_skill_fallback"
ROUTING_POLICY_INLINE_SKILL_COMPATIBILITY = "inline_skill_compatibility"
ROUTING_POLICY_PENDING_ACTION_OWNER = "pending_action_owner"
ROUTING_POLICY_STATE_ROUTE = "state_route"
ROUTING_POLICY_MODEL_ROUTER = "model_router"
ROUTING_POLICY_GENERAL_FALLBACK = "general_fallback"


@dataclass(frozen=True)
class AgentRouteSelection:
    agent_name: str
    policy_step: str
    diagnostics: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RoutingPolicyContext:
    state: AgentState
    latest_user_text: str
    requested_agent: str
    explicit_skill_definitions: tuple[SkillDefinition, ...]
    agent_registrations: tuple[AgentRegistration, ...]
    registrations_by_name: dict[str, AgentRegistration]
    default_route: str
    general_assistant_name: str
    model_router: ModelRouter | None


def resolve_general_assistant_name(
    agent_registrations: tuple[AgentRegistration, ...],
    registrations_by_name: dict[str, AgentRegistration],
) -> str:
    for registration in agent_registrations:
        if registration.is_general_assistant:
            return registration.name
    if "general_chat_agent" in registrations_by_name:
        return "general_chat_agent"
    return ""


def normalize_agent_name(raw_value: str, general_assistant_name: str) -> str:
    if not raw_value:
        return ""
    if raw_value == GENERAL_ASSISTANT_ALIAS:
        return general_assistant_name or raw_value
    return raw_value


def build_route_reason(agent_diagnostics: list[dict[str, Any]], route: str) -> str:
    for item in reversed(agent_diagnostics):
        reason = str(item.get("reason", "")).strip()
        if reason:
            return reason
    return f"Selected `{route}` from gateway routing policy."


def select_agent_by_policy(
    context: RoutingPolicyContext,
    *,
    warnings: list[str],
) -> tuple[str, str, list[dict[str, Any]], list[str]]:
    selectors = (
        _select_requested_agent,
        _select_explicit_forked_skill_delegate,
        _select_explicit_forked_skill_fallback,
        _select_explicit_inline_skill_candidate,
        _select_pending_action_owner,
        _select_state_route,
        _select_model_router,
    )

    for selector in selectors:
        selection = selector(context, warnings)
        if selection is not None:
            return selection.agent_name, selection.policy_step, list(selection.diagnostics), warnings

    selection = _select_general_fallback(context, warnings)
    return selection.agent_name, selection.policy_step, list(selection.diagnostics), warnings


def _select_requested_agent(
    context: RoutingPolicyContext,
    warnings: list[str],
) -> AgentRouteSelection | None:
    requested_agent = context.requested_agent
    if not requested_agent:
        return None
    if requested_agent in context.registrations_by_name:
        return AgentRouteSelection(
            agent_name=requested_agent,
            policy_step=ROUTING_POLICY_REQUESTED_AGENT,
            diagnostics=(
                {
                    "kind": "requested_agent",
                    "policy_step": ROUTING_POLICY_REQUESTED_AGENT,
                    "selected_agent": requested_agent,
                    "reason": f"Explicit requested agent `{requested_agent}` was honored.",
                },
            ),
        )
    warnings.append(f"Requested agent `{requested_agent}` is not active; falling back to gateway policy.")
    return None


def _select_explicit_forked_skill_delegate(
    context: RoutingPolicyContext,
    warnings: list[str],
) -> AgentRouteSelection | None:
    explicit_delegate_agents = [
        normalize_agent_name(definition.delegate_agent, context.general_assistant_name)
        for definition in context.explicit_skill_definitions
        if definition.execution_mode == "forked" and definition.delegate_agent
    ]
    explicit_delegate_agents = [name for name in explicit_delegate_agents if name in context.registrations_by_name]
    if not explicit_delegate_agents:
        return None
    return _choose_from_candidates(
        context,
        tuple(dict.fromkeys(explicit_delegate_agents)),
        warnings,
        policy_step=ROUTING_POLICY_FORKED_SKILL_DELEGATE,
        reason_prefix="Selected from explicit forked skill delegates",
    )


def _select_explicit_forked_skill_fallback(
    context: RoutingPolicyContext,
    warnings: list[str],
) -> AgentRouteSelection | None:
    if not any(definition.execution_mode == "forked" for definition in context.explicit_skill_definitions):
        return None
    selected_agent = _fallback_general_assistant(
        context,
        warnings,
        "Forked skill could not resolve an active `delegate_agent`; GeneralAssistant fallback applied.",
    )
    return AgentRouteSelection(
        agent_name=selected_agent,
        policy_step=ROUTING_POLICY_FORKED_SKILL_FALLBACK,
        diagnostics=(
            {
                "kind": "forked_skill_fallback",
                "policy_step": ROUTING_POLICY_FORKED_SKILL_FALLBACK,
                "selected_agent": selected_agent,
                "reason": "Forked skill could not resolve an active `delegate_agent`; used GeneralAssistant fallback.",
            },
        ),
    )


def _select_explicit_inline_skill_candidate(
    context: RoutingPolicyContext,
    warnings: list[str],
) -> AgentRouteSelection | None:
    explicit_inline_candidates: list[str] = []
    for definition in context.explicit_skill_definitions:
        if definition.execution_mode != "inline":
            continue
        explicit_inline_candidates.extend(
            normalize_agent_name(agent_name, context.general_assistant_name)
            for agent_name in definition.available_to_agents
            if normalize_agent_name(agent_name, context.general_assistant_name) in context.registrations_by_name
        )
    explicit_inline_candidates = list(dict.fromkeys(explicit_inline_candidates))
    if not explicit_inline_candidates:
        return None
    return _choose_from_candidates(
        context,
        tuple(explicit_inline_candidates),
        warnings,
        policy_step=ROUTING_POLICY_INLINE_SKILL_COMPATIBILITY,
        reason_prefix="Selected from explicit inline skill compatibility",
    )


def _select_pending_action_owner(
    context: RoutingPolicyContext,
    warnings: list[str],
) -> AgentRouteSelection | None:
    pending_action = get_pending_action(context.state)
    if not is_pending_action_active(pending_action):
        return None

    owner_agent = normalize_agent_name(
        str(pending_action.get("requested_by_agent", "")).strip(),
        context.general_assistant_name,
    )
    if owner_agent in context.registrations_by_name:
        return AgentRouteSelection(
            agent_name=owner_agent,
            policy_step=ROUTING_POLICY_PENDING_ACTION_OWNER,
            diagnostics=(
                {
                    "kind": "pending_action",
                    "policy_step": ROUTING_POLICY_PENDING_ACTION_OWNER,
                    "selected_agent": owner_agent,
                    "reason": f"Active pending action is owned by `{owner_agent}`.",
                },
            ),
        )

    warnings.append(f"Pending action owner `{owner_agent}` is not active; falling back to gateway policy.")
    return None


def _select_state_route(
    context: RoutingPolicyContext,
    warnings: list[str],
) -> AgentRouteSelection | None:
    document_conversion_agent = "document_conversion_agent"
    if document_conversion_agent not in context.registrations_by_name:
        return None

    uploaded_files = context.state.get("uploaded_files")
    if isinstance(uploaded_files, list) and uploaded_files:
        return AgentRouteSelection(
            agent_name=document_conversion_agent,
            policy_step=ROUTING_POLICY_STATE_ROUTE,
            diagnostics=(
                {
                    "kind": "state_route",
                    "policy_step": ROUTING_POLICY_STATE_ROUTE,
                    "selected_agent": document_conversion_agent,
                    "reason": "Uploaded files require document conversion handling.",
                },
            ),
        )

    conversion_session_id = str(context.state.get("conversion_session_id", "")).strip()
    if conversion_session_id:
        return AgentRouteSelection(
            agent_name=document_conversion_agent,
            policy_step=ROUTING_POLICY_STATE_ROUTE,
            diagnostics=(
                {
                    "kind": "state_route",
                    "policy_step": ROUTING_POLICY_STATE_ROUTE,
                    "selected_agent": document_conversion_agent,
                    "reason": "Active conversion session should stay on document conversion.",
                },
            ),
        )

    return None


def _select_model_router(
    context: RoutingPolicyContext,
    warnings: list[str],
) -> AgentRouteSelection | None:
    if context.model_router is None:
        return None

    selection = context.model_router.select_specialist(
        agent_registrations=context.agent_registrations,
        general_assistant_name=context.general_assistant_name,
        latest_user_text=context.latest_user_text,
        state=context.state,
    )
    if selection is None:
        return None

    selected_agent, diagnostics = selection
    return AgentRouteSelection(
        agent_name=selected_agent,
        policy_step=ROUTING_POLICY_MODEL_ROUTER,
        diagnostics=tuple(diagnostics),
    )


def _select_general_fallback(
    context: RoutingPolicyContext,
    warnings: list[str],
) -> AgentRouteSelection:
    selected_agent = _fallback_general_assistant(
        context,
        warnings,
        "No specialist route applied; GeneralAssistant fallback applied.",
    )
    return AgentRouteSelection(
        agent_name=selected_agent,
        policy_step=ROUTING_POLICY_GENERAL_FALLBACK,
        diagnostics=(
            {
                "kind": "fallback",
                "policy_step": ROUTING_POLICY_GENERAL_FALLBACK,
                "selected_agent": selected_agent,
                "reason": "No specialist route applied; used GeneralAssistant fallback.",
            },
        ),
    )


def _choose_from_candidates(
    context: RoutingPolicyContext,
    candidate_agent_names: tuple[str, ...],
    warnings: list[str],
    *,
    policy_step: str,
    reason_prefix: str,
) -> AgentRouteSelection:
    candidates = [
        context.registrations_by_name[name]
        for name in candidate_agent_names
        if name in context.registrations_by_name
    ]
    if not candidates:
        selected_agent = _fallback_general_assistant(
            context,
            warnings,
            f"{reason_prefix}; GeneralAssistant fallback applied because no valid candidate remained.",
        )
        return AgentRouteSelection(
            agent_name=selected_agent,
            policy_step=policy_step,
            diagnostics=(
                {
                    "kind": "candidate_fallback",
                    "policy_step": policy_step,
                    "selected_agent": selected_agent,
                    "reason": f"{reason_prefix}; no valid candidate remained.",
                },
            ),
        )

    diagnostics: list[dict[str, Any]] = []
    for registration in candidates:
        diagnostics.append(
            {
                "kind": "candidate_available",
                "policy_step": policy_step,
                "agent": registration.name,
                "selection_order": registration.selection_order,
            }
        )

    selected_registration = sorted(
        candidates,
        key=lambda item: (
            item.selection_order,
            item.name,
        ),
    )[0]
    diagnostics.append(
        {
            "kind": "candidate_selected",
            "policy_step": policy_step,
            "selected_agent": selected_registration.name,
            "reason": reason_prefix,
        }
    )
    return AgentRouteSelection(
        agent_name=selected_registration.name,
        policy_step=policy_step,
        diagnostics=tuple(diagnostics),
    )


def _fallback_general_assistant(
    context: RoutingPolicyContext,
    warnings: list[str],
    warning_text: str,
) -> str:
    if context.general_assistant_name and context.general_assistant_name in context.registrations_by_name:
        return context.general_assistant_name

    ordered_registrations = sorted(
        context.agent_registrations,
        key=lambda item: (item.selection_order, item.name),
    )
    if not ordered_registrations:
        return context.default_route

    warnings.append(f"{warning_text} GeneralAssistant is unavailable; used `{ordered_registrations[0].name}` instead.")
    return ordered_registrations[0].name
