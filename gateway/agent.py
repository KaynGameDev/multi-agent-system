from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

from app.conversation_mode import (
    CONVERSATION_MODE_COMMAND_AGENT,
    ConversationModeCommand,
    parse_conversation_mode_command,
)
from app.contracts import (
    AgentRouteDecision,
    ExecutionContract,
    SkillInvocationContract,
    build_routing_decision,
    validate_agent_route_decision,
)
from app.interpretation.intent_parser import IntentParser
from app.messages import extract_latest_human_text, render_message_for_routing_context
from app.pending_actions import PendingAction, expire_pending_action
from app.routing.agent_router import AgentRouter, AgentRouterResult
from app.routing.pending_action_router import (
    PendingActionRouter,
    PendingActionTurnResult,
    resolve_pending_action_turn_from_state,
)
from app.skill_runtime import build_skill_runtime_state
from app.skills import SkillRegistry
from app.state import AgentState
from gateway.skill_policy import (
    SelectedSkillRuntime,
    normalize_context_paths,
    normalize_requested_skill_ids,
    resolve_explicit_skill_definitions,
    select_skills_for_agent,
)

logger = logging.getLogger(__name__)

ROUTE_POLICY_REQUESTED_AGENT = "requested_agent"
ROUTE_POLICY_FORKED_SKILL_DELEGATE = "forked_skill_delegate"
ROUTE_POLICY_FORKED_SKILL_FALLBACK = "forked_skill_fallback"
ROUTE_POLICY_INLINE_SKILL_COMPATIBILITY = "inline_skill_compatibility"
ROUTE_POLICY_PENDING_ACTION_OWNER = "pending_action_owner"
ROUTE_POLICY_PENDING_ACTION_OWNER_FALLBACK = "pending_action_owner_fallback"
ROUTE_POLICY_CONVERSATION_MODE_COMMAND = "conversation_mode_command"


@dataclass(frozen=True)
class GatewayRequestContext:
    latest_user_text: str
    context_paths: tuple[str, ...]
    requested_skill_ids: tuple[str, ...]
    requested_agent: str


@dataclass(frozen=True)
class GatewayRouteSelection:
    selected_agent: str
    policy_step: str
    route_reason: str
    diagnostics: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    agent_route_decision: AgentRouteDecision


@dataclass(frozen=True)
class _PendingActionContext:
    pending_action_turn: PendingActionTurnResult | None
    pending_action_decision: dict[str, Any] | None
    pending_action_resolution_key: str | None
    execution_contract: ExecutionContract | None


@dataclass(frozen=True)
class _SkillSelectionResult:
    resolved_skill_ids: tuple[str, ...]
    skill_invocation_contracts: tuple[SkillInvocationContract, ...]
    skill_runtime_state: dict[str, Any]
    warnings: list[str]
    skill_diagnostics: list[dict[str, Any]]


class GatewayNode:
    def __init__(
        self,
        _llm=None,
        *,
        agent_registrations: tuple[Any, ...],
        default_route: str,
        skill_registry: SkillRegistry,
        pending_action_router: PendingActionRouter | None = None,
        agent_router: AgentRouter | None = None,
    ) -> None:
        self.agent_registrations = tuple(agent_registrations)
        self.default_route = default_route
        self.skill_registry = skill_registry
        self.pending_action_router = pending_action_router
        self.agent_router = agent_router or AgentRouter(IntentParser(_llm))
        self.registrations_by_name = {registration.name: registration for registration in self.agent_registrations}
        self.general_assistant_name = resolve_general_assistant_name(
            self.agent_registrations,
            self.registrations_by_name,
        )

    def __call__(self, state: AgentState) -> dict[str, Any]:
        request_context = build_gateway_request_context(
            state,
            general_assistant_name=self.general_assistant_name,
        )
        conversation_mode_command = resolve_conversation_mode_command(
            latest_user_text=request_context.latest_user_text,
            current_requested_agent=request_context.requested_agent,
            registrations_by_name=self.registrations_by_name,
        )
        routing_state = dict(state)

        pa_ctx = self._resolve_pending_action_context(routing_state)

        explicit_skill_definitions, skill_diagnostics, warnings = resolve_explicit_skill_definitions(
            self.skill_registry,
            requested_skill_ids=request_context.requested_skill_ids,
            context_paths=request_context.context_paths,
        )
        route_selection = resolve_gateway_route_selection(
            pending_action_turn=pa_ctx.pending_action_turn,
            latest_user_text=request_context.latest_user_text,
            requested_agent=request_context.requested_agent,
            conversation_mode_command=conversation_mode_command,
            explicit_skill_definitions=explicit_skill_definitions,
            state=routing_state,
            agent_router=self.agent_router,
            agent_registrations=self.agent_registrations,
            registrations_by_name=self.registrations_by_name,
            general_assistant_name=self.general_assistant_name,
            default_route=self.default_route,
            warnings=warnings,
        )

        selected_agent = route_selection.selected_agent
        policy_step = route_selection.policy_step
        agent_diagnostics = list(route_selection.diagnostics)
        warnings = list(route_selection.warnings)

        skill_result = self._select_skills_and_contracts(
            selected_agent=selected_agent,
            latest_user_text=request_context.latest_user_text,
            context_paths=request_context.context_paths,
            explicit_skill_definitions=explicit_skill_definitions,
            warnings=warnings,
            skill_diagnostics=skill_diagnostics,
        )

        effective_requested_agent = (
            conversation_mode_command.requested_agent
            if conversation_mode_command is not None
            else request_context.requested_agent
        )
        routing_decision = build_routing_decision(
            selected_agent,
            reason=route_selection.route_reason,
            policy_step=policy_step,
            warnings=skill_result.warnings,
            diagnostics=agent_diagnostics,
            selected_agent=selected_agent,
            requested_agent=effective_requested_agent,
            requested_skill_ids=list(request_context.requested_skill_ids),
            resolved_skill_ids=list(skill_result.resolved_skill_ids),
            skill_invocation_contracts=list(skill_result.skill_invocation_contracts),
        )
        logger.info(
            "Gateway selected route=%s policy_step=%s requested_agent=%s requested_skills=%s resolved_skills=%s warnings=%s",
            selected_agent,
            policy_step,
            effective_requested_agent,
            request_context.requested_skill_ids,
            skill_result.resolved_skill_ids,
            skill_result.warnings,
        )

        expired_pending_action, expiry_messages = self._handle_pending_action_expiry(
            policy_step, pa_ctx.pending_action_turn,
        )

        state_update: dict[str, Any] = {
            "route": selected_agent,
            "route_reason": route_selection.route_reason,
            "route_policy_step": policy_step,
            "pending_action_decision": pa_ctx.pending_action_decision,
            "pending_action_resolution_key": pa_ctx.pending_action_resolution_key,
            "agent_route_decision": route_selection.agent_route_decision.model_dump(),
            "execution_contract": pa_ctx.execution_contract,
            "requested_skill_ids": list(request_context.requested_skill_ids),
            "resolved_skill_ids": list(skill_result.resolved_skill_ids),
            "context_paths": list(request_context.context_paths),
            "skill_resolution_diagnostics": skill_result.skill_diagnostics,
            "agent_selection_diagnostics": agent_diagnostics,
            "selection_warnings": skill_result.warnings,
            "skill_invocation_contracts": list(skill_result.skill_invocation_contracts),
            "active_skill_invocation_contracts": skill_result.skill_runtime_state["active_skill_invocation_contracts"],
            "skill_execution_diagnostics": skill_result.skill_runtime_state["skill_execution_diagnostics"],
            "routing_decision": routing_decision,
        }
        if conversation_mode_command is not None:
            state_update["requested_agent"] = conversation_mode_command.requested_agent
        if expired_pending_action is not None:
            state_update["pending_action"] = expired_pending_action
        if expiry_messages:
            state_update["messages"] = expiry_messages
        return state_update

    def _resolve_pending_action_context(
        self,
        routing_state: dict[str, Any],
    ) -> _PendingActionContext:
        pending_action_turn = resolve_pending_action_turn_from_state(
            routing_state,
            pending_action_router=self.pending_action_router,
        )
        if pending_action_turn is not None:
            pending_action_decision = pending_action_turn.pending_action_decision.model_dump()
            pending_action_resolution_key = pending_action_turn.pending_action_resolution_key
            execution_contract = pending_action_turn.execution_contract
            routing_state["pending_action_decision"] = pending_action_decision
            routing_state["pending_action_resolution_key"] = pending_action_resolution_key
            routing_state["execution_contract"] = execution_contract
        else:
            pending_action_decision = None
            pending_action_resolution_key = None
            execution_contract = None
            routing_state["pending_action_decision"] = None
            routing_state["pending_action_resolution_key"] = None
            routing_state["execution_contract"] = None
        return _PendingActionContext(
            pending_action_turn=pending_action_turn,
            pending_action_decision=pending_action_decision,
            pending_action_resolution_key=pending_action_resolution_key,
            execution_contract=execution_contract,
        )

    def _select_skills_and_contracts(
        self,
        *,
        selected_agent: str,
        latest_user_text: str,
        context_paths: tuple[str, ...],
        explicit_skill_definitions: tuple[Any, ...],
        warnings: list[str],
        skill_diagnostics: list[dict[str, Any]],
    ) -> _SkillSelectionResult:
        selected_skills, selected_skill_diagnostics, warnings = select_skills_for_agent(
            self.skill_registry,
            agent_name=selected_agent,
            latest_user_text=latest_user_text,
            general_assistant_name=self.general_assistant_name,
            context_paths=context_paths,
            explicit_skill_definitions=explicit_skill_definitions,
            warnings=warnings,
        )
        resolved_skill_ids = tuple(item.definition.skill_id for item in selected_skills)
        skill_diagnostics.extend(selected_skill_diagnostics)
        skill_invocation_contracts = tuple(
            self.skill_registry.build_skill_invocation_contract_from_definition(
                item.definition,
                fallback_skill_id=item.definition.skill_id,
                target_agent=selected_agent,
                source=item.source,
                reason=item.reason,
                context_paths=context_paths,
            )
            for item in selected_skills
        )
        skill_runtime_state = build_skill_runtime_state(
            skill_invocation_contracts,
            agent_name=selected_agent,
        )
        return _SkillSelectionResult(
            resolved_skill_ids=resolved_skill_ids,
            skill_invocation_contracts=skill_invocation_contracts,
            skill_runtime_state=skill_runtime_state,
            warnings=warnings,
            skill_diagnostics=skill_diagnostics,
        )

    @staticmethod
    def _handle_pending_action_expiry(
        policy_step: str,
        pending_action_turn: PendingActionTurnResult | None,
    ) -> tuple[PendingAction | None, list[AIMessage]]:
        if policy_step != ROUTE_POLICY_PENDING_ACTION_OWNER_FALLBACK or pending_action_turn is None:
            return None, []
        expired_pending_action = expire_pending_action(pending_action_turn.pending_action)
        expiry_messages = [
            AIMessage(
                content=(
                    "The approval request could not be completed because the required "
                    "service is currently unavailable. The request has been cancelled."
                )
            )
        ]
        logger.warning(
            "Gateway expired orphaned pending action due to missing owner agent. "
            "pending_action_id=%s owner_agent=%s",
            str(expired_pending_action.get("id", "")).strip(),
            str(pending_action_turn.pending_action.get("requested_by_agent", "")).strip(),
        )
        return expired_pending_action, expiry_messages


__all__ = [
    "GatewayNode",
    "GatewayRequestContext",
    "GatewayRouteSelection",
    "SelectedSkillRuntime",
]


def build_gateway_request_context(
    state: AgentState,
    *,
    general_assistant_name: str,
) -> GatewayRequestContext:
    return GatewayRequestContext(
        latest_user_text=extract_latest_human_text(state),
        context_paths=normalize_context_paths(state),
        requested_skill_ids=normalize_requested_skill_ids(state),
        requested_agent=resolve_requested_agent(state, general_assistant_name=general_assistant_name),
    )


def resolve_requested_agent(
    state: AgentState,
    *,
    general_assistant_name: str,
) -> str:
    return normalize_agent_name(
        str(state.get("requested_agent") or "").strip(),
        general_assistant_name,
    )


def resolve_general_assistant_name(
    agent_registrations: tuple[Any, ...],
    registrations_by_name: dict[str, Any],
) -> str:
    for registration in agent_registrations:
        if getattr(registration, "is_general_assistant", False):
            return str(getattr(registration, "name", "")).strip()
    if "general_chat_agent" in registrations_by_name:
        return "general_chat_agent"
    return ""


def normalize_agent_name(raw_value: str, general_assistant_name: str) -> str:
    cleaned = str(raw_value or "").strip()
    if not cleaned:
        return ""
    if cleaned == "GeneralAssistant":
        return general_assistant_name or cleaned
    return cleaned


def resolve_gateway_route_selection(
    *,
    pending_action_turn: PendingActionTurnResult | None,
    latest_user_text: str,
    requested_agent: str,
    conversation_mode_command: ConversationModeCommand | None,
    explicit_skill_definitions: tuple[Any, ...],
    state: AgentState,
    agent_router: AgentRouter,
    agent_registrations: tuple[Any, ...],
    registrations_by_name: dict[str, Any],
    general_assistant_name: str,
    default_route: str,
    warnings: list[str],
) -> GatewayRouteSelection:
    resolved_warnings = list(warnings)
    if conversation_mode_command is not None:
        return resolve_conversation_mode_route_selection(
            conversation_mode_command,
            warnings=resolved_warnings,
        )

    if pending_action_turn is not None and not pending_action_turn.allow_fresh_routing:
        return resolve_pending_action_route_selection(
            pending_action_turn,
            agent_registrations=agent_registrations,
            registrations_by_name=registrations_by_name,
            general_assistant_name=general_assistant_name,
            default_route=default_route,
            warnings=resolved_warnings,
        )

    explicit_override = resolve_explicit_route_override(
        requested_agent=requested_agent,
        explicit_skill_definitions=explicit_skill_definitions,
        agent_registrations=agent_registrations,
        registrations_by_name=registrations_by_name,
        general_assistant_name=general_assistant_name,
        default_route=default_route,
        warnings=resolved_warnings,
    )
    if explicit_override is not None:
        return explicit_override

    parser_route = resolve_parser_route(
        agent_router,
        latest_user_text=latest_user_text,
        state=state,
        registrations_by_name=registrations_by_name,
        default_agent=general_assistant_name or default_route,
    )
    resolved_warnings.extend(parser_route.warnings)
    return GatewayRouteSelection(
        selected_agent=parser_route.selected_agent,
        policy_step=parser_route.policy_step,
        route_reason=parser_route.agent_route_decision.reason,
        diagnostics=tuple(parser_route.diagnostics),
        warnings=tuple(resolved_warnings),
        agent_route_decision=parser_route.agent_route_decision,
    )


def resolve_pending_action_route_selection(
    pending_action_turn: PendingActionTurnResult,
    *,
    agent_registrations: tuple[Any, ...],
    registrations_by_name: dict[str, Any],
    general_assistant_name: str,
    default_route: str,
    warnings: list[str],
) -> GatewayRouteSelection:
    owner_agent = normalize_agent_name(
        str(pending_action_turn.pending_action.get("requested_by_agent", "")).strip(),
        general_assistant_name,
    )
    pending_action_id = str(pending_action_turn.pending_action.get("id", "")).strip()
    if owner_agent in registrations_by_name:
        reason = f"Active pending action is owned by `{owner_agent}`."
        diagnostics = (
            build_gateway_route_diagnostic(
                kind="pending_action",
                policy_step=ROUTE_POLICY_PENDING_ACTION_OWNER,
                selected_agent=owner_agent,
                reason=reason,
                pending_action_id=pending_action_id,
                pending_action_decision=pending_action_turn.pending_action_decision.decision,
            ),
        )
        return GatewayRouteSelection(
            selected_agent=owner_agent,
            policy_step=ROUTE_POLICY_PENDING_ACTION_OWNER,
            route_reason=reason,
            diagnostics=diagnostics,
            warnings=tuple(warnings),
            agent_route_decision=build_gateway_agent_route_decision(
                selected_agent=owner_agent,
                reason=reason,
                fallback_used=False,
                diagnostics={
                    "kind": "pending_action",
                    "policy_step": ROUTE_POLICY_PENDING_ACTION_OWNER,
                    "pending_action_id": pending_action_id,
                    "decision": pending_action_turn.pending_action_decision.decision,
                },
            ),
        )

    selected_agent, fallback_warning = fallback_general_assistant(
        agent_registrations=agent_registrations,
        registrations_by_name=registrations_by_name,
        general_assistant_name=general_assistant_name,
        default_route=default_route,
        warning_prefix=f"Pending action owner `{owner_agent}` is not active; pending-action fallback applied.",
    )
    if fallback_warning:
        warnings.append(fallback_warning)
    reason = f"Pending action owner `{owner_agent}` is not active; fallback to `{selected_agent}`."
    warnings.append(reason)
    diagnostics = (
        build_gateway_route_diagnostic(
            kind="pending_action_fallback",
            policy_step=ROUTE_POLICY_PENDING_ACTION_OWNER_FALLBACK,
            selected_agent=selected_agent,
            reason=reason,
            pending_action_id=pending_action_id,
            pending_action_decision=pending_action_turn.pending_action_decision.decision,
        ),
    )
    return GatewayRouteSelection(
        selected_agent=selected_agent,
        policy_step=ROUTE_POLICY_PENDING_ACTION_OWNER_FALLBACK,
        route_reason=reason,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
        agent_route_decision=build_gateway_agent_route_decision(
            selected_agent=selected_agent,
            reason=reason,
            fallback_used=True,
            diagnostics={
                "kind": "pending_action_fallback",
                "policy_step": ROUTE_POLICY_PENDING_ACTION_OWNER_FALLBACK,
                "pending_action_id": pending_action_id,
                "decision": pending_action_turn.pending_action_decision.decision,
            },
        ),
    )


def resolve_conversation_mode_command(
    *,
    latest_user_text: str,
    current_requested_agent: str,
    registrations_by_name: dict[str, Any],
) -> ConversationModeCommand | None:
    command = parse_conversation_mode_command(
        latest_user_text,
        current_requested_agent=current_requested_agent,
    )
    if command is None:
        return None
    if CONVERSATION_MODE_COMMAND_AGENT not in registrations_by_name:
        return None
    if command.requested_agent and command.requested_agent not in registrations_by_name:
        return None
    return command


def resolve_conversation_mode_route_selection(
    command: ConversationModeCommand,
    *,
    warnings: list[str],
) -> GatewayRouteSelection:
    mode_state = "enabled" if command.requested_agent else "disabled"
    reason = f"Conversation mode command {mode_state} `{command.mode}`."
    diagnostics = (
        build_gateway_route_diagnostic(
            kind="conversation_mode_command",
            policy_step=ROUTE_POLICY_CONVERSATION_MODE_COMMAND,
            selected_agent=CONVERSATION_MODE_COMMAND_AGENT,
            reason=reason,
            mode=command.mode,
            action=command.action,
            requested_agent=command.requested_agent,
        ),
    )
    return GatewayRouteSelection(
        selected_agent=CONVERSATION_MODE_COMMAND_AGENT,
        policy_step=ROUTE_POLICY_CONVERSATION_MODE_COMMAND,
        route_reason=reason,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
        agent_route_decision=build_gateway_agent_route_decision(
            selected_agent=CONVERSATION_MODE_COMMAND_AGENT,
            reason=reason,
            fallback_used=False,
            diagnostics={
                "kind": "conversation_mode_command",
                "policy_step": ROUTE_POLICY_CONVERSATION_MODE_COMMAND,
                "mode": command.mode,
                "action": command.action,
                "requested_agent": command.requested_agent,
            },
        ),
    )


def resolve_explicit_route_override(
    *,
    requested_agent: str,
    explicit_skill_definitions: tuple[Any, ...],
    agent_registrations: tuple[Any, ...],
    registrations_by_name: dict[str, Any],
    general_assistant_name: str,
    default_route: str,
    warnings: list[str],
) -> GatewayRouteSelection | None:
    requested_route = resolve_requested_agent_route_selection(
        requested_agent=requested_agent,
        warnings=warnings,
        registrations_by_name=registrations_by_name,
    )
    if requested_route is not None:
        return requested_route

    forked_skill_route = resolve_forked_skill_route_selection(
        explicit_skill_definitions=explicit_skill_definitions,
        agent_registrations=agent_registrations,
        registrations_by_name=registrations_by_name,
        general_assistant_name=general_assistant_name,
        default_route=default_route,
        warnings=warnings,
    )
    if forked_skill_route is not None:
        return forked_skill_route

    return resolve_inline_skill_route_selection(
        explicit_skill_definitions=explicit_skill_definitions,
        agent_registrations=agent_registrations,
        registrations_by_name=registrations_by_name,
        general_assistant_name=general_assistant_name,
        warnings=warnings,
    )


def resolve_requested_agent_route_selection(
    *,
    requested_agent: str,
    warnings: list[str],
    registrations_by_name: dict[str, Any],
) -> GatewayRouteSelection | None:
    if not requested_agent:
        return None
    if requested_agent not in registrations_by_name:
        warnings.append(f"Requested agent `{requested_agent}` is not active; continuing with parser routing.")
        return None

    reason = f"Explicit requested agent `{requested_agent}` was honored."
    diagnostics = (
        build_gateway_route_diagnostic(
            kind="requested_agent",
            policy_step=ROUTE_POLICY_REQUESTED_AGENT,
            selected_agent=requested_agent,
            reason=reason,
        ),
    )
    return GatewayRouteSelection(
        selected_agent=requested_agent,
        policy_step=ROUTE_POLICY_REQUESTED_AGENT,
        route_reason=reason,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
        agent_route_decision=build_gateway_agent_route_decision(
            selected_agent=requested_agent,
            reason=reason,
            fallback_used=False,
            diagnostics={
                "kind": "requested_agent",
                "policy_step": ROUTE_POLICY_REQUESTED_AGENT,
            },
        ),
    )


def resolve_forked_skill_route_selection(
    *,
    explicit_skill_definitions: tuple[Any, ...],
    agent_registrations: tuple[Any, ...],
    registrations_by_name: dict[str, Any],
    general_assistant_name: str,
    default_route: str,
    warnings: list[str],
) -> GatewayRouteSelection | None:
    forked_skill_definitions = tuple(
        definition
        for definition in explicit_skill_definitions
        if str(getattr(definition, "execution_mode", "")).strip().lower() == "forked"
    )
    if not forked_skill_definitions:
        return None

    candidate_agent_names = tuple(
        name
        for name in dict.fromkeys(
            normalize_agent_name(str(getattr(definition, "delegate_agent", "")).strip(), general_assistant_name)
            for definition in forked_skill_definitions
            if str(getattr(definition, "delegate_agent", "")).strip()
        )
        if name in registrations_by_name
    )
    if candidate_agent_names:
        selected_agent = select_best_candidate_agent(
            candidate_agent_names,
            agent_registrations=agent_registrations,
            registrations_by_name=registrations_by_name,
        )
        reason = f"Selected `{selected_agent}` from explicit forked skill delegates."
        diagnostics = (
            build_gateway_route_diagnostic(
                kind="forked_skill_delegate",
                policy_step=ROUTE_POLICY_FORKED_SKILL_DELEGATE,
                selected_agent=selected_agent,
                reason=reason,
            ),
        )
        return GatewayRouteSelection(
            selected_agent=selected_agent,
            policy_step=ROUTE_POLICY_FORKED_SKILL_DELEGATE,
            route_reason=reason,
            diagnostics=diagnostics,
            warnings=tuple(warnings),
            agent_route_decision=build_gateway_agent_route_decision(
                selected_agent=selected_agent,
                reason=reason,
                fallback_used=False,
                diagnostics={
                    "kind": "forked_skill_delegate",
                    "policy_step": ROUTE_POLICY_FORKED_SKILL_DELEGATE,
                },
            ),
        )

    selected_agent, fallback_warning = fallback_general_assistant(
        agent_registrations=agent_registrations,
        registrations_by_name=registrations_by_name,
        general_assistant_name=general_assistant_name,
        default_route=default_route,
        warning_prefix="Forked skill could not resolve an active `delegate_agent`; fallback applied.",
    )
    if fallback_warning:
        warnings.append(fallback_warning)
    reason = f"Forked skill could not resolve an active `delegate_agent`; fallback to `{selected_agent}`."
    diagnostics = (
        build_gateway_route_diagnostic(
            kind="forked_skill_fallback",
            policy_step=ROUTE_POLICY_FORKED_SKILL_FALLBACK,
            selected_agent=selected_agent,
            reason=reason,
        ),
    )
    return GatewayRouteSelection(
        selected_agent=selected_agent,
        policy_step=ROUTE_POLICY_FORKED_SKILL_FALLBACK,
        route_reason=reason,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
        agent_route_decision=build_gateway_agent_route_decision(
            selected_agent=selected_agent,
            reason=reason,
            fallback_used=True,
            diagnostics={
                "kind": "forked_skill_fallback",
                "policy_step": ROUTE_POLICY_FORKED_SKILL_FALLBACK,
            },
        ),
    )


def resolve_inline_skill_route_selection(
    *,
    explicit_skill_definitions: tuple[Any, ...],
    agent_registrations: tuple[Any, ...],
    registrations_by_name: dict[str, Any],
    general_assistant_name: str,
    warnings: list[str],
) -> GatewayRouteSelection | None:
    candidate_agent_names = tuple(
        name
        for name in dict.fromkeys(
            normalize_agent_name(str(agent_name).strip(), general_assistant_name)
            for definition in explicit_skill_definitions
            if str(getattr(definition, "execution_mode", "inline")).strip().lower() == "inline"
            for agent_name in tuple(getattr(definition, "available_to_agents", ()) or ())
        )
        if name in registrations_by_name
    )
    if not candidate_agent_names:
        return None

    selected_agent = select_best_candidate_agent(
        candidate_agent_names,
        agent_registrations=agent_registrations,
        registrations_by_name=registrations_by_name,
    )
    reason = f"Selected `{selected_agent}` from explicit inline skill compatibility."
    diagnostics = (
        build_gateway_route_diagnostic(
            kind="inline_skill_compatibility",
            policy_step=ROUTE_POLICY_INLINE_SKILL_COMPATIBILITY,
            selected_agent=selected_agent,
            reason=reason,
        ),
    )
    return GatewayRouteSelection(
        selected_agent=selected_agent,
        policy_step=ROUTE_POLICY_INLINE_SKILL_COMPATIBILITY,
        route_reason=reason,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
        agent_route_decision=build_gateway_agent_route_decision(
            selected_agent=selected_agent,
            reason=reason,
            fallback_used=False,
            diagnostics={
                "kind": "inline_skill_compatibility",
                "policy_step": ROUTE_POLICY_INLINE_SKILL_COMPATIBILITY,
            },
        ),
    )


def collect_recent_messages_for_parser(state: AgentState) -> list[str]:
    messages = state.get("messages") or []
    rendered: list[str] = []
    for message in messages[-6:]:
        text = render_message_for_routing_context(message)
        if text:
            rendered.append(text)
    return rendered


def build_top_level_routing_context(state: AgentState) -> dict[str, Any]:
    uploaded_files = state.get("uploaded_files")
    return {
        "interface_name": str(state.get("interface_name", "")).strip().lower() or "unknown",
        "uploaded_files_count": len(uploaded_files) if isinstance(uploaded_files, list) else 0,
        "conversion_session_active": bool(str(state.get("conversion_session_id", "")).strip()),
    }


def resolve_parser_route(
    agent_router: AgentRouter,
    *,
    latest_user_text: str,
    state: AgentState,
    registrations_by_name: dict[str, Any],
    default_agent: str,
) -> AgentRouterResult:
    return agent_router.route_request(
        latest_user_text=latest_user_text,
        recent_messages=collect_recent_messages_for_parser(state),
        routing_context=build_top_level_routing_context(state),
        registrations_by_name=registrations_by_name,
        default_agent=default_agent,
    )


def select_best_candidate_agent(
    candidate_agent_names: tuple[str, ...],
    *,
    agent_registrations: tuple[Any, ...],
    registrations_by_name: dict[str, Any],
) -> str:
    candidates = [
        registrations_by_name[name]
        for name in candidate_agent_names
        if name in registrations_by_name
    ]
    if not candidates:
        return candidate_agent_names[0]
    selected_registration = sorted(
        candidates,
        key=lambda item: (
            int(getattr(item, "selection_order", 100)),
            str(getattr(item, "name", "")),
        ),
    )[0]
    return str(getattr(selected_registration, "name", "")).strip() or candidate_agent_names[0]


def fallback_general_assistant(
    *,
    agent_registrations: tuple[Any, ...],
    registrations_by_name: dict[str, Any],
    general_assistant_name: str,
    default_route: str,
    warning_prefix: str,
) -> tuple[str, str | None]:
    if general_assistant_name and general_assistant_name in registrations_by_name:
        return general_assistant_name, None

    ordered_registrations = sorted(
        agent_registrations,
        key=lambda item: (
            int(getattr(item, "selection_order", 100)),
            str(getattr(item, "name", "")),
        ),
    )
    if not ordered_registrations:
        return default_route, None

    selected_agent = str(getattr(ordered_registrations[0], "name", "")).strip() or default_route
    warning = f"{warning_prefix} GeneralAssistant is unavailable; used `{selected_agent}` instead."
    return selected_agent, warning


def build_gateway_route_diagnostic(
    *,
    kind: str,
    policy_step: str,
    selected_agent: str,
    reason: str,
    **extra: Any,
) -> dict[str, Any]:
    diagnostic = {
        "kind": kind,
        "policy_step": policy_step,
        "selected_agent": selected_agent,
        "reason": reason,
    }
    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        diagnostic[key] = value
    return diagnostic


def build_gateway_agent_route_decision(
    *,
    selected_agent: str,
    reason: str,
    fallback_used: bool,
    diagnostics: dict[str, Any],
) -> AgentRouteDecision:
    return validate_agent_route_decision(
        {
            "selected_agent": selected_agent,
            "reason": reason,
            "fallback_used": fallback_used,
            "diagnostics": diagnostics,
        }
    )
