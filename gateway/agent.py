from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

from app.contracts import (
    AgentRouteDecision,
    ExecutionContract,
    SkillInvocationContract,
    build_routing_decision,
    validate_agent_route_decision,
)
from app.messages import extract_latest_human_text
from app.pending_actions import PendingAction, expire_pending_action
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

ROUTE_POLICY_SINGLE_AGENT = "single_agent_kbb"
ROUTE_POLICY_PENDING_ACTION_OWNER = "pending_action_owner"
ROUTE_POLICY_PENDING_ACTION_OWNER_FALLBACK = "pending_action_owner_fallback"


@dataclass(frozen=True)
class GatewayRequestContext:
    latest_user_text: str
    context_paths: tuple[str, ...]
    requested_skill_ids: tuple[str, ...]


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
        agent_router=None,
    ) -> None:
        del _llm, agent_router
        self.agent_registrations = tuple(agent_registrations)
        self.default_route = default_route
        self.skill_registry = skill_registry
        self.pending_action_router = pending_action_router
        self.registrations_by_name = {
            registration.name: registration for registration in self.agent_registrations
        }

    def __call__(self, state: AgentState) -> dict[str, Any]:
        request_context = build_gateway_request_context(state)
        routing_state = dict(state)

        pa_ctx = self._resolve_pending_action_context(routing_state)
        explicit_skill_definitions, skill_diagnostics, warnings = resolve_explicit_skill_definitions(
            self.skill_registry,
            requested_skill_ids=request_context.requested_skill_ids,
            context_paths=request_context.context_paths,
        )
        route_selection = resolve_single_agent_route_selection(
            pending_action_turn=pa_ctx.pending_action_turn,
            default_route=self.default_route,
            registrations_by_name=self.registrations_by_name,
            warnings=warnings,
        )

        skill_result = self._select_skills_and_contracts(
            selected_agent=route_selection.selected_agent,
            latest_user_text=request_context.latest_user_text,
            context_paths=request_context.context_paths,
            explicit_skill_definitions=explicit_skill_definitions,
            warnings=list(route_selection.warnings),
            skill_diagnostics=skill_diagnostics,
        )

        routing_decision = build_routing_decision(
            route_selection.selected_agent,
            reason=route_selection.route_reason,
            policy_step=route_selection.policy_step,
            warnings=skill_result.warnings,
            diagnostics=list(route_selection.diagnostics),
            selected_agent=route_selection.selected_agent,
            requested_agent=route_selection.selected_agent,
            requested_skill_ids=list(request_context.requested_skill_ids),
            resolved_skill_ids=list(skill_result.resolved_skill_ids),
            skill_invocation_contracts=list(skill_result.skill_invocation_contracts),
        )
        logger.info(
            "Gateway selected route=%s policy_step=%s requested_skills=%s resolved_skills=%s warnings=%s",
            route_selection.selected_agent,
            route_selection.policy_step,
            request_context.requested_skill_ids,
            skill_result.resolved_skill_ids,
            skill_result.warnings,
        )

        expired_pending_action, expiry_messages = self._handle_pending_action_expiry(
            route_selection.policy_step,
            pa_ctx.pending_action_turn,
        )

        state_update: dict[str, Any] = {
            "route": route_selection.selected_agent,
            "route_reason": route_selection.route_reason,
            "route_policy_step": route_selection.policy_step,
            "requested_agent": route_selection.selected_agent,
            "pending_action_decision": pa_ctx.pending_action_decision,
            "pending_action_resolution_key": pa_ctx.pending_action_resolution_key,
            "agent_route_decision": route_selection.agent_route_decision.model_dump(),
            "execution_contract": pa_ctx.execution_contract,
            "requested_skill_ids": list(request_context.requested_skill_ids),
            "resolved_skill_ids": list(skill_result.resolved_skill_ids),
            "context_paths": list(request_context.context_paths),
            "skill_resolution_diagnostics": skill_result.skill_diagnostics,
            "agent_selection_diagnostics": list(route_selection.diagnostics),
            "selection_warnings": skill_result.warnings,
            "skill_invocation_contracts": list(skill_result.skill_invocation_contracts),
            "active_skill_invocation_contracts": skill_result.skill_runtime_state["active_skill_invocation_contracts"],
            "skill_execution_diagnostics": skill_result.skill_runtime_state["skill_execution_diagnostics"],
            "routing_decision": routing_decision,
        }
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
            general_assistant_name=selected_agent,
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
                    "当前待审批动作属于一个已经下线的旧工作流。"
                    "我已取消这条旧审批，请重新告诉我你现在想推进的知识库动作。"
                )
            )
        ]
        logger.warning(
            "Gateway expired orphaned pending action due to single-agent runtime fallback. "
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


def build_gateway_request_context(state: AgentState) -> GatewayRequestContext:
    return GatewayRequestContext(
        latest_user_text=extract_latest_human_text(state),
        context_paths=normalize_context_paths(state),
        requested_skill_ids=normalize_requested_skill_ids(state),
    )


def resolve_single_agent_route_selection(
    *,
    pending_action_turn: PendingActionTurnResult | None,
    default_route: str,
    registrations_by_name: dict[str, Any],
    warnings: list[str],
) -> GatewayRouteSelection:
    owner_agent = (
        str(pending_action_turn.pending_action.get("requested_by_agent", "")).strip()
        if pending_action_turn is not None
        else ""
    )
    pending_action_id = (
        str(pending_action_turn.pending_action.get("id", "")).strip()
        if pending_action_turn is not None
        else ""
    )

    if pending_action_turn is not None and not pending_action_turn.allow_fresh_routing:
        if not owner_agent or owner_agent == default_route or owner_agent in registrations_by_name:
            reason = f"Active pending action continues on `{default_route}`."
            diagnostics = (
                build_gateway_route_diagnostic(
                    kind="pending_action",
                    policy_step=ROUTE_POLICY_PENDING_ACTION_OWNER,
                    selected_agent=default_route,
                    reason=reason,
                    pending_action_id=pending_action_id,
                    pending_action_decision=pending_action_turn.pending_action_decision.decision,
                ),
            )
            return GatewayRouteSelection(
                selected_agent=default_route,
                policy_step=ROUTE_POLICY_PENDING_ACTION_OWNER,
                route_reason=reason,
                diagnostics=diagnostics,
                warnings=tuple(warnings),
                agent_route_decision=build_gateway_agent_route_decision(
                    selected_agent=default_route,
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

        reason = (
            f"Pending action owner `{owner_agent}` is not active in the single-agent runtime; "
            f"fallback to `{default_route}`."
        )
        warnings.append(reason)
        diagnostics = (
            build_gateway_route_diagnostic(
                kind="pending_action_fallback",
                policy_step=ROUTE_POLICY_PENDING_ACTION_OWNER_FALLBACK,
                selected_agent=default_route,
                reason=reason,
                pending_action_id=pending_action_id,
                pending_action_decision=pending_action_turn.pending_action_decision.decision,
            ),
        )
        return GatewayRouteSelection(
            selected_agent=default_route,
            policy_step=ROUTE_POLICY_PENDING_ACTION_OWNER_FALLBACK,
            route_reason=reason,
            diagnostics=diagnostics,
            warnings=tuple(warnings),
            agent_route_decision=build_gateway_agent_route_decision(
                selected_agent=default_route,
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

    reason = f"Single-agent runtime: all fresh user turns route to `{default_route}`."
    diagnostics = (
        build_gateway_route_diagnostic(
            kind="single_agent",
            policy_step=ROUTE_POLICY_SINGLE_AGENT,
            selected_agent=default_route,
            reason=reason,
        ),
    )
    return GatewayRouteSelection(
        selected_agent=default_route,
        policy_step=ROUTE_POLICY_SINGLE_AGENT,
        route_reason=reason,
        diagnostics=diagnostics,
        warnings=tuple(warnings),
        agent_route_decision=build_gateway_agent_route_decision(
            selected_agent=default_route,
            reason=reason,
            fallback_used=False,
            diagnostics={
                "kind": "single_agent",
                "policy_step": ROUTE_POLICY_SINGLE_AGENT,
            },
        ),
    )


def build_gateway_route_diagnostic(
    *,
    kind: str,
    policy_step: str,
    selected_agent: str,
    reason: str,
    pending_action_id: str = "",
    pending_action_decision: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": kind,
        "policy_step": policy_step,
        "selected_agent": selected_agent,
        "reason": reason,
    }
    if pending_action_id:
        payload["pending_action_id"] = pending_action_id
    if pending_action_decision:
        payload["pending_action_decision"] = pending_action_decision
    return payload


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
