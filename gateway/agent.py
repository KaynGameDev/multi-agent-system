from __future__ import annotations

import logging
from typing import Any

from app.contracts import build_routing_decision
from app.messages import extract_latest_human_text
from app.skill_runtime import build_skill_runtime_state
from app.skills import SkillRegistry
from app.state import AgentState
from gateway.matchers import (
    AgentMatchResult,
    document_conversion_matcher,
    general_chat_matcher,
    knowledge_base_builder_matcher,
    knowledge_matcher,
    project_task_matcher_factory,
)
from gateway.routing_policy import (
    RoutingPolicyContext,
    build_route_reason,
    normalize_agent_name,
    resolve_general_assistant_name,
    select_agent_by_policy,
)
from gateway.skill_policy import (
    SelectedSkillRuntime,
    normalize_context_paths,
    normalize_requested_skill_ids,
    resolve_explicit_skill_definitions,
    select_skills_for_agent,
)

logger = logging.getLogger(__name__)


class GatewayNode:
    def __init__(
        self,
        _llm=None,
        *,
        agent_registrations: tuple[Any, ...],
        default_route: str,
        skill_registry: SkillRegistry,
    ) -> None:
        self.agent_registrations = tuple(agent_registrations)
        self.default_route = default_route
        self.skill_registry = skill_registry
        self.registrations_by_name = {registration.name: registration for registration in self.agent_registrations}
        self.general_assistant_name = resolve_general_assistant_name(
            self.agent_registrations,
            self.registrations_by_name,
        )

    def __call__(self, state: AgentState) -> dict[str, Any]:
        latest_user_text = extract_latest_human_text(state)
        context_paths = normalize_context_paths(state)
        requested_skill_ids = normalize_requested_skill_ids(state)
        requested_agent = normalize_agent_name(
            str(state.get("requested_agent") or "").strip(),
            self.general_assistant_name,
        )

        explicit_skill_definitions, skill_diagnostics, warnings = resolve_explicit_skill_definitions(
            self.skill_registry,
            requested_skill_ids=requested_skill_ids,
            context_paths=context_paths,
        )
        selected_agent, policy_step, agent_diagnostics, warnings = select_agent_by_policy(
            RoutingPolicyContext(
                state=state,
                latest_user_text=latest_user_text,
                requested_agent=requested_agent,
                explicit_skill_definitions=explicit_skill_definitions,
                agent_registrations=self.agent_registrations,
                registrations_by_name=self.registrations_by_name,
                default_route=self.default_route,
                general_assistant_name=self.general_assistant_name,
            ),
            warnings=warnings,
        )

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

        route_reason = build_route_reason(agent_diagnostics, selected_agent)
        routing_decision = build_routing_decision(
            selected_agent,
            reason=route_reason,
            policy_step=policy_step,
            warnings=warnings,
            diagnostics=agent_diagnostics,
            selected_agent=selected_agent,
            requested_agent=requested_agent,
            requested_skill_ids=list(requested_skill_ids),
            resolved_skill_ids=list(resolved_skill_ids),
            skill_invocation_contracts=list(skill_invocation_contracts),
        )
        logger.info(
            "Gateway selected route=%s policy_step=%s requested_agent=%s requested_skills=%s resolved_skills=%s warnings=%s",
            selected_agent,
            policy_step,
            requested_agent,
            requested_skill_ids,
            resolved_skill_ids,
            warnings,
        )

        return {
            "route": selected_agent,
            "route_reason": route_reason,
            "route_policy_step": policy_step,
            "requested_skill_ids": list(requested_skill_ids),
            "resolved_skill_ids": list(resolved_skill_ids),
            "context_paths": list(context_paths),
            "skill_resolution_diagnostics": skill_diagnostics,
            "agent_selection_diagnostics": agent_diagnostics,
            "selection_warnings": warnings,
            "skill_invocation_contracts": list(skill_invocation_contracts),
            "active_skill_invocation_contracts": skill_runtime_state["active_skill_invocation_contracts"],
            "skill_execution_diagnostics": skill_runtime_state["skill_execution_diagnostics"],
            "routing_decision": routing_decision,
        }


__all__ = [
    "AgentMatchResult",
    "GatewayNode",
    "SelectedSkillRuntime",
    "document_conversion_matcher",
    "general_chat_matcher",
    "knowledge_base_builder_matcher",
    "knowledge_matcher",
    "project_task_matcher_factory",
]
