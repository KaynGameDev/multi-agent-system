from __future__ import annotations

from typing import Any

from app.contracts import SkillInvocationContract, normalize_skill_invocation_contract
from app.prompt_loader import join_prompt_layers
from app.skills import SkillDefinition, SkillRegistry

def get_active_skill_invocation_contracts(
    state: dict[str, Any] | None,
    *,
    agent_name: str,
) -> tuple[SkillInvocationContract, ...]:
    if not isinstance(state, dict) or not agent_name:
        return ()

    active_contracts: list[SkillInvocationContract] = []
    seen_keys: set[tuple[str, str, str]] = set()
    source_contracts = state.get("active_skill_invocation_contracts")
    if not isinstance(source_contracts, list):
        source_contracts = state.get("skill_invocation_contracts") or []

    for raw_contract in source_contracts:
        if not isinstance(raw_contract, dict):
            continue
        contract = normalize_skill_invocation_contract(raw_contract)
        if not is_skill_contract_active_for_agent(contract, agent_name=agent_name):
            continue
        contract_key = (
            str(contract.get("skill_id", "")).strip(),
            str(contract.get("mode", "")).strip(),
            str(contract.get("target_agent", "")).strip() or agent_name,
        )
        if contract_key in seen_keys:
            continue
        seen_keys.add(contract_key)
        active_contracts.append(contract)
    return tuple(active_contracts)


def is_skill_contract_active_for_agent(
    contract: SkillInvocationContract,
    *,
    agent_name: str,
) -> bool:
    if not agent_name:
        return False

    target_agent = str(contract.get("target_agent", "")).strip()
    if target_agent:
        return target_agent == agent_name

    available_to_agents = contract.get("available_to_agents")
    if isinstance(available_to_agents, list) and available_to_agents:
        return agent_name in {str(item).strip() for item in available_to_agents if str(item).strip()}

    return True


def build_skill_execution_diagnostics(
    contracts: list[SkillInvocationContract] | tuple[SkillInvocationContract, ...],
    *,
    agent_name: str,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for raw_contract in contracts:
        if not isinstance(raw_contract, dict):
            continue
        contract = normalize_skill_invocation_contract(raw_contract)
        diagnostics.append(
            {
                "kind": "skill_execution_contract",
                "skill_id": str(contract.get("skill_id", "")).strip(),
                "mode": str(contract.get("mode", "")).strip() or "inline",
                "target_agent": str(contract.get("target_agent", "")).strip() or agent_name,
                "executed_by_agent": agent_name,
                "source": str(contract.get("source", "")).strip(),
                "reason": str(contract.get("reason", "")).strip(),
            }
        )
    return diagnostics


def build_skill_runtime_state(
    contracts: list[SkillInvocationContract] | tuple[SkillInvocationContract, ...],
    *,
    agent_name: str,
) -> dict[str, Any]:
    normalized_contracts = [normalize_skill_invocation_contract(contract) for contract in contracts if isinstance(contract, dict)]
    return {
        "active_skill_invocation_contracts": normalized_contracts,
        "skill_execution_diagnostics": build_skill_execution_diagnostics(
            normalized_contracts,
            agent_name=agent_name,
        ),
    }


def build_skill_prompt_context(
    state: dict[str, Any] | None,
    *,
    skill_registry: SkillRegistry | None,
    agent_name: str,
) -> str:
    if skill_registry is None or not agent_name:
        return ""

    contracts = get_active_skill_invocation_contracts(state, agent_name=agent_name)
    if not contracts:
        return ""

    return join_prompt_layers(
        *[
            render_skill_prompt_context(
                contract,
                skill_registry=skill_registry,
                agent_name=agent_name,
            )
            for contract in contracts
        ]
    )


def render_skill_prompt_context(
    contract: SkillInvocationContract,
    *,
    skill_registry: SkillRegistry,
    agent_name: str,
) -> str:
    normalized = normalize_skill_invocation_contract(contract)
    skill_id = str(normalized.get("skill_id", "")).strip()
    skill_name = str(normalized.get("name", "")).strip() or skill_id
    skill_description = str(normalized.get("description", "")).strip()
    mode = str(normalized.get("mode", "")).strip() or "inline"
    target_agent = str(normalized.get("target_agent", "")).strip() or agent_name
    source = str(normalized.get("source", "")).strip()
    reason = str(normalized.get("reason", "")).strip()

    lines = [
        "## Active Skill Runtime",
        f"- Skill: `{skill_id}`",
        f"- Name: {skill_name}",
        f"- Execution mode: `{mode}`",
        f"- Executing agent: `{target_agent}`",
    ]
    if skill_description:
        lines.append(f"- Description: {skill_description}")
    if source:
        lines.append(f"- Invocation source: `{source}`")
    if reason:
        lines.append(f"- Invocation reason: {reason}")
    lines.append("- Treat this as runtime-selected context. Do not reinterpret skill routing or delegation inside the prompt.")

    skill_body = load_skill_body_for_contract(skill_registry, normalized)
    if skill_body:
        lines.extend(
            (
                "",
                "### Skill Instructions",
                "The following SKILL.md body is the instruction body for the runtime-selected skill.",
                skill_body,
            )
        )

    return "\n".join(line for line in lines if line is not None).strip()


def load_skill_body_for_contract(
    skill_registry: SkillRegistry,
    contract: SkillInvocationContract,
) -> str:
    definition = resolve_skill_definition_for_contract(skill_registry, contract)
    if definition is None:
        return ""
    return skill_registry.load_skill_body(definition)


def resolve_skill_definition_for_contract(
    skill_registry: SkillRegistry,
    contract: SkillInvocationContract,
) -> SkillDefinition | None:
    skill_id = str(contract.get("skill_id", "")).strip()
    if not skill_id:
        return None

    context_paths = contract.get("context_paths")
    resolved_context_paths = context_paths if isinstance(context_paths, list) else []
    resolution = skill_registry.resolve_skill(skill_id, context_paths=resolved_context_paths)
    return resolution.effective_definition
