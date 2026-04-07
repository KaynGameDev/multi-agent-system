from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.skills import SkillDefinition, SkillRegistry, normalize_skill_id
from gateway.text_utils import normalize_text, tokenize_text


@dataclass(frozen=True)
class SelectedSkillRuntime:
    definition: SkillDefinition
    source: str
    reason: str


def normalize_requested_skill_ids(state: dict[str, Any]) -> tuple[str, ...]:
    raw_value = state.get("requested_skill_ids") or []
    if isinstance(raw_value, str):
        raw_items = [item.strip() for item in raw_value.split(",") if item.strip()]
    elif isinstance(raw_value, (list, tuple, set)):
        raw_items = [str(item).strip() for item in raw_value if str(item).strip()]
    else:
        raw_items = []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        skill_id = normalize_skill_id(item)
        if not skill_id or skill_id in seen:
            continue
        normalized.append(skill_id)
        seen.add(skill_id)
    return tuple(normalized)


def normalize_context_paths(state: dict[str, Any]) -> tuple[str, ...]:
    raw_value = state.get("context_paths") or []
    if isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, (list, tuple, set)):
        values = [str(item).strip() for item in raw_value if str(item).strip()]
    else:
        values = []
    return tuple(values)


def resolve_explicit_skill_definitions(
    skill_registry: SkillRegistry,
    *,
    requested_skill_ids: tuple[str, ...],
    context_paths: tuple[str, ...],
) -> tuple[tuple[SkillDefinition, ...], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    diagnostics: list[dict[str, Any]] = list(skill_registry.discovery_diagnostics)
    explicit_skill_definitions: list[SkillDefinition] = []

    for skill_id in requested_skill_ids:
        resolution = skill_registry.resolve_skill(skill_id, context_paths=context_paths)
        diagnostics.extend(resolution.diagnostics)
        if resolution.effective_definition is None:
            warnings.append(f"Requested skill `{skill_id}` could not be resolved from the shared registry.")
            continue
        explicit_skill_definitions.append(resolution.effective_definition)
        diagnostics.append(
            {
                "kind": "explicit_request",
                "skill_id": resolution.effective_definition.skill_id,
                "selected": True,
                "reason": "Skill was explicitly requested in state.",
            }
        )

    return tuple(explicit_skill_definitions), diagnostics, warnings


def select_skills_for_agent(
    skill_registry: SkillRegistry,
    *,
    agent_name: str,
    latest_user_text: str,
    general_assistant_name: str,
    context_paths: tuple[str, ...],
    explicit_skill_definitions: tuple[SkillDefinition, ...],
    warnings: list[str],
) -> tuple[tuple[SelectedSkillRuntime, ...], list[dict[str, Any]], list[str]]:
    diagnostics: list[dict[str, Any]] = []
    selected_skills: list[SelectedSkillRuntime] = []
    selected_ids: set[str] = set()

    for definition in explicit_skill_definitions:
        eligible = is_skill_eligible_for_agent(
            definition,
            target_agent=agent_name,
            general_assistant_name=general_assistant_name,
        )
        diagnostics.append(
            {
                "kind": "explicit_skill_selection",
                "skill_id": definition.skill_id,
                "selected": eligible,
                "reason": (
                    f"Explicit skill applies to `{agent_name}`."
                    if eligible
                    else f"Explicit skill does not apply to `{agent_name}`."
                ),
            }
        )
        if not eligible:
            warnings.append(
                f"Explicit skill `{definition.skill_id}` was skipped because it does not apply to `{agent_name}`."
            )
            continue
        if definition.skill_id in selected_ids:
            continue
        selected_skills.append(
            SelectedSkillRuntime(
                definition=definition,
                source="gateway.explicit_skill_request",
                reason=f"Explicit skill `{definition.skill_id}` was requested and applies to `{agent_name}`.",
            )
        )
        selected_ids.add(definition.skill_id)

    scored_auto_matches: list[tuple[int, SkillDefinition, str]] = []
    for skill_id in skill_registry.list_skill_ids():
        resolution = skill_registry.resolve_skill(skill_id, context_paths=context_paths)
        diagnostics.extend(resolution.diagnostics)
        definition = resolution.effective_definition
        if definition is None:
            continue
        if definition.skill_id in selected_ids:
            continue
        if definition.execution_mode != "inline":
            continue
        if not is_skill_eligible_for_agent(
            definition,
            target_agent=agent_name,
            general_assistant_name=general_assistant_name,
        ):
            diagnostics.append(
                {
                    "kind": "auto_skill_ineligible",
                    "skill_id": definition.skill_id,
                    "selected": False,
                    "reason": f"Effective skill does not apply to `{agent_name}`.",
                }
            )
            continue
        score, reason = score_skill_match(definition, latest_user_text)
        diagnostics.append(
            {
                "kind": "auto_skill_match",
                "skill_id": definition.skill_id,
                "selected": score > 0,
                "score": score,
                "reason": reason,
            }
        )
        if score <= 0:
            continue
        scored_auto_matches.append((score, definition, reason))

    for _, definition, reason in sorted(
        scored_auto_matches,
        key=lambda item: (-item[0], item[1].skill_id),
    )[:3]:
        selected_skills.append(
            SelectedSkillRuntime(
                definition=definition,
                source="gateway.auto_skill_match",
                reason=reason,
            )
        )
        selected_ids.add(definition.skill_id)

    return tuple(selected_skills), diagnostics, warnings


def is_skill_eligible_for_agent(
    definition: SkillDefinition,
    *,
    target_agent: str,
    general_assistant_name: str,
) -> bool:
    if definition.execution_mode == "inline":
        return target_agent in definition.available_to_agents
    if definition.delegate_agent:
        return target_agent == definition.delegate_agent
    return target_agent == general_assistant_name


def score_skill_match(definition: SkillDefinition, latest_user_text: str) -> tuple[int, str]:
    normalized_user_text = normalize_text(latest_user_text)
    if not normalized_user_text:
        return 0, "No user text available for deterministic skill matching."

    text_tokens = tokenize_text(normalized_user_text)
    skill_tokens = tokenize_text(
        normalize_text(" ".join([definition.skill_id, definition.name, definition.description]))
    )
    overlaps = sorted(text_tokens & skill_tokens)
    if len(overlaps) >= 2:
        return 20 + len(overlaps), f"Token overlap matched: {', '.join(overlaps[:5])}."

    normalized_skill_id = definition.skill_id.replace("-", " ")
    if normalized_skill_id and normalized_skill_id in normalized_user_text:
        return 15, f"Skill id phrase `{normalized_skill_id}` appeared in the request."

    return 0, "No deterministic metadata overlap matched."
