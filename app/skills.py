from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.agent_registry import AgentRegistration
from app.paths import PROJECT_ROOT, resolve_project_path
from app.prompt_loader import join_prompt_layers

logger = logging.getLogger(__name__)

PROJECT_SHARED_SCOPE = "project_shared"
AGENT_LOCAL_SCOPE = "agent_local"
PATH_SCOPED_SCOPE = "path_scoped"
DEFAULT_PROJECT_SKILLS_DIR = ".jade/skills"
SKILL_FRONTMATTER_DELIMITER = "---"
SCOPE_PRECEDENCE = {
    PROJECT_SHARED_SCOPE: 1,
    AGENT_LOCAL_SCOPE: 2,
    PATH_SCOPED_SCOPE: 3,
}


@dataclass(frozen=True)
class SkillDefinition:
    skill_id: str
    name: str
    description: str
    scope: str
    source_path: Path
    execution_mode: str = "inline"
    delegate_agent: str = ""
    available_to_agents: tuple[str, ...] = field(default_factory=tuple)
    path_patterns: tuple[str, ...] = field(default_factory=tuple)
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillResolution:
    skill_id: str
    effective_definition: SkillDefinition | None
    diagnostics: tuple[dict[str, Any], ...]


class SkillRegistry:
    def __init__(
        self,
        agent_registrations: tuple[AgentRegistration, ...],
        *,
        project_skills_dir: str | Path = DEFAULT_PROJECT_SKILLS_DIR,
        project_root: Path | None = None,
    ) -> None:
        self.agent_registrations = tuple(agent_registrations)
        self.project_root = Path(project_root).resolve() if project_root is not None else PROJECT_ROOT
        self.project_skills_dir = self._resolve_path(project_skills_dir, DEFAULT_PROJECT_SKILLS_DIR)
        self.general_assistant_name = self._resolve_general_assistant_name()
        self._definitions_by_id: dict[str, tuple[SkillDefinition, ...]] = {}
        self._discovery_diagnostics: list[dict[str, Any]] = []
        self._body_cache: dict[Path, str] = {}
        self._build_catalog()

    @property
    def discovery_diagnostics(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._discovery_diagnostics)

    def resolve_skill(self, skill_id: str, *, context_paths: list[str] | tuple[str, ...] | None = None) -> SkillResolution:
        normalized_id = normalize_skill_id(skill_id)
        definitions = self._definitions_by_id.get(normalized_id, ())
        if not definitions:
            return SkillResolution(
                skill_id=normalized_id,
                effective_definition=None,
                diagnostics=(
                    {
                        "kind": "missing_skill",
                        "skill_id": normalized_id,
                        "reason": f"Skill `{normalized_id}` was not found in the shared registry.",
                    },
                ),
            )

        resolved_paths = tuple(path for path in (context_paths or ()) if str(path).strip())
        diagnostics: list[dict[str, Any]] = []
        eligible_by_scope: dict[str, list[SkillDefinition]] = {}

        for definition in definitions:
            eligible, reason = self._is_definition_eligible(definition, resolved_paths)
            diagnostics.append(
                {
                    "kind": "candidate",
                    "skill_id": definition.skill_id,
                    "scope": definition.scope,
                    "source_path": str(definition.source_path),
                    "available_to_agents": list(definition.available_to_agents),
                    "path_patterns": list(definition.path_patterns),
                    "eligible": eligible,
                    "reason": reason,
                }
            )
            if not eligible:
                continue
            eligible_by_scope.setdefault(definition.scope, []).append(definition)

        if not eligible_by_scope:
            diagnostics.append(
                {
                    "kind": "unresolved",
                    "skill_id": normalized_id,
                    "reason": "No eligible skill definition matched the current request context.",
                }
            )
            return SkillResolution(
                skill_id=normalized_id,
                effective_definition=None,
                diagnostics=tuple(diagnostics),
            )

        winning_scope = max(eligible_by_scope, key=lambda item: SCOPE_PRECEDENCE.get(item, 0))
        winning_definitions = eligible_by_scope[winning_scope]
        if len(winning_definitions) > 1:
            diagnostics.append(
                {
                    "kind": "conflict",
                    "skill_id": normalized_id,
                    "scope": winning_scope,
                    "reason": "Multiple definitions were eligible at the same precedence level.",
                    "candidates": [str(item.source_path) for item in winning_definitions],
                }
            )
            return SkillResolution(
                skill_id=normalized_id,
                effective_definition=None,
                diagnostics=tuple(diagnostics),
            )

        effective_definition = winning_definitions[0]
        for definition in definitions:
            if definition == effective_definition:
                continue
            diagnostics.append(
                {
                    "kind": "shadowed",
                    "skill_id": normalized_id,
                    "scope": definition.scope,
                    "source_path": str(definition.source_path),
                    "shadowed_by": str(effective_definition.source_path),
                }
            )

        diagnostics.append(
            {
                "kind": "resolved",
                "skill_id": normalized_id,
                "effective_scope": effective_definition.scope,
                "effective_source_path": str(effective_definition.source_path),
                "effective_execution_mode": effective_definition.execution_mode,
            }
        )
        return SkillResolution(
            skill_id=normalized_id,
            effective_definition=effective_definition,
            diagnostics=tuple(diagnostics),
        )

    def list_effective_skills(
        self,
        *,
        context_paths: list[str] | tuple[str, ...] | None = None,
        agent_name: str | None = None,
    ) -> tuple[SkillDefinition, ...]:
        definitions: list[SkillDefinition] = []
        for skill_id in sorted(self._definitions_by_id):
            resolution = self.resolve_skill(skill_id, context_paths=context_paths)
            effective = resolution.effective_definition
            if effective is None:
                continue
            if agent_name and agent_name not in effective.available_to_agents:
                continue
            definitions.append(effective)
        return tuple(definitions)

    def list_skill_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions_by_id))

    def build_prompt_layers(
        self,
        skill_ids: list[str] | tuple[str, ...],
        *,
        agent_name: str,
        context_paths: list[str] | tuple[str, ...] | None = None,
    ) -> str:
        layers: list[str] = []
        seen_skill_ids: set[str] = set()
        for raw_skill_id in skill_ids:
            normalized_id = normalize_skill_id(raw_skill_id)
            if not normalized_id or normalized_id in seen_skill_ids:
                continue
            seen_skill_ids.add(normalized_id)
            resolution = self.resolve_skill(normalized_id, context_paths=context_paths)
            definition = resolution.effective_definition
            if definition is None:
                continue
            if agent_name not in definition.available_to_agents:
                continue
            body = self.load_skill_body(definition)
            if not body:
                continue
            layers.append(body)
        return join_prompt_layers(*layers)

    def load_skill_body(self, definition: SkillDefinition) -> str:
        cached = self._body_cache.get(definition.source_path)
        if cached is not None:
            return cached

        raw_text = definition.source_path.read_text(encoding="utf-8")
        _, body = split_skill_frontmatter(raw_text)
        normalized_body = body.strip()
        self._body_cache[definition.source_path] = normalized_body
        return normalized_body

    def _build_catalog(self) -> None:
        definitions: list[SkillDefinition] = []
        discovery_diagnostics: list[dict[str, Any]] = []

        for registration in self.agent_registrations:
            if not registration.skill_namespace:
                continue
            agent_skills_dir = self._resolve_path(Path("agents") / registration.skill_namespace / "Skills")
            for definition, diagnostics in self._discover_scope(
                agent_skills_dir,
                source_scope=AGENT_LOCAL_SCOPE,
                owning_agent=registration.name,
                all_agents=tuple(item.name for item in self.agent_registrations),
            ):
                definitions.append(definition)
                discovery_diagnostics.extend(diagnostics)

        for definition, diagnostics in self._discover_scope(
            self.project_skills_dir,
            source_scope=PROJECT_SHARED_SCOPE,
            owning_agent="",
            all_agents=tuple(item.name for item in self.agent_registrations),
        ):
            definitions.append(definition)
            discovery_diagnostics.extend(diagnostics)

        grouped: dict[str, list[SkillDefinition]] = {}
        for definition in definitions:
            grouped.setdefault(definition.skill_id, []).append(definition)

        normalized_groups: dict[str, tuple[SkillDefinition, ...]] = {}
        for skill_id, items in grouped.items():
            normalized_groups[skill_id] = tuple(
                sorted(
                    items,
                    key=lambda definition: (
                        -SCOPE_PRECEDENCE.get(definition.scope, 0),
                        str(definition.source_path),
                    ),
                )
            )

            same_scope_groups: dict[str, list[SkillDefinition]] = {}
            for definition in items:
                same_scope_groups.setdefault(definition.scope, []).append(definition)
            for scope_name, scoped_items in same_scope_groups.items():
                if len(scoped_items) < 2:
                    continue
                discovery_diagnostics.append(
                    {
                        "kind": "same_scope_conflict",
                        "skill_id": skill_id,
                        "scope": scope_name,
                        "reason": "Multiple definitions share the same scope and skill_id.",
                        "candidates": [str(item.source_path) for item in scoped_items],
                    }
                )

        self._definitions_by_id = normalized_groups
        self._discovery_diagnostics = discovery_diagnostics

    def _discover_scope(
        self,
        skills_dir: Path,
        *,
        source_scope: str,
        owning_agent: str,
        all_agents: tuple[str, ...],
    ) -> list[tuple[SkillDefinition, list[dict[str, Any]]]]:
        if not skills_dir.is_dir():
            return []

        discovered: list[tuple[SkillDefinition, list[dict[str, Any]]]] = []
        for skill_path in sorted(skills_dir.glob("*/SKILL.md")):
            definition, diagnostics = self._normalize_definition(
                skill_path,
                source_scope=source_scope,
                owning_agent=owning_agent,
                all_agents=all_agents,
            )
            if definition is None:
                continue
            discovered.append((definition, diagnostics))
        return discovered

    def _normalize_definition(
        self,
        skill_path: Path,
        *,
        source_scope: str,
        owning_agent: str,
        all_agents: tuple[str, ...],
    ) -> tuple[SkillDefinition | None, list[dict[str, Any]]]:
        diagnostics: list[dict[str, Any]] = []
        raw_text = skill_path.read_text(encoding="utf-8")
        metadata, body = split_skill_frontmatter(raw_text)
        metadata = normalize_metadata_keys(metadata)

        fallback_name = extract_skill_name(body) or skill_path.parent.name
        fallback_description = extract_skill_description(body) or f"Skill loaded from {skill_path.parent.name}."
        skill_id = normalize_skill_id(metadata.get("skill_id") or skill_path.parent.name or fallback_name)
        name = str(metadata.get("name") or fallback_name).strip() or skill_id
        description = str(metadata.get("description") or fallback_description).strip() or fallback_description
        execution_mode = normalize_execution_mode(metadata.get("execution_mode") or metadata.get("context"))
        delegate_agent = str(metadata.get("delegate_agent") or metadata.get("agent") or "").strip()
        delegate_agent = self._normalize_agent_reference(delegate_agent)
        path_patterns = normalize_string_tuple(metadata.get("path_patterns") or metadata.get("paths"))

        if owning_agent:
            available_to_agents = (owning_agent,)
        else:
            raw_available = metadata.get("available_to_agents") or metadata.get("compatible_agents") or metadata.get("agents")
            available_to_agents = tuple(
                self._normalize_agent_reference(item) or item
                for item in normalize_string_tuple(raw_available)
            )
            if not available_to_agents:
                available_to_agents = tuple(all_agents)
                diagnostics.append(
                    {
                        "kind": "default_available_to_agents",
                        "skill_id": skill_id,
                        "source_path": str(skill_path),
                        "reason": "Project-shared skill did not declare `available_to_agents`; defaulted to all active agents.",
                        "available_to_agents": list(available_to_agents),
                    }
                )

        scope = PATH_SCOPED_SCOPE if path_patterns else source_scope
        definition = SkillDefinition(
            skill_id=skill_id,
            name=name,
            description=description,
            scope=scope,
            source_path=skill_path,
            execution_mode=execution_mode,
            delegate_agent=delegate_agent,
            available_to_agents=available_to_agents,
            path_patterns=path_patterns,
            raw_metadata=dict(metadata),
        )
        diagnostics.append(
            {
                "kind": "discovered",
                "skill_id": definition.skill_id,
                "scope": definition.scope,
                "source_path": str(definition.source_path),
                "execution_mode": definition.execution_mode,
                "available_to_agents": list(definition.available_to_agents),
            }
        )
        return definition, diagnostics

    def _is_definition_eligible(
        self,
        definition: SkillDefinition,
        context_paths: tuple[str, ...],
    ) -> tuple[bool, str]:
        if definition.scope != PATH_SCOPED_SCOPE:
            return True, "Always eligible for this scope."
        if not context_paths:
            return False, "Path-scoped skill requires context_paths but none were provided."
        for context_path in context_paths:
            if any(fnmatch.fnmatch(context_path, pattern) for pattern in definition.path_patterns):
                return True, f"Matched path pattern for `{context_path}`."
        return False, "No context path matched the path-scoped patterns."

    def _resolve_path(self, configured_value: str | Path, default_value: str | Path = "") -> Path:
        if self.project_root == PROJECT_ROOT:
            return resolve_project_path(configured_value, default_value)
        candidate = configured_value or default_value
        path = Path(candidate).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.project_root / path).resolve()

    def _resolve_general_assistant_name(self) -> str:
        for registration in self.agent_registrations:
            if registration.is_general_assistant:
                return registration.name
        for registration in self.agent_registrations:
            if registration.name == "general_chat_agent":
                return registration.name
        return ""

    def _normalize_agent_reference(self, raw_value: str) -> str:
        if raw_value == "GeneralAssistant":
            return self.general_assistant_name or raw_value
        return raw_value


def split_skill_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    if not raw_text.startswith(f"{SKILL_FRONTMATTER_DELIMITER}\n"):
        return {}, raw_text

    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != SKILL_FRONTMATTER_DELIMITER:
        return {}, raw_text

    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == SKILL_FRONTMATTER_DELIMITER:
            end_index = index
            break

    if end_index is None:
        return {}, raw_text

    frontmatter_text = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :]).lstrip("\n")
    return parse_frontmatter(frontmatter_text), body


def parse_frontmatter(frontmatter_text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    current_key = ""

    for raw_line in frontmatter_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if current_key and stripped.startswith("- "):
            current_value = values.setdefault(current_key, [])
            if isinstance(current_value, list):
                current_value.append(parse_frontmatter_value(stripped[2:].strip()))
            continue

        match = re.match(r"^([A-Za-z0-9_-]+):(?:\s*(.*))?$", stripped)
        if not match:
            current_key = ""
            continue

        current_key = match.group(1)
        raw_value = (match.group(2) or "").strip()
        if not raw_value:
            values[current_key] = []
            continue
        values[current_key] = parse_frontmatter_value(raw_value)
        if not isinstance(values[current_key], list):
            current_key = ""

    return values


def parse_frontmatter_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""
    lowered = value.casefold()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_frontmatter_value(item.strip()) for item in inner.split(",")]
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def normalize_metadata_keys(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        normalized[key.replace("-", "_")] = value
    return normalized


def normalize_execution_mode(raw_value: Any) -> str:
    value = str(raw_value or "").strip().lower()
    if value in {"fork", "forked"}:
        return "forked"
    return "inline"


def normalize_string_tuple(raw_value: Any) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        parts = [item.strip() for item in raw_value.split(",") if item.strip()]
        return tuple(parts)
    if isinstance(raw_value, (list, tuple, set)):
        return tuple(str(item).strip() for item in raw_value if str(item).strip())
    return ()


def normalize_skill_id(raw_value: Any) -> str:
    text = str(raw_value or "").strip().casefold()
    if not text:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return normalized


def extract_skill_name(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        name = re.sub(r"^#+\s*", "", stripped)
        name = re.sub(r"^Skill:\s*", "", name, flags=re.IGNORECASE)
        if name:
            return name.strip()
    return ""


def extract_skill_description(body: str) -> str:
    lines = body.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped in {"## Purpose", "## 目的", "## Description", "## 描述"}:
            paragraph = collect_following_paragraph(lines[index + 1 :])
            if paragraph:
                return paragraph

    paragraph = collect_following_paragraph(lines)
    return paragraph


def collect_following_paragraph(lines: list[str]) -> str:
    collected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == SKILL_FRONTMATTER_DELIMITER:
            if collected:
                break
            continue
        if stripped.startswith("#"):
            if collected:
                break
            continue
        if stripped.startswith("- ") or re.match(r"^\d+\.\s+", stripped):
            if collected:
                break
            continue
        collected.append(stripped)
    return " ".join(collected).strip()
