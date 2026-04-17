from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.memory.paths import resolve_long_term_memory_dir
from app.memory.types import AgentMemoryScope

DEFAULT_USER_MEMORY_KEY = "anonymous"
DEFAULT_PROJECT_MEMORY_KEY = "default-project"
DEFAULT_LOCAL_MEMORY_KEY = "local"


@dataclass(frozen=True)
class ResolvedAgentMemoryContext:
    agent_name: str
    scope: AgentMemoryScope
    scope_key: str
    root_dir: Path


def resolve_agent_memory_context(
    settings: Settings,
    *,
    agent_name: str,
    memory_scope: AgentMemoryScope,
    state: dict[str, Any] | None = None,
) -> ResolvedAgentMemoryContext:
    normalized_agent_name = str(agent_name or "").strip()
    if not normalized_agent_name:
        raise ValueError("agent_name must not be empty.")

    normalized_scope = str(memory_scope or "").strip().lower()
    if normalized_scope not in {"user", "project", "local"}:
        raise ValueError(f"Unsupported agent memory scope: {memory_scope}")

    scope_key = resolve_agent_memory_scope_key(normalized_scope, state=state)
    base_dir = resolve_long_term_memory_dir(settings) / "agents" / normalized_agent_name
    if normalized_scope == "user":
        root_dir = base_dir / "users" / scope_key
    elif normalized_scope == "project":
        root_dir = base_dir / "projects" / scope_key
    else:
        root_dir = base_dir / "local"

    return ResolvedAgentMemoryContext(
        agent_name=normalized_agent_name,
        scope=normalized_scope,  # type: ignore[arg-type]
        scope_key=scope_key,
        root_dir=root_dir.resolve(),
    )


def resolve_agent_memory_scope_key(
    memory_scope: AgentMemoryScope | str,
    *,
    state: dict[str, Any] | None = None,
) -> str:
    normalized_scope = str(memory_scope or "").strip().lower()
    state_dict = state if isinstance(state, dict) else {}

    if normalized_scope == "local":
        return DEFAULT_LOCAL_MEMORY_KEY

    if normalized_scope == "user":
        candidates = (
            state_dict.get("user_id"),
            state_dict.get("user_email"),
            state_dict.get("user_display_name"),
            state_dict.get("user_real_name"),
            state_dict.get("thread_id"),
        )
        for candidate in candidates:
            cleaned = normalize_agent_memory_scope_path(candidate)
            if cleaned:
                return cleaned
        return DEFAULT_USER_MEMORY_KEY

    if normalized_scope == "project":
        project_candidates = (
            (
                state_dict.get("target_market_slug"),
                state_dict.get("target_game_slug"),
                state_dict.get("target_feature_slug"),
            ),
            (
                "conversion",
                state_dict.get("conversion_session_id"),
            ),
            (
                "channel",
                state_dict.get("channel_id"),
            ),
            (
                "thread",
                state_dict.get("thread_id"),
            ),
        )
        for candidate_group in project_candidates:
            normalized_parts = [
                normalize_agent_memory_scope_segment(part)
                for part in candidate_group
                if normalize_agent_memory_scope_segment(part)
            ]
            if normalized_parts:
                return "/".join(normalized_parts)
        return DEFAULT_PROJECT_MEMORY_KEY

    raise ValueError(f"Unsupported agent memory scope: {memory_scope}")


def build_agent_memory_prompt(
    settings: Settings,
    *,
    agent_name: str,
    memory_scope: AgentMemoryScope | str,
    state: dict[str, Any] | None = None,
) -> str:
    normalized_scope = str(memory_scope or "").strip().lower()
    if normalized_scope not in {"user", "project", "local"}:
        return ""

    context = resolve_agent_memory_context(
        settings,
        agent_name=agent_name,
        memory_scope=normalized_scope,  # type: ignore[arg-type]
        state=state,
    )
    scope_label = {
        "user": "user-scoped",
        "project": "project-scoped",
        "local": "agent-local",
    }[context.scope]
    lines = [
        "# Memory Scope",
        f"- This agent has {scope_label} long-term memory enabled.",
        f"- Current memory scope: `{context.scope}`",
        f"- Current memory key: `{context.scope_key}`",
        f"- Allowed memory root: `{context.root_dir}`",
        "- Use only the memory tools to inspect or mutate memory.",
        "- Do not claim access to other agents' memory or any path outside the allowed memory root.",
        "- Treat the memory root as strictly path-scoped runtime state, not as a general filesystem capability.",
    ]
    return "\n".join(lines).strip()


def normalize_agent_memory_scope_path(value: Any) -> str:
    if value is None:
        return ""
    normalized = str(value).replace("\\", "/").strip().strip("/")
    if not normalized:
        return ""

    parts = [normalize_agent_memory_scope_segment(part) for part in normalized.split("/")]
    cleaned_parts = [part for part in parts if part]
    return "/".join(cleaned_parts)


def normalize_agent_memory_scope_segment(value: Any) -> str:
    cleaned = str(value or "").strip().lower()
    if not cleaned:
        return ""
    cleaned = re.sub(r"[^a-z0-9._-]+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    cleaned = cleaned.strip("-._")
    if cleaned in {"", ".", ".."}:
        return ""
    return cleaned
