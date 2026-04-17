from __future__ import annotations

from typing import Annotated, Any

from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState

from app.config import Settings
from app.memory.agent_scope import resolve_agent_memory_context
from app.memory.long_term import FileLongTermMemoryStore
from app.memory.types import AgentMemoryScope


def build_agent_memory_tools(
    settings: Settings,
    *,
    agent_name: str,
    memory_scope: AgentMemoryScope,
) -> tuple[BaseTool, ...]:
    normalized_agent_name = str(agent_name or "").strip()
    normalized_memory_scope = str(memory_scope or "").strip().lower()
    if not normalized_agent_name or normalized_memory_scope not in {"user", "project", "local"}:
        return ()

    def build_context(state: dict[str, Any] | None) -> dict[str, object]:
        resolved = resolve_agent_memory_context(
            settings,
            agent_name=normalized_agent_name,
            memory_scope=normalized_memory_scope,  # type: ignore[arg-type]
            state=state,
        )
        return {
            "agent_name": normalized_agent_name,
            "memory_scope": normalized_memory_scope,
            "memory_scope_key": resolved.scope_key,
            "memory_root": str(resolved.root_dir),
            "path_scoped": True,
        }

    def build_store(state: dict[str, Any] | None) -> tuple[FileLongTermMemoryStore, dict[str, object]]:
        context = build_context(state)
        return FileLongTermMemoryStore(str(context["memory_root"])), context

    @tool
    def list_agent_memories(
        state: Annotated[dict | None, InjectedState] = None,
    ) -> dict[str, object]:
        """List short memory index entries for this agent's scoped memory root."""
        try:
            store, context = build_store(state)
            entries = [entry.model_dump(mode="json") for entry in store.list()]
            return {
                "ok": True,
                **context,
                "entries": entries,
                "count": len(entries),
            }
        except Exception as exc:
            return {
                "ok": False,
                **build_context(state),
                "error": str(exc),
                "entries": [],
                "count": 0,
            }

    @tool
    def read_agent_memory(
        memory_id: str,
        state: Annotated[dict | None, InjectedState] = None,
    ) -> dict[str, object]:
        """Load a single memory topic from this agent's scoped memory root by memory id."""
        try:
            store, context = build_store(state)
            memory = store.get(memory_id)
            if memory is None:
                return {
                    "ok": False,
                    **context,
                    "memory_id": str(memory_id or "").strip(),
                    "error": "Memory not found.",
                }
            return {
                "ok": True,
                **context,
                "memory": memory.model_dump(mode="json"),
            }
        except Exception as exc:
            return {
                "ok": False,
                **build_context(state),
                "memory_id": str(memory_id or "").strip(),
                "error": str(exc),
            }

    @tool
    def write_agent_memory(
        memory_id: str,
        name: str,
        description: str,
        memory_type: str,
        content: str,
        state: Annotated[dict | None, InjectedState] = None,
    ) -> dict[str, object]:
        """Create or update a scoped long-term memory topic for this agent."""
        try:
            store, context = build_store(state)
            memory = store.upsert(
                {
                    "memory_id": memory_id,
                    "name": name,
                    "description": description,
                    "memory_type": memory_type,
                    "content_markdown": content,
                }
            )
            return {
                "ok": True,
                **context,
                "memory": memory.model_dump(mode="json"),
            }
        except Exception as exc:
            return {
                "ok": False,
                **build_context(state),
                "memory_id": str(memory_id or "").strip(),
                "error": str(exc),
            }

    @tool
    def delete_agent_memory(
        memory_id: str,
        state: Annotated[dict | None, InjectedState] = None,
    ) -> dict[str, object]:
        """Delete a scoped long-term memory topic for this agent."""
        try:
            store, context = build_store(state)
            deleted = store.delete(memory_id)
            return {
                "ok": deleted,
                **context,
                "memory_id": str(memory_id or "").strip(),
                "deleted": deleted,
                "error": "" if deleted else "Memory not found.",
            }
        except Exception as exc:
            return {
                "ok": False,
                **build_context(state),
                "memory_id": str(memory_id or "").strip(),
                "deleted": False,
                "error": str(exc),
            }

    return (
        list_agent_memories,
        read_agent_memory,
        write_agent_memory,
        delete_agent_memory,
    )
