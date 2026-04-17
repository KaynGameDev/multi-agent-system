from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool
from app.memory.types import AgentMemoryScope


AgentNodeFactory = Callable[[Any], Any]
AgentMatcher = Callable[[Any, str], Any]


@dataclass(frozen=True)
class AgentRegistration:
    name: str
    description: str
    build_node: AgentNodeFactory
    tools: tuple[BaseTool, ...] = field(default_factory=tuple)
    tool_ids: tuple[str, ...] = field(default_factory=tuple)
    selection_order: int = 100
    is_general_assistant: bool = False
    skill_namespace: str = ""
    matcher: AgentMatcher | None = None
    memory_scope: AgentMemoryScope | None = None
