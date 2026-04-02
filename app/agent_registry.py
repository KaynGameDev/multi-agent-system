from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool


AgentNodeFactory = Callable[[Any], Any]


@dataclass(frozen=True)
class AgentRegistration:
    name: str
    description: str
    build_node: AgentNodeFactory
    tools: tuple[BaseTool, ...] = field(default_factory=tuple)
