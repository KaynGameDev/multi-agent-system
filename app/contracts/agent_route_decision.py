from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentRouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_agent: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    fallback_used: bool
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("selected_agent", "reason")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("This field must not be empty.")
        return cleaned


def validate_agent_route_decision(value: Any) -> AgentRouteDecision:
    return AgentRouteDecision.model_validate(value)
