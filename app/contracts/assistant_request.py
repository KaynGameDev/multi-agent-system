from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Literal

AssistantRequestDomain = Literal[
    "general",
    "knowledge",
    "project_task",
    "knowledge_base_builder",
    "document_conversion",
]


class AssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["assistant_request"] = "assistant_request"
    user_goal: str = Field(min_length=1)
    likely_domain: AssistantRequestDomain
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None = None

    @field_validator("user_goal")
    @classmethod
    def validate_user_goal(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("user_goal must not be empty.")
        return cleaned

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None


def validate_assistant_request(value: Any) -> AssistantRequest:
    return AssistantRequest.model_validate(value)


def build_fallback_assistant_request(
    user_goal: str,
    *,
    likely_domain: AssistantRequestDomain = "general",
    confidence: float = 0.0,
    notes: str | None = None,
) -> AssistantRequest:
    normalized_goal = str(user_goal or "").strip() or "Clarify the user's request."
    return AssistantRequest(
        user_goal=normalized_goal,
        likely_domain=likely_domain,
        confidence=confidence,
        notes=notes,
    )
