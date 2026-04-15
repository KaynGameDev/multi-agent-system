from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
from typing_extensions import Literal

PendingActionDecisionKind = Literal["approve", "reject", "modify", "select", "unrelated", "unclear"]


class PendingActionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["pending_action_decision"] = "pending_action_decision"
    pending_action_id: str = Field(min_length=1)
    decision: PendingActionDecisionKind
    notes: str | None = None
    selected_item_id: str | None = None
    constraints: list[str] = Field(default_factory=list)

    @field_validator("pending_action_id")
    @classmethod
    def validate_pending_action_id(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("pending_action_id must not be empty.")
        return cleaned

    @field_validator("notes", "selected_item_id")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @field_validator("constraints", mode="before")
    @classmethod
    def validate_constraints(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("constraints must be a list.")
        return [str(item).strip() for item in value if str(item).strip()]

    @model_validator(mode="after")
    def validate_selection_requirements(self) -> PendingActionDecision:
        if self.decision == "select" and not self.selected_item_id:
            raise ValueError("selected_item_id is required when decision is `select`.")
        return self


def validate_pending_action_decision(value: Any) -> PendingActionDecision:
    return PendingActionDecision.model_validate(value)


def build_unclear_pending_action_decision(
    pending_action_id: str,
    *,
    notes: str | None = None,
) -> PendingActionDecision:
    normalized_id = str(pending_action_id or "").strip() or "unknown_pending_action"
    return PendingActionDecision(
        pending_action_id=normalized_id,
        decision="unclear",
        notes=notes,
        selected_item_id=None,
        constraints=[],
    )
