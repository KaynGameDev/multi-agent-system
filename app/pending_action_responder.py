from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import AIMessage

from app.contracts import ExecutionContractValidation, PendingAction, PendingActionDecision, build_assistant_response
from app.pending_actions import update_pending_action
from app.routing.pending_action_router import PendingActionTurnResult

PendingActionResponse = dict[str, Any]
ClarificationTextBuilder = Callable[[PendingAction, ExecutionContractValidation], str]
ValidResolutionHandler = Callable[[PendingAction, PendingActionDecision, ExecutionContractValidation], PendingActionResponse]


def run_pending_action_response(
    *,
    pending_action: PendingAction,
    pending_action_turn: PendingActionTurnResult,
    cannot_validate_text: str,
    cancel_text: str,
    build_clarification_text: ClarificationTextBuilder,
    on_valid_resolution: ValidResolutionHandler,
    cannot_validate_result_updates: Mapping[str, Any] | None = None,
    cancel_result_updates: Mapping[str, Any] | None = None,
    clarification_result_updates: Mapping[str, Any] | None = None,
) -> PendingActionResponse:
    contract = pending_action_turn.execution_contract
    validation = pending_action_turn.validation
    if contract is None or validation is None:
        return _build_response_result(
            content=cannot_validate_text,
            kind="await_confirmation",
            pending_action=pending_action,
            result_updates=cannot_validate_result_updates,
        )

    if validation.get("runtime_action") == "cancel":
        return _build_response_result(
            content=cancel_text,
            kind="text",
            pending_action=None,
            result_updates=cancel_result_updates,
        )

    normalized_scope = validation.get("normalized_scope") or {}
    updated_action = update_pending_action(
        pending_action,
        status=str(validation.get("next_status", "ask_clarification")).strip() or "ask_clarification",
        target_scope=normalized_scope or None,
        metadata_updates={"last_contract": dict(contract)},
    )

    if not validation.get("valid"):
        content = build_clarification_text(updated_action, validation)
        return _build_response_result(
            content=content,
            kind="await_confirmation",
            pending_action=updated_action,
            result_updates=clarification_result_updates,
        )

    return on_valid_resolution(updated_action, pending_action_turn.pending_action_decision, validation)


def _build_response_result(
    *,
    content: str,
    kind: str,
    pending_action: PendingAction | None,
    result_updates: Mapping[str, Any] | None = None,
) -> PendingActionResponse:
    result: PendingActionResponse = {
        "messages": [AIMessage(content=content)],
        "assistant_response": build_assistant_response(
            kind=kind,
            content=content,
            pending_action=pending_action,
        ),
        "pending_action": pending_action,
    }
    if result_updates:
        result.update(dict(result_updates))
    return result
