from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from app.contracts import (
    ExecutionContract,
    ExecutionContractValidation,
    ExecutionDecision,
    PendingAction,
    PendingActionDecision,
    PendingActionSelectionOption,
    PendingActionStatus,
    PendingActionTargetScope,
    RuntimeAction,
    validate_pending_action_decision,
)

ACTIVE_PENDING_ACTION_STATUSES = {"awaiting_confirmation", "request_revision", "ask_clarification"}
LEGACY_PENDING_ACTION_STATUS_ALIASES = {
    "modified": "request_revision",
    "cancelled": "rejected",
}
ALLOWED_EXECUTION_DECISIONS = {"approve", "reject", "modify", "select", "unclear"}
ALLOWED_REQUESTED_OUTPUTS = {"diff", "preview", "plan", "summary", "details"}


def build_pending_action(
    *,
    session_id: str,
    action_type: str,
    requested_by_agent: str,
    summary: str,
    target_scope: PendingActionTargetScope | None = None,
    risk_level: str = "medium",
    requires_explicit_approval: bool = True,
    metadata: dict[str, Any] | None = None,
) -> PendingAction:
    return PendingAction(
        id=f"pending_{uuid4().hex}",
        session_id=session_id.strip(),
        type=action_type.strip(),
        requested_by_agent=requested_by_agent.strip(),
        summary=summary.strip(),
        target_scope=dict(target_scope or {}),
        risk_level=risk_level.strip() or "medium",
        requires_explicit_approval=bool(requires_explicit_approval),
        created_at=datetime.now(timezone.utc).isoformat(),
        status="awaiting_confirmation",
        metadata=dict(metadata or {}),
    )


def get_pending_action(state: dict[str, Any]) -> PendingAction | None:
    action = state.get("pending_action")
    if isinstance(action, dict) and str(action.get("status", "")).strip():
        normalized_action = dict(action)
        normalized_status = normalize_pending_action_status(str(normalized_action.get("status", "")).strip())
        if normalized_status:
            normalized_action["status"] = normalized_status
        return normalized_action
    return None


def is_pending_action_active(action: PendingAction | None) -> bool:
    if not action:
        return False
    return normalize_pending_action_status(str(action.get("status", "")).strip()) in ACTIVE_PENDING_ACTION_STATUSES


def get_execution_contract(state: dict[str, Any]) -> ExecutionContract | None:
    contract = state.get("execution_contract")
    if isinstance(contract, dict) and str(contract.get("decision", "")).strip():
        return contract
    return None


def get_pending_action_decision(state: dict[str, Any]) -> PendingActionDecision | None:
    raw_decision = state.get("pending_action_decision")
    if raw_decision is None:
        return None
    try:
        return validate_pending_action_decision(raw_decision)
    except ValidationError:
        return None


def resolve_pending_action_decision(
    action: PendingAction,
    decision: PendingActionDecision | dict[str, Any],
    *,
    user_text: str = "",
) -> dict[str, Any]:
    validated_decision = validate_pending_action_decision(decision)
    if validated_decision.decision == "unrelated":
        return {
            "decision": validated_decision,
            "contract": None,
            "validation": None,
        }

    contract = build_execution_contract_from_pending_action_decision(
        action,
        validated_decision,
        user_text=user_text,
    )
    validation = validate_execution_contract(action, contract)
    return {
        "decision": validated_decision,
        "contract": contract,
        "validation": validation,
    }


def normalize_pending_action_status(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if not normalized:
        return ""
    return LEGACY_PENDING_ACTION_STATUS_ALIASES.get(normalized, normalized)


def get_pending_action_metadata(action: PendingAction | None) -> dict[str, Any]:
    if not action:
        return {}
    metadata = action.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def get_pending_action_selection_options(action: PendingAction | None) -> list[PendingActionSelectionOption]:
    metadata = get_pending_action_metadata(action)
    raw_options = metadata.get("selection_options")
    if not isinstance(raw_options, list):
        return []

    options: list[PendingActionSelectionOption] = []
    for option in raw_options:
        if isinstance(option, dict):
            options.append(option)
    return options


def get_pending_action_selection_phase(action: PendingAction | None) -> str:
    metadata = get_pending_action_metadata(action)
    return str(metadata.get("selection_phase", "")).strip().lower()


def get_pending_action_approval_payload(action: PendingAction | None) -> dict[str, Any]:
    metadata = get_pending_action_metadata(action)
    payload = metadata.get("approval_payload")
    if isinstance(payload, dict):
        return payload
    return {}


def get_pending_action_approval_payload_hash(action: PendingAction | None) -> str:
    metadata = get_pending_action_metadata(action)
    return str(metadata.get("approval_payload_hash", "")).strip()


def build_execution_contract_from_pending_action_decision(
    action: PendingAction,
    decision: PendingActionDecision,
    *,
    user_text: str = "",
) -> ExecutionContract:
    normalized_scope, requested_outputs, unparsed_constraints = parse_pending_action_constraints(decision.constraints)
    resolved_selection = None
    if decision.selected_item_id:
        resolved_selection = resolve_selection_option_by_item_id(action, decision.selected_item_id)

    selected_option = resolved_selection[0] if resolved_selection is not None else None
    selected_index = resolved_selection[1] if resolved_selection is not None else None
    should_execute = decision.decision in {"approve", "select"}
    if decision.decision == "modify":
        should_execute = bool(normalized_scope) and not requested_outputs and not unparsed_constraints
    if decision.decision in {"reject", "unclear"}:
        should_execute = False

    return ExecutionContract(
        pending_action_id=str(decision.pending_action_id).strip(),
        session_id=str(action.get("session_id", "")).strip(),
        action_type=str(action.get("type", "")).strip(),
        requested_by_agent=str(action.get("requested_by_agent", "")).strip(),
        decision=normalize_execution_decision(decision.decision),
        summary=str(decision.notes or "").strip() or default_execution_summary(decision.decision),
        reply_text=str(user_text or "").strip(),
        target_scope=normalized_scope,
        selected_option=selected_option,
        selected_index=selected_index,
        requested_outputs=requested_outputs,
        constraints={
            "raw_constraints": list(decision.constraints),
            "selected_item_id": str(decision.selected_item_id or "").strip(),
            "unparsed_constraints": list(unparsed_constraints),
        },
        should_execute=should_execute,
        interpretation_source="pending_action_decision",
        confidence=1.0 if decision.decision != "unclear" else 0.0,
    )


def default_execution_summary(decision: str) -> str:
    if decision == "approve":
        return "The user approved the pending action."
    if decision == "reject":
        return "The user rejected the pending action."
    if decision == "modify":
        return "The user requested a modified version of the pending action."
    if decision == "select":
        return "The user selected one of the pending options."
    return "The reply did not clearly approve, reject, or modify the pending action."


def normalize_execution_decision(value: Any) -> str:
    decision = str(value or "").strip().lower()
    if decision in ALLOWED_EXECUTION_DECISIONS:
        return decision
    return "unclear"


def validate_execution_contract(
    action: PendingAction,
    contract: ExecutionContract,
) -> ExecutionContractValidation:
    action_id = str(action.get("id", "")).strip()
    if not action_id or contract.get("pending_action_id") != action_id:
        return ExecutionContractValidation(
            valid=False,
            runtime_action="ask_clarification",
            next_status="awaiting_confirmation",
            reason="The execution contract did not match the active pending action.",
            normalized_scope={},
        )

    decision = str(contract.get("decision", "")).strip().lower()
    requested_outputs = list(contract.get("requested_outputs") or [])
    should_execute = contract.get("should_execute") is True
    selected_index = contract.get("selected_index")
    raw_constraints = contract.get("constraints") if isinstance(contract.get("constraints"), dict) else {}
    unparsed_constraints = list(raw_constraints.get("unparsed_constraints") or [])
    if unparsed_constraints:
        return ExecutionContractValidation(
            valid=False,
            runtime_action="ask_clarification",
            next_status="ask_clarification",
            reason=(
                "The pending-action decision included unsupported constraints: "
                f"{', '.join(str(item) for item in unparsed_constraints if str(item).strip())}."
            ),
            normalized_scope={},
        )

    if decision == "reject":
        return ExecutionContractValidation(
            valid=True,
            runtime_action="cancel",
            next_status="rejected",
            reason="The user rejected the pending action.",
            normalized_scope={},
        )

    if decision == "select":
        if requested_outputs or contract.get("target_scope") or not should_execute:
            return ExecutionContractValidation(
                valid=False,
                runtime_action="ask_clarification",
                next_status="ask_clarification",
                reason="The selection reply mixed option picking with unsupported execution instructions.",
                normalized_scope={},
            )
        resolved_selection = resolve_contract_selection(action, contract)
        if resolved_selection is None:
            return ExecutionContractValidation(
                valid=False,
                runtime_action="ask_clarification",
                next_status="ask_clarification",
                reason="The reply did not identify a selectable option.",
                normalized_scope={},
            )
        selected_option, normalized_index = resolved_selection
        return ExecutionContractValidation(
            valid=True,
            runtime_action="select",
            next_status="awaiting_confirmation",
            reason="The user selected a pending option.",
            normalized_scope={},
            selected_option=selected_option,
            selected_index=normalized_index,
        )

    if decision == "unclear":
        return ExecutionContractValidation(
            valid=False,
            runtime_action="ask_clarification",
            next_status="ask_clarification",
            reason="The reply was too ambiguous to execute deterministically.",
            normalized_scope={},
        )

    normalized_scope, scope_errors = normalize_requested_scope(
        action_scope=action.get("target_scope"),
        requested_scope=contract.get("target_scope"),
    )
    if scope_errors:
        return ExecutionContractValidation(
            valid=False,
            runtime_action="ask_clarification",
            next_status="ask_clarification",
            reason=" ".join(scope_errors),
            normalized_scope=normalized_scope,
        )

    if decision == "approve":
        if requested_outputs:
            return ExecutionContractValidation(
                valid=True,
                runtime_action="request_revision",
                next_status="request_revision",
                reason="The user requested additional preview material before execution.",
                normalized_scope=normalized_scope,
            )
        if normalized_scope:
            return ExecutionContractValidation(
                valid=False,
                runtime_action="ask_clarification",
                next_status="ask_clarification",
                reason="The approval reply included a narrowed scope that requires clarification.",
                normalized_scope=normalized_scope,
            )
        if selected_index is not None:
            return ExecutionContractValidation(
                valid=False,
                runtime_action="ask_clarification",
                next_status="ask_clarification",
                reason="The approval reply also identified a selection, so it could not be executed safely.",
                normalized_scope={},
            )
        if not should_execute:
            return ExecutionContractValidation(
                valid=False,
                runtime_action="ask_clarification",
                next_status="ask_clarification",
                reason="The parsed approval did not authorize execution.",
                normalized_scope={},
            )
        return ExecutionContractValidation(
            valid=True,
            runtime_action="execute",
            next_status="approved",
            reason="The user approved the pending action.",
            normalized_scope={},
        )

    if decision == "modify":
        if selected_index is not None:
            return ExecutionContractValidation(
                valid=False,
                runtime_action="ask_clarification",
                next_status="ask_clarification",
                reason="The requested modification also identified a selection, so it could not be handled deterministically.",
                normalized_scope={},
            )
        if not normalized_scope and not requested_outputs:
            return ExecutionContractValidation(
                valid=False,
                runtime_action="ask_clarification",
                next_status="ask_clarification",
                reason="The requested modification did not map to a supported deterministic scope.",
                normalized_scope=normalized_scope,
            )
        if requested_outputs and not should_execute:
            return ExecutionContractValidation(
                valid=True,
                runtime_action="request_revision",
                next_status="request_revision",
                reason="The user requested additional preview material before execution.",
                normalized_scope=normalized_scope,
            )
        if should_execute:
            return ExecutionContractValidation(
                valid=True,
                runtime_action="execute",
                next_status="approved",
                reason="The user approved a narrowed execution contract.",
                normalized_scope=normalized_scope,
            )
        return ExecutionContractValidation(
            valid=True,
            runtime_action="request_revision",
            next_status="request_revision",
            reason="The user requested a modified action that still needs agent handling.",
            normalized_scope=normalized_scope,
        )

    return ExecutionContractValidation(
        valid=False,
        runtime_action="ask_clarification",
        next_status="ask_clarification",
        reason="The reply did not map to a supported execution decision.",
        normalized_scope={},
    )


def resolve_contract_selection(
    action: PendingAction,
    contract: ExecutionContract,
) -> tuple[PendingActionSelectionOption, int] | None:
    options = get_pending_action_selection_options(action)
    if not options:
        return None

    selected_index = contract.get("selected_index")
    if isinstance(selected_index, int):
        if 0 <= selected_index < len(options):
            return options[selected_index], selected_index
        return None

    selected_option = contract.get("selected_option")
    if not isinstance(selected_option, dict):
        selected_item_id = ""
        constraints = contract.get("constraints")
        if isinstance(constraints, dict):
            selected_item_id = str(constraints.get("selected_item_id", "")).strip()
        if not selected_item_id:
            return None
        return resolve_selection_option_by_item_id(action, selected_item_id)
    for index, option in enumerate(options):
        if option == selected_option:
            return option, index
    return None


def update_pending_action(
    action: PendingAction,
    *,
    status: PendingActionStatus,
    target_scope: PendingActionTargetScope | None = None,
    metadata_updates: dict[str, Any] | None = None,
) -> PendingAction:
    updated: PendingAction = dict(action)
    normalized_status = normalize_pending_action_status(str(status).strip())
    updated["status"] = normalized_status or str(status).strip()
    if target_scope is not None:
        updated["target_scope"] = dict(target_scope)
    if metadata_updates:
        updated_metadata = dict(updated.get("metadata") or {})
        updated_metadata.update(metadata_updates)
        updated["metadata"] = updated_metadata
    return updated


def expire_pending_action(action: PendingAction) -> PendingAction:
    return update_pending_action(action, status="expired")


def action_allows_execution(
    action: PendingAction | None,
    contract: ExecutionContract | None,
    *,
    action_type: str,
    file_path: str = "",
    approval_payload: dict[str, Any] | None = None,
    approval_payload_hash: str = "",
) -> bool:
    if not action or not contract:
        return False

    validation = validate_execution_contract(action, contract)
    if not validation.get("valid") or validation.get("runtime_action") != "execute":
        return False
    if str(action.get("type", "")).strip() != action_type.strip():
        return False

    if file_path:
        normalized_scope = validation.get("normalized_scope") or {}
        allowed_files = list(normalized_scope.get("files") or action.get("target_scope", {}).get("files", []) or [])
        if allowed_files and file_path not in allowed_files:
            return False
    return approval_payload_matches_action(
        action,
        approval_payload=approval_payload,
        approval_payload_hash=approval_payload_hash,
    )


def normalize_requested_scope(
    *,
    action_scope: PendingActionTargetScope | None,
    requested_scope: PendingActionTargetScope | None,
) -> tuple[PendingActionTargetScope, list[str]]:
    if not requested_scope:
        return {}, []

    available_scope = action_scope or {}
    normalized_scope: PendingActionTargetScope = {}
    errors: list[str] = []

    for field_name in ("files", "modules"):
        requested_items = list(requested_scope.get(field_name) or [])
        if not requested_items:
            continue
        available_items = list(available_scope.get(field_name) or [])
        if not available_items:
            errors.append(f"The pending action does not support narrowing by {field_name}.")
            continue
        missing = [item for item in requested_items if item not in available_items]
        if missing:
            errors.append(f"Requested {field_name} were outside the pending action scope: {', '.join(missing)}.")
            continue
        normalized_scope[field_name] = requested_items

    requested_skill_name = str(requested_scope.get("skill_name", "")).strip()
    if requested_skill_name:
        available_skill_name = str(available_scope.get("skill_name", "")).strip()
        if not available_skill_name:
            errors.append("The pending action does not support narrowing by skill.")
        elif requested_skill_name != available_skill_name:
            errors.append(f"Requested skill `{requested_skill_name}` did not match the pending action skill.")
        else:
            normalized_scope["skill_name"] = requested_skill_name

    return normalized_scope, errors


def parse_pending_action_constraints(
    constraints: list[str] | tuple[str, ...] | None,
) -> tuple[PendingActionTargetScope, list[str], list[str]]:
    normalized_scope: PendingActionTargetScope = {}
    requested_outputs: list[str] = []
    unparsed_constraints: list[str] = []

    for raw_constraint in constraints or ():
        cleaned_constraint = str(raw_constraint or "").strip()
        if not cleaned_constraint:
            continue
        key, separator, raw_value = cleaned_constraint.partition(":")
        constraint_key = key.strip().lower()
        constraint_value = raw_value.strip()
        if not separator or not constraint_key or not constraint_value:
            unparsed_constraints.append(cleaned_constraint)
            continue

        if constraint_key in {"file", "files"}:
            normalized_scope.setdefault("files", [])
            normalized_scope["files"].append(constraint_value)
            continue
        if constraint_key in {"module", "modules"}:
            normalized_scope.setdefault("modules", [])
            normalized_scope["modules"].append(constraint_value)
            continue
        if constraint_key in {"skill", "skill_name"}:
            normalized_scope["skill_name"] = constraint_value
            continue
        if constraint_key == "output":
            normalized_output = constraint_value.lower()
            if normalized_output in ALLOWED_REQUESTED_OUTPUTS:
                requested_outputs.append(normalized_output)
                continue
        unparsed_constraints.append(cleaned_constraint)

    if "files" in normalized_scope:
        normalized_scope["files"] = list(dict.fromkeys(normalized_scope["files"]))
    if "modules" in normalized_scope:
        normalized_scope["modules"] = list(dict.fromkeys(normalized_scope["modules"]))
    requested_outputs = list(dict.fromkeys(requested_outputs))
    return normalized_scope, requested_outputs, unparsed_constraints


def resolve_selection_option_by_item_id(
    action: PendingAction,
    selected_item_id: str,
) -> tuple[PendingActionSelectionOption, int] | None:
    normalized_target = normalize_selection_item_id(selected_item_id)
    if not normalized_target:
        return None

    options = get_pending_action_selection_options(action)
    for index, option in enumerate(options):
        if normalized_target in build_selection_option_identifiers(option, fallback_index=index + 1):
            return option, index
    return None


def build_selection_option_identifiers(
    option: PendingActionSelectionOption,
    *,
    fallback_index: int,
) -> set[str]:
    payload = option.get("payload")
    payload_id = payload.get("id") if isinstance(payload, dict) else None
    raw_candidates = (
        option.get("id"),
        option.get("value"),
        option.get("label"),
        payload_id,
        str(fallback_index),
    )
    return {
        normalize_selection_item_id(candidate)
        for candidate in raw_candidates
        if normalize_selection_item_id(candidate)
    }


def normalize_selection_item_id(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def approval_payload_matches_action(
    action: PendingAction | None,
    *,
    approval_payload: dict[str, Any] | None = None,
    approval_payload_hash: str = "",
) -> bool:
    expected_hash = get_pending_action_approval_payload_hash(action)
    if not expected_hash:
        return False

    resolved_hash = str(approval_payload_hash or "").strip()
    if not resolved_hash and isinstance(approval_payload, dict):
        resolved_hash = compute_approval_payload_hash(approval_payload)
    if not resolved_hash or resolved_hash != expected_hash:
        return False

    expected_payload = get_pending_action_approval_payload(action)
    if expected_payload and isinstance(approval_payload, dict):
        return normalize_approval_payload(approval_payload) == normalize_approval_payload(expected_payload)
    return True


def normalize_approval_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): normalize_approval_payload_value(value)
        for key, value in sorted(payload.items(), key=lambda item: str(item[0]))
    }


def normalize_approval_payload_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): normalize_approval_payload_value(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize_approval_payload_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def compute_approval_payload_hash(payload: dict[str, Any] | None) -> str:
    normalized_payload = normalize_approval_payload(payload)
    if not normalized_payload:
        return ""
    encoded = json.dumps(
        normalized_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_write_knowledge_approval_payload(
    *,
    relative_path: str,
    content: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    return {
        "action_type": "write_knowledge_markdown",
        "tool_name": "write_knowledge_markdown_document",
        "arguments": {
            "relative_path": str(relative_path or "").strip(),
            "content": str(content or ""),
            "overwrite": bool(overwrite),
        },
    }


def compute_directory_digest(root_path: str | Path) -> str:
    resolved_root = Path(root_path)
    if not resolved_root.exists() or not resolved_root.is_dir():
        return ""

    hasher = hashlib.sha256()
    for candidate in sorted(path for path in resolved_root.rglob("*") if path.is_file()):
        relative_path = candidate.relative_to(resolved_root).as_posix()
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(candidate.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def build_conversion_publish_approval_payload(
    *,
    relative_package_path: str,
    staged_package_path: str,
) -> dict[str, Any]:
    return {
        "action_type": "publish_conversion_package",
        "relative_package_path": str(relative_package_path or "").strip(),
        "staged_package_digest": compute_directory_digest(staged_package_path),
    }
