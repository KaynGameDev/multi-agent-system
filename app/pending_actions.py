from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.contracts import (
    ExecutionContract,
    ExecutionContractValidation,
    ExecutionDecision,
    PendingAction,
    PendingActionSelectionOption,
    PendingActionStatus,
    PendingActionTargetScope,
    RuntimeAction,
)
from app.pending_action_parser import (
    DEFAULT_UNAVAILABLE_REASON,
    INTERPRETATION_SOURCE_DETERMINISTIC_SELECTION,
    INTERPRETATION_SOURCE_LLM_PARSER,
    PendingActionReplyInterpreter,
    build_unclear_pending_action_parse,
)

ACTIVE_PENDING_ACTION_STATUSES = {"awaiting_confirmation", "request_revision", "ask_clarification"}
LEGACY_PENDING_ACTION_STATUS_ALIASES = {
    "modified": "request_revision",
    "cancelled": "rejected",
}
ALLOWED_EXECUTION_DECISIONS = {"approve", "reject", "modify", "select", "unclear"}


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


def resolve_pending_action_reply(
    action: PendingAction,
    user_text: str,
    *,
    interpreter: PendingActionReplyInterpreter | None = None,
) -> dict[str, Any]:
    contract = interpret_pending_action_reply(action, user_text, interpreter=interpreter)
    validation = validate_execution_contract(action, contract)
    return {
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


def extract_selection_index(normalized_text: str) -> int | None:
    if normalized_text.isdigit():
        return max(int(normalized_text) - 1, 0)

    patterns = (
        r"^第\s*(\d+)\s*(?:个|篇|份|条|项)?$",
        r"^(?:option|doc|document|task)\s*(\d+)$",
        r"^(\d+)[\).]$",
    )
    for pattern in patterns:
        match = re.match(pattern, normalized_text)
        if match:
            return max(int(match.group(1)) - 1, 0)
    return None


def match_pending_action_selection_option(
    options: list[PendingActionSelectionOption],
    normalized_text: str,
) -> dict[str, Any] | None:
    selected_index = extract_selection_index(normalized_text)
    if selected_index is not None and 0 <= selected_index < len(options):
        option = options[selected_index]
        return {"option": option, "index": selected_index}

    for index, option in enumerate(options):
        labels = [option.get("label", ""), option.get("value", ""), *option.get("aliases", [])]
        normalized_labels = {
            normalize_pending_action_text(label)
            for label in labels
            if normalize_pending_action_text(label)
        }
        if normalized_text in normalized_labels:
            return {"option": option, "index": index}
    return None


def interpret_pending_action_reply(
    action: PendingAction,
    user_text: str,
    *,
    interpreter: PendingActionReplyInterpreter | None = None,
) -> ExecutionContract:
    prepared_input = prepare_pending_action_reply_input(action, user_text)
    exact_selection_match = prepared_input.get("exact_selection_match")
    if isinstance(exact_selection_match, dict):
        return build_selection_execution_contract(action, user_text, exact_selection_match)

    parsed_reply = build_unclear_pending_action_parse(DEFAULT_UNAVAILABLE_REASON)
    if interpreter is not None:
        try:
            parsed_reply = interpreter.parse_pending_action_reply(action, prepared_input)
        except Exception:
            parsed_reply = build_unclear_pending_action_parse(DEFAULT_UNAVAILABLE_REASON)
    return build_interpreted_execution_contract(action, user_text, parsed_reply=parsed_reply)


def prepare_pending_action_reply_input(action: PendingAction, user_text: str) -> dict[str, Any]:
    normalized_user_text = normalize_pending_action_text(user_text)
    selection_options = get_pending_action_selection_options(action)
    exact_selection_match = None
    if selection_options and normalized_user_text:
        exact_selection_match = match_pending_action_selection_option(selection_options, normalized_user_text)

    return {
        "user_reply": str(user_text or "").strip(),
        "normalized_user_reply": normalized_user_text,
        "summary": str(action.get("summary", "")).strip(),
        "action_type": str(action.get("type", "")).strip(),
        "target_scope": dict(action.get("target_scope") or {}),
        "selection_options": [dict(option) for option in selection_options],
        "exact_selection_match": exact_selection_match,
    }


def build_selection_execution_contract(
    action: PendingAction,
    user_text: str,
    selection_match: dict[str, Any],
) -> ExecutionContract:
    selected_option = selection_match["option"]
    selected_index = int(selection_match["index"])
    return ExecutionContract(
        pending_action_id=str(action.get("id", "")).strip(),
        session_id=str(action.get("session_id", "")).strip(),
        action_type=str(action.get("type", "")).strip(),
        requested_by_agent=str(action.get("requested_by_agent", "")).strip(),
        decision="select",
        summary="The user selected one of the pending options.",
        reply_text=str(user_text or "").strip(),
        target_scope={},
        selected_option=selected_option,
        selected_index=selected_index,
        requested_outputs=[],
        constraints={},
        should_execute=True,
        interpretation_source=INTERPRETATION_SOURCE_DETERMINISTIC_SELECTION,
        confidence=1.0,
    )


def build_interpreted_execution_contract(
    action: PendingAction,
    user_text: str,
    *,
    parsed_reply: dict[str, Any],
) -> ExecutionContract:
    decision = normalize_execution_decision(parsed_reply.get("decision"))
    interpretation_source = normalize_interpretation_source(parsed_reply.get("interpretation_source"))
    normalized_confidence = normalize_contract_confidence(parsed_reply.get("confidence", 0.0))
    normalized_scope = dict(parsed_reply.get("target_scope") or {}) if isinstance(parsed_reply.get("target_scope"), dict) else {}
    requested_outputs = normalize_contract_requested_outputs(parsed_reply.get("requested_outputs"))
    selected_index = normalize_contract_selected_index(parsed_reply.get("selected_index"))
    should_execute = parsed_reply.get("should_execute") is True

    if decision == "reject":
        should_execute = False
    if decision == "unclear":
        normalized_scope = {}
        requested_outputs = []
        selected_index = None
        should_execute = False

    summary = str(parsed_reply.get("reason", "")).strip() or default_execution_summary(decision)
    return ExecutionContract(
        pending_action_id=str(action.get("id", "")).strip(),
        session_id=str(action.get("session_id", "")).strip(),
        action_type=str(action.get("type", "")).strip(),
        requested_by_agent=str(action.get("requested_by_agent", "")).strip(),
        decision=decision,
        summary=summary,
        reply_text=str(user_text or "").strip(),
        target_scope=normalized_scope,
        selected_index=selected_index,
        requested_outputs=requested_outputs,
        constraints={},
        should_execute=should_execute,
        interpretation_source=interpretation_source,
        confidence=normalized_confidence,
    )


def default_execution_summary(decision: str) -> str:
    if decision == "approve":
        return "The user approved the pending action."
    if decision == "reject":
        return "The user rejected the pending action."
    if decision == "modify":
        return "The user requested a modified version of the pending action."
    return "The reply did not clearly approve, reject, or modify the pending action."


def normalize_execution_decision(value: Any) -> str:
    decision = str(value or "").strip().lower()
    if decision in ALLOWED_EXECUTION_DECISIONS:
        return decision
    return "unclear"


def normalize_interpretation_source(value: Any) -> str:
    source = str(value or "").strip()
    if source:
        return source
    return INTERPRETATION_SOURCE_LLM_PARSER


def normalize_contract_confidence(value: Any) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(resolved, 1.0))


def normalize_contract_requested_outputs(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip().lower() for item in value if str(item).strip()]


def normalize_contract_selected_index(value: Any) -> int | None:
    if isinstance(value, int) and value >= 0:
        return value
    return None


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
        return None
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


def normalize_pending_action_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().casefold())
    return normalized.strip("`'\"“”‘’.,!?，。！？")


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


def scope_item_matches_text(item: str, normalized_text: str) -> bool:
    normalized_item = normalize_pending_action_text(item)
    if not normalized_item:
        return False
    if normalized_item in normalized_text:
        return True

    basename = normalized_item.rsplit("/", 1)[-1]
    stem = basename.rsplit(".", 1)[0]
    return any(candidate and candidate in normalized_text for candidate in {basename, stem})


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
