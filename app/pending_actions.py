from __future__ import annotations

import re
from datetime import datetime, timezone
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

ACTIVE_PENDING_ACTION_STATUSES = {"awaiting_confirmation", "request_revision", "ask_clarification"}
LEGACY_PENDING_ACTION_STATUS_ALIASES = {
    "modified": "request_revision",
    "cancelled": "rejected",
}
APPROVAL_KEYWORDS = {
    "ok",
    "okay",
    "yes",
    "yeah",
    "yep",
    "sure",
    "continue",
    "proceed",
    "run",
    "execute",
    "apply",
    "ship",
    "approve",
    "approved",
    "confirm",
    "confirmed",
    "ahead",
    "go",
    "continue",
    "继续",
    "继续吧",
    "可以",
    "好的",
    "好",
    "行",
    "执行",
    "批准",
    "确认",
    "同意",
}
REJECTION_KEYWORDS = {
    "cancel",
    "stop",
    "abort",
    "reject",
    "decline",
    "skip",
    "no",
    "never",
    "取消",
    "停止",
    "拒绝",
    "不用",
    "算了",
    "不要",
    "别",
}
MODIFIER_KEYWORDS = {
    "only",
    "just",
    "except",
    "without",
    "but",
    "limit",
    "limited",
    "just",
    "only",
    "仅",
    "只",
    "只要",
    "但是",
    "但",
    "除了",
    "不要",
}
OUTPUT_KEYWORDS = {
    "diff": "diff",
    "patch": "diff",
    "changes": "diff",
    "change": "diff",
    "preview": "preview",
    "plan": "plan",
    "summary": "summary",
    "details": "details",
    "detail": "details",
    "difference": "diff",
    "show": "preview",
    "比较": "diff",
    "差异": "diff",
    "预览": "preview",
    "计划": "plan",
    "总结": "summary",
    "详情": "details",
}
DEFER_KEYWORDS = {"first", "before", "prior", "先", "先看", "先给"}


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


def resolve_pending_action_reply(action: PendingAction, user_text: str) -> dict[str, Any]:
    contract = interpret_pending_action_reply(action, user_text)
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


def interpret_pending_action_reply(action: PendingAction, user_text: str) -> ExecutionContract:
    normalized_text = normalize_pending_action_text(user_text)
    tokens = tokenize_pending_action_text(normalized_text)
    requested_outputs = detect_requested_outputs(normalized_text, tokens)
    target_scope = detect_requested_scope(action, normalized_text)
    has_modifier_language = contains_any(normalized_text, MODIFIER_KEYWORDS)
    has_scope_modifier = bool(target_scope) and has_modifier_language
    approval_score = score_keyword_hits(tokens, normalized_text, APPROVAL_KEYWORDS)
    rejection_score = score_keyword_hits(tokens, normalized_text, REJECTION_KEYWORDS)
    defer_requested = contains_any(normalized_text, DEFER_KEYWORDS)
    selection_options = get_pending_action_selection_options(action)

    decision: ExecutionDecision = "unclear"
    summary = "The reply did not clearly approve, reject, or modify the pending action."
    should_execute = False
    constraints: dict[str, Any] = {}

    if selection_options:
        selection_match = match_pending_action_selection_option(selection_options, normalized_text)
        if selection_match is None and len(selection_options) == 1 and approval_score > 0 and not defer_requested:
            selection_match = {"option": selection_options[0], "index": 0}
        if selection_match is not None:
            selected_option = selection_match["option"]
            selected_index = int(selection_match["index"])
            decision = "select"
            summary = "The user selected one of the pending options."
            should_execute = True
            return ExecutionContract(
                pending_action_id=str(action.get("id", "")).strip(),
                session_id=str(action.get("session_id", "")).strip(),
                action_type=str(action.get("type", "")).strip(),
                requested_by_agent=str(action.get("requested_by_agent", "")).strip(),
                decision=decision,
                summary=summary,
                reply_text=user_text.strip(),
                target_scope=target_scope,
                selected_option=selected_option,
                selected_index=selected_index,
                requested_outputs=requested_outputs,
                constraints=constraints,
                should_execute=should_execute,
            )
        if rejection_score > approval_score and rejection_score > 0 and not requested_outputs:
            decision = "reject"
            summary = "The user rejected the pending action."
        else:
            summary = "The reply did not clearly select one of the pending options."
    elif rejection_score > approval_score and rejection_score > 0 and not requested_outputs:
        decision = "reject"
        summary = "The user rejected the pending action."
    elif requested_outputs or has_scope_modifier or (target_scope and approval_score > 0) or (approval_score > 0 and has_modifier_language):
        decision = "modify"
        should_execute = approval_score > 0 and not requested_outputs and not defer_requested
        if requested_outputs:
            constraints["preview_before_execute"] = True
            if not should_execute:
                summary = "The user requested a preview or diff before execution."
            else:
                summary = "The user approved the action with an additional preview request."
        elif target_scope:
            summary = "The user approved the action with a narrower target scope."
        elif has_modifier_language:
            summary = "The user approved the action with additional constraints."
        else:
            summary = "The user requested a modified version of the pending action."
    elif approval_score > 0 and rejection_score == 0:
        decision = "approve"
        should_execute = True
        summary = "The user approved the pending action."

    return ExecutionContract(
        pending_action_id=str(action.get("id", "")).strip(),
        session_id=str(action.get("session_id", "")).strip(),
        action_type=str(action.get("type", "")).strip(),
        requested_by_agent=str(action.get("requested_by_agent", "")).strip(),
        decision=decision,
        summary=summary,
        reply_text=user_text.strip(),
        target_scope=target_scope,
        requested_outputs=requested_outputs,
        constraints=constraints,
        should_execute=should_execute,
    )


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
    if decision == "reject":
        return ExecutionContractValidation(
            valid=True,
            runtime_action="cancel",
            next_status="rejected",
            reason="The user rejected the pending action.",
            normalized_scope={},
        )

    if decision == "select":
        selected_option = contract.get("selected_option")
        if not isinstance(selected_option, dict):
            return ExecutionContractValidation(
                valid=False,
                runtime_action="ask_clarification",
                next_status="ask_clarification",
                reason="The reply did not identify a selectable option.",
                normalized_scope={},
            )
        selected_index = contract.get("selected_index")
        normalized_index = int(selected_index) if isinstance(selected_index, int) else 0
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

    requested_outputs = list(contract.get("requested_outputs") or [])
    should_execute = contract.get("should_execute") is True
    if decision == "modify":
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
        valid=True,
        runtime_action="execute",
        next_status="approved",
        reason="The user approved the pending action.",
        normalized_scope=normalized_scope,
    )


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
    if target_scope:
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
    return True


def normalize_pending_action_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip().casefold())
    return normalized.strip("`'\"“”‘’.,!?，。！？")


def tokenize_pending_action_text(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9\u4e00-\u9fff]+", text)
        if token
    }


def detect_requested_outputs(normalized_text: str, tokens: set[str]) -> list[str]:
    outputs: list[str] = []
    for keyword, output_name in OUTPUT_KEYWORDS.items():
        if keyword in normalized_text or keyword in tokens:
            outputs.append(output_name)
    return list(dict.fromkeys(outputs))


def detect_requested_scope(action: PendingAction, normalized_text: str) -> PendingActionTargetScope:
    target_scope = action.get("target_scope") or {}
    resolved_scope: PendingActionTargetScope = {}

    for field_name in ("files", "modules"):
        raw_items = target_scope.get(field_name)
        if not isinstance(raw_items, list):
            continue
        matched_items = [item for item in raw_items if scope_item_matches_text(str(item), normalized_text)]
        if matched_items:
            resolved_scope[field_name] = matched_items

    skill_name = str(target_scope.get("skill_name", "")).strip()
    if skill_name and scope_item_matches_text(skill_name, normalized_text):
        resolved_scope["skill_name"] = skill_name
    return resolved_scope


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


def score_keyword_hits(tokens: set[str], normalized_text: str, keywords: set[str]) -> int:
    score = 0
    for keyword in keywords:
        normalized_keyword = normalize_pending_action_text(keyword)
        if not normalized_keyword:
            continue
        if normalized_keyword in tokens:
            score += 2
            continue
        if keyword_matches_text(normalized_keyword, normalized_text):
            score += 1
    return score


def contains_any(normalized_text: str, keywords: set[str]) -> bool:
    return any(
        keyword_matches_text(normalize_pending_action_text(keyword), normalized_text)
        for keyword in keywords
        if keyword
    )


def keyword_matches_text(keyword: str, normalized_text: str) -> bool:
    if not keyword:
        return False
    if re.search(r"[\u4e00-\u9fff]", keyword):
        return keyword in normalized_text
    if " " in keyword:
        return keyword in normalized_text
    return bool(re.search(rf"\b{re.escape(keyword)}\b", normalized_text))
