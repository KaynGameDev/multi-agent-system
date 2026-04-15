from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, TypeVar
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage

from app.contracts import (
    ToolExecutionRecord,
    ToolInvocationEnvelope,
    ToolResultEnvelope,
    build_tool_execution_record,
    build_tool_invocation_envelope,
    build_tool_result_envelope,
    normalize_tool_invocation_envelope,
    normalize_tool_result_envelope,
    tool_invocation_to_tool_call,
)
from app.tool_registry import get_tool_metadata_by_runtime_name
from app.utils import safe_get_str

LANGGRAPH_TOOL_NODE_BACKEND = "langgraph_tool_node"
INTERNAL_WORKFLOW_BACKEND = "internal_workflow"

T = TypeVar("T")


def build_runtime_tool_invocation(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    status: str = "requested",
    payload: dict[str, Any] | None = None,
    error: str = "",
    diagnostics: list[dict[str, Any]] | None = None,
    source: str = "",
    reason: str = "",
    tool_call_id: str = "",
    execution_backend: str = LANGGRAPH_TOOL_NODE_BACKEND,
) -> ToolInvocationEnvelope:
    normalized_tool_name = str(tool_name or "").strip()
    metadata = resolve_tool_metadata_fields(normalized_tool_name)
    normalized_tool_call_id = str(tool_call_id or "").strip()
    if not normalized_tool_call_id and execution_backend == INTERNAL_WORKFLOW_BACKEND and normalized_tool_name:
        normalized_tool_call_id = f"internal_{normalized_tool_name}_{uuid4().hex}"
    return build_tool_invocation_envelope(
        normalized_tool_name,
        arguments or {},
        tool_id=metadata.get("tool_id", ""),
        display_name=metadata.get("display_name", ""),
        tool_family=metadata.get("tool_family", ""),
        execution_backend=execution_backend,
        status=status,  # type: ignore[arg-type]
        payload=payload,
        error=error,
        diagnostics=diagnostics,
        source=source,
        reason=reason,
        tool_call_id=normalized_tool_call_id,
    )


def build_runtime_tool_result(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    status: str = "ok",
    payload: dict[str, Any] | None = None,
    error: str = "",
    diagnostics: list[dict[str, Any]] | None = None,
    source: str = "",
    reason: str = "",
    tool_call_id: str = "",
    execution_backend: str = LANGGRAPH_TOOL_NODE_BACKEND,
) -> ToolResultEnvelope:
    normalized_tool_name = str(tool_name or "").strip()
    metadata = resolve_tool_metadata_fields(normalized_tool_name)
    return build_tool_result_envelope(
        normalized_tool_name,
        arguments or {},
        tool_id=metadata.get("tool_id", ""),
        display_name=metadata.get("display_name", ""),
        tool_family=metadata.get("tool_family", ""),
        execution_backend=execution_backend,
        status=status,  # type: ignore[arg-type]
        payload=payload,
        error=error,
        diagnostics=diagnostics,
        source=source,
        reason=reason,
        tool_call_id=str(tool_call_id or "").strip(),
    )


def normalize_runtime_tool_invocation(
    envelope: dict[str, Any] | None,
    *,
    tool_name: str = "",
    arguments: dict[str, Any] | None = None,
    source: str = "",
    reason: str = "",
    execution_backend: str = LANGGRAPH_TOOL_NODE_BACKEND,
) -> ToolInvocationEnvelope:
    normalized = normalize_tool_invocation_envelope(
        envelope,
        tool_name=tool_name,
        arguments=arguments,
        execution_backend=execution_backend,
        source=source,
        reason=reason,
    )
    return enrich_runtime_tool_envelope(normalized, execution_backend=execution_backend)


def normalize_runtime_tool_result(
    result: dict[str, Any] | None,
    *,
    tool_name: str = "",
    arguments: dict[str, Any] | None = None,
    source: str = "",
    reason: str = "",
    execution_backend: str = LANGGRAPH_TOOL_NODE_BACKEND,
) -> ToolResultEnvelope:
    normalized = normalize_tool_result_envelope(
        result,
        tool_name=tool_name,
        arguments=arguments,
        execution_backend=execution_backend,
        source=source,
        reason=reason,
    )
    return enrich_runtime_tool_result(normalized, execution_backend=execution_backend)


def enrich_runtime_tool_envelope(
    envelope: ToolInvocationEnvelope,
    *,
    execution_backend: str = LANGGRAPH_TOOL_NODE_BACKEND,
) -> ToolInvocationEnvelope:
    normalized = dict(envelope)
    tool_name = safe_get_str(normalized, "tool_name")
    metadata = resolve_tool_metadata_fields(tool_name)
    for key, value in metadata.items():
        if value and not safe_get_str(normalized, key):
            normalized[key] = value
    backend_value = safe_get_str(normalized, "execution_backend", execution_backend).lower()
    if backend_value:
        normalized["execution_backend"] = backend_value
    return normalized  # type: ignore[return-value]


def enrich_runtime_tool_result(
    result: ToolResultEnvelope,
    *,
    execution_backend: str = LANGGRAPH_TOOL_NODE_BACKEND,
) -> ToolResultEnvelope:
    normalized = dict(result)
    tool_name = safe_get_str(normalized, "tool_name")
    metadata = resolve_tool_metadata_fields(tool_name)
    for key, value in metadata.items():
        if value and not safe_get_str(normalized, key):
            normalized[key] = value
    backend_value = safe_get_str(normalized, "execution_backend", execution_backend).lower()
    if backend_value:
        normalized["execution_backend"] = backend_value
    return normalized  # type: ignore[return-value]


def extract_first_tool_invocation(
    response: Any,
    *,
    source: str = "",
    reason: str = "",
    execution_backend: str = LANGGRAPH_TOOL_NODE_BACKEND,
) -> ToolInvocationEnvelope | None:
    tool_calls = getattr(response, "tool_calls", []) or []
    if not tool_calls or not isinstance(tool_calls[0], dict):
        return None
    return normalize_runtime_tool_invocation(
        tool_calls[0],
        source=source,
        reason=reason,
        execution_backend=execution_backend,
    )


def find_matching_tool_call_request(
    messages: list[Any] | tuple[Any, ...],
    tool_call_id: str,
    *,
    tool_name: str = "",
) -> dict[str, Any] | None:
    normalized_tool_call_id = str(tool_call_id or "").strip()
    normalized_tool_name = str(tool_name or "").strip()

    fallback_match: dict[str, Any] | None = None
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        for tool_call in getattr(message, "tool_calls", []) or []:
            if not isinstance(tool_call, dict):
                continue
            current_name = safe_get_str(tool_call, "name")
            current_id = safe_get_str(tool_call, "id")
            if normalized_tool_call_id and current_id == normalized_tool_call_id:
                return tool_call
            if not normalized_tool_call_id and normalized_tool_name and current_name == normalized_tool_name and fallback_match is None:
                fallback_match = tool_call
    return fallback_match


def extract_tool_result_from_message(
    message: Any,
    *,
    messages: list[Any] | tuple[Any, ...] | None = None,
    tool_name: str = "",
    arguments: dict[str, Any] | None = None,
    source: str = "",
    reason: str = "",
    execution_backend: str = LANGGRAPH_TOOL_NODE_BACKEND,
) -> ToolResultEnvelope | None:
    if not isinstance(message, ToolMessage):
        return None

    content = getattr(message, "content", "")
    parsed_payload = parse_tool_message_content(content)
    if parsed_payload is None:
        return None

    tool_call_id = str(getattr(message, "tool_call_id", "") or "").strip()
    matched_request = find_matching_tool_call_request(
        list(messages or ()),
        tool_call_id,
        tool_name=tool_name,
    )
    resolved_tool_name = str((matched_request or {}).get("name", "") or tool_name).strip()
    resolved_arguments = arguments
    if not isinstance(resolved_arguments, dict):
        matched_arguments = (matched_request or {}).get("args")
        resolved_arguments = dict(matched_arguments) if isinstance(matched_arguments, dict) else {}

    result = normalize_runtime_tool_result(
        parsed_payload,
        tool_name=resolved_tool_name,
        arguments=resolved_arguments,
        source=source,
        reason=reason,
        execution_backend=execution_backend,
    )
    if tool_call_id and not safe_get_str(result, "tool_call_id"):
        result["tool_call_id"] = tool_call_id
    return result


def build_tool_execution_record_for_message(
    message: Any,
    *,
    messages: list[Any] | tuple[Any, ...] | None = None,
    tool_name: str = "",
    source: str = "",
    reason: str = "",
    execution_backend: str = LANGGRAPH_TOOL_NODE_BACKEND,
) -> ToolExecutionRecord | None:
    result = extract_tool_result_from_message(
        message,
        messages=messages,
        tool_name=tool_name,
        source=source,
        reason=reason,
        execution_backend=execution_backend,
    )
    if result is None:
        return None

    tool_call_id = safe_get_str(result, "tool_call_id")
    matched_request = find_matching_tool_call_request(list(messages or ()), tool_call_id, tool_name=tool_name)
    invocation = None
    if matched_request is not None:
        invocation = normalize_runtime_tool_invocation(
            matched_request,
            source=source,
            reason=reason,
            execution_backend=execution_backend,
        )
        if tool_call_id and not safe_get_str(invocation, "tool_call_id"):
            invocation["tool_call_id"] = tool_call_id
    return build_tool_execution_record(invocation=invocation, result=result)


def get_pending_action_tool_invocation(
    pending_action: dict[str, Any] | None,
    *,
    source: str = "",
    reason: str = "",
    execution_backend: str = LANGGRAPH_TOOL_NODE_BACKEND,
) -> ToolInvocationEnvelope | None:
    if not isinstance(pending_action, dict):
        return None

    metadata = pending_action.get("metadata") if isinstance(pending_action.get("metadata"), dict) else {}
    tool_invocation = metadata.get("tool_invocation")
    if isinstance(tool_invocation, dict):
        return normalize_runtime_tool_invocation(
            tool_invocation,
            source=source,
            reason=reason,
            execution_backend=execution_backend,
        )

    tool_name = safe_get_str(metadata, "tool_name")
    tool_args = metadata.get("tool_args")
    if not tool_name or not isinstance(tool_args, dict):
        return None
    return build_runtime_tool_invocation(
        tool_name,
        dict(tool_args),
        source=source,
        reason=reason,
        tool_call_id=safe_get_str(metadata, "tool_call_id"),
        execution_backend=execution_backend,
    )


def build_pending_action_retry_tool_call(
    pending_action: dict[str, Any] | None,
    *,
    source: str = "",
    reason: str = "",
    fallback_tool_call_id: str = "",
    execution_backend: str = LANGGRAPH_TOOL_NODE_BACKEND,
) -> tuple[AIMessage, ToolInvocationEnvelope] | None:
    tool_invocation = get_pending_action_tool_invocation(
        pending_action,
        source=source,
        reason=reason,
        execution_backend=execution_backend,
    )
    if tool_invocation is None:
        return None

    tool_call = tool_invocation_to_tool_call(tool_invocation)
    if not tool_call.get("name"):
        return None
    tool_call["id"] = safe_get_str(tool_call, "id") or str(fallback_tool_call_id).strip()
    if not str(tool_call["id"]).strip():
        return None
    return AIMessage(content="", tool_calls=[tool_call]), tool_invocation


def run_internal_tool_operation(
    tool_name: str,
    *,
    arguments: dict[str, Any] | None = None,
    source: str = "",
    reason: str = "",
    payload_builder: Callable[[T], dict[str, Any]] | None = None,
    operation: Callable[[], T],
) -> tuple[ToolInvocationEnvelope, ToolResultEnvelope, T | None, Exception | None]:
    invocation = build_runtime_tool_invocation(
        tool_name,
        arguments or {},
        source=source,
        reason=reason,
        execution_backend=INTERNAL_WORKFLOW_BACKEND,
    )
    try:
        raw_result = operation()
    except Exception as exc:  # pragma: no cover - callers decide whether to recover or re-raise
        result = build_runtime_tool_result(
            tool_name,
            arguments or {},
            status="error",
            payload={},
            error=str(exc),
            source=source,
            reason=reason,
            tool_call_id=safe_get_str(invocation, "tool_call_id"),
            execution_backend=INTERNAL_WORKFLOW_BACKEND,
        )
        return invocation, result, None, exc

    payload = payload_builder(raw_result) if payload_builder is not None else coerce_internal_tool_payload(raw_result)
    result = build_runtime_tool_result(
        tool_name,
        arguments or {},
        status="ok",
        payload=payload,
        source=source,
        reason=reason,
        tool_call_id=safe_get_str(invocation, "tool_call_id"),
        execution_backend=INTERNAL_WORKFLOW_BACKEND,
    )
    return invocation, result, raw_result, None


def parse_tool_message_content(content: Any) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return dict(content)
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {"content": stripped}
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"items": [serialize_internal_value(item) for item in parsed]}
    return {"value": serialize_internal_value(parsed)}


def resolve_tool_metadata_fields(tool_name: str) -> dict[str, str]:
    normalized_tool_name = str(tool_name or "").strip()
    if not normalized_tool_name:
        return {}
    try:
        metadata = get_tool_metadata_by_runtime_name(normalized_tool_name)
    except KeyError:
        return {}
    return {
        "tool_id": metadata.tool_id,
        "display_name": metadata.display_name,
        "tool_family": metadata.tool_family,
    }


def coerce_internal_tool_payload(value: Any) -> dict[str, Any]:
    serialized = serialize_internal_value(value)
    if isinstance(serialized, dict):
        return serialized
    if isinstance(serialized, list):
        return {"items": serialized}
    return {"value": serialized}


def serialize_internal_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: serialize_internal_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): serialize_internal_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_internal_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
