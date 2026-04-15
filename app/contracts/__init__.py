from __future__ import annotations

from typing import Any
from uuid import uuid4

from typing_extensions import Literal, TypedDict

from app.utils import safe_get_str

from .agent_route_decision import AgentRouteDecision, validate_agent_route_decision
from .assistant_request import (
    AssistantRequest,
    AssistantRequestDomain,
    build_fallback_assistant_request,
    validate_assistant_request,
)
from .pending_action_decision import (
    PendingActionDecision,
    PendingActionDecisionKind,
    build_unclear_pending_action_decision,
    validate_pending_action_decision,
)

PendingActionStatus = Literal[
    "awaiting_confirmation",
    "approved",
    "rejected",
    "request_revision",
    "ask_clarification",
    "expired",
]
ExecutionDecision = Literal["approve", "reject", "modify", "select", "unclear"]
RuntimeAction = Literal["execute", "cancel", "request_revision", "ask_clarification", "select"]
AssistantResponseKind = Literal[
    "text",
    "await_confirmation",
    "execute",
    "invoke_skill",
    "invoke_tool",
    "tool_result",
    "route",
]
SkillInvocationMode = Literal["inline", "fork"]
ToolEnvelopeStatus = Literal["requested", "running", "ok", "error", "blocked", "cancelled"]
ToolExecutionBackend = Literal["langgraph_tool_node", "internal_workflow"]


class PendingActionTargetScope(TypedDict, total=False):
    files: list[str]
    modules: list[str]
    skill_name: str


class PendingActionSelectionOption(TypedDict, total=False):
    id: str
    label: str
    aliases: list[str]
    value: str
    payload: dict[str, Any]


class PendingAction(TypedDict, total=False):
    id: str
    session_id: str
    type: str
    requested_by_agent: str
    summary: str
    target_scope: PendingActionTargetScope
    risk_level: str
    requires_explicit_approval: bool
    created_at: str
    status: PendingActionStatus
    metadata: dict[str, Any]


class ExecutionContract(TypedDict, total=False):
    pending_action_id: str
    session_id: str
    action_type: str
    requested_by_agent: str
    decision: ExecutionDecision
    summary: str
    reply_text: str
    target_scope: PendingActionTargetScope
    selected_option: PendingActionSelectionOption
    selected_index: int
    requested_outputs: list[str]
    constraints: dict[str, Any]
    should_execute: bool
    interpretation_source: str
    confidence: float


class ExecutionContractValidation(TypedDict, total=False):
    valid: bool
    runtime_action: RuntimeAction
    next_status: PendingActionStatus
    reason: str
    normalized_scope: PendingActionTargetScope
    selected_option: PendingActionSelectionOption
    selected_index: int


class SkillInvocationContract(TypedDict, total=False):
    skill_id: str
    name: str
    description: str
    source_path: str
    mode: SkillInvocationMode
    target_agent: str
    arguments: dict[str, Any]
    source: str
    reason: str
    context_paths: list[str]
    available_to_agents: list[str]
    execution_mode: str
    diagnostics: list[dict[str, Any]]


class ToolInvocationEnvelope(TypedDict, total=False):
    tool_name: str
    tool_id: str
    display_name: str
    tool_family: str
    execution_backend: ToolExecutionBackend
    arguments: dict[str, Any]
    status: ToolEnvelopeStatus
    payload: dict[str, Any]
    error: str
    diagnostics: list[dict[str, Any]]
    source: str
    reason: str
    tool_call_id: str


class ToolResultEnvelope(TypedDict, total=False):
    tool_name: str
    tool_id: str
    display_name: str
    tool_family: str
    execution_backend: ToolExecutionBackend
    arguments: dict[str, Any]
    status: ToolEnvelopeStatus
    payload: dict[str, Any]
    error: str
    diagnostics: list[dict[str, Any]]
    source: str
    reason: str
    tool_call_id: str


class ToolExecutionRecord(TypedDict, total=False):
    invocation: ToolInvocationEnvelope
    result: ToolResultEnvelope


class RoutingDecision(TypedDict, total=False):
    route: str
    reason: str
    policy_step: str
    warnings: list[str]
    diagnostics: list[dict[str, Any]]
    selected_agent: str
    requested_agent: str
    requested_skill_ids: list[str]
    resolved_skill_ids: list[str]
    skill_invocation_contracts: list[SkillInvocationContract]


class AssistantResponse(TypedDict, total=False):
    kind: AssistantResponseKind
    content: str
    pending_action: PendingAction | None
    execution_contract: ExecutionContract | None
    skill_invocation: SkillInvocationContract | None
    tool_invocation: ToolInvocationEnvelope | None
    tool_result: ToolResultEnvelope | None
    routing_decision: RoutingDecision | None
    metadata: dict[str, Any]


def build_assistant_response(
    *,
    kind: AssistantResponseKind = "text",
    content: str = "",
    pending_action: PendingAction | None = None,
    execution_contract: ExecutionContract | None = None,
    skill_invocation: SkillInvocationContract | None = None,
    tool_invocation: ToolInvocationEnvelope | None = None,
    tool_result: ToolResultEnvelope | None = None,
    routing_decision: RoutingDecision | None = None,
    metadata: dict[str, Any] | None = None,
) -> AssistantResponse:
    response: AssistantResponse = {"kind": kind}
    cleaned_content = str(content or "").strip()
    if cleaned_content:
        response["content"] = cleaned_content
    if pending_action is not None:
        response["pending_action"] = pending_action
    if execution_contract is not None:
        response["execution_contract"] = execution_contract
    if skill_invocation is not None:
        response["skill_invocation"] = skill_invocation
    if tool_invocation is not None:
        response["tool_invocation"] = tool_invocation
    if tool_result is not None:
        response["tool_result"] = tool_result
    if routing_decision is not None:
        response["routing_decision"] = routing_decision
    if metadata:
        response["metadata"] = dict(metadata)
    return response


def build_routing_decision(
    route: str,
    *,
    reason: str = "",
    policy_step: str = "",
    warnings: list[str] | None = None,
    diagnostics: list[dict[str, Any]] | None = None,
    selected_agent: str = "",
    requested_agent: str = "",
    requested_skill_ids: list[str] | None = None,
    resolved_skill_ids: list[str] | None = None,
    skill_invocation_contracts: list[SkillInvocationContract] | None = None,
) -> RoutingDecision:
    decision: RoutingDecision = {"route": str(route).strip()}
    cleaned_reason = str(reason or "").strip()
    if cleaned_reason:
        decision["reason"] = cleaned_reason
    cleaned_policy_step = str(policy_step or "").strip()
    if cleaned_policy_step:
        decision["policy_step"] = cleaned_policy_step
    if warnings is not None:
        decision["warnings"] = [str(item) for item in warnings if str(item).strip()]
    if diagnostics is not None:
        decision["diagnostics"] = [dict(item) for item in diagnostics if isinstance(item, dict)]
    cleaned_selected_agent = str(selected_agent or "").strip()
    if cleaned_selected_agent:
        decision["selected_agent"] = cleaned_selected_agent
    cleaned_requested_agent = str(requested_agent or "").strip()
    if cleaned_requested_agent:
        decision["requested_agent"] = cleaned_requested_agent
    if requested_skill_ids is not None:
        decision["requested_skill_ids"] = [str(item).strip() for item in requested_skill_ids if str(item).strip()]
    if resolved_skill_ids is not None:
        decision["resolved_skill_ids"] = [str(item).strip() for item in resolved_skill_ids if str(item).strip()]
    if skill_invocation_contracts is not None:
        decision["skill_invocation_contracts"] = [
            normalize_skill_invocation_contract(contract)
            for contract in skill_invocation_contracts
            if isinstance(contract, dict)
        ]
    return decision


def build_skill_invocation_contract(
    *,
    skill_id: str,
    name: str = "",
    description: str = "",
    source_path: str = "",
    mode: str = "inline",
    target_agent: str = "",
    arguments: dict[str, Any] | None = None,
    source: str = "",
    reason: str = "",
    context_paths: list[str] | tuple[str, ...] | None = None,
    available_to_agents: list[str] | tuple[str, ...] | None = None,
    execution_mode: str = "",
    diagnostics: list[dict[str, Any]] | None = None,
) -> SkillInvocationContract:
    contract: SkillInvocationContract = {
        "skill_id": str(skill_id).strip(),
        "mode": normalize_skill_invocation_mode(mode or execution_mode),
        "target_agent": str(target_agent).strip(),
        "arguments": dict(arguments or {}),
        "source": str(source).strip(),
        "reason": str(reason).strip(),
    }
    cleaned_name = str(name or "").strip()
    if cleaned_name:
        contract["name"] = cleaned_name
    cleaned_description = str(description or "").strip()
    if cleaned_description:
        contract["description"] = cleaned_description
    cleaned_source_path = str(source_path or "").strip()
    if cleaned_source_path:
        contract["source_path"] = cleaned_source_path
    cleaned_context_paths = [str(item).strip() for item in (context_paths or ()) if str(item).strip()]
    if cleaned_context_paths:
        contract["context_paths"] = cleaned_context_paths
    cleaned_agents = [str(item).strip() for item in (available_to_agents or ()) if str(item).strip()]
    if cleaned_agents:
        contract["available_to_agents"] = cleaned_agents
    cleaned_execution_mode = str(execution_mode or "").strip().lower()
    if cleaned_execution_mode:
        contract["execution_mode"] = cleaned_execution_mode
    if diagnostics is not None:
        contract["diagnostics"] = [dict(item) for item in diagnostics if isinstance(item, dict)]
    return contract


def build_tool_invocation_envelope(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    tool_id: str = "",
    display_name: str = "",
    tool_family: str = "",
    execution_backend: str = "",
    status: ToolEnvelopeStatus = "requested",
    payload: dict[str, Any] | None = None,
    error: str = "",
    diagnostics: list[dict[str, Any]] | None = None,
    source: str = "",
    reason: str = "",
    tool_call_id: str = "",
) -> ToolInvocationEnvelope:
    envelope: ToolInvocationEnvelope = {
        "tool_name": str(tool_name).strip(),
        "arguments": dict(arguments or {}),
        "status": status,
    }
    cleaned_tool_id = str(tool_id or "").strip()
    if cleaned_tool_id:
        envelope["tool_id"] = cleaned_tool_id
    cleaned_display_name = str(display_name or "").strip()
    if cleaned_display_name:
        envelope["display_name"] = cleaned_display_name
    cleaned_tool_family = str(tool_family or "").strip()
    if cleaned_tool_family:
        envelope["tool_family"] = cleaned_tool_family
    cleaned_execution_backend = str(execution_backend or "").strip().lower()
    if cleaned_execution_backend:
        envelope["execution_backend"] = cleaned_execution_backend  # type: ignore[assignment]
    if payload is not None:
        envelope["payload"] = dict(payload)
    cleaned_error = str(error or "").strip()
    if cleaned_error:
        envelope["error"] = cleaned_error
    if diagnostics is not None:
        envelope["diagnostics"] = [dict(item) for item in diagnostics if isinstance(item, dict)]
    cleaned_source = str(source or "").strip()
    if cleaned_source:
        envelope["source"] = cleaned_source
    cleaned_reason = str(reason or "").strip()
    if cleaned_reason:
        envelope["reason"] = cleaned_reason
    cleaned_tool_call_id = str(tool_call_id or "").strip()
    if cleaned_tool_call_id:
        envelope["tool_call_id"] = cleaned_tool_call_id
    return envelope


def build_tool_result_envelope(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    tool_id: str = "",
    display_name: str = "",
    tool_family: str = "",
    execution_backend: str = "",
    status: ToolEnvelopeStatus = "ok",
    payload: dict[str, Any] | None = None,
    error: str = "",
    diagnostics: list[dict[str, Any]] | None = None,
    source: str = "",
    reason: str = "",
    tool_call_id: str = "",
) -> ToolResultEnvelope:
    envelope: ToolResultEnvelope = {
        "tool_name": str(tool_name).strip(),
        "arguments": dict(arguments or {}),
        "status": status,
        "payload": dict(payload or {}),
    }
    cleaned_tool_id = str(tool_id or "").strip()
    if cleaned_tool_id:
        envelope["tool_id"] = cleaned_tool_id
    cleaned_display_name = str(display_name or "").strip()
    if cleaned_display_name:
        envelope["display_name"] = cleaned_display_name
    cleaned_tool_family = str(tool_family or "").strip()
    if cleaned_tool_family:
        envelope["tool_family"] = cleaned_tool_family
    cleaned_execution_backend = str(execution_backend or "").strip().lower()
    if cleaned_execution_backend:
        envelope["execution_backend"] = cleaned_execution_backend  # type: ignore[assignment]
    cleaned_error = str(error or "").strip()
    if cleaned_error:
        envelope["error"] = cleaned_error
    if diagnostics is not None:
        envelope["diagnostics"] = [dict(item) for item in diagnostics if isinstance(item, dict)]
    cleaned_source = str(source or "").strip()
    if cleaned_source:
        envelope["source"] = cleaned_source
    cleaned_reason = str(reason or "").strip()
    if cleaned_reason:
        envelope["reason"] = cleaned_reason
    cleaned_tool_call_id = str(tool_call_id or "").strip()
    if cleaned_tool_call_id:
        envelope["tool_call_id"] = cleaned_tool_call_id
    return envelope


def normalize_skill_invocation_mode(raw_value: Any) -> SkillInvocationMode:
    value = str(raw_value or "").strip().lower()
    if value in {"fork", "forked"}:
        return "fork"
    return "inline"


def normalize_skill_invocation_contract(contract: SkillInvocationContract) -> SkillInvocationContract:
    normalized = dict(contract)
    normalized["skill_id"] = safe_get_str(normalized, "skill_id")
    name = safe_get_str(normalized, "name")
    if name:
        normalized["name"] = name
    else:
        normalized.pop("name", None)
    description = safe_get_str(normalized, "description")
    if description:
        normalized["description"] = description
    else:
        normalized.pop("description", None)
    source_path = safe_get_str(normalized, "source_path")
    if source_path:
        normalized["source_path"] = source_path
    else:
        normalized.pop("source_path", None)
    normalized["mode"] = normalize_skill_invocation_mode(normalized.get("mode", "inline"))
    normalized["target_agent"] = safe_get_str(normalized, "target_agent")
    normalized["arguments"] = dict(normalized.get("arguments") or {})
    normalized["source"] = safe_get_str(normalized, "source")
    normalized["reason"] = safe_get_str(normalized, "reason")
    context_paths = normalized.get("context_paths")
    if isinstance(context_paths, list):
        normalized["context_paths"] = [str(item).strip() for item in context_paths if str(item).strip()]
    available_to_agents = normalized.get("available_to_agents")
    if isinstance(available_to_agents, list):
        normalized["available_to_agents"] = [str(item).strip() for item in available_to_agents if str(item).strip()]
    diagnostics = normalized.get("diagnostics")
    if isinstance(diagnostics, list):
        normalized["diagnostics"] = [dict(item) for item in diagnostics if isinstance(item, dict)]
    return normalized


def normalize_tool_invocation_envelope(
    envelope: dict[str, Any] | None,
    *,
    tool_name: str = "",
    arguments: dict[str, Any] | None = None,
    execution_backend: str = "",
    source: str = "",
    reason: str = "",
) -> ToolInvocationEnvelope:
    if not isinstance(envelope, dict):
        return build_tool_invocation_envelope(
            tool_name,
            arguments or {},
            source=source,
            reason=reason,
        )

    if "tool_name" in envelope and "arguments" in envelope and "status" in envelope:
        normalized: ToolInvocationEnvelope = {
            "tool_name": safe_get_str(envelope, "tool_name", tool_name),
            "arguments": dict(envelope.get("arguments") or arguments or {}),
            "status": safe_get_str(envelope, "status", "requested").lower() or "requested",
        }
        tool_id = safe_get_str(envelope, "tool_id")
        if tool_id:
            normalized["tool_id"] = tool_id
        display_name = safe_get_str(envelope, "display_name")
        if display_name:
            normalized["display_name"] = display_name
        tool_family = safe_get_str(envelope, "tool_family")
        if tool_family:
            normalized["tool_family"] = tool_family
        execution_backend_value = safe_get_str(envelope, "execution_backend", execution_backend).lower()
        if execution_backend_value:
            normalized["execution_backend"] = execution_backend_value  # type: ignore[assignment]
        payload = envelope.get("payload")
        if isinstance(payload, dict):
            normalized["payload"] = dict(payload)
        error = safe_get_str(envelope, "error")
        if error:
            normalized["error"] = error
        diagnostics = envelope.get("diagnostics")
        if isinstance(diagnostics, list):
            normalized["diagnostics"] = [dict(item) for item in diagnostics if isinstance(item, dict)]
        source_value = safe_get_str(envelope, "source", source)
        if source_value:
            normalized["source"] = source_value
        reason_value = safe_get_str(envelope, "reason", reason)
        if reason_value:
            normalized["reason"] = reason_value
        tool_call_id = safe_get_str(envelope, "tool_call_id", envelope.get("id", ""))
        if tool_call_id:
            normalized["tool_call_id"] = tool_call_id
        return normalized

    inferred_tool_name = str(envelope.get("tool_name") or envelope.get("name") or tool_name).strip()
    inferred_arguments = envelope.get("arguments")
    if not isinstance(inferred_arguments, dict):
        inferred_arguments = envelope.get("args")
    if not isinstance(inferred_arguments, dict):
        inferred_arguments = arguments or {}

    status = safe_get_str(envelope, "status").lower()
    if not status:
        if envelope.get("ok") is True:
            status = "ok"
        elif envelope.get("requires_confirmation") is True:
            status = "blocked"
        elif envelope.get("cancelled") is True:
            status = "cancelled"
        elif envelope.get("error"):
            status = "error"
        else:
            status = "ok"

    normalized = build_tool_invocation_envelope(
        inferred_tool_name,
        inferred_arguments,
        execution_backend=execution_backend or safe_get_str(envelope, "execution_backend"),
        status=status,  # type: ignore[arg-type]
        payload=dict(envelope),
        error=safe_get_str(envelope, "error"),
        diagnostics=normalize_diagnostics(envelope.get("diagnostics")),
        source=source or safe_get_str(envelope, "source"),
        reason=reason or safe_get_str(envelope, "reason"),
        tool_call_id=safe_get_str(envelope, "tool_call_id", envelope.get("id", "")),
    )
    return normalized


def normalize_tool_result_envelope(
    result: dict[str, Any] | None,
    *,
    tool_name: str = "",
    arguments: dict[str, Any] | None = None,
    execution_backend: str = "",
    source: str = "",
    reason: str = "",
) -> ToolResultEnvelope:
    if not isinstance(result, dict):
        return build_tool_result_envelope(
            tool_name,
            arguments or {},
            execution_backend=execution_backend,
            status="error",
            payload={},
            error="Tool result payload was not a dictionary.",
            source=source,
            reason=reason,
        )

    if "tool_name" in result and "arguments" in result and "status" in result and "payload" in result:
        payload = result.get("payload")
        normalized: ToolResultEnvelope = {
            "tool_name": safe_get_str(result, "tool_name", tool_name),
            "arguments": dict(result.get("arguments") or arguments or {}),
            "status": safe_get_str(result, "status", "ok").lower() or "ok",
            "payload": dict(payload) if isinstance(payload, dict) else {},
        }
        tool_id = safe_get_str(result, "tool_id")
        if tool_id:
            normalized["tool_id"] = tool_id
        display_name = safe_get_str(result, "display_name")
        if display_name:
            normalized["display_name"] = display_name
        tool_family = safe_get_str(result, "tool_family")
        if tool_family:
            normalized["tool_family"] = tool_family
        execution_backend_value = safe_get_str(result, "execution_backend", execution_backend).lower()
        if execution_backend_value:
            normalized["execution_backend"] = execution_backend_value  # type: ignore[assignment]
        error = safe_get_str(result, "error")
        if error:
            normalized["error"] = error
        diagnostics = result.get("diagnostics")
        if isinstance(diagnostics, list):
            normalized["diagnostics"] = [dict(item) for item in diagnostics if isinstance(item, dict)]
        source_value = safe_get_str(result, "source", source)
        if source_value:
            normalized["source"] = source_value
        reason_value = safe_get_str(result, "reason", reason)
        if reason_value:
            normalized["reason"] = reason_value
        tool_call_id = safe_get_str(result, "tool_call_id", result.get("id", ""))
        if tool_call_id:
            normalized["tool_call_id"] = tool_call_id
        return normalized

    status = infer_tool_result_status(result)
    return build_tool_result_envelope(
        tool_name or str(result.get("tool_name") or result.get("name") or "").strip(),
        dict(arguments or result.get("arguments") or result.get("args") or {}),
        execution_backend=execution_backend or safe_get_str(result, "execution_backend"),
        status=status,
        payload=dict(result),
        error=safe_get_str(result, "error"),
        diagnostics=normalize_diagnostics(result.get("diagnostics")),
        source=source or safe_get_str(result, "source"),
        reason=reason or safe_get_str(result, "reason"),
        tool_call_id=safe_get_str(result, "tool_call_id", result.get("id", "")),
    )


def tool_invocation_to_tool_call(envelope: ToolInvocationEnvelope | None) -> dict[str, Any]:
    normalized = normalize_tool_invocation_envelope(
        envelope if isinstance(envelope, dict) else None,
    )
    tool_call_id = safe_get_str(normalized, "tool_call_id") or f"call_{normalized['tool_name']}_{uuid4().hex}"
    return {
        "name": normalized["tool_name"],
        "args": dict(normalized.get("arguments") or {}),
        "id": tool_call_id,
    }


def build_tool_execution_record(
    *,
    invocation: ToolInvocationEnvelope | None = None,
    result: ToolResultEnvelope | None = None,
) -> ToolExecutionRecord:
    record: ToolExecutionRecord = {}
    if invocation is not None:
        record["invocation"] = normalize_tool_invocation_envelope(invocation)
    if result is not None:
        record["result"] = normalize_tool_result_envelope(result)
    return record


def extract_assistant_response_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response.strip()
    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            joined = _stringify_content(content)
            if joined:
                return joined
        tool_result = response.get("tool_result")
        if isinstance(tool_result, dict):
            result_text = extract_tool_result_text(tool_result)
            if result_text:
                return result_text
        return ""
    content = getattr(response, "content", None)
    if content is not None:
        text = _stringify_content(content)
        if text:
            return text
    return _stringify_content(response)


def extract_tool_result_text(result: Any) -> str:
    if not isinstance(result, dict):
        return _stringify_content(result)

    payload = result.get("payload")
    if isinstance(payload, dict):
        for key in ("content", "markdown", "text", "message", "summary", "description"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        content = payload.get("content")
        if content is not None:
            text = _stringify_content(content)
            if text:
                return text
    error = safe_get_str(result, "error")
    if error:
        return error
    return ""


def infer_tool_result_status(result: dict[str, Any]) -> ToolEnvelopeStatus:
    status = safe_get_str(result, "status").lower()
    if status in {"requested", "running", "ok", "error", "blocked", "cancelled"}:
        return status  # type: ignore[return-value]
    if result.get("ok") is True:
        return "ok"
    if result.get("requires_confirmation") is True:
        return "blocked"
    if result.get("cancelled") is True:
        return "cancelled"
    if result.get("error"):
        return "error"
    return "ok"


def normalize_diagnostics(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text.strip())
        return "\n".join(part for part in parts if part).strip()
    if content is None:
        return ""
    return str(content).strip()
