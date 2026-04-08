from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.contracts import PendingAction

ALLOWED_REQUESTED_OUTPUTS = {"diff", "preview", "plan", "summary", "details"}
INTERPRETATION_SOURCE_DETERMINISTIC_SELECTION = "deterministic_selection"
INTERPRETATION_SOURCE_LLM_PARSER = "llm_parser"
DEFAULT_LOW_CONFIDENCE_REASON = "The pending-action reply parser was not confident enough to produce a safe executable interpretation."
DEFAULT_UNAVAILABLE_REASON = "The pending-action reply parser was unavailable."
DEFAULT_MALFORMED_REASON = "The pending-action reply parser returned malformed output."


class ParsedPendingActionReply(TypedDict, total=False):
    decision: str
    requested_outputs: list[str]
    target_scope: dict[str, Any]
    selected_index: int | None
    should_execute: bool
    reason: str
    confidence: float
    interpretation_source: str


class PendingActionReplyInterpreter(Protocol):
    def parse_pending_action_reply(
        self,
        action: PendingAction,
        prepared_input: dict[str, Any],
    ) -> ParsedPendingActionReply:
        ...


class PendingActionReplyLLMOutput(BaseModel):
    decision: str = "unclear"
    requested_outputs: list[str] = Field(default_factory=list)
    target_scope: dict[str, Any] = Field(default_factory=dict)
    selected_index: int | None = None
    should_execute: bool = False
    reason: str = ""
    confidence: float = 0.0


class LLMPendingActionInterpreter:
    def __init__(self, llm: Any, *, confidence_threshold: float = 0.75) -> None:
        self._parser = llm.with_structured_output(PendingActionReplyLLMOutput)
        self.confidence_threshold = float(confidence_threshold)

    def parse_pending_action_reply(
        self,
        action: PendingAction,
        prepared_input: dict[str, Any],
    ) -> ParsedPendingActionReply:
        try:
            raw_output = self._parser.invoke(build_pending_action_reply_parser_prompt(action, prepared_input))
        except Exception:
            return build_unclear_pending_action_parse(DEFAULT_UNAVAILABLE_REASON)
        return normalize_llm_pending_action_parse(
            raw_output,
            confidence_threshold=self.confidence_threshold,
        )


def build_unclear_pending_action_parse(reason: str = "") -> ParsedPendingActionReply:
    return ParsedPendingActionReply(
        decision="unclear",
        requested_outputs=[],
        target_scope={},
        selected_index=None,
        should_execute=False,
        reason=str(reason or "").strip(),
        confidence=0.0,
        interpretation_source=INTERPRETATION_SOURCE_LLM_PARSER,
    )


def normalize_llm_pending_action_parse(
    raw_output: Any,
    *,
    confidence_threshold: float,
) -> ParsedPendingActionReply:
    parsed = coerce_pending_action_parse_dict(raw_output)
    if parsed is None:
        return build_unclear_pending_action_parse(DEFAULT_MALFORMED_REASON)

    decision = str(parsed.get("decision", "")).strip().lower()
    if decision not in {"approve", "reject", "modify", "select", "unclear"}:
        return build_unclear_pending_action_parse(DEFAULT_MALFORMED_REASON)

    confidence = clamp_confidence(parsed.get("confidence", 0.0))
    if confidence < float(confidence_threshold):
        reason = str(parsed.get("reason", "")).strip() or DEFAULT_LOW_CONFIDENCE_REASON
        return build_unclear_pending_action_parse(reason)

    requested_outputs = normalize_requested_outputs(parsed.get("requested_outputs"))
    target_scope = normalize_target_scope(parsed.get("target_scope"))
    selected_index = normalize_selected_index(parsed.get("selected_index"))
    should_execute = bool(parsed.get("should_execute", False))
    reason = str(parsed.get("reason", "")).strip()

    if decision == "reject":
        should_execute = False
    elif decision == "unclear":
        should_execute = False
        requested_outputs = []
        target_scope = {}
        selected_index = None

    return ParsedPendingActionReply(
        decision=decision,
        requested_outputs=requested_outputs,
        target_scope=target_scope,
        selected_index=selected_index,
        should_execute=should_execute,
        reason=reason,
        confidence=confidence,
        interpretation_source=INTERPRETATION_SOURCE_LLM_PARSER,
    )


def build_pending_action_reply_parser_prompt(
    action: PendingAction,
    prepared_input: dict[str, Any],
) -> str:
    summary = str(action.get("summary", "")).strip()
    action_type = str(action.get("type", "")).strip()
    user_reply = str(prepared_input.get("user_reply", "")).strip()
    target_scope = prepared_input.get("target_scope")
    selection_options = prepared_input.get("selection_options")

    lines = [
        "You interpret a user's reply to a pending action.",
        "Return only a structured parse.",
        "Do not authorize execution; that is handled later by deterministic validation.",
        "",
        "Allowed decisions: approve, reject, modify, select, unclear",
        "Allowed requested_outputs values: diff, preview, plan, summary, details",
        "Only use `select` when the reply explicitly identifies one of the provided options.",
        "If the reply is ambiguous, unsafe, or uncertain, return `unclear`.",
        "",
        f"Pending action type: {action_type}",
        f"Pending action summary: {summary}",
        f"User reply: {user_reply}",
        f"Allowed target scope: {target_scope if isinstance(target_scope, dict) else {}}",
        f"Selection options: {selection_options if isinstance(selection_options, list) else []}",
    ]
    return "\n".join(lines).strip()


def coerce_pending_action_parse_dict(raw_output: Any) -> dict[str, Any] | None:
    if isinstance(raw_output, dict):
        return dict(raw_output)
    model_dump = getattr(raw_output, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return None


def clamp_confidence(value: Any) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(resolved, 1.0))


def normalize_requested_outputs(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = [str(item).strip().lower() for item in value if str(item).strip()]
    else:
        raw_items = []

    normalized: list[str] = []
    for item in raw_items:
        if item in ALLOWED_REQUESTED_OUTPUTS and item not in normalized:
            normalized.append(item)
    return normalized


def normalize_target_scope(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, Any] = {}
    for field_name in ("files", "modules"):
        raw_items = value.get(field_name)
        if not isinstance(raw_items, (list, tuple, set)):
            continue
        cleaned_items = [str(item).strip() for item in raw_items if str(item).strip()]
        if cleaned_items:
            normalized[field_name] = cleaned_items

    skill_name = str(value.get("skill_name", "")).strip()
    if skill_name:
        normalized["skill_name"] = skill_name
    return normalized


def normalize_selected_index(value: Any) -> int | None:
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, str) and value.strip().isdigit():
        resolved = int(value.strip())
        return resolved if resolved >= 0 else None
    return None
