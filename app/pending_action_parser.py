from __future__ import annotations

import json
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
    def __init__(
        self,
        llm: Any,
        *,
        backup_llm: Any | None = None,
        confidence_threshold: float = 0.75,
        max_parse_attempts: int = 2,
    ) -> None:
        self._primary_llm = llm
        self._primary_parser = llm.with_structured_output(PendingActionReplyLLMOutput)
        self._backup_llm = backup_llm
        self._backup_parser = backup_llm.with_structured_output(PendingActionReplyLLMOutput) if backup_llm is not None else None
        self.confidence_threshold = float(confidence_threshold)
        self.max_parse_attempts = max(int(max_parse_attempts), 1)

    def parse_pending_action_reply(
        self,
        action: PendingAction,
        prepared_input: dict[str, Any],
    ) -> ParsedPendingActionReply:
        prompt = build_pending_action_reply_parser_prompt(action, prepared_input)
        latest_failure_reason = DEFAULT_UNAVAILABLE_REASON

        parsed, latest_failure_reason = self._try_structured_parser(
            self._primary_parser,
            self._primary_llm,
            prompt,
        )
        if parsed is not None:
            return parsed

        if self._backup_parser is not None and self._backup_llm is not None:
            parsed, latest_failure_reason = self._try_structured_parser(
                self._backup_parser,
                self._backup_llm,
                prompt,
            )
            if parsed is not None:
                return parsed

        return build_unclear_pending_action_parse(latest_failure_reason)

    def _try_structured_parser(
        self,
        parser: Any,
        raw_llm: Any,
        prompt: str,
    ) -> tuple[ParsedPendingActionReply | None, str]:
        latest_failure_reason = DEFAULT_UNAVAILABLE_REASON
        for _ in range(self.max_parse_attempts):
            try:
                raw_output = parser.invoke(prompt)
            except Exception:
                latest_failure_reason = DEFAULT_UNAVAILABLE_REASON
                continue

            parsed, failure_reason = attempt_normalize_llm_pending_action_parse(
                raw_output,
                confidence_threshold=self.confidence_threshold,
            )
            if parsed is not None:
                return parsed, ""

            latest_failure_reason = failure_reason
            repaired = self._repair_pending_action_reply(
                parser,
                raw_llm,
                prompt,
                raw_output,
                failure_reason=failure_reason,
            )
            if repaired is not None:
                return repaired, ""

        return None, latest_failure_reason

    def _repair_pending_action_reply(
        self,
        parser: Any,
        raw_llm: Any,
        prompt: str,
        raw_output: Any,
        *,
        failure_reason: str,
    ) -> ParsedPendingActionReply | None:
        try:
            raw_completion = raw_llm.invoke(prompt)
        except Exception:
            raw_completion = raw_output

        try:
            repaired_output = parser.invoke(
                build_pending_action_reply_repair_prompt(
                    original_prompt=prompt,
                    original_output=raw_completion if raw_completion is not None else raw_output,
                    failure_reason=failure_reason,
                )
            )
        except Exception:
            return None

        parsed, _ = attempt_normalize_llm_pending_action_parse(
            repaired_output,
            confidence_threshold=self.confidence_threshold,
        )
        return parsed


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
    parsed, failure_reason = attempt_normalize_llm_pending_action_parse(
        raw_output,
        confidence_threshold=confidence_threshold,
    )
    if parsed is not None:
        return parsed
    return build_unclear_pending_action_parse(failure_reason)


def attempt_normalize_llm_pending_action_parse(
    raw_output: Any,
    *,
    confidence_threshold: float,
) -> tuple[ParsedPendingActionReply | None, str]:
    parsed = coerce_pending_action_parse_dict(raw_output)
    if parsed is None:
        return None, DEFAULT_MALFORMED_REASON

    decision = str(parsed.get("decision", "")).strip().lower()
    if decision not in {"approve", "reject", "modify", "select", "unclear"}:
        return None, DEFAULT_MALFORMED_REASON

    confidence = clamp_confidence(parsed.get("confidence", 0.0))
    if confidence < float(confidence_threshold):
        reason = str(parsed.get("reason", "")).strip() or DEFAULT_LOW_CONFIDENCE_REASON
        return None, reason

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

    return (
        ParsedPendingActionReply(
            decision=decision,
            requested_outputs=requested_outputs,
            target_scope=target_scope,
            selected_index=selected_index,
            should_execute=should_execute,
            reason=reason,
            confidence=confidence,
            interpretation_source=INTERPRETATION_SOURCE_LLM_PARSER,
        ),
        "",
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
        "Only use `select` when the reply clearly identifies one of the provided options, including ordinal references like `the second one`.",
        "Do not map a generic `show` request to `preview` unless the user explicitly asked for a preview or diff.",
        "Use `summary` for requests like `show me a summary`.",
        "Use `details` for requests like `details` or `show more detail` on a selected item.",
        "Treat negation or deferment like `do not continue` or `not yet` as non-approval.",
        "If the reply is ambiguous, unsafe, or uncertain, return `unclear`.",
        "",
        "Examples:",
        '- Reply: "show me a summary" -> decision=modify requested_outputs=["summary"] should_execute=false',
        '- Reply: "show me the diff first" -> decision=modify requested_outputs=["diff"] should_execute=false',
        '- Reply: "the second one" with selection options -> decision=select selected_index=1 should_execute=true',
        '- Reply: "do not continue" -> decision=reject should_execute=false',
        "",
        f"Pending action type: {action_type}",
        f"Pending action summary: {summary}",
        f"User reply: {user_reply}",
        f"Allowed target scope: {target_scope if isinstance(target_scope, dict) else {}}",
        f"Selection options: {selection_options if isinstance(selection_options, list) else []}",
    ]
    return "\n".join(lines).strip()


def build_pending_action_reply_repair_prompt(
    *,
    original_prompt: str,
    original_output: Any,
    failure_reason: str,
) -> str:
    serialized_output = stringify_llm_output(original_output)
    return "\n".join(
        [
            "Repair the following pending-action parse attempt.",
            "Return only a structured parse that matches the required schema.",
            "Fix malformed, low-confidence, or incomplete output without changing the user's intent.",
            f"Failure reason: {failure_reason}",
            "",
            "Original parsing task:",
            original_prompt,
            "",
            "Previous model output:",
            serialized_output,
        ]
    ).strip()


def coerce_pending_action_parse_dict(raw_output: Any) -> dict[str, Any] | None:
    if isinstance(raw_output, dict):
        return dict(raw_output)
    model_dump = getattr(raw_output, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    return None


def stringify_llm_output(raw_output: Any) -> str:
    if isinstance(raw_output, str):
        return raw_output
    if isinstance(raw_output, dict):
        return json.dumps(raw_output, ensure_ascii=False, sort_keys=True)
    content = getattr(raw_output, "content", None)
    if isinstance(content, str):
        return content
    model_dump = getattr(raw_output, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return json.dumps(dumped, ensure_ascii=False, sort_keys=True)
        return str(dumped)
    return str(raw_output)


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
