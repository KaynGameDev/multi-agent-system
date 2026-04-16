from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import AIMessage

ReactiveRecoveryKind = Literal["prompt_too_long", "max_output_tokens"]

PROMPT_TOO_LONG_PATTERNS = (
    "maximum context length",
    "context length exceeded",
    "prompt is too long",
    "prompt too long",
    "request too large",
    "input is too long",
    "input too long",
    "too many input tokens",
    "context window exceeded",
    "too many tokens in prompt",
)
MAX_OUTPUT_PATTERNS = (
    "max_output_tokens",
    "maximum output tokens",
    "maximum completion tokens",
    "output token limit",
    "too many output tokens",
    "finish_reason=length",
    "finish reason=length",
    "finish_reason: length",
    "finish reason: length",
    "response was truncated",
)
MAX_OUTPUT_STOP_REASONS = {
    "length",
    "max_tokens",
    "max_output_tokens",
    "output_token_limit",
}


@dataclass(frozen=True)
class ReactiveRecoverySignal:
    kind: ReactiveRecoveryKind
    detail: str = ""


def detect_reactive_recovery_signal_from_exception(exc: Exception | None) -> ReactiveRecoverySignal | None:
    for detail in _iter_exception_details(exc):
        normalized = detail.lower()
        if any(pattern in normalized for pattern in PROMPT_TOO_LONG_PATTERNS):
            return ReactiveRecoverySignal(kind="prompt_too_long", detail=detail)
        if any(pattern in normalized for pattern in MAX_OUTPUT_PATTERNS):
            return ReactiveRecoverySignal(kind="max_output_tokens", detail=detail)
    return None


def detect_reactive_recovery_signal_from_final_state(final_state: dict[str, Any] | None) -> ReactiveRecoverySignal | None:
    if not isinstance(final_state, dict):
        return None

    messages = final_state.get("messages")
    if not isinstance(messages, list):
        return None

    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        stop_reason = _extract_stop_reason(message)
        if stop_reason in MAX_OUTPUT_STOP_REASONS:
            detail = stop_reason
            return ReactiveRecoverySignal(kind="max_output_tokens", detail=detail)
    return None


def build_reactive_recovery_detail(
    signal: ReactiveRecoverySignal,
    *,
    retry_exhausted: bool = False,
    compaction_failed: bool = False,
) -> str:
    if signal.kind == "prompt_too_long":
        prefix = "The model rejected this request because the prompt/context was too large."
        guidance = "Please compact the conversation manually and try again."
    else:
        prefix = "The model hit its output limit before it could finish the reply."
        guidance = "Please compact the conversation manually or ask for a narrower answer, then try again."

    if compaction_failed:
        return f"{prefix} I tried to compact and retry automatically, but the recovery compaction failed. {guidance}"
    if retry_exhausted:
        return f"{prefix} I compacted and retried once, but it still hit the same limit. {guidance}"
    return f"{prefix} I couldn't recover automatically. {guidance}"


def _iter_exception_details(exc: Exception | None) -> list[str]:
    details: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        detail = str(current).strip()
        if detail:
            details.append(detail)
        current = current.__cause__ or current.__context__
    return details


def _extract_stop_reason(message: AIMessage) -> str:
    metadata_sources = [
        getattr(message, "response_metadata", {}) or {},
        getattr(message, "additional_kwargs", {}) or {},
    ]
    for metadata in metadata_sources:
        if not isinstance(metadata, dict):
            continue
        for key in ("finish_reason", "finishReason", "stop_reason", "stopReason"):
            value = metadata.get(key)
            normalized = str(value or "").strip().lower()
            if normalized:
                return normalized
    return ""
