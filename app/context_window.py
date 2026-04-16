from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

DEFAULT_FALLBACK_CONTEXT_WINDOW = 128_000
DEFAULT_OPENAI_CONTEXT_WINDOW = 400_000
DEFAULT_GEMINI_FLASH_CONTEXT_WINDOW = 1_048_576
DEFAULT_GEMINI_PRO_CONTEXT_WINDOW = 2_097_152
DEFAULT_MINIMAX_M2_CONTEXT_WINDOW = 204_800

DEFAULT_WARNING_THRESHOLD_RATIO = 0.70
DEFAULT_AUTO_COMPACT_THRESHOLD_RATIO = 0.85
DEFAULT_HARD_BLOCK_THRESHOLD_RATIO = 0.95
DEFAULT_AUTO_COMPACT_FAILURE_LIMIT = 3

USAGE_BASELINE_STAGE_INCLUSIVE = "inclusive"
USAGE_BASELINE_STAGE_BEFORE_MESSAGE = "before_message"
ContextWindowDecisionLevel = Literal["ok", "warning", "error", "auto_compact", "hard_block"]


@dataclass(frozen=True)
class ContextWindowThresholdOverrides:
    effective_window: int | None = None
    warning_threshold: int | None = None
    auto_compact_threshold: int | None = None
    hard_block_threshold: int | None = None


@dataclass(frozen=True)
class ContextWindowThresholds:
    effective_window: int
    warning_threshold: int
    auto_compact_threshold: int
    hard_block_threshold: int

    def __post_init__(self) -> None:
        values = (
            self.effective_window,
            self.warning_threshold,
            self.auto_compact_threshold,
            self.hard_block_threshold,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Context window thresholds must be positive integers.")
        if not (
            self.warning_threshold
            <= self.auto_compact_threshold
            <= self.hard_block_threshold
            <= self.effective_window
        ):
            raise ValueError(
                "Context window thresholds must satisfy "
                "warning <= auto_compact <= hard_block <= effective_window."
            )


@dataclass(frozen=True)
class TokenCountEstimate:
    baseline_message_id: str | None
    baseline_tokens: int
    estimated_tail_tokens: int
    estimated_total_tokens: int
    used_baseline_usage: bool


@dataclass(frozen=True)
class ContextWindowDecision:
    level: ContextWindowDecisionLevel
    should_warn: bool
    should_auto_compact: bool
    should_hard_block: bool
    should_block: bool
    auto_compact_enabled: bool
    auto_compact_available: bool
    auto_compact_breaker_open: bool


@dataclass(frozen=True)
class ContextWindowSnapshot:
    model: str
    context_window: int
    thresholds: ContextWindowThresholds
    used_tokens: int
    remaining_tokens: int
    remaining_percentage: float
    estimate: TokenCountEstimate
    decision: ContextWindowDecision


def get_context_window_for_model(model: str) -> int:
    normalized = str(model or "").strip().lower()
    if not normalized:
        return DEFAULT_FALLBACK_CONTEXT_WINDOW

    if normalized.startswith("gpt-5"):
        return DEFAULT_OPENAI_CONTEXT_WINDOW
    if normalized.startswith("gpt-4.1") or normalized.startswith("gpt-4o"):
        return DEFAULT_FALLBACK_CONTEXT_WINDOW

    if normalized.startswith("gemini"):
        if "pro" in normalized:
            return DEFAULT_GEMINI_PRO_CONTEXT_WINDOW
        return DEFAULT_GEMINI_FLASH_CONTEXT_WINDOW

    if normalized.startswith("minimax-m2") or normalized.startswith("abab") or normalized.startswith("minimax-"):
        return DEFAULT_MINIMAX_M2_CONTEXT_WINDOW

    return DEFAULT_FALLBACK_CONTEXT_WINDOW


def resolve_context_window_thresholds(
    model: str,
    overrides: ContextWindowThresholdOverrides | None = None,
) -> ContextWindowThresholds:
    resolved_overrides = overrides or ContextWindowThresholdOverrides()
    context_window = get_context_window_for_model(model)
    effective_window = resolved_overrides.effective_window or context_window
    warning_threshold = resolved_overrides.warning_threshold or max(
        1,
        math.floor(effective_window * DEFAULT_WARNING_THRESHOLD_RATIO),
    )
    auto_compact_threshold = resolved_overrides.auto_compact_threshold or max(
        warning_threshold,
        math.floor(effective_window * DEFAULT_AUTO_COMPACT_THRESHOLD_RATIO),
    )
    hard_block_threshold = resolved_overrides.hard_block_threshold or max(
        auto_compact_threshold,
        math.floor(effective_window * DEFAULT_HARD_BLOCK_THRESHOLD_RATIO),
    )
    return ContextWindowThresholds(
        effective_window=effective_window,
        warning_threshold=warning_threshold,
        auto_compact_threshold=auto_compact_threshold,
        hard_block_threshold=hard_block_threshold,
    )


def estimate_transcript_message_tokens(message: dict[str, Any] | Any) -> int:
    normalized = _normalize_accounting_message(message)
    payload = {
        "role": normalized["role"],
        "type": normalized["type"],
        "content": normalized["content"],
    }
    metadata = normalized.get("metadata")
    if normalized["type"] != "message" and isinstance(metadata, dict) and metadata:
        payload["metadata"] = metadata
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    ascii_chars = sum(1 for character in serialized if character.isascii())
    non_ascii_chars = len(serialized) - ascii_chars
    structural_markers = sum(serialized.count(marker) for marker in "\n{}[]:,`")
    estimated = math.ceil(ascii_chars / 4) + non_ascii_chars + math.ceil(structural_markers / 8)
    return max(1, estimated + 4)


def token_count_with_estimation(messages: list[dict[str, Any]] | list[Any]) -> TokenCountEstimate:
    normalized_messages = [_normalize_accounting_message(message) for message in messages]
    baseline_index: int | None = None
    baseline_tokens = 0
    baseline_message_id: str | None = None
    usage_stage = USAGE_BASELINE_STAGE_INCLUSIVE

    for index in range(len(normalized_messages) - 1, -1, -1):
        normalized_message = normalized_messages[index]
        usage_baseline = _extract_usage_baseline_tokens(normalized_message)
        if usage_baseline is None:
            continue
        baseline_index = index
        baseline_tokens = usage_baseline
        baseline_message_id = normalized_message["id"] or None
        usage_stage = str(
            (normalized_message.get("metadata") or {}).get(
                "usage_baseline_stage",
                USAGE_BASELINE_STAGE_INCLUSIVE,
            )
        ).strip() or USAGE_BASELINE_STAGE_INCLUSIVE
        break

    estimated_tail_tokens = 0
    if baseline_index is None:
        estimated_tail_tokens = sum(
            estimate_transcript_message_tokens(message)
            for message in normalized_messages
        )
    else:
        if usage_stage == USAGE_BASELINE_STAGE_BEFORE_MESSAGE:
            estimated_tail_tokens += estimate_transcript_message_tokens(
                normalized_messages[baseline_index]
            )
        for message in normalized_messages[baseline_index + 1 :]:
            estimated_tail_tokens += estimate_transcript_message_tokens(message)

    return TokenCountEstimate(
        baseline_message_id=baseline_message_id,
        baseline_tokens=baseline_tokens,
        estimated_tail_tokens=estimated_tail_tokens,
        estimated_total_tokens=baseline_tokens + estimated_tail_tokens,
        used_baseline_usage=baseline_index is not None,
    )


def evaluate_context_window(
    messages: list[dict[str, Any]] | list[Any],
    *,
    model: str,
    threshold_overrides: ContextWindowThresholdOverrides | None = None,
    auto_compact_enabled: bool = True,
    auto_compact_failure_count: int = 0,
    auto_compact_failure_limit: int = DEFAULT_AUTO_COMPACT_FAILURE_LIMIT,
) -> ContextWindowSnapshot:
    context_window = get_context_window_for_model(model)
    thresholds = resolve_context_window_thresholds(model, threshold_overrides)
    estimate = token_count_with_estimation(messages)
    used_tokens = estimate.estimated_total_tokens
    remaining_tokens = max(thresholds.effective_window - used_tokens, 0)
    remaining_percentage = (
        (remaining_tokens / thresholds.effective_window) * 100
        if thresholds.effective_window > 0
        else 0.0
    )
    decision = _build_context_window_decision(
        used_tokens,
        thresholds,
        auto_compact_enabled=auto_compact_enabled,
        auto_compact_failure_count=auto_compact_failure_count,
        auto_compact_failure_limit=auto_compact_failure_limit,
    )
    return ContextWindowSnapshot(
        model=str(model or "").strip(),
        context_window=context_window,
        thresholds=thresholds,
        used_tokens=used_tokens,
        remaining_tokens=remaining_tokens,
        remaining_percentage=remaining_percentage,
        estimate=estimate,
        decision=decision,
    )


def format_context_window_status(snapshot: ContextWindowSnapshot) -> str:
    return (
        f"context={snapshot.used_tokens}/{snapshot.thresholds.effective_window} tokens "
        f"({snapshot.remaining_percentage:.1f}% remaining) "
        f"status={snapshot.decision.level} "
        f"baseline={snapshot.estimate.baseline_tokens} "
        f"tail_estimate={snapshot.estimate.estimated_tail_tokens}"
    )


def serialize_context_window_snapshot(snapshot: ContextWindowSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def should_auto_compact(
    messages: list[dict[str, Any]] | list[Any],
    *,
    model: str,
    threshold_overrides: ContextWindowThresholdOverrides | None = None,
    auto_compact_enabled: bool = True,
    auto_compact_failure_count: int = 0,
    auto_compact_failure_limit: int = DEFAULT_AUTO_COMPACT_FAILURE_LIMIT,
) -> bool:
    snapshot = evaluate_context_window(
        messages,
        model=model,
        threshold_overrides=threshold_overrides,
        auto_compact_enabled=auto_compact_enabled,
        auto_compact_failure_count=auto_compact_failure_count,
        auto_compact_failure_limit=auto_compact_failure_limit,
    )
    return snapshot.decision.should_auto_compact and snapshot.decision.auto_compact_available


def _build_context_window_decision(
    used_tokens: int,
    thresholds: ContextWindowThresholds,
    *,
    auto_compact_enabled: bool,
    auto_compact_failure_count: int,
    auto_compact_failure_limit: int,
) -> ContextWindowDecision:
    should_warn = used_tokens >= thresholds.warning_threshold
    should_auto_compact = used_tokens >= thresholds.auto_compact_threshold
    should_hard_block = used_tokens >= thresholds.hard_block_threshold
    normalized_failure_limit = max(int(auto_compact_failure_limit or DEFAULT_AUTO_COMPACT_FAILURE_LIMIT), 1)
    normalized_failure_count = max(int(auto_compact_failure_count or 0), 0)
    auto_compact_breaker_open = bool(
        auto_compact_enabled and normalized_failure_count >= normalized_failure_limit
    )
    auto_compact_available = bool(auto_compact_enabled and not auto_compact_breaker_open)
    level: ContextWindowDecisionLevel = "ok"
    if should_hard_block:
        level = "hard_block"
    elif should_auto_compact and auto_compact_available:
        level = "auto_compact"
    elif should_auto_compact:
        level = "error"
    elif should_warn:
        level = "warning"
    should_block = level in {"error", "hard_block"}
    return ContextWindowDecision(
        level=level,
        should_warn=should_warn,
        should_auto_compact=should_auto_compact,
        should_hard_block=should_hard_block,
        should_block=should_block,
        auto_compact_enabled=bool(auto_compact_enabled),
        auto_compact_available=auto_compact_available,
        auto_compact_breaker_open=auto_compact_breaker_open,
    )


def _normalize_accounting_message(message: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return {
            "id": str(message.get("id", "")).strip(),
            "role": str(message.get("role", "")).strip(),
            "type": str(message.get("type", "message")).strip() or "message",
            "content": _stringify_message_content(
                message.get("markdown", message.get("content", ""))
            ),
            "usage": _normalize_usage_dict(message.get("usage")),
            "metadata": _normalize_optional_mapping(message.get("metadata")),
        }

    message_type = str(getattr(message, "type", "")).strip().lower()
    content = _stringify_message_content(getattr(message, "content", ""))
    role = message_type
    transcript_type = "message"
    metadata = None
    usage = None

    if message_type == "human":
        role = "user"
    elif message_type in {"ai", "assistant"}:
        role = "assistant"
        usage = _normalize_usage_dict(getattr(message, "usage_metadata", None))
    elif message_type == "system":
        role = "system"
        additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
        transcript_type = str(additional_kwargs.get("transcript_type", "message")).strip() or "message"
        metadata = _normalize_optional_mapping(additional_kwargs.get("metadata"))

    return {
        "id": str(getattr(message, "id", "") or "").strip(),
        "role": role,
        "type": transcript_type,
        "content": content,
        "usage": usage,
        "metadata": metadata,
    }


def _normalize_optional_mapping(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    normalized = {str(key): item for key, item in value.items()}
    return normalized or None


def _normalize_usage_dict(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    normalized: dict[str, int] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(raw_value, bool):
            continue
        if isinstance(raw_value, int):
            normalized[key] = raw_value
            continue
        if isinstance(raw_value, float) and raw_value.is_integer():
            normalized[key] = int(raw_value)
    return normalized or None


def _extract_usage_baseline_tokens(message: dict[str, Any]) -> int | None:
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None
    metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
    usage_stage = str(metadata.get("usage_baseline_stage", USAGE_BASELINE_STAGE_INCLUSIVE)).strip()
    if usage_stage == USAGE_BASELINE_STAGE_BEFORE_MESSAGE:
        return _usage_input_tokens(usage) or _usage_total_tokens(usage)
    return _usage_total_tokens(usage) or _usage_input_plus_output(usage)


def _usage_total_tokens(usage: dict[str, int]) -> int | None:
    total = usage.get("total_tokens")
    if isinstance(total, int) and total > 0:
        return total
    return None


def _usage_input_tokens(usage: dict[str, int]) -> int | None:
    for key in ("input_tokens", "prompt_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _usage_output_tokens(usage: dict[str, int]) -> int | None:
    for key in ("output_tokens", "completion_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _usage_input_plus_output(usage: dict[str, int]) -> int | None:
    input_tokens = _usage_input_tokens(usage)
    output_tokens = _usage_output_tokens(usage)
    if input_tokens is None and output_tokens is None:
        return None
    return (input_tokens or 0) + (output_tokens or 0)


def _stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return str(content or "")
