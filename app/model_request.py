from __future__ import annotations

from datetime import datetime, timezone
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from app.tool_runtime import parse_tool_message_content

from interfaces.web.conversations import TRANSCRIPT_TYPE_COMPACT_BOUNDARY

RequestReducer = Callable[[list[Any]], list[Any]]
DEFAULT_MICROCOMPACT_TOOL_RESULT_THRESHOLD_CHARS = 600
DEFAULT_MICROCOMPACT_PRESERVE_RECENT_TOOL_RESULTS = 1
DEFAULT_COLD_CACHE_CLEAR_AFTER_SECONDS = 0
DEFAULT_COLD_CACHE_MIN_FOLLOWING_MESSAGES = 3


@dataclass(frozen=True)
class ModelRequestReductionConfig:
    microcompact_tool_result_threshold_chars: int = DEFAULT_MICROCOMPACT_TOOL_RESULT_THRESHOLD_CHARS
    preserve_recent_tool_results: int = DEFAULT_MICROCOMPACT_PRESERVE_RECENT_TOOL_RESULTS
    cold_cache_clear_after_seconds: int = DEFAULT_COLD_CACHE_CLEAR_AFTER_SECONDS
    cold_cache_min_following_messages: int = DEFAULT_COLD_CACHE_MIN_FOLLOWING_MESSAGES


@dataclass(frozen=True)
class ModelRequestReducerHooks:
    snip: RequestReducer | None = None
    microcompact: RequestReducer | None = None
    collapse: RequestReducer | None = None
    auto_compact: RequestReducer | None = None


def get_messages_after_compact_boundary(messages: Sequence[Any] | None) -> list[Any]:
    normalized_messages = list(messages or [])
    last_boundary_index = -1
    for index, message in enumerate(normalized_messages):
        if is_compact_boundary_message(message):
            last_boundary_index = index
    if last_boundary_index < 0:
        return normalized_messages
    return normalized_messages[last_boundary_index + 1 :]


def build_model_request_messages(
    *,
    system_prompt: str = "",
    transcript_messages: Sequence[Any] | None = None,
    extra_messages: Sequence[Any] | None = None,
    reducer_hooks: ModelRequestReducerHooks | None = None,
    reduction_config: ModelRequestReductionConfig | None = None,
    use_projection_pipeline: bool = True,
) -> list[Any]:
    request_messages: list[Any] = []
    cleaned_system_prompt = str(system_prompt or "").strip()
    if cleaned_system_prompt:
        request_messages.append(SystemMessage(content=cleaned_system_prompt))

    projected_messages = (
        project_transcript_messages(
            transcript_messages,
            reducer_hooks=reducer_hooks,
            reduction_config=reduction_config,
        )
        if use_projection_pipeline
        else list(transcript_messages or [])
    )
    request_messages.extend(projected_messages)
    request_messages.extend(list(extra_messages or []))
    return request_messages


def project_transcript_messages(
    transcript_messages: Sequence[Any] | None,
    *,
    reducer_hooks: ModelRequestReducerHooks | None = None,
    reduction_config: ModelRequestReductionConfig | None = None,
) -> list[Any]:
    projected_messages = get_messages_after_compact_boundary(transcript_messages)
    config = reduction_config or ModelRequestReductionConfig()
    hooks = reducer_hooks or ModelRequestReducerHooks()
    projected_messages = _apply_builtin_snip_reducer(projected_messages, config)
    projected_messages = _apply_optional_reducer(projected_messages, hooks.snip)
    projected_messages = _apply_builtin_microcompact_reducer(projected_messages, config)
    projected_messages = _apply_optional_reducer(projected_messages, hooks.microcompact)
    projected_messages = _apply_optional_reducer(projected_messages, hooks.collapse)
    projected_messages = _apply_optional_reducer(projected_messages, hooks.auto_compact)
    return projected_messages


def project_model_facing_messages(
    transcript_messages: Sequence[Any] | None,
    *,
    reducer_hooks: ModelRequestReducerHooks | None = None,
    reduction_config: ModelRequestReductionConfig | None = None,
) -> list[Any]:
    return project_transcript_messages(
        transcript_messages,
        reducer_hooks=reducer_hooks,
        reduction_config=reduction_config,
    )


def is_compact_boundary_message(message: Any) -> bool:
    if isinstance(message, dict):
        message_type = str(message.get("type", "")).strip()
        if message_type == TRANSCRIPT_TYPE_COMPACT_BOUNDARY:
            return True
        additional_kwargs = message.get("additional_kwargs")
        if isinstance(additional_kwargs, dict):
            return str(additional_kwargs.get("transcript_type", "")).strip() == TRANSCRIPT_TYPE_COMPACT_BOUNDARY
        return False

    message_type = str(getattr(message, "type", "")).strip().lower()
    if message_type != "system":
        return False
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    if not isinstance(additional_kwargs, dict):
        return False
    return str(additional_kwargs.get("transcript_type", "")).strip() == TRANSCRIPT_TYPE_COMPACT_BOUNDARY


def _apply_optional_reducer(messages: list[Any], reducer: RequestReducer | None) -> list[Any]:
    if not callable(reducer):
        return list(messages)
    reduced = reducer(list(messages))
    return list(reduced or [])


def _apply_builtin_snip_reducer(
    messages: list[Any],
    config: ModelRequestReductionConfig,
) -> list[Any]:
    clear_after_seconds = max(int(config.cold_cache_clear_after_seconds or 0), 0)
    min_following_messages = max(int(config.cold_cache_min_following_messages or 0), 0)
    if clear_after_seconds <= 0:
        return list(messages)

    latest_timestamp = _latest_message_timestamp(messages)
    if latest_timestamp is None:
        return list(messages)

    reduced_messages: list[Any] = []
    for index, message in enumerate(messages):
        if not _is_cold_cache_clearable_message(message):
            reduced_messages.append(message)
            continue

        message_timestamp = _extract_message_timestamp(message)
        if message_timestamp is None:
            reduced_messages.append(message)
            continue

        age_seconds = (latest_timestamp - message_timestamp).total_seconds()
        following_messages = len(messages) - index - 1
        if age_seconds >= clear_after_seconds and following_messages >= min_following_messages:
            continue
        reduced_messages.append(message)
    return reduced_messages


def _apply_builtin_microcompact_reducer(
    messages: list[Any],
    config: ModelRequestReductionConfig,
) -> list[Any]:
    threshold_chars = max(int(config.microcompact_tool_result_threshold_chars or 0), 0)
    if threshold_chars <= 0:
        return list(messages)

    tool_result_indices = [index for index, message in enumerate(messages) if isinstance(message, ToolMessage)]
    if not tool_result_indices:
        return list(messages)

    preserve_recent_tool_results = max(int(config.preserve_recent_tool_results or 0), 0)
    preserved_tool_indices = set(tool_result_indices[-preserve_recent_tool_results:]) if preserve_recent_tool_results > 0 else set()
    preserved_invocation_indices = {
        invocation_index
        for invocation_index in (
            _find_matching_tool_invocation_index(messages, tool_result_index)
            for tool_result_index in preserved_tool_indices
        )
        if invocation_index is not None
    }

    reduced_messages: list[Any] = []
    for index, message in enumerate(messages):
        if isinstance(message, ToolMessage):
            if index in preserved_tool_indices or len(_stringify_message_content(getattr(message, "content", ""))) < threshold_chars:
                reduced_messages.append(message)
                continue
            reduced_messages.append(_build_microcompact_tool_result_message(messages, index))
            continue

        if _is_tool_invocation_message(message) and index not in preserved_invocation_indices:
            paired_tool_index = _find_following_tool_result_index(messages, index)
            if paired_tool_index is not None and paired_tool_index not in preserved_tool_indices:
                replacement = _strip_or_drop_tool_invocation_message(message)
                if replacement is not None:
                    reduced_messages.append(replacement)
                continue

        reduced_messages.append(message)
    return reduced_messages


def _build_microcompact_tool_result_message(messages: list[Any], tool_result_index: int) -> SystemMessage:
    tool_message = messages[tool_result_index]
    tool_call_id = str(getattr(tool_message, "tool_call_id", "") or "").strip()
    tool_name = _resolve_tool_name(messages, tool_result_index)
    content = _stringify_message_content(getattr(tool_message, "content", ""))
    parsed_payload = parse_tool_message_content(content)
    summary = _summarize_tool_payload(parsed_payload, raw_content=content)
    created_at = _extract_message_created_at_value(tool_message)
    metadata: dict[str, Any] = {
        "cheap_context_reduction": {
            "kind": "microcompact_tool_result",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
        }
    }
    if created_at:
        metadata["created_at"] = created_at
    return SystemMessage(
        content=(
            f"Earlier tool result `{tool_name or 'tool'}` was compressed to keep context small. "
            f"{summary}"
        ).strip(),
        additional_kwargs=metadata,
    )


def _strip_or_drop_tool_invocation_message(message: Any) -> AIMessage | None:
    content = _stringify_message_content(getattr(message, "content", ""))
    if not content.strip():
        return None
    return AIMessage(
        content=content.strip(),
        id=getattr(message, "id", None),
        name=getattr(message, "name", None),
        additional_kwargs=_copy_message_additional_kwargs(message),
        response_metadata=dict(getattr(message, "response_metadata", {}) or {}),
    )


def _summarize_tool_payload(payload: dict[str, Any] | None, *, raw_content: str) -> str:
    if not isinstance(payload, dict):
        return _truncate_text(raw_content, limit=220)

    if payload.get("ok") is False:
        error_text = _first_non_empty_value(payload, keys=("error", "message", "detail", "content"))
        return f"The tool reported an error: {_truncate_text(error_text or 'request failed', limit=180)}"

    summary_parts: list[str] = []
    for key in ("document", "task", "item", "result"):
        value = payload.get(key)
        if isinstance(value, dict):
            label = _first_non_empty_value(value, keys=("title", "name", "path", "id"))
            if label:
                summary_parts.append(f"{key}={label}")
                break

    for key in ("documents", "tasks", "items", "results", "files"):
        value = payload.get(key)
        if isinstance(value, list):
            sample_labels = _extract_item_labels(value)
            if sample_labels:
                summary_parts.append(f"{key}={len(value)} ({', '.join(sample_labels)})")
            else:
                summary_parts.append(f"{key}={len(value)}")
            break

    for key in ("path", "relative_path", "section_query"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            summary_parts.append(f"{key}={value.strip()}")
            break

    content_text = payload.get("content")
    if isinstance(content_text, str) and content_text.strip():
        summary_parts.append(
            f"content={_truncate_text(content_text, limit=180)} ({len(content_text.strip())} chars)"
        )

    if not summary_parts:
        scalar_parts = []
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)) and str(value).strip():
                scalar_parts.append(f"{key}={value}")
            if len(scalar_parts) >= 3:
                break
        summary_parts.extend(scalar_parts)

    if not summary_parts:
        return f"Raw result size was {len(raw_content)} characters."
    return "; ".join(summary_parts[:4])


def _extract_item_labels(values: list[Any], *, limit: int = 3) -> list[str]:
    labels: list[str] = []
    for item in values:
        if len(labels) >= limit:
            break
        if isinstance(item, dict):
            label = _first_non_empty_value(item, keys=("title", "name", "path", "id", "content"))
        else:
            label = str(item or "").strip()
        if not label:
            continue
        truncated = _truncate_text(label, limit=48)
        if truncated not in labels:
            labels.append(truncated)
    return labels


def _first_non_empty_value(value: dict[str, Any], *, keys: tuple[str, ...]) -> str:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return str(candidate)
    return ""


def _resolve_tool_name(messages: list[Any], tool_result_index: int) -> str:
    tool_message = messages[tool_result_index]
    invocation_index = _find_matching_tool_invocation_index(messages, tool_result_index)
    if invocation_index is not None:
        tool_calls = getattr(messages[invocation_index], "tool_calls", []) or []
        tool_call_id = str(getattr(tool_message, "tool_call_id", "") or "").strip()
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            current_id = str(tool_call.get("id", "") or "").strip()
            current_name = str(tool_call.get("name", "") or "").strip()
            if tool_call_id and current_id == tool_call_id and current_name:
                return current_name
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            current_name = str(tool_call.get("name", "") or "").strip()
            if current_name:
                return current_name
    return str(getattr(tool_message, "name", "") or "").strip()


def _find_matching_tool_invocation_index(messages: list[Any], tool_result_index: int) -> int | None:
    tool_message = messages[tool_result_index]
    tool_call_id = str(getattr(tool_message, "tool_call_id", "") or "").strip()
    for index in range(tool_result_index - 1, -1, -1):
        message = messages[index]
        if not _is_tool_invocation_message(message):
            continue
        tool_calls = getattr(message, "tool_calls", []) or []
        if tool_call_id:
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and str(tool_call.get("id", "") or "").strip() == tool_call_id:
                    return index
        if tool_calls:
            return index
    return None


def _find_following_tool_result_index(messages: list[Any], invocation_index: int) -> int | None:
    tool_calls = getattr(messages[invocation_index], "tool_calls", []) or []
    tool_call_ids = {
        str(tool_call.get("id", "") or "").strip()
        for tool_call in tool_calls
        if isinstance(tool_call, dict) and str(tool_call.get("id", "") or "").strip()
    }
    for index in range(invocation_index + 1, len(messages)):
        message = messages[index]
        if not isinstance(message, ToolMessage):
            continue
        tool_call_id = str(getattr(message, "tool_call_id", "") or "").strip()
        if not tool_call_ids or (tool_call_id and tool_call_id in tool_call_ids):
            return index
    return None


def _is_tool_invocation_message(message: Any) -> bool:
    return isinstance(message, AIMessage) and bool(getattr(message, "tool_calls", []) or [])


def _is_cold_cache_clearable_message(message: Any) -> bool:
    if isinstance(message, ToolMessage):
        return True
    if _is_tool_invocation_message(message):
        return True
    if isinstance(message, SystemMessage):
        additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
        if not isinstance(additional_kwargs, dict):
            return False
        cheap_context_reduction = additional_kwargs.get("cheap_context_reduction")
        return isinstance(cheap_context_reduction, dict)
    return False


def _latest_message_timestamp(messages: Sequence[Any]) -> datetime | None:
    timestamps = [_extract_message_timestamp(message) for message in messages]
    timestamps = [timestamp for timestamp in timestamps if timestamp is not None]
    if not timestamps:
        return None
    return max(timestamps)


def _extract_message_timestamp(message: Any) -> datetime | None:
    raw_value = _extract_message_created_at_value(message)
    if not raw_value:
        return None
    normalized_value = raw_value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized_value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _extract_message_created_at_value(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("created_at", "") or "").strip()
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    if isinstance(additional_kwargs, dict):
        created_at = str(additional_kwargs.get("created_at", "") or "").strip()
        if created_at:
            return created_at
    response_metadata = getattr(message, "response_metadata", {}) or {}
    if isinstance(response_metadata, dict):
        created_at = str(response_metadata.get("created_at", "") or "").strip()
        if created_at:
            return created_at
    metadata = getattr(message, "metadata", None)
    if isinstance(metadata, dict):
        return str(metadata.get("created_at", "") or "").strip()
    return ""


def _copy_message_additional_kwargs(message: Any) -> dict[str, Any]:
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    if not isinstance(additional_kwargs, dict):
        return {}
    copied = dict(additional_kwargs)
    copied.pop("tool_calls", None)
    return copied


def _stringify_message_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
                    continue
            serialized = _serialize_scalar(item)
            if serialized:
                parts.append(serialized)
        return "\n".join(part for part in parts if part).strip()
    return _serialize_scalar(value)


def _serialize_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value).strip()


def _truncate_text(text: str, *, limit: int) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(limit - 1, 0)].rstrip() + "…"
