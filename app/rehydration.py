from __future__ import annotations

from copy import deepcopy
from typing import Any

RUNTIME_REHYDRATION_METADATA_KEY = "runtime_rehydration_state"
TRANSCRIPT_TYPE_COMPACT_BOUNDARY = "compact_boundary"

STRING_STATE_FIELDS = (
    "requested_agent",
    "pending_action_resolution_key",
    "conversion_session_id",
    "target_game_slug",
    "target_market_slug",
    "target_feature_slug",
    "conversion_status",
    "approval_state",
)
STRING_LIST_STATE_FIELDS = (
    "requested_skill_ids",
    "resolved_skill_ids",
    "context_paths",
    "missing_required_fields",
    "recent_file_reads",
)
JSON_STATE_FIELDS = (
    "pending_action",
    "pending_action_decision",
    "execution_contract",
    "routing_decision",
    "skill_invocation_contracts",
    "active_skill_invocation_contracts",
    "skill_execution_diagnostics",
    "tool_invocation",
    "tool_result",
    "tool_execution_trace",
    "uploaded_files",
)
PATH_LIKE_KEYS = {
    "absolute_path",
    "file_path",
    "package_path",
    "path",
    "raw_path",
    "relative_package_path",
    "relative_path",
    "staged_package_path",
}
FILES_CONTAINER_KEYS = {"files"}
MAX_RECENT_FILE_READS = 8


def build_runtime_rehydration_state(state: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}

    snapshot: dict[str, Any] = {}
    for field_name in STRING_STATE_FIELDS:
        cleaned = str(state.get(field_name, "") or "").strip()
        if cleaned:
            snapshot[field_name] = cleaned

    for field_name in STRING_LIST_STATE_FIELDS:
        cleaned_items = _normalize_string_list(state.get(field_name))
        if cleaned_items:
            snapshot[field_name] = cleaned_items

    for field_name in JSON_STATE_FIELDS:
        cleaned_value = _sanitize_jsonish(state.get(field_name))
        if _has_serialized_value(cleaned_value):
            snapshot[field_name] = cleaned_value

    recent_file_reads = collect_recent_file_reads(state)
    if recent_file_reads:
        snapshot["recent_file_reads"] = recent_file_reads
        snapshot["context_paths"] = _merge_unique_strings(
            snapshot.get("context_paths"),
            recent_file_reads,
        )

    return snapshot


def collect_recent_file_reads(state: dict[str, Any] | None) -> list[str]:
    if not isinstance(state, dict):
        return []

    collected: list[str] = []
    _extend_unique_strings(collected, _normalize_string_list(state.get("recent_file_reads")))
    _extend_unique_strings(collected, _normalize_string_list(state.get("context_paths")))

    uploaded_files = state.get("uploaded_files")
    if isinstance(uploaded_files, list):
        for item in uploaded_files:
            if not isinstance(item, dict):
                continue
            _extend_unique_strings(
                collected,
                [
                    str(item.get("path", "")).strip(),
                    str(item.get("name", "")).strip(),
                ],
            )

    pending_action = state.get("pending_action")
    if isinstance(pending_action, dict):
        target_scope = pending_action.get("target_scope")
        if isinstance(target_scope, dict):
            for key in FILES_CONTAINER_KEYS:
                _extend_unique_strings(collected, _normalize_string_list(target_scope.get(key)))

    for field_name in ("tool_result", "tool_execution_trace", "pending_action", "execution_contract"):
        _extract_path_strings(state.get(field_name), collected)

    return collected[:MAX_RECENT_FILE_READS]


def extract_runtime_rehydration_state_from_transcript(
    messages: list[Any] | tuple[Any, ...] | None,
    *,
    require_compact_boundary: bool = True,
) -> dict[str, Any]:
    normalized_messages = list(messages or [])
    if not normalized_messages:
        return {}
    if require_compact_boundary and not any(_is_compact_boundary_message(message) for message in normalized_messages):
        return {}

    for message in reversed(normalized_messages):
        metadata = _extract_message_metadata(message)
        raw_snapshot = metadata.get(RUNTIME_REHYDRATION_METADATA_KEY)
        if not isinstance(raw_snapshot, dict):
            continue
        snapshot = build_runtime_rehydration_state(raw_snapshot)
        if snapshot:
            return snapshot
    return {}


def merge_runtime_rehydration_state(
    base_state: dict[str, Any] | None,
    rehydration_state: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(base_state or {})
    normalized_snapshot = build_runtime_rehydration_state(rehydration_state)
    if not normalized_snapshot:
        return merged

    for field_name in STRING_STATE_FIELDS:
        existing = str(merged.get(field_name, "") or "").strip()
        if not existing and field_name in normalized_snapshot:
            merged[field_name] = normalized_snapshot[field_name]

    for field_name in STRING_LIST_STATE_FIELDS:
        merged_items = _merge_unique_strings(
            merged.get(field_name),
            normalized_snapshot.get(field_name),
        )
        if merged_items:
            merged[field_name] = merged_items

    for field_name in JSON_STATE_FIELDS:
        if field_name not in normalized_snapshot:
            continue
        existing = merged.get(field_name)
        if _has_runtime_value(existing):
            continue
        merged[field_name] = deepcopy(normalized_snapshot[field_name])

    return merged


def _normalize_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []
    if not isinstance(value, (list, tuple)):
        return []

    normalized: list[str] = []
    for item in value:
        cleaned = str(item or "").strip()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _sanitize_jsonish(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return deepcopy(value)
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_item = _sanitize_jsonish(item)
            if normalized_item is None:
                continue
            normalized[str(key)] = normalized_item
        return normalized or None
    if isinstance(value, (list, tuple)):
        normalized_list = []
        for item in value:
            normalized_item = _sanitize_jsonish(item)
            if normalized_item is None:
                continue
            normalized_list.append(normalized_item)
        return normalized_list or None
    return str(value)


def _extract_path_strings(
    value: Any,
    collected: list[str],
    *,
    allow_plain_string: bool = False,
) -> None:
    if len(collected) >= MAX_RECENT_FILE_READS:
        return
    if isinstance(value, str):
        cleaned = value.strip()
        if allow_plain_string and cleaned and cleaned not in collected:
            collected.append(cleaned)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key or "").strip()
            if normalized_key in PATH_LIKE_KEYS:
                _extract_path_strings(item, collected, allow_plain_string=True)
                continue
            if normalized_key in FILES_CONTAINER_KEYS:
                _extract_path_strings(item, collected, allow_plain_string=True)
                continue
            _extract_path_strings(item, collected, allow_plain_string=False)
            if len(collected) >= MAX_RECENT_FILE_READS:
                return
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _extract_path_strings(item, collected, allow_plain_string=allow_plain_string)
            if len(collected) >= MAX_RECENT_FILE_READS:
                return


def _extract_message_metadata(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        metadata = message.get("metadata")
        return metadata if isinstance(metadata, dict) else {}
    metadata = getattr(message, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _is_compact_boundary_message(message: Any) -> bool:
    if isinstance(message, dict):
        message_type = str(message.get("type", "")).strip()
        return message_type == TRANSCRIPT_TYPE_COMPACT_BOUNDARY

    message_type = str(getattr(message, "type", "")).strip().lower()
    if message_type != "system":
        return False
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    if not isinstance(additional_kwargs, dict):
        return False
    return str(additional_kwargs.get("transcript_type", "")).strip() == TRANSCRIPT_TYPE_COMPACT_BOUNDARY


def _merge_unique_strings(existing: object, incoming: object) -> list[str]:
    merged: list[str] = []
    _extend_unique_strings(merged, _normalize_string_list(existing))
    _extend_unique_strings(merged, _normalize_string_list(incoming))
    return merged


def _extend_unique_strings(target: list[str], values: list[str]) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)


def _has_serialized_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True


def _has_runtime_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return True
