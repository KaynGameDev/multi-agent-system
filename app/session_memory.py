from __future__ import annotations

from copy import deepcopy
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Any

from app.context_window import token_count_with_estimation
from app.memory.session_files import delete_session_memory_file, update_session_memory_file

logger = logging.getLogger(__name__)

SESSION_MEMORY_SCHEMA_VERSION = 1
DEFAULT_SESSION_MEMORY_ENABLED = True
DEFAULT_SESSION_MEMORY_INITIALIZE_THRESHOLD_TOKENS = 2_048
DEFAULT_SESSION_MEMORY_UPDATE_GROWTH_THRESHOLD_TOKENS = 768
TRANSCRIPT_TYPE_MESSAGE = "message"
TRANSCRIPT_TYPE_COMPACT_BOUNDARY = "compact_boundary"


@dataclass(frozen=True)
class SessionMemoryRecord:
    thread_id: str
    updated_at: str
    last_message_id: str
    last_message_created_at: str
    covered_message_count: int
    covered_tokens: int
    summary_markdown: str
    source: str = "initialize"


@dataclass(frozen=True)
class SessionMemoryCompactionPlan:
    record: SessionMemoryRecord
    compacted_source_count: int
    preserved_tail_messages: list[dict[str, Any]]


class SessionMemoryStore:
    def __init__(self, storage_path: str | Path) -> None:
        self._storage_path = Path(storage_path).expanduser().resolve()
        self._session_files_root_dir = self._storage_path.parent / "sessions"
        self._records: dict[str, SessionMemoryRecord] = {}
        self._lock = RLock()
        self._load()

    def get(self, thread_id: str) -> SessionMemoryRecord | None:
        normalized_thread_id = str(thread_id or "").strip()
        if not normalized_thread_id:
            return None
        with self._lock:
            record = self._records.get(normalized_thread_id)
            return deepcopy(record) if record is not None else None

    def upsert(self, record: SessionMemoryRecord | dict[str, Any]) -> SessionMemoryRecord:
        normalized_record = normalize_session_memory_record(record)
        if normalized_record is None:
            raise ValueError("Session memory record must include a thread_id and summary_markdown.")
        with self._lock:
            self._records[normalized_record.thread_id] = normalized_record
            self._persist_locked()
            self._sync_session_file_locked(normalized_record)
            return deepcopy(normalized_record)

    def delete(self, thread_id: str) -> None:
        normalized_thread_id = str(thread_id or "").strip()
        if not normalized_thread_id:
            return
        with self._lock:
            if normalized_thread_id not in self._records:
                return
            self._records.pop(normalized_thread_id, None)
            self._persist_locked()
            self._delete_session_file_locked(normalized_thread_id)

    def _load(self) -> None:
        with self._lock:
            self._records = {}
            if not self._storage_path.exists():
                return

            try:
                raw_payload = json.loads(self._storage_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Failed to load session memory from %s", self._storage_path, exc_info=True)
                return

            if not isinstance(raw_payload, dict):
                return

            raw_records = raw_payload.get("records", [])
            if not isinstance(raw_records, list):
                return

            for item in raw_records:
                normalized_record = normalize_session_memory_record(item)
                if normalized_record is None:
                    continue
                self._records[normalized_record.thread_id] = normalized_record

    def _persist_locked(self) -> None:
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SESSION_MEMORY_SCHEMA_VERSION,
            "records": [asdict(record) for record in sorted(self._records.values(), key=lambda item: item.thread_id)],
        }
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(self._storage_path.parent),
            delete=False,
        ) as temp_file:
            json.dump(payload, temp_file, ensure_ascii=False, indent=2)
            temp_file.write("\n")
            temp_path = Path(temp_file.name)
        temp_path.replace(self._storage_path)

    def _sync_session_file_locked(self, record: SessionMemoryRecord) -> None:
        try:
            update_session_memory_file(
                self._session_files_root_dir,
                record.thread_id,
                {"current_state": record.summary_markdown},
            )
        except Exception:
            logger.warning(
                "Failed to sync session memory file for thread %s under %s",
                record.thread_id,
                self._session_files_root_dir,
                exc_info=True,
            )

    def _delete_session_file_locked(self, thread_id: str) -> None:
        try:
            delete_session_memory_file(self._session_files_root_dir, thread_id)
        except Exception:
            logger.warning(
                "Failed to delete session memory file for thread %s under %s",
                thread_id,
                self._session_files_root_dir,
                exc_info=True,
            )


def normalize_session_memory_record(value: SessionMemoryRecord | dict[str, Any] | None) -> SessionMemoryRecord | None:
    if isinstance(value, SessionMemoryRecord):
        return value
    if not isinstance(value, dict):
        return None

    thread_id = str(value.get("thread_id", "") or "").strip()
    summary_markdown = str(value.get("summary_markdown", "") or "").strip()
    if not thread_id or not summary_markdown:
        return None

    return SessionMemoryRecord(
        thread_id=thread_id,
        updated_at=str(value.get("updated_at", "") or "").strip() or utc_now_iso(),
        last_message_id=str(value.get("last_message_id", "") or "").strip(),
        last_message_created_at=str(value.get("last_message_created_at", "") or "").strip(),
        covered_message_count=max(int(value.get("covered_message_count", 0) or 0), 0),
        covered_tokens=max(int(value.get("covered_tokens", 0) or 0), 0),
        summary_markdown=summary_markdown,
        source=str(value.get("source", "") or "").strip() or "initialize",
    )


def is_safe_session_memory_extraction_point(messages: list[dict[str, Any]] | list[Any] | None) -> bool:
    active_slice = _project_active_slice(messages)
    if not active_slice:
        return False

    for message in reversed(active_slice):
        if message["type"] != TRANSCRIPT_TYPE_MESSAGE:
            continue
        if message["role"] not in {"user", "assistant"}:
            continue
        return message["role"] == "assistant"
    return False


def should_initialize_session_memory(
    messages: list[dict[str, Any]] | list[Any] | None,
    *,
    initialize_threshold_tokens: int = DEFAULT_SESSION_MEMORY_INITIALIZE_THRESHOLD_TOKENS,
) -> bool:
    active_slice = _project_active_slice(messages)
    if not active_slice or not is_safe_session_memory_extraction_point(active_slice):
        return False
    active_tokens = token_count_with_estimation(active_slice).estimated_total_tokens
    return active_tokens >= max(int(initialize_threshold_tokens or 0), 0)


def should_update_session_memory(
    messages: list[dict[str, Any]] | list[Any] | None,
    session_memory: SessionMemoryRecord | dict[str, Any] | None,
    *,
    update_growth_threshold_tokens: int = DEFAULT_SESSION_MEMORY_UPDATE_GROWTH_THRESHOLD_TOKENS,
) -> bool:
    existing_record = normalize_session_memory_record(session_memory)
    active_slice = _project_active_slice(messages)
    if existing_record is None or not active_slice or not is_safe_session_memory_extraction_point(active_slice):
        return False

    last_message_index = _find_message_index(active_slice, existing_record.last_message_id)
    if last_message_index < 0:
        return False

    growth_messages = active_slice[last_message_index + 1 :]
    if not growth_messages:
        return False

    growth_tokens = token_count_with_estimation(growth_messages).estimated_total_tokens
    return growth_tokens >= max(int(update_growth_threshold_tokens or 0), 0)


def build_session_memory_record(
    thread_id: str,
    messages: list[dict[str, Any]] | list[Any] | None,
    *,
    session_memory: SessionMemoryRecord | dict[str, Any] | None = None,
    initialize_threshold_tokens: int = DEFAULT_SESSION_MEMORY_INITIALIZE_THRESHOLD_TOKENS,
    update_growth_threshold_tokens: int = DEFAULT_SESSION_MEMORY_UPDATE_GROWTH_THRESHOLD_TOKENS,
    llm: Any | None = None,
) -> SessionMemoryRecord | None:
    normalized_thread_id = str(thread_id or "").strip()
    if not normalized_thread_id:
        return None

    active_slice = _project_active_slice(messages)
    if not active_slice or not is_safe_session_memory_extraction_point(active_slice):
        return None

    existing_record = normalize_session_memory_record(session_memory)
    source = "initialize"
    if existing_record is None:
        if not should_initialize_session_memory(
            active_slice,
            initialize_threshold_tokens=initialize_threshold_tokens,
        ):
            return None
    else:
        last_message_index = _find_message_index(active_slice, existing_record.last_message_id)
        if last_message_index >= 0:
            if not should_update_session_memory(
                active_slice,
                existing_record,
                update_growth_threshold_tokens=update_growth_threshold_tokens,
            ):
                return None
            source = "update"
        else:
            if not should_initialize_session_memory(
                active_slice,
                initialize_threshold_tokens=initialize_threshold_tokens,
            ):
                return None
            source = "reinitialize"

    from app.compaction import compact_conversation

    bundle = compact_conversation(
        list(messages or []),
        llm=llm,
        trigger="session_memory",
        preserved_tail_count=0,
        session_memory=None,
    )
    summary_markdown = str(bundle.summary_message.get("markdown", "") or "").strip()
    if not summary_markdown:
        return None

    active_tokens = token_count_with_estimation(active_slice).estimated_total_tokens
    last_message = active_slice[-1]
    return SessionMemoryRecord(
        thread_id=normalized_thread_id,
        updated_at=utc_now_iso(),
        last_message_id=last_message["id"],
        last_message_created_at=last_message["created_at"],
        covered_message_count=len(active_slice),
        covered_tokens=active_tokens,
        summary_markdown=summary_markdown,
        source=source,
    )


def build_session_memory_compaction_plan(
    messages: list[dict[str, Any]] | list[Any] | None,
    session_memory: SessionMemoryRecord | dict[str, Any] | None,
    *,
    preserved_tail_count: int,
) -> SessionMemoryCompactionPlan | None:
    existing_record = normalize_session_memory_record(session_memory)
    if existing_record is None:
        return None

    active_slice = _project_active_slice(messages)
    if not active_slice:
        return None

    last_message_index = _find_message_index(active_slice, existing_record.last_message_id)
    if last_message_index < 0:
        return None

    preserved_tail_messages = [deepcopy(message) for message in active_slice[last_message_index + 1 :]]
    if len(preserved_tail_messages) > max(int(preserved_tail_count or 0), 0):
        return None

    return SessionMemoryCompactionPlan(
        record=existing_record,
        compacted_source_count=last_message_index + 1,
        preserved_tail_messages=preserved_tail_messages,
    )


def _project_active_slice(messages: list[dict[str, Any]] | list[Any] | None) -> list[dict[str, Any]]:
    projected_messages = _get_messages_after_compact_boundary(messages)
    return [_normalize_transcript_message(message) for message in projected_messages]


def _normalize_transcript_message(message: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return {
            "id": str(message.get("id", "") or "").strip(),
            "role": str(message.get("role", "") or "").strip(),
            "type": str(message.get("type", TRANSCRIPT_TYPE_MESSAGE) or "").strip() or TRANSCRIPT_TYPE_MESSAGE,
            "markdown": str(message.get("markdown", message.get("content", "")) or ""),
            "created_at": str(message.get("created_at", "") or "").strip(),
            "metadata": deepcopy(message.get("metadata")) if isinstance(message.get("metadata"), dict) else None,
        }

    role = str(getattr(message, "type", "") or "").strip().lower()
    normalized_role = role
    if role == "human":
        normalized_role = "user"
    elif role in {"ai", "assistant"}:
        normalized_role = "assistant"
    elif role == "system":
        normalized_role = "system"

    return {
        "id": str(getattr(message, "id", "") or "").strip(),
        "role": normalized_role,
        "type": TRANSCRIPT_TYPE_MESSAGE,
        "markdown": str(getattr(message, "content", "") or ""),
        "created_at": "",
        "metadata": None,
    }


def _find_message_index(messages: list[dict[str, Any]], message_id: str) -> int:
    normalized_message_id = str(message_id or "").strip()
    if not normalized_message_id:
        return -1
    for index, message in enumerate(messages):
        if message["id"] == normalized_message_id:
            return index
    return -1


def _get_messages_after_compact_boundary(messages: list[dict[str, Any]] | list[Any] | None) -> list[Any]:
    normalized_messages = list(messages or [])
    last_boundary_index = -1
    for index, message in enumerate(normalized_messages):
        if _is_compact_boundary_message(message):
            last_boundary_index = index
    if last_boundary_index < 0:
        return normalized_messages
    return normalized_messages[last_boundary_index + 1 :]


def _is_compact_boundary_message(message: Any) -> bool:
    if isinstance(message, dict):
        return str(message.get("type", "") or "").strip() == TRANSCRIPT_TYPE_COMPACT_BOUNDARY

    message_type = str(getattr(message, "type", "") or "").strip().lower()
    if message_type != "system":
        return False
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    if not isinstance(additional_kwargs, dict):
        return False
    return str(additional_kwargs.get("transcript_type", "") or "").strip() == TRANSCRIPT_TYPE_COMPACT_BOUNDARY


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
