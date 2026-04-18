from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from app.memory.file_transaction import RootFileTransaction
from app.memory.long_term import (
    LongTermMemoryFormatError,
    LongTermMemoryIndexEntry,
    LongTermMemoryWrite,
    _load_existing_index_file_or_default,
    _resolve_indexed_topic_path,
    build_long_term_memory_topic_relative_path,
    load_long_term_memory_catalog,
    load_long_term_memory_file,
    normalize_long_term_memory_id,
    parse_long_term_memory_index,
    render_long_term_memory_index_document,
    render_long_term_memory_topic,
)
from app.memory.observability import emit_memory_telemetry
from app.memory.types import (
    LongTermMemoryFile,
    LongTermMemorySnapshot,
    LongTermMemorySnapshotApplySummary,
    LongTermMemorySnapshotChoice,
    LongTermMemorySnapshotSyncState,
)

DEFAULT_MEMORY_SNAPSHOT_ID = "default"
LONG_TERM_MEMORY_SNAPSHOTS_DIRNAME = "snapshots"
LONG_TERM_MEMORY_SNAPSHOT_SYNC_FILENAME = ".snapshot_sync.json"
_LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")
_WHITESPACE_PATTERN = re.compile(r"\s+")

logger = logging.getLogger(__name__)


class LongTermMemorySnapshotError(ValueError):
    """Raised when project-provided memory snapshots are missing or invalid."""


def resolve_long_term_memory_snapshots_dir(project_root_dir: str | Path) -> Path:
    return Path(project_root_dir).expanduser().resolve() / LONG_TERM_MEMORY_SNAPSHOTS_DIRNAME


def resolve_long_term_memory_snapshot_dir(project_root_dir: str | Path, snapshot_id: str) -> Path:
    normalized_snapshot_id = _normalize_snapshot_id(snapshot_id)
    return resolve_long_term_memory_snapshots_dir(project_root_dir) / normalized_snapshot_id


def resolve_long_term_memory_snapshot_sync_path(user_root_dir: str | Path) -> Path:
    return Path(user_root_dir).expanduser().resolve() / LONG_TERM_MEMORY_SNAPSHOT_SYNC_FILENAME


def list_long_term_memory_snapshots(project_root_dir: str | Path) -> list[LongTermMemorySnapshot]:
    snapshots_dir = resolve_long_term_memory_snapshots_dir(project_root_dir)
    if not snapshots_dir.is_dir():
        return []

    snapshots: list[LongTermMemorySnapshot] = []
    for child in sorted(snapshots_dir.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or child.name.startswith("."):
            continue
        try:
            snapshots.append(load_long_term_memory_snapshot(project_root_dir, child.name))
        except LongTermMemorySnapshotError:
            continue
    return snapshots


def load_long_term_memory_snapshot(
    project_root_dir: str | Path,
    snapshot_id: str,
) -> LongTermMemorySnapshot:
    normalized_snapshot_id = _normalize_snapshot_id(snapshot_id)
    snapshot_dir = resolve_long_term_memory_snapshot_dir(project_root_dir, normalized_snapshot_id)
    try:
        catalog = load_long_term_memory_catalog(snapshot_dir)
    except LongTermMemoryFormatError as exc:
        raise LongTermMemorySnapshotError(
            f"Invalid long-term memory snapshot `{normalized_snapshot_id}` in {snapshot_dir}: {exc}"
        ) from exc

    return LongTermMemorySnapshot(
        snapshot_id=normalized_snapshot_id,
        root_dir=str(snapshot_dir),
        fingerprint=build_long_term_memory_snapshot_fingerprint(catalog.topic_files),
        memory_count=len(catalog.topic_files),
        memories=catalog.topic_files,
    )


def select_long_term_memory_snapshot(
    project_root_dir: str | Path,
    *,
    snapshot_id: str = "",
) -> LongTermMemorySnapshot | None:
    explicit_snapshot_id = str(snapshot_id or "").strip()
    if explicit_snapshot_id:
        return load_long_term_memory_snapshot(project_root_dir, explicit_snapshot_id)

    snapshots = list_long_term_memory_snapshots(project_root_dir)
    if not snapshots:
        return None

    default_snapshot = next(
        (snapshot for snapshot in snapshots if snapshot.snapshot_id == DEFAULT_MEMORY_SNAPSHOT_ID),
        None,
    )
    if default_snapshot is not None:
        return default_snapshot
    return max(snapshots, key=lambda item: item.snapshot_id)


def load_long_term_memory_snapshot_sync_state(
    user_root_dir: str | Path,
) -> LongTermMemorySnapshotSyncState | None:
    sync_path = resolve_long_term_memory_snapshot_sync_path(user_root_dir)
    if not sync_path.is_file():
        return None

    try:
        payload = json.loads(sync_path.read_text(encoding="utf-8"))
        return LongTermMemorySnapshotSyncState.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError):
        return None


def write_long_term_memory_snapshot_sync_state(
    user_root_dir: str | Path,
    *,
    snapshot: LongTermMemorySnapshot,
    action: LongTermMemorySnapshotChoice,
) -> LongTermMemorySnapshotSyncState:
    resolved_user_root_dir = Path(user_root_dir).expanduser().resolve()
    resolved_user_root_dir.mkdir(parents=True, exist_ok=True)

    sync_state = _build_long_term_memory_snapshot_sync_state(
        snapshot=snapshot,
        action=action,
    )
    sync_path = resolve_long_term_memory_snapshot_sync_path(resolved_user_root_dir)
    sync_path.write_text(_render_long_term_memory_snapshot_sync_state(sync_state), encoding="utf-8")
    return sync_state


def get_pending_long_term_memory_snapshot(
    project_root_dir: str | Path,
    user_root_dir: str | Path,
    *,
    snapshot_id: str = "",
) -> LongTermMemorySnapshot | None:
    snapshot = select_long_term_memory_snapshot(project_root_dir, snapshot_id=snapshot_id)
    if snapshot is None:
        emit_memory_telemetry(
            logger,
            "snapshots.pending",
            status="skip",
            project_root_dir=project_root_dir,
            user_root_dir=user_root_dir,
            reason="no_snapshot",
        )
        return None

    sync_state = load_long_term_memory_snapshot_sync_state(user_root_dir)
    if (
        sync_state is not None
        and sync_state.snapshot_id == snapshot.snapshot_id
        and sync_state.fingerprint == snapshot.fingerprint
    ):
        emit_memory_telemetry(
            logger,
            "snapshots.pending",
            status="skip",
            project_root_dir=project_root_dir,
            user_root_dir=user_root_dir,
            snapshot_id=snapshot.snapshot_id,
            reason="already_synced",
        )
        return None
    emit_memory_telemetry(
        logger,
        "snapshots.pending",
        project_root_dir=project_root_dir,
        user_root_dir=user_root_dir,
        snapshot_id=snapshot.snapshot_id,
        memory_count=snapshot.memory_count,
    )
    return snapshot


def apply_long_term_memory_snapshot(
    user_root_dir: str | Path,
    project_root_dir: str | Path,
    *,
    action: LongTermMemorySnapshotChoice,
    snapshot_id: str = "",
) -> LongTermMemorySnapshotApplySummary:
    normalized_action = normalize_long_term_memory_snapshot_choice(action)
    action_map = {
        "keep": keep_long_term_memory_snapshot,
        "merge": merge_long_term_memory_snapshot,
        "replace": replace_long_term_memory_snapshot,
    }
    return action_map[normalized_action](
        user_root_dir,
        project_root_dir,
        snapshot_id=snapshot_id,
    )


def keep_long_term_memory_snapshot(
    user_root_dir: str | Path,
    project_root_dir: str | Path,
    *,
    snapshot_id: str = "",
) -> LongTermMemorySnapshotApplySummary:
    try:
        resolved_user_root_dir, resolved_project_root_dir, snapshot = _resolve_snapshot_apply_inputs(
            user_root_dir,
            project_root_dir,
            snapshot_id=snapshot_id,
        )
        write_long_term_memory_snapshot_sync_state(
            resolved_user_root_dir,
            snapshot=snapshot,
            action="keep",
        )
        summary = LongTermMemorySnapshotApplySummary(
            snapshot_id=snapshot.snapshot_id,
            fingerprint=snapshot.fingerprint,
            action="keep",
            user_root_dir=str(resolved_user_root_dir),
            project_root_dir=str(resolved_project_root_dir),
        )
    except Exception as exc:
        _emit_snapshot_apply_error(
            action="keep",
            user_root_dir=Path(user_root_dir).expanduser().resolve(),
            project_root_dir=Path(project_root_dir).expanduser().resolve(),
            snapshot_id=str(snapshot_id or "").strip(),
            error=exc,
        )
        raise

    _emit_snapshot_apply_success(summary)
    return summary


def merge_long_term_memory_snapshot(
    user_root_dir: str | Path,
    project_root_dir: str | Path,
    *,
    snapshot_id: str = "",
) -> LongTermMemorySnapshotApplySummary:
    created_memory_ids: list[str] = []
    updated_memory_ids: list[str] = []
    try:
        resolved_user_root_dir, resolved_project_root_dir, snapshot = _resolve_snapshot_apply_inputs(
            user_root_dir,
            project_root_dir,
            snapshot_id=snapshot_id,
        )
        existing_index_file = _load_existing_index_file_or_default(resolved_user_root_dir)
        existing_entries = {
            entry.memory_id: entry
            for entry in parse_long_term_memory_index(existing_index_file.content_markdown)
        }
        desired_entry_map = dict(existing_entries)
        transaction = RootFileTransaction(resolved_user_root_dir)
        for snapshot_memory in snapshot.memories:
            existing_entry = existing_entries.get(snapshot_memory.memory_id)
            existing_memory = (
                load_long_term_memory_file(
                    _resolve_indexed_topic_path(resolved_user_root_dir, existing_entry.relative_path),
                    root_dir=resolved_user_root_dir,
                )
                if existing_entry is not None
                else None
            )
            if existing_memory is None:
                merged_payload = _build_snapshot_memory_write(snapshot_memory)
                created_memory_ids.append(snapshot_memory.memory_id)
            else:
                merged_payload = _build_merged_snapshot_memory(existing_memory, snapshot_memory)
                if (
                    existing_memory.name == merged_payload["name"]
                    and existing_memory.description == merged_payload["description"]
                    and existing_memory.memory_type == merged_payload["memory_type"]
                    and existing_memory.content_markdown == merged_payload["content_markdown"]
                ):
                    continue
                updated_memory_ids.append(snapshot_memory.memory_id)

            memory_write = LongTermMemoryWrite.model_validate(merged_payload)
            normalized_memory_id = normalize_long_term_memory_id(memory_write.memory_id)
            topic_relative_path = build_long_term_memory_topic_relative_path(normalized_memory_id)
            transaction.write_text(
                topic_relative_path,
                render_long_term_memory_topic(
                    memory_write,
                    relative_path=topic_relative_path.as_posix(),
                ),
            )
            desired_entry_map[normalized_memory_id] = LongTermMemoryIndexEntry(
                memory_id=normalized_memory_id,
                relative_path=topic_relative_path.as_posix(),
                name=memory_write.name,
                description=memory_write.description,
                memory_type=memory_write.memory_type,
            )

        sync_state = _build_long_term_memory_snapshot_sync_state(
            snapshot=snapshot,
            action="merge",
        )
        transaction.write_text(
            LONG_TERM_MEMORY_SNAPSHOT_SYNC_FILENAME,
            _render_long_term_memory_snapshot_sync_state(sync_state),
        )
        transaction.write_text(
            "MEMORY.md",
            render_long_term_memory_index_document(
                existing_index_file,
                sorted(desired_entry_map.values(), key=lambda item: item.memory_id),
            ),
        )
        transaction.commit()
        summary = LongTermMemorySnapshotApplySummary(
            snapshot_id=snapshot.snapshot_id,
            fingerprint=snapshot.fingerprint,
            action="merge",
            user_root_dir=str(resolved_user_root_dir),
            project_root_dir=str(resolved_project_root_dir),
            created_memory_ids=created_memory_ids,
            updated_memory_ids=updated_memory_ids,
        )
    except Exception as exc:
        _emit_snapshot_apply_error(
            action="merge",
            user_root_dir=Path(user_root_dir).expanduser().resolve(),
            project_root_dir=Path(project_root_dir).expanduser().resolve(),
            snapshot_id=str(snapshot_id or "").strip(),
            error=exc,
        )
        raise

    _emit_snapshot_apply_success(summary)
    return summary


def replace_long_term_memory_snapshot(
    user_root_dir: str | Path,
    project_root_dir: str | Path,
    *,
    snapshot_id: str = "",
) -> LongTermMemorySnapshotApplySummary:
    created_memory_ids: list[str] = []
    deleted_memory_ids: list[str] = []
    try:
        resolved_user_root_dir, resolved_project_root_dir, snapshot = _resolve_snapshot_apply_inputs(
            user_root_dir,
            project_root_dir,
            snapshot_id=snapshot_id,
        )
        existing_index_file = _load_existing_index_file_or_default(resolved_user_root_dir)
        existing_entries = {
            entry.memory_id: entry
            for entry in parse_long_term_memory_index(existing_index_file.content_markdown)
        }
        transaction = RootFileTransaction(resolved_user_root_dir)
        replacement_relative_paths: set[str] = set()
        replacement_entries: list[LongTermMemoryIndexEntry] = []
        for snapshot_memory in snapshot.memories:
            memory_write = LongTermMemoryWrite.model_validate(_build_snapshot_memory_write(snapshot_memory))
            normalized_memory_id = normalize_long_term_memory_id(memory_write.memory_id)
            topic_relative_path = build_long_term_memory_topic_relative_path(normalized_memory_id)
            replacement_relative_paths.add(topic_relative_path.as_posix())
            transaction.write_text(
                topic_relative_path,
                render_long_term_memory_topic(
                    memory_write,
                    relative_path=topic_relative_path.as_posix(),
                ),
            )
            replacement_entries.append(
                LongTermMemoryIndexEntry(
                    memory_id=normalized_memory_id,
                    relative_path=topic_relative_path.as_posix(),
                    name=memory_write.name,
                    description=memory_write.description,
                    memory_type=memory_write.memory_type,
                )
            )
            created_memory_ids.append(snapshot_memory.memory_id)

        for memory_id in sorted(existing_entries):
            deleted_memory_ids.append(memory_id)
            existing_entry = existing_entries[memory_id]
            if existing_entry.relative_path not in replacement_relative_paths:
                transaction.delete(existing_entry.relative_path)

        sync_state = _build_long_term_memory_snapshot_sync_state(
            snapshot=snapshot,
            action="replace",
        )
        transaction.write_text(
            LONG_TERM_MEMORY_SNAPSHOT_SYNC_FILENAME,
            _render_long_term_memory_snapshot_sync_state(sync_state),
        )
        transaction.write_text(
            "MEMORY.md",
            render_long_term_memory_index_document(
                existing_index_file,
                replacement_entries,
            ),
        )
        transaction.commit()
        summary = LongTermMemorySnapshotApplySummary(
            snapshot_id=snapshot.snapshot_id,
            fingerprint=snapshot.fingerprint,
            action="replace",
            user_root_dir=str(resolved_user_root_dir),
            project_root_dir=str(resolved_project_root_dir),
            created_memory_ids=created_memory_ids,
            deleted_memory_ids=deleted_memory_ids,
        )
    except Exception as exc:
        _emit_snapshot_apply_error(
            action="replace",
            user_root_dir=Path(user_root_dir).expanduser().resolve(),
            project_root_dir=Path(project_root_dir).expanduser().resolve(),
            snapshot_id=str(snapshot_id or "").strip(),
            error=exc,
        )
        raise

    _emit_snapshot_apply_success(summary)
    return summary


def normalize_long_term_memory_snapshot_choice(value: LongTermMemorySnapshotChoice | str) -> LongTermMemorySnapshotChoice:
    normalized = str(value or "").strip().lower()
    if normalized not in {"keep", "merge", "replace"}:
        raise LongTermMemorySnapshotError(f"Unsupported snapshot update choice: {value}")
    return normalized  # type: ignore[return-value]


def _resolve_snapshot_apply_inputs(
    user_root_dir: str | Path,
    project_root_dir: str | Path,
    *,
    snapshot_id: str = "",
) -> tuple[Path, Path, LongTermMemorySnapshot]:
    resolved_user_root_dir = Path(user_root_dir).expanduser().resolve()
    resolved_project_root_dir = Path(project_root_dir).expanduser().resolve()
    snapshot = select_long_term_memory_snapshot(project_root_dir, snapshot_id=snapshot_id)
    if snapshot is None:
        raise LongTermMemorySnapshotError("No project-provided memory snapshot is available.")
    resolved_user_root_dir.mkdir(parents=True, exist_ok=True)
    return resolved_user_root_dir, resolved_project_root_dir, snapshot


def _emit_snapshot_apply_success(summary: LongTermMemorySnapshotApplySummary) -> None:
    emit_memory_telemetry(
        logger,
        "snapshots.apply",
        action=summary.action,
        snapshot_id=summary.snapshot_id,
        user_root_dir=summary.user_root_dir,
        project_root_dir=summary.project_root_dir,
        created_count=len(summary.created_memory_ids),
        updated_count=len(summary.updated_memory_ids),
        deleted_count=len(summary.deleted_memory_ids),
    )


def _emit_snapshot_apply_error(
    *,
    action: str,
    user_root_dir: Path,
    project_root_dir: Path,
    snapshot_id: str,
    error: Exception,
) -> None:
    emit_memory_telemetry(
        logger,
        "snapshots.apply",
        status="error",
        action=action,
        user_root_dir=user_root_dir,
        project_root_dir=project_root_dir,
        snapshot_id=snapshot_id,
        error=str(error),
    )


def build_long_term_memory_snapshot_fingerprint(memories: list[LongTermMemoryFile]) -> str:
    hasher = hashlib.sha256()
    for memory in sorted(memories, key=lambda item: item.memory_id):
        payload = "|".join(
            [
                memory.memory_id,
                memory.name,
                memory.description,
                memory.memory_type,
                memory.content_markdown,
            ]
        )
        hasher.update(payload.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _build_snapshot_memory_write(snapshot_memory: LongTermMemoryFile) -> dict[str, str]:
    return {
        "memory_id": snapshot_memory.memory_id,
        "name": snapshot_memory.name,
        "description": snapshot_memory.description,
        "memory_type": snapshot_memory.memory_type,
        "content_markdown": snapshot_memory.content_markdown,
    }


def _build_merged_snapshot_memory(
    existing_memory: LongTermMemoryFile,
    snapshot_memory: LongTermMemoryFile,
) -> dict[str, str]:
    return {
        "memory_id": existing_memory.memory_id,
        "name": existing_memory.name,
        "description": existing_memory.description,
        "memory_type": existing_memory.memory_type,
        "content_markdown": _merge_memory_markdown(
            existing_memory.content_markdown,
            snapshot_memory.content_markdown,
        ),
    }


def _merge_memory_markdown(existing_markdown: str, snapshot_markdown: str) -> str:
    fragments: list[str] = []
    seen: set[str] = set()
    for fragment in _extract_memory_fragments(existing_markdown):
        normalized_fragment = _normalize_text(fragment)
        if not normalized_fragment or normalized_fragment in seen:
            continue
        seen.add(normalized_fragment)
        fragments.append(fragment)
    for fragment in _extract_memory_fragments(snapshot_markdown):
        normalized_fragment = _normalize_text(fragment)
        if not normalized_fragment or normalized_fragment in seen:
            continue
        seen.add(normalized_fragment)
        fragments.append(fragment)

    if not fragments:
        return ""
    if len(fragments) == 1:
        return fragments[0]
    return "\n".join(f"- {fragment}" for fragment in fragments)


def _extract_memory_fragments(content_markdown: str) -> list[str]:
    text = str(content_markdown or "").strip()
    if not text:
        return []

    fragments: list[str] = []
    for raw_line in text.splitlines():
        cleaned_line = _LIST_PREFIX_PATTERN.sub("", raw_line).strip()
        if cleaned_line:
            fragments.append(cleaned_line)
    if fragments:
        return fragments

    paragraphs: list[str] = []
    for block in text.split("\n\n"):
        cleaned_block = block.strip()
        if cleaned_block:
            paragraphs.append(cleaned_block)
    return paragraphs


def _normalize_snapshot_id(value: str) -> str:
    normalized = str(value or "").strip().replace("\\", "/").strip("/")
    if not normalized:
        raise LongTermMemorySnapshotError("Snapshot id must not be empty.")

    parts: list[str] = []
    for part in Path(normalized).parts:
        cleaned_part = str(part or "").strip()
        if cleaned_part in {"", ".", ".."} or cleaned_part.startswith("."):
            raise LongTermMemorySnapshotError(f"Invalid snapshot id: {value}")
        parts.append(cleaned_part)
    return "/".join(parts)


def _build_long_term_memory_snapshot_sync_state(
    *,
    snapshot: LongTermMemorySnapshot,
    action: LongTermMemorySnapshotChoice,
) -> LongTermMemorySnapshotSyncState:
    return LongTermMemorySnapshotSyncState(
        snapshot_id=snapshot.snapshot_id,
        fingerprint=snapshot.fingerprint,
        action=normalize_long_term_memory_snapshot_choice(action),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def _render_long_term_memory_snapshot_sync_state(sync_state: LongTermMemorySnapshotSyncState) -> str:
    return json.dumps(sync_state.model_dump(mode="python"), indent=2, sort_keys=True) + "\n"


def _normalize_text(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", str(value or "").strip()).lower()
