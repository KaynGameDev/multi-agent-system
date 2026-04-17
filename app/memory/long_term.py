from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.frontmatter import normalize_metadata_keys, render_frontmatter_document, split_frontmatter
from app.memory.types import (
    LongTermMemoryCatalog,
    LongTermMemoryFile,
    LongTermMemoryFrontmatter,
    LongTermMemoryIndexEntry,
    LongTermMemoryWrite,
)

LONG_TERM_MEMORY_INDEX_BASENAME = "MEMORY"
LONG_TERM_MEMORY_INDEX_CANDIDATE_NAMES = (
    LONG_TERM_MEMORY_INDEX_BASENAME,
    f"{LONG_TERM_MEMORY_INDEX_BASENAME}.md",
)
LONG_TERM_MEMORY_TOPICS_DIRNAME = "topics"
LONG_TERM_MEMORY_INDEX_SECTION_TITLE = "## Topics"
LONG_TERM_MEMORY_EMPTY_INDEX_SENTINEL = "_No topic memories yet._"
DEFAULT_LONG_TERM_MEMORY_INDEX_NAME = "Memory Index"
DEFAULT_LONG_TERM_MEMORY_INDEX_DESCRIPTION = "Top-level catalog of long-term memory topics."
DEFAULT_LONG_TERM_MEMORY_INDEX_TYPE = "reference"

LONG_TERM_MEMORY_INDEX_ENTRY_PATTERN = re.compile(
    r"^- \[(?P<name>[^\]]+)\]\((?P<relative_path>[^)]+)\)\s+\(`(?P<memory_type>user|feedback|project|reference)`\):\s+(?P<description>.+)$"
)


class LongTermMemoryFormatError(ValueError):
    """Raised when long-term memory files are missing or invalid."""


class FileLongTermMemoryStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()

    def load_catalog(self) -> LongTermMemoryCatalog:
        return load_long_term_memory_catalog(self.root_dir)

    def list(self) -> list[LongTermMemoryIndexEntry]:
        return list_long_term_memories(self.root_dir)

    def get(self, memory_id: str) -> LongTermMemoryFile | None:
        return get_long_term_memory(self.root_dir, memory_id)

    def upsert(self, memory: LongTermMemoryWrite | dict[str, Any]) -> LongTermMemoryFile:
        return upsert_long_term_memory(self.root_dir, memory)

    def delete(self, memory_id: str) -> bool:
        return delete_long_term_memory(self.root_dir, memory_id)


def load_long_term_memory_catalog(root_dir: str | Path) -> LongTermMemoryCatalog:
    resolved_root_dir = Path(root_dir).expanduser().resolve()
    if not resolved_root_dir.exists():
        raise LongTermMemoryFormatError(f"Long-term memory directory does not exist: {resolved_root_dir}")
    if not resolved_root_dir.is_dir():
        raise LongTermMemoryFormatError(f"Long-term memory path must be a directory: {resolved_root_dir}")

    index_path = resolve_long_term_memory_index_file(resolved_root_dir)
    index_file = load_long_term_memory_file(index_path, root_dir=resolved_root_dir)
    index_entries = parse_long_term_memory_index(index_file.content_markdown)

    topic_files: list[LongTermMemoryFile] = []
    seen_memory_ids: set[str] = set()
    seen_relative_paths: set[str] = set()
    for entry in index_entries:
        if entry.memory_id in seen_memory_ids:
            raise LongTermMemoryFormatError(f"Duplicate long-term memory id in index: {entry.memory_id}")
        if entry.relative_path in seen_relative_paths:
            raise LongTermMemoryFormatError(f"Duplicate long-term memory path in index: {entry.relative_path}")
        seen_memory_ids.add(entry.memory_id)
        seen_relative_paths.add(entry.relative_path)

        topic_path = _resolve_indexed_topic_path(resolved_root_dir, entry.relative_path)
        topic_file = load_long_term_memory_file(topic_path, root_dir=resolved_root_dir)
        _validate_index_entry_against_topic_file(entry, topic_file)
        topic_files.append(topic_file)

    return LongTermMemoryCatalog(
        root_dir=str(resolved_root_dir),
        index_file=index_file,
        index_entries=index_entries,
        topic_files=topic_files,
    )


def load_long_term_memory_file(path: str | Path, *, root_dir: str | Path) -> LongTermMemoryFile:
    resolved_path = Path(path).expanduser().resolve()
    resolved_root_dir = Path(root_dir).expanduser().resolve()
    if not resolved_path.exists():
        raise LongTermMemoryFormatError(f"Long-term memory file does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise LongTermMemoryFormatError(f"Long-term memory path must be a file: {resolved_path}")

    try:
        relative_path = resolved_path.relative_to(resolved_root_dir)
    except ValueError as exc:
        raise LongTermMemoryFormatError(
            f"Long-term memory file {resolved_path} must live under root {resolved_root_dir}"
        ) from exc

    raw_text = resolved_path.read_text(encoding="utf-8")
    metadata, body = split_frontmatter(raw_text)
    if not metadata:
        raise LongTermMemoryFormatError(
            f"Long-term memory file {resolved_path} must start with frontmatter containing name, description, and type."
        )

    normalized_metadata = normalize_metadata_keys(metadata)
    try:
        frontmatter = LongTermMemoryFrontmatter.model_validate(normalized_metadata)
    except ValidationError as exc:
        raise LongTermMemoryFormatError(f"Invalid long-term memory frontmatter in {resolved_path}: {exc}") from exc

    return LongTermMemoryFile(
        memory_id=_build_memory_id(relative_path),
        relative_path=relative_path.as_posix(),
        source_path=str(resolved_path),
        name=frontmatter.name,
        description=frontmatter.description,
        memory_type=frontmatter.memory_type,
        content_markdown=body.strip(),
    )


def list_long_term_memories(root_dir: str | Path) -> list[LongTermMemoryIndexEntry]:
    resolved_root_dir = Path(root_dir).expanduser().resolve()
    if not resolved_root_dir.exists():
        return []

    try:
        index_path = resolve_long_term_memory_index_file(resolved_root_dir)
    except LongTermMemoryFormatError as exc:
        if not _is_missing_index_error(exc):
            raise
        return []

    index_file = load_long_term_memory_file(index_path, root_dir=resolved_root_dir)
    return parse_long_term_memory_index(index_file.content_markdown)


def get_long_term_memory(root_dir: str | Path, memory_id: str) -> LongTermMemoryFile | None:
    normalized_memory_id = normalize_long_term_memory_id(memory_id)
    entry_map = {entry.memory_id: entry for entry in list_long_term_memories(root_dir)}
    entry = entry_map.get(normalized_memory_id)
    if entry is None:
        return None
    return load_long_term_memory_file(
        _resolve_indexed_topic_path(root_dir, entry.relative_path),
        root_dir=root_dir,
    )


def upsert_long_term_memory(
    root_dir: str | Path,
    memory: LongTermMemoryWrite | dict[str, Any],
) -> LongTermMemoryFile:
    resolved_root_dir = Path(root_dir).expanduser().resolve()
    resolved_root_dir.mkdir(parents=True, exist_ok=True)

    memory_write = LongTermMemoryWrite.model_validate(memory)
    normalized_memory_id = normalize_long_term_memory_id(memory_write.memory_id)
    topic_relative_path = build_long_term_memory_topic_relative_path(normalized_memory_id)
    topic_path = (resolved_root_dir / topic_relative_path).resolve()
    topic_path.parent.mkdir(parents=True, exist_ok=True)

    topic_document = render_long_term_memory_topic(memory_write, relative_path=topic_relative_path.as_posix())
    topic_path.write_text(topic_document, encoding="utf-8")

    existing_index_file = _load_existing_index_file_or_default(resolved_root_dir)
    entry_map = {entry.memory_id: entry for entry in parse_long_term_memory_index(existing_index_file.content_markdown)}
    entry_map[normalized_memory_id] = LongTermMemoryIndexEntry(
        memory_id=normalized_memory_id,
        relative_path=topic_relative_path.as_posix(),
        name=memory_write.name,
        description=memory_write.description,
        memory_type=memory_write.memory_type,
    )
    _write_index_file(
        resolved_root_dir,
        index_file=existing_index_file,
        index_entries=sorted(entry_map.values(), key=lambda item: item.memory_id),
    )
    return load_long_term_memory_file(topic_path, root_dir=resolved_root_dir)


def delete_long_term_memory(root_dir: str | Path, memory_id: str) -> bool:
    resolved_root_dir = Path(root_dir).expanduser().resolve()
    if not resolved_root_dir.exists():
        return False

    normalized_memory_id = normalize_long_term_memory_id(memory_id)
    try:
        existing_index_file = _load_existing_index_file_or_default(resolved_root_dir, require_existing=True)
    except LongTermMemoryFormatError:
        return False

    entry_map = {entry.memory_id: entry for entry in parse_long_term_memory_index(existing_index_file.content_markdown)}
    entry = entry_map.pop(normalized_memory_id, None)
    if entry is None:
        return False

    topic_path = _resolve_indexed_topic_path(resolved_root_dir, entry.relative_path)
    if topic_path.exists():
        topic_path.unlink()
        _prune_empty_parent_directories(topic_path.parent, stop_at=(resolved_root_dir / LONG_TERM_MEMORY_TOPICS_DIRNAME).resolve())

    _write_index_file(
        resolved_root_dir,
        index_file=existing_index_file,
        index_entries=sorted(entry_map.values(), key=lambda item: item.memory_id),
    )
    return True


def resolve_long_term_memory_index_file(root_dir: str | Path) -> Path:
    resolved_root_dir = Path(root_dir).expanduser().resolve()
    candidates = [
        (resolved_root_dir / candidate_name).resolve()
        for candidate_name in LONG_TERM_MEMORY_INDEX_CANDIDATE_NAMES
        if (resolved_root_dir / candidate_name).is_file()
    ]
    if not candidates:
        raise LongTermMemoryFormatError(
            f"Long-term memory directory {resolved_root_dir} must contain a {LONG_TERM_MEMORY_INDEX_BASENAME} index file."
        )
    if len(candidates) > 1:
        rendered_candidates = ", ".join(str(candidate) for candidate in candidates)
        raise LongTermMemoryFormatError(
            f"Long-term memory directory {resolved_root_dir} has multiple MEMORY index candidates: {rendered_candidates}"
        )
    return candidates[0]


def resolve_long_term_memory_topic_path(root_dir: str | Path, memory_id: str) -> Path:
    resolved_root_dir = Path(root_dir).expanduser().resolve()
    return (resolved_root_dir / build_long_term_memory_topic_relative_path(memory_id)).resolve()


def build_long_term_memory_topic_relative_path(memory_id: str) -> Path:
    normalized_memory_id = normalize_long_term_memory_id(memory_id)
    return Path(LONG_TERM_MEMORY_TOPICS_DIRNAME) / f"{normalized_memory_id}.md"


def normalize_long_term_memory_id(memory_id: str) -> str:
    normalized = str(memory_id or "").strip().replace("\\", "/")
    if normalized.endswith(".md"):
        normalized = normalized[:-3]
    normalized = normalized.strip("/")
    if not normalized:
        raise LongTermMemoryFormatError("Long-term memory id must not be empty.")

    parts: list[str] = []
    for part in Path(normalized).parts:
        cleaned_part = str(part or "").strip()
        if cleaned_part in {"", ".", ".."}:
            raise LongTermMemoryFormatError(f"Invalid long-term memory id: {memory_id}")
        if cleaned_part.startswith("."):
            raise LongTermMemoryFormatError(f"Invalid long-term memory id: {memory_id}")
        parts.append(cleaned_part)
    return "/".join(parts)


def parse_long_term_memory_index(body: str) -> list[LongTermMemoryIndexEntry]:
    normalized_body = str(body or "").strip()
    if not normalized_body:
        return []

    entries: list[LongTermMemoryIndexEntry] = []
    for raw_line in normalized_body.splitlines():
        line = raw_line.strip()
        if not line or line == LONG_TERM_MEMORY_INDEX_SECTION_TITLE or line == LONG_TERM_MEMORY_EMPTY_INDEX_SENTINEL:
            continue
        if not line.startswith("- "):
            continue

        match = LONG_TERM_MEMORY_INDEX_ENTRY_PATTERN.match(line)
        if not match:
            raise LongTermMemoryFormatError(f"Invalid long-term memory index entry: {line}")

        relative_path = str(match.group("relative_path") or "").strip()
        entry = LongTermMemoryIndexEntry(
            memory_id=_build_memory_id(Path(relative_path)),
            relative_path=relative_path,
            name=str(match.group("name") or "").strip(),
            description=str(match.group("description") or "").strip(),
            memory_type=str(match.group("memory_type") or "").strip(),
        )
        _validate_topic_relative_path(entry.relative_path)
        entries.append(entry)
    return entries


def format_long_term_memory_index(entries: list[LongTermMemoryIndexEntry]) -> str:
    normalized_entries = sorted(entries, key=lambda item: item.memory_id)
    lines = [LONG_TERM_MEMORY_INDEX_SECTION_TITLE, ""]
    if not normalized_entries:
        lines.append(LONG_TERM_MEMORY_EMPTY_INDEX_SENTINEL)
        return "\n".join(lines)

    for entry in normalized_entries:
        lines.append(
            f"- [{entry.name}]({entry.relative_path}) (`{entry.memory_type}`): {entry.description}"
        )
    return "\n".join(lines)


def render_long_term_memory_topic(
    memory: LongTermMemoryWrite | dict[str, Any],
    *,
    relative_path: str = "",
) -> str:
    memory_write = LongTermMemoryWrite.model_validate(memory)
    metadata = {
        "name": memory_write.name,
        "description": memory_write.description,
        "type": memory_write.memory_type,
    }
    document = render_frontmatter_document(metadata, memory_write.content_markdown)
    if relative_path:
        return document
    return document


def _load_existing_index_file_or_default(
    root_dir: str | Path,
    *,
    require_existing: bool = False,
) -> LongTermMemoryFile:
    resolved_root_dir = Path(root_dir).expanduser().resolve()
    try:
        index_path = resolve_long_term_memory_index_file(resolved_root_dir)
    except LongTermMemoryFormatError as exc:
        if require_existing or not _is_missing_index_error(exc):
            raise
        return LongTermMemoryFile(
            memory_id=LONG_TERM_MEMORY_INDEX_BASENAME,
            relative_path=f"{LONG_TERM_MEMORY_INDEX_BASENAME}.md",
            source_path=str((resolved_root_dir / f"{LONG_TERM_MEMORY_INDEX_BASENAME}.md").resolve()),
            name=DEFAULT_LONG_TERM_MEMORY_INDEX_NAME,
            description=DEFAULT_LONG_TERM_MEMORY_INDEX_DESCRIPTION,
            memory_type=DEFAULT_LONG_TERM_MEMORY_INDEX_TYPE,
            content_markdown=format_long_term_memory_index([]),
        )
    return load_long_term_memory_file(index_path, root_dir=resolved_root_dir)


def _write_index_file(
    root_dir: str | Path,
    *,
    index_file: LongTermMemoryFile,
    index_entries: list[LongTermMemoryIndexEntry],
) -> LongTermMemoryFile:
    resolved_root_dir = Path(root_dir).expanduser().resolve()
    index_path = (resolved_root_dir / f"{LONG_TERM_MEMORY_INDEX_BASENAME}.md").resolve()
    index_document = render_frontmatter_document(
        {
            "name": index_file.name,
            "description": index_file.description,
            "type": index_file.memory_type,
        },
        format_long_term_memory_index(index_entries),
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(index_document, encoding="utf-8")
    return load_long_term_memory_file(index_path, root_dir=resolved_root_dir)


def _resolve_indexed_topic_path(root_dir: str | Path, relative_path: str) -> Path:
    resolved_root_dir = Path(root_dir).expanduser().resolve()
    _validate_topic_relative_path(relative_path)
    resolved_path = (resolved_root_dir / relative_path).resolve()
    try:
        resolved_path.relative_to(resolved_root_dir)
    except ValueError as exc:
        raise LongTermMemoryFormatError(
            f"Indexed long-term memory path must stay under root {resolved_root_dir}: {relative_path}"
        ) from exc
    return resolved_path


def _validate_index_entry_against_topic_file(
    entry: LongTermMemoryIndexEntry,
    topic_file: LongTermMemoryFile,
) -> None:
    if topic_file.relative_path != entry.relative_path:
        raise LongTermMemoryFormatError(
            f"Long-term memory index path mismatch for {entry.memory_id}: {entry.relative_path} != {topic_file.relative_path}"
        )
    if topic_file.memory_id != entry.memory_id:
        raise LongTermMemoryFormatError(
            f"Long-term memory index id mismatch for {entry.memory_id}: {entry.memory_id} != {topic_file.memory_id}"
        )
    if topic_file.name != entry.name:
        raise LongTermMemoryFormatError(
            f"Long-term memory index name mismatch for {entry.memory_id}: {entry.name} != {topic_file.name}"
        )
    if topic_file.description != entry.description:
        raise LongTermMemoryFormatError(
            f"Long-term memory index description mismatch for {entry.memory_id}: {entry.description} != {topic_file.description}"
        )
    if topic_file.memory_type != entry.memory_type:
        raise LongTermMemoryFormatError(
            f"Long-term memory index type mismatch for {entry.memory_id}: {entry.memory_type} != {topic_file.memory_type}"
        )


def _validate_topic_relative_path(relative_path: str) -> None:
    normalized_relative_path = str(relative_path or "").strip()
    if not normalized_relative_path:
        raise LongTermMemoryFormatError("Long-term memory topic path must not be empty.")

    path = Path(normalized_relative_path)
    if path.suffix.lower() != ".md":
        raise LongTermMemoryFormatError(f"Long-term memory topic path must end in .md: {relative_path}")
    if not path.parts or path.parts[0] != LONG_TERM_MEMORY_TOPICS_DIRNAME:
        raise LongTermMemoryFormatError(
            f"Long-term memory topic path must live under {LONG_TERM_MEMORY_TOPICS_DIRNAME}/: {relative_path}"
        )
    for part in path.parts:
        if part in {"", ".", ".."} or part.startswith("."):
            raise LongTermMemoryFormatError(f"Invalid long-term memory topic path: {relative_path}")


def _build_memory_id(relative_path: Path) -> str:
    normalized_relative_path = relative_path.as_posix().strip()
    if not normalized_relative_path:
        return ""
    path = relative_path
    if path.parts and path.parts[0] == LONG_TERM_MEMORY_TOPICS_DIRNAME:
        path = Path(*path.parts[1:])
    if path.suffix:
        path = path.with_suffix("")
    return path.as_posix().strip()


def _prune_empty_parent_directories(start_dir: Path, *, stop_at: Path) -> None:
    current = start_dir.resolve()
    resolved_stop_at = stop_at.resolve()
    while current != resolved_stop_at and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _is_missing_index_error(exc: LongTermMemoryFormatError) -> bool:
    return "must contain a MEMORY index file" in str(exc)
