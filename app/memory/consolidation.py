from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from app.memory.long_term import (
    FileLongTermMemoryStore,
    LongTermMemoryFormatError,
    load_long_term_memory_catalog,
)
from app.memory.types import (
    LongTermMemoryCatalog,
    LongTermMemoryConsolidationSummary,
    LongTermMemoryFile,
)

DEFAULT_MEMORY_CONSOLIDATION_MIN_ENTRIES = 8
DEFAULT_MEMORY_CONSOLIDATION_NOISY_PREFIXES = ("daily", "session", "turn")
_DATE_SEGMENT_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_NON_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
_LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+\.\s+)")


class FileLongTermMemoryConsolidator:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()

    def consolidate(self, *, min_entries: int = DEFAULT_MEMORY_CONSOLIDATION_MIN_ENTRIES) -> LongTermMemoryConsolidationSummary:
        return consolidate_long_term_memory(self.root_dir, min_entries=min_entries)


def should_schedule_long_term_memory_consolidation(
    root_dir: str | Path,
    *,
    min_entries: int = DEFAULT_MEMORY_CONSOLIDATION_MIN_ENTRIES,
) -> bool:
    catalog = _load_catalog_or_none(root_dir)
    if catalog is None:
        return False

    topic_files = catalog.topic_files
    if len(topic_files) >= max(int(min_entries or 0), 0):
        return True
    if any(_is_noisy_memory_id(topic_file.memory_id) for topic_file in topic_files):
        return True
    return any(len(group) > 1 for group in _group_by_name_and_type(topic_files).values())


def consolidate_long_term_memory(
    root_dir: str | Path,
    *,
    min_entries: int = DEFAULT_MEMORY_CONSOLIDATION_MIN_ENTRIES,
) -> LongTermMemoryConsolidationSummary:
    resolved_root_dir = Path(root_dir).expanduser().resolve()
    initial_catalog = _load_catalog_or_none(resolved_root_dir)
    if initial_catalog is None:
        return LongTermMemoryConsolidationSummary(root_dir=str(resolved_root_dir))

    initial_topic_files = initial_catalog.topic_files
    if not initial_topic_files:
        return LongTermMemoryConsolidationSummary(root_dir=str(resolved_root_dir))

    store = FileLongTermMemoryStore(resolved_root_dir)
    updated_memory_ids: list[str] = []
    deleted_memory_ids: list[str] = []
    noisy_group_count = 0
    duplicate_group_count = 0
    effective_min_entries = max(int(min_entries or 0), 0)

    if len(initial_topic_files) >= effective_min_entries or any(
        _is_noisy_memory_id(topic_file.memory_id) for topic_file in initial_topic_files
    ):
        for group_key, group_files in _group_by_name_and_type(initial_topic_files).items():
            noisy_members = [topic_file for topic_file in group_files if _is_noisy_memory_id(topic_file.memory_id)]
            if not noisy_members:
                continue

            canonical_memory_id = _resolve_canonical_memory_id(group_files)
            canonical_existing = next(
                (topic_file for topic_file in group_files if topic_file.memory_id == canonical_memory_id),
                None,
            )
            merged_content = _build_consolidated_content(
                [topic_file for topic_file in group_files if topic_file is canonical_existing or topic_file in noisy_members]
            )
            if not merged_content:
                continue

            canonical_name = canonical_existing.name if canonical_existing is not None else noisy_members[0].name
            canonical_description = _choose_canonical_description(group_files)
            if (
                canonical_existing is None
                or canonical_existing.name != canonical_name
                or canonical_existing.description != canonical_description
                or canonical_existing.content_markdown != merged_content
            ):
                store.upsert(
                    {
                        "memory_id": canonical_memory_id,
                        "name": canonical_name,
                        "description": canonical_description,
                        "memory_type": group_files[0].memory_type,
                        "content_markdown": merged_content,
                    }
                )
                updated_memory_ids.append(canonical_memory_id)

            noisy_group_count += 1
            for noisy_member in noisy_members:
                if noisy_member.memory_id == canonical_memory_id:
                    continue
                if store.delete(noisy_member.memory_id):
                    deleted_memory_ids.append(noisy_member.memory_id)

    reloaded_catalog = _load_catalog_or_none(resolved_root_dir)
    if reloaded_catalog is None:
        return LongTermMemoryConsolidationSummary(
            root_dir=str(resolved_root_dir),
            examined_count=len(initial_topic_files),
            updated_memory_ids=updated_memory_ids,
            deleted_memory_ids=deleted_memory_ids,
            noisy_group_count=noisy_group_count,
        )

    duplicate_groups = _group_by_fingerprint(reloaded_catalog.topic_files)
    for group_files in duplicate_groups.values():
        if len(group_files) <= 1:
            continue
        duplicate_group_count += 1
        canonical = _choose_best_canonical_file(group_files)
        for duplicate in group_files:
            if duplicate.memory_id == canonical.memory_id:
                continue
            if store.delete(duplicate.memory_id):
                deleted_memory_ids.append(duplicate.memory_id)

    return LongTermMemoryConsolidationSummary(
        root_dir=str(resolved_root_dir),
        examined_count=len(initial_topic_files),
        updated_memory_ids=updated_memory_ids,
        deleted_memory_ids=deleted_memory_ids,
        noisy_group_count=noisy_group_count,
        duplicate_group_count=duplicate_group_count,
    )


def _load_catalog_or_none(root_dir: str | Path) -> LongTermMemoryCatalog | None:
    try:
        return load_long_term_memory_catalog(root_dir)
    except LongTermMemoryFormatError:
        return None


def _group_by_name_and_type(
    topic_files: list[LongTermMemoryFile],
) -> dict[tuple[str, str], list[LongTermMemoryFile]]:
    grouped: dict[tuple[str, str], list[LongTermMemoryFile]] = defaultdict(list)
    for topic_file in topic_files:
        key = (topic_file.memory_type, _normalize_text(topic_file.name))
        grouped[key].append(topic_file)
    return dict(grouped)


def _group_by_fingerprint(
    topic_files: list[LongTermMemoryFile],
) -> dict[tuple[str, str, str], list[LongTermMemoryFile]]:
    grouped: dict[tuple[str, str, str], list[LongTermMemoryFile]] = defaultdict(list)
    for topic_file in topic_files:
        key = (
            topic_file.memory_type,
            _normalize_text(topic_file.name),
            _normalize_text(topic_file.content_markdown),
        )
        grouped[key].append(topic_file)
    return dict(grouped)


def _resolve_canonical_memory_id(group_files: list[LongTermMemoryFile]) -> str:
    non_noisy_candidates = [topic_file for topic_file in group_files if not _is_noisy_memory_id(topic_file.memory_id)]
    if non_noisy_candidates:
        return _choose_best_canonical_file(non_noisy_candidates).memory_id

    sample = _choose_best_canonical_file(group_files)
    name_slug = _slugify_text(sample.name)
    if not name_slug:
        return sample.memory_id
    return f"{sample.memory_type}/{name_slug}"


def _choose_best_canonical_file(group_files: list[LongTermMemoryFile]) -> LongTermMemoryFile:
    return min(
        group_files,
        key=lambda topic_file: (
            1 if _is_noisy_memory_id(topic_file.memory_id) else 0,
            topic_file.memory_id.count("/"),
            len(topic_file.memory_id),
            topic_file.memory_id,
        ),
    )


def _choose_canonical_description(group_files: list[LongTermMemoryFile]) -> str:
    canonical = _choose_best_canonical_file(group_files)
    if canonical.description:
        return canonical.description

    descriptions = [topic_file.description for topic_file in group_files if topic_file.description]
    if descriptions:
        return min(descriptions, key=lambda item: (len(item), item))
    return f"Consolidated durable memory for {canonical.name.lower()}."


def _build_consolidated_content(group_files: list[LongTermMemoryFile]) -> str:
    fragments: list[str] = []
    seen: set[str] = set()
    for topic_file in group_files:
        for fragment in _extract_content_fragments(topic_file.content_markdown):
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


def _extract_content_fragments(content_markdown: str) -> list[str]:
    text = str(content_markdown or "").strip()
    if not text:
        return []

    raw_lines = []
    for block in text.splitlines():
        cleaned = _LIST_PREFIX_PATTERN.sub("", block).strip()
        if cleaned:
            raw_lines.append(cleaned)
    if raw_lines:
        return raw_lines

    paragraphs = []
    for block in text.split("\n\n"):
        cleaned = block.strip()
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def _is_noisy_memory_id(memory_id: str) -> bool:
    normalized_memory_id = str(memory_id or "").strip().strip("/")
    if not normalized_memory_id:
        return False
    parts = [part for part in normalized_memory_id.split("/") if part]
    if not parts:
        return False
    if parts[0] in DEFAULT_MEMORY_CONSOLIDATION_NOISY_PREFIXES:
        return True
    return any(_DATE_SEGMENT_PATTERN.match(part) is not None for part in parts)


def _normalize_text(value: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", str(value or "").strip().lower())


def _slugify_text(value: str) -> str:
    normalized = _NON_SLUG_PATTERN.sub("-", _normalize_text(value)).strip("-")
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized
