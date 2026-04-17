from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from app.frontmatter import normalize_metadata_keys, split_frontmatter
from app.memory.types import (
    LongTermMemoryCatalog,
    LongTermMemoryFile,
    LongTermMemoryFrontmatter,
)

LONG_TERM_MEMORY_INDEX_BASENAME = "MEMORY"
LONG_TERM_MEMORY_INDEX_CANDIDATE_NAMES = (
    LONG_TERM_MEMORY_INDEX_BASENAME,
    f"{LONG_TERM_MEMORY_INDEX_BASENAME}.md",
)
SUPPORTED_LONG_TERM_MEMORY_TOPIC_SUFFIXES = {".md"}


class LongTermMemoryFormatError(ValueError):
    """Raised when long-term memory files are missing or invalid."""


def load_long_term_memory_catalog(root_dir: str | Path) -> LongTermMemoryCatalog:
    resolved_root_dir = Path(root_dir).expanduser().resolve()
    if not resolved_root_dir.exists():
        raise LongTermMemoryFormatError(f"Long-term memory directory does not exist: {resolved_root_dir}")
    if not resolved_root_dir.is_dir():
        raise LongTermMemoryFormatError(f"Long-term memory path must be a directory: {resolved_root_dir}")

    index_path = resolve_long_term_memory_index_file(resolved_root_dir)
    index_file = load_long_term_memory_file(index_path, root_dir=resolved_root_dir)

    topic_files: list[LongTermMemoryFile] = []
    for path in sorted(resolved_root_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.resolve() == index_path:
            continue
        if _is_hidden_relative_path(path.relative_to(resolved_root_dir)):
            continue
        if path.suffix.lower() not in SUPPORTED_LONG_TERM_MEMORY_TOPIC_SUFFIXES:
            continue
        topic_files.append(load_long_term_memory_file(path, root_dir=resolved_root_dir))

    return LongTermMemoryCatalog(
        root_dir=str(resolved_root_dir),
        index_file=index_file,
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


def _build_memory_id(relative_path: Path) -> str:
    normalized_relative_path = relative_path.as_posix()
    if relative_path.suffix:
        normalized_relative_path = relative_path.with_suffix("").as_posix()
    return normalized_relative_path.strip()


def _is_hidden_relative_path(relative_path: Path) -> bool:
    return any(part.startswith(".") for part in relative_path.parts)
