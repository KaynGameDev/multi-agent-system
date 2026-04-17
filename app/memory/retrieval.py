from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Sequence

from app.memory.long_term import (
    LongTermMemoryFormatError,
    list_long_term_memories,
    load_long_term_memory_file,
)
from app.memory.types import AgentMemoryRetrievalResult, AgentMemoryScope, LongTermMemoryIndexEntry

QUERY_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "for",
    "from",
    "get",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "show",
    "tell",
    "the",
    "to",
    "us",
    "what",
    "with",
}


def retrieve_relevant_long_term_memories(
    root_dir: str | Path,
    *,
    query_text: str,
    memory_scope: AgentMemoryScope,
    scope_key: str,
    top_k: int = 3,
    loaded_paths: Sequence[str] | None = None,
) -> list[AgentMemoryRetrievalResult]:
    normalized_query = str(query_text or "").strip()
    if not normalized_query:
        return []

    effective_limit = max(int(top_k or 0), 0)
    if effective_limit <= 0:
        return []

    try:
        entries = list_long_term_memories(root_dir)
    except LongTermMemoryFormatError:
        return []
    if not entries:
        return []

    resolved_root_dir = Path(root_dir).expanduser().resolve()
    excluded_refs = normalize_loaded_memory_refs(loaded_paths or (), root_dir=resolved_root_dir)
    scored_entries = [
        (entry, score)
        for entry in entries
        if not _is_excluded_entry(entry, excluded_refs, root_dir=resolved_root_dir)
        for score in [_score_index_entry(entry, normalized_query)]
        if score > 0
    ]
    if not scored_entries:
        return []

    ranked_entries = sorted(
        scored_entries,
        key=lambda item: (-item[1], item[0].memory_id),
    )

    results: list[AgentMemoryRetrievalResult] = []
    for entry, score in ranked_entries[:effective_limit]:
        topic_path = _resolve_memory_topic_path(resolved_root_dir, entry.relative_path)
        topic_file = load_long_term_memory_file(topic_path, root_dir=resolved_root_dir)
        results.append(
            AgentMemoryRetrievalResult(
                memory_id=topic_file.memory_id,
                scope=memory_scope,
                scope_key=scope_key,
                relative_path=topic_file.relative_path,
                source_path=topic_file.source_path,
                name=topic_file.name,
                description=topic_file.description,
                memory_type=topic_file.memory_type,
                content_markdown=topic_file.content_markdown,
                score=score,
            )
        )
    return results


def format_retrieved_memories_prompt(results: Sequence[AgentMemoryRetrievalResult]) -> str:
    normalized_results = [result for result in results if isinstance(result, AgentMemoryRetrievalResult)]
    if not normalized_results:
        return ""

    lines = ["# Relevant Memories"]
    for result in normalized_results:
        lines.extend(
            [
                f"## {result.name}",
                f"- Memory id: `{result.memory_id}`",
                f"- Type: `{result.memory_type}`",
                f"- Description: {result.description}",
                f"- Source: `{result.relative_path}`",
            ]
        )
        if result.content_markdown:
            lines.append(result.content_markdown)
    return "\n".join(lines).strip()


def normalize_loaded_memory_refs(
    loaded_paths: Iterable[str],
    *,
    root_dir: str | Path,
) -> set[str]:
    resolved_root_dir = Path(root_dir).expanduser().resolve()
    normalized_refs: set[str] = set()
    for raw_path in loaded_paths:
        cleaned = str(raw_path or "").strip()
        if not cleaned:
            continue

        normalized_text = cleaned.replace("\\", "/").strip()
        normalized_refs.add(normalized_text)
        path = Path(normalized_text)
        if path.suffix.lower() == ".md":
            normalized_refs.add(path.with_suffix("").as_posix())

        try:
            resolved_candidate = Path(cleaned).expanduser().resolve()
        except OSError:
            resolved_candidate = None

        if resolved_candidate is None:
            continue
        normalized_refs.add(str(resolved_candidate))
        try:
            relative_candidate = resolved_candidate.relative_to(resolved_root_dir)
        except ValueError:
            continue

        relative_text = relative_candidate.as_posix()
        normalized_refs.add(relative_text)
        if relative_candidate.suffix.lower() == ".md":
            normalized_refs.add(relative_candidate.with_suffix("").as_posix())

        memory_id = _memory_id_from_relative_path(relative_candidate)
        if memory_id:
            normalized_refs.add(memory_id)
    return normalized_refs


def _score_index_entry(entry: LongTermMemoryIndexEntry, query_text: str) -> float:
    normalized_query = " ".join(_tokenize_query(query_text))
    if not normalized_query:
        return 0.0

    name_text = entry.name.lower()
    description_text = entry.description.lower()
    memory_id_text = entry.memory_id.lower()
    relative_path_text = entry.relative_path.lower()
    memory_type_text = entry.memory_type.lower()
    header_text = " ".join(
        (
            name_text,
            description_text,
            memory_id_text,
            relative_path_text,
            memory_type_text,
        )
    )
    query_tokens = _tokenize_query(query_text)
    score = 0.0
    if normalized_query and normalized_query in header_text:
        score += 6.0

    for token in query_tokens:
        if token in name_text:
            score += 3.0
        if token in description_text:
            score += 2.0
        if token in memory_id_text:
            score += 1.5
        if token in relative_path_text:
            score += 1.0
        if token == memory_type_text:
            score += 1.0
    return score


def _tokenize_query(query_text: str) -> list[str]:
    seen_tokens: set[str] = set()
    tokens: list[str] = []
    for token in QUERY_TOKEN_PATTERN.findall(str(query_text or "").lower()):
        if len(token) <= 1:
            continue
        if len(token) <= 2 and token.isalpha():
            continue
        if token in QUERY_STOPWORDS:
            continue
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        tokens.append(token)
    return tokens


def _is_excluded_entry(
    entry: LongTermMemoryIndexEntry,
    excluded_refs: set[str],
    *,
    root_dir: Path,
) -> bool:
    candidate_path = _resolve_memory_topic_path(root_dir, entry.relative_path)
    comparison_refs = {
        entry.memory_id,
        entry.relative_path,
        str(candidate_path),
    }
    relative_without_suffix = Path(entry.relative_path).with_suffix("").as_posix()
    comparison_refs.add(relative_without_suffix)
    if relative_without_suffix.startswith("topics/"):
        comparison_refs.add(relative_without_suffix[len("topics/") :])
    return any(ref in excluded_refs for ref in comparison_refs)


def _resolve_memory_topic_path(root_dir: Path, relative_path: str) -> Path:
    resolved_path = (root_dir / relative_path).resolve()
    try:
        resolved_path.relative_to(root_dir)
    except ValueError as exc:
        raise LongTermMemoryFormatError(
            f"Indexed long-term memory path must stay under root {root_dir}: {relative_path}"
        ) from exc
    return resolved_path


def _memory_id_from_relative_path(path: Path) -> str:
    normalized = path.as_posix().strip()
    if not normalized:
        return ""
    relative_path = path
    if relative_path.parts and relative_path.parts[0] == "topics":
        relative_path = Path(*relative_path.parts[1:])
    if relative_path.suffix.lower() == ".md":
        relative_path = relative_path.with_suffix("")
    return relative_path.as_posix().strip()
