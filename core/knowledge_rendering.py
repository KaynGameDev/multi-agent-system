from __future__ import annotations


def render_knowledge_payload(payload: dict) -> str | None:
    if not isinstance(payload, dict):
        return None

    if payload.get("ok") is False:
        return render_knowledge_error(payload)
    if is_read_payload(payload):
        return render_read_payload(payload)
    if is_search_payload(payload):
        return render_search_payload(payload)
    if is_list_payload(payload):
        return render_list_payload(payload)
    return None


def is_knowledge_payload(payload: dict) -> bool:
    return render_knowledge_payload(payload) is not None


def is_list_payload(payload: dict) -> bool:
    return isinstance(payload.get("documents"), list) and "document_count" in payload and "document" not in payload


def is_search_payload(payload: dict) -> bool:
    return isinstance(payload.get("documents"), list) and "query" in payload and "document" not in payload


def is_read_payload(payload: dict) -> bool:
    return isinstance(payload.get("document"), dict) and "content" in payload


def render_list_payload(payload: dict) -> str:
    documents = payload.get("documents") if isinstance(payload.get("documents"), list) else []
    document_count = payload.get("document_count")
    if not documents:
        if isinstance(document_count, int):
            return f"Documents ({document_count})"
        return "Documents (0)"

    count = document_count if isinstance(document_count, int) else len(documents)
    lines = [f"Documents ({count})"]
    for index, document in enumerate(documents, start=1):
        if not isinstance(document, dict):
            continue
        lines.extend(format_document_block(index, document))
    return "\n\n".join(line for line in lines if line).strip()


def render_search_payload(payload: dict) -> str:
    query = first_non_empty(payload.get("query"), "document")
    documents = payload.get("documents") if isinstance(payload.get("documents"), list) else []
    match_count = payload.get("match_count")
    if isinstance(match_count, int):
        header = f'Matches for "{query}" ({match_count})'
    else:
        header = f'Matches for "{query}" ({len(documents)})'

    if not documents:
        return header

    lines = [header]
    for index, document in enumerate(documents, start=1):
        if not isinstance(document, dict):
            continue
        lines.extend(format_document_block(index, document, include_snippet=True))
    return "\n\n".join(line for line in lines if line).strip()


def render_read_payload(payload: dict) -> str:
    document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    title = first_non_empty(document.get("title"), document.get("name"), "Document")
    path = first_non_empty(document.get("path"))
    section_query = first_non_empty(payload.get("section_query"))
    start_line = payload.get("start_line")
    end_line = payload.get("end_line")
    content = first_non_empty(payload.get("content"))

    metadata_parts: list[str] = []
    if path:
        metadata_parts.append(f"Path: {path}")
    if section_query:
        metadata_parts.append(f"Section: {section_query}")
    if isinstance(start_line, int) and isinstance(end_line, int):
        metadata_parts.append(f"Lines: {start_line}-{end_line}")

    lines = [title]
    if metadata_parts:
        lines.append(" | ".join(metadata_parts))
    if content:
        lines.append(f"```text\n{content}\n```")
    else:
        lines.append("No content available.")
    if payload.get("truncated"):
        lines.append("Excerpt truncated.")
    return "\n\n".join(line for line in lines if line).strip()


def render_knowledge_error(payload: dict) -> str:
    error = first_non_empty(payload.get("error"), "I couldn't retrieve that document information.")
    return f"I couldn't retrieve that document information: {error}"


def format_document_block(index: int, document: dict, *, include_snippet: bool = False) -> list[str]:
    title = first_non_empty(document.get("title"), document.get("name"), "Untitled document")
    path = first_non_empty(document.get("path"))
    file_type = first_non_empty(document.get("file_type"))
    section_title = first_non_empty(document.get("section_title"))
    block_type = first_non_empty(document.get("block_type"))
    snippet = compact_snippet(first_non_empty(document.get("snippet")))

    lines = [f"{index}. {title}"]
    metadata = join_parts(
        [
            labeled_value("Path", path),
            labeled_value("Type", file_type),
            labeled_value("Section", section_title),
            labeled_value("Block", block_type),
        ]
    )
    if metadata:
        lines.append(metadata)
    if include_snippet and snippet:
        lines.append(f"Snippet: {snippet}")
    return lines


def compact_snippet(snippet: str) -> str:
    if not snippet:
        return ""
    parts = [part.strip() for part in snippet.splitlines() if part.strip()]
    return " / ".join(parts)


def labeled_value(label: str, value: str) -> str:
    text = first_non_empty(value)
    if not text:
        return ""
    return f"{label}: {text}"


def join_parts(parts: list[str]) -> str:
    return " | ".join(part for part in parts if part)


def first_non_empty(*values) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
