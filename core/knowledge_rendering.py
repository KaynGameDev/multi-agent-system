from __future__ import annotations


def render_knowledge_payload(payload: dict, *, preferred_language: str = "en") -> str | None:
    if not isinstance(payload, dict):
        return None

    if payload.get("ok") is False:
        return render_knowledge_error(payload, preferred_language=preferred_language)
    if is_read_payload(payload):
        return render_read_payload(payload, preferred_language=preferred_language)
    if is_search_payload(payload):
        return render_search_payload(payload, preferred_language=preferred_language)
    if is_list_payload(payload):
        return render_list_payload(payload, preferred_language=preferred_language)
    return None


def is_knowledge_payload(payload: dict) -> bool:
    return render_knowledge_payload(payload) is not None


def is_list_payload(payload: dict) -> bool:
    return isinstance(payload.get("documents"), list) and "document_count" in payload and "document" not in payload


def is_search_payload(payload: dict) -> bool:
    return isinstance(payload.get("documents"), list) and "query" in payload and "document" not in payload


def is_read_payload(payload: dict) -> bool:
    return isinstance(payload.get("document"), dict) and "content" in payload


def render_list_payload(payload: dict, *, preferred_language: str = "en") -> str:
    documents = payload.get("documents") if isinstance(payload.get("documents"), list) else []
    document_count = payload.get("document_count")
    documents_label = translate("Documents", preferred_language)
    if not documents:
        if isinstance(document_count, int):
            return f"{documents_label} ({document_count})"
        return f"{documents_label} (0)"

    count = document_count if isinstance(document_count, int) else len(documents)
    lines = [f"{documents_label} ({count})"]
    for index, document in enumerate(documents, start=1):
        if not isinstance(document, dict):
            continue
        lines.extend(format_document_block(index, document, preferred_language=preferred_language))
    return "\n\n".join(line for line in lines if line).strip()


def render_search_payload(payload: dict, *, preferred_language: str = "en") -> str:
    query = first_non_empty(payload.get("query"), translate("document", preferred_language))
    documents = payload.get("documents") if isinstance(payload.get("documents"), list) else []
    match_count = payload.get("match_count")
    if isinstance(match_count, int):
        header = translate_with_query("Matches for", query, match_count, preferred_language)
    else:
        header = translate_with_query("Matches for", query, len(documents), preferred_language)

    if not documents:
        return header

    lines = [header]
    for index, document in enumerate(documents, start=1):
        if not isinstance(document, dict):
            continue
        lines.extend(
            format_document_block(
                index,
                document,
                include_snippet=True,
                preferred_language=preferred_language,
            )
        )
    return "\n\n".join(line for line in lines if line).strip()


def render_read_payload(payload: dict, *, preferred_language: str = "en") -> str:
    document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
    title = first_non_empty(document.get("title"), document.get("name"), translate("Document", preferred_language))
    path = first_non_empty(document.get("path"))
    section_query = first_non_empty(payload.get("section_query"))
    start_line = payload.get("start_line")
    end_line = payload.get("end_line")
    content = first_non_empty(payload.get("content"))

    metadata_parts: list[str] = []
    if path:
        metadata_parts.append(f"{translate('Path', preferred_language)}: {path}")
    if section_query:
        metadata_parts.append(f"{translate('Section', preferred_language)}: {section_query}")
    if isinstance(start_line, int) and isinstance(end_line, int):
        metadata_parts.append(f"{translate('Lines', preferred_language)}: {start_line}-{end_line}")

    lines = [title]
    if metadata_parts:
        lines.append(" | ".join(metadata_parts))
    if content:
        lines.append(f"```text\n{content}\n```")
    else:
        lines.append(translate("No content available.", preferred_language))
    if payload.get("truncated"):
        lines.append(translate("Excerpt truncated.", preferred_language))
    return "\n\n".join(line for line in lines if line).strip()


def render_knowledge_error(payload: dict, *, preferred_language: str = "en") -> str:
    fallback = translate("I couldn't retrieve that document information.", preferred_language)
    error = first_non_empty(payload.get("error"))
    if not error:
        return fallback
    prefix = fallback.rstrip(".。")
    separator = "：" if preferred_language == "zh" else ":"
    return f"{prefix}{separator} {error}"


def format_document_block(
    index: int,
    document: dict,
    *,
    include_snippet: bool = False,
    preferred_language: str = "en",
) -> list[str]:
    title = first_non_empty(document.get("title"), document.get("name"), translate("Untitled document", preferred_language))
    path = first_non_empty(document.get("path"))
    file_type = first_non_empty(document.get("file_type"))
    section_title = first_non_empty(document.get("section_title"))
    block_type = first_non_empty(document.get("block_type"))
    snippet = compact_snippet(first_non_empty(document.get("snippet")))

    lines = [f"{index}. {title}"]
    metadata = join_parts(
        [
            labeled_value(translate("Path", preferred_language), path),
            labeled_value(translate("Type", preferred_language), file_type),
            labeled_value(translate("Section", preferred_language), section_title),
            labeled_value(translate("Block", preferred_language), block_type),
        ]
    )
    if metadata:
        lines.append(metadata)
    if include_snippet and snippet:
        lines.append(f"{translate('Snippet', preferred_language)}: {snippet}")
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


def translate(text: str, preferred_language: str) -> str:
    if preferred_language != "zh":
        return text

    translations = {
        "Documents": "文档",
        "document": "文档",
        "Matches for": "搜索结果",
        "Document": "文档",
        "Path": "路径",
        "Type": "类型",
        "Section": "章节",
        "Lines": "行号",
        "No content available.": "没有可用内容。",
        "Excerpt truncated.": "摘录已截断。",
        "I couldn't retrieve that document information.": "我无法获取该文档信息。",
        "Untitled document": "未命名文档",
        "Block": "区块",
        "Snippet": "摘要",
    }
    return translations.get(text, text)


def translate_with_query(prefix: str, query: str, count: int, preferred_language: str) -> str:
    if preferred_language == "zh":
        return f'搜索“{query}”的结果 ({count})'
    return f'{prefix} "{query}" ({count})'
