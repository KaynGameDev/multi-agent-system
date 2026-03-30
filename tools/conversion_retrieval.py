from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tools.document_conversion import (
    ConversionSourceRecord,
    compact_source_bundle_for_extraction,
)
from tools.knowledge_base import extract_document_blocks, extract_document_title, load_knowledge_document

logger = logging.getLogger(__name__)

TABLE_WINDOW_ROWS = 30
TABLE_WINDOW_OVERLAP = 5
PROSE_CHUNK_MAX_CHARS = 3_000
RETRIEVED_SOURCE_BUNDLE_MAX_CHARS = 90_000
MAX_CHUNKS_PER_SHEET = 6
MAX_CHUNKS_PER_SOURCE = 12

QUERY_LIMITS = {
    "target": 4,
    "overview": 4,
    "terminology": 5,
    "entities": 5,
    "rules": 5,
    "config": 2,
    "economy": 2,
    "localization": 2,
    "ui": 2,
    "analytics": 2,
    "qa": 2,
}

QUERY_HINTS = {
    "target": "game market feature package variant target summary title design purpose 游戏 市场 功能 包体 设计目的 活动简介",
    "overview": "overview summary introduction purpose core flow feature description 玩法 简介 设计目的 主要内容 核心流程",
    "terminology": "terminology glossary terms naming wording 术语 名词 命名 文案",
    "entities": "entities entity objects items currencies rewards tasks modules 系统对象 奖励 任务 道具 货币",
    "rules": "rules logic conditions thresholds reset unlock limitations 规则 条件 阈值 重置 解锁 限制",
    "config": "config configuration parameters values server switch tables 配置 参数 数值 开关 配表 服务端",
    "economy": "economy sink source rewards price cost currency economy 经济 消耗 产出 奖励 价格 成本 货币",
    "localization": "localization copy strings text naming wording 本地化 文案 字符串 命名",
    "ui": "ui ux interface screen popup button page layout 界面 页面 弹窗 按钮 布局",
    "analytics": "analytics tracking events telemetry metrics 埋点 事件 统计 指标",
    "qa": "qa test testing edge cases validation 验证 测试 边界 case",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "can",
    "convert",
    "conversion",
    "document",
    "for",
    "from",
    "help",
    "i",
    "into",
    "it",
    "me",
    "need",
    "of",
    "on",
    "please",
    "the",
    "this",
    "to",
    "we",
    "you",
    "your",
    "请",
    "一下",
    "帮我",
    "我们",
    "文档",
    "转换",
    "这个",
}


@dataclass(frozen=True)
class ConversionChunk:
    chunk_id: str
    source_doc_id: str
    source_type: str
    source_name: str
    sheet_name: str
    section_title: str
    row_start: int
    row_end: int
    text: str
    char_count: int


def build_conversion_chunks(sources: Sequence[ConversionSourceRecord]) -> list[ConversionChunk]:
    chunks: list[ConversionChunk] = []
    for source in sources:
        content = load_conversion_source_text(source)
        source_title = extract_document_title(content, default=source.original_name)
        if should_treat_as_google_sheet(source, content):
            chunks.extend(build_google_sheet_chunks(source, source_title, content))
            continue
        chunks.extend(build_structured_document_chunks(source, source_title, content))
    return chunks


def retrieve_conversion_chunks(
    chunks: Sequence[ConversionChunk],
    queries: dict[str, str],
    *,
    max_chunks_per_sheet: int = MAX_CHUNKS_PER_SHEET,
    max_chunks_per_source: int = MAX_CHUNKS_PER_SOURCE,
) -> list[ConversionChunk]:
    summary_chunks = sorted(
        (chunk for chunk in chunks if chunk.section_title == "Source Summary"),
        key=lambda chunk: (chunk.source_name.lower(), chunk.source_doc_id, chunk.chunk_id),
    )
    content_chunks = [chunk for chunk in chunks if chunk.section_title != "Source Summary"]

    selected_by_id = {chunk.chunk_id: chunk for chunk in summary_chunks}
    per_sheet_counts: Counter[tuple[str, str]] = Counter()
    per_source_counts: Counter[str] = Counter()

    for query_name, limit in QUERY_LIMITS.items():
        query_text = str(queries.get(query_name, "")).strip()
        if not query_text:
            continue
        terms = normalize_conversion_query_terms(query_text)
        if not terms and not query_text.strip():
            continue

        ranked = sorted(
            (
                (score_conversion_chunk(chunk, query_text, terms), chunk)
                for chunk in content_chunks
            ),
            key=lambda item: (
                -item[0],
                item[1].source_name.lower(),
                item[1].sheet_name.lower(),
                item[1].row_start,
                item[1].chunk_id,
            ),
        )

        added = 0
        for score, chunk in ranked:
            if score <= 0:
                break
            if chunk.chunk_id in selected_by_id:
                continue
            if per_source_counts[chunk.source_doc_id] >= max_chunks_per_source:
                continue
            if chunk.sheet_name and per_sheet_counts[(chunk.source_doc_id, chunk.sheet_name)] >= max_chunks_per_sheet:
                continue

            selected_by_id[chunk.chunk_id] = chunk
            per_source_counts[chunk.source_doc_id] += 1
            if chunk.sheet_name:
                per_sheet_counts[(chunk.source_doc_id, chunk.sheet_name)] += 1
            added += 1
            if added >= limit:
                break

    selected_chunks = list(selected_by_id.values())
    summary_ids = {chunk.chunk_id for chunk in summary_chunks}
    summary_order = {chunk.chunk_id: index for index, chunk in enumerate(summary_chunks)}
    selected_chunks.sort(
        key=lambda chunk: (
            0 if chunk.chunk_id in summary_ids else 1,
            summary_order.get(chunk.chunk_id, 10_000),
            chunk.source_name.lower(),
            chunk.source_doc_id,
            chunk.sheet_name.lower(),
            chunk.row_start,
            chunk.section_title.lower(),
            chunk.chunk_id,
        )
    )
    return selected_chunks


def build_retrieved_source_bundle(
    sources: Sequence[ConversionSourceRecord],
    *,
    shared_context: str,
    existing_package_context: str,
    answer_history: list[str],
    latest_user_text: str = "",
    game_slug: str = "",
    market_slug: str = "",
    feature_slug: str = "",
    max_chars: int = RETRIEVED_SOURCE_BUNDLE_MAX_CHARS,
) -> str:
    chunks = build_conversion_chunks(sources)
    queries = build_conversion_queries(
        latest_user_text=latest_user_text,
        answer_history=answer_history,
        game_slug=game_slug,
        market_slug=market_slug,
        feature_slug=feature_slug,
    )
    selected_chunks = retrieve_conversion_chunks(chunks, queries)

    parts: list[str] = []
    if shared_context.strip():
        parts.append("## Shared Context\n" + shared_context.strip())
    if existing_package_context.strip():
        parts.append("## Existing Approved Package\n" + existing_package_context.strip())
    if answer_history:
        clarifications = "\n".join(f"- {item}" for item in answer_history if item.strip())
        if clarifications.strip():
            parts.append("## User Clarifications\n" + clarifications)

    summary_chunks = [chunk for chunk in selected_chunks if chunk.section_title == "Source Summary"]
    for chunk in summary_chunks:
        parts.append(
            "\n".join(
                [
                    f"## Source Summary: {chunk.source_name}",
                    f"Source ID: {chunk.source_doc_id}",
                    chunk.text.strip(),
                ]
            ).strip()
        )

    content_chunks = [chunk for chunk in selected_chunks if chunk.section_title != "Source Summary"]
    grouped_chunks: dict[tuple[str, str], list[ConversionChunk]] = {}
    group_order: list[tuple[str, str]] = []
    for chunk in content_chunks:
        key = (chunk.source_doc_id, chunk.sheet_name)
        if key not in grouped_chunks:
            grouped_chunks[key] = []
            group_order.append(key)
        grouped_chunks[key].append(chunk)

    for source_doc_id, sheet_name in group_order:
        group = grouped_chunks[(source_doc_id, sheet_name)]
        group.sort(key=lambda chunk: (chunk.row_start, chunk.row_end, chunk.section_title.lower(), chunk.chunk_id))
        source_name = group[0].source_name
        header_lines = [f"## Retrieved Evidence: {source_name}", f"Source ID: {source_doc_id}"]
        if sheet_name:
            header_lines.append(f"### Sheet: {sheet_name}")
        entries: list[str] = ["\n".join(header_lines)]
        for chunk in group:
            label_parts: list[str] = []
            if chunk.section_title:
                label_parts.append(chunk.section_title)
            if chunk.row_start or chunk.row_end:
                label_parts.append(f"rows {chunk.row_start}-{chunk.row_end}")
            if label_parts:
                entries.append("#### " + " | ".join(label_parts))
            entries.append(chunk.text.strip())
        parts.append("\n\n".join(entry for entry in entries if entry.strip()).strip())

    raw_bundle = "\n\n".join(part for part in parts if part.strip()).strip()
    compacted_bundle = compact_source_bundle_for_extraction(raw_bundle, max_chars=max_chars)
    logger.debug(
        "Built retrieved conversion bundle source_count=%s chunk_count=%s selected_count=%s selected_chunk_ids=%s sheets=%s raw_chars=%s final_chars=%s fallback_compacted=%s",
        len(sources),
        len(chunks),
        len(selected_chunks),
        [chunk.chunk_id for chunk in selected_chunks],
        sorted({chunk.sheet_name for chunk in selected_chunks if chunk.sheet_name}),
        len(raw_bundle),
        len(compacted_bundle),
        compacted_bundle != raw_bundle,
    )
    return compacted_bundle


def build_conversion_queries(
    *,
    latest_user_text: str,
    answer_history: list[str],
    game_slug: str,
    market_slug: str,
    feature_slug: str,
) -> dict[str, str]:
    recent_answers = " ".join(
        sanitize_query_text(item)
        for item in answer_history[-4:]
        if sanitize_query_text(item)
    ).strip()
    latest = sanitize_query_text(latest_user_text)
    identifiers = " ".join(value for value in (game_slug, market_slug, feature_slug) if value).strip()
    shared_context = " ".join(part for part in (latest, recent_answers, identifiers) if part).strip()

    queries: dict[str, str] = {}
    for query_name, hint_text in QUERY_HINTS.items():
        queries[query_name] = " ".join(part for part in (shared_context, hint_text) if part).strip()
    return queries


def load_conversion_source_text(source: ConversionSourceRecord) -> str:
    source_path = Path(source.raw_path)
    try:
        return load_knowledge_document(source_path)
    except Exception:
        return source_path.read_text(encoding="utf-8", errors="replace")


def should_treat_as_google_sheet(source: ConversionSourceRecord, content: str) -> bool:
    if source.source_type == "google_sheet":
        return True
    return "\n## Sheet:" in content


def build_google_sheet_chunks(
    source: ConversionSourceRecord,
    source_title: str,
    content: str,
) -> list[ConversionChunk]:
    chunks: list[ConversionChunk] = []
    chunks.append(
        ConversionChunk(
            chunk_id=f"{source.source_doc_id}:summary",
            source_doc_id=source.source_doc_id,
            source_type=source.source_type,
            source_name=source_title,
            sheet_name="",
            section_title="Source Summary",
            row_start=0,
            row_end=0,
            text=build_google_sheet_summary_text(content),
            char_count=0,
        )
    )

    sections = split_google_sheet_sections(content)
    for sheet_name, body_lines in sections:
        chunks.extend(build_sheet_section_chunks(source, source_title, sheet_name, body_lines))

    return finalize_chunk_sizes(chunks)


def build_structured_document_chunks(
    source: ConversionSourceRecord,
    source_title: str,
    content: str,
) -> list[ConversionChunk]:
    chunks: list[ConversionChunk] = []
    summary_text = build_document_summary_text(content)
    chunks.append(
        ConversionChunk(
            chunk_id=f"{source.source_doc_id}:summary",
            source_doc_id=source.source_doc_id,
            source_type=source.source_type,
            source_name=source_title,
            sheet_name="",
            section_title="Source Summary",
            row_start=0,
            row_end=0,
            text=summary_text,
            char_count=0,
        )
    )

    blocks = extract_document_blocks(content.splitlines())
    for block_index, block in enumerate(blocks):
        section_title = block.title.strip() or f"Section {block_index + 1}"
        if is_markdown_table_block(block.content):
            table_lines = [line.rstrip() for line in block.content.splitlines() if line.strip()]
            chunks.extend(
                build_table_window_chunks(
                    source=source,
                    source_title=source_title,
                    sheet_name="",
                    section_title=section_title,
                    table_lines=table_lines,
                    chunk_prefix=f"{source.source_doc_id}:block:{block_index}",
                )
            )
            continue

        for prose_index, text in enumerate(split_prose_text(block.content, max_chars=PROSE_CHUNK_MAX_CHARS), start=1):
            if not text.strip():
                continue
            suffix = str(prose_index)
            chunks.append(
                ConversionChunk(
                    chunk_id=f"{source.source_doc_id}:block:{block_index}:prose:{suffix}",
                    source_doc_id=source.source_doc_id,
                    source_type=source.source_type,
                    source_name=source_title,
                    sheet_name="",
                    section_title=section_title,
                    row_start=0,
                    row_end=0,
                    text=text.strip(),
                    char_count=0,
                )
            )

    return finalize_chunk_sizes(chunks)


def finalize_chunk_sizes(chunks: Sequence[ConversionChunk]) -> list[ConversionChunk]:
    return [
        ConversionChunk(
            chunk_id=chunk.chunk_id,
            source_doc_id=chunk.source_doc_id,
            source_type=chunk.source_type,
            source_name=chunk.source_name,
            sheet_name=chunk.sheet_name,
            section_title=chunk.section_title,
            row_start=chunk.row_start,
            row_end=chunk.row_end,
            text=chunk.text.strip(),
            char_count=len(chunk.text.strip()),
        )
        for chunk in chunks
        if chunk.text.strip()
    ]


def build_google_sheet_summary_text(content: str) -> str:
    sheet_names = [name for name, _ in split_google_sheet_sections(content)]
    lines = [f"Sheets: {', '.join(sheet_names)}" if sheet_names else "Sheets: none"]
    non_table_lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("|") and not line.strip().startswith("## Sheet:")
    ]
    if non_table_lines:
        lines.append("")
        lines.extend(non_table_lines[:8])
    return "\n".join(lines).strip()


def build_document_summary_text(content: str) -> str:
    headings = [line.strip().lstrip("#").strip() for line in content.splitlines() if line.strip().startswith("#")]
    summary_lines: list[str] = []
    if headings:
        summary_lines.append("Headings: " + ", ".join(headings[:8]))
    preview = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("|")
    ]
    if preview:
        if summary_lines:
            summary_lines.append("")
        summary_lines.extend(preview[:8])
    return "\n".join(summary_lines).strip()


def split_google_sheet_sections(content: str) -> list[tuple[str, list[str]]]:
    lines = content.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_name = ""
    current_body: list[str] = []

    for line in lines[1:]:
        if line.startswith("## Sheet:"):
            if current_name:
                sections.append((current_name, current_body))
            current_name = line.replace("## Sheet:", "", 1).strip()
            current_body = []
            continue
        if current_name:
            current_body.append(line.rstrip())

    if current_name:
        sections.append((current_name, current_body))
    return sections


def build_sheet_section_chunks(
    source: ConversionSourceRecord,
    source_title: str,
    sheet_name: str,
    body_lines: list[str],
) -> list[ConversionChunk]:
    chunks: list[ConversionChunk] = []
    prose_buffer: list[str] = []
    block_index = 0
    line_index = 0

    while line_index < len(body_lines):
        line = body_lines[line_index].rstrip()
        stripped = line.strip()
        if not stripped:
            prose_buffer.append("")
            line_index += 1
            continue

        if is_markdown_table_start(body_lines, line_index):
            chunks.extend(
                flush_prose_chunks(
                    source=source,
                    source_title=source_title,
                    sheet_name=sheet_name,
                    section_title=sheet_name or "Sheet Content",
                    prose_lines=prose_buffer,
                    chunk_prefix=f"{source.source_doc_id}:sheet:{sheet_name}:prose:{block_index}",
                )
            )
            prose_buffer = []
            table_lines = [body_lines[line_index].rstrip(), body_lines[line_index + 1].rstrip()]
            line_index += 2
            while line_index < len(body_lines) and body_lines[line_index].strip().startswith("|"):
                table_lines.append(body_lines[line_index].rstrip())
                line_index += 1
            chunks.extend(
                build_table_window_chunks(
                    source=source,
                    source_title=source_title,
                    sheet_name=sheet_name,
                    section_title=sheet_name,
                    table_lines=table_lines,
                    chunk_prefix=f"{source.source_doc_id}:sheet:{sheet_name}:table:{block_index}",
                )
            )
            block_index += 1
            continue

        prose_buffer.append(line)
        line_index += 1

    chunks.extend(
        flush_prose_chunks(
            source=source,
            source_title=source_title,
            sheet_name=sheet_name,
            section_title=sheet_name or "Sheet Content",
            prose_lines=prose_buffer,
            chunk_prefix=f"{source.source_doc_id}:sheet:{sheet_name}:prose:{block_index}",
        )
    )
    return chunks


def flush_prose_chunks(
    *,
    source: ConversionSourceRecord,
    source_title: str,
    sheet_name: str,
    section_title: str,
    prose_lines: list[str],
    chunk_prefix: str,
) -> list[ConversionChunk]:
    text = "\n".join(line.rstrip() for line in prose_lines).strip()
    if not text:
        return []

    chunks: list[ConversionChunk] = []
    for index, part in enumerate(split_prose_text(text, max_chars=PROSE_CHUNK_MAX_CHARS), start=1):
        chunks.append(
            ConversionChunk(
                chunk_id=f"{chunk_prefix}:{index}",
                source_doc_id=source.source_doc_id,
                source_type=source.source_type,
                source_name=source_title,
                sheet_name=sheet_name,
                section_title=section_title,
                row_start=0,
                row_end=0,
                text=part.strip(),
                char_count=0,
            )
        )
    return chunks


def build_table_window_chunks(
    *,
    source: ConversionSourceRecord,
    source_title: str,
    sheet_name: str,
    section_title: str,
    table_lines: list[str],
    chunk_prefix: str,
) -> list[ConversionChunk]:
    if len(table_lines) <= 2:
        return []

    header_lines = table_lines[:2]
    data_rows = [line.rstrip() for line in table_lines[2:] if line.strip()]
    if not data_rows:
        return [
            ConversionChunk(
                chunk_id=f"{chunk_prefix}:empty",
                source_doc_id=source.source_doc_id,
                source_type=source.source_type,
                source_name=source_title,
                sheet_name=sheet_name,
                section_title=section_title,
                row_start=0,
                row_end=0,
                text="\n".join(header_lines).strip(),
                char_count=0,
            )
        ]

    window_size = TABLE_WINDOW_ROWS
    step = max(TABLE_WINDOW_ROWS - TABLE_WINDOW_OVERLAP, 1)
    chunks: list[ConversionChunk] = []
    chunk_index = 0
    start = 0
    while start < len(data_rows):
        end = min(start + window_size, len(data_rows))
        window_rows = data_rows[start:end]
        row_start = start + 1
        row_end = end
        chunk_text = "\n".join([*header_lines, *window_rows]).strip()
        chunks.append(
            ConversionChunk(
                chunk_id=f"{chunk_prefix}:{chunk_index}",
                source_doc_id=source.source_doc_id,
                source_type=source.source_type,
                source_name=source_title,
                sheet_name=sheet_name,
                section_title=section_title,
                row_start=row_start,
                row_end=row_end,
                text=chunk_text,
                char_count=0,
            )
        )
        if end >= len(data_rows):
            break
        start += step
        chunk_index += 1

    return chunks


def split_prose_text(text: str, *, max_chars: int) -> list[str]:
    normalized = text.strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    if len(paragraphs) <= 1:
        return split_long_text(normalized, max_chars=max_chars)

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0
    for paragraph in paragraphs:
        addition = len(paragraph) + (2 if current_parts else 0)
        if current_parts and current_length + addition > max_chars:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = []
            current_length = 0
        if len(paragraph) > max_chars:
            if current_parts:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_length = 0
            chunks.extend(split_long_text(paragraph, max_chars=max_chars))
            continue
        current_parts.append(paragraph)
        current_length += addition

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())
    return chunks


def split_long_text(text: str, *, max_chars: int) -> list[str]:
    words = text.split()
    if not words:
        return [text[:max_chars].strip()]

    chunks: list[str] = []
    current_words: list[str] = []
    current_length = 0
    for word in words:
        addition = len(word) + (1 if current_words else 0)
        if current_words and current_length + addition > max_chars:
            chunks.append(" ".join(current_words).strip())
            current_words = []
            current_length = 0
        if len(word) > max_chars:
            if current_words:
                chunks.append(" ".join(current_words).strip())
                current_words = []
                current_length = 0
            for index in range(0, len(word), max_chars):
                chunks.append(word[index : index + max_chars].strip())
            continue
        current_words.append(word)
        current_length += addition

    if current_words:
        chunks.append(" ".join(current_words).strip())
    return chunks


def score_conversion_chunk(chunk: ConversionChunk, query: str, terms: list[str]) -> int:
    metadata_text = " ".join(
        part for part in (chunk.source_name, chunk.sheet_name, chunk.section_title) if part
    ).lower()
    content_text = chunk.text.lower()
    normalized_query = query.strip().lower()

    score = 0
    if normalized_query and normalized_query in content_text:
        score += max(8, len(terms) * 2)
    if normalized_query and normalized_query in metadata_text:
        score += max(10, len(terms) * 3)

    for term in terms:
        score += content_text.count(term)
        score += metadata_text.count(term) * 3

    if score > 0 and chunk.row_start and chunk.row_end:
        score += 1
    return score


def normalize_conversion_query_terms(query: str) -> list[str]:
    terms = re.findall(r"[\w\u4e00-\u9fff]+", sanitize_query_text(query).lower())
    return [term for term in terms if len(term) >= 2 and term not in STOPWORDS]


def sanitize_query_text(text: str) -> str:
    without_urls = re.sub(r"https?://\S+", " ", text or "")
    return re.sub(r"\s+", " ", without_urls).strip()


def is_markdown_table_block(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    return lines[0].startswith("|") and bool(re.match(r"^\|\s*:?-{3,}", lines[1]))


def is_markdown_table_start(lines: Sequence[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    current = lines[index].strip()
    next_line = lines[index + 1].strip()
    return current.startswith("|") and bool(re.match(r"^\|\s*:?-{3,}", next_line))
