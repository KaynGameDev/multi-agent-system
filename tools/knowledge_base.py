from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

from langchain_core.tools import tool
from openpyxl import load_workbook

from core.config import load_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_knowledge_base_root() -> Path:
    settings = load_settings()
    configured_root = Path(settings.knowledge_base_dir).expanduser()
    if not configured_root.is_absolute():
        configured_root = PROJECT_ROOT / configured_root
    return configured_root.resolve()


def get_supported_knowledge_file_types() -> tuple[str, ...]:
    settings = load_settings()
    return tuple(file_type.lower() for file_type in settings.knowledge_file_types)


def get_knowledge_document_paths() -> list[Path]:
    root = get_knowledge_base_root()
    if not root.exists() or not root.is_dir():
        return []

    supported_types = set(get_supported_knowledge_file_types())
    documents: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if is_hidden_path(path, root):
            continue
        if path.suffix.lower() not in supported_types:
            continue
        documents.append(path)
    return documents


def is_hidden_path(path: Path, root: Path) -> bool:
    try:
        relative_path = path.relative_to(root)
    except ValueError:
        relative_path = path
    return any(part.startswith(".") for part in relative_path.parts)


def load_knowledge_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt", ".rst"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        return render_delimited_file_as_text(path, delimiter=delimiter)
    if suffix in {".xlsx", ".xlsm"}:
        return render_workbook_as_text(path)
    raise ValueError(f"Unsupported knowledge document type: {suffix}")


def build_document_metadata(path: Path) -> dict[str, str]:
    root = get_knowledge_base_root()
    content = load_knowledge_document(path)
    relative_path = safe_relative_path(path, root)
    title = extract_document_title(content, default=humanize_path_name(path))
    return {
        "name": path.name,
        "title": title,
        "path": relative_path,
        "file_type": path.suffix.lower(),
    }


def safe_relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def humanize_path_name(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip() or path.stem


def extract_document_title(content: str, *, default: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or default
        if stripped:
            return stripped
    return default


def normalize_query_terms(query: str) -> list[str]:
    return [term for term in re.findall(r"[\w\u4e00-\u9fff]+", query.lower()) if len(term) >= 2]


def score_document(content: str, query: str, terms: list[str]) -> int:
    lowered = content.lower()
    score = 0
    normalized_query = query.strip().lower()
    if normalized_query and normalized_query in lowered:
        score += max(8, len(terms) * 2)

    for term in terms:
        score += lowered.count(term)

    return score


def find_best_match_line(lines: list[str], query: str, terms: list[str]) -> int:
    normalized_query = query.strip().lower()
    best_index = 0
    best_score = -1

    for index, line in enumerate(lines):
        lowered = line.lower()
        line_score = 0
        if normalized_query and normalized_query in lowered:
            line_score += max(8, len(terms) * 2)
        for term in terms:
            line_score += lowered.count(term)
        if line_score > best_score:
            best_score = line_score
            best_index = index

    return best_index


def build_snippet(lines: list[str], match_index: int, *, radius: int = 2) -> str:
    start = max(match_index - radius, 0)
    end = min(match_index + radius + 1, len(lines))
    snippet_lines = [line.rstrip() for line in lines[start:end] if line.strip()]
    return "\n".join(snippet_lines).strip()


def is_markdown_heading(line: str) -> bool:
    return bool(re.match(r"^\s*#{1,6}\s+", line))


def heading_level(line: str) -> int:
    match = re.match(r"^\s*(#{1,6})\s+", line)
    if not match:
        return 0
    return len(match.group(1))


def resolve_document(document_name: str) -> Path | None:
    normalized_query = document_name.strip().lower()
    if not normalized_query:
        return None

    exact_matches: list[Path] = []
    partial_matches: list[Path] = []

    for path in get_knowledge_document_paths():
        metadata = build_document_metadata(path)
        candidates = {
            path.name.lower(),
            path.stem.lower(),
            metadata["title"].lower(),
            metadata["path"].lower(),
        }
        if normalized_query in candidates:
            exact_matches.append(path)
            continue
        if any(normalized_query in candidate for candidate in candidates):
            partial_matches.append(path)

    if exact_matches:
        return exact_matches[0]
    if partial_matches:
        return partial_matches[0]
    return None


def locate_section(lines: list[str], section_query: str) -> tuple[int, int]:
    normalized_query = section_query.strip().lower()
    if not normalized_query:
        return 0, len(lines)

    for index, line in enumerate(lines):
        if normalized_query not in line.lower():
            continue

        if is_markdown_heading(line):
            section_start = index
            section_level = heading_level(line)
            section_end = len(lines)
            for next_index in range(index + 1, len(lines)):
                next_line = lines[next_index]
                if is_markdown_heading(next_line) and heading_level(next_line) <= section_level:
                    section_end = next_index
                    break
            return section_start, section_end

        return max(index - 3, 0), min(index + 4, len(lines))

    return 0, len(lines)


def render_delimited_file_as_text(path: Path, *, delimiter: str) -> str:
    title = humanize_path_name(path)
    lines = [f"# {title}", f"## Table: {path.name}"]
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        lines.extend(render_row_stream(reader))
    return "\n".join(line for line in lines if line).strip()


def render_workbook_as_text(path: Path) -> str:
    title = humanize_path_name(path)
    lines = [f"# {title}"]
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if not workbook.worksheets:
            lines.append("(empty workbook)")
            return "\n".join(lines)

        for worksheet in workbook.worksheets:
            lines.append(f"## Sheet: {worksheet.title}")
            lines.extend(
                render_row_stream(
                    (
                        [format_cell_value(value) for value in row]
                        for row in worksheet.iter_rows(values_only=True)
                    )
                )
            )
    finally:
        workbook.close()

    return "\n".join(line for line in lines if line).strip()


def render_row_stream(rows: Iterable[list[str]]) -> list[str]:
    rendered_lines: list[str] = []
    header: list[str] | None = None
    saw_content = False

    for row_index, row in enumerate(rows, start=1):
        normalized_row = [cell.strip() for cell in row]
        if not any(normalized_row):
            continue

        saw_content = True
        if header is None:
            header = build_header_row(normalized_row)
            rendered_lines.append(f"Columns: {' | '.join(header)}")
            continue

        rendered_row = format_row_with_header(normalized_row, header)
        if rendered_row:
            rendered_lines.append(f"Row {row_index}: {rendered_row}")

    if not saw_content:
        rendered_lines.append("(empty)")

    return rendered_lines


def build_header_row(row: list[str]) -> list[str]:
    headers: list[str] = []
    for index, value in enumerate(row, start=1):
        header_value = value or f"column_{index}"
        headers.append(header_value)
    return headers


def format_row_with_header(row: list[str], header: list[str]) -> str:
    parts: list[str] = []
    column_count = max(len(row), len(header))
    for index in range(column_count):
        header_value = header[index] if index < len(header) else f"column_{index + 1}"
        cell_value = row[index] if index < len(row) else ""
        if not cell_value:
            continue
        parts.append(f"{header_value}: {cell_value}")
    return " | ".join(parts)


def format_cell_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value).strip()


def build_knowledge_base_context() -> dict[str, object]:
    root = get_knowledge_base_root()
    return {
        "knowledge_base_dir": str(root),
        "exists": root.exists() and root.is_dir(),
        "supported_file_types": list(get_supported_knowledge_file_types()),
    }


@tool
def list_knowledge_documents() -> dict[str, object]:
    """List the internal documents available to the knowledge agent."""
    context = build_knowledge_base_context()
    documents = [build_document_metadata(path) for path in get_knowledge_document_paths()]
    return {
        "ok": True,
        **context,
        "document_count": len(documents),
        "documents": documents,
    }


@tool
def search_knowledge_documents(query: str, limit: int = 5) -> dict[str, object]:
    """Search local knowledge documents and return structured matches."""
    normalized_limit = max(limit, 1)
    context = build_knowledge_base_context()
    if not query.strip():
        return {
            "ok": False,
            **context,
            "error": "Query cannot be empty.",
            "query": query,
            "match_count": 0,
            "documents": [],
        }

    terms = normalize_query_terms(query)
    matches: list[dict[str, object]] = []

    for path in get_knowledge_document_paths():
        content = load_knowledge_document(path)
        score = score_document(content, query, terms)
        if score <= 0:
            continue

        lines = content.splitlines()
        match_index = find_best_match_line(lines, query, terms)
        metadata = build_document_metadata(path)
        matches.append(
            {
                **metadata,
                "score": score,
                "line_number": match_index + 1,
                "snippet": build_snippet(lines, match_index),
            }
        )

    matches.sort(key=lambda item: (-int(item["score"]), str(item["path"])))
    return {
        "ok": True,
        **context,
        "query": query,
        "match_count": len(matches),
        "documents": matches[:normalized_limit],
    }


@tool
def read_knowledge_document(document_name: str, section_query: str = "", max_lines: int = 80) -> dict[str, object]:
    """Read a local knowledge document, optionally focusing on a matching section."""
    context = build_knowledge_base_context()
    path = resolve_document(document_name)
    if path is None:
        return {
            "ok": False,
            **context,
            "error": f"Document not found: {document_name}",
            "document_name": document_name,
        }

    normalized_max_lines = max(max_lines, 1)
    content = load_knowledge_document(path)
    lines = content.splitlines()
    start, end = locate_section(lines, section_query)
    excerpt_lines = lines[start:min(end, start + normalized_max_lines)]
    metadata = build_document_metadata(path)

    return {
        "ok": True,
        **context,
        "document": metadata,
        "section_query": section_query.strip(),
        "start_line": start + 1,
        "end_line": start + len(excerpt_lines),
        "content": "\n".join(excerpt_lines).strip(),
        "truncated": start + len(excerpt_lines) < end,
    }
