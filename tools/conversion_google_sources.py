from __future__ import annotations

import re
from dataclasses import dataclass

from tools.google_workspace_services import get_google_docs_service, get_google_sheets_service

DOC_URL_PATTERN = re.compile(r"https://docs\.google\.com/document/d/([A-Za-z0-9\-_]+)")
SHEET_URL_PATTERN = re.compile(r"https://docs\.google\.com/spreadsheets/d/([A-Za-z0-9\-_]+)")


@dataclass(frozen=True)
class GoogleDocumentReference:
    document_type: str
    document_id: str
    url: str


def extract_google_document_references(text: str) -> list[GoogleDocumentReference]:
    references: list[GoogleDocumentReference] = []
    seen: set[tuple[str, str]] = set()

    for match in DOC_URL_PATTERN.finditer(text or ""):
        key = ("google_doc", match.group(1))
        if key in seen:
            continue
        seen.add(key)
        references.append(
            GoogleDocumentReference(
                document_type="google_doc",
                document_id=match.group(1),
                url=match.group(0),
            )
        )

    for match in SHEET_URL_PATTERN.finditer(text or ""):
        key = ("google_sheet", match.group(1))
        if key in seen:
            continue
        seen.add(key)
        references.append(
            GoogleDocumentReference(
                document_type="google_sheet",
                document_id=match.group(1),
                url=match.group(0),
            )
        )

    return references

def fetch_google_document_source(reference: GoogleDocumentReference) -> tuple[str, str, str]:
    if reference.document_type == "google_doc":
        return fetch_google_doc(reference.document_id)
    if reference.document_type == "google_sheet":
        return fetch_google_sheet(reference.document_id)
    raise RuntimeError(f"Unsupported Google document type: {reference.document_type}")


def fetch_google_doc(document_id: str) -> tuple[str, str, str]:
    service = get_google_docs_service()
    document = service.documents().get(documentId=document_id).execute()
    title = str(document.get("title", "")).strip() or document_id
    lines = [f"# {title}"]
    lines.extend(render_doc_structural_elements(document.get("body", {}).get("content", [])))
    content = "\n".join(line for line in lines if line.strip()).strip() or f"# {title}\n(empty document)"
    return title, "google_doc", content


def render_doc_structural_elements(elements: list[dict]) -> list[str]:
    lines: list[str] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        if "paragraph" in element:
            paragraph_lines = render_doc_paragraph(element["paragraph"])
            lines.extend(paragraph_lines)
            continue
        if "table" in element:
            table_lines = render_doc_table(element["table"])
            lines.extend(table_lines)
            continue
        if "tableOfContents" in element:
            toc_lines = render_doc_structural_elements(element["tableOfContents"].get("content", []))
            lines.extend(toc_lines)
    return lines


def render_doc_paragraph(paragraph: dict) -> list[str]:
    elements = paragraph.get("elements", [])
    text_parts: list[str] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        text_run = element.get("textRun")
        if isinstance(text_run, dict):
            text_parts.append(str(text_run.get("content", "")))
    text = "".join(text_parts).replace("\u000b", "\n").strip()
    if not text:
        return []

    named_style = str(paragraph.get("paragraphStyle", {}).get("namedStyleType", "")).strip()
    bullet = paragraph.get("bullet")
    if bullet:
        return [f"- {text}"]

    heading_level = named_style_to_heading_level(named_style)
    if heading_level:
        return [f"{'#' * heading_level} {text}"]
    return [text]


def named_style_to_heading_level(named_style: str) -> int:
    mapping = {
        "TITLE": 1,
        "SUBTITLE": 2,
        "HEADING_1": 2,
        "HEADING_2": 3,
        "HEADING_3": 4,
        "HEADING_4": 5,
        "HEADING_5": 6,
        "HEADING_6": 6,
    }
    return mapping.get(named_style, 0)


def render_doc_table(table: dict) -> list[str]:
    rows = table.get("tableRows", [])
    extracted_rows: list[list[str]] = []

    for row in rows:
        cells = row.get("tableCells", []) if isinstance(row, dict) else []
        rendered_cells: list[str] = []
        for cell in cells:
            cell_lines = render_doc_structural_elements(cell.get("content", []))
            rendered_cells.append(" / ".join(line.strip() for line in cell_lines if line.strip()))
        if any(value.strip() for value in rendered_cells):
            extracted_rows.append(rendered_cells)

    if not extracted_rows:
        return []

    width = max(len(row) for row in extracted_rows)
    header = pad_row(extracted_rows[0], width)
    lines = [
        f"| {' | '.join(escape_table_cell(value) or f'column_{index + 1}' for index, value in enumerate(header))} |",
        f"| {' | '.join('---' for _ in range(width))} |",
    ]
    for row in extracted_rows[1:]:
        padded_row = pad_row(row, width)
        lines.append(f"| {' | '.join(escape_table_cell(value) or ' ' for value in padded_row)} |")
    return lines


def fetch_google_sheet(spreadsheet_id: str) -> tuple[str, str, str]:
    service = get_google_sheets_service()
    spreadsheet = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    title = str(spreadsheet.get("properties", {}).get("title", "")).strip() or spreadsheet_id
    sheet_titles = [
        str(sheet.get("properties", {}).get("title", "")).strip()
        for sheet in spreadsheet.get("sheets", [])
        if str(sheet.get("properties", {}).get("title", "")).strip()
    ]
    requested_ranges = [quote_sheet_name(name) for name in sheet_titles]
    values_result = (
        service.spreadsheets()
        .values()
        .batchGet(spreadsheetId=spreadsheet_id, ranges=requested_ranges)
        .execute()
    )

    lines = [f"# {title}"]
    for index, sheet_name in enumerate(sheet_titles):
        lines.append(f"## Sheet: {sheet_name}")
        value_range = values_result.get("valueRanges", [])[index] if index < len(values_result.get("valueRanges", [])) else {}
        rows = [
            [str(cell).strip() for cell in row]
            for row in value_range.get("values", [])
            if isinstance(row, list)
        ]
        lines.extend(render_sheet_rows(rows))

    content = "\n".join(line for line in lines if line.strip()).strip() or f"# {title}\n(empty spreadsheet)"
    return title, "google_sheet", content


def render_sheet_rows(rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["(empty sheet)"]

    width = max(len(row) for row in rows)
    header = pad_row(rows[0], width)
    lines = [
        f"| {' | '.join(escape_table_cell(value) or f'column_{index + 1}' for index, value in enumerate(header))} |",
        f"| {' | '.join('---' for _ in range(width))} |",
    ]
    for row in rows[1:]:
        padded_row = pad_row(row, width)
        if not any(value.strip() for value in padded_row):
            continue
        lines.append(f"| {' | '.join(escape_table_cell(value) or ' ' for value in padded_row)} |")
    return lines


def quote_sheet_name(sheet_name: str) -> str:
    escaped = sheet_name.replace("'", "''")
    return f"'{escaped}'"


def pad_row(row: list[str], width: int) -> list[str]:
    return list(row) + [""] * max(width - len(row), 0)


def escape_table_cell(value: str) -> str:
    return value.replace("|", "/").strip()
