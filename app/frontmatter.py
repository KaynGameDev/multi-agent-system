from __future__ import annotations

import re
from typing import Any

FRONTMATTER_DELIMITER = "---"


def split_frontmatter(raw_text: str, *, delimiter: str = FRONTMATTER_DELIMITER) -> tuple[dict[str, Any], str]:
    if not raw_text.startswith(f"{delimiter}\n"):
        return {}, raw_text

    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != delimiter:
        return {}, raw_text

    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == delimiter:
            end_index = index
            break

    if end_index is None:
        return {}, raw_text

    frontmatter_text = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :]).lstrip("\n")
    return parse_frontmatter(frontmatter_text), body


def parse_frontmatter(frontmatter_text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    current_key = ""

    for raw_line in frontmatter_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if current_key and stripped.startswith("- "):
            current_value = values.setdefault(current_key, [])
            if isinstance(current_value, list):
                current_value.append(parse_frontmatter_value(stripped[2:].strip()))
            continue

        match = re.match(r"^([A-Za-z0-9_-]+):(?:\s*(.*))?$", stripped)
        if not match:
            current_key = ""
            continue

        current_key = match.group(1)
        raw_value = (match.group(2) or "").strip()
        if not raw_value:
            values[current_key] = []
            continue
        values[current_key] = parse_frontmatter_value(raw_value)
        if not isinstance(values[current_key], list):
            current_key = ""

    return values


def parse_frontmatter_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""
    lowered = value.casefold()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_frontmatter_value(item.strip()) for item in inner.split(",")]
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def normalize_metadata_keys(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        normalized[key.replace("-", "_")] = value
    return normalized


def dump_frontmatter(metadata: dict[str, Any], *, delimiter: str = FRONTMATTER_DELIMITER) -> str:
    lines = [delimiter]
    for key, value in metadata.items():
        normalized_key = str(key or "").strip()
        if not normalized_key:
            continue
        if isinstance(value, bool):
            rendered_value = "true" if value else "false"
            lines.append(f"{normalized_key}: {rendered_value}")
            continue
        if isinstance(value, (list, tuple)):
            lines.append(f"{normalized_key}:")
            for item in value:
                lines.append(f"- {_dump_frontmatter_scalar(item)}")
            continue
        lines.append(f"{normalized_key}: {_dump_frontmatter_scalar(value)}")
    lines.append(delimiter)
    return "\n".join(lines)


def render_frontmatter_document(
    metadata: dict[str, Any],
    body: str = "",
    *,
    delimiter: str = FRONTMATTER_DELIMITER,
) -> str:
    normalized_body = str(body or "").strip()
    parts = [
        dump_frontmatter(metadata, delimiter=delimiter),
        "",
    ]
    if normalized_body:
        parts.extend([normalized_body, ""])
    return "\n".join(parts)


def _dump_frontmatter_scalar(value: Any) -> str:
    if value is None:
        return '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)

    rendered = str(value)
    if not rendered:
        return '""'
    if re.search(r"[:#\[\],'\"]", rendered) or rendered != rendered.strip():
        escaped = rendered.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return rendered
