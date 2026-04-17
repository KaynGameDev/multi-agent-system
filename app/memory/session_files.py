from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.frontmatter import normalize_metadata_keys, render_frontmatter_document, split_frontmatter
from app.memory.types import SessionMemoryFile, SessionMemoryFileUpdate

SESSION_MEMORY_FILE_KIND = "session_memory"
SESSION_MEMORY_TEMPLATE_VERSION = 1
SESSION_MEMORY_TITLE = "# Session Memory"
SESSION_MEMORY_EMPTY_KEY_FILES_SENTINEL = "_None yet._"
SESSION_MEMORY_SECTION_LABELS = {
    "current_state": "Current State",
    "task_spec": "Task Spec",
    "key_files": "Key Files",
    "workflow": "Workflow",
    "errors_corrections": "Errors/Corrections",
    "learnings": "Learnings",
    "worklog": "Worklog",
}


class SessionMemoryFormatError(ValueError):
    """Raised when a session memory file is missing or invalid."""


class FileSessionMemoryStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()

    def ensure(self, thread_id: str) -> SessionMemoryFile:
        return ensure_session_memory_file(self.root_dir, thread_id)

    def get(self, thread_id: str) -> SessionMemoryFile | None:
        return get_session_memory_file(self.root_dir, thread_id)

    def update(self, thread_id: str, patch: SessionMemoryFileUpdate | dict[str, Any]) -> SessionMemoryFile:
        return update_session_memory_file(self.root_dir, thread_id, patch)

    def delete(self, thread_id: str) -> bool:
        return delete_session_memory_file(self.root_dir, thread_id)


def ensure_session_memory_file(root_dir: str | Path, thread_id: str) -> SessionMemoryFile:
    existing = get_session_memory_file(root_dir, thread_id)
    if existing is not None:
        return existing

    normalized_thread_id = normalize_session_memory_thread_id(thread_id)
    path = resolve_session_memory_file_path(root_dir, normalized_thread_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = SessionMemoryFile(
        thread_id=normalized_thread_id,
        source_path=str(path),
    )
    return _write_session_memory_file(path, document)


def get_session_memory_file(root_dir: str | Path, thread_id: str) -> SessionMemoryFile | None:
    path = resolve_session_memory_file_path(root_dir, thread_id)
    if not path.exists():
        return None
    return load_session_memory_file(path)


def update_session_memory_file(
    root_dir: str | Path,
    thread_id: str,
    patch: SessionMemoryFileUpdate | dict[str, Any],
) -> SessionMemoryFile:
    existing = ensure_session_memory_file(root_dir, thread_id)
    update = SessionMemoryFileUpdate.model_validate(patch)
    merged_payload = existing.model_dump(exclude={"source_path"})
    merged_payload.update(update.model_dump(exclude_none=True))

    path = resolve_session_memory_file_path(root_dir, existing.thread_id)
    return _write_session_memory_file(
        path,
        SessionMemoryFile.model_validate(
            {
                **merged_payload,
                "source_path": str(path),
            }
        ),
    )


def delete_session_memory_file(root_dir: str | Path, thread_id: str) -> bool:
    path = resolve_session_memory_file_path(root_dir, thread_id)
    if not path.exists():
        return False
    path.unlink()
    _prune_empty_parent_directories(path.parent, stop_at=Path(root_dir).expanduser().resolve())
    return True


def load_session_memory_file(path: str | Path) -> SessionMemoryFile:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.exists():
        raise SessionMemoryFormatError(f"Session memory file does not exist: {resolved_path}")
    if not resolved_path.is_file():
        raise SessionMemoryFormatError(f"Session memory path must be a file: {resolved_path}")

    raw_text = resolved_path.read_text(encoding="utf-8")
    metadata, body = split_frontmatter(raw_text)
    normalized_metadata = normalize_metadata_keys(metadata)

    thread_id = str(normalized_metadata.get("thread_id", "") or "").strip()
    if not thread_id:
        raise SessionMemoryFormatError(f"Session memory file {resolved_path} must declare thread_id in frontmatter.")
    kind = str(normalized_metadata.get("kind", "") or "").strip()
    if kind and kind != SESSION_MEMORY_FILE_KIND:
        raise SessionMemoryFormatError(
            f"Session memory file {resolved_path} must declare kind={SESSION_MEMORY_FILE_KIND}."
        )
    template_version = normalized_metadata.get("template_version", SESSION_MEMORY_TEMPLATE_VERSION)
    if int(template_version or 0) != SESSION_MEMORY_TEMPLATE_VERSION:
        raise SessionMemoryFormatError(
            f"Session memory file {resolved_path} has unsupported template_version: {template_version}"
        )

    sections = parse_session_memory_sections(body)
    return SessionMemoryFile(
        thread_id=thread_id,
        source_path=str(resolved_path),
        current_state=sections["current_state"],
        task_spec=sections["task_spec"],
        key_files=sections["key_files"],
        workflow=sections["workflow"],
        errors_corrections=sections["errors_corrections"],
        learnings=sections["learnings"],
        worklog=sections["worklog"],
    )


def resolve_session_memory_file_path(root_dir: str | Path, thread_id: str) -> Path:
    resolved_root_dir = Path(root_dir).expanduser().resolve()
    return (resolved_root_dir / build_session_memory_relative_path(thread_id)).resolve()


def build_session_memory_relative_path(thread_id: str) -> Path:
    normalized_thread_id = normalize_session_memory_thread_id(thread_id)
    raw_segments: list[str] = []
    for slash_part in normalized_thread_id.replace("\\", "/").split("/"):
        raw_segments.extend(slash_part.split(":"))

    normalized_segments = [normalize_session_memory_path_segment(part) for part in raw_segments]
    cleaned_segments = [part for part in normalized_segments if part]
    if not cleaned_segments:
        raise SessionMemoryFormatError("Session thread id must resolve to at least one safe path segment.")
    return Path(*cleaned_segments[:-1], f"{cleaned_segments[-1]}.md")


def normalize_session_memory_thread_id(thread_id: str) -> str:
    cleaned = str(thread_id or "").strip()
    if not cleaned:
        raise SessionMemoryFormatError("Session thread id must not be empty.")
    return cleaned


def normalize_session_memory_path_segment(value: str) -> str:
    cleaned = str(value or "").strip().lower()
    if not cleaned:
        return ""
    cleaned = re.sub(r"[^a-z0-9._-]+", "-", cleaned)
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    cleaned = cleaned.strip("-._")
    if cleaned in {"", ".", ".."}:
        return ""
    return cleaned


def render_session_memory_document(document: SessionMemoryFile | dict[str, Any]) -> str:
    session_file = SessionMemoryFile.model_validate(document)
    body_lines = [SESSION_MEMORY_TITLE, ""]
    for field_name, label in SESSION_MEMORY_SECTION_LABELS.items():
        body_lines.extend([f"## {label}", ""])
        if field_name == "key_files":
            if session_file.key_files:
                body_lines.extend(f"- {item}" for item in session_file.key_files)
            else:
                body_lines.append(SESSION_MEMORY_EMPTY_KEY_FILES_SENTINEL)
        else:
            rendered_value = str(getattr(session_file, field_name, "") or "").strip()
            if rendered_value:
                body_lines.append(rendered_value)
        body_lines.append("")

    return render_frontmatter_document(
        {
            "thread_id": session_file.thread_id,
            "kind": SESSION_MEMORY_FILE_KIND,
            "template_version": SESSION_MEMORY_TEMPLATE_VERSION,
        },
        "\n".join(body_lines).strip(),
    )


def parse_session_memory_sections(body: str) -> dict[str, Any]:
    normalized_body = str(body or "").strip()
    if not normalized_body:
        raise SessionMemoryFormatError("Session memory file must include the session template body.")

    heading_to_key = {f"## {label}": key for key, label in SESSION_MEMORY_SECTION_LABELS.items()}
    section_buffers: dict[str, list[str]] = {}
    current_key = ""
    saw_title = False

    for raw_line in normalized_body.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped == SESSION_MEMORY_TITLE:
            saw_title = True
            continue
        if stripped in heading_to_key:
            current_key = heading_to_key[stripped]
            section_buffers[current_key] = []
            continue
        if not current_key:
            if stripped:
                raise SessionMemoryFormatError(f"Unexpected content before first session section: {stripped}")
            continue
        section_buffers[current_key].append(line)

    if not saw_title:
        raise SessionMemoryFormatError("Session memory file must start with the Session Memory title.")

    missing_keys = [key for key in SESSION_MEMORY_SECTION_LABELS if key not in section_buffers]
    if missing_keys:
        raise SessionMemoryFormatError(
            f"Session memory file is missing required sections: {', '.join(missing_keys)}"
        )

    parsed_sections: dict[str, Any] = {}
    for key in SESSION_MEMORY_SECTION_LABELS:
        if key == "key_files":
            parsed_sections[key] = _parse_key_files_section(section_buffers[key])
        else:
            parsed_sections[key] = "\n".join(line.rstrip() for line in section_buffers[key]).strip()
    return parsed_sections


def _parse_key_files_section(lines: list[str]) -> list[str]:
    key_files: list[str] = []
    seen: set[str] = set()
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped == SESSION_MEMORY_EMPTY_KEY_FILES_SENTINEL:
            continue
        if not stripped.startswith("- "):
            raise SessionMemoryFormatError("Session memory Key Files section must use markdown bullet items.")
        value = stripped[2:].strip()
        if not value or value in seen:
            continue
        seen.add(value)
        key_files.append(value)
    return key_files


def _write_session_memory_file(path: Path, document: SessionMemoryFile) -> SessionMemoryFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_session_memory_document(document), encoding="utf-8")
    return load_session_memory_file(path)


def _prune_empty_parent_directories(start_dir: Path, *, stop_at: Path) -> None:
    current = start_dir.resolve()
    resolved_stop_at = stop_at.resolve()
    while current != resolved_stop_at and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent
