from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(configured_value: str | Path, default_value: str | Path = "") -> Path:
    candidate = configured_value or default_value
    path = Path(candidate).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()
