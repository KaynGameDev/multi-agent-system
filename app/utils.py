from __future__ import annotations

from typing import Any


def safe_get_str(mapping: dict[str, Any], key: str, default: str = "") -> str:
    """Extract a string from a dict, returning a stripped, non-None string.

    Equivalent to ``str(mapping.get(key, default)).strip()``.
    """
    return str(mapping.get(key, default)).strip()
