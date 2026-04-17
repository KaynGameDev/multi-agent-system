from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def build_memory_telemetry_payload(
    event: str,
    *,
    status: str = "ok",
    **fields: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event": str(event or "").strip() or "memory.unknown",
        "status": str(status or "").strip().lower() or "ok",
    }
    for key, value in fields.items():
        cleaned_key = str(key or "").strip()
        if not cleaned_key:
            continue
        payload[cleaned_key] = _normalize_telemetry_value(value)
    return payload


def emit_memory_telemetry(
    logger: logging.Logger,
    event: str,
    *,
    status: str = "ok",
    **fields: Any,
) -> dict[str, Any]:
    payload = build_memory_telemetry_payload(event, status=status, **fields)
    level = logging.WARNING if payload["status"] == "error" else logging.INFO
    logger.log(level, "memory telemetry %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return payload


def _normalize_telemetry_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _normalize_telemetry_value(item)
            for key, item in value.items()
            if str(key).strip()
        }
    if isinstance(value, (list, tuple, set)):
        return [_normalize_telemetry_value(item) for item in value]
    return str(value)
