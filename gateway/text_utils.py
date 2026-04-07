from __future__ import annotations

import re

CASUAL_GREETING_NORMALIZED_TEXTS = {
    "hi",
    "hello",
    "hey",
    "yo",
    "good morning",
    "good afternoon",
    "good evening",
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "早上好",
    "下午好",
    "晚上好",
}

_TOKEN_STOP_WORDS = {"the", "and", "for", "with", "that", "this", "what", "when", "where"}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().casefold())


def tokenize_text(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", text.casefold())
        if token not in _TOKEN_STOP_WORDS
    }


def is_casual_greeting(text: str) -> bool:
    normalized = re.sub(r"[!?,.，。！？\s]+", " ", (text or "").strip().lower()).strip()
    if not normalized:
        return False
    return normalized in CASUAL_GREETING_NORMALIZED_TEXTS
