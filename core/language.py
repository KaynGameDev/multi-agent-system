from __future__ import annotations

import re


LANGUAGE_MATCHING_PROMPT = (
    "Reply in the same language as the user's latest message unless the user explicitly asks for a different language."
)

_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_WORD_PATTERN = re.compile(r"[A-Za-z]+")
_CHINESE_PUNCTUATION_PATTERN = re.compile(r"[，。！？；：、“”‘’（）《》【】]")


def detect_response_language(text: str) -> str:
    if not isinstance(text, str):
        return "en"
    cjk_count = len(_CJK_PATTERN.findall(text))
    if cjk_count == 0:
        return "en"

    latin_word_count = len(_LATIN_WORD_PATTERN.findall(text))
    if latin_word_count == 0:
        return "zh"
    if _CHINESE_PUNCTUATION_PATTERN.search(text) and cjk_count >= 2:
        return "zh"
    if cjk_count >= latin_word_count:
        return "zh"
    return "en"
