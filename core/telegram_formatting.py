from __future__ import annotations

import html
import re


FENCED_CODE_BLOCK_PATTERN = re.compile(r"```(?:[^\n`]*\n)?(.*?)```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`([^`\n]+)`")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)]\((https?://[^)\s]+)\)")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")
ITALIC_PATTERN = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)")


def to_telegram_html(text: str) -> str:
    if not text:
        return ""

    protected_segments: list[str] = []

    def protect(value: str) -> str:
        protected_segments.append(value)
        return f"@@TELEGRAM_SEGMENT_{len(protected_segments) - 1}@@"

    def protect_code_block(match: re.Match[str]) -> str:
        code = match.group(1).strip("\n")
        return protect(f"<pre><code>{html.escape(code)}</code></pre>")

    def protect_inline_code(match: re.Match[str]) -> str:
        code = match.group(1)
        return protect(f"<code>{html.escape(code)}</code>")

    def protect_link(match: re.Match[str]) -> str:
        label = html.escape(match.group(1).strip())
        url = html.escape(match.group(2).strip(), quote=True)
        return protect(f'<a href="{url}">{label}</a>')

    formatted = text.strip()
    formatted = FENCED_CODE_BLOCK_PATTERN.sub(protect_code_block, formatted)
    formatted = INLINE_CODE_PATTERN.sub(protect_inline_code, formatted)
    formatted = MARKDOWN_LINK_PATTERN.sub(protect_link, formatted)
    formatted = html.escape(formatted)
    formatted = HEADING_PATTERN.sub(_convert_heading, formatted)
    formatted = BOLD_PATTERN.sub(r"<b>\1</b>", formatted)
    formatted = ITALIC_PATTERN.sub(r"<i>\1</i>", formatted)
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)

    for index, segment in enumerate(protected_segments):
        formatted = formatted.replace(f"@@TELEGRAM_SEGMENT_{index}@@", segment)

    return formatted.strip()


def _convert_heading(match: re.Match[str]) -> str:
    title = match.group(2).strip()
    if not title:
        return ""
    return f"<b>{title}</b>"
