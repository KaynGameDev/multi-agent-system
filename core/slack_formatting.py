from __future__ import annotations

import re


FENCED_CODE_BLOCK_PATTERN = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`[^`\n]+`")
BOLD_PATTERN = re.compile(r"\*\*(.+?)\*\*")
UNDERLINE_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^]]+)]\(([^)]+)\)")
HTML_BOLD_OPEN = re.compile(r"<\s*(b|strong)\s*>", re.IGNORECASE)
HTML_BOLD_CLOSE = re.compile(r"<\s*/\s*(b|strong)\s*>", re.IGNORECASE)
HTML_ITALIC_OPEN = re.compile(r"<\s*(i|em)\s*>", re.IGNORECASE)
HTML_ITALIC_CLOSE = re.compile(r"<\s*/\s*(i|em)\s*>", re.IGNORECASE)


def to_slack_mrkdwn(text: str) -> str:
    if not text:
        return ""

    protected_blocks: list[str] = []
    protected_inline: list[str] = []

    def protect_block(match: re.Match[str]) -> str:
        protected_blocks.append(match.group(0))
        return f"@@SLACK_CODE_BLOCK_{len(protected_blocks) - 1}@@"

    def protect_inline(match: re.Match[str]) -> str:
        protected_inline.append(match.group(0))
        return f"@@SLACK_INLINE_CODE_{len(protected_inline) - 1}@@"

    formatted = text

    # Protect code first so we don't rewrite Markdown inside code.
    formatted = FENCED_CODE_BLOCK_PATTERN.sub(protect_block, formatted)
    formatted = INLINE_CODE_PATTERN.sub(protect_inline, formatted)

    # Escape only Slack control chars outside protected code.
    formatted = _escape_slack_control_chars(formatted)

    # Markdown headings -> Slack bold line
    formatted = UNDERLINE_HEADING_PATTERN.sub(_convert_heading, formatted)

    # Markdown links [text](url) -> <url|text>
    formatted = MARKDOWN_LINK_PATTERN.sub(r"<\2|\1>", formatted)

    # HTML-ish tags -> Slack style
    formatted = HTML_BOLD_OPEN.sub("*", formatted)
    formatted = HTML_BOLD_CLOSE.sub("*", formatted)
    formatted = HTML_ITALIC_OPEN.sub("_", formatted)
    formatted = HTML_ITALIC_CLOSE.sub("_", formatted)

    # Standard Markdown bold -> Slack bold
    formatted = BOLD_PATTERN.sub(r"*\1*", formatted)

    # Normalize bullet markers a bit
    formatted = _normalize_lists(formatted)

    # Remove excessive blank lines
    formatted = re.sub(r"\n{3,}", "\n\n", formatted)

    # Restore code
    formatted = _restore_protected_inline(formatted, protected_inline)
    formatted = _restore_protected_blocks(formatted, protected_blocks)

    return formatted.strip()


def _escape_slack_control_chars(text: str) -> str:
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    # Re-open valid Slack entities that we intentionally support later
    text = re.sub(r"&lt;((?:https?://|mailto:)[^|>]+)\|([^>]+)&gt;", r"<\1|\2>", text)
    text = re.sub(r"&lt;(https?://[^>]+)&gt;", r"<\1>", text)
    text = re.sub(r"&lt;(@[A-Z0-9]+)&gt;", r"<\1>", text)
    text = re.sub(r"&lt;(#?[A-Z0-9]+)&gt;", r"<\1>", text)
    text = re.sub(r"&lt;(!(?:channel|here|everyone))&gt;", r"<\1>", text)
    return text


def _convert_heading(match: re.Match[str]) -> str:
    heading_text = match.group(2).strip()
    return f"*{heading_text}*"


def _normalize_lists(text: str) -> str:
    lines = text.splitlines()
    normalized: list[str] = []

    for line in lines:
        stripped = line.strip()

        if re.match(r"^\d+\.\s+", stripped):
            normalized.append(stripped)
            continue

        if re.match(r"^[-*+]\s+", stripped):
            normalized.append(f"• {re.sub(r'^[-*+]\s+', '', stripped)}")
            continue

        normalized.append(line)

    return "\n".join(normalized)


def _restore_protected_inline(text: str, protected_inline: list[str]) -> str:
    for index, block in enumerate(protected_inline):
        text = text.replace(f"@@SLACK_INLINE_CODE_{index}@@", block)
    return text


def _restore_protected_blocks(text: str, protected_blocks: list[str]) -> str:
    for index, block in enumerate(protected_blocks):
        text = text.replace(f"@@SLACK_CODE_BLOCK_{index}@@", block)
    return text
