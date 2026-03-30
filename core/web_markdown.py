from __future__ import annotations

import html

import bleach
import markdown


ALLOWED_TAGS = [
    "a",
    "blockquote",
    "br",
    "code",
    "em",
    "h1",
    "h2",
    "h3",
    "h4",
    "hr",
    "li",
    "ol",
    "p",
    "pre",
    "strong",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
]
ALLOWED_ATTRIBUTES = {
    "a": ["href", "title", "rel", "target"],
}
ALLOWED_PROTOCOLS = ["http", "https", "mailto"]


def render_markdown_html(text: str) -> str:
    normalized = (text or "").strip()
    if not normalized:
        return "<p></p>"

    rendered = markdown.markdown(
        normalized,
        extensions=[
            "fenced_code",
            "tables",
            "sane_lists",
            "nl2br",
        ],
        output_format="html5",
    )
    sanitized = bleach.clean(
        rendered,
        tags=ALLOWED_TAGS,
        attributes=ALLOWED_ATTRIBUTES,
        protocols=ALLOWED_PROTOCOLS,
        strip=True,
    )
    return bleach.linkify(sanitized)


def render_plain_text_html(text: str) -> str:
    lines = html.escape((text or "").strip()).splitlines() or [""]
    joined = "<br>".join(line for line in lines)
    return f"<p>{joined}</p>"
