from __future__ import annotations

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types


def build_text_event(
    context: InvocationContext,
    *,
    author: str,
    text: str,
) -> Event:
    return Event(
        invocationId=context.invocation_id,
        author=author,
        branch=context.branch,
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=text)],
        ),
    )


def extract_user_text(context: InvocationContext) -> str:
    user_content = context.user_content
    if user_content is None:
        return ""
    parts = user_content.parts or []
    fragments = [part.text for part in parts if getattr(part, "text", None)]
    return "\n".join(fragment for fragment in fragments if fragment).strip()

