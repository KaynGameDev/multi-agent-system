from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.contracts import extract_assistant_response_text

THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def stringify_message_content(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()

    return str(content)


def strip_internal_reasoning(text: str) -> str:
    cleaned = THINK_BLOCK_PATTERN.sub("", str(text or ""))
    return cleaned.strip()


def normalize_text_for_routing_context(text: str) -> str:
    stripped = strip_internal_reasoning(text)
    if not stripped:
        return ""
    return " ".join(part.strip() for part in stripped.splitlines() if part.strip()).strip()


def render_message_for_routing_context(message) -> str:
    if isinstance(message, ToolMessage):
        return ""

    text = normalize_text_for_routing_context(
        stringify_message_content(getattr(message, "content", ""))
    )
    if not text:
        return ""

    if isinstance(message, HumanMessage):
        return f"user: {text}"
    if isinstance(message, AIMessage):
        return f"assistant: {text}"

    message_type = str(getattr(message, "type", "")).strip().lower()
    if message_type == "human":
        return f"user: {text}"
    if message_type in {"ai", "assistant"}:
        return f"assistant: {text}"
    return text


def extract_final_text(final_state: dict) -> str:
    assistant_response = final_state.get("assistant_response")
    assistant_text = extract_assistant_response_text(assistant_response)
    if assistant_text:
        return assistant_text

    messages = final_state.get("messages") or []
    if not messages:
        return "I couldn't generate a response."

    last_message = messages[-1]
    content = getattr(last_message, "content", "")
    return stringify_message_content(content) or "I couldn't generate a response."


def extract_latest_human_text(state: dict) -> str:
    messages = state.get("messages") or []
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return stringify_message_content(getattr(message, "content", ""))
    return ""
