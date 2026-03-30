from __future__ import annotations


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


def extract_final_text(final_state: dict) -> str:
    messages = final_state.get("messages") or []
    if not messages:
        return "I couldn't generate a response."

    last_message = messages[-1]
    content = getattr(last_message, "content", "")
    return stringify_message_content(content) or "I couldn't generate a response."
