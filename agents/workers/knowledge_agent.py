from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core.language import LANGUAGE_MATCHING_PROMPT, detect_response_language
from core.knowledge_rendering import is_knowledge_payload, render_knowledge_payload
from core.state import AgentState


KNOWLEDGE_AGENT_PROMPT = (
    "You are the Knowledge Agent for Jade Games Ltd. "
    "Answer questions about internal documentation, architecture, setup, workflow, and operational guidance that are documented in the knowledge base. "
    "Use the knowledge tools whenever the answer depends on internal docs or project documentation. "
    "Do not invent undocumented behavior. If the documentation is missing or unclear, say so plainly. "
    "After using a knowledge tool, answer the user's question directly instead of repeating raw tool output unless the user explicitly asks to see the document or excerpt. "
    "For spreadsheet or CSV-style documents, extract the relevant rules, limits, steps, or conclusions instead of reciting raw rows. "
    "Write concise, plain Markdown. "
    "When helpful, mention which document you used. "
    f"{LANGUAGE_MATCHING_PROMPT}"
)

REFERENTIAL_KNOWLEDGE_QUERY_PATTERNS = (
    r"\bwhat are those docs\b",
    r"\bwhat are those documents\b",
    r"\bshow that doc\b",
    r"\bshow that document\b",
    r"\bread that\b",
    r"\bread more\b",
    r"\bshow more\b",
    r"^details?\??$",
    r"those docs",
    r"those documents",
    r"that doc",
    r"that document",
    r"那些文档",
    r"这些文档",
    r"那个文档",
    r"这个文档",
    r"展开",
    r"详情",
)

LIST_KNOWLEDGE_QUERY_PATTERNS = (
    r"\blist\b.*\bdocs?\b",
    r"\bshow\b.*\bdocs?\b",
    r"\bwhat docs\b",
    r"\bwhat documents\b",
    r"\bavailable docs?\b",
    r"\bavailable documents\b",
    r"有哪些文档",
    r"文档列表",
)

READ_KNOWLEDGE_QUERY_PATTERNS = (
    r"\bread\b",
    r"\bshow\b.*\bdoc",
    r"\bopen\b.*\bdoc",
    r"\bsection\b",
    r"\bdocument\b",
    r"\bfull text\b",
    r"\boriginal text\b",
    r"\braw\b",
    r"\bexcerpt\b",
    r"\bshow me\b",
    r"\bread that\b",
    r"\bshow that\b",
    r"\bdetails?\b",
    r"\b全文\b",
    r"\b原文\b",
    r"\b内容\b",
    r"\b读一下\b",
    r"\b展示\b",
)


class KnowledgeAgentNode:
    def __init__(self, llm, tools: list) -> None:
        self.llm = llm.bind_tools(tools)

    def __call__(self, state: AgentState) -> dict:
        rendered_response = build_knowledge_response(state)
        if rendered_response is not None:
            return {"messages": [AIMessage(content=rendered_response)]}

        messages = [SystemMessage(content=KNOWLEDGE_AGENT_PROMPT), *state["messages"]]
        response = self.llm.invoke(messages)
        return {"messages": [response]}


def build_knowledge_response(state: AgentState) -> str | None:
    latest_user_text = get_latest_user_text(state)
    preferred_language = detect_response_language(latest_user_text)
    latest_messages = state.get("messages", [])
    if latest_messages:
        latest_message = latest_messages[-1]
        payload = get_tool_payload(latest_message)
        if payload is not None and should_render_latest_tool_payload(latest_user_text, payload):
            return render_knowledge_payload(payload, preferred_language=preferred_language)

    if not should_render_knowledge_follow_up(latest_user_text):
        return None

    payload = get_latest_knowledge_payload(state)
    if payload is None:
        return None
    return render_knowledge_payload(payload, preferred_language=preferred_language)


def should_render_knowledge_follow_up(user_text: str) -> bool:
    normalized = user_text.strip().lower()
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in REFERENTIAL_KNOWLEDGE_QUERY_PATTERNS)


def should_render_latest_tool_payload(user_text: str, payload: dict) -> bool:
    normalized = user_text.strip().lower()
    if not normalized:
        return False

    if payload.get("ok") is False:
        return True
    if is_list_like_payload(payload):
        return any(re.search(pattern, normalized) for pattern in LIST_KNOWLEDGE_QUERY_PATTERNS)
    if is_read_like_payload(payload):
        return any(re.search(pattern, normalized) for pattern in READ_KNOWLEDGE_QUERY_PATTERNS)
    return False


def get_latest_user_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return stringify_message_content(message.content)
    return ""


def stringify_message_content(content) -> str:
    if isinstance(content, str):
        return content

    parts: list[str] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
    return " ".join(parts).strip()


def get_latest_knowledge_payload(state: AgentState) -> dict | None:
    for message in reversed(state.get("messages", [])):
        payload = get_tool_payload(message)
        if payload is not None:
            return payload
    return None


def get_tool_payload(message) -> dict | None:
    if not isinstance(message, ToolMessage):
        return None
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if not is_knowledge_payload(payload):
        return None
    return payload


def is_list_like_payload(payload: dict) -> bool:
    return isinstance(payload.get("documents"), list) and "document" not in payload


def is_read_like_payload(payload: dict) -> bool:
    return isinstance(payload.get("document"), dict) and "content" in payload
