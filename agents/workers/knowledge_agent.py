from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core.knowledge_rendering import is_knowledge_payload, render_knowledge_payload
from core.state import AgentState


KNOWLEDGE_AGENT_PROMPT = (
    "You are the Knowledge Agent for Jade Games Ltd. "
    "Answer questions about internal documentation, architecture, setup, workflow, and operational guidance that are documented in the knowledge base. "
    "Use the knowledge tools whenever the answer depends on internal docs or project documentation. "
    "Do not invent undocumented behavior. If the documentation is missing or unclear, say so plainly. "
    "Write concise, plain Markdown. "
    "When helpful, mention which document you used."
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
    latest_messages = state.get("messages", [])
    if latest_messages:
        latest_message = latest_messages[-1]
        payload = get_tool_payload(latest_message)
        if payload is not None:
            return render_knowledge_payload(payload)

    latest_user_text = get_latest_user_text(state)
    if not should_render_knowledge_follow_up(latest_user_text):
        return None

    payload = get_latest_knowledge_payload(state)
    if payload is None:
        return None
    return render_knowledge_payload(payload)


def should_render_knowledge_follow_up(user_text: str) -> bool:
    normalized = user_text.strip().lower()
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in REFERENTIAL_KNOWLEDGE_QUERY_PATTERNS)


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
