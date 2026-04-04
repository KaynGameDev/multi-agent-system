from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agents.knowledge.rendering import first_non_empty, is_search_payload
from app.language import detect_response_language
from app.pending_interactions import (
    PendingInteractionOption,
    build_selection_interaction,
    get_pending_interaction,
    match_pending_interaction_reply,
)
from app.prompt_loader import join_prompt_layers, load_prompt_sections, load_shared_instruction_text
from app.skills import SkillRegistry
from app.tool_registry import (
    TOOL_KNOWLEDGE_LIST_DOCUMENTS,
    TOOL_KNOWLEDGE_READ_DOCUMENT,
    TOOL_KNOWLEDGE_SEARCH_DOCUMENTS,
    build_agent_tool_prompt,
)
from agents.knowledge.rendering import is_knowledge_payload, render_knowledge_payload
from app.state import AgentState

PROMPT_PATH = "agents/knowledge/AGENT.md"


class KnowledgeAgentNode:
    def __init__(
        self,
        llm,
        tools: list,
        *,
        skill_registry: SkillRegistry | None = None,
        agent_name: str = "",
        tool_ids: tuple[str, ...] = (),
    ) -> None:
        self.llm = llm.bind_tools(tools)
        self.skill_registry = skill_registry
        self.agent_name = agent_name
        self.tool_ids = tuple(tool_ids)

    def __call__(self, state: AgentState) -> dict:
        rendered_response = build_knowledge_response(state, agent_name=self.agent_name)
        if rendered_response is not None:
            return rendered_response

        messages = [
            SystemMessage(
                content=build_knowledge_prompt(
                    state,
                    skill_registry=self.skill_registry,
                    agent_name=self.agent_name,
                    tool_ids=self.tool_ids,
                )
            ),
            *state["messages"],
        ]
        response = self.llm.invoke(messages)
        result: dict[str, Any] = {"messages": [response]}
        latest_payload = get_latest_tool_payload(state)
        if latest_payload is not None:
            interaction = build_knowledge_pending_interaction(latest_payload)
            if interaction is not None:
                result["pending_interaction"] = interaction
        return result


def build_knowledge_prompt(
    state: AgentState,
    *,
    skill_registry: SkillRegistry | None = None,
    agent_name: str = "",
    tool_ids: tuple[str, ...] = (),
) -> str:
    sections = load_prompt_sections(
        PROMPT_PATH,
        required_sections=("role", "responsibilities", "tool_usage", "boundaries", "output"),
    )
    skill_prompt = (
        skill_registry.build_prompt_layers(
            state.get("resolved_skill_ids", []),
            agent_name=agent_name,
            context_paths=state.get("context_paths", []),
        )
        if skill_registry is not None and agent_name
        else ""
    )
    tool_prompt = build_agent_tool_prompt(tool_ids)
    return join_prompt_layers(
        load_shared_instruction_text(),
        sections["role"],
        sections["responsibilities"],
        sections["tool_usage"],
        tool_prompt,
        skill_prompt,
        sections["boundaries"],
        sections["output"],
    )


def build_knowledge_response(state: AgentState, *, agent_name: str) -> dict[str, Any] | None:
    latest_user_text = get_latest_user_text(state)
    preferred_language = detect_response_language(latest_user_text)
    interaction = get_pending_interaction(state)
    if interaction and interaction.get("owner_agent") == agent_name:
        if str(interaction.get("status", "")).strip() == "render_after_tool_result":
            latest_payload = get_latest_tool_payload(state)
            if latest_payload is not None:
                return render_knowledge_update(latest_payload, preferred_language=preferred_language)
        follow_up_result = build_knowledge_interaction_response(
            state,
            interaction=interaction,
            preferred_language=preferred_language,
        )
        if follow_up_result is not None:
            return follow_up_result

    latest_payload = get_latest_tool_payload(state)
    if latest_payload is not None:
        return render_knowledge_update(latest_payload, preferred_language=preferred_language)
    return None


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


def get_latest_tool_payload(state: AgentState) -> dict | None:
    messages = state.get("messages", [])
    if not messages:
        return None
    return get_tool_payload(messages[-1])


def get_latest_knowledge_payload(state: AgentState) -> dict | None:
    for message in reversed(state.get("messages", [])):
        payload = get_tool_payload(message)
        if payload is not None:
            return payload
    return None


def render_knowledge_update(payload: dict, *, preferred_language: str) -> dict[str, Any] | None:
    rendered = render_knowledge_payload(payload, preferred_language=preferred_language)
    if rendered is None:
        return None

    prompt_context = ""
    interaction = build_knowledge_pending_interaction(payload)
    if interaction is not None:
        prompt_context = str(interaction.get("prompt_context", "")).strip()
    content = rendered
    if prompt_context:
        content = f"{rendered}\n\n{prompt_context}"
    return {
        "messages": [AIMessage(content=content)],
        "pending_interaction": interaction,
    }


def build_knowledge_interaction_response(
    state: AgentState,
    *,
    interaction: dict[str, Any],
    preferred_language: str,
) -> dict[str, Any] | None:
    latest_user_text = get_latest_user_text(state)
    outcome = match_pending_interaction_reply(interaction, latest_user_text)
    if outcome is None:
        return None

    if outcome["action"] == "cancel":
        return {
            "messages": [AIMessage(content=translate_interaction_text("Cancelled the pending document follow-up.", preferred_language))],
            "pending_interaction": None,
        }

    if outcome["action"] != "select":
        return None

    source_tool_id = str(interaction.get("source_tool_id", "")).strip()
    if source_tool_id == TOOL_KNOWLEDGE_READ_DOCUMENT:
        payload = get_latest_knowledge_payload(state)
        if payload is None:
            return {
                "messages": [AIMessage(content=translate_interaction_text("I couldn't find the latest document payload to reopen.", preferred_language))],
                "pending_interaction": None,
            }
        return render_knowledge_update(payload, preferred_language=preferred_language)

    option = outcome["option"]
    document_name = str(option.get("value") or option.get("payload", {}).get("document_name", "")).strip()
    if not document_name:
        return None
    tool_call_id = f"call_read_knowledge_follow_up_{len(state.get('messages', []))}"
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_knowledge_document",
                        "args": {"document_name": document_name},
                        "id": tool_call_id,
                    }
                ],
            )
        ],
        "pending_interaction": {
            "kind": "selection",
            "owner_agent": "knowledge_agent",
            "source_tool_id": TOOL_KNOWLEDGE_READ_DOCUMENT,
            "status": "render_after_tool_result",
            "prompt_context": "",
            "options": [],
            "accepted_replies": [],
            "cancel_replies": [],
            "payload": {"document_name": document_name},
        },
    }


def build_knowledge_pending_interaction(payload: dict) -> dict[str, Any] | None:
    if payload.get("ok") is False:
        return None

    if is_list_like_payload(payload) or is_search_payload(payload):
        documents = payload.get("documents") if isinstance(payload.get("documents"), list) else []
        if not documents:
            return None
        single_option = len(documents) == 1
        options = [
            build_document_option(document, index=index, include_referential_aliases=single_option)
            for index, document in enumerate(documents, start=1)
            if isinstance(document, dict)
        ]
        if not options:
            return None
        return build_selection_interaction(
            owner_agent="knowledge_agent",
            source_tool_id=TOOL_KNOWLEDGE_SEARCH_DOCUMENTS if is_search_payload(payload) else TOOL_KNOWLEDGE_LIST_DOCUMENTS,
            prompt_context="回复文档编号或文档名即可打开；回复 `取消` 结束。" if has_chinese_documents(documents) else "Reply with the document number or exact document name to open it, or reply `cancel` to stop.",
            options=options,
            payload={"document_count": len(options)},
        )

    if is_read_like_payload(payload):
        document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
        if not document:
            return None
        return build_selection_interaction(
            owner_agent="knowledge_agent",
            source_tool_id=TOOL_KNOWLEDGE_READ_DOCUMENT,
            prompt_context="回复 `详情`、文档名或 `取消`。" if prefers_chinese_document(document) else "Reply with `details`, the document name, or `cancel`.",
            options=[build_document_option(document, index=1, include_referential_aliases=True)],
            payload={"document_name": build_document_name(document)},
        )
    return None


def build_document_option(
    document: dict[str, Any],
    *,
    index: int,
    include_referential_aliases: bool,
) -> PendingInteractionOption:
    title = build_document_name(document)
    path = first_non_empty(document.get("path"))
    aliases = [str(index)]
    if title:
        aliases.append(title)
    if path:
        aliases.append(path)
        aliases.append(path.rsplit("/", 1)[-1])
        aliases.append(path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    if include_referential_aliases:
        aliases.extend(["that one", "this one", "details", "detail", "那个", "这个", "详情"])
    return PendingInteractionOption(
        label=title or path or f"Document {index}",
        aliases=list(dict.fromkeys(alias for alias in aliases if alias)),
        value=title or path,
        payload={"document_name": title or path or ""},
    )


def build_document_name(document: dict[str, Any]) -> str:
    return first_non_empty(document.get("name"), document.get("title"), document.get("path"))


def has_chinese_documents(documents: list[dict[str, Any]]) -> bool:
    return any(prefers_chinese_document(document) for document in documents if isinstance(document, dict))


def prefers_chinese_document(document: dict[str, Any]) -> bool:
    text = " ".join(
        str(value)
        for value in (document.get("title"), document.get("name"), document.get("path"))
        if isinstance(value, str) and value.strip()
    )
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def translate_interaction_text(text: str, preferred_language: str) -> str:
    if preferred_language != "zh":
        return text

    translations = {
        "Cancelled the pending document follow-up.": "已取消待处理的文档跟进。",
        "I couldn't find the latest document payload to reopen.": "我找不到最近的文档结果，无法继续展开。",
    }
    return translations.get(text, text)
