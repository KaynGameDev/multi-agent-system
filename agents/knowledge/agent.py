from __future__ import annotations
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from agents.knowledge.rendering import first_non_empty, is_search_payload
from app.contracts import (
    build_assistant_response,
    tool_invocation_to_tool_call,
)
from app.language import detect_response_language
from app.messages import extract_latest_human_text, stringify_message_content
from app.pending_actions import (
    PendingActionSelectionOption,
    build_pending_action,
    get_pending_action,
    get_pending_action_metadata,
    get_pending_action_selection_phase,
    is_pending_action_active,
    update_pending_action,
)
from app.prompt_loader import join_prompt_layers, load_prompt_sections, load_shared_instruction_text
from app.routing.pending_action_router import PendingActionRouter, PendingActionTurnResult, resolve_owned_pending_action_turn
from app.skill_runtime import build_skill_prompt_context
from app.skills import SkillRegistry
from app.tool_runtime import (
    build_runtime_tool_invocation,
    build_tool_execution_record_for_message,
    extract_first_tool_invocation,
    extract_tool_result_from_message,
)
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
        pending_action_router: PendingActionRouter | None = None,
        agent_name: str = "",
        tool_ids: tuple[str, ...] = (),
    ) -> None:
        self.llm = llm.bind_tools(tools)
        self.skill_registry = skill_registry
        self.pending_action_router = pending_action_router
        self.agent_name = agent_name
        self.tool_ids = tuple(tool_ids)

    def __call__(self, state: AgentState) -> dict:
        rendered_response = build_knowledge_response(
            state,
            agent_name=self.agent_name,
            pending_action_router=self.pending_action_router,
        )
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
        assistant_text = stringify_message_content(getattr(response, "content", ""))
        tool_invocation = extract_first_tool_invocation(
            response,
            source="knowledge_agent",
            reason="The model requested a follow-up knowledge tool call.",
        )
        result: dict[str, Any] = {
            "messages": [response],
            "assistant_response": build_assistant_response(
                kind="text" if assistant_text else ("invoke_tool" if tool_invocation else "text"),
                content=assistant_text,
                tool_invocation=tool_invocation,
            ),
        }
        if tool_invocation is not None:
            result["tool_invocation"] = tool_invocation
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
    skill_prompt = build_skill_prompt_context(
        state,
        skill_registry=skill_registry,
        agent_name=agent_name,
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


def build_knowledge_response(
    state: AgentState,
    *,
    agent_name: str,
    pending_action_router: PendingActionRouter | None = None,
) -> dict[str, Any] | None:
    latest_user_text = extract_latest_human_text(state)
    preferred_language = detect_response_language(latest_user_text)
    pending_action, pending_action_turn = resolve_owned_pending_action_turn(
        state, agent_name=agent_name, pending_action_router=pending_action_router,
    )
    if pending_action is not None:
        follow_up_result = build_knowledge_pending_action_response(
            state,
            pending_action=pending_action,
            pending_action_turn=pending_action_turn,
            preferred_language=preferred_language,
        )
        if follow_up_result is not None:
            return follow_up_result

    return None


def get_tool_result(message, *, messages: list[Any] | None = None) -> dict[str, Any] | None:
    result = extract_tool_result_from_message(
        message,
        messages=messages,
        tool_name="knowledge_documents",
        source="knowledge_agent",
        reason="Knowledge ToolNode returned a result.",
    )
    if result is None:
        return None
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    if not is_knowledge_payload(payload):
        return None
    return result


def is_list_like_payload(payload: dict) -> bool:
    return isinstance(payload.get("documents"), list) and "document" not in payload


def is_read_like_payload(payload: dict) -> bool:
    return isinstance(payload.get("document"), dict) and "content" in payload


def get_latest_tool_result(state: AgentState) -> dict[str, Any] | None:
    messages = state.get("messages", [])
    if not messages:
        return None
    return get_tool_result(messages[-1], messages=messages)


def get_latest_knowledge_result(state: AgentState) -> dict[str, Any] | None:
    for message in reversed(state.get("messages", [])):
        result = get_tool_result(message, messages=state.get("messages", []))
        if result is not None:
            return result
    return None


def render_knowledge_update(state: AgentState, tool_result: dict[str, Any], *, preferred_language: str) -> dict[str, Any] | None:
    payload = tool_result.get("payload") if isinstance(tool_result.get("payload"), dict) else {}
    rendered = render_knowledge_payload(payload, preferred_language=preferred_language)
    if rendered is None:
        return None

    prompt_context = ""
    pending_action = build_knowledge_pending_action(state, tool_result)
    if pending_action is not None:
        prompt_context = str(get_pending_action_metadata(pending_action).get("prompt_context", "")).strip()
    content = rendered
    if prompt_context:
        content = f"{rendered}\n\n{prompt_context}"
    tool_execution_record = None
    messages = state.get("messages", [])
    if messages:
        tool_execution_record = build_tool_execution_record_for_message(
            messages[-1],
            messages=messages,
            tool_name=str(tool_result.get("tool_name", "")).strip() or "knowledge_documents",
            source="knowledge_agent",
            reason="Knowledge ToolNode returned a result.",
        )
    result: dict[str, Any] = {
        "messages": [AIMessage(content=content)],
        "pending_action": pending_action,
        "tool_result": tool_result,
        "assistant_response": build_assistant_response(
            kind="text",
            content=content,
            pending_action=pending_action,
            tool_result=tool_result,
        ),
    }
    if tool_execution_record is not None:
        result["tool_execution_trace"] = [tool_execution_record]
    return result


def build_knowledge_pending_action(state: AgentState, tool_result: dict[str, Any]) -> dict[str, Any] | None:
    payload = tool_result.get("payload") if isinstance(tool_result.get("payload"), dict) else {}
    if payload.get("ok") is False:
        return None

    thread_id = str(state.get("thread_id", "")).strip()
    if not thread_id:
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
        prompt_context = (
            "回复文档编号或文档名即可打开；回复 `取消` 结束。"
            if has_chinese_documents(documents)
            else "Reply with the document number or exact document name to open it, or reply `cancel` to stop."
        )
        return build_pending_action(
            session_id=thread_id,
            action_type="select_knowledge_document",
            requested_by_agent="knowledge_agent",
            summary="Select a document to open from the knowledge results.",
            risk_level="low",
            requires_explicit_approval=False,
            metadata={
                "source_tool_id": resolve_knowledge_source_tool_id(tool_result, payload),
                "prompt_context": prompt_context,
                "selection_options": options,
                "selection_phase": "awaiting_selection",
                "payload": {"document_count": len(options)},
            },
        )

    if is_read_like_payload(payload):
        document = payload.get("document") if isinstance(payload.get("document"), dict) else {}
        if not document:
            return None
        option = build_document_option(document, index=1, include_referential_aliases=True)
        prompt_context = (
            "回复 `详情`、文档名或 `取消`。"
            if prefers_chinese_document(document)
            else "Reply with `details`, the document name, or `cancel`."
        )
        return build_pending_action(
            session_id=thread_id,
            action_type="review_knowledge_document",
            requested_by_agent="knowledge_agent",
            summary="Review the opened knowledge document.",
            risk_level="low",
            requires_explicit_approval=False,
            metadata={
                "source_tool_id": resolve_knowledge_source_tool_id(tool_result, payload),
                "prompt_context": prompt_context,
                "selection_options": [option],
                "selection_phase": "awaiting_selection",
                "payload": {"document_name": build_document_name(document)},
            },
        )
    return None


def build_knowledge_pending_action_response(
    state: AgentState,
    *,
    pending_action: dict[str, Any],
    pending_action_turn: PendingActionTurnResult,
    preferred_language: str,
) -> dict[str, Any] | None:
    selection_phase = get_pending_action_selection_phase(pending_action)
    if selection_phase == "render_after_tool_result":
        latest_tool_result = get_latest_tool_result(state)
        if latest_tool_result is None:
            content = translate_knowledge_text("I couldn't find the latest document payload to reopen.", preferred_language)
            reopened_action = update_pending_action(
                pending_action,
                status="ask_clarification",
                metadata_updates={"selection_phase": "ask_clarification"},
            )
            return {
                "messages": [AIMessage(content=content)],
                "assistant_response": build_assistant_response(
                    kind="text",
                    content=content,
                    pending_action=reopened_action,
                ),
                "pending_action": reopened_action,
            }
        return render_knowledge_update(state, latest_tool_result, preferred_language=preferred_language)

    contract = pending_action_turn.execution_contract
    validation = pending_action_turn.validation
    if contract is None or validation is None:
        fallback_validation = {"reason": "The pending-action decision could not be validated."}
        content = build_knowledge_clarification_text(
            pending_action=pending_action,
            validation=fallback_validation,
            preferred_language=preferred_language,
        )
        return {
            "messages": [AIMessage(content=content)],
            "assistant_response": build_assistant_response(
                kind="await_confirmation",
                content=content,
                pending_action=pending_action,
            ),
            "pending_action": pending_action,
        }

    if validation.get("runtime_action") == "cancel":
        content = translate_knowledge_text("Cancelled the pending document follow-up.", preferred_language)
        return {
            "messages": [AIMessage(content=content)],
            "assistant_response": build_assistant_response(kind="text", content=content),
            "pending_action": None,
        }

    normalized_scope = validation.get("normalized_scope") or {}
    updated_action = update_pending_action(
        pending_action,
        status=str(validation.get("next_status", "ask_clarification")).strip() or "ask_clarification",
        target_scope=normalized_scope or None,
        metadata_updates={"last_contract": dict(contract)},
    )

    if not validation.get("valid"):
        content = build_knowledge_clarification_text(
            pending_action=updated_action,
            validation=validation,
            preferred_language=preferred_language,
        )
        return {
            "messages": [AIMessage(content=content)],
            "assistant_response": build_assistant_response(
                kind="await_confirmation",
                content=content,
                pending_action=updated_action,
            ),
            "pending_action": updated_action,
        }

    runtime_action = validation.get("runtime_action")
    if runtime_action == "select":
        selected_option = validation.get("selected_option") if isinstance(validation.get("selected_option"), dict) else None
        if selected_option is None:
            content = build_knowledge_clarification_text(
                pending_action=updated_action,
                validation=validation,
                preferred_language=preferred_language,
            )
            return {
                "messages": [AIMessage(content=content)],
                "assistant_response": build_assistant_response(
                    kind="await_confirmation",
                    content=content,
                    pending_action=updated_action,
                ),
                "pending_action": updated_action,
            }

        source_tool_id = str(get_pending_action_metadata(pending_action).get("source_tool_id", "")).strip()
        if source_tool_id == TOOL_KNOWLEDGE_READ_DOCUMENT:
            tool_result = get_latest_knowledge_result(state)
            if tool_result is None:
                content = translate_knowledge_text("I couldn't find the latest document payload to reopen.", preferred_language)
                reopened_action = update_pending_action(
                    updated_action,
                    status="ask_clarification",
                    metadata_updates={"selection_phase": "ask_clarification"},
                )
                return {
                    "messages": [AIMessage(content=content)],
                    "assistant_response": build_assistant_response(
                        kind="text",
                        content=content,
                        pending_action=reopened_action,
                    ),
                    "pending_action": reopened_action,
                }
            return render_knowledge_update(state, tool_result, preferred_language=preferred_language)

        document_name = str(selected_option.get("value") or selected_option.get("payload", {}).get("document_name", "")).strip()
        if not document_name:
            content = build_knowledge_clarification_text(
                pending_action=updated_action,
                validation=validation,
                preferred_language=preferred_language,
            )
            return {
                "messages": [AIMessage(content=content)],
                "assistant_response": build_assistant_response(
                    kind="await_confirmation",
                    content=content,
                    pending_action=updated_action,
                ),
                "pending_action": updated_action,
            }

        tool_call_id = f"call_read_knowledge_follow_up_{len(state.get('messages', []))}"
        tool_invocation = build_runtime_tool_invocation(
            "read_knowledge_document",
            {"document_name": document_name},
            source="knowledge_agent",
            reason="The user selected a knowledge document to open.",
            tool_call_id=tool_call_id,
        )
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[tool_invocation_to_tool_call(tool_invocation)],
                )
            ],
            "assistant_response": build_assistant_response(
                kind="invoke_tool",
                content="",
                pending_action=updated_action,
                tool_invocation=tool_invocation,
            ),
            "pending_action": update_pending_action(
                updated_action,
                status="awaiting_confirmation",
                metadata_updates={
                    "selection_phase": "render_after_tool_result",
                    "selected_option": dict(selected_option),
                    "selected_index": int(validation.get("selected_index", 0) or 0),
                    "last_contract": dict(contract),
                },
            ),
            "tool_invocation": tool_invocation,
        }

    content = build_knowledge_clarification_text(
        pending_action=updated_action,
        validation=validation,
        preferred_language=preferred_language,
    )
    return {
        "messages": [AIMessage(content=content)],
        "assistant_response": build_assistant_response(
            kind="await_confirmation",
            content=content,
            pending_action=updated_action,
        ),
        "pending_action": updated_action,
    }


def resolve_knowledge_source_tool_id(tool_result: dict[str, Any], payload: dict[str, Any]) -> str:
    tool_id = str(tool_result.get("tool_id", "")).strip()
    if tool_id:
        return tool_id
    if is_read_like_payload(payload):
        return TOOL_KNOWLEDGE_READ_DOCUMENT
    if is_search_payload(payload):
        return TOOL_KNOWLEDGE_SEARCH_DOCUMENTS
    return TOOL_KNOWLEDGE_LIST_DOCUMENTS


def build_document_option(
    document: dict[str, Any],
    *,
    index: int,
    include_referential_aliases: bool,
) -> PendingActionSelectionOption:
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
    return PendingActionSelectionOption(
        id=title or path or str(index),
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


def build_knowledge_clarification_text(
    *,
    pending_action: dict[str, Any],
    validation: dict[str, Any],
    preferred_language: str,
) -> str:
    summary = str(pending_action.get("summary", "")).strip()
    reason = str(validation.get("reason", "")).strip()
    prompt_context = str(get_pending_action_metadata(pending_action).get("prompt_context", "")).strip()
    if preferred_language == "zh":
        parts = [f"我还不能确定下一步：{summary}"]
        if reason:
            parts.append(f"原因：{reason}")
        if prompt_context:
            parts.append(prompt_context)
        return "\n\n".join(parts).strip()

    parts = [f"I still can't determine the next step: {summary}"]
    if reason:
        parts.append(f"Reason: {reason}")
    if prompt_context:
        parts.append(prompt_context)
    return "\n\n".join(parts).strip()


def translate_knowledge_text(text: str, preferred_language: str) -> str:
    if preferred_language != "zh":
        return text

    translations = {
        "Cancelled the pending document follow-up.": "已取消待处理的文档跟进。",
        "I couldn't find the latest document payload to reopen.": "我找不到最近的文档结果，无法继续展开。",
    }
    return translations.get(text, text)
