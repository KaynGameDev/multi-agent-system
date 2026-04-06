from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.language import detect_response_language
from app.pending_actions import (
    build_pending_action,
    get_pending_action,
    is_pending_action_active,
    resolve_pending_action_reply,
    update_pending_action,
)
from app.prompt_loader import join_prompt_layers, load_prompt_sections, load_shared_instruction_text
from app.skills import SkillRegistry
from app.state import AgentState
from app.tool_registry import TOOL_KNOWLEDGE_WRITE_MARKDOWN, build_agent_tool_prompt

PROMPT_PATH = "agents/knowledge_base_builder/AGENT.md"


class KnowledgeBaseBuilderAgentNode:
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
        pending_action = get_pending_action(state)
        if (
            pending_action
            and pending_action.get("requested_by_agent") == self.agent_name
            and is_pending_action_active(pending_action)
        ):
            pending_action_response = build_pending_action_response(state, pending_action)
            if pending_action_response is not None:
                return pending_action_response

        rendered_response = build_knowledge_base_builder_response(state)
        if rendered_response is not None:
            return rendered_response

        messages = [
            SystemMessage(
                content=build_knowledge_base_builder_prompt(
                    state,
                    skill_registry=self.skill_registry,
                    agent_name=self.agent_name,
                    tool_ids=self.tool_ids,
                )
            ),
            *state["messages"],
        ]
        response = self.llm.invoke(messages)
        return {"messages": [response]}


def build_knowledge_base_builder_prompt(
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


def build_knowledge_base_builder_response(state: AgentState) -> dict[str, Any] | None:
    messages = state.get("messages", [])
    if not messages:
        return None

    latest_message = messages[-1]
    payload = get_builder_tool_payload(latest_message)
    if payload is None:
        return None

    if payload.get("knowledge_mutation") == "write_markdown":
        rendered = render_write_knowledge_payload(state, payload)
        pending_action = build_builder_pending_action(state, latest_message, payload)
        return {
            "messages": [AIMessage(content=rendered)],
            "pending_action": pending_action,
            "execution_contract": None,
        }

    return None


def get_builder_tool_payload(message) -> dict | None:
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
    return payload


def render_write_knowledge_payload(state: AgentState, payload: dict) -> str:
    relative_path = str(payload.get("relative_path", "")).strip()
    error = str(payload.get("error", "")).strip()
    preferred_language = detect_response_language(get_latest_user_text(state))

    if payload.get("ok") is True:
        if preferred_language == "zh":
            action = "已更新" if payload.get("overwritten") else "已创建"
            return f"{action}知识库草稿：`{relative_path}`"
        action = "Updated" if payload.get("overwritten") else "Created"
        return f"{action} knowledge-base draft: `{relative_path}`"

    if payload.get("requires_confirmation") is True:
        target_exists = payload.get("target_exists") is True
        if preferred_language == "zh":
            action_text = "覆盖更新" if target_exists else "创建草稿"
            return (
                "这次知识库文件操作需要你先确认，我还没有实际写入。\n\n"
                f"- 目标路径：`{relative_path}`\n"
                f"- 计划动作：{action_text}\n\n"
                "你可以直接自然回复，例如“继续”“可以，执行吧”“先看 diff”“取消”。"
            )
        action_text = "overwrite the draft" if target_exists else "create the draft"
        return (
            "This knowledge-base file action is waiting for confirmation. Nothing has been written yet.\n\n"
            f"- Target path: `{relative_path}`\n"
            f"- Planned action: {action_text}\n\n"
            "You can reply naturally, for example: `continue`, `go ahead`, `show me the diff first`, or `cancel`."
        )

    if error:
        if preferred_language == "zh":
            return f"知识库写入失败：{error}"
        return f"Knowledge-base write failed: {error}"

    if preferred_language == "zh":
        return "知识库写入未完成。"
    return "The knowledge-base write did not complete."


def build_builder_pending_action(state: AgentState, message, payload: dict) -> dict[str, Any] | None:
    if payload.get("knowledge_mutation") != "write_markdown":
        return None
    if payload.get("requires_confirmation") is not True:
        return None

    relative_path = str(payload.get("relative_path", "")).strip()
    target_exists = payload.get("target_exists") is True
    tool_request = find_builder_tool_request(state.get("messages", []), getattr(message, "tool_call_id", ""))
    if tool_request is None:
        return None

    metadata = {
        "source_tool_id": TOOL_KNOWLEDGE_WRITE_MARKDOWN,
        "relative_path": relative_path,
        "absolute_path": str(payload.get("absolute_path", "")).strip(),
        "target_exists": target_exists,
        "tool_name": str(tool_request.get("name", "")).strip(),
        "tool_args": dict(tool_request.get("args") or {}),
    }
    return build_pending_action(
        session_id=str(state.get("thread_id", "")).strip(),
        action_type="write_knowledge_markdown",
        requested_by_agent="knowledge_base_builder_agent",
        summary=f"Write knowledge-base draft to `{relative_path}`.",
        target_scope={"files": [relative_path]} if relative_path else {},
        risk_level="high" if target_exists else "medium",
        requires_explicit_approval=True,
        metadata=metadata,
    )


def build_pending_action_response(state: AgentState, pending_action: dict[str, Any]) -> dict[str, Any] | None:
    latest_user_text = get_latest_user_text(state)
    if not latest_user_text.strip():
        return None

    preferred_language = detect_response_language(latest_user_text)
    resolution = resolve_pending_action_reply(pending_action, latest_user_text)
    contract = resolution["contract"]
    validation = resolution["validation"]

    if validation.get("runtime_action") == "cancel":
        return {
            "messages": [AIMessage(content=translate_builder_text("Cancelled the pending knowledge-base write.", preferred_language))],
            "pending_action": None,
            "execution_contract": None,
        }

    normalized_scope = validation.get("normalized_scope") or {}
    next_status = str(validation.get("next_status", "ask_clarification")).strip() or "ask_clarification"
    updated_action = update_pending_action(
        pending_action,
        status=next_status,
        target_scope=normalized_scope or None,
        metadata_updates={"last_contract": dict(contract)},
    )

    if not validation.get("valid"):
        return {
            "messages": [
                AIMessage(
                    content=build_pending_action_clarification(
                        updated_action,
                        validation,
                        preferred_language=preferred_language,
                    )
                )
            ],
            "pending_action": updated_action,
            "execution_contract": None,
        }

    runtime_action = validation.get("runtime_action")

    if runtime_action == "request_revision":
        return {
            "messages": [
                AIMessage(
                    content=build_pending_action_revision_response(
                        updated_action,
                        contract,
                        preferred_language=preferred_language,
                    )
                )
            ],
            "pending_action": updated_action,
            "execution_contract": None,
        }

    if runtime_action != "execute":
        return {
            "messages": [
                AIMessage(
                    content=build_pending_action_clarification(
                        updated_action,
                        validation,
                        preferred_language=preferred_language,
                    )
                )
            ],
            "pending_action": updated_action,
            "execution_contract": None,
        }

    retry_message = build_builder_retry_tool_call(state, updated_action)
    if retry_message is None:
        return {
            "messages": [AIMessage(content=translate_builder_text("I could not reconstruct the pending write request to execute it safely.", preferred_language))],
            "pending_action": updated_action,
            "execution_contract": None,
        }

    return {
        "messages": [retry_message],
        "pending_action": updated_action,
        "execution_contract": contract,
    }


def build_builder_retry_tool_call(state: AgentState, pending_action: dict[str, Any]) -> AIMessage | None:
    metadata = pending_action.get("metadata") if isinstance(pending_action.get("metadata"), dict) else {}
    tool_name = str(metadata.get("tool_name", "")).strip()
    tool_args = metadata.get("tool_args")
    if not tool_name or not isinstance(tool_args, dict):
        return None

    tool_call_id = f"call_retry_write_{len(state.get('messages', []))}"
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": dict(tool_args),
                "id": tool_call_id,
            }
        ],
    )


def build_pending_action_clarification(
    pending_action: dict[str, Any],
    validation: dict[str, Any],
    *,
    preferred_language: str,
) -> str:
    summary = str(pending_action.get("summary", "")).strip()
    reason = str(validation.get("reason", "")).strip()
    if preferred_language == "zh":
        details = f"\n原因：{reason}" if reason else ""
        return (
            f"我还不能确定是否执行这项待确认操作：{summary}{details}\n\n"
            "你可以直接自然回复，例如“继续”“执行吧”“先看 diff”，或“取消”。"
        )
    details = f"\nReason: {reason}" if reason else ""
    return (
        f"I still cannot execute this pending action deterministically: {summary}{details}\n\n"
        "You can reply naturally, for example: `continue`, `go ahead`, `show me the diff first`, or `cancel`."
    )


def build_pending_action_revision_response(
    pending_action: dict[str, Any],
    contract: dict[str, Any],
    *,
    preferred_language: str,
) -> str:
    requested_outputs = list(contract.get("requested_outputs") or [])
    if any(output in {"diff", "preview"} for output in requested_outputs):
        rendered_diff = render_pending_write_diff(pending_action)
        if preferred_language == "zh":
            return (
                "这是本次知识库写入的 diff 预览，尚未实际写入。\n\n"
                f"{rendered_diff}\n\n"
                "如果可以继续，直接回复“继续”或“执行吧”；如果不要执行，回复“取消”。"
            )
        return (
            "Here is the diff preview for the pending knowledge-base write. Nothing has been written yet.\n\n"
            f"{rendered_diff}\n\n"
            "If you want to continue, reply naturally with something like `continue` or `go ahead`. Reply `cancel` to stop."
        )

    if preferred_language == "zh":
        return "我已记录这次修改请求，但还不能直接执行。请继续说明限制，或直接确认执行。"
    return "I recorded the requested modification, but I still need a clearer execution confirmation."


def render_pending_write_diff(pending_action: dict[str, Any]) -> str:
    metadata = pending_action.get("metadata") if isinstance(pending_action.get("metadata"), dict) else {}
    tool_args = metadata.get("tool_args") if isinstance(metadata.get("tool_args"), dict) else {}
    relative_path = str(metadata.get("relative_path", "")).strip()
    absolute_path = str(metadata.get("absolute_path", "")).strip()
    proposed_content = str(tool_args.get("content", ""))

    current_content = ""
    if absolute_path:
        absolute_candidate = Path(absolute_path)
        if absolute_candidate.exists():
            current_content = absolute_candidate.read_text(encoding="utf-8")

    diff_lines = list(
        difflib.unified_diff(
            current_content.splitlines(),
            proposed_content.splitlines(),
            fromfile=f"a/{relative_path}" if current_content else "/dev/null",
            tofile=f"b/{relative_path or 'proposed.md'}",
            lineterm="",
        )
    )
    rendered = "\n".join(diff_lines) or "(no textual diff available)"
    return f"```diff\n{rendered}\n```"


def find_builder_tool_request(messages: list[Any], tool_call_id: str) -> dict[str, Any] | None:
    normalized_tool_call_id = str(tool_call_id or "").strip()
    if not normalized_tool_call_id:
        return None

    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        for tool_call in getattr(message, "tool_calls", []) or []:
            if not isinstance(tool_call, dict):
                continue
            if str(tool_call.get("id", "")).strip() != normalized_tool_call_id:
                continue
            if str(tool_call.get("name", "")).strip() != "write_knowledge_markdown_document":
                continue
            return tool_call
    return None


def translate_builder_text(text: str, preferred_language: str) -> str:
    if preferred_language != "zh":
        return text

    translations = {
        "Cancelled the pending knowledge-base write.": "已取消待处理的知识库写入操作。",
        "I could not reconstruct the pending write request to execute it safely.": "我无法安全地还原这次待执行的知识库写入请求。",
    }
    return translations.get(text, text)


def get_latest_user_text(state: AgentState) -> str:
    messages = state.get("messages", [])
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if isinstance(message, HumanMessage) and isinstance(content, str):
            return content
    return ""
