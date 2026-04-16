from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from app.contracts import build_assistant_response
from app.language import detect_response_language
from app.messages import extract_latest_human_text, stringify_message_content
from app.model_request import build_model_request_messages
from app.pending_actions import (
    build_pending_action,
    build_write_knowledge_approval_payload,
    compute_approval_payload_hash,
    get_pending_action,
    is_pending_action_active,
    update_pending_action,
)
from app.prompt_loader import join_prompt_layers, load_prompt_sections, load_shared_instruction_text
from app.routing.pending_action_router import PendingActionRouter, PendingActionTurnResult, resolve_owned_pending_action_turn
from app.skill_runtime import build_skill_prompt_context
from app.skills import SkillRegistry
from app.state import AgentState
from app.tool_runtime import (
    build_pending_action_retry_tool_call,
    build_tool_execution_record_for_message,
    extract_first_tool_invocation,
    extract_tool_result_from_message,
    find_matching_tool_call_request,
    get_persisted_tool_result,
    normalize_runtime_tool_invocation,
)
from app.tool_registry import TOOL_KNOWLEDGE_WRITE_MARKDOWN, build_agent_tool_prompt

PROMPT_PATH = "agents/knowledge_base_builder/AGENT.md"


class KnowledgeBaseBuilderAgentNode:
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
        pending_action, pending_action_turn = resolve_owned_pending_action_turn(
            state, agent_name=self.agent_name, pending_action_router=self.pending_action_router,
        )
        if pending_action is not None:
            pending_action_response = build_pending_action_response(
                state,
                pending_action,
                pending_action_turn=pending_action_turn,
            )
            if pending_action_response is not None:
                return pending_action_response

        rendered_response = build_knowledge_base_builder_response(state)
        if rendered_response is not None:
            return rendered_response

        messages = build_model_request_messages(
            system_prompt=build_knowledge_base_builder_prompt(
                state,
                skill_registry=self.skill_registry,
                agent_name=self.agent_name,
                tool_ids=self.tool_ids,
            ),
            transcript_messages=state["messages"],
        )
        response = self.llm.invoke(messages)
        assistant_text = stringify_message_content(getattr(response, "content", ""))
        tool_invocation = extract_first_tool_invocation(
            response,
            source="knowledge_base_builder_agent",
            reason="The model requested a follow-up knowledge-base tool call.",
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


def build_knowledge_base_builder_response(state: AgentState) -> dict[str, Any] | None:
    messages = state.get("messages", [])
    latest_message = messages[-1] if messages else None
    tool_result = (
        get_builder_tool_result(latest_message, messages=messages)
        if latest_message is not None
        else None
    )
    if tool_result is None:
        tool_result = get_persisted_tool_result(
            state,
            source="knowledge_base_builder_agent",
            reason="Using persisted knowledge-base tool state after transcript rehydration.",
        )
    if tool_result is None:
        return None
    payload = tool_result.get("payload") if isinstance(tool_result.get("payload"), dict) else {}

    if str(payload.get("knowledge_mutation", "")).strip() == "write_markdown":
        rendered = render_write_knowledge_payload(state, payload)
        pending_action = build_builder_pending_action(state, latest_message, tool_result, payload)
        assistant_kind = "await_confirmation" if payload.get("requires_confirmation") is True else "tool_result"
        result: dict[str, Any] = {
            "messages": [AIMessage(content=rendered)],
            "assistant_response": build_assistant_response(
                kind=assistant_kind,
                content=rendered,
                pending_action=pending_action,
                tool_result=tool_result,
            ),
            "pending_action": pending_action,
            "tool_result": tool_result,
            "execution_contract": None,
        }
        tool_execution_record = None
        if latest_message is not None:
            tool_execution_record = build_tool_execution_record_for_message(
                latest_message,
                messages=messages,
                tool_name=str(tool_result.get("tool_name", "")).strip() or "write_knowledge_markdown_document",
                source="knowledge_base_builder_agent",
                reason="Knowledge-base builder ToolNode returned a result.",
            )
        if tool_execution_record is not None:
            result["tool_execution_trace"] = [tool_execution_record]
        return result

    return None


def get_builder_tool_result(message, *, messages: list[Any] | None = None) -> dict[str, Any] | None:
    return extract_tool_result_from_message(
        message,
        messages=messages,
        tool_name="write_knowledge_markdown_document",
        source="knowledge_base_builder_agent",
        reason="Knowledge-base builder ToolNode returned a result.",
    )


def render_write_knowledge_payload(state: AgentState, payload: dict) -> str:
    relative_path = str(payload.get("relative_path", "")).strip()
    error = str(payload.get("error", "")).strip()
    preferred_language = detect_response_language(extract_latest_human_text(state))

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


def build_builder_pending_action(
    state: AgentState,
    message,
    tool_result: dict[str, Any],
    payload: dict,
) -> dict[str, Any] | None:
    if payload.get("knowledge_mutation") != "write_markdown":
        return None
    if payload.get("requires_confirmation") is not True:
        return None

    relative_path = str(payload.get("relative_path", "")).strip()
    target_exists = payload.get("target_exists") is True
    tool_request = find_matching_tool_call_request(
        state.get("messages", []),
        getattr(message, "tool_call_id", ""),
        tool_name="write_knowledge_markdown_document",
    )
    if tool_request is None:
        return None

    tool_invocation = normalize_runtime_tool_invocation(
        tool_request,
        source="knowledge_base_builder_agent",
        reason="The pending knowledge-base write requires explicit confirmation.",
    )

    metadata = {
        "source_tool_id": TOOL_KNOWLEDGE_WRITE_MARKDOWN,
        "relative_path": relative_path,
        "absolute_path": str(payload.get("absolute_path", "")).strip(),
        "target_exists": target_exists,
        "tool_name": str(tool_request.get("name", "")).strip(),
        "tool_args": dict(tool_request.get("args") or {}),
        "tool_call_id": str(tool_request.get("id", "")).strip(),
        "tool_invocation": tool_invocation,
        "tool_result": dict(tool_result),
    }
    approval_payload = build_write_knowledge_approval_payload(
        relative_path=relative_path,
        content=str((tool_request.get("args") or {}).get("content", "")),
        overwrite=bool((tool_request.get("args") or {}).get("overwrite", False)),
    )
    metadata["approval_payload"] = approval_payload
    metadata["approval_payload_hash"] = compute_approval_payload_hash(approval_payload)
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


def build_pending_action_response(
    state: AgentState,
    pending_action: dict[str, Any],
    *,
    pending_action_turn: PendingActionTurnResult,
) -> dict[str, Any] | None:
    latest_user_text = extract_latest_human_text(state)
    preferred_language = detect_response_language(latest_user_text)
    contract = pending_action_turn.execution_contract
    validation = pending_action_turn.validation
    if contract is None or validation is None:
        content = build_pending_action_clarification(
            pending_action,
            {"reason": "The pending-action decision could not be validated."},
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
            "execution_contract": None,
        }

    if validation.get("runtime_action") == "cancel":
        content = translate_builder_text("Cancelled the pending knowledge-base write.", preferred_language)
        return {
            "messages": [AIMessage(content=content)],
            "assistant_response": build_assistant_response(
                kind="text",
                content=content,
                pending_action=None,
                execution_contract=None,
            ),
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
        content = build_pending_action_clarification(
            updated_action,
            validation,
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
            "execution_contract": None,
        }

    runtime_action = validation.get("runtime_action")

    if runtime_action == "request_revision":
        content = build_pending_action_revision_response(
            updated_action,
            contract,
            preferred_language=preferred_language,
        )
        return {
            "messages": [AIMessage(content=content)],
            "assistant_response": build_assistant_response(
                kind="await_confirmation",
                content=content,
                pending_action=updated_action,
                execution_contract=None,
            ),
            "pending_action": updated_action,
            "execution_contract": None,
        }

    if runtime_action != "execute":
        content = build_pending_action_clarification(
            updated_action,
            validation,
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
            "execution_contract": None,
        }

    retry_payload = build_pending_action_retry_tool_call(
        updated_action,
        source="knowledge_base_builder_agent",
        reason="The pending knowledge-base write requires explicit confirmation.",
        fallback_tool_call_id=f"call_retry_write_{len(state.get('messages', []))}",
    )
    if retry_payload is None:
        content = translate_builder_text("I could not reconstruct the pending write request to execute it safely.", preferred_language)
        return {
            "messages": [AIMessage(content=content)],
            "assistant_response": build_assistant_response(
                kind="text",
                content=content,
                pending_action=updated_action,
                execution_contract=None,
            ),
            "pending_action": updated_action,
            "execution_contract": None,
        }

    retry_message, tool_invocation = retry_payload
    return {
        "messages": [retry_message],
        "assistant_response": build_assistant_response(
            kind="invoke_tool",
            content="",
            pending_action=updated_action,
            execution_contract=contract,
            tool_invocation=tool_invocation,
        ),
        "pending_action": updated_action,
        "execution_contract": contract,
        "tool_invocation": tool_invocation,
    }


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

    if any(output in {"summary", "details", "plan"} for output in requested_outputs):
        rendered_summary = render_pending_write_summary(
            pending_action,
            detailed=any(output in {"details", "plan"} for output in requested_outputs),
        )
        if preferred_language == "zh":
            return (
                "这是本次知识库写入的摘要，尚未实际写入。\n\n"
                f"{rendered_summary}\n\n"
                "如果可以继续，直接回复“继续”或“执行吧”；如果不要执行，回复“取消”。"
            )
        return (
            "Here is the summary for the pending knowledge-base write. Nothing has been written yet.\n\n"
            f"{rendered_summary}\n\n"
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


def render_pending_write_summary(pending_action: dict[str, Any], *, detailed: bool = False) -> str:
    metadata = pending_action.get("metadata") if isinstance(pending_action.get("metadata"), dict) else {}
    tool_args = metadata.get("tool_args") if isinstance(metadata.get("tool_args"), dict) else {}
    relative_path = str(metadata.get("relative_path", "")).strip()
    target_exists = metadata.get("target_exists") is True
    proposed_content = str(tool_args.get("content", ""))
    lines = proposed_content.splitlines()
    heading = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            break

    summary_lines = [
        f"- Target: `{relative_path or 'pending.md'}`",
        f"- Mode: {'update existing document' if target_exists else 'create new document'}",
        f"- Content length: {len(proposed_content)} characters across {len(lines)} lines",
    ]
    if heading:
        summary_lines.append(f"- Heading: {heading}")
    if detailed and lines:
        preview = "\n".join(lines[: min(8, len(lines))])
        summary_lines.append("\n```markdown\n" + preview + "\n```")
    return "\n".join(summary_lines).strip()

def translate_builder_text(text: str, preferred_language: str) -> str:
    if preferred_language != "zh":
        return text

    translations = {
        "Cancelled the pending knowledge-base write.": "已取消待处理的知识库写入操作。",
        "I could not reconstruct the pending write request to execute it safely.": "我无法安全地还原这次待执行的知识库写入请求。",
    }
    return translations.get(text, text)
