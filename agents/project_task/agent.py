from __future__ import annotations
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage

from app.contracts import build_assistant_response
from app.language import detect_response_language
from app.messages import extract_latest_human_text, stringify_message_content
from app.memory.agent_scope import build_agent_memory_prompt, resolve_agent_memory_context
from app.memory.snapshots import (
    LongTermMemorySnapshotError,
    apply_long_term_memory_snapshot,
    get_pending_long_term_memory_snapshot,
)
from app.model_request import build_model_request_messages
from app.pending_actions import (
    PendingActionSelectionOption,
    build_pending_action,
    get_pending_action,
    get_pending_action_metadata,
    is_pending_action_active,
    update_pending_action,
)
from app.prompt_loader import join_prompt_layers, load_prompt_sections, load_shared_instruction_text
from app.routing.pending_action_router import PendingActionRouter, PendingActionTurnResult, resolve_owned_pending_action_turn
from app.skill_runtime import build_skill_prompt_context
from app.skills import SkillRegistry
from app.state import AgentState
from app.tool_runtime import (
    build_tool_execution_record_for_message,
    extract_first_tool_invocation,
    extract_tool_result_from_message,
)
from app.tool_registry import TOOL_PROJECT_READ_TASKS, TOOL_PROJECT_SHEET_OVERVIEW, build_agent_tool_prompt

PROMPT_PATH = "agents/project_task/AGENT.md"


class ProjectTaskAgentNode:
    def __init__(
        self,
        llm,
        tools: list,
        *,
        settings=None,
        skill_registry: SkillRegistry | None = None,
        pending_action_router: PendingActionRouter | None = None,
        agent_name: str = "",
        tool_ids: tuple[str, ...] = (),
        memory_scope: str = "",
    ) -> None:
        self.llm = llm.bind_tools(tools)
        self.settings = settings
        self.skill_registry = skill_registry
        self.pending_action_router = pending_action_router
        self.agent_name = agent_name
        self.tool_ids = tuple(tool_ids)
        self.memory_scope = str(memory_scope or "").strip().lower()

    def __call__(self, state: AgentState) -> dict:
        rendered_response = build_project_task_response(
            state,
            settings=self.settings,
            agent_name=self.agent_name,
            memory_scope=self.memory_scope,
            pending_action_router=self.pending_action_router,
        )
        if rendered_response is not None:
            return rendered_response

        messages = build_model_request_messages(
            system_prompt=build_project_task_prompt(
                state,
                settings=self.settings,
                skill_registry=self.skill_registry,
                agent_name=self.agent_name,
                tool_ids=self.tool_ids,
                memory_scope=self.memory_scope,
            ),
            transcript_messages=state["messages"],
        )
        response = self.llm.invoke(messages)
        assistant_text = stringify_message_content(getattr(response, "content", ""))
        tool_invocation = extract_first_tool_invocation(
            response,
            source="project_task_agent",
            reason="The model requested a follow-up project task tool call.",
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


def build_project_task_prompt(
    state: AgentState,
    *,
    settings=None,
    skill_registry: SkillRegistry | None = None,
    agent_name: str = "",
    tool_ids: tuple[str, ...] = (),
    memory_scope: str = "",
) -> str:
    user_sheet_name = str(state.get("user_sheet_name", "")).strip()
    user_mapped_slack_name = str(
        state.get("user_mapped_slack_name", state.get("user_display_name", ""))
    ).strip()
    interface_name = str(state.get("interface_name", "")).strip().lower()
    today = date.today().isoformat()
    latest_user_text = extract_latest_human_text(state)
    sections = load_prompt_sections(
        PROMPT_PATH,
        required_sections=(
            "role",
            "responsibilities",
            "tool_usage",
            "boundaries",
            "slack_output",
            "web_output",
            "default_output",
            "date_context",
            "tool_guidance",
            "identity_context",
            "name_resolution",
        ),
        today=today,
        user_sheet_name=user_sheet_name,
        user_google_name=str(state.get("user_google_name", "")).strip(),
        user_job_title=str(state.get("user_job_title", "")).strip(),
        user_mapped_slack_name=user_mapped_slack_name,
    )
    lines = [
        sections["role"],
        sections["responsibilities"],
        sections["tool_usage"],
        build_agent_tool_prompt(tool_ids),
        sections["boundaries"],
    ]
    if settings is not None and agent_name and memory_scope:
        lines.append(
            build_agent_memory_prompt(
                settings,
                agent_name=agent_name,
                memory_scope=memory_scope,
                state=state,
                query_text=latest_user_text,
            )
        )
    if interface_name == "slack":
        lines.append(sections["slack_output"])
    elif interface_name == "web":
        lines.append(sections["web_output"])
    else:
        lines.append(sections["default_output"])
    lines.append(sections["date_context"])
    lines.append(sections["tool_guidance"])
    if skill_registry is not None and agent_name:
        lines.append(
            build_skill_prompt_context(
                state,
                skill_registry=skill_registry,
                agent_name=agent_name,
            )
        )
    if user_sheet_name:
        lines.append(sections["identity_context"])
    lines.append(sections["name_resolution"])

    return join_prompt_layers(load_shared_instruction_text(), *lines)

DUE_SCOPE_LABELS = {
    "overdue": "Overdue tasks",
    "today": "Tasks due today",
    "this_week": "Tasks due this week",
    "next_7_days": "Tasks due in the next 7 days",
}

DUE_SCOPE_LABELS_ZH = {
    "overdue": "逾期任务",
    "today": "今日到期任务",
    "this_week": "本周到期任务",
    "next_7_days": "未来 7 天到期任务",
}

TASK_FIELD_LABELS_ZH = {
    "project": "项目",
    "iteration": "迭代",
    "platform": "平台",
    "priority": "优先级",
    "assignee": "负责人",
    "due": "截止",
    "status": "状态",
    "client": "客户端",
    "server": "服务器",
    "qa": "测试",
    "pm": "产品",
}


def build_project_task_response(
    state: AgentState,
    *,
    settings=None,
    agent_name: str,
    memory_scope: str = "",
    pending_action_router: PendingActionRouter | None = None,
) -> dict[str, Any] | None:
    latest_user_text = extract_latest_human_text(state)
    preferred_language = detect_response_language(latest_user_text)
    pending_action, pending_action_turn = resolve_owned_pending_action_turn(
        state, agent_name=agent_name, pending_action_router=pending_action_router,
    )
    if pending_action is not None:
        follow_up_result = build_project_task_pending_action_response(
            state,
            settings=settings,
            agent_name=agent_name,
            memory_scope=memory_scope,
            pending_action=pending_action,
            pending_action_turn=pending_action_turn,
            preferred_language=preferred_language,
        )
        if follow_up_result is not None:
            return follow_up_result

    latest_tool_result = get_latest_task_tool_result(state)
    if latest_tool_result is not None:
        return render_task_update(state, latest_tool_result, preferred_language=preferred_language)

    active_pending_action = get_pending_action(state)
    if (
        settings is not None
        and str(memory_scope or "").strip().lower() == "user"
        and getattr(settings, "long_term_memory_enabled", False)
        and not is_pending_action_active(active_pending_action)
    ):
        snapshot_pending_action = build_memory_snapshot_pending_action(
            state,
            settings=settings,
            agent_name=agent_name,
            preferred_language=preferred_language,
        )
        if snapshot_pending_action is not None:
            content = str(get_pending_action_metadata(snapshot_pending_action).get("prompt_context", "")).strip()
            return {
                "messages": [AIMessage(content=content)],
                "assistant_response": build_assistant_response(
                    kind="await_confirmation",
                    content=content,
                    pending_action=snapshot_pending_action,
                ),
                "pending_action": snapshot_pending_action,
            }
    return None


def get_latest_task_tool_result(state: AgentState) -> dict[str, Any] | None:
    messages = state.get("messages", [])
    if messages:
        latest_result = get_task_tool_result(messages[-1], messages=messages)
        if latest_result is not None:
            return latest_result
    return None


def get_task_tool_result(message, *, messages: list[Any] | None = None) -> dict[str, Any] | None:
    result = extract_tool_result_from_message(
        message,
        messages=messages,
        tool_name="project_task",
        source="project_task_agent",
        reason="Project task ToolNode returned a result.",
    )
    if result is None:
        return None
    payload = result.get("payload") if isinstance(result.get("payload"), dict) else {}
    if isinstance(payload.get("tasks"), list):
        return result
    return None


def render_task_update(state: AgentState, tool_result: dict[str, Any], *, preferred_language: str) -> dict[str, Any] | None:
    payload = tool_result.get("payload") if isinstance(tool_result.get("payload"), dict) else {}
    rendered = render_task_payload(payload, preferred_language=preferred_language)
    if rendered is None:
        return None

    prompt_context = ""
    pending_action = build_task_pending_action(state, tool_result)
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
            tool_name=str(tool_result.get("tool_name", "")).strip() or "project_task",
            source="project_task_agent",
            reason="Project task ToolNode returned a result.",
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


def build_task_pending_action(state: AgentState, tool_result: dict[str, Any]) -> dict[str, Any] | None:
    payload = tool_result.get("payload") if isinstance(tool_result.get("payload"), dict) else {}
    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []
    if not tasks:
        return None

    thread_id = str(state.get("thread_id", "")).strip()
    if not thread_id:
        return None

    preferred_language = "zh" if any(task_contains_chinese(task) for task in tasks if isinstance(task, dict)) else "en"
    header = build_task_list_header(payload, preferred_language=preferred_language)
    single_option = len(tasks) == 1
    options = [
        build_task_option(task, index=index, include_referential_aliases=single_option)
        for index, task in enumerate(tasks, start=1)
        if isinstance(task, dict)
    ]
    if not options:
        return None

    source_tool_id = resolve_project_source_tool_id(tool_result, payload)
    prompt_context = (
        "回复任务编号或任务标题可查看详情；回复 `取消` 结束。"
        if preferred_language == "zh"
        else "Reply with the task number or exact task title for details, or reply `cancel` to stop."
    )
    return build_pending_action(
        session_id=thread_id,
        action_type="select_project_task",
        requested_by_agent="project_task_agent",
        summary="Select a task to inspect from the project tracker results.",
        risk_level="low",
        requires_explicit_approval=False,
        metadata={
            "source_tool_id": source_tool_id,
            "prompt_context": prompt_context,
            "selection_options": options,
            "payload": {"header": header},
        },
    )


def resolve_project_source_tool_id(tool_result: dict[str, Any], payload: dict[str, Any]) -> str:
    tool_id = str(tool_result.get("tool_id", "")).strip()
    if tool_id:
        return tool_id
    if "total_rows" in payload:
        return TOOL_PROJECT_SHEET_OVERVIEW
    return TOOL_PROJECT_READ_TASKS


def build_project_task_pending_action_response(
    state: AgentState,
    *,
    settings=None,
    agent_name: str,
    memory_scope: str,
    pending_action: dict[str, Any],
    pending_action_turn: PendingActionTurnResult,
    preferred_language: str,
) -> dict[str, Any] | None:
    action_type = str(pending_action.get("type", "")).strip()
    if action_type == "apply_memory_snapshot":
        return build_memory_snapshot_pending_action_response(
            state,
            settings=settings,
            agent_name=agent_name,
            memory_scope=memory_scope,
            pending_action=pending_action,
            pending_action_turn=pending_action_turn,
            preferred_language=preferred_language,
        )
    if action_type == "select_project_task":
        return build_task_pending_action_response(
            state,
            pending_action=pending_action,
            pending_action_turn=pending_action_turn,
            preferred_language=preferred_language,
        )
    return None


def build_memory_snapshot_pending_action(
    state: AgentState,
    *,
    settings,
    agent_name: str,
    preferred_language: str,
) -> dict[str, Any] | None:
    thread_id = str(state.get("thread_id", "")).strip()
    if not thread_id:
        return None

    user_context = resolve_agent_memory_context(
        settings,
        agent_name=agent_name,
        memory_scope="user",
        state=state,
    )
    project_context = resolve_agent_memory_context(
        settings,
        agent_name=agent_name,
        memory_scope="project",
        state=state,
    )
    snapshot = get_pending_long_term_memory_snapshot(
        project_context.root_dir,
        user_context.root_dir,
    )
    if snapshot is None:
        return None

    options = build_memory_snapshot_selection_options(preferred_language=preferred_language)
    prompt_context = build_memory_snapshot_prompt_text(
        snapshot_id=snapshot.snapshot_id,
        memory_count=snapshot.memory_count,
        preferred_language=preferred_language,
    )
    return build_pending_action(
        session_id=thread_id,
        action_type="apply_memory_snapshot",
        requested_by_agent=agent_name,
        summary="Choose how to apply the project memory snapshot to personal memory.",
        risk_level="low",
        requires_explicit_approval=False,
        metadata={
            "prompt_context": prompt_context,
            "selection_options": options,
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_fingerprint": snapshot.fingerprint,
            "snapshot_memory_count": snapshot.memory_count,
            "project_memory_root": str(project_context.root_dir),
            "user_memory_root": str(user_context.root_dir),
        },
    )


def build_memory_snapshot_selection_options(*, preferred_language: str) -> list[PendingActionSelectionOption]:
    if preferred_language == "zh":
        return [
            {
                "id": "keep",
                "label": "保留当前记忆",
                "aliases": ["keep", "保留", "跳过", "保持现状"],
                "value": "keep",
                "payload": {"action": "keep"},
            },
            {
                "id": "merge",
                "label": "合并快照",
                "aliases": ["merge", "合并", "组合"],
                "value": "merge",
                "payload": {"action": "merge"},
            },
            {
                "id": "replace",
                "label": "替换为快照",
                "aliases": ["replace", "替换", "覆盖"],
                "value": "replace",
                "payload": {"action": "replace"},
            },
        ]

    return [
        {
            "id": "keep",
            "label": "Keep current memory",
            "aliases": ["keep", "skip", "keep mine", "leave it"],
            "value": "keep",
            "payload": {"action": "keep"},
        },
        {
            "id": "merge",
            "label": "Merge snapshot",
            "aliases": ["merge", "combine", "blend"],
            "value": "merge",
            "payload": {"action": "merge"},
        },
        {
            "id": "replace",
            "label": "Replace with snapshot",
            "aliases": ["replace", "overwrite", "use snapshot"],
            "value": "replace",
            "payload": {"action": "replace"},
        },
    ]


def build_memory_snapshot_prompt_text(
    *,
    snapshot_id: str,
    memory_count: int,
    preferred_language: str,
) -> str:
    if preferred_language == "zh":
        return (
            f"检测到项目记忆快照 `{snapshot_id}`，包含 {memory_count} 条主题记忆。\n\n"
            "请选择如何更新你的个人记忆：`keep` 保持现有记忆不变，`merge` 将快照内容合并进去，"
            "`replace` 用快照替换现有个人记忆。"
        )

    return (
        f"Project memory snapshot `{snapshot_id}` is available with {memory_count} topic memories.\n\n"
        "Choose how to update your personal memory: `keep` leaves it unchanged, `merge` combines the snapshot "
        "with it, and `replace` swaps your current personal memory for the snapshot."
    )


def build_memory_snapshot_pending_action_response(
    state: AgentState,
    *,
    settings=None,
    agent_name: str,
    memory_scope: str,
    pending_action: dict[str, Any],
    pending_action_turn: PendingActionTurnResult,
    preferred_language: str,
) -> dict[str, Any] | None:
    if settings is None or str(memory_scope or "").strip().lower() != "user":
        return None

    contract = pending_action_turn.execution_contract
    validation = pending_action_turn.validation
    if contract is None or validation is None:
        content = build_task_clarification_text(
            pending_action=pending_action,
            validation={"reason": "The snapshot decision could not be validated."},
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
        content = (
            "已取消此次快照更新，之后需要时可以再选择。"
            if preferred_language == "zh"
            else "Cancelled this snapshot update for now."
        )
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

    if not validation.get("valid") or validation.get("runtime_action") != "select":
        content = build_task_clarification_text(
            pending_action=updated_action,
            validation=validation or {"reason": "The snapshot decision was unclear."},
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

    selected_option = validation.get("selected_option") if isinstance(validation.get("selected_option"), dict) else None
    payload = selected_option.get("payload") if isinstance(selected_option, dict) else {}
    action = str(payload.get("action") or selected_option.get("value") or selected_option.get("id") or "").strip().lower()
    if action not in {"keep", "merge", "replace"}:
        content = build_task_clarification_text(
            pending_action=updated_action,
            validation={"reason": "The snapshot choice did not resolve to keep, merge, or replace."},
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

    metadata = get_pending_action_metadata(updated_action)
    project_memory_root = str(metadata.get("project_memory_root", "")).strip()
    user_memory_root = str(metadata.get("user_memory_root", "")).strip()
    snapshot_id = str(metadata.get("snapshot_id", "")).strip()
    if not project_memory_root or not user_memory_root:
        content = build_task_clarification_text(
            pending_action=updated_action,
            validation={"reason": "The snapshot roots were missing from the pending action."},
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

    try:
        summary = apply_long_term_memory_snapshot(
            user_memory_root,
            project_memory_root,
            action=action,
            snapshot_id=snapshot_id,
        )
    except LongTermMemorySnapshotError as exc:
        content = build_task_clarification_text(
            pending_action=updated_action,
            validation={"reason": str(exc)},
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
    content = build_memory_snapshot_result_text(summary, preferred_language=preferred_language)
    return {
        "messages": [AIMessage(content=content)],
        "assistant_response": build_assistant_response(kind="text", content=content),
        "pending_action": None,
    }


def build_memory_snapshot_result_text(summary, *, preferred_language: str) -> str:
    if preferred_language == "zh":
        if summary.action == "keep":
            return f"已记录：项目快照 `{summary.snapshot_id}` 暂不应用到你的个人记忆。"
        if summary.action == "merge":
            return (
                f"已将项目快照 `{summary.snapshot_id}` 合并到你的个人记忆中。"
                f" 新增 {len(summary.created_memory_ids)} 条，更新 {len(summary.updated_memory_ids)} 条。"
            )
        return (
            f"已用项目快照 `{summary.snapshot_id}` 替换你的个人记忆基线。"
            f" 删除 {len(summary.deleted_memory_ids)} 条，写入 {len(summary.created_memory_ids)} 条。"
        )

    if summary.action == "keep":
        return f"Recorded that snapshot `{summary.snapshot_id}` should be kept out of your personal memory for now."
    if summary.action == "merge":
        return (
            f"Merged snapshot `{summary.snapshot_id}` into your personal memory. "
            f"Created {len(summary.created_memory_ids)} memories and updated {len(summary.updated_memory_ids)}."
        )
    return (
        f"Replaced your personal memory baseline with snapshot `{summary.snapshot_id}`. "
        f"Deleted {len(summary.deleted_memory_ids)} memories and wrote {len(summary.created_memory_ids)}."
    )


def build_task_pending_action_response(
    state: AgentState,
    *,
    pending_action: dict[str, Any],
    pending_action_turn: PendingActionTurnResult,
    preferred_language: str,
) -> dict[str, Any] | None:
    contract = pending_action_turn.execution_contract
    validation = pending_action_turn.validation
    if contract is None or validation is None:
        content = build_task_clarification_text(
            pending_action=pending_action,
            validation={"reason": "The pending-action decision could not be validated."},
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
        content = "已取消待处理的任务跟进。" if preferred_language == "zh" else "Cancelled the pending task follow-up."
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
        content = build_task_clarification_text(
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

    if validation.get("runtime_action") != "select":
        content = build_task_clarification_text(
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

    selected_option = validation.get("selected_option") if isinstance(validation.get("selected_option"), dict) else None
    if selected_option is None:
        content = build_task_clarification_text(
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

    task = selected_option.get("payload", {}).get("task")
    if not isinstance(task, dict):
        content = build_task_clarification_text(
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

    metadata = get_pending_action_metadata(pending_action)
    header = str(metadata.get("payload", {}).get("header", "")).strip() if isinstance(metadata.get("payload"), dict) else ""
    rendered = render_task_payload(
        {"tasks": [task], "filters": {}, "match_count": 1},
        preferred_language=preferred_language,
        header_override=header,
    )
    if rendered is None:
        content = build_task_clarification_text(
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
    return {
        "messages": [AIMessage(content=rendered)],
        "assistant_response": build_assistant_response(
            kind="text",
            content=rendered,
            pending_action=updated_action,
        ),
        "pending_action": None,
    }


def build_task_clarification_text(
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


def build_task_list_header(payload: dict, *, preferred_language: str = "en") -> str:
    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []
    shown_count = len(tasks)
    total_count = payload.get("match_count")
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    due_scope = filters.get("due_scope", "") if isinstance(filters, dict) else ""
    assignee = filters.get("assignee", "") if isinstance(filters, dict) else ""

    if preferred_language == "zh":
        base = DUE_SCOPE_LABELS_ZH.get(str(due_scope), "任务")
        if assignee:
            base = f"{assignee} 的{base}"
        if isinstance(total_count, int) and total_count > shown_count:
            return f"{base}（显示 {shown_count} / 共 {total_count}）"
        return f"{base}（{shown_count}）"

    base = DUE_SCOPE_LABELS.get(str(due_scope), "Tasks")
    if assignee:
        base = f"{base} for {assignee}"
    if isinstance(total_count, int) and total_count > shown_count:
        return f"{base} ({shown_count} shown of {total_count})"
    return f"{base} ({shown_count})"


def render_task_payload(
    payload: dict,
    *,
    preferred_language: str = "en",
    header_override: str = "",
) -> str | None:
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return None

    header = header_override.strip() or build_task_list_header(payload, preferred_language=preferred_language)
    lines = [header]
    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            continue
        lines.extend(format_task_block(index, task, preferred_language=preferred_language))
    return "\n\n".join(line for line in lines if line).strip() or None


def format_task_block(index: int, task: dict, *, preferred_language: str = "en") -> list[str]:
    title = first_non_empty(task.get("content"), "Untitled task")
    line_one = f"{index}. {title}"

    summary = join_parts(
        [
            labeled_value(task_label("project", preferred_language), task.get("project")),
            labeled_value(task_label("iteration", preferred_language), task.get("iteration")),
            labeled_value(task_label("platform", preferred_language), task.get("platform")),
            labeled_value(task_label("priority", preferred_language), task.get("priority")),
        ]
    )

    schedule = join_parts(
        [
            labeled_value(task_label("assignee", preferred_language), task.get("assignee")),
            labeled_value(task_label("due", preferred_language), task.get("end_date")),
            labeled_value(
                task_label("status", preferred_language),
                humanize_due_status(task.get("due_status"), preferred_language=preferred_language),
            ),
        ]
    )

    owners = join_parts(
        [
            labeled_value(task_label("client", preferred_language), task.get("client_owner")),
            labeled_value(task_label("server", preferred_language), task.get("server_owner")),
            labeled_value(task_label("qa", preferred_language), task.get("test_owner")),
            labeled_value(task_label("pm", preferred_language), task.get("product_owner")),
        ]
    )

    lines = [line_one]
    if summary:
        lines.append(summary)
    if schedule:
        lines.append(schedule)
    if owners:
        lines.append(owners)
    return lines


def build_task_option(task: dict[str, Any], *, index: int, include_referential_aliases: bool) -> PendingActionSelectionOption:
    title = first_non_empty(task.get("content"), "Untitled task")
    aliases = [str(index), title]
    project = first_non_empty(task.get("project"))
    if project:
        aliases.append(f"{project} {title}")
    if include_referential_aliases:
        aliases.extend(["that one", "this one", "details", "detail", "那个", "这个", "详情"])
    return PendingActionSelectionOption(
        id=first_non_empty(task.get("id"), title, str(index)),
        label=title,
        aliases=list(dict.fromkeys(alias for alias in aliases if alias)),
        value=title,
        payload={"task": task},
    )


def labeled_value(label: str, value) -> str:
    text = first_non_empty(value)
    if not text:
        return ""
    return f"{label}: {text}"


def first_non_empty(*values) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def join_parts(parts: list[str]) -> str:
    return " | ".join(part for part in parts if part)


def task_label(key: str, preferred_language: str) -> str:
    if preferred_language == "zh":
        return TASK_FIELD_LABELS_ZH.get(key, key)

    mapping = {
        "project": "Project",
        "iteration": "Iteration",
        "platform": "Platform",
        "priority": "Priority",
        "assignee": "Assignee",
        "due": "Due",
        "status": "Status",
        "client": "Client",
        "server": "Server",
        "qa": "QA",
        "pm": "PM",
    }
    return mapping.get(key, key)


def humanize_due_status(value, *, preferred_language: str = "en") -> str:
    mapping = {
        "zh": {
            "overdue": "已逾期",
            "today": "今日到期",
            "upcoming": "即将到期",
        },
        "en": {
            "overdue": "Overdue",
            "today": "Due today",
            "upcoming": "Upcoming",
        },
    }
    if not isinstance(value, str):
        return ""
    localized_mapping = mapping.get(preferred_language, mapping["en"])
    return localized_mapping.get(value, value)


def task_contains_chinese(task: dict[str, Any]) -> bool:
    text = " ".join(
        str(value)
        for value in (task.get("content"), task.get("project"), task.get("assignee"))
        if isinstance(value, str) and value.strip()
    )
    return any("\u4e00" <= char <= "\u9fff" for char in text)
