from __future__ import annotations

import json
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.language import detect_response_language
from app.pending_actions import (
    PendingActionSelectionOption,
    build_pending_action,
    get_pending_action,
    get_pending_action_metadata,
    is_pending_action_active,
    resolve_pending_action_reply,
    update_pending_action,
)
from app.prompt_loader import join_prompt_layers, load_prompt_sections, load_shared_instruction_text
from app.skills import SkillRegistry
from app.state import AgentState
from app.tool_registry import TOOL_PROJECT_READ_TASKS, TOOL_PROJECT_SHEET_OVERVIEW, build_agent_tool_prompt

PROMPT_PATH = "agents/project_task/AGENT.md"


class ProjectTaskAgentNode:
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
        task_list_response = build_task_list_response(state, agent_name=self.agent_name)
        if task_list_response is not None:
            return task_list_response

        messages = [
            SystemMessage(
                content=build_project_task_prompt(
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
        latest_payload = get_latest_task_tool_payload(state)
        if latest_payload is not None:
            pending_action = build_task_pending_action(state, latest_payload)
            if pending_action is not None:
                result["pending_action"] = pending_action
        return result


def build_project_task_prompt(
    state: AgentState,
    *,
    skill_registry: SkillRegistry | None = None,
    agent_name: str = "",
    tool_ids: tuple[str, ...] = (),
) -> str:
    user_sheet_name = str(state.get("user_sheet_name", "")).strip()
    user_mapped_slack_name = str(
        state.get("user_mapped_slack_name", state.get("user_display_name", ""))
    ).strip()
    interface_name = str(state.get("interface_name", "")).strip().lower()
    today = date.today().isoformat()
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
            skill_registry.build_prompt_layers(
                state.get("resolved_skill_ids", []),
                agent_name=agent_name,
                context_paths=state.get("context_paths", []),
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


def build_task_list_response(state: AgentState, *, agent_name: str) -> dict[str, Any] | None:
    latest_user_text = get_latest_user_text(state)
    preferred_language = detect_response_language(latest_user_text)
    pending_action = get_pending_action(state)
    if pending_action and pending_action.get("requested_by_agent") == agent_name and is_pending_action_active(pending_action):
        follow_up_result = build_task_pending_action_response(
            state,
            pending_action=pending_action,
            preferred_language=preferred_language,
        )
        if follow_up_result is not None:
            return follow_up_result
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


def get_latest_task_tool_payload(state: AgentState) -> dict | None:
    messages = state.get("messages", [])
    if not messages:
        return None
    return get_task_tool_payload(messages[-1])


def get_task_tool_payload(message) -> dict | None:
    if not isinstance(message, ToolMessage):
        return None
    content = getattr(message, "content", "")
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        return payload
    return None


def build_task_pending_action(state: AgentState, payload: dict) -> dict[str, Any] | None:
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

    source_tool_id = TOOL_PROJECT_SHEET_OVERVIEW if "total_rows" in payload else TOOL_PROJECT_READ_TASKS
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


def build_task_pending_action_response(
    state: AgentState,
    *,
    pending_action: dict[str, Any],
    preferred_language: str,
) -> dict[str, Any] | None:
    latest_user_text = get_latest_user_text(state)
    resolution = resolve_pending_action_reply(pending_action, latest_user_text)
    contract = resolution["contract"]
    validation = resolution["validation"]

    if validation.get("runtime_action") == "cancel":
        return {
            "messages": [AIMessage(content="已取消待处理的任务跟进。" if preferred_language == "zh" else "Cancelled the pending task follow-up.")],
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
        return {
            "messages": [
                AIMessage(
                    content=build_task_clarification_text(
                        pending_action=updated_action,
                        validation=validation,
                        preferred_language=preferred_language,
                    )
                )
            ],
            "pending_action": updated_action,
        }

    if validation.get("runtime_action") != "select":
        return {
            "messages": [
                AIMessage(
                    content=build_task_clarification_text(
                        pending_action=updated_action,
                        validation=validation,
                        preferred_language=preferred_language,
                    )
                )
            ],
            "pending_action": updated_action,
        }

    selected_option = validation.get("selected_option") if isinstance(validation.get("selected_option"), dict) else None
    if selected_option is None:
        return {
            "messages": [
                AIMessage(
                    content=build_task_clarification_text(
                        pending_action=updated_action,
                        validation=validation,
                        preferred_language=preferred_language,
                    )
                )
            ],
            "pending_action": updated_action,
        }

    task = selected_option.get("payload", {}).get("task")
    if not isinstance(task, dict):
        return {
            "messages": [
                AIMessage(
                    content=build_task_clarification_text(
                        pending_action=updated_action,
                        validation=validation,
                        preferred_language=preferred_language,
                    )
                )
            ],
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
        return {
            "messages": [
                AIMessage(
                    content=build_task_clarification_text(
                        pending_action=updated_action,
                        validation=validation,
                        preferred_language=preferred_language,
                    )
                )
            ],
            "pending_action": updated_action,
        }
    return {
        "messages": [AIMessage(content=rendered)],
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
