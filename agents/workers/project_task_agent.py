from __future__ import annotations

import json
import re
from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core.language import LANGUAGE_MATCHING_PROMPT, detect_response_language
from core.state import AgentState


PROJECT_TASK_PROMPT = (
    "You are the Project Task Agent for Jade Games Ltd. "
    "Answer questions about the Google Sheets project tracker. "
    "Use the project tools whenever the answer depends on task data, assignees, owners, deadlines, statuses, priorities, or project schedule information. "
    "Project tools return structured JSON data, not preformatted prose. Read the tool output carefully, reason over it, and summarize the relevant facts for the user. "
    "Do not dump raw JSON unless the user explicitly asks for it. "
    "Do not invent sheet data. If the sheet does not contain the requested information, say so clearly. "
    "Write concise, plain Markdown that stays easy to scan in Slack after boundary formatting. "
    "When listing tasks, use a numbered list with one task per block. "
    "Do not make every metadata line a bullet. "
    "Bold at most the task title if needed, not the whole block. "
    "Avoid headings that are too large or noisy. "
    f"{LANGUAGE_MATCHING_PROMPT}"
)


class ProjectTaskAgentNode:
    def __init__(self, llm, tools: list) -> None:
        self.llm = llm.bind_tools(tools)

    def __call__(self, state: AgentState) -> dict:
        task_list_response = build_task_list_response(state)
        if task_list_response is not None:
            return {"messages": [AIMessage(content=task_list_response)]}

        messages = [SystemMessage(content=build_project_task_prompt(state)), *state["messages"]]
        response = self.llm.invoke(messages)
        return {"messages": [response]}


def build_project_task_prompt(state: AgentState) -> str:
    lines = [PROJECT_TASK_PROMPT]
    today = date.today().isoformat()

    lines.append(
        f"Today's local date is {today}. For deadline questions, prefer tool filters instead of doing date math in your head."
    )
    lines.append(
        "For time-based task lookups, use read_project_tasks with due_scope values like 'overdue', 'today', 'this_week', or 'next_7_days'. "
        "For explicit date ranges, use end_date_from and end_date_to. "
        "Interpret 'this week' as Monday through Sunday."
    )

    user_sheet_name = state.get("user_sheet_name", "")
    if user_sheet_name:
        lines.append(
            "Current user identity context: "
            f"sheet_name={user_sheet_name}; "
            f"google_name={state.get('user_google_name', '')}; "
            f"job_title={state.get('user_job_title', '')}; "
            f"slack_name={state.get('user_mapped_slack_name', state.get('user_display_name', ''))}."
        )
        lines.append(
            "If the user asks about 'my tasks', 'my work', 'my deadlines', or 'what am I doing', "
            f"treat the assignee as '{user_sheet_name}' unless the user clearly specifies someone else."
        )

    lines.append(
        "When a person is mentioned using a Slack-style name, alias, email, or English name, prefer the canonical Chinese sheet name when calling tools."
    )

    return "\n\n".join(lines)


REFERENTIAL_TASK_QUERY_PATTERNS = (
    r"\bwhat are those tasks\b",
    r"\bwhat are these tasks\b",
    r"\bwhat are those\b",
    r"\bwhat are these\b",
    r"\bwhat are they\b",
    r"\bshow (?:them|those|these)\b",
    r"\blist (?:them|those|these)\b",
    r"\bmore details\b",
    r"^details?\??$",
    r"those tasks",
    r"these tasks",
    r"那些任务",
    r"这些任务",
    r"把它们列出来",
    r"把这些任务列出来",
    r"把那些任务列出来",
    r"列一下这些任务",
    r"这些任务详情",
    r"那些任务详情",
    r"^详情$",
)

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


def build_task_list_response(state: AgentState) -> str | None:
    latest_user_text = get_latest_user_text(state)
    if not should_render_task_list(latest_user_text):
        return None

    payload = get_latest_task_payload(state)
    if payload is None:
        return None

    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return None

    preferred_language = detect_response_language(latest_user_text)
    lines = [build_task_list_header(payload, preferred_language=preferred_language)]
    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            continue
        lines.extend(format_task_block(index, task, preferred_language=preferred_language))

    return "\n\n".join(line for line in lines if line).strip() or None


def should_render_task_list(user_text: str) -> bool:
    normalized = user_text.strip().lower()
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in REFERENTIAL_TASK_QUERY_PATTERNS)


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


def get_latest_task_payload(state: AgentState) -> dict | None:
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, ToolMessage):
            continue
        content = getattr(message, "content", "")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
            return payload
    return None


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
