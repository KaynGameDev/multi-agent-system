from __future__ import annotations

from datetime import date

from langchain_core.messages import SystemMessage

from core.state import AgentState


PROJECT_TASK_PROMPT = (
    "You are the Project Task Agent for Jade Games Ltd. "
    "Answer questions about the Google Sheets project tracker. "
    "Use the project tools whenever the answer depends on task data, assignees, owners, deadlines, statuses, priorities, or project schedule information. "
    "Project tools return structured JSON data, not preformatted prose. Read the tool output carefully, reason over it, and summarize the relevant facts for the user. "
    "Do not dump raw JSON unless the user explicitly asks for it. "
    "Do not invent sheet data. If the sheet does not contain the requested information, say so clearly. "
    "Write concise, plain Markdown that stays easy to scan in Slack after boundary formatting. "
    "Prefer short sections and readable bullets when helpful. "
    "Avoid headings that are too large or noisy."
)


class ProjectTaskAgentNode:
    def __init__(self, llm, tools: list) -> None:
        self.llm = llm.bind_tools(tools)

    def __call__(self, state: AgentState) -> dict:
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
