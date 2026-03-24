from __future__ import annotations

from langchain_core.messages import SystemMessage

from core.state import AgentState


GENERAL_CHAT_BASE_PROMPT = (
    "Your name is Jarvis."
    "You are the General Chat Agent for Jade Games."
    "Handle greetings, casual conversation, and general questions that do not require project sheet data. "
    "If the user is asking about project tasks, assignees, schedules, deadlines, priorities, or project tracker content, "
    "do not invent an answer; those should be handled by the project-task flow instead. "
    "If the user is asking about internal architecture, setup instructions, repository documentation, or company process docs, "
    "do not invent an answer; those should be handled by the knowledge-agent flow instead."
)

SLACK_FORMAT_PROMPT = (
    "The current interface is Slack. "
    "Write concise plain Markdown that will still read cleanly after the Slack boundary converts it to mrkdwn. "
    "Use short paragraphs or flat lists when needed. "
    "Avoid raw Slack entities like <@U123>, <#C123>, or <url|label>. "
    "Avoid oversized headings and noisy formatting."
)

DEFAULT_FORMAT_PROMPT = (
    "Write concise, plain Markdown that stays readable across chat interfaces."
)


class GeneralChatAgentNode:
    def __init__(self, llm) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        messages = [SystemMessage(content=build_general_chat_prompt(state)), *state["messages"]]
        response = self.llm.invoke(messages)
        return {"messages": [response]}


def build_general_chat_prompt(state: AgentState) -> str:
    interface_name = str(state.get("interface_name", "")).strip().lower()
    format_prompt = DEFAULT_FORMAT_PROMPT
    if interface_name == "slack":
        format_prompt = SLACK_FORMAT_PROMPT
    return "\n\n".join((GENERAL_CHAT_BASE_PROMPT, format_prompt))
