from __future__ import annotations

from langchain_core.messages import SystemMessage

from core.state import AgentState


GENERAL_CHAT_PROMPT = (
    "You are the General Chat Agent for Jade Games Ltd. "
    "Handle greetings, casual conversation, and general questions that do not require project sheet data. "
    "If the user is asking about project tasks, assignees, schedules, deadlines, priorities, or project tracker content, "
    "do not invent an answer; those should be handled by the project-task flow instead. "
    "If the user is asking about internal architecture, setup instructions, repository documentation, or company process docs, "
    "do not invent an answer; those should be handled by the knowledge-agent flow instead. "
    "Format final answers for Slack mrkdwn, not standard Markdown. "
    "Use *single asterisks* for bold, not double asterisks. "
    "Avoid Markdown headings like # or ##. "
    "Keep answers concise, clear, and easy to read in Slack."
)


class GeneralChatAgentNode:
    def __init__(self, llm) -> None:
        self.llm = llm

    def __call__(self, state: AgentState) -> dict:
        messages = [SystemMessage(content=GENERAL_CHAT_PROMPT), *state["messages"]]
        response = self.llm.invoke(messages)
        return {"messages": [response]}
