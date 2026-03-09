from __future__ import annotations

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

RouteName = Literal["project_task_agent", "general_chat_agent"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    route: RouteName
    route_reason: str
    user_id: str
    channel_id: str
