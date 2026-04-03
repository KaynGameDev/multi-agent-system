from __future__ import annotations

from typing import Any
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

RouteName = str


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    route: RouteName
    route_reason: str
    interface_name: str
    requested_agent: str
    requested_skill_ids: list[str]
    resolved_skill_ids: list[str]
    context_paths: list[str]
    skill_resolution_diagnostics: list[dict[str, Any]]
    agent_selection_diagnostics: list[dict[str, Any]]
    selection_warnings: list[str]
    thread_id: str
    user_id: str
    channel_id: str
    user_display_name: str
    user_real_name: str
    user_email: str
    user_google_name: str
    user_sheet_name: str
    user_job_title: str
    user_mapped_slack_name: str
    uploaded_files: list[dict[str, str]]
    conversion_session_id: str
    target_game_slug: str
    target_market_slug: str
    target_feature_slug: str
    conversion_status: str
    missing_required_fields: list[str]
    approval_state: str
