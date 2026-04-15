from __future__ import annotations

from typing import Any

from app.contracts import AssistantRequest


def build_assistant_request_diagnostic(
    assistant_request: AssistantRequest,
) -> dict[str, Any]:
    return {
        "kind": "assistant_request",
        "type": assistant_request.type,
        "user_goal": assistant_request.user_goal,
        "likely_domain": assistant_request.likely_domain,
        "confidence": assistant_request.confidence,
        "notes": assistant_request.notes,
    }


def build_agent_route_diagnostic(
    *,
    selected_agent: str,
    fallback_used: bool,
    reason: str,
    policy_step: str,
) -> dict[str, Any]:
    return {
        "kind": "agent_route_decision",
        "policy_step": policy_step,
        "selected_agent": selected_agent,
        "fallback_used": fallback_used,
        "reason": reason,
    }
