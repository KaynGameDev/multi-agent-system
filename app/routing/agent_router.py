from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

from pydantic import ValidationError

from app.contracts import (
    AgentRouteDecision,
    AssistantRequest,
    build_fallback_assistant_request,
    validate_agent_route_decision,
    validate_assistant_request,
)
from app.interpretation.intent_parser import (
    DEFAULT_ASSISTANT_REQUEST_LOW_CONFIDENCE_REASON,
    DEFAULT_INTENT_PARSER_MALFORMED_REASON,
    DEFAULT_INTENT_PARSER_UNAVAILABLE_REASON,
    IntentParser,
)
from app.interpretation.model_config import DEFAULT_INTENT_PARSER_CONFIDENCE_THRESHOLD

from .domain_map import resolve_agent_for_domain
from .routing_diagnostics import (
    build_agent_route_diagnostic,
    build_assistant_request_diagnostic,
)

ROUTE_POLICY_STEP_ASSISTANT_REQUEST_DOMAIN = "assistant_request_domain"
ROUTE_POLICY_STEP_ASSISTANT_REQUEST_FALLBACK = "assistant_request_fallback"


@dataclass(frozen=True)
class AgentRouterResult:
    assistant_request: AssistantRequest
    agent_route_decision: AgentRouteDecision
    selected_agent: str
    policy_step: str
    diagnostics: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]


class AgentRouter:
    def __init__(
        self,
        parser: IntentParser | Any | None = None,
        *,
        confidence_threshold: float | None = None,
    ) -> None:
        self.parser = parser
        self.confidence_threshold = resolve_agent_router_confidence_threshold(
            parser,
            confidence_threshold=confidence_threshold,
        )

    def route_request(
        self,
        *,
        latest_user_text: str,
        recent_messages: list[str] | tuple[str, ...] | None,
        routing_context: dict[str, Any],
        registrations_by_name: dict[str, Any],
        default_agent: str = "general_chat_agent",
    ) -> AgentRouterResult:
        assistant_request = self._parse_assistant_request(
            latest_user_text,
            recent_messages=recent_messages,
            routing_context=routing_context,
        )

        selected_agent, fallback_used, fallback_reason = resolve_agent_for_domain(
            assistant_request.likely_domain,
            registrations_by_name=registrations_by_name,
            default_agent=default_agent,
        )

        if assistant_request.confidence < self.confidence_threshold:
            logger.info(
                "AssistantRequest confidence below threshold but trusting parsed domain. "
                "confidence=%s threshold=%s domain=%s agent=%s",
                assistant_request.confidence,
                self.confidence_threshold,
                assistant_request.likely_domain,
                selected_agent,
            )

        policy_step = (
            ROUTE_POLICY_STEP_ASSISTANT_REQUEST_FALLBACK
            if fallback_used
            else ROUTE_POLICY_STEP_ASSISTANT_REQUEST_DOMAIN
        )
        reason = (
            f"Fallback to `{selected_agent}` because {fallback_reason}"
            if fallback_used
            else f"AssistantRequest mapped `{assistant_request.likely_domain}` to `{selected_agent}`."
        )
        route_decision = validate_agent_route_decision(
            {
                "selected_agent": selected_agent,
                "reason": reason,
                "fallback_used": fallback_used,
                "diagnostics": {
                    "assistant_request": assistant_request.model_dump(),
                },
            }
        )

        diagnostics: list[dict[str, Any]] = [
            build_assistant_request_diagnostic(assistant_request),
            build_agent_route_diagnostic(
                selected_agent=selected_agent,
                fallback_used=fallback_used,
                reason=reason,
                policy_step=policy_step,
            ),
        ]
        warnings: list[str] = []
        if fallback_used and reason:
            warnings.append(reason)

        return AgentRouterResult(
            assistant_request=assistant_request,
            agent_route_decision=route_decision,
            selected_agent=selected_agent,
            policy_step=policy_step,
            diagnostics=tuple(diagnostics),
            warnings=tuple(warnings),
        )

    def _parse_assistant_request(
        self,
        latest_user_text: str,
        *,
        recent_messages: list[str] | tuple[str, ...] | None,
        routing_context: dict[str, Any],
    ) -> AssistantRequest:
        normalized_user_text = str(latest_user_text or "").strip()
        if self.parser is None:
            return build_fallback_assistant_request(
                normalized_user_text,
                notes=DEFAULT_INTENT_PARSER_UNAVAILABLE_REASON,
            )

        try:
            parsed = self.parser.parse_assistant_request(
                normalized_user_text,
                recent_messages=recent_messages,
                routing_context=routing_context,
            )
        except TypeError:
            try:
                parsed = self.parser.parse_assistant_request(normalized_user_text)
            except Exception:
                return build_fallback_assistant_request(
                    normalized_user_text,
                    notes=DEFAULT_INTENT_PARSER_UNAVAILABLE_REASON,
                )
        except Exception:
            return build_fallback_assistant_request(
                normalized_user_text,
                notes=DEFAULT_INTENT_PARSER_UNAVAILABLE_REASON,
            )

        try:
            return validate_assistant_request(parsed)
        except ValidationError:
            return build_fallback_assistant_request(
                normalized_user_text,
                notes=DEFAULT_INTENT_PARSER_MALFORMED_REASON,
            )


def resolve_agent_router_confidence_threshold(
    parser: IntentParser | Any | None,
    *,
    confidence_threshold: float | None,
) -> float:
    if confidence_threshold is not None:
        return float(confidence_threshold)
    if isinstance(parser, IntentParser):
        return float(parser.config.confidence_threshold)
    return float(DEFAULT_INTENT_PARSER_CONFIDENCE_THRESHOLD)
