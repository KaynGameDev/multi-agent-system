from __future__ import annotations

import hashlib
from dataclasses import dataclass
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from app.contracts import (
    ExecutionContract,
    ExecutionContractValidation,
    PendingAction,
    PendingActionDecision,
    build_unclear_pending_action_decision,
    validate_pending_action_decision,
)
from app.interpretation.intent_parser import (
    DEFAULT_INTENT_PARSER_UNAVAILABLE_REASON,
    IntentParser,
)
from app.messages import extract_latest_human_text
from app.pending_actions import (
    get_execution_contract,
    get_pending_action,
    get_pending_action_decision,
    is_pending_action_active,
    resolve_pending_action_decision,
    validate_execution_contract,
)
from app.state import AgentState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingActionTurnResult:
    pending_action: PendingAction
    pending_action_decision: PendingActionDecision
    pending_action_resolution_key: str
    execution_contract: ExecutionContract | None
    validation: ExecutionContractValidation | None
    allow_fresh_routing: bool


class PendingActionRouter:
    def __init__(self, parser: IntentParser | None = None) -> None:
        self.parser = parser

    def resolve_turn(self, state: AgentState) -> PendingActionTurnResult | None:
        pending_action = get_pending_action(state)
        if not is_pending_action_active(pending_action):
            return None

        existing_resolution = resolve_pending_action_turn_from_state(
            state,
            pending_action_router=None,
            allow_fallback_decision=False,
        )
        if existing_resolution is not None:
            return existing_resolution

        latest_user_text = extract_latest_human_text(state)
        decision = self.parse_pending_action_decision(
            pending_action,
            latest_user_text,
        )
        resolution_key = build_pending_action_resolution_key(state, pending_action)
        return build_pending_action_turn_result(
            pending_action,
            decision,
            resolution_key=resolution_key,
            user_text=latest_user_text,
        )

    def parse_pending_action_decision(
        self,
        pending_action: PendingAction,
        user_text: str,
    ) -> PendingActionDecision:
        pending_action_id = str(pending_action.get("id", "")).strip()
        if self.parser is None:
            logger.warning(
                "PendingActionRouter has no parser; using unclear pending-action decision. pending_action_id=%s",
                pending_action_id,
            )
            return build_unclear_pending_action_decision(
                pending_action_id,
                notes=DEFAULT_INTENT_PARSER_UNAVAILABLE_REASON,
            )
        try:
            return validate_pending_action_decision(
                self.parser.parse_pending_action_decision(pending_action, user_text)
            )
        except (ValidationError, TypeError, ValueError):
            logger.warning(
                "PendingActionRouter received malformed parser output; using unclear pending-action decision. pending_action_id=%s",
                pending_action_id,
            )
            return build_unclear_pending_action_decision(
                pending_action_id,
                notes="The pending-action decision parser returned malformed output.",
            )
        except Exception as exc:
            logger.warning(
                "PendingActionRouter parser invocation failed; using unclear pending-action decision. pending_action_id=%s error=%s",
                pending_action_id,
                exc,
            )
            return build_unclear_pending_action_decision(
                pending_action_id,
                notes=DEFAULT_INTENT_PARSER_UNAVAILABLE_REASON,
            )


def resolve_pending_action_turn_from_state(
    state: AgentState,
    *,
    pending_action_router: PendingActionRouter | None = None,
    allow_fallback_decision: bool = True,
) -> PendingActionTurnResult | None:
    pending_action = get_pending_action(state)
    if not is_pending_action_active(pending_action):
        return None

    resolution_key = build_pending_action_resolution_key(state, pending_action)
    decision = get_pending_action_decision(state)
    if decision is not None and pending_action_cache_matches_state(
        state,
        pending_action,
        decision,
        resolution_key=resolution_key,
    ):
        return hydrate_pending_action_turn_result(
            pending_action,
            decision,
            resolution_key=resolution_key,
            execution_contract=get_execution_contract(state),
        )

    if pending_action_router is not None:
        return pending_action_router.resolve_turn(state)
    if not allow_fallback_decision:
        return None

    latest_user_text = extract_latest_human_text(state)
    fallback_decision = build_unclear_pending_action_decision(
        str(pending_action.get("id", "")).strip(),
        notes=DEFAULT_INTENT_PARSER_UNAVAILABLE_REASON,
    )
    return build_pending_action_turn_result(
        pending_action,
        fallback_decision,
        resolution_key=resolution_key,
        user_text=latest_user_text,
    )


def build_pending_action_resolution_key(
    state: AgentState,
    pending_action: PendingAction,
) -> str:
    messages = state.get("messages") or []
    latest_user_text = extract_latest_human_text(state).strip()
    human_message_count = sum(1 for message in messages if isinstance(message, HumanMessage))
    pending_action_id = str(pending_action.get("id", "")).strip()
    content_hash = hashlib.sha256(latest_user_text.encode("utf-8")).hexdigest()
    return f"{pending_action_id}:{human_message_count}:{content_hash}"


def get_pending_action_resolution_key(state: AgentState) -> str | None:
    raw_value = state.get("pending_action_resolution_key")
    if isinstance(raw_value, str):
        normalized = raw_value.strip()
        if normalized:
            return normalized
    return None


def pending_action_cache_matches_state(
    state: AgentState,
    pending_action: PendingAction,
    decision: PendingActionDecision,
    *,
    resolution_key: str,
) -> bool:
    cached_resolution_key = get_pending_action_resolution_key(state)
    if cached_resolution_key != resolution_key:
        return False

    pending_action_id = str(pending_action.get("id", "")).strip()
    return str(decision.pending_action_id).strip() == pending_action_id


def hydrate_pending_action_turn_result(
    pending_action: PendingAction,
    decision: PendingActionDecision,
    *,
    resolution_key: str,
    execution_contract: ExecutionContract | None,
) -> PendingActionTurnResult:
    if decision.decision == "unrelated":
        return PendingActionTurnResult(
            pending_action=pending_action,
            pending_action_decision=decision,
            pending_action_resolution_key=resolution_key,
            execution_contract=None,
            validation=None,
            allow_fresh_routing=True,
        )

    resolved_contract = execution_contract
    if resolved_contract is None:
        resolution = resolve_pending_action_decision(
            pending_action,
            decision,
            user_text="",
        )
        resolved_contract = resolution["contract"]
    validation = (
        validate_execution_contract(pending_action, resolved_contract)
        if resolved_contract is not None
        else None
    )
    return PendingActionTurnResult(
        pending_action=pending_action,
        pending_action_decision=decision,
        pending_action_resolution_key=resolution_key,
        execution_contract=resolved_contract,
        validation=validation,
        allow_fresh_routing=False,
    )


def build_pending_action_turn_result(
    pending_action: PendingAction,
    decision: PendingActionDecision,
    *,
    resolution_key: str,
    user_text: str,
) -> PendingActionTurnResult:
    if decision.decision == "unrelated":
        return PendingActionTurnResult(
            pending_action=pending_action,
            pending_action_decision=decision,
            pending_action_resolution_key=resolution_key,
            execution_contract=None,
            validation=None,
            allow_fresh_routing=True,
        )

    resolution = resolve_pending_action_decision(
        pending_action,
        decision,
        user_text=user_text,
    )
    return PendingActionTurnResult(
        pending_action=pending_action,
        pending_action_decision=decision,
        pending_action_resolution_key=resolution_key,
        execution_contract=resolution["contract"],
        validation=resolution["validation"],
        allow_fresh_routing=False,
    )


def resolve_owned_pending_action_turn(
    state: AgentState,
    *,
    agent_name: str,
    pending_action_router: PendingActionRouter | None = None,
) -> tuple[PendingAction, PendingActionTurnResult] | tuple[None, None]:
    """Return (pending_action, turn) when this agent owns an active pending action
    that is ready for follow-up handling.

    Returns (None, None) when:
    - there is no active pending action
    - the pending action belongs to a different agent
    - the turn result allows fresh routing (i.e. the user changed topic)
    """
    pending_action = get_pending_action(state)
    if not (
        pending_action
        and pending_action.get("requested_by_agent") == agent_name
        and is_pending_action_active(pending_action)
    ):
        return None, None
    turn = resolve_pending_action_turn_from_state(state, pending_action_router=pending_action_router)
    if turn is None or turn.allow_fresh_routing:
        return None, None
    return pending_action, turn
