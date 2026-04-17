from __future__ import annotations

import ast
from collections.abc import Sequence
import json
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.contracts import (
    AssistantRequest,
    PendingAction,
    PendingActionDecision,
    build_fallback_assistant_request,
    build_unclear_pending_action_decision,
    validate_assistant_request,
    validate_pending_action_decision,
)
from app.contracts.assistant_request import AssistantRequestDomain
from app.contracts.pending_action_decision import PendingActionDecisionKind
from app.messages import stringify_message_content
from app.pending_actions import get_pending_action_selection_options
from app.prompt_loader import load_prompt_text

from .model_config import IntentParserModelConfig

logger = logging.getLogger(__name__)

DEFAULT_ASSISTANT_REQUEST_LOW_CONFIDENCE_REASON = (
    "The assistant-request parser was not confident enough to classify the request safely."
)
DEFAULT_PENDING_ACTION_LOW_CONFIDENCE_REASON = (
    "The pending-action decision parser was not confident enough to interpret the reply safely."
)
DEFAULT_INTENT_PARSER_UNAVAILABLE_REASON = "The lightweight intent parser was unavailable."
DEFAULT_INTENT_PARSER_MALFORMED_REASON = "The lightweight intent parser returned malformed output."


class AssistantRequestCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_goal: str
    likely_domain: AssistantRequestDomain
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str | None = None

    @field_validator("user_goal")
    @classmethod
    def validate_user_goal(cls, value: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError("user_goal must not be empty.")
        return cleaned

    @field_validator("notes")
    @classmethod
    def validate_notes(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None


class PendingActionDecisionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: PendingActionDecisionKind
    notes: str | None = None
    selected_item_id: str | None = None
    constraints: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("notes", "selected_item_id")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @field_validator("constraints", mode="before")
    @classmethod
    def validate_constraints(cls, value: Any) -> list[str]:
        if value is None:
            raise ValueError("constraints must be provided.")
        if not isinstance(value, list):
            raise ValueError("constraints must be a list.")
        return [str(item).strip() for item in value if str(item).strip()]


class IntentParser:
    def __init__(
        self,
        llm: Any | None,
        *,
        backup_llm: Any | None = None,
        config: IntentParserModelConfig | None = None,
    ) -> None:
        self.llm = llm
        self.backup_llm = None if backup_llm is llm else backup_llm
        self.config = config or IntentParserModelConfig()
        self._assistant_request_parser = build_structured_output_parser(llm, AssistantRequestCandidate)
        self._pending_action_parser = build_structured_output_parser(llm, PendingActionDecisionCandidate)
        self._assistant_request_backup_parser = build_structured_output_parser(
            self.backup_llm,
            AssistantRequestCandidate,
        )
        self._pending_action_backup_parser = build_structured_output_parser(
            self.backup_llm,
            PendingActionDecisionCandidate,
        )

    def parse_assistant_request(
        self,
        user_message: str,
        *,
        recent_messages: Sequence[str] | None = None,
        routing_context: dict[str, Any] | None = None,
    ) -> AssistantRequest:
        normalized_user_message = normalize_user_message(user_message)
        prompt = build_assistant_request_prompt(
            normalized_user_message,
            recent_messages=recent_messages,
            routing_context=routing_context,
            prompt_path=self.config.assistant_request_prompt_path,
        )
        failure_reason = DEFAULT_INTENT_PARSER_UNAVAILABLE_REASON
        assistant_request: AssistantRequest | None = None
        for parser_role, parser, llm in self._iter_parser_attempts(
            self._assistant_request_parser,
            self._assistant_request_backup_parser,
        ):
            parsed, failure_reason = self._invoke_parser_once(
                parser,
                prompt,
                llm=llm,
                parser_role=parser_role,
            )
            if parsed is None:
                if parser_role == "primary" and self.backup_llm is not None:
                    logger.warning(
                        "AssistantRequest parser primary LLM failed; retrying with backup LLM. reason=%s",
                        failure_reason,
                    )
                continue
            assistant_request = self._build_assistant_request_from_parsed(
                parsed,
            )
            if assistant_request is not None:
                break
            failure_reason = DEFAULT_INTENT_PARSER_MALFORMED_REASON
            if parser_role == "primary" and self.backup_llm is not None:
                logger.warning(
                    "AssistantRequest parser primary LLM returned malformed output; retrying with backup LLM."
                )
                continue
            break

        if assistant_request is None:
            logger.warning(
                "AssistantRequest parser failed; using general fallback. reason=%s",
                failure_reason,
            )
            return build_fallback_assistant_request(normalized_user_message, notes=failure_reason)

        if assistant_request.confidence < self.config.confidence_threshold:
            logger.info(
                "AssistantRequest parser confidence below threshold; preserving parsed domain for router. confidence=%s threshold=%s domain=%s",
                assistant_request.confidence,
                self.config.confidence_threshold,
                assistant_request.likely_domain,
            )

        return assistant_request

    def parse_pending_action_decision(
        self,
        pending_action: PendingAction,
        user_message: str,
    ) -> PendingActionDecision:
        pending_action_id = str(pending_action.get("id", "")).strip()
        normalized_user_message = normalize_user_message(user_message)
        prompt = load_prompt_text(
            self.config.pending_action_prompt_path,
            pending_action_id=pending_action_id or "unknown_pending_action",
            pending_action_summary=str(pending_action.get("summary", "")).strip() or "(no summary)",
            selection_items=render_selection_items(pending_action),
            user_message=normalized_user_message or "(empty)",
        )
        failure_reason = DEFAULT_INTENT_PARSER_UNAVAILABLE_REASON
        pending_action_decision: PendingActionDecision | None = None
        for parser_role, parser, llm in self._iter_parser_attempts(
            self._pending_action_parser,
            self._pending_action_backup_parser,
        ):
            parsed, failure_reason = self._invoke_parser_once(
                parser,
                prompt,
                llm=llm,
                parser_role=parser_role,
            )
            if parsed is None:
                if parser_role == "primary" and self.backup_llm is not None:
                    logger.warning(
                        "PendingActionDecision parser primary LLM failed; retrying with backup LLM. pending_action_id=%s reason=%s",
                        pending_action_id,
                        failure_reason,
                    )
                continue
            pending_action_decision = self._build_pending_action_decision_from_parsed(
                parsed,
                pending_action_id,
            )
            if pending_action_decision is not None:
                return pending_action_decision
            failure_reason = DEFAULT_INTENT_PARSER_MALFORMED_REASON
            if parser_role == "primary" and self.backup_llm is not None:
                logger.warning(
                    "PendingActionDecision parser primary LLM returned malformed output; retrying with backup LLM. pending_action_id=%s",
                    pending_action_id,
                )
                continue
            break

        logger.warning(
            "PendingActionDecision parser failed; using unclear decision. pending_action_id=%s reason=%s",
            pending_action_id,
            failure_reason,
        )
        return build_unclear_pending_action_decision(pending_action_id, notes=failure_reason)

    def _iter_parser_attempts(
        self,
        primary_parser: Any,
        backup_parser: Any | None,
    ) -> tuple[tuple[str, Any, Any | None], ...]:
        attempts: list[tuple[str, Any, Any | None]] = [
            ("primary", primary_parser, self.llm),
        ]
        if self.backup_llm is not None:
            attempts.append(("backup", backup_parser, self.backup_llm))
        return tuple(attempts)

    def _build_assistant_request_from_parsed(
        self,
        parsed: Any,
    ) -> AssistantRequest | None:
        try:
            candidate = AssistantRequestCandidate.model_validate(parsed)
        except ValidationError:
            logger.warning(
                "AssistantRequest parser returned malformed output during candidate validation."
            )
            return None

        try:
            return validate_assistant_request(
                {
                    "type": "assistant_request",
                    "user_goal": candidate.user_goal,
                    "likely_domain": candidate.likely_domain,
                    "confidence": candidate.confidence,
                    "notes": candidate.notes,
                }
            )
        except ValidationError:
            logger.warning(
                "AssistantRequest parser failed schema validation after normalization."
            )
            return None

    def _build_pending_action_decision_from_parsed(
        self,
        parsed: Any,
        pending_action_id: str,
    ) -> PendingActionDecision | None:
        try:
            candidate = PendingActionDecisionCandidate.model_validate(parsed)
        except ValidationError:
            logger.warning(
                "PendingActionDecision parser returned malformed output during candidate validation. pending_action_id=%s",
                pending_action_id,
            )
            return None

        if candidate.confidence < self.config.confidence_threshold:
            logger.info(
                "PendingActionDecision parser confidence below threshold; using unclear decision. pending_action_id=%s confidence=%s threshold=%s",
                pending_action_id,
                candidate.confidence,
                self.config.confidence_threshold,
            )
            return build_unclear_pending_action_decision(
                pending_action_id,
                notes=candidate.notes or DEFAULT_PENDING_ACTION_LOW_CONFIDENCE_REASON,
            )

        try:
            return validate_pending_action_decision(
                {
                    "type": "pending_action_decision",
                    "pending_action_id": pending_action_id,
                    "decision": candidate.decision,
                    "notes": candidate.notes,
                    "selected_item_id": candidate.selected_item_id,
                    "constraints": candidate.constraints,
                }
            )
        except ValidationError:
            logger.warning(
                "PendingActionDecision parser failed schema validation. pending_action_id=%s",
                pending_action_id,
            )
            return None

    def _invoke_parser_once(
        self,
        parser: Any,
        prompt: str,
        *,
        llm: Any | None,
        parser_role: str,
    ) -> tuple[Any | None, str]:
        if parser is None:
            logger.warning(
                "Intent parser was unavailable because no structured parser was configured. parser_role=%s",
                parser_role,
            )
            raw_recovered = self._recover_from_raw_llm(prompt, llm=llm)
            if raw_recovered is not None:
                return raw_recovered, ""
            return None, DEFAULT_INTENT_PARSER_UNAVAILABLE_REASON

        latest_failure_reason = DEFAULT_INTENT_PARSER_UNAVAILABLE_REASON
        for attempt in range(1, self.config.max_parse_attempts + 1):
            try:
                result = parser.invoke(prompt)
                normalized = normalize_parser_invoke_result(result)
                if normalized is not None:
                    return normalized, ""
                latest_failure_reason = DEFAULT_INTENT_PARSER_MALFORMED_REASON
            except ValidationError as exc:
                latest_failure_reason = DEFAULT_INTENT_PARSER_MALFORMED_REASON
                logger.warning(
                    "Intent parser structured output validation failed. parser_role=%s attempt=%s max_attempts=%s error=%s",
                    parser_role,
                    attempt,
                    self.config.max_parse_attempts,
                    exc,
                )
            except Exception as exc:
                latest_failure_reason = DEFAULT_INTENT_PARSER_UNAVAILABLE_REASON
                logger.warning(
                    "Intent parser invocation failed. parser_role=%s attempt=%s max_attempts=%s error=%s",
                    parser_role,
                    attempt,
                    self.config.max_parse_attempts,
                    exc,
                )
            raw_recovered = self._recover_from_raw_llm(prompt, llm=llm)
            if raw_recovered is not None:
                return raw_recovered, ""
        return None, latest_failure_reason

    def _recover_from_raw_llm(self, prompt: str, *, llm: Any | None) -> Any | None:
        if llm is None:
            return None

        raw_invoke = getattr(llm, "invoke", None)
        if not callable(raw_invoke):
            return None

        try:
            raw_result = raw_invoke(prompt)
        except Exception as exc:
            logger.warning("Intent parser raw fallback invocation failed. error=%s", exc)
            return None

        recovered = recover_structured_output_payload(raw_result)
        if recovered is None:
            return None

        logger.warning("Intent parser recovered structured payload from raw LLM output.")
        return recovered


def build_structured_output_parser(llm: Any | None, schema: type[BaseModel]) -> Any | None:
    structured_output = getattr(llm, "with_structured_output", None) if llm is not None else None
    if not callable(structured_output):
        return None

    candidate_kwargs = (
        {"method": "json_schema", "include_raw": True},
        {"include_raw": True},
        {"method": "json_schema"},
        {},
    )
    for kwargs in candidate_kwargs:
        try:
            return structured_output(schema, **kwargs)
        except TypeError:
            continue
        except ValueError:
            continue
    return None


def normalize_parser_invoke_result(result: Any) -> Any | None:
    if isinstance(result, dict) and {"raw", "parsed", "parsing_error"}.issubset(result.keys()):
        parsed = result.get("parsed")
        if parsed is not None:
            return parsed
        raw = result.get("raw")
        recovered = recover_structured_output_payload(raw)
        if recovered is not None:
            logger.warning(
                "Intent parser recovered malformed structured output from raw content. parsing_error=%s",
                result.get("parsing_error"),
            )
            return recovered
        return None

    recovered = recover_structured_output_payload(result)
    if recovered is not None:
        return recovered
    return result


def recover_structured_output_payload(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)

    content = getattr(value, "content", None)
    if content is not None:
        text = stringify_message_content(content)
        if not text:
            return None

        for candidate in iter_json_like_snippets(text):
            parsed = parse_json_like_payload(candidate)
            if parsed is not None:
                return parsed
        return None

    if isinstance(value, BaseModel):
        return value

    content = value
    text = stringify_message_content(content)
    if not text:
        return None

    for candidate in iter_json_like_snippets(text):
        parsed = parse_json_like_payload(candidate)
        if parsed is not None:
            return parsed
    return None


def iter_json_like_snippets(text: str) -> list[str]:
    snippets: list[str] = []
    cleaned = str(text or "").strip()
    if not cleaned:
        return snippets
    snippets.append(cleaned)

    fence_marker = "```"
    if fence_marker in cleaned:
        parts = cleaned.split(fence_marker)
        for index in range(1, len(parts), 2):
            block = parts[index].strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block:
                snippets.append(block)

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        snippets.append(cleaned[first_brace : last_brace + 1])

    deduped: list[str] = []
    seen: set[str] = set()
    for snippet in snippets:
        normalized = snippet.strip()
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped


def parse_json_like_payload(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
    if isinstance(parsed, dict):
        return parsed
    return None


def normalize_user_message(user_message: str) -> str:
    return str(user_message or "").strip()


def build_assistant_request_prompt(
    user_message: str,
    *,
    recent_messages: Sequence[str] | None = None,
    routing_context: dict[str, Any] | None = None,
    prompt_path=None,
) -> str:
    return load_prompt_text(
        prompt_path or IntentParserModelConfig().assistant_request_prompt_path,
        user_message=normalize_user_message(user_message) or "(empty)",
        recent_context=render_recent_messages(recent_messages),
        routing_context=render_routing_context(routing_context),
    )


def render_recent_messages(recent_messages: Sequence[str] | None) -> str:
    if not recent_messages:
        return "- (none)"
    cleaned = [str(message).strip() for message in recent_messages if str(message).strip()]
    if not cleaned:
        return "- (none)"
    return "\n".join(f"- {message}" for message in cleaned[-6:])


def render_routing_context(routing_context: dict[str, Any] | None) -> str:
    if not isinstance(routing_context, dict) or not routing_context:
        return "- interface_name: unknown\n- uploaded_files_count: 0\n- conversion_session_active: false"

    interface_name = str(routing_context.get("interface_name", "")).strip().lower() or "unknown"
    uploaded_files_count = int(routing_context.get("uploaded_files_count", 0) or 0)
    conversion_session_active = bool(routing_context.get("conversion_session_active", False))
    return (
        f"- interface_name: {interface_name}\n"
        f"- uploaded_files_count: {uploaded_files_count}\n"
        f"- conversion_session_active: {str(conversion_session_active).lower()}"
    )


def render_selection_items(pending_action: PendingAction) -> str:
    options = get_pending_action_selection_options(pending_action)
    if not options:
        return "- (none)"

    rendered: list[str] = []
    for index, option in enumerate(options, start=1):
        selected_item_id = derive_selection_item_id(option, fallback_index=index)
        label = str(option.get("label") or option.get("value") or selected_item_id).strip()
        rendered.append(f"- id: {selected_item_id} | label: {label}")
    return "\n".join(rendered)


def derive_selection_item_id(option: dict[str, Any], *, fallback_index: int) -> str:
    for candidate in (
        option.get("id"),
        option.get("value"),
        option.get("label"),
        (option.get("payload") or {}).get("id") if isinstance(option.get("payload"), dict) else None,
    ):
        cleaned = str(candidate or "").strip()
        if cleaned:
            return cleaned
    return f"item_{fallback_index}"
