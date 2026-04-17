from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.context_window import token_count_with_estimation
from app.messages import strip_internal_reasoning, stringify_message_content
from app.memory.observability import emit_memory_telemetry
from app.model_request import build_model_request_messages, is_compact_boundary_message
from app.rehydration import (
    RUNTIME_REHYDRATION_METADATA_KEY,
    extract_runtime_rehydration_state_from_transcript,
)
from app.session_memory import (
    SessionMemoryRecord,
    build_session_memory_compaction_plan,
    resolve_safe_preserved_tail_start,
)
from interfaces.web.conversations import (
    TRANSCRIPT_TYPE_COMPACT_BOUNDARY,
    TRANSCRIPT_TYPE_MESSAGE,
    utc_now_iso,
)

DRAFT_ANALYSIS_BLOCK_PATTERN = re.compile(r"<draft_analysis>.*?</draft_analysis>", re.IGNORECASE | re.DOTALL)

logger = logging.getLogger(__name__)


class ContinuationSummaryDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_analysis: str = ""
    durable_summary: str = Field(
        description="Durable continuation context that should survive transcript compaction.",
    )
    open_loops: list[str] = Field(default_factory=list)
    user_preferences: list[str] = Field(default_factory=list)
    assistant_commitments: list[str] = Field(default_factory=list)
    attachment_notes: list[str] = Field(default_factory=list)

    @field_validator("durable_summary")
    @classmethod
    def validate_durable_summary(cls, value: str) -> str:
        cleaned = strip_draft_analysis_text(value)
        if not cleaned:
            raise ValueError("durable_summary must not be empty.")
        return cleaned

    @field_validator(
        "open_loops",
        "user_preferences",
        "assistant_commitments",
        "attachment_notes",
        mode="before",
    )
    @classmethod
    def normalize_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("Expected a list.")
        normalized = [strip_draft_analysis_text(item) for item in value]
        return [item for item in normalized if item]


@dataclass(frozen=True)
class ConversationCompactionBundle:
    prefix_messages: list[dict[str, Any]]
    boundary_message: dict[str, Any]
    summary_message: dict[str, Any]
    preserved_tail_messages: list[dict[str, Any]]
    attachments: list[dict[str, Any]]
    compacted_messages: list[dict[str, Any]]
    active_slice_start: int
    active_slice_count: int
    compacted_source_count: int
    used_session_memory: bool = False
    session_memory_record: SessionMemoryRecord | None = None


def compact_conversation(
    messages: list[dict[str, Any]] | list[Any],
    *,
    llm: Any | None = None,
    trigger: str = "manual",
    preserved_tail_count: int = 0,
    attachments: list[dict[str, Any] | str] | None = None,
    pre_tokens: int | None = None,
    summary_draft: ContinuationSummaryDraft | dict[str, Any] | None = None,
    session_memory: SessionMemoryRecord | dict[str, Any] | None = None,
) -> ConversationCompactionBundle:
    normalized_messages = [_normalize_transcript_message_payload(message) for message in messages]
    active_slice_count = 0
    compacted_source_count = 0
    resolved_preserved_tail_count = 0
    used_session_memory = False
    try:
        active_slice_start = _find_active_slice_start(normalized_messages)
        prefix_messages = [deepcopy(message) for message in normalized_messages[:active_slice_start]]
        active_slice = [deepcopy(message) for message in normalized_messages[active_slice_start:]]
        active_slice_count = len(active_slice)
        if not active_slice:
            raise ValueError("Cannot compact an empty active transcript slice.")

        normalized_attachments = _normalize_attachments(attachments)
        resolved_preserved_tail_count = min(
            max(int(preserved_tail_count or 0), 0),
            max(len(active_slice) - 1, 0),
        )
        session_memory_plan = build_session_memory_compaction_plan(
            active_slice,
            session_memory,
            preserved_tail_count=resolved_preserved_tail_count,
        )
        used_session_memory = session_memory_plan is not None
        session_memory_record = session_memory_plan.record if session_memory_plan is not None else None
        if session_memory_plan is not None:
            compacted_source_messages = [
                deepcopy(message)
                for message in active_slice[: session_memory_plan.compacted_source_count]
            ]
            preserved_tail_messages = [deepcopy(message) for message in session_memory_plan.preserved_tail_messages]
            resolved_preserved_tail_count = len(preserved_tail_messages)
        else:
            preserved_tail_start = resolve_safe_preserved_tail_start(
                active_slice,
                preserved_tail_count=resolved_preserved_tail_count,
            )
            compacted_source_messages = (
                active_slice[:preserved_tail_start]
                if preserved_tail_start < len(active_slice)
                else active_slice
            )
            preserved_tail_messages = (
                [deepcopy(message) for message in active_slice[preserved_tail_start:]]
                if preserved_tail_start < len(active_slice)
                else []
            )
            resolved_preserved_tail_count = len(preserved_tail_messages)

        if not compacted_source_messages:
            raise ValueError("Manual compaction must summarize at least one message.")

        compacted_source_count = len(compacted_source_messages)
        compaction_timestamp = utc_now_iso()
        estimated_pre_tokens = pre_tokens
        if estimated_pre_tokens is None:
            estimated_pre_tokens = token_count_with_estimation(active_slice).estimated_total_tokens

        if used_session_memory and session_memory_record is not None:
            formatted_summary = session_memory_record.summary_markdown
        else:
            resolved_summary_draft = _resolve_continuation_summary_draft(
                llm,
                compacted_source_messages,
                attachments=normalized_attachments,
                summary_draft=summary_draft,
            )
            formatted_summary = format_continuation_summary(
                resolved_summary_draft,
                attachments=normalized_attachments,
            )
        runtime_rehydration_state = extract_runtime_rehydration_state_from_transcript(
            normalized_messages,
            require_compact_boundary=False,
        )

        boundary_message = {
            "id": _generate_message_id(),
            "role": "system",
            "type": TRANSCRIPT_TYPE_COMPACT_BOUNDARY,
            "markdown": "",
            "created_at": compaction_timestamp,
            "metadata": {
                "trigger": str(trigger or "").strip() or "manual",
                "preTokens": int(estimated_pre_tokens),
                "preservedTail": {
                    "count": resolved_preserved_tail_count,
                    "messageIds": [message["id"] for message in preserved_tail_messages],
                },
            },
        }
        if normalized_attachments:
            boundary_message["metadata"]["attachments"] = deepcopy(normalized_attachments)

        summary_message = {
            "id": _generate_message_id(),
            "role": "assistant",
            "type": TRANSCRIPT_TYPE_MESSAGE,
            "markdown": formatted_summary,
            "created_at": compaction_timestamp,
            "metadata": {
                "compaction": {
                    "kind": "continuation_summary",
                    "source": "session_memory" if used_session_memory else "fresh_summary",
                    "sourceMessageIds": [message["id"] for message in compacted_source_messages],
                    "sourceMessageCount": len(compacted_source_messages),
                }
            },
        }
        if used_session_memory and session_memory_record is not None:
            summary_message["metadata"]["compaction"]["sessionMemory"] = {
                "updatedAt": session_memory_record.updated_at,
                "lastMessageId": session_memory_record.last_message_id,
                "coveredMessageCount": session_memory_record.covered_message_count,
                "coveredTokens": session_memory_record.covered_tokens,
                "source": session_memory_record.source,
            }
        if runtime_rehydration_state:
            summary_message["metadata"][RUNTIME_REHYDRATION_METADATA_KEY] = deepcopy(runtime_rehydration_state)
        if normalized_attachments:
            summary_message["metadata"]["attachments"] = deepcopy(normalized_attachments)

        compacted_messages = [
            *prefix_messages,
            boundary_message,
            summary_message,
            *preserved_tail_messages,
        ]
        bundle = ConversationCompactionBundle(
            prefix_messages=prefix_messages,
            boundary_message=boundary_message,
            summary_message=summary_message,
            preserved_tail_messages=preserved_tail_messages,
            attachments=normalized_attachments,
            compacted_messages=compacted_messages,
            active_slice_start=active_slice_start,
            active_slice_count=len(active_slice),
            compacted_source_count=len(compacted_source_messages),
            used_session_memory=used_session_memory,
            session_memory_record=session_memory_record,
        )
    except Exception as exc:
        emit_memory_telemetry(
            logger,
            "compaction.run",
            status="error",
            trigger=str(trigger or "").strip() or "manual",
            active_slice_count=active_slice_count,
            compacted_source_count=compacted_source_count,
            preserved_tail_count=resolved_preserved_tail_count,
            used_session_memory=used_session_memory,
            error=str(exc),
        )
        raise

    emit_memory_telemetry(
        logger,
        "compaction.run",
        trigger=str(trigger or "").strip() or "manual",
        active_slice_count=bundle.active_slice_count,
        compacted_source_count=bundle.compacted_source_count,
        preserved_tail_count=len(bundle.preserved_tail_messages),
        used_session_memory=bundle.used_session_memory,
        attachments_count=len(bundle.attachments),
    )
    return bundle


def build_continuation_summary_prompt(
    compacted_messages: list[dict[str, Any]] | list[Any],
    *,
    attachments: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        "You are preparing a durable continuation summary for a compacted conversation transcript.",
        "Return a structured summary that lets a future assistant continue the conversation safely after history is replaced.",
        "Keep only durable context. Exclude draft reasoning, private analysis, or speculative thoughts.",
        "Capture facts, current goals, unresolved questions, user preferences, and assistant commitments that still matter.",
        "Do not quote large verbatim passages from the conversation.",
        "If attachments are provided, mention only the durable facts they contribute.",
        "",
        "Conversation slice to summarize:",
        render_compaction_source_messages(compacted_messages),
    ]
    normalized_attachments = _normalize_attachments(attachments)
    if normalized_attachments:
        lines.extend(
            [
                "",
                "Related attachments:",
                *[
                    f"- {attachment.get('label') or attachment.get('name') or attachment.get('id') or 'attachment'}"
                    for attachment in normalized_attachments
                ],
            ]
        )
    return "\n".join(lines).strip()


def format_continuation_summary(
    summary: ContinuationSummaryDraft | dict[str, Any] | str,
    *,
    attachments: list[dict[str, Any] | str] | None = None,
) -> str:
    draft = _coerce_summary_draft(summary)
    lines = [
        "## Continuation Summary",
        strip_draft_analysis_text(draft.durable_summary),
    ]
    if draft.open_loops:
        lines.extend(["", "### Open Loops", *[f"- {item}" for item in draft.open_loops]])
    if draft.user_preferences:
        lines.extend(["", "### User Preferences", *[f"- {item}" for item in draft.user_preferences]])
    if draft.assistant_commitments:
        lines.extend(
            ["", "### Assistant Commitments", *[f"- {item}" for item in draft.assistant_commitments]]
        )

    attachment_lines = list(draft.attachment_notes)
    normalized_attachments = _normalize_attachments(attachments)
    if normalized_attachments and not attachment_lines:
        attachment_lines = [
            str(attachment.get("label") or attachment.get("name") or attachment.get("id") or "").strip()
            for attachment in normalized_attachments
            if str(attachment.get("label") or attachment.get("name") or attachment.get("id") or "").strip()
        ]
    if attachment_lines:
        lines.extend(["", "### Attachments", *[f"- {item}" for item in attachment_lines]])
    return "\n".join(lines).strip()


def strip_draft_analysis_text(text: Any) -> str:
    cleaned = strip_internal_reasoning(stringify_message_content(text))
    cleaned = DRAFT_ANALYSIS_BLOCK_PATTERN.sub("", cleaned)
    durable_heading_match = re.search(r"(?im)^(?:#+\s*)?durable summary\s*:?\s*$", cleaned)
    if durable_heading_match is not None:
        cleaned = cleaned[durable_heading_match.end() :].strip()
    cleaned = re.sub(r"(?im)^(?:#+\s*)?draft analysis\s*:?\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^(?:#+\s*)?analysis\s*:?\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^(?:#+\s*)?reasoning\s*:?\s*$", "", cleaned)
    return cleaned.strip()


def render_compaction_source_messages(messages: list[dict[str, Any]] | list[Any]) -> str:
    rendered_lines: list[str] = []
    for message in messages:
        normalized = _normalize_transcript_message_payload(message)
        content = strip_draft_analysis_text(normalized.get("markdown", ""))
        if not content:
            continue
        rendered_lines.append(f"- {normalized['role']}: {content}")
    return "\n".join(rendered_lines).strip() or "- (empty)"


def _resolve_continuation_summary_draft(
    llm: Any | None,
    compacted_messages: list[dict[str, Any]],
    *,
    attachments: list[dict[str, Any]] | None = None,
    summary_draft: ContinuationSummaryDraft | dict[str, Any] | None = None,
) -> ContinuationSummaryDraft:
    if summary_draft is not None:
        return _coerce_summary_draft(summary_draft)

    prompt = build_continuation_summary_prompt(compacted_messages, attachments=attachments)

    if llm is None or not hasattr(llm, "invoke"):
        return _fallback_summary_draft(compacted_messages, attachments=attachments)

    structured_builder = getattr(llm, "with_structured_output", None)
    structured_llm = structured_builder(ContinuationSummaryDraft) if callable(structured_builder) else None
    request_messages = build_model_request_messages(
        system_prompt=prompt,
        use_projection_pipeline=False,
    )
    if structured_llm is not None:
        try:
            return _coerce_summary_draft(structured_llm.invoke(request_messages))
        except Exception:
            pass
    try:
        raw_result = llm.invoke(request_messages)
    except Exception:
        return _fallback_summary_draft(compacted_messages, attachments=attachments)
    return _coerce_summary_draft(raw_result)


def _coerce_summary_draft(value: ContinuationSummaryDraft | dict[str, Any] | str | Any) -> ContinuationSummaryDraft:
    if isinstance(value, ContinuationSummaryDraft):
        return value
    if isinstance(value, dict):
        return ContinuationSummaryDraft.model_validate(value)
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return ContinuationSummaryDraft.model_validate(value.model_dump())
        except ValidationError:
            pass
    if hasattr(value, "final_answer"):
        return ContinuationSummaryDraft(durable_summary=str(getattr(value, "final_answer", "")).strip())
    if hasattr(value, "content"):
        return ContinuationSummaryDraft(durable_summary=stringify_message_content(getattr(value, "content", "")))
    return ContinuationSummaryDraft(durable_summary=stringify_message_content(value))


def _fallback_summary_draft(
    compacted_messages: list[dict[str, Any]],
    *,
    attachments: list[dict[str, Any]] | None = None,
) -> ContinuationSummaryDraft:
    durable_lines = render_compaction_source_messages(compacted_messages).splitlines()
    attachment_notes = [
        str(attachment.get("label") or attachment.get("name") or attachment.get("id") or "").strip()
        for attachment in _normalize_attachments(attachments)
        if str(attachment.get("label") or attachment.get("name") or attachment.get("id") or "").strip()
    ]
    return ContinuationSummaryDraft(
        durable_summary="\n".join(line[2:] if line.startswith("- ") else line for line in durable_lines[:8]).strip(),
        attachment_notes=attachment_notes,
    )


def _normalize_transcript_message_payload(message: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return {
            "id": str(message.get("id", "")).strip() or _generate_message_id(),
            "role": str(message.get("role", "")).strip(),
            "type": str(message.get("type", TRANSCRIPT_TYPE_MESSAGE)).strip() or TRANSCRIPT_TYPE_MESSAGE,
            "markdown": str(message.get("markdown", message.get("content", ""))),
            "created_at": str(message.get("created_at", "")).strip() or utc_now_iso(),
            "usage": deepcopy(message.get("usage")) if isinstance(message.get("usage"), dict) else None,
            "metadata": deepcopy(message.get("metadata")) if isinstance(message.get("metadata"), dict) else None,
            "tool_calls": deepcopy(message.get("tool_calls")) if isinstance(message.get("tool_calls"), list) else [],
            "tool_call_id": str(message.get("tool_call_id", "") or "").strip(),
        }

    role = str(getattr(message, "type", "")).strip().lower()
    normalized_role = role
    if role == "human":
        normalized_role = "user"
    elif role in {"ai", "assistant"}:
        normalized_role = "assistant"
    elif role == "system":
        normalized_role = "system"
    normalized_type = TRANSCRIPT_TYPE_MESSAGE
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    if isinstance(additional_kwargs, dict):
        normalized_type = (
            str(additional_kwargs.get("transcript_type", TRANSCRIPT_TYPE_MESSAGE)).strip()
            or TRANSCRIPT_TYPE_MESSAGE
        )
    metadata = deepcopy(additional_kwargs.get("metadata")) if isinstance(additional_kwargs, dict) and isinstance(additional_kwargs.get("metadata"), dict) else None
    usage = deepcopy(getattr(message, "usage_metadata", None)) if isinstance(getattr(message, "usage_metadata", None), dict) else None
    return {
        "id": str(getattr(message, "id", "") or "").strip() or _generate_message_id(),
        "role": normalized_role,
        "type": normalized_type,
        "markdown": stringify_message_content(getattr(message, "content", "")),
        "created_at": utc_now_iso(),
        "usage": usage,
        "metadata": metadata,
        "tool_calls": deepcopy(getattr(message, "tool_calls", [])) if isinstance(getattr(message, "tool_calls", []), list) else [],
        "tool_call_id": str(getattr(message, "tool_call_id", "") or "").strip(),
    }


def _normalize_attachments(attachments: list[dict[str, Any] | str] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for attachment in attachments or []:
        if isinstance(attachment, dict):
            normalized.append({str(key): deepcopy(value) for key, value in attachment.items()})
            continue
        label = str(attachment or "").strip()
        if label:
            normalized.append({"label": label})
    return normalized


def _find_active_slice_start(messages: list[dict[str, Any]]) -> int:
    last_boundary_index = -1
    for index, message in enumerate(messages):
        if is_compact_boundary_message(message):
            last_boundary_index = index
    return last_boundary_index + 1


def _generate_message_id() -> str:
    from uuid import uuid4

    return uuid4().hex
