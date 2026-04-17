from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from app.config import Settings
from app.memory.agent_scope import resolve_agent_memory_context
from app.memory.long_term import FileLongTermMemoryStore
from app.memory.types import AgentMemoryScope, LongTermMemoryFile, LongTermMemoryWrite
from app.rehydration import RUNTIME_REHYDRATION_METADATA_KEY, TRANSCRIPT_TYPE_COMPACT_BOUNDARY
from app.tool_registry import TOOL_MEMORY_WRITE

DIRECT_MEMORY_WRITE_TOOL_NAMES = {"write_agent_memory"}
MAX_AUTOMATIC_DURABLE_MEMORIES_PER_TURN = 3
_NAME_PATTERNS = (
    re.compile(r"\bcall me (?P<value>[A-Za-z][A-Za-z0-9 ._-]{0,40})\b", re.IGNORECASE),
    re.compile(r"\bmy name is (?P<value>[A-Za-z][A-Za-z0-9 ._-]{0,40})\b", re.IGNORECASE),
)
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?:\r?\n)+|(?<=[.!?;])\s+")
_NON_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


def persist_durable_turn_memories(
    settings: Settings,
    *,
    agent_name: str,
    memory_scope: AgentMemoryScope | str,
    state: dict[str, Any] | None = None,
    transcript_messages: list[dict[str, Any]] | list[Any] | None = None,
    max_candidates: int = MAX_AUTOMATIC_DURABLE_MEMORIES_PER_TURN,
) -> list[LongTermMemoryFile]:
    candidates = extract_durable_turn_memories(
        transcript_messages,
        max_candidates=max_candidates,
    )
    if not candidates:
        return []

    context = resolve_agent_memory_context(
        settings,
        agent_name=agent_name,
        memory_scope=str(memory_scope or "").strip().lower(),  # type: ignore[arg-type]
        state=state,
    )
    store = FileLongTermMemoryStore(context.root_dir)
    saved: list[LongTermMemoryFile] = []
    for candidate in candidates:
        saved.append(store.upsert(candidate.model_dump(mode="python")))
    return saved


def extract_durable_turn_memories(
    transcript_messages: list[dict[str, Any]] | list[Any] | None,
    *,
    max_candidates: int = MAX_AUTOMATIC_DURABLE_MEMORIES_PER_TURN,
) -> list[LongTermMemoryWrite]:
    latest_turn = _extract_latest_completed_turn(transcript_messages)
    if not latest_turn:
        return []

    user_messages = [message for message in latest_turn if message["role"] == "user"]
    if not user_messages:
        return []

    candidate_map: dict[str, LongTermMemoryWrite] = {}
    for user_message in user_messages:
        for sentence in _split_sentences(user_message["markdown"]):
            for candidate in _extract_candidates_from_sentence(sentence):
                candidate_map[candidate.memory_id] = candidate
                if len(candidate_map) >= max(int(max_candidates or 0), 0):
                    return list(candidate_map.values())[: max(int(max_candidates or 0), 0)]

    return list(candidate_map.values())[: max(int(max_candidates or 0), 0)]


def turn_has_direct_memory_write(assistant_metadata: dict[str, Any] | None) -> bool:
    if not isinstance(assistant_metadata, dict):
        return False

    runtime_state = assistant_metadata.get(RUNTIME_REHYDRATION_METADATA_KEY)
    if not isinstance(runtime_state, dict):
        return False

    for key in ("tool_invocation", "tool_result"):
        if _envelope_is_memory_write(runtime_state.get(key)):
            return True

    trace = runtime_state.get("tool_execution_trace")
    if not isinstance(trace, list):
        return False

    for record in trace:
        if not isinstance(record, dict):
            continue
        if _envelope_is_memory_write(record.get("invocation")) or _envelope_is_memory_write(record.get("result")):
            return True
    return False


def _extract_latest_completed_turn(
    transcript_messages: list[dict[str, Any]] | list[Any] | None,
) -> list[dict[str, Any]]:
    visible_messages = [
        message
        for message in _normalize_messages_after_compact_boundary(transcript_messages)
        if message["type"] == "message" and message["role"] in {"user", "assistant"}
    ]
    if len(visible_messages) < 2 or visible_messages[-1]["role"] != "assistant":
        return []

    for index in range(len(visible_messages) - 2, -1, -1):
        if visible_messages[index]["role"] != "user":
            continue
        return visible_messages[index:]
    return []


def _normalize_messages_after_compact_boundary(
    transcript_messages: list[dict[str, Any]] | list[Any] | None,
) -> list[dict[str, Any]]:
    normalized_messages = list(transcript_messages or [])
    last_boundary_index = -1
    for index, message in enumerate(normalized_messages):
        normalized_type = _extract_message_type(message)
        if normalized_type == TRANSCRIPT_TYPE_COMPACT_BOUNDARY:
            last_boundary_index = index

    active_messages = normalized_messages[last_boundary_index + 1 :] if last_boundary_index >= 0 else normalized_messages
    return [_normalize_message(message) for message in active_messages]


def _normalize_message(message: dict[str, Any] | Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return {
            "role": str(message.get("role", "") or "").strip(),
            "type": str(message.get("type", "message") or "").strip() or "message",
            "markdown": str(message.get("markdown", message.get("content", "")) or ""),
            "metadata": deepcopy(message.get("metadata")) if isinstance(message.get("metadata"), dict) else None,
        }

    message_type = str(getattr(message, "type", "") or "").strip().lower()
    normalized_role = message_type
    if message_type == "human":
        normalized_role = "user"
    elif message_type in {"ai", "assistant"}:
        normalized_role = "assistant"
    elif message_type == "system":
        normalized_role = "system"

    metadata = getattr(message, "metadata", None)
    return {
        "role": normalized_role,
        "type": "message",
        "markdown": str(getattr(message, "content", "") or ""),
        "metadata": deepcopy(metadata) if isinstance(metadata, dict) else None,
    }


def _extract_message_type(message: dict[str, Any] | Any) -> str:
    if isinstance(message, dict):
        return str(message.get("type", "message") or "").strip() or "message"

    message_type = str(getattr(message, "type", "") or "").strip().lower()
    if message_type != "system":
        return "message"
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    if not isinstance(additional_kwargs, dict):
        return "message"
    return str(additional_kwargs.get("transcript_type", "") or "").strip() or "message"


def _split_sentences(text: str) -> list[str]:
    raw_parts = _SENTENCE_SPLIT_PATTERN.split(str(text or "").strip())
    normalized: list[str] = []
    for raw_part in raw_parts:
        cleaned = " ".join(raw_part.strip().split())
        if cleaned:
            normalized.append(cleaned)
    return normalized


def _extract_candidates_from_sentence(sentence: str) -> list[LongTermMemoryWrite]:
    sentence_text = str(sentence or "").strip()
    if not sentence_text:
        return []

    normalized_sentence = _normalize_sentence(sentence_text)
    lowered = normalized_sentence.lower()
    candidates: dict[str, LongTermMemoryWrite] = {}

    preferred_name = _extract_preferred_name(normalized_sentence)
    if preferred_name:
        candidates["profile/preferred-name"] = LongTermMemoryWrite(
            memory_id="profile/preferred-name",
            name="Preferred Name",
            description="User's preferred name for future replies.",
            memory_type="user",
            content_markdown=f"Call the user `{preferred_name}` in future replies.",
        )

    if _looks_like_preference_sentence(lowered):
        response_style = _extract_response_style(lowered)
        if response_style:
            candidates[response_style.memory_id] = response_style

        response_format = _extract_response_format(lowered)
        if response_format:
            candidates[response_format.memory_id] = response_format

        file_reference_preference = _extract_file_reference_preference(lowered)
        if file_reference_preference:
            candidates[file_reference_preference.memory_id] = file_reference_preference

        workflow_preference = _extract_workflow_preference(lowered)
        if workflow_preference:
            candidates[workflow_preference.memory_id] = workflow_preference

        language_preference = _extract_language_preference(lowered)
        if language_preference:
            candidates[language_preference.memory_id] = language_preference

    if _looks_like_feedback_sentence(lowered):
        feedback_id = _slugify_sentence(normalized_sentence)
        if feedback_id:
            candidates[f"feedback/{feedback_id}"] = LongTermMemoryWrite(
                memory_id=f"feedback/{feedback_id}",
                name="User Feedback",
                description="Behavior correction to preserve across future turns.",
                memory_type="feedback",
                content_markdown=normalized_sentence,
            )

    project_candidate = _extract_project_fact(normalized_sentence, lowered)
    if project_candidate is not None:
        candidates[project_candidate.memory_id] = project_candidate

    return list(candidates.values())


def _extract_preferred_name(sentence: str) -> str:
    for pattern in _NAME_PATTERNS:
        match = pattern.search(sentence)
        if match is None:
            continue
        value = str(match.group("value") or "").strip().strip("\"'`.,!?;: ")
        if not value:
            continue
        parts = [part for part in value.split() if part]
        if 0 < len(parts) <= 4:
            return " ".join(parts)
    return ""


def _extract_response_style(lowered: str) -> LongTermMemoryWrite | None:
    if any(token in lowered for token in ("concise", "brief", "shorter", "short ")):
        return LongTermMemoryWrite(
            memory_id="preferences/response-style",
            name="Response Style Preference",
            description="Preferred level of detail for future replies.",
            memory_type="user",
            content_markdown="Prefer concise updates by default.",
        )
    if any(token in lowered for token in ("detailed", "detail", "thorough", "more context")):
        return LongTermMemoryWrite(
            memory_id="preferences/response-style",
            name="Response Style Preference",
            description="Preferred level of detail for future replies.",
            memory_type="user",
            content_markdown="Prefer detailed replies with extra context when useful.",
        )
    return None


def _extract_response_format(lowered: str) -> LongTermMemoryWrite | None:
    if any(token in lowered for token in ("bullet", "bullets", "bullet-point", "bullet points", "list format")):
        return LongTermMemoryWrite(
            memory_id="preferences/response-format",
            name="Response Format Preference",
            description="Preferred reply format for future answers.",
            memory_type="user",
            content_markdown="Prefer bullet-point responses when the answer is list-shaped.",
        )
    return None


def _extract_file_reference_preference(lowered: str) -> LongTermMemoryWrite | None:
    if any(token in lowered for token in ("file reference", "file references", "file path", "file paths", "path references")):
        return LongTermMemoryWrite(
            memory_id="preferences/file-references",
            name="File Reference Preference",
            description="Preference for including file references in future code discussions.",
            memory_type="user",
            content_markdown="Include concrete file references when discussing repo changes.",
        )
    return None


def _extract_workflow_preference(lowered: str) -> LongTermMemoryWrite | None:
    if any(token in lowered for token in ("step by step", "step-by-step", "walk me through", "workflow")):
        return LongTermMemoryWrite(
            memory_id="preferences/workflow-style",
            name="Workflow Preference",
            description="Preferred workflow framing for future explanations.",
            memory_type="user",
            content_markdown="Prefer step-by-step workflow guidance for complex changes.",
        )
    return None


def _extract_language_preference(lowered: str) -> LongTermMemoryWrite | None:
    if "english" in lowered and any(token in lowered for token in ("reply", "respond", "answer", "write")):
        return LongTermMemoryWrite(
            memory_id="preferences/output-language",
            name="Output Language Preference",
            description="Preferred response language for future turns.",
            memory_type="user",
            content_markdown="Respond in English unless the user asks for another language.",
        )
    if "chinese" in lowered and any(token in lowered for token in ("reply", "respond", "answer", "write")):
        return LongTermMemoryWrite(
            memory_id="preferences/output-language",
            name="Output Language Preference",
            description="Preferred response language for future turns.",
            memory_type="user",
            content_markdown="Respond in Chinese unless the user asks for another language.",
        )
    return None


def _extract_project_fact(sentence: str, lowered: str) -> LongTermMemoryWrite | None:
    if not any(
        token in lowered
        for token in (
            "we are working on",
            "we're working on",
            "this project is",
            "the project is",
            "target market is",
            "target game is",
            "target feature is",
        )
    ):
        return None

    project_slug = _slugify_sentence(sentence)
    if not project_slug:
        return None

    return LongTermMemoryWrite(
        memory_id=f"project/{project_slug}",
        name="Project Context",
        description="Durable project context mentioned during conversation.",
        memory_type="project",
        content_markdown=sentence,
    )


def _looks_like_preference_sentence(lowered: str) -> bool:
    return any(
        cue in lowered
        for cue in (
            "prefer",
            "please keep",
            "please use",
            "please include",
            "please respond",
            "please answer",
            "default to",
            "from now on",
            "for future",
            "next time",
            "going forward",
            "always",
            "remember",
            "call me",
        )
    )


def _looks_like_feedback_sentence(lowered: str) -> bool:
    if not any(cue in lowered for cue in ("don't", "do not", "stop", "avoid", "that's wrong", "that was wrong", "instead")):
        return False
    return any(
        token in lowered
        for token in (
            "reply",
            "respond",
            "answer",
            "update",
            "use",
            "format",
            "assume",
            "remember",
            "call me",
        )
    )


def _normalize_sentence(sentence: str) -> str:
    cleaned = " ".join(str(sentence or "").strip().split())
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def _slugify_sentence(sentence: str) -> str:
    slug = _NON_SLUG_PATTERN.sub("-", str(sentence or "").strip().lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug:
        return ""
    parts = [part for part in slug.split("-") if part][:8]
    return "-".join(parts)


def _envelope_is_memory_write(envelope: Any) -> bool:
    if not isinstance(envelope, dict):
        return False
    tool_id = str(envelope.get("tool_id", "") or "").strip()
    tool_name = str(envelope.get("tool_name", "") or "").strip()
    return tool_id == TOOL_MEMORY_WRITE or tool_name in DIRECT_MEMORY_WRITE_TOOL_NAMES
