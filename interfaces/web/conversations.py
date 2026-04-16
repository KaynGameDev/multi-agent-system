from __future__ import annotations

from copy import deepcopy
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from interfaces.web.markdown import render_markdown_html

logger = logging.getLogger(__name__)
TRANSCRIPT_SCHEMA_VERSION = 2
TRANSCRIPT_TYPE_MESSAGE = "message"
TRANSCRIPT_TYPE_COMPACT_BOUNDARY = "compact_boundary"
VISIBLE_TRANSCRIPT_ROLES = {"user", "assistant"}
LEGACY_KB_WRITE_PERMISSION_LINES = (
    "由于我目前的权限主要是**读取和检索**知识库，无法直接在你的本地磁盘上创建新文件，建议你将以下内容保存到对应的目录下（参考 `KnowledgeTopology.md` 的结构）。",
    "我目前拥有**读取**和**检索**公司知识库的权限，但**没有直接修改或创建物理文件**的权限。",
)
LEGACY_KB_WRITE_HISTORY_NOTE = (
    "这是旧版本保存的历史回复。旧版本当时错误地声称自己不能直接写知识库文件；"
    "当前版本会先预览目标路径，并在你确认后再执行写入。\n\n"
    "注意：如果下面出现 `Docs/10_Projects/...` 之类路径，它们也是旧版本路径，仅供历史参考。"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def derive_conversation_title(text: str, *, fallback: str = "New chat", limit: int = 60) -> str:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return fallback
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def transcript_to_langchain_messages(
    messages: list[dict[str, Any]] | list["TranscriptMessage"],
) -> list[HumanMessage | AIMessage | SystemMessage]:
    converted: list[HumanMessage | AIMessage | SystemMessage] = []
    for message in messages:
        message_id = ""
        role = ""
        message_type = TRANSCRIPT_TYPE_MESSAGE
        markdown = ""
        usage = None
        metadata = None
        if isinstance(message, TranscriptMessage):
            message_id = message.id
            role = message.role
            message_type = message.type
            markdown = message.markdown
            usage = deepcopy(message.usage)
            metadata = deepcopy(message.metadata)
        elif isinstance(message, dict):
            message_id = str(message.get("id", "")).strip()
            role = str(message.get("role", "")).strip()
            message_type = str(message.get("type", TRANSCRIPT_TYPE_MESSAGE)).strip() or TRANSCRIPT_TYPE_MESSAGE
            markdown = str(message.get("markdown", ""))
            usage = _normalize_optional_mapping(message.get("usage"))
            metadata = _normalize_optional_mapping(message.get("metadata"))

        if message_type == TRANSCRIPT_TYPE_MESSAGE:
            if role == "user":
                converted.append(HumanMessage(content=markdown, id=message_id or None))
            elif role == "assistant":
                kwargs: dict[str, Any] = {
                    "content": markdown,
                    "id": message_id or None,
                }
                if usage is not None:
                    kwargs["usage_metadata"] = usage
                converted.append(AIMessage(**kwargs))
        elif message_type == TRANSCRIPT_TYPE_COMPACT_BOUNDARY and role == "system":
            additional_kwargs: dict[str, Any] = {
                "transcript_type": TRANSCRIPT_TYPE_COMPACT_BOUNDARY,
            }
            if metadata is not None:
                additional_kwargs["metadata"] = metadata
            converted.append(
                SystemMessage(
                    content=markdown,
                    id=message_id or None,
                    additional_kwargs=additional_kwargs,
                )
            )
    return converted


def normalize_legacy_assistant_markdown(markdown: str) -> str:
    normalized = str(markdown or "")
    replaced_legacy_intro = False
    for legacy_line in LEGACY_KB_WRITE_PERMISSION_LINES:
        if legacy_line in normalized:
            normalized = normalized.replace(legacy_line, LEGACY_KB_WRITE_HISTORY_NOTE, 1)
            replaced_legacy_intro = True

    if replaced_legacy_intro and "`Docs/10_Projects/" in normalized:
        normalized = re.sub(
            r"\*\*建议路径：\*\*\s*`Docs/10_Projects/",
            "**历史旧路径（已过时）：** `Docs/10_Projects/",
            normalized,
        )
    return normalized


@dataclass
class TranscriptMessage:
    id: str
    role: str
    type: str
    markdown: str
    html: str
    created_at: str
    usage: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    @property
    def is_visible(self) -> bool:
        return self.type == TRANSCRIPT_TYPE_MESSAGE and self.role in VISIBLE_TRANSCRIPT_ROLES


@dataclass
class Conversation:
    conversation_id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[TranscriptMessage] = field(default_factory=list)


class ConversationNotFoundError(KeyError):
    pass


class WebConversationStore:
    def __init__(self, storage_path: str | Path | None = None) -> None:
        self._storage_path = Path(storage_path).expanduser().resolve() if storage_path else None
        self._conversations: dict[str, Conversation] = {}
        self._lock = RLock()
        self._legacy_normalization_applied = False
        self._load()

    def create_conversation(self, *, title: str = "New chat") -> dict:
        with self._lock:
            timestamp = utc_now_iso()
            conversation = Conversation(
                conversation_id=uuid4().hex,
                title=title,
                created_at=timestamp,
                updated_at=timestamp,
            )
            self._conversations[conversation.conversation_id] = conversation
            self._persist_locked()
            return self._serialize_conversation(conversation)

    def list_conversations(self) -> list[dict]:
        with self._lock:
            conversations = sorted(
                self._conversations.values(),
                key=lambda conversation: conversation.updated_at,
                reverse=True,
            )
            return [self._serialize_metadata(conversation) for conversation in conversations]

    def get_conversation(self, conversation_id: str) -> dict:
        with self._lock:
            conversation = self._require_conversation_locked(conversation_id)
            return self._serialize_conversation(conversation)

    def get_full_conversation(self, conversation_id: str) -> dict:
        with self._lock:
            conversation = self._require_conversation_locked(conversation_id)
            return self._serialize_conversation(
                conversation,
                include_hidden_messages=True,
                include_internal_fields=True,
            )

    def append_message(self, conversation_id: str, *, role: str, markdown: str) -> dict:
        self.append_transcript_message(
            conversation_id,
            role=role,
            message_type=TRANSCRIPT_TYPE_MESSAGE,
            markdown=markdown,
        )
        return self.get_conversation(conversation_id)

    def append_transcript_message(
        self,
        conversation_id: str,
        *,
        role: str,
        message_type: str,
        markdown: str = "",
        usage: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        message_id: str | None = None,
        created_at: str | None = None,
    ) -> dict:
        with self._lock:
            conversation = self._require_conversation_locked(conversation_id)
            timestamp = str(created_at or "").strip() or utc_now_iso()
            normalized_message = self._build_transcript_message(
                role=role,
                message_type=message_type,
                markdown=markdown,
                usage=usage,
                metadata=metadata,
                message_id=message_id,
                created_at=timestamp,
            )
            is_first_visible_user_message = (
                normalized_message.type == TRANSCRIPT_TYPE_MESSAGE
                and normalized_message.role == "user"
                and not any(message.is_visible for message in conversation.messages)
            )
            conversation.messages.append(normalized_message)
            if is_first_visible_user_message:
                conversation.title = derive_conversation_title(markdown)
            conversation.updated_at = normalized_message.created_at
            self._persist_locked()
            return self._serialize_conversation(
                conversation,
                include_hidden_messages=True,
                include_internal_fields=True,
            )

    def append_compact_boundary(
        self,
        conversation_id: str,
        *,
        trigger: str,
        pre_tokens: int | float | str,
        preserved_tail: dict[str, Any] | None = None,
    ) -> dict:
        metadata: dict[str, Any] = {
            "trigger": str(trigger).strip(),
            "preTokens": pre_tokens,
        }
        if preserved_tail is not None:
            metadata["preservedTail"] = deepcopy(preserved_tail)
        return self.append_transcript_message(
            conversation_id,
            role="system",
            message_type=TRANSCRIPT_TYPE_COMPACT_BOUNDARY,
            metadata=metadata,
        )

    def replace_transcript(
        self,
        conversation_id: str,
        *,
        messages: list[dict[str, Any]] | list[TranscriptMessage],
        updated_at: str | None = None,
    ) -> dict:
        with self._lock:
            conversation = self._require_conversation_locked(conversation_id)
            rewritten_messages: list[TranscriptMessage] = []
            for message in messages:
                if isinstance(message, TranscriptMessage):
                    rewritten_messages.append(message)
                    continue
                if not isinstance(message, dict):
                    continue
                deserialized = self._deserialize_message(message)
                if deserialized is not None:
                    rewritten_messages.append(deserialized)
            conversation.messages = rewritten_messages
            conversation.updated_at = str(updated_at or "").strip() or utc_now_iso()
            self._persist_locked()
            return self._serialize_conversation(
                conversation,
                include_hidden_messages=True,
                include_internal_fields=True,
            )

    def rename_conversation(self, conversation_id: str, *, title: str) -> dict:
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(conversation_id)

            conversation.title = self._normalize_title(title)
            self._persist_locked()
            return self._serialize_conversation(conversation)

    def delete_conversation(self, conversation_id: str) -> None:
        with self._lock:
            if conversation_id not in self._conversations:
                raise ConversationNotFoundError(conversation_id)

            del self._conversations[conversation_id]
            self._persist_locked()

    def _load(self) -> None:
        if self._storage_path is None or not self._storage_path.exists():
            return

        try:
            payload = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load web conversation store path=%s", self._storage_path, exc_info=True)
            return

        raw_conversations = payload.get("conversations", []) if isinstance(payload, dict) else []
        if not isinstance(raw_conversations, list):
            logger.warning("Ignoring invalid web conversation store payload path=%s", self._storage_path)
            return

        conversations: dict[str, Conversation] = {}
        for item in raw_conversations:
            conversation = self._deserialize_conversation(item)
            if conversation is None:
                continue
            conversations[conversation.conversation_id] = conversation
        self._conversations = conversations
        if self._legacy_normalization_applied:
            self._persist_locked()
            self._legacy_normalization_applied = False

    def _persist_locked(self) -> None:
        if self._storage_path is None:
            return

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": TRANSCRIPT_SCHEMA_VERSION,
            "conversations": [
                self._serialize_conversation_for_disk(conversation)
                for conversation in sorted(
                    self._conversations.values(),
                    key=lambda conversation: conversation.updated_at,
                )
            ]
        }
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(self._storage_path.parent),
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            temp_path = Path(handle.name)
        temp_path.replace(self._storage_path)

    def _serialize_conversation_for_disk(self, conversation: Conversation) -> dict:
        return {
            "conversation_id": conversation.conversation_id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "messages": [
                self._serialize_message(
                    message,
                    include_html=False,
                    include_internal_fields=True,
                )
                for message in conversation.messages
            ],
        }

    def _deserialize_conversation(self, payload: object) -> Conversation | None:
        if not isinstance(payload, dict):
            return None
        conversation_id = str(payload.get("conversation_id", "")).strip()
        if not conversation_id:
            return None
        title = str(payload.get("title", "")).strip() or "New chat"
        created_at = str(payload.get("created_at", "")).strip() or utc_now_iso()
        updated_at = str(payload.get("updated_at", "")).strip() or created_at
        raw_messages = payload.get("messages", [])
        messages: list[TranscriptMessage] = []
        if isinstance(raw_messages, list):
            for raw_message in raw_messages:
                message = self._deserialize_message(raw_message)
                if message is not None:
                    messages.append(message)
        return Conversation(
            conversation_id=conversation_id,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            messages=messages,
        )

    def _deserialize_message(self, payload: object) -> TranscriptMessage | None:
        if not isinstance(payload, dict):
            return None
        message_id = str(payload.get("id", "")).strip() or uuid4().hex
        role = str(payload.get("role", "")).strip()
        message_type = str(payload.get("type", TRANSCRIPT_TYPE_MESSAGE)).strip() or TRANSCRIPT_TYPE_MESSAGE
        markdown = str(payload.get("markdown", ""))
        created_at = str(payload.get("created_at", "")).strip() or utc_now_iso()
        usage = _normalize_optional_mapping(payload.get("usage"))
        metadata = _normalize_optional_mapping(payload.get("metadata"))
        try:
            return self._build_transcript_message(
                role=role,
                message_type=message_type,
                markdown=markdown,
                usage=usage,
                metadata=metadata,
                message_id=message_id,
                created_at=created_at,
                allow_legacy_assistant_normalization=True,
            )
        except ValueError:
            return None

    def _serialize_metadata(self, conversation: Conversation) -> dict:
        return {
            "conversation_id": conversation.conversation_id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
        }

    def _serialize_conversation(
        self,
        conversation: Conversation,
        *,
        include_hidden_messages: bool = False,
        include_internal_fields: bool = False,
    ) -> dict:
        return {
            **self._serialize_metadata(conversation),
            "messages": [
                self._serialize_message(
                    message,
                    include_internal_fields=include_internal_fields,
                )
                for message in conversation.messages
                if include_hidden_messages or message.is_visible
            ],
        }

    def _serialize_message(
        self,
        message: TranscriptMessage,
        *,
        include_html: bool = True,
        include_internal_fields: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": message.id,
            "role": message.role,
            "type": message.type,
            "markdown": message.markdown,
            "created_at": message.created_at,
        }
        if include_html:
            payload["html"] = message.html
        if message.usage is not None:
            payload["usage"] = deepcopy(message.usage)
        if include_internal_fields and message.metadata is not None:
            payload["metadata"] = deepcopy(message.metadata)
        return payload

    def _require_conversation_locked(self, conversation_id: str) -> Conversation:
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        return conversation

    def _build_transcript_message(
        self,
        *,
        role: str,
        message_type: str,
        markdown: str,
        usage: dict[str, Any] | None,
        metadata: dict[str, Any] | None,
        message_id: str | None,
        created_at: str,
        allow_legacy_assistant_normalization: bool = False,
    ) -> TranscriptMessage:
        normalized_role = str(role).strip()
        normalized_type = str(message_type).strip() or TRANSCRIPT_TYPE_MESSAGE
        normalized_markdown = str(markdown or "")
        if normalized_type == TRANSCRIPT_TYPE_MESSAGE:
            if normalized_role not in VISIBLE_TRANSCRIPT_ROLES:
                raise ValueError(f"Unsupported message role: {normalized_role}")
            if normalized_role == "assistant" and allow_legacy_assistant_normalization:
                rewritten_markdown = normalize_legacy_assistant_markdown(normalized_markdown)
                if rewritten_markdown != normalized_markdown:
                    self._legacy_normalization_applied = True
                normalized_markdown = rewritten_markdown
        elif normalized_type == TRANSCRIPT_TYPE_COMPACT_BOUNDARY:
            if normalized_role != "system":
                raise ValueError(f"Unsupported compact boundary role: {normalized_role}")
            normalized_markdown = ""
        else:
            raise ValueError(f"Unsupported message type: {normalized_type}")

        return TranscriptMessage(
            id=str(message_id or "").strip() or uuid4().hex,
            role=normalized_role,
            type=normalized_type,
            markdown=normalized_markdown,
            html=_render_transcript_html(normalized_markdown, message_type=normalized_type),
            created_at=str(created_at or "").strip() or utc_now_iso(),
            usage=_normalize_optional_mapping(usage),
            metadata=_normalize_optional_mapping(metadata),
        )

    def _normalize_title(self, title: str) -> str:
        cleaned = " ".join(str(title or "").strip().split())
        return cleaned or "New chat"


def _normalize_optional_mapping(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    normalized = {str(key): deepcopy(item) for key, item in value.items()}
    return normalized or None


def _render_transcript_html(markdown: str, *, message_type: str) -> str:
    if message_type != TRANSCRIPT_TYPE_MESSAGE:
        return ""
    return render_markdown_html(markdown)
