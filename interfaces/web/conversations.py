from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from interfaces.web.markdown import render_markdown_html

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def derive_conversation_title(text: str, *, fallback: str = "New chat", limit: int = 60) -> str:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return fallback
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def transcript_to_langchain_messages(messages: list[dict[str, Any]] | list["TranscriptMessage"]) -> list[HumanMessage | AIMessage]:
    converted: list[HumanMessage | AIMessage] = []
    for message in messages:
        role = ""
        markdown = ""
        if isinstance(message, TranscriptMessage):
            role = message.role
            markdown = message.markdown
        elif isinstance(message, dict):
            role = str(message.get("role", "")).strip()
            markdown = str(message.get("markdown", ""))

        if role == "user":
            converted.append(HumanMessage(content=markdown))
        elif role == "assistant":
            converted.append(AIMessage(content=markdown))
    return converted


@dataclass
class TranscriptMessage:
    id: str
    role: str
    markdown: str
    html: str
    created_at: str


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
            conversation = self._conversations.get(conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(conversation_id)
            return self._serialize_conversation(conversation)

    def append_message(self, conversation_id: str, *, role: str, markdown: str) -> dict:
        with self._lock:
            conversation = self._conversations.get(conversation_id)
            if conversation is None:
                raise ConversationNotFoundError(conversation_id)

            timestamp = utc_now_iso()
            conversation.messages.append(
                TranscriptMessage(
                    id=uuid4().hex,
                    role=role,
                    markdown=markdown,
                    html=render_markdown_html(markdown),
                    created_at=timestamp,
                )
            )
            if role == "user" and len(conversation.messages) == 1:
                conversation.title = derive_conversation_title(markdown)
            conversation.updated_at = timestamp
            self._persist_locked()
            return self._serialize_conversation(conversation)

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

    def _persist_locked(self) -> None:
        if self._storage_path is None:
            return

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
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
                {
                    "id": message.id,
                    "role": message.role,
                    "markdown": message.markdown,
                    "created_at": message.created_at,
                }
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
        markdown = str(payload.get("markdown", ""))
        created_at = str(payload.get("created_at", "")).strip() or utc_now_iso()
        if role not in {"user", "assistant"}:
            return None
        return TranscriptMessage(
            id=message_id,
            role=role,
            markdown=markdown,
            html=render_markdown_html(markdown),
            created_at=created_at,
        )

    def _serialize_metadata(self, conversation: Conversation) -> dict:
        return {
            "conversation_id": conversation.conversation_id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
        }

    def _serialize_conversation(self, conversation: Conversation) -> dict:
        return {
            **self._serialize_metadata(conversation),
            "messages": [asdict(message) for message in conversation.messages],
        }
