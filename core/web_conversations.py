from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from core.web_markdown import render_markdown_html


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def derive_conversation_title(text: str, *, fallback: str = "New chat", limit: int = 60) -> str:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return fallback
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


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
    def __init__(self) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._lock = RLock()

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
            return self._serialize_conversation(conversation)

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
