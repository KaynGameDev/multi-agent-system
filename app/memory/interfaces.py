from __future__ import annotations

from typing import Protocol

from app.memory.types import (
    ConversationCompactionRequest,
    ConversationCompactionSummary,
    LongTermMemoryRecord,
    MemoryRetrievalQuery,
    MemoryRetrievalResult,
    SessionMemorySnapshot,
)


class SessionMemoryBackend(Protocol):
    def get(self, thread_id: str) -> SessionMemorySnapshot | None:
        ...

    def upsert(self, snapshot: SessionMemorySnapshot) -> SessionMemorySnapshot:
        ...

    def delete(self, thread_id: str) -> None:
        ...


class LongTermMemoryBackend(Protocol):
    def get(self, memory_id: str) -> LongTermMemoryRecord | None:
        ...

    def upsert(self, record: LongTermMemoryRecord) -> LongTermMemoryRecord:
        ...

    def delete(self, memory_id: str) -> None:
        ...


class MemoryRetriever(Protocol):
    def search(self, query: MemoryRetrievalQuery) -> list[MemoryRetrievalResult]:
        ...


class ConversationCompactor(Protocol):
    def compact(self, request: ConversationCompactionRequest) -> ConversationCompactionSummary:
        ...
