from __future__ import annotations

from typing import Protocol

from app.memory.types import (
    ConversationCompactionRequest,
    ConversationCompactionSummary,
    LongTermMemoryFile,
    LongTermMemoryIndexEntry,
    LongTermMemoryWrite,
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
    def list(self) -> list[LongTermMemoryIndexEntry]:
        ...

    def get(self, memory_id: str) -> LongTermMemoryFile | None:
        ...

    def upsert(self, memory: LongTermMemoryWrite) -> LongTermMemoryFile:
        ...

    def delete(self, memory_id: str) -> bool:
        ...


class MemoryRetriever(Protocol):
    def search(self, query: MemoryRetrievalQuery) -> list[MemoryRetrievalResult]:
        ...


class ConversationCompactor(Protocol):
    def compact(self, request: ConversationCompactionRequest) -> ConversationCompactionSummary:
        ...
