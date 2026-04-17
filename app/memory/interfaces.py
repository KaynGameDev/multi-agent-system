from __future__ import annotations

from typing import Protocol

from app.memory.types import (
    ConversationCompactionRequest,
    ConversationCompactionSummary,
    LongTermMemoryFile,
    LongTermMemoryConsolidationSummary,
    LongTermMemoryIndexEntry,
    LongTermMemoryWrite,
    MemoryRetrievalQuery,
    MemoryRetrievalResult,
    SessionMemoryFile,
    SessionMemoryFileUpdate,
    SessionMemorySnapshot,
)


class SessionMemoryBackend(Protocol):
    def get(self, thread_id: str) -> SessionMemorySnapshot | None:
        ...

    def upsert(self, snapshot: SessionMemorySnapshot) -> SessionMemorySnapshot:
        ...

    def delete(self, thread_id: str) -> None:
        ...


class SessionMemoryFileBackend(Protocol):
    def ensure(self, thread_id: str) -> SessionMemoryFile:
        ...

    def get(self, thread_id: str) -> SessionMemoryFile | None:
        ...

    def update(self, thread_id: str, patch: SessionMemoryFileUpdate) -> SessionMemoryFile:
        ...

    def delete(self, thread_id: str) -> bool:
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


class MemoryConsolidator(Protocol):
    def consolidate(self) -> LongTermMemoryConsolidationSummary:
        ...


class ConversationCompactor(Protocol):
    def compact(self, request: ConversationCompactionRequest) -> ConversationCompactionSummary:
        ...
