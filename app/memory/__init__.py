"""Shared memory subsystem scaffolding for future Jade runtime work."""

from app.memory.interfaces import (
    ConversationCompactor,
    LongTermMemoryBackend,
    MemoryRetriever,
    SessionMemoryBackend,
)
from app.memory.paths import (
    MemorySubsystemPaths,
    build_memory_subsystem_paths,
    resolve_compaction_dir,
    resolve_long_term_memory_dir,
    resolve_memory_work_dir,
    resolve_retrieval_dir,
    resolve_session_memory_store_path,
)
from app.memory.types import (
    ConversationCompactionRequest,
    ConversationCompactionSummary,
    LongTermMemoryRecord,
    MemoryReference,
    MemoryRetrievalQuery,
    MemoryRetrievalResult,
    MemoryScope,
    SessionMemorySnapshot,
)

__all__ = [
    "ConversationCompactionRequest",
    "ConversationCompactionSummary",
    "ConversationCompactor",
    "LongTermMemoryBackend",
    "LongTermMemoryRecord",
    "MemoryReference",
    "MemoryRetriever",
    "MemoryRetrievalQuery",
    "MemoryRetrievalResult",
    "MemoryScope",
    "MemorySubsystemPaths",
    "SessionMemoryBackend",
    "SessionMemorySnapshot",
    "build_memory_subsystem_paths",
    "resolve_compaction_dir",
    "resolve_long_term_memory_dir",
    "resolve_memory_work_dir",
    "resolve_retrieval_dir",
    "resolve_session_memory_store_path",
]
