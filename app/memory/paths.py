from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import DEFAULT_MEMORY_WORK_DIR, Settings
from app.paths import resolve_project_path

DEFAULT_SESSION_MEMORY_STORE_FILENAME = "session_memory.json"
DEFAULT_SESSION_MEMORY_DIRNAME = "sessions"
DEFAULT_LONG_TERM_MEMORY_DIRNAME = "long_term"
DEFAULT_LONG_TERM_MEMORY_INDEX_FILENAME = "MEMORY.md"
DEFAULT_LONG_TERM_MEMORY_TOPICS_DIRNAME = "topics"
DEFAULT_RETRIEVAL_DIRNAME = "retrieval"
DEFAULT_COMPACTION_DIRNAME = "compaction"


@dataclass(frozen=True)
class MemorySubsystemPaths:
    work_dir: Path
    session_memory_store_path: Path
    session_memory_dir: Path
    long_term_memory_dir: Path
    long_term_memory_index_path: Path
    long_term_memory_topics_dir: Path
    retrieval_dir: Path
    compaction_dir: Path


def resolve_memory_work_dir(settings: Settings) -> Path:
    return resolve_project_path(settings.memory_work_dir, DEFAULT_MEMORY_WORK_DIR)


def resolve_session_memory_store_path(settings: Settings) -> Path:
    return resolve_memory_work_dir(settings) / DEFAULT_SESSION_MEMORY_STORE_FILENAME


def resolve_session_memory_dir(settings: Settings) -> Path:
    return resolve_memory_work_dir(settings) / DEFAULT_SESSION_MEMORY_DIRNAME


def resolve_long_term_memory_dir(settings: Settings) -> Path:
    return resolve_memory_work_dir(settings) / DEFAULT_LONG_TERM_MEMORY_DIRNAME


def resolve_long_term_memory_index_path(settings: Settings) -> Path:
    return resolve_long_term_memory_dir(settings) / DEFAULT_LONG_TERM_MEMORY_INDEX_FILENAME


def resolve_long_term_memory_topics_dir(settings: Settings) -> Path:
    return resolve_long_term_memory_dir(settings) / DEFAULT_LONG_TERM_MEMORY_TOPICS_DIRNAME


def resolve_retrieval_dir(settings: Settings) -> Path:
    return resolve_memory_work_dir(settings) / DEFAULT_RETRIEVAL_DIRNAME


def resolve_compaction_dir(settings: Settings) -> Path:
    return resolve_memory_work_dir(settings) / DEFAULT_COMPACTION_DIRNAME


def build_memory_subsystem_paths(settings: Settings) -> MemorySubsystemPaths:
    return MemorySubsystemPaths(
        work_dir=resolve_memory_work_dir(settings),
        session_memory_store_path=resolve_session_memory_store_path(settings),
        session_memory_dir=resolve_session_memory_dir(settings),
        long_term_memory_dir=resolve_long_term_memory_dir(settings),
        long_term_memory_index_path=resolve_long_term_memory_index_path(settings),
        long_term_memory_topics_dir=resolve_long_term_memory_topics_dir(settings),
        retrieval_dir=resolve_retrieval_dir(settings),
        compaction_dir=resolve_compaction_dir(settings),
    )
