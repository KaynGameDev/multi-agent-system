from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import Settings
from app.paths import resolve_project_path

DEFAULT_CHECKPOINT_DB_NAME = "langgraph_checkpoints.sqlite3"
CORRUPT_DB_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S%fZ"

logger = logging.getLogger(__name__)


class GraphCheckpointStore:
    def __init__(self, saver: Any, *, database_path: Path, connection: sqlite3.Connection | None = None) -> None:
        self.saver = saver
        self.database_path = database_path
        self._connection = connection

    def has_checkpoint(self, thread_id: str) -> bool:
        return self.get_checkpoint(thread_id) is not None

    def get_checkpoint(self, thread_id: str):
        return self.saver.get_tuple(_build_thread_config(thread_id))

    def delete_thread(self, thread_id: str) -> None:
        self.saver.delete_thread(thread_id)

    def close(self) -> None:
        saver_close = getattr(self.saver, "close", None)
        if callable(saver_close):
            saver_close()
            self._connection = None
            return
        if self._connection is not None:
            self._connection.close()
            self._connection = None


def resolve_checkpoint_db_path(settings: Settings) -> Path:
    configured_path = settings.langgraph_checkpoint_db_path.strip()
    default_path = str(Path(settings.conversion_work_dir) / DEFAULT_CHECKPOINT_DB_NAME)
    return resolve_project_path(configured_path or default_path, default_path)


def build_checkpoint_store(settings: Settings) -> GraphCheckpointStore:
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing optional dependency `langgraph-checkpoint-sqlite`. "
            "Install project dependencies before starting Jade."
        ) from exc

    database_path = resolve_checkpoint_db_path(settings)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    saver = _build_resilient_sqlite_saver(SqliteSaver, database_path)
    saver.setup()
    return GraphCheckpointStore(
        saver,
        database_path=database_path,
    )


def _build_thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def _build_resilient_sqlite_saver(sqlite_saver_cls, database_path: Path):
    class ResilientSqliteSaver(sqlite_saver_cls):
        def __init__(self, path: Path) -> None:
            self.database_path = path
            self._recovery_lock = threading.RLock()
            self._operation_lock = threading.RLock()
            super().__init__(_open_sqlite_connection(path))

        def close(self) -> None:
            try:
                self.conn.close()
            except Exception:
                logger.debug("Failed to close checkpoint SQLite connection cleanly.", exc_info=True)

        def setup(self) -> None:
            return self._run_with_recovery(super().setup)

        def get_tuple(self, config):
            return self._run_with_recovery(super().get_tuple, config)

        def get(self, config):
            return self._run_with_recovery(super().get, config)

        def list(self, *args, **kwargs):
            return self._run_with_recovery(super().list, *args, **kwargs)

        def put(self, config, checkpoint, metadata, new_versions):
            return self._run_with_recovery(super().put, config, checkpoint, metadata, new_versions)

        def put_writes(self, config, writes, task_id, task_path=""):
            return self._run_with_recovery(super().put_writes, config, writes, task_id, task_path)

        def delete_thread(self, thread_id: str) -> None:
            return self._run_with_recovery(super().delete_thread, thread_id)

        def _run_with_recovery(self, operation, *args, **kwargs):
            with self._operation_lock:
                try:
                    return operation(*args, **kwargs)
                except sqlite3.ProgrammingError as exc:
                    if "closed database" not in str(exc).lower():
                        raise
                    with self._recovery_lock:
                        logger.warning(
                            "Checkpoint SQLite connection was closed unexpectedly; reconnecting. path=%s error=%s",
                            self.database_path,
                            exc,
                        )
                        self._reconnect()
                    return operation(*args, **kwargs)
                except sqlite3.DatabaseError as exc:
                    if not is_sqlite_corruption_error(exc):
                        raise
                    with self._recovery_lock:
                        logger.warning(
                            "Checkpoint SQLite database is corrupted; rebuilding fresh database. path=%s error=%s",
                            self.database_path,
                            exc,
                        )
                        self._recover_from_corruption()
                    return operation(*args, **kwargs)

        def _reconnect(self) -> None:
            try:
                self.conn.close()
            except Exception:
                logger.debug("Failed to close stale checkpoint SQLite connection.", exc_info=True)
            self.conn = _open_sqlite_connection(self.database_path)
            self.is_setup = False
            super().setup()

        def _recover_from_corruption(self) -> None:
            try:
                self.conn.close()
            except Exception:
                logger.debug("Failed to close corrupted checkpoint SQLite connection cleanly.", exc_info=True)
            quarantine_sqlite_database(self.database_path)
            self.conn = _open_sqlite_connection(self.database_path)
            self.is_setup = False
            super().setup()

    return ResilientSqliteSaver(database_path)


def _open_sqlite_connection(database_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(str(database_path), check_same_thread=False)


def is_sqlite_corruption_error(exc: BaseException) -> bool:
    message = str(exc).strip().lower()
    return any(
        marker in message
        for marker in (
            "database disk image is malformed",
            "malformed",
            "file is not a database",
            "not a database",
        )
    )


def quarantine_sqlite_database(database_path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime(CORRUPT_DB_TIMESTAMP_FORMAT)
    quarantine_path = database_path.with_name(f"{database_path.name}.corrupt.{timestamp}")
    _rename_if_exists(database_path, quarantine_path)
    _rename_if_exists(
        Path(f"{database_path}-wal"),
        Path(f"{quarantine_path}-wal"),
    )
    _rename_if_exists(
        Path(f"{database_path}-shm"),
        Path(f"{quarantine_path}-shm"),
    )
    return quarantine_path


def _rename_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
