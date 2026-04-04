from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.config import Settings
from app.paths import resolve_project_path

DEFAULT_CHECKPOINT_DB_NAME = "langgraph_checkpoints.sqlite3"


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
    connection = sqlite3.connect(str(database_path), check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return GraphCheckpointStore(
        saver,
        database_path=database_path,
        connection=connection,
    )


def _build_thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}
