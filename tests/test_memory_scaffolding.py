from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app import config as config_module
from app.config import load_settings
from app.memory.paths import (
    build_memory_subsystem_paths,
    resolve_long_term_memory_index_path,
    resolve_long_term_memory_topics_dir,
    resolve_memory_work_dir,
    resolve_session_memory_dir,
)
from app.memory.types import (
    ConversationCompactionRequest,
    LongTermMemoryIndexEntry,
    LongTermMemoryRecord,
    LongTermMemoryWrite,
    MemoryReference,
    MemoryRetrievalQuery,
    MemoryRetrievalResult,
    SessionMemoryFile,
    SessionMemoryFileUpdate,
    SessionMemorySnapshot,
)
from app.paths import PROJECT_ROOT
from tests.common import make_settings


class MemoryScaffoldingTests(unittest.TestCase):
    def tearDown(self) -> None:
        config_module._cached_settings = None

    def test_load_settings_parses_memory_scaffolding_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WEB_ENABLED": "true",
                "SLACK_ENABLED": "false",
                "GOOGLE_API_KEY": "test-google-key",
                "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/credentials.json",
                "JADE_PROJECT_SHEET_ID": "sheet-id",
                "MEMORY_WORK_DIR": "runtime/custom-memory",
                "LONG_TERM_MEMORY_ENABLED": "true",
                "MEMORY_RETRIEVAL_ENABLED": "true",
                "MEMORY_RETRIEVAL_DEFAULT_LIMIT": "12",
            },
            clear=True,
        ):
            settings = load_settings(force_reload=True)

        self.assertEqual(settings.memory_work_dir, "runtime/custom-memory")
        self.assertTrue(settings.long_term_memory_enabled)
        self.assertTrue(settings.memory_retrieval_enabled)
        self.assertEqual(settings.memory_retrieval_default_limit, 12)

    def test_memory_paths_resolve_from_memory_work_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            settings = replace(
                make_settings(Path(tempdir)),
                memory_work_dir="runtime/test-memory",
            )

        paths = build_memory_subsystem_paths(settings)
        expected_root = (PROJECT_ROOT / "runtime" / "test-memory").resolve()

        self.assertEqual(resolve_memory_work_dir(settings), expected_root)
        self.assertEqual(paths.work_dir, expected_root)
        self.assertEqual(paths.session_memory_store_path, expected_root / "session_memory.json")
        self.assertEqual(paths.session_memory_dir, expected_root / "sessions")
        self.assertEqual(paths.long_term_memory_dir, expected_root / "long_term")
        self.assertEqual(paths.long_term_memory_index_path, expected_root / "long_term" / "MEMORY.md")
        self.assertEqual(paths.long_term_memory_topics_dir, expected_root / "long_term" / "topics")
        self.assertEqual(paths.retrieval_dir, expected_root / "retrieval")
        self.assertEqual(paths.compaction_dir, expected_root / "compaction")
        self.assertEqual(resolve_session_memory_dir(settings), expected_root / "sessions")
        self.assertEqual(resolve_long_term_memory_index_path(settings), expected_root / "long_term" / "MEMORY.md")
        self.assertEqual(resolve_long_term_memory_topics_dir(settings), expected_root / "long_term" / "topics")

    def test_memory_contract_models_cover_session_long_term_retrieval_and_compaction(self) -> None:
        session_snapshot = SessionMemorySnapshot(
            thread_id="web:test-thread",
            summary_markdown="## Continuation Summary\nKeep the current plan.",
            last_message_id="a1",
            covered_message_count=4,
            covered_tokens=512,
        )
        self.assertEqual(session_snapshot.thread_id, "web:test-thread")

        session_file = SessionMemoryFile(
            thread_id="web:test-thread",
            source_path="/tmp/runtime/memory/sessions/web/test-thread.md",
            current_state="Implementing session memory files.",
            task_spec="Create one file per conversation.",
            key_files=["app/memory/session_files.py"],
            workflow="Inspect, implement, test.",
            errors_corrections="Adjusted the path layout to use sessions/.",
            learnings="Separate the file layer from the JSON store.",
            worklog="- Added the scaffold.",
        )
        self.assertEqual(session_file.key_files, ["app/memory/session_files.py"])

        session_update = SessionMemoryFileUpdate(
            current_state="Updated current state.",
            key_files=["app/session_memory.py", "app/memory/session_files.py"],
        )
        self.assertEqual(
            session_update.key_files,
            ["app/session_memory.py", "app/memory/session_files.py"],
        )

        record = LongTermMemoryRecord(
            memory_id="mem-001",
            scope="workspace",
            namespace="jade",
            content_markdown="Remember the release checklist decision.",
            references=[MemoryReference(kind="document_path", value="knowledge/Docs/00_Shared/Standards/MemorySubsystem.md")],
        )
        self.assertEqual(record.scope, "workspace")
        self.assertEqual(record.references[0].kind, "document_path")

        index_entry = LongTermMemoryIndexEntry(
            memory_id="project_overview",
            relative_path="topics/project_overview.md",
            name="Project Overview",
            description="Shared roadmap and release context.",
            memory_type="project",
        )
        self.assertEqual(index_entry.relative_path, "topics/project_overview.md")

        write_request = LongTermMemoryWrite(
            memory_id="project_overview",
            name="Project Overview",
            description="Shared roadmap and release context.",
            memory_type="project",
            content_markdown="The roadmap is currently focused on the memory subsystem.",
        )
        self.assertEqual(write_request.memory_type, "project")

        query = MemoryRetrievalQuery(
            query="release checklist",
            namespace="jade",
            scope="workspace",
            thread_id="web:test-thread",
            top_k=5,
        )
        self.assertEqual(query.top_k, 5)

        result = MemoryRetrievalResult(
            memory_id="mem-001",
            content_markdown="Remember the release checklist decision.",
            score=0.92,
            scope="workspace",
        )
        self.assertEqual(result.score, 0.92)

        compaction_request = ConversationCompactionRequest(
            thread_id="web:test-thread",
            trigger="auto",
            preserved_tail_count=1,
        )
        self.assertEqual(compaction_request.trigger, "auto")


if __name__ == "__main__":
    unittest.main()
