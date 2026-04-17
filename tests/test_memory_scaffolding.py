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
    resolve_memory_work_dir,
)
from app.memory.types import (
    ConversationCompactionRequest,
    LongTermMemoryRecord,
    MemoryReference,
    MemoryRetrievalQuery,
    MemoryRetrievalResult,
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
        self.assertEqual(paths.long_term_memory_dir, expected_root / "long_term")
        self.assertEqual(paths.retrieval_dir, expected_root / "retrieval")
        self.assertEqual(paths.compaction_dir, expected_root / "compaction")

    def test_memory_contract_models_cover_session_long_term_retrieval_and_compaction(self) -> None:
        session_snapshot = SessionMemorySnapshot(
            thread_id="web:test-thread",
            summary_markdown="## Continuation Summary\nKeep the current plan.",
            last_message_id="a1",
            covered_message_count=4,
            covered_tokens=512,
        )
        self.assertEqual(session_snapshot.thread_id, "web:test-thread")

        record = LongTermMemoryRecord(
            memory_id="mem-001",
            scope="workspace",
            namespace="jade",
            content_markdown="Remember the release checklist decision.",
            references=[MemoryReference(kind="document_path", value="knowledge/Docs/00_Shared/Standards/MemorySubsystem.md")],
        )
        self.assertEqual(record.scope, "workspace")
        self.assertEqual(record.references[0].kind, "document_path")

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
