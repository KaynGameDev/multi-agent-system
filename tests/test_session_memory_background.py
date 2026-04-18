from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.session_memory import SessionMemoryRecord, SessionMemoryStore
from app.session_memory_background import (
    BackgroundSessionMemoryUpdater,
    SessionMemoryRefreshTarget,
)


class SessionMemoryBackgroundUpdaterTests(unittest.TestCase):
    def test_updater_flush_coalesces_to_latest_target_for_thread(self) -> None:
        calls: list[SessionMemoryRefreshTarget] = []
        updater = BackgroundSessionMemoryUpdater(calls.append, debounce_seconds=60.0)
        self.addCleanup(updater.close)

        updater.schedule(
            SessionMemoryRefreshTarget(
                conversation_id="c1",
                thread_id="web:test",
                allowed_session_file_path="/tmp/sessions/web/test.md",
            )
        )
        updater.schedule(
            SessionMemoryRefreshTarget(
                conversation_id="c2",
                thread_id="web:test",
                allowed_session_file_path="/tmp/sessions/web/test.md",
            )
        )
        updater.flush("web:test")

        self.assertEqual([target.conversation_id for target in calls], ["c2"])

    def test_updater_preserves_force_refresh_when_coalescing_targets(self) -> None:
        calls: list[SessionMemoryRefreshTarget] = []
        updater = BackgroundSessionMemoryUpdater(calls.append, debounce_seconds=60.0)
        self.addCleanup(updater.close)

        updater.schedule(
            SessionMemoryRefreshTarget(
                conversation_id="c1",
                thread_id="web:test",
                allowed_session_file_path="/tmp/sessions/web/test.md",
                force_refresh=True,
            )
        )
        updater.schedule(
            SessionMemoryRefreshTarget(
                conversation_id="c2",
                thread_id="web:test",
                allowed_session_file_path="/tmp/sessions/web/test.md",
            )
        )
        updater.flush("web:test")

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].conversation_id, "c2")
        self.assertTrue(calls[0].force_refresh)

    def test_scoped_upsert_rejects_mismatched_thread_or_path(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        storage_path = Path(temp_dir.name) / "session_memory.json"
        store = SessionMemoryStore(storage_path)
        record = SessionMemoryRecord(
            thread_id="web:test",
            updated_at="2026-04-17T00:00:03+00:00",
            last_message_id="a1",
            last_message_created_at="2026-04-17T00:00:03+00:00",
            covered_message_count=2,
            covered_tokens=128,
            summary_markdown="## Continuation Summary\nScoped update.",
            source="update",
        )

        with self.assertRaisesRegex(ValueError, "allowed thread"):
            store.upsert_scoped(
                record,
                allowed_thread_id="web:other",
                allowed_session_file_path=store.resolve_session_file_path("web:test"),
            )

        with self.assertRaisesRegex(ValueError, "allowed session file path"):
            store.upsert_scoped(
                record,
                allowed_thread_id="web:test",
                allowed_session_file_path=store.resolve_session_file_path("web:other"),
            )


if __name__ == "__main__":
    unittest.main()
