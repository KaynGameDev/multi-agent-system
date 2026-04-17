from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.memory.consolidation import (
    FileLongTermMemoryConsolidator,
    consolidate_long_term_memory,
    should_schedule_long_term_memory_consolidation,
)
from app.memory.consolidation_background import BackgroundMemoryConsolidator, MemoryConsolidationTarget
from app.memory.long_term import FileLongTermMemoryStore


class MemoryConsolidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root_dir = Path(self.tempdir.name) / "long_term"
        self.store = FileLongTermMemoryStore(self.root_dir)

    def test_consolidate_long_term_memory_rolls_noisy_entries_into_topic_memory_and_prunes_duplicates(self) -> None:
        self.store.upsert(
            {
                "memory_id": "session/2026-04-17/preferred-name",
                "name": "Preferred Name",
                "description": "User's preferred name for future replies.",
                "memory_type": "user",
                "content_markdown": "Call the user `Kay` in future replies.",
            }
        )
        self.store.upsert(
            {
                "memory_id": "daily/2026-04-18/preferred-name",
                "name": "Preferred Name",
                "description": "User's preferred name for future replies.",
                "memory_type": "user",
                "content_markdown": "Call the user `Kay` in future replies.",
            }
        )
        self.store.upsert(
            {
                "memory_id": "feedback/duplicate-one",
                "name": "User Feedback",
                "description": "Behavior correction to preserve across future turns.",
                "memory_type": "feedback",
                "content_markdown": "Prefer concise updates.",
            }
        )
        self.store.upsert(
            {
                "memory_id": "feedback/duplicate-two",
                "name": "User Feedback",
                "description": "Behavior correction to preserve across future turns.",
                "memory_type": "feedback",
                "content_markdown": "Prefer concise updates.",
            }
        )

        summary = consolidate_long_term_memory(self.root_dir, min_entries=1)

        self.assertEqual(summary.noisy_group_count, 1)
        self.assertEqual(summary.duplicate_group_count, 1)
        self.assertIn("user/preferred-name", summary.updated_memory_ids)
        self.assertIn("session/2026-04-17/preferred-name", summary.deleted_memory_ids)
        self.assertIn("daily/2026-04-18/preferred-name", summary.deleted_memory_ids)

        entries = self.store.list()
        self.assertEqual(
            [entry.memory_id for entry in entries],
            ["feedback/duplicate-one", "user/preferred-name"],
        )
        self.assertEqual(
            self.store.get("user/preferred-name").content_markdown,
            "Call the user `Kay` in future replies.",
        )

    def test_should_schedule_long_term_memory_consolidation_detects_noisy_or_duplicate_roots(self) -> None:
        self.store.upsert(
            {
                "memory_id": "user/preferred-name",
                "name": "Preferred Name",
                "description": "User's preferred name for future replies.",
                "memory_type": "user",
                "content_markdown": "Call the user `Kay` in future replies.",
            }
        )
        self.assertFalse(should_schedule_long_term_memory_consolidation(self.root_dir, min_entries=4))

        self.store.upsert(
            {
                "memory_id": "session/2026-04-17/preferred-name",
                "name": "Preferred Name",
                "description": "User's preferred name for future replies.",
                "memory_type": "user",
                "content_markdown": "Call the user `Kay` in future replies.",
            }
        )
        self.assertTrue(should_schedule_long_term_memory_consolidation(self.root_dir, min_entries=4))

    def test_background_memory_consolidator_flushes_scheduled_root(self) -> None:
        self.store.upsert(
            {
                "memory_id": "turn/2026-04-17/preferred-name",
                "name": "Preferred Name",
                "description": "User's preferred name for future replies.",
                "memory_type": "user",
                "content_markdown": "Call the user `Kay` in future replies.",
            }
        )

        consolidator = BackgroundMemoryConsolidator(
            lambda target: FileLongTermMemoryConsolidator(target.root_dir).consolidate(min_entries=1),
            debounce_seconds=0,
        )
        self.addCleanup(consolidator.close)

        consolidator.schedule(
            MemoryConsolidationTarget(
                root_dir=str(self.root_dir),
                agent_name="project_task_agent",
                memory_scope="user",
                scope_key="tester",
            )
        )
        consolidator.flush(str(self.root_dir))

        self.assertTrue(consolidator.wait_for_idle(str(self.root_dir), timeout=1.0))
        self.assertEqual([entry.memory_id for entry in self.store.list()], ["user/preferred-name"])


if __name__ == "__main__":
    unittest.main()
