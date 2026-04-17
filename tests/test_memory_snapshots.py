from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.memory.long_term import get_long_term_memory, list_long_term_memories, upsert_long_term_memory
from app.memory.snapshots import (
    apply_long_term_memory_snapshot,
    get_pending_long_term_memory_snapshot,
    keep_long_term_memory_snapshot,
    load_long_term_memory_snapshot_sync_state,
    merge_long_term_memory_snapshot,
    replace_long_term_memory_snapshot,
    resolve_long_term_memory_snapshot_dir,
    select_long_term_memory_snapshot,
    LongTermMemorySnapshotError,
)


class MemorySnapshotsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.tempdir.name) / "project-memory"
        self.user_root = Path(self.tempdir.name) / "user-memory"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_keep_long_term_memory_snapshot_records_sync_state_without_mutating_personal_memory(self) -> None:
        self._write_snapshot_memory(
            "default",
            memory_id="project/overview",
            name="Project Overview",
            description="Shared roadmap context.",
            memory_type="project",
            content="Roadmap v1.",
        )
        self._write_snapshot_memory(
            "2026-04-17",
            memory_id="project/release",
            name="Release Notes",
            description="Release-specific memory.",
            memory_type="project",
            content="Release details.",
        )

        selected = select_long_term_memory_snapshot(self.project_root)
        pending_before = get_pending_long_term_memory_snapshot(self.project_root, self.user_root)
        summary = keep_long_term_memory_snapshot(self.user_root, self.project_root)
        sync_state = load_long_term_memory_snapshot_sync_state(self.user_root)
        pending_after = get_pending_long_term_memory_snapshot(self.project_root, self.user_root)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.snapshot_id, "default")
        self.assertIsNotNone(pending_before)
        self.assertEqual(summary.action, "keep")
        self.assertIsNotNone(sync_state)
        self.assertEqual(sync_state.action, "keep")
        self.assertEqual(sync_state.snapshot_id, "default")
        self.assertIsNone(pending_after)

        self._write_snapshot_memory(
            "default",
            memory_id="project/overview",
            name="Project Overview",
            description="Shared roadmap context.",
            memory_type="project",
            content="Roadmap v2.",
        )
        pending_after_update = get_pending_long_term_memory_snapshot(self.project_root, self.user_root)
        self.assertIsNotNone(pending_after_update)
        self.assertNotEqual(pending_after_update.fingerprint, sync_state.fingerprint)

    def test_merge_long_term_memory_snapshot_combines_snapshot_with_existing_memory(self) -> None:
        self._write_snapshot_memory(
            "default",
            memory_id="project/roadmap",
            name="Roadmap Guidance",
            description="Shared roadmap guidance.",
            memory_type="project",
            content="Prefer milestone-first summaries.\nUse concise file-linked updates.",
        )
        upsert_long_term_memory(
            self.user_root,
            {
                "memory_id": "project/roadmap",
                "name": "Roadmap Guidance",
                "description": "Shared roadmap guidance.",
                "memory_type": "project",
                "content_markdown": "Prefer milestone-first summaries.\nCall out owner changes.",
            },
        )

        summary = merge_long_term_memory_snapshot(self.user_root, self.project_root)
        merged_memory = get_long_term_memory(self.user_root, "project/roadmap")
        sync_state = load_long_term_memory_snapshot_sync_state(self.user_root)

        self.assertEqual(summary.action, "merge")
        self.assertEqual(summary.created_memory_ids, [])
        self.assertEqual(summary.updated_memory_ids, ["project/roadmap"])
        self.assertIsNotNone(merged_memory)
        self.assertIn("Prefer milestone-first summaries.", merged_memory.content_markdown)
        self.assertIn("Call out owner changes.", merged_memory.content_markdown)
        self.assertIn("Use concise file-linked updates.", merged_memory.content_markdown)
        self.assertEqual(merged_memory.content_markdown.count("Prefer milestone-first summaries."), 1)
        self.assertIsNotNone(sync_state)
        self.assertEqual(sync_state.action, "merge")

    def test_replace_long_term_memory_snapshot_rewrites_personal_memory_from_snapshot(self) -> None:
        self._write_snapshot_memory(
            "default",
            memory_id="project/overview",
            name="Project Overview",
            description="Shared roadmap context.",
            memory_type="project",
            content="Snapshot overview.",
        )
        upsert_long_term_memory(
            self.user_root,
            {
                "memory_id": "preferences/task-view",
                "name": "Task View Preference",
                "description": "Personal preference.",
                "memory_type": "user",
                "content_markdown": "Prefer grouped updates.",
            },
        )
        upsert_long_term_memory(
            self.user_root,
            {
                "memory_id": "feedback/old",
                "name": "Old Feedback",
                "description": "Old personal feedback.",
                "memory_type": "feedback",
                "content_markdown": "Do not keep this.",
            },
        )

        summary = replace_long_term_memory_snapshot(self.user_root, self.project_root)
        remaining_ids = [entry.memory_id for entry in list_long_term_memories(self.user_root)]
        sync_state = load_long_term_memory_snapshot_sync_state(self.user_root)

        self.assertEqual(summary.action, "replace")
        self.assertEqual(summary.created_memory_ids, ["project/overview"])
        self.assertEqual(summary.deleted_memory_ids, ["feedback/old", "preferences/task-view"])
        self.assertEqual(remaining_ids, ["project/overview"])
        self.assertIsNotNone(sync_state)
        self.assertEqual(sync_state.action, "replace")

    def test_apply_long_term_memory_snapshot_dispatches_to_first_class_actions(self) -> None:
        self._write_snapshot_memory(
            "default",
            memory_id="project/overview",
            name="Project Overview",
            description="Shared roadmap context.",
            memory_type="project",
            content="Snapshot overview.",
        )

        keep_summary = apply_long_term_memory_snapshot(self.user_root, self.project_root, action="keep")
        merge_summary = apply_long_term_memory_snapshot(self.user_root, self.project_root, action="merge")
        replace_summary = apply_long_term_memory_snapshot(self.user_root, self.project_root, action="replace")

        self.assertEqual(keep_summary.action, "keep")
        self.assertEqual(merge_summary.action, "merge")
        self.assertEqual(replace_summary.action, "replace")

    def test_keep_long_term_memory_snapshot_requires_available_snapshot(self) -> None:
        with self.assertRaisesRegex(LongTermMemorySnapshotError, "No project-provided memory snapshot is available"):
            keep_long_term_memory_snapshot(self.user_root, self.project_root)

    def test_merge_long_term_memory_snapshot_requires_available_snapshot(self) -> None:
        with self.assertRaisesRegex(LongTermMemorySnapshotError, "No project-provided memory snapshot is available"):
            merge_long_term_memory_snapshot(self.user_root, self.project_root)

    def test_replace_long_term_memory_snapshot_requires_available_snapshot(self) -> None:
        with self.assertRaisesRegex(LongTermMemorySnapshotError, "No project-provided memory snapshot is available"):
            replace_long_term_memory_snapshot(self.user_root, self.project_root)

    def _write_snapshot_memory(
        self,
        snapshot_id: str,
        *,
        memory_id: str,
        name: str,
        description: str,
        memory_type: str,
        content: str,
    ) -> None:
        snapshot_root = resolve_long_term_memory_snapshot_dir(self.project_root, snapshot_id)
        upsert_long_term_memory(
            snapshot_root,
            {
                "memory_id": memory_id,
                "name": name,
                "description": description,
                "memory_type": memory_type,
                "content_markdown": content,
            },
        )


if __name__ == "__main__":
    unittest.main()
