from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app.memory.file_transaction as file_transaction
from app.frontmatter import render_frontmatter_document
from app.memory.long_term import get_long_term_memory, list_long_term_memories, upsert_long_term_memory
from app.memory.snapshots import (
    apply_long_term_memory_snapshot,
    get_pending_long_term_memory_snapshot,
    keep_long_term_memory_snapshot,
    load_long_term_memory_snapshot_sync_state,
    merge_long_term_memory_snapshot,
    replace_long_term_memory_snapshot,
    resolve_long_term_memory_snapshot_dir,
    resolve_long_term_memory_snapshot_sync_path,
    select_long_term_memory_snapshot,
    LongTermMemorySnapshotError,
    write_long_term_memory_snapshot_sync_state,
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

    def test_merge_long_term_memory_snapshot_ignores_unrelated_broken_topic_entries(self) -> None:
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

        broken_topic_path = self.user_root / "topics" / "reference" / "broken.md"
        broken_topic_path.parent.mkdir(parents=True, exist_ok=True)
        broken_topic_path.write_text("Missing frontmatter.", encoding="utf-8")
        index_path = self.user_root / "MEMORY.md"
        index_path.write_text(
            render_frontmatter_document(
                {
                    "name": "Memory Index",
                    "description": "Top-level catalog of long-term memory topics.",
                    "type": "reference",
                },
                "\n".join(
                    [
                        "## Topics",
                        "",
                        "- [Roadmap Guidance](topics/project/roadmap.md) (`project`): Shared roadmap guidance.",
                        "- [Broken Reference](topics/reference/broken.md) (`reference`): Unrelated broken topic.",
                    ]
                ),
            ),
            encoding="utf-8",
        )

        summary = merge_long_term_memory_snapshot(self.user_root, self.project_root)
        merged_memory = get_long_term_memory(self.user_root, "project/roadmap")

        self.assertEqual(summary.action, "merge")
        self.assertEqual(summary.updated_memory_ids, ["project/roadmap"])
        self.assertIsNotNone(merged_memory)
        self.assertIn("Use concise file-linked updates.", merged_memory.content_markdown)
        self.assertTrue(broken_topic_path.exists())

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

    def test_merge_long_term_memory_snapshot_rolls_back_if_transaction_commit_fails(self) -> None:
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
        snapshot = select_long_term_memory_snapshot(self.project_root)
        self.assertIsNotNone(snapshot)
        write_long_term_memory_snapshot_sync_state(self.user_root, snapshot=snapshot, action="keep")

        index_path = self.user_root / "MEMORY.md"
        topic_path = self.user_root / "topics" / "project" / "roadmap.md"
        sync_path = resolve_long_term_memory_snapshot_sync_path(self.user_root)
        original_index_text = index_path.read_text(encoding="utf-8")
        original_topic_text = topic_path.read_text(encoding="utf-8")
        original_sync_text = sync_path.read_text(encoding="utf-8")
        apply_call_count = 0
        real_apply = file_transaction._apply_root_file_transaction_operation

        def failing_apply(operation):
            nonlocal apply_call_count
            apply_call_count += 1
            if apply_call_count == 3:
                raise OSError("boom")
            return real_apply(operation)

        with patch(
            "app.memory.file_transaction._apply_root_file_transaction_operation",
            side_effect=failing_apply,
        ):
            with self.assertRaisesRegex(OSError, "boom"):
                merge_long_term_memory_snapshot(self.user_root, self.project_root)

        restored_memory = get_long_term_memory(self.user_root, "project/roadmap")
        self.assertIsNotNone(restored_memory)
        self.assertEqual(restored_memory.content_markdown, "Prefer milestone-first summaries.\nCall out owner changes.")
        self.assertEqual(index_path.read_text(encoding="utf-8"), original_index_text)
        self.assertEqual(topic_path.read_text(encoding="utf-8"), original_topic_text)
        self.assertEqual(sync_path.read_text(encoding="utf-8"), original_sync_text)

    def test_replace_long_term_memory_snapshot_rolls_back_if_transaction_commit_fails(self) -> None:
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
        snapshot = select_long_term_memory_snapshot(self.project_root)
        self.assertIsNotNone(snapshot)
        write_long_term_memory_snapshot_sync_state(self.user_root, snapshot=snapshot, action="keep")

        index_path = self.user_root / "MEMORY.md"
        task_view_path = self.user_root / "topics" / "preferences" / "task-view.md"
        feedback_path = self.user_root / "topics" / "feedback" / "old.md"
        sync_path = resolve_long_term_memory_snapshot_sync_path(self.user_root)
        original_index_text = index_path.read_text(encoding="utf-8")
        original_task_view_text = task_view_path.read_text(encoding="utf-8")
        original_feedback_text = feedback_path.read_text(encoding="utf-8")
        original_sync_text = sync_path.read_text(encoding="utf-8")
        apply_call_count = 0
        real_apply = file_transaction._apply_root_file_transaction_operation

        def failing_apply(operation):
            nonlocal apply_call_count
            apply_call_count += 1
            if apply_call_count == 5:
                raise OSError("boom")
            return real_apply(operation)

        with patch(
            "app.memory.file_transaction._apply_root_file_transaction_operation",
            side_effect=failing_apply,
        ):
            with self.assertRaisesRegex(OSError, "boom"):
                replace_long_term_memory_snapshot(self.user_root, self.project_root)

        self.assertEqual(index_path.read_text(encoding="utf-8"), original_index_text)
        self.assertEqual(task_view_path.read_text(encoding="utf-8"), original_task_view_text)
        self.assertEqual(feedback_path.read_text(encoding="utf-8"), original_feedback_text)
        self.assertEqual(sync_path.read_text(encoding="utf-8"), original_sync_text)
        self.assertEqual(
            [entry.memory_id for entry in list_long_term_memories(self.user_root)],
            ["feedback/old", "preferences/task-view"],
        )

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
