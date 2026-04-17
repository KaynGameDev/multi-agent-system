from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.memory.paths import resolve_session_memory_dir
from app.memory.session_files import (
    FileSessionMemoryStore,
    SessionMemoryFormatError,
    build_session_memory_relative_path,
    ensure_session_memory_file,
    get_session_memory_file,
    load_session_memory_file,
    resolve_session_memory_file_path,
    update_session_memory_file,
)
from tests.common import make_settings


class SessionMemoryFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.runtime_root = Path(self.tempdir.name) / "runtime"
        self.settings = make_settings(self.runtime_root)
        self.session_root = resolve_session_memory_dir(self.settings)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_build_session_memory_relative_path_scopes_by_thread_id(self) -> None:
        relative_path = build_session_memory_relative_path("web:test-thread")

        self.assertEqual(relative_path.as_posix(), "web/test-thread.md")

    def test_ensure_session_memory_file_creates_fixed_template(self) -> None:
        created = ensure_session_memory_file(self.session_root, "web:test-thread")
        file_path = resolve_session_memory_file_path(self.session_root, "web:test-thread")
        raw_text = file_path.read_text(encoding="utf-8")

        self.assertEqual(created.thread_id, "web:test-thread")
        self.assertTrue(file_path.exists())
        self.assertIn("# Session Memory", raw_text)
        self.assertIn("## Current State", raw_text)
        self.assertIn("## Task Spec", raw_text)
        self.assertIn("## Key Files", raw_text)
        self.assertIn("## Workflow", raw_text)
        self.assertIn("## Errors/Corrections", raw_text)
        self.assertIn("## Learnings", raw_text)
        self.assertIn("## Worklog", raw_text)

    def test_update_session_memory_file_preserves_unset_sections(self) -> None:
        ensure_session_memory_file(self.session_root, "web:test-thread")

        updated = update_session_memory_file(
            self.session_root,
            "web:test-thread",
            {
                "current_state": "Investigating the session memory template.",
                "task_spec": "Add a per-session file-backed summary.",
                "key_files": ["app/memory/session_files.py", "app/session_memory.py"],
                "workflow": "Inspect, implement, test.",
            },
        )
        reloaded = get_session_memory_file(self.session_root, "web:test-thread")

        self.assertIsNotNone(reloaded)
        self.assertEqual(updated.current_state, "Investigating the session memory template.")
        self.assertEqual(updated.task_spec, "Add a per-session file-backed summary.")
        self.assertEqual(updated.key_files, ["app/memory/session_files.py", "app/session_memory.py"])
        self.assertEqual(updated.workflow, "Inspect, implement, test.")
        self.assertEqual(reloaded.errors_corrections, "")
        self.assertEqual(reloaded.learnings, "")
        self.assertEqual(reloaded.worklog, "")

    def test_load_session_memory_file_rejects_missing_required_sections(self) -> None:
        path = resolve_session_memory_file_path(self.session_root, "web:test-thread")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "---",
                    "thread_id: web:test-thread",
                    "kind: session_memory",
                    "template_version: 1",
                    "---",
                    "",
                    "# Session Memory",
                    "",
                    "## Current State",
                    "",
                    "Only one section is present.",
                ]
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(SessionMemoryFormatError, "missing required sections"):
            load_session_memory_file(path)

    def test_file_store_supports_ensure_read_update_and_delete(self) -> None:
        store = FileSessionMemoryStore(self.session_root)

        created = store.ensure("web:test-thread")
        updated = store.update(
            "web:test-thread",
            {
                "current_state": "Session file created.",
                "worklog": "- Created the new session memory file.",
            },
        )
        loaded = store.get("web:test-thread")
        deleted = store.delete("web:test-thread")

        self.assertEqual(created.thread_id, "web:test-thread")
        self.assertEqual(updated.worklog, "- Created the new session memory file.")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.current_state, "Session file created.")
        self.assertTrue(deleted)
        self.assertIsNone(get_session_memory_file(self.session_root, "web:test-thread"))


if __name__ == "__main__":
    unittest.main()
