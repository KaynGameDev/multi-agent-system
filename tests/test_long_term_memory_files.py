from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.frontmatter import render_frontmatter_document
from app.memory.long_term import (
    FileLongTermMemoryStore,
    LongTermMemoryFormatError,
    delete_long_term_memory,
    format_long_term_memory_index,
    get_long_term_memory,
    list_long_term_memories,
    load_long_term_memory_catalog,
    load_long_term_memory_file,
    normalize_long_term_memory_id,
    parse_long_term_memory_index,
    resolve_long_term_memory_index_file,
    resolve_long_term_memory_topic_path,
    upsert_long_term_memory,
)
from app.memory.types import LongTermMemoryIndexEntry


def write_memory_markdown(
    path: Path,
    *,
    name: str,
    description: str,
    memory_type: str,
    body: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_frontmatter_document(
            {
                "name": name,
                "description": description,
                "type": memory_type,
            },
            body,
        ),
        encoding="utf-8",
    )


class LongTermMemoryFilesTests(unittest.TestCase):
    def test_load_long_term_memory_catalog_reads_index_and_topic_files(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "long_term"
            project_path = root / "topics" / "project_overview.md"
            user_path = root / "topics" / "users" / "tester_preferences.md"

            write_memory_markdown(
                project_path,
                name="Project Overview",
                description="Shared roadmap and release context.",
                memory_type="project",
                body="The roadmap is currently focused on the memory subsystem.",
            )
            write_memory_markdown(
                user_path,
                name="Tester Preferences",
                description="Stable user preferences for the current workspace.",
                memory_type="user",
                body="Prefer concise updates with file references.",
            )
            write_memory_markdown(
                root / "MEMORY.md",
                name="Memory Index",
                description="Top-level long-term memory catalog.",
                memory_type="reference",
                body=format_long_term_memory_index(
                    [
                        LongTermMemoryIndexEntry(
                            memory_id="project_overview",
                            relative_path="topics/project_overview.md",
                            name="Project Overview",
                            description="Shared roadmap and release context.",
                            memory_type="project",
                        ),
                        LongTermMemoryIndexEntry(
                            memory_id="users/tester_preferences",
                            relative_path="topics/users/tester_preferences.md",
                            name="Tester Preferences",
                            description="Stable user preferences for the current workspace.",
                            memory_type="user",
                        ),
                    ]
                ),
            )

            catalog = load_long_term_memory_catalog(root)

        self.assertEqual(catalog.index_file.memory_id, "MEMORY")
        self.assertEqual(
            [entry.memory_id for entry in catalog.index_entries],
            ["project_overview", "users/tester_preferences"],
        )
        self.assertEqual(
            [topic.memory_id for topic in catalog.topic_files],
            ["project_overview", "users/tester_preferences"],
        )
        self.assertEqual(catalog.topic_files[1].relative_path, "topics/users/tester_preferences.md")

    def test_parse_long_term_memory_index_reads_short_link_entries(self) -> None:
        entries = parse_long_term_memory_index(
            "\n".join(
                [
                    "## Topics",
                    "",
                    "- [Project Overview](topics/project_overview.md) (`project`): Shared roadmap and release context.",
                    "- [Tester Preferences](topics/users/tester_preferences.md) (`user`): Stable user preferences for the current workspace.",
                ]
            )
        )

        self.assertEqual([entry.memory_id for entry in entries], ["project_overview", "users/tester_preferences"])
        self.assertEqual(entries[0].relative_path, "topics/project_overview.md")
        self.assertEqual(entries[1].memory_type, "user")

    def test_load_long_term_memory_catalog_requires_memory_index_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "long_term"
            root.mkdir(parents=True, exist_ok=True)
            write_memory_markdown(
                root / "topics" / "project_overview.md",
                name="Project Overview",
                description="Shared roadmap and release context.",
                memory_type="project",
            )

            with self.assertRaisesRegex(LongTermMemoryFormatError, "MEMORY"):
                load_long_term_memory_catalog(root)

    def test_load_long_term_memory_file_rejects_missing_required_frontmatter_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "long_term"
            path = root / "MEMORY.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "\n".join(
                    [
                        "---",
                        "name: Memory Index",
                        "type: reference",
                        "---",
                        "",
                        "Missing a description field.",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(LongTermMemoryFormatError, "frontmatter"):
                load_long_term_memory_file(path, root_dir=root)

    def test_load_long_term_memory_catalog_rejects_index_entries_outside_topics_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "long_term"
            write_memory_markdown(
                root / "project_overview.md",
                name="Project Overview",
                description="Shared roadmap and release context.",
                memory_type="project",
            )
            write_memory_markdown(
                root / "MEMORY.md",
                name="Memory Index",
                description="Top-level long-term memory catalog.",
                memory_type="reference",
                body="\n".join(
                    [
                        "## Topics",
                        "",
                        "- [Project Overview](project_overview.md) (`project`): Shared roadmap and release context.",
                    ]
                ),
            )

            with self.assertRaisesRegex(LongTermMemoryFormatError, "topics/"):
                load_long_term_memory_catalog(root)

    def test_load_long_term_memory_catalog_rejects_index_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "long_term"
            write_memory_markdown(
                root / "topics" / "feedback.md",
                name="Feedback Summary",
                description="Recent product feedback.",
                memory_type="feedback",
                body="Real feedback content.",
            )
            write_memory_markdown(
                root / "MEMORY.md",
                name="Memory Index",
                description="Top-level long-term memory catalog.",
                memory_type="reference",
                body="\n".join(
                    [
                        "## Topics",
                        "",
                        "- [Feedback Summary](topics/feedback.md) (`project`): Recent product feedback.",
                    ]
                ),
            )

            with self.assertRaisesRegex(LongTermMemoryFormatError, "type mismatch"):
                load_long_term_memory_catalog(root)

    def test_upsert_long_term_memory_creates_topic_file_and_short_index_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "long_term"

            saved = upsert_long_term_memory(
                root,
                {
                    "memory_id": "project_overview",
                    "name": "Project Overview",
                    "description": "Shared roadmap and release context.",
                    "memory_type": "project",
                    "content_markdown": "The roadmap is currently focused on the memory subsystem.",
                },
            )

            index_path = resolve_long_term_memory_index_file(root)
            index_text = index_path.read_text(encoding="utf-8")

        self.assertEqual(saved.relative_path, "topics/project_overview.md")
        self.assertIn("[Project Overview](topics/project_overview.md)", index_text)
        self.assertNotIn("The roadmap is currently focused on the memory subsystem.", index_text)

    def test_upsert_long_term_memory_updates_existing_memory_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "long_term"
            upsert_long_term_memory(
                root,
                {
                    "memory_id": "feedback_summary",
                    "name": "Feedback Summary",
                    "description": "Recent product feedback.",
                    "memory_type": "feedback",
                    "content_markdown": "Version one.",
                },
            )

            updated = upsert_long_term_memory(
                root,
                {
                    "memory_id": "feedback_summary",
                    "name": "Feedback Summary",
                    "description": "Updated durable feedback summary.",
                    "memory_type": "feedback",
                    "content_markdown": "Version two.",
                },
            )

            reloaded = get_long_term_memory(root, "feedback_summary")
            index_entries = list_long_term_memories(root)

        self.assertIsNotNone(reloaded)
        self.assertEqual(updated.description, "Updated durable feedback summary.")
        self.assertEqual(reloaded.content_markdown, "Version two.")
        self.assertEqual(index_entries[0].description, "Updated durable feedback summary.")

    def test_delete_long_term_memory_removes_topic_file_and_index_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "long_term"
            upsert_long_term_memory(
                root,
                {
                    "memory_id": "users/tester_preferences",
                    "name": "Tester Preferences",
                    "description": "Stable user preferences for the current workspace.",
                    "memory_type": "user",
                    "content_markdown": "Prefer concise updates with file references.",
                },
            )
            topic_path = resolve_long_term_memory_topic_path(root, "users/tester_preferences")

            deleted = delete_long_term_memory(root, "users/tester_preferences")
            index_entries = list_long_term_memories(root)

        self.assertTrue(deleted)
        self.assertFalse(topic_path.exists())
        self.assertEqual(index_entries, [])

    def test_file_store_supports_save_load_list_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "long_term"
            store = FileLongTermMemoryStore(root)

            store.upsert(
                {
                    "memory_id": "reference/api_contract",
                    "name": "API Contract",
                    "description": "Reference notes for the memory API.",
                    "memory_type": "reference",
                    "content_markdown": "Use the file store for durable memory mutations.",
                }
            )
            listed = store.list()
            loaded = store.get("reference/api_contract")
            deleted = store.delete("reference/api_contract")

        self.assertEqual([entry.memory_id for entry in listed], ["reference/api_contract"])
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.memory_type, "reference")
        self.assertTrue(deleted)

    def test_resolve_long_term_memory_index_file_rejects_ambiguous_index_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "long_term"
            write_memory_markdown(
                root / "MEMORY.md",
                name="Memory Index",
                description="Markdown index file.",
                memory_type="reference",
            )
            write_memory_markdown(
                root / "MEMORY",
                name="Memory Index Duplicate",
                description="Duplicate suffixless index file.",
                memory_type="reference",
            )

            with self.assertRaisesRegex(LongTermMemoryFormatError, "multiple MEMORY index candidates"):
                resolve_long_term_memory_index_file(root)

    def test_list_long_term_memories_does_not_hide_ambiguous_index_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "long_term"
            write_memory_markdown(
                root / "MEMORY.md",
                name="Memory Index",
                description="Markdown index file.",
                memory_type="reference",
            )
            write_memory_markdown(
                root / "MEMORY",
                name="Memory Index Duplicate",
                description="Duplicate suffixless index file.",
                memory_type="reference",
            )

            with self.assertRaisesRegex(LongTermMemoryFormatError, "multiple MEMORY index candidates"):
                list_long_term_memories(root)

    def test_normalize_long_term_memory_id_rejects_path_escape_segments(self) -> None:
        with self.assertRaisesRegex(LongTermMemoryFormatError, "Invalid long-term memory id"):
            normalize_long_term_memory_id("../escape")


if __name__ == "__main__":
    unittest.main()
