from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.memory.long_term import (
    LongTermMemoryFormatError,
    load_long_term_memory_catalog,
    load_long_term_memory_file,
    resolve_long_term_memory_index_file,
)


def write_memory_markdown(
    path: Path,
    *,
    name: str,
    description: str,
    memory_type: str,
    body: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        [
            "---",
            f"name: {name}",
            f"description: {description}",
            f"type: {memory_type}",
            "---",
            "",
            body.strip(),
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


class LongTermMemoryFilesTests(unittest.TestCase):
    def test_load_long_term_memory_catalog_reads_index_and_topic_files(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "long_term"
            write_memory_markdown(
                root / "MEMORY.md",
                name="Memory Index",
                description="Top-level long-term memory catalog.",
                memory_type="reference",
                body="This index summarizes the durable memory set.",
            )
            write_memory_markdown(
                root / "project_overview.md",
                name="Project Overview",
                description="Shared roadmap and release context.",
                memory_type="project",
                body="The roadmap is currently focused on the memory subsystem.",
            )
            write_memory_markdown(
                root / "users" / "tester_preferences.md",
                name="Tester Preferences",
                description="Stable user preferences for the current workspace.",
                memory_type="user",
                body="Prefer concise updates with file references.",
            )
            (root / ".gitkeep").write_text("", encoding="utf-8")
            (root / "notes.txt").write_text("ignore me", encoding="utf-8")

            catalog = load_long_term_memory_catalog(root)

        self.assertEqual(catalog.index_file.memory_id, "MEMORY")
        self.assertEqual(catalog.index_file.memory_type, "reference")
        self.assertEqual(
            [topic.memory_id for topic in catalog.topic_files],
            ["project_overview", "users/tester_preferences"],
        )
        self.assertEqual(
            [topic.memory_type for topic in catalog.topic_files],
            ["project", "user"],
        )
        self.assertEqual(catalog.topic_files[1].relative_path, "users/tester_preferences.md")

    def test_load_long_term_memory_catalog_requires_memory_index_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "long_term"
            root.mkdir(parents=True, exist_ok=True)
            write_memory_markdown(
                root / "project_overview.md",
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

    def test_load_long_term_memory_file_rejects_unsupported_type(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir) / "long_term"
            path = root / "feedback.md"
            write_memory_markdown(
                path,
                name="Feedback Summary",
                description="Recent product feedback.",
                memory_type="incident",
                body="This should fail validation.",
            )

            with self.assertRaisesRegex(LongTermMemoryFormatError, "incident"):
                load_long_term_memory_file(path, root_dir=root)

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


if __name__ == "__main__":
    unittest.main()
