from __future__ import annotations

import unittest

from core.knowledge_rendering import render_knowledge_payload


LIST_PAYLOAD = {
    "ok": True,
    "document_count": 2,
    "documents": [
        {
            "title": "Game Design Spec",
            "path": "design/game_design.csv",
            "file_type": ".csv",
        },
        {
            "title": "Ops Guide",
            "path": "ops/guide.md",
            "file_type": ".md",
        },
    ],
}

SEARCH_PAYLOAD = {
    "ok": True,
    "query": "reward rule",
    "match_count": 1,
    "documents": [
        {
            "title": "Game Design Spec",
            "path": "design/game_design.csv",
            "file_type": ".csv",
            "snippet": "## Sheet: Rewards\nRow 2: Reward Rule | Base reward * crit",
        }
    ],
}

READ_PAYLOAD = {
    "ok": True,
    "document": {
        "title": "Game Design Spec",
        "path": "design/game_design.csv",
        "file_type": ".csv",
    },
    "section_query": "Rewards",
    "start_line": 10,
    "end_line": 14,
    "content": "## Sheet: Rewards\nColumns: Rule | Value\nRow 2: Base | 100",
    "truncated": False,
}

TRUNCATED_READ_PAYLOAD = {
    **READ_PAYLOAD,
    "truncated": True,
}

ERROR_PAYLOAD = {
    "ok": False,
    "error": "Document not found: missing_spec",
}


class KnowledgeRenderingTests(unittest.TestCase):
    def test_render_list_payload_uses_numbered_document_blocks(self) -> None:
        rendered = render_knowledge_payload(LIST_PAYLOAD)

        self.assertIsNotNone(rendered)
        self.assertIn("Documents (2)", rendered)
        self.assertIn("1. Game Design Spec", rendered)
        self.assertIn("Path: design/game_design.csv | Type: .csv", rendered)
        self.assertIn("2. Ops Guide", rendered)

    def test_render_search_payload_includes_snippet(self) -> None:
        rendered = render_knowledge_payload(SEARCH_PAYLOAD)

        self.assertIsNotNone(rendered)
        self.assertIn('Matches for "reward rule" (1)', rendered)
        self.assertIn("1. Game Design Spec", rendered)
        self.assertIn("Snippet: ## Sheet: Rewards / Row 2: Reward Rule | Base reward * crit", rendered)

    def test_render_read_payload_uses_fenced_excerpt(self) -> None:
        rendered = render_knowledge_payload(READ_PAYLOAD)

        self.assertIsNotNone(rendered)
        self.assertIn("Game Design Spec", rendered)
        self.assertIn("Path: design/game_design.csv | Section: Rewards | Lines: 10-14", rendered)
        self.assertIn("```text", rendered)
        self.assertIn("## Sheet: Rewards", rendered)

    def test_render_truncated_read_payload_includes_truncation_note(self) -> None:
        rendered = render_knowledge_payload(TRUNCATED_READ_PAYLOAD)

        self.assertIsNotNone(rendered)
        self.assertIn("Excerpt truncated.", rendered)

    def test_render_error_payload_is_a_short_sentence(self) -> None:
        rendered = render_knowledge_payload(ERROR_PAYLOAD)

        self.assertEqual(
            rendered,
            "I couldn't retrieve that document information: Document not found: missing_spec",
        )


if __name__ == "__main__":
    unittest.main()
