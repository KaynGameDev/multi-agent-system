from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from agents.project_task.agent import build_project_task_prompt
from app.memory.agent_scope import resolve_agent_memory_context, retrieve_relevant_agent_memories
from app.memory.long_term import upsert_long_term_memory
from app.memory.retrieval import retrieve_relevant_long_term_memories
import app.memory.retrieval as retrieval_module
from tests.common import make_settings


class MemoryRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.runtime_root = Path(self.tempdir.name) / "runtime"
        self.settings = replace(
            make_settings(self.runtime_root),
            memory_retrieval_enabled=True,
            memory_retrieval_default_limit=4,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_retrieval_ranks_headers_before_loading_top_matches(self) -> None:
        root = self.runtime_root / "memory" / "long_term"
        upsert_long_term_memory(
            root,
            {
                "memory_id": "preferences/task-view",
                "name": "Task View Preference",
                "description": "Preferred task response layout for due-date summaries.",
                "memory_type": "user",
                "content_markdown": "Prefer grouped due-date summaries.",
            },
        )
        upsert_long_term_memory(
            root,
            {
                "memory_id": "project/roadmap",
                "name": "Roadmap Snapshot",
                "description": "Current roadmap priorities for the memory subsystem.",
                "memory_type": "project",
                "content_markdown": "Roadmap details.",
            },
        )
        upsert_long_term_memory(
            root,
            {
                "memory_id": "reference/release-checklist",
                "name": "Release Checklist",
                "description": "Reference deployment checklist.",
                "memory_type": "reference",
                "content_markdown": "Checklist body.",
            },
        )

        with patch("app.memory.retrieval.load_long_term_memory_file", wraps=retrieval_module.load_long_term_memory_file) as mocked_loader:
            results = retrieve_relevant_long_term_memories(
                root,
                query_text="show my task preference summary",
                memory_scope="user",
                scope_key="user-123",
                top_k=1,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].memory_id, "preferences/task-view")
        self.assertEqual(mocked_loader.call_count, 1)

    def test_retrieval_skips_already_loaded_memory_paths(self) -> None:
        root = self.runtime_root / "memory" / "long_term"
        first = upsert_long_term_memory(
            root,
            {
                "memory_id": "preferences/task-view",
                "name": "Task View Preference",
                "description": "Preferred task response layout.",
                "memory_type": "user",
                "content_markdown": "Prefer grouped due-date summaries.",
            },
        )
        upsert_long_term_memory(
            root,
            {
                "memory_id": "preferences/slack-tone",
                "name": "Slack Tone Preference",
                "description": "Preferred tone for Slack updates.",
                "memory_type": "user",
                "content_markdown": "Keep updates concise and direct.",
            },
        )

        results = retrieve_relevant_long_term_memories(
            root,
            query_text="what preference should I use",
            memory_scope="user",
            scope_key="user-123",
            top_k=2,
            loaded_paths=[first.source_path],
        )

        self.assertEqual([result.memory_id for result in results], ["preferences/slack-tone"])

    def test_retrieve_relevant_agent_memories_uses_scope_specific_root(self) -> None:
        state = {
            "user_id": "user-123",
            "thread_id": "web:thread-1",
            "messages": [HumanMessage(content="Remember my preferred task view.")],
        }
        context = resolve_agent_memory_context(
            self.settings,
            agent_name="project_task_agent",
            memory_scope="user",
            state=state,
        )
        upsert_long_term_memory(
            context.root_dir,
            {
                "memory_id": "preferences/task-view",
                "name": "Task View Preference",
                "description": "Preferred task response layout.",
                "memory_type": "user",
                "content_markdown": "Prefer grouped due-date summaries.",
            },
        )

        results = retrieve_relevant_agent_memories(
            self.settings,
            agent_name="project_task_agent",
            memory_scope="user",
            state=state,
            query_text="what is my task view preference",
            top_k=2,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].scope, "user")
        self.assertEqual(results[0].scope_key, "user-123")
        self.assertEqual(results[0].memory_id, "preferences/task-view")

    def test_project_task_prompt_injects_relevant_memories(self) -> None:
        state = {
            "interface_name": "web",
            "user_id": "user-123",
            "user_sheet_name": "Tester",
            "user_google_name": "",
            "user_job_title": "",
            "user_mapped_slack_name": "Tester",
            "thread_id": "web:thread-1",
            "messages": [HumanMessage(content="How should you format my task updates?")],
            "context_paths": [],
            "recent_file_reads": [],
        }
        context = resolve_agent_memory_context(
            self.settings,
            agent_name="project_task_agent",
            memory_scope="user",
            state=state,
        )
        upsert_long_term_memory(
            context.root_dir,
            {
                "memory_id": "preferences/task-view",
                "name": "Task View Preference",
                "description": "Preferred task response layout.",
                "memory_type": "user",
                "content_markdown": "Prefer grouped due-date summaries.",
            },
        )

        prompt = build_project_task_prompt(
            state,
            settings=self.settings,
            agent_name="project_task_agent",
            tool_ids=(),
            memory_scope="user",
        )

        self.assertIn("# Relevant Memories", prompt)
        self.assertIn("Task View Preference", prompt)
        self.assertIn("Prefer grouped due-date summaries.", prompt)


if __name__ == "__main__":
    unittest.main()
