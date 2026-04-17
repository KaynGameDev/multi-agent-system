from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app.memory.agent_scope import resolve_agent_memory_context
from app.memory.extraction import (
    extract_durable_turn_memories,
    persist_durable_turn_memories,
    turn_has_direct_memory_write,
)
from app.memory.long_term import FileLongTermMemoryStore
from app.rehydration import RUNTIME_REHYDRATION_METADATA_KEY
from tests.common import make_settings


class MemoryExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.settings = replace(
            make_settings(self.root / "runtime"),
            long_term_memory_enabled=True,
        )

    def test_extract_durable_turn_memories_promotes_explicit_user_preferences(self) -> None:
        candidates = extract_durable_turn_memories(
            [
                {
                    "role": "user",
                    "type": "message",
                    "markdown": "Please keep future updates concise, include file references, and call me Kay.",
                },
                {
                    "role": "assistant",
                    "type": "message",
                    "markdown": "I will keep future updates concise and include file references.",
                },
            ]
        )

        self.assertEqual(
            [candidate.memory_id for candidate in candidates],
            [
                "profile/preferred-name",
                "preferences/response-style",
                "preferences/file-references",
            ],
        )
        self.assertEqual(candidates[0].content_markdown, "Call the user `Kay` in future replies.")
        self.assertEqual(candidates[1].memory_type, "user")

    def test_extract_durable_turn_memories_skips_ephemeral_task_requests(self) -> None:
        candidates = extract_durable_turn_memories(
            [
                {
                    "role": "user",
                    "type": "message",
                    "markdown": "What tasks are due today for the client team?",
                },
                {
                    "role": "assistant",
                    "type": "message",
                    "markdown": "Here are the tasks due today.",
                },
            ]
        )

        self.assertEqual(candidates, [])

    def test_turn_has_direct_memory_write_detects_runtime_memory_tool_usage(self) -> None:
        assistant_metadata = {
            RUNTIME_REHYDRATION_METADATA_KEY: {
                "tool_execution_trace": [
                    {
                        "result": {
                            "tool_name": "write_agent_memory",
                            "tool_id": "memory.write",
                            "status": "ok",
                        }
                    }
                ]
            }
        }

        self.assertTrue(turn_has_direct_memory_write(assistant_metadata))

    def test_persist_durable_turn_memories_writes_scoped_files_to_disk(self) -> None:
        state = {
            "user_id": "Tester@example.com",
            "thread_id": "web:test-thread",
            "channel_id": "test-conversation",
        }

        saved = persist_durable_turn_memories(
            self.settings,
            agent_name="project_task_agent",
            memory_scope="user",
            state=state,
            transcript_messages=[
                {
                    "role": "user",
                    "type": "message",
                    "markdown": "Please keep future updates concise and call me Kay.",
                },
                {
                    "role": "assistant",
                    "type": "message",
                    "markdown": "Understood. I will keep future updates concise.",
                },
            ],
        )

        context = resolve_agent_memory_context(
            self.settings,
            agent_name="project_task_agent",
            memory_scope="user",
            state=state,
        )
        store = FileLongTermMemoryStore(context.root_dir)

        self.assertEqual([memory.memory_id for memory in saved], ["profile/preferred-name", "preferences/response-style"])
        self.assertEqual(store.get("profile/preferred-name").content_markdown, "Call the user `Kay` in future replies.")
        self.assertEqual(store.get("preferences/response-style").memory_type, "user")


if __name__ == "__main__":
    unittest.main()
