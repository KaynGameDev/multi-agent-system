from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.memory.session_files import get_session_memory_file
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.session_memory import (
    DEFAULT_SESSION_MEMORY_BACKGROUND_MIN_TURNS,
    SessionMemoryRefreshActivity,
    SessionMemoryRecord,
    SessionMemoryStore,
    build_session_memory_compaction_plan,
    count_session_memory_refresh_activity,
    is_safe_session_memory_extraction_point,
    should_schedule_background_session_memory_refresh,
    should_initialize_session_memory,
    should_update_session_memory,
)


class SessionMemoryTests(unittest.TestCase):
    def test_should_initialize_only_after_assistant_turn_and_threshold(self) -> None:
        assistant_finished_messages = [
            {
                "id": "u1",
                "role": "user",
                "type": "message",
                "markdown": "Open the architecture plan",
                "created_at": "2026-04-16T00:00:00+00:00",
            },
            {
                "id": "a1",
                "role": "assistant",
                "type": "message",
                "markdown": "I opened the architecture plan and summarized the next steps.",
                "created_at": "2026-04-16T00:00:01+00:00",
            },
        ]
        user_finished_messages = [*assistant_finished_messages, {
            "id": "u2",
            "role": "user",
            "type": "message",
            "markdown": "Keep going",
            "created_at": "2026-04-16T00:00:02+00:00",
        }]

        self.assertTrue(
            should_initialize_session_memory(
                assistant_finished_messages,
                initialize_threshold_tokens=1,
            )
        )
        self.assertFalse(
            should_initialize_session_memory(
                assistant_finished_messages,
                initialize_threshold_tokens=1_000_000,
            )
        )
        self.assertFalse(
            should_initialize_session_memory(
                user_finished_messages,
                initialize_threshold_tokens=1,
            )
        )

    def test_safe_extraction_point_rejects_pending_tool_call_turn(self) -> None:
        messages = [
            HumanMessage(content="Open the architecture guide", id="u1"),
            AIMessage(
                content="",
                id="a_tool",
                tool_calls=[{"id": "call_read", "name": "read_knowledge_document", "args": {"document": "Guide"}}],
            ),
        ]

        self.assertFalse(is_safe_session_memory_extraction_point(messages))

    def test_should_update_after_enough_growth_from_last_memory_point(self) -> None:
        messages = [
            {
                "id": "u1",
                "role": "user",
                "type": "message",
                "markdown": "Open the architecture plan",
                "created_at": "2026-04-16T00:00:00+00:00",
            },
            {
                "id": "a1",
                "role": "assistant",
                "type": "message",
                "markdown": "I opened the architecture plan.",
                "created_at": "2026-04-16T00:00:01+00:00",
            },
            {
                "id": "u2",
                "role": "user",
                "type": "message",
                "markdown": "Please compare it with the release checklist.",
                "created_at": "2026-04-16T00:00:02+00:00",
            },
            {
                "id": "a2",
                "role": "assistant",
                "type": "message",
                "markdown": "The architecture plan and release checklist disagree on sequencing.",
                "created_at": "2026-04-16T00:00:03+00:00",
            },
        ]
        existing_record = SessionMemoryRecord(
            thread_id="web:test",
            updated_at="2026-04-16T00:00:01+00:00",
            last_message_id="a1",
            last_message_created_at="2026-04-16T00:00:01+00:00",
            covered_message_count=2,
            covered_tokens=200,
            summary_markdown="## Continuation Summary\nInitial memory",
        )

        self.assertTrue(
            should_update_session_memory(
                messages,
                existing_record,
                update_growth_threshold_tokens=1,
            )
        )
        self.assertFalse(
            should_update_session_memory(
                messages,
                existing_record,
                update_growth_threshold_tokens=1_000_000,
            )
        )

    def test_background_refresh_counts_turns_and_tool_activity(self) -> None:
        messages = [
            {
                "id": "u1",
                "role": "user",
                "type": "message",
                "markdown": "Open the architecture plan",
                "created_at": "2026-04-16T00:00:00+00:00",
            },
            {
                "id": "a1",
                "role": "assistant",
                "type": "message",
                "markdown": "I opened the architecture plan.",
                "created_at": "2026-04-16T00:00:01+00:00",
                "metadata": {
                    "runtime_rehydration_state": {
                        "tool_result": {"tool_name": "read_knowledge_document"},
                        "tool_execution_trace": [{"result": {"tool_name": "read_knowledge_document"}}],
                    }
                },
            },
        ]

        activity = count_session_memory_refresh_activity(messages)

        self.assertEqual(
            activity,
            SessionMemoryRefreshActivity(turn_count=2, tool_activity_count=2),
        )

    def test_background_refresh_schedules_after_turns_or_tool_activity(self) -> None:
        messages = [
            {
                "id": "u1",
                "role": "user",
                "type": "message",
                "markdown": "Open the architecture plan",
                "created_at": "2026-04-16T00:00:00+00:00",
            },
            {
                "id": "a1",
                "role": "assistant",
                "type": "message",
                "markdown": "I opened the architecture plan.",
                "created_at": "2026-04-16T00:00:01+00:00",
                "metadata": {
                    "runtime_rehydration_state": {
                        "tool_result": {"tool_name": "read_knowledge_document"},
                    }
                },
            },
        ]

        self.assertTrue(
            should_schedule_background_session_memory_refresh(
                messages,
                None,
                initialize_threshold_tokens=1_000_000,
                update_growth_threshold_tokens=1_000_000,
                min_turns=DEFAULT_SESSION_MEMORY_BACKGROUND_MIN_TURNS,
            )
        )

    def test_session_memory_store_round_trips_records(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        storage_path = Path(temp_dir.name) / "session_memory.json"
        store = SessionMemoryStore(storage_path)

        record = SessionMemoryRecord(
            thread_id="web:test",
            updated_at="2026-04-16T00:00:03+00:00",
            last_message_id="a2",
            last_message_created_at="2026-04-16T00:00:03+00:00",
            covered_message_count=4,
            covered_tokens=512,
            summary_markdown="## Continuation Summary\nRemember the architecture checklist comparison.",
            source="update",
        )
        store.upsert(record)

        reloaded_store = SessionMemoryStore(storage_path)
        self.assertEqual(reloaded_store.get("web:test"), record)
        synced_file = get_session_memory_file(storage_path.parent / "sessions", "web:test")
        self.assertIsNotNone(synced_file)
        self.assertEqual(synced_file.current_state, record.summary_markdown)

    def test_session_memory_store_delete_removes_session_file(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        storage_path = Path(temp_dir.name) / "session_memory.json"
        store = SessionMemoryStore(storage_path)

        store.upsert(
            SessionMemoryRecord(
                thread_id="web:test",
                updated_at="2026-04-16T00:00:03+00:00",
                last_message_id="a2",
                last_message_created_at="2026-04-16T00:00:03+00:00",
                covered_message_count=4,
                covered_tokens=512,
                summary_markdown="## Continuation Summary\nRemember the architecture checklist comparison.",
                source="update",
            )
        )

        self.assertIsNotNone(get_session_memory_file(storage_path.parent / "sessions", "web:test"))
        store.delete("web:test")
        self.assertIsNone(get_session_memory_file(storage_path.parent / "sessions", "web:test"))

    def test_compaction_plan_requires_delta_to_fit_preserved_tail(self) -> None:
        messages = [
            {
                "id": "u1",
                "role": "user",
                "type": "message",
                "markdown": "Open the architecture plan",
                "created_at": "2026-04-16T00:00:00+00:00",
            },
            {
                "id": "a1",
                "role": "assistant",
                "type": "message",
                "markdown": "I opened the architecture plan.",
                "created_at": "2026-04-16T00:00:01+00:00",
            },
            {
                "id": "u2",
                "role": "user",
                "type": "message",
                "markdown": "Keep going",
                "created_at": "2026-04-16T00:00:02+00:00",
            },
        ]
        existing_record = SessionMemoryRecord(
            thread_id="web:test",
            updated_at="2026-04-16T00:00:01+00:00",
            last_message_id="a1",
            last_message_created_at="2026-04-16T00:00:01+00:00",
            covered_message_count=2,
            covered_tokens=200,
            summary_markdown="## Continuation Summary\nInitial memory",
        )

        fitting_plan = build_session_memory_compaction_plan(
            messages,
            existing_record,
            preserved_tail_count=1,
        )
        self.assertIsNotNone(fitting_plan)
        self.assertEqual([message["id"] for message in fitting_plan.preserved_tail_messages], ["u2"])

        stale_plan = build_session_memory_compaction_plan(
            messages,
            existing_record,
            preserved_tail_count=0,
        )
        self.assertIsNone(stale_plan)

    def test_compaction_plan_preserves_whole_tool_pair_in_tail(self) -> None:
        messages = [
            HumanMessage(content="Open the architecture guide", id="u1"),
            AIMessage(content="I opened the guide.", id="a1"),
            AIMessage(
                content="",
                id="a_tool",
                tool_calls=[{"id": "call_read", "name": "read_knowledge_document", "args": {"document": "Checklist"}}],
            ),
            ToolMessage(
                content='{"ok": true, "document": {"title": "Checklist"}, "content": "short"}',
                tool_call_id="call_read",
                id="t_read",
            ),
        ]
        existing_record = SessionMemoryRecord(
            thread_id="web:test",
            updated_at="2026-04-16T00:00:01+00:00",
            last_message_id="a1",
            last_message_created_at="2026-04-16T00:00:01+00:00",
            covered_message_count=2,
            covered_tokens=200,
            summary_markdown="## Continuation Summary\nInitial memory",
        )

        fitting_plan = build_session_memory_compaction_plan(
            messages,
            existing_record,
            preserved_tail_count=1,
        )

        self.assertIsNotNone(fitting_plan)
        self.assertEqual([message["id"] for message in fitting_plan.preserved_tail_messages], ["a_tool", "t_read"])


if __name__ == "__main__":
    unittest.main()
