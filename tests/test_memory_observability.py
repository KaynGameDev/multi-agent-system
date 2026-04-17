from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app.compaction import ContinuationSummaryDraft, compact_conversation
from app.frontmatter import render_frontmatter_document
from app.memory.extraction import persist_durable_turn_memories
from app.memory.long_term import (
    LongTermMemoryFormatError,
    delete_long_term_memory,
    get_long_term_memory,
    list_long_term_memories,
    upsert_long_term_memory,
)
from app.memory.retrieval import retrieve_relevant_long_term_memories
from app.memory.snapshots import (
    apply_long_term_memory_snapshot,
    get_pending_long_term_memory_snapshot,
    resolve_long_term_memory_snapshot_dir,
)
from app.session_memory import SessionMemoryStore, build_session_memory_record
from tests.common import make_settings


class DummySummaryLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return ContinuationSummaryDraft(
            durable_summary="Keep the durable implementation context.",
        )


def parse_memory_telemetry_logs(lines: list[str]) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    marker = "memory telemetry "
    for line in lines:
        if marker not in line:
            continue
        payloads.append(json.loads(line.split(marker, 1)[1]))
    return payloads


def find_telemetry_payload(
    payloads: list[dict[str, object]],
    *,
    event: str,
    status: str | None = None,
    action: str | None = None,
) -> dict[str, object]:
    for payload in payloads:
        if payload.get("event") != event:
            continue
        if status is not None and payload.get("status") != status:
            continue
        if action is not None and payload.get("action") != action:
            continue
        return payload
    raise AssertionError(f"Missing telemetry payload event={event!r} status={status!r} action={action!r}: {payloads!r}")


class MemoryObservabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.runtime_root = Path(self.tempdir.name) / "runtime"
        self.settings = replace(make_settings(self.runtime_root), long_term_memory_enabled=True)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_long_term_crud_and_failure_emit_telemetry(self) -> None:
        root = self.runtime_root / "memory" / "long_term"

        with self.assertLogs("app.memory.long_term", level="INFO") as logs:
            self.assertEqual(list_long_term_memories(root), [])
            saved = upsert_long_term_memory(
                root,
                {
                    "memory_id": "preferences/task-view",
                    "name": "Task View Preference",
                    "description": "Preferred task response layout.",
                    "memory_type": "user",
                    "content_markdown": "Prefer grouped due-date summaries.",
                },
            )
            loaded = get_long_term_memory(root, "preferences/task-view")
            deleted = delete_long_term_memory(root, "preferences/task-view")

        payloads = parse_memory_telemetry_logs(logs.output)
        self.assertEqual(saved.memory_id, "preferences/task-view")
        self.assertIsNotNone(loaded)
        self.assertTrue(deleted)
        self.assertEqual(find_telemetry_payload(payloads, event="long_term.list", status="empty")["count"], 0)
        self.assertEqual(find_telemetry_payload(payloads, event="long_term.upsert", status="ok")["memory_id"], "preferences/task-view")
        self.assertEqual(find_telemetry_payload(payloads, event="long_term.get", status="ok")["relative_path"], "topics/preferences/task-view.md")
        self.assertTrue(find_telemetry_payload(payloads, event="long_term.delete", status="ok")["deleted"])

        with self.assertLogs("app.memory.long_term", level="WARNING") as error_logs:
            with self.assertRaisesRegex(LongTermMemoryFormatError, "Invalid long-term memory id"):
                upsert_long_term_memory(
                    root,
                    {
                        "memory_id": "../escape",
                        "name": "Escape Attempt",
                        "description": "Should fail.",
                        "memory_type": "user",
                        "content_markdown": "bad",
                    },
                )

        error_payload = find_telemetry_payload(
            parse_memory_telemetry_logs(error_logs.output),
            event="long_term.upsert",
            status="error",
        )
        self.assertIn("Invalid long-term memory id", str(error_payload["error"]))

    def test_retrieval_quality_and_failure_emit_telemetry(self) -> None:
        root = self.runtime_root / "memory" / "retrieval-long-term"
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

        with self.assertLogs("app.memory.retrieval", level="INFO") as logs:
            results = retrieve_relevant_long_term_memories(
                root,
                query_text="task view preference",
                memory_scope="user",
                scope_key="user-123",
                top_k=1,
            )

        payload = find_telemetry_payload(
            parse_memory_telemetry_logs(logs.output),
            event="retrieval.search",
            status="ok",
        )
        self.assertEqual(results[0].memory_id, "preferences/task-view")
        self.assertEqual(payload["candidate_count"], 2)
        self.assertEqual(payload["selected_count"], 1)
        self.assertGreater(float(payload["top_score"]), 0.0)

        bad_root = self.runtime_root / "memory" / "bad-retrieval"
        bad_root.mkdir(parents=True, exist_ok=True)
        (bad_root / "MEMORY.md").write_text(
            render_frontmatter_document(
                {
                    "name": "Memory Index",
                    "description": "Broken index for retrieval testing.",
                    "type": "reference",
                },
                "## Topics\n\n- invalid entry",
            ),
            encoding="utf-8",
        )

        with self.assertLogs("app.memory.retrieval", level="WARNING") as error_logs:
            bad_results = retrieve_relevant_long_term_memories(
                bad_root,
                query_text="anything useful",
                memory_scope="user",
                scope_key="user-123",
            )

        self.assertEqual(bad_results, [])
        error_payload = find_telemetry_payload(
            parse_memory_telemetry_logs(error_logs.output),
            event="retrieval.search",
            status="error",
        )
        self.assertIn("Invalid long-term memory index entry", str(error_payload["error"]))

    def test_session_memory_updates_and_failure_emit_telemetry(self) -> None:
        messages = [
            {
                "id": "u1",
                "role": "user",
                "type": "message",
                "markdown": "Please summarize the current state.",
                "created_at": "2026-04-17T00:00:00+00:00",
            },
            {
                "id": "a1",
                "role": "assistant",
                "type": "message",
                "markdown": "Working on the memory subsystem summary now.",
                "created_at": "2026-04-17T00:00:01+00:00",
            },
        ]
        store = SessionMemoryStore(self.runtime_root / "session_memory.json")

        with self.assertLogs("app.session_memory", level="INFO") as logs:
            record = build_session_memory_record(
                "web:test-thread",
                messages,
                initialize_threshold_tokens=1,
                llm=DummySummaryLLM(),
            )
            self.assertIsNotNone(record)
            store.upsert(record)
            store.delete("web:test-thread")
            skipped = build_session_memory_record(
                "",
                messages,
                initialize_threshold_tokens=1,
                llm=DummySummaryLLM(),
            )

        payloads = parse_memory_telemetry_logs(logs.output)
        self.assertIsNone(skipped)
        self.assertEqual(find_telemetry_payload(payloads, event="session_memory.build_record", status="ok")["source"], "initialize")
        self.assertEqual(find_telemetry_payload(payloads, event="session_memory.store_upsert", status="ok")["thread_id"], "web:test-thread")
        self.assertTrue(find_telemetry_payload(payloads, event="session_memory.store_delete", status="ok")["deleted"])
        self.assertEqual(
            find_telemetry_payload(payloads, event="session_memory.build_record", status="skip")["reason"],
            "empty_thread_id",
        )

        with self.assertLogs("app.session_memory", level="WARNING") as error_logs:
            with self.assertRaises(ValueError):
                store.upsert({"thread_id": "", "summary_markdown": ""})

        error_payload = find_telemetry_payload(
            parse_memory_telemetry_logs(error_logs.output),
            event="session_memory.store_upsert",
            status="error",
        )
        self.assertIn("thread_id and summary_markdown", str(error_payload["error"]))

    def test_compaction_boundaries_and_failure_emit_telemetry(self) -> None:
        messages = [
            {
                "id": "u1",
                "role": "user",
                "type": "message",
                "markdown": "First request",
                "created_at": "2026-04-17T00:00:00+00:00",
            },
            {
                "id": "a1",
                "role": "assistant",
                "type": "message",
                "markdown": "First answer",
                "created_at": "2026-04-17T00:00:01+00:00",
            },
            {
                "id": "u2",
                "role": "user",
                "type": "message",
                "markdown": "Latest follow-up",
                "created_at": "2026-04-17T00:00:02+00:00",
            },
        ]

        with self.assertLogs("app.compaction", level="INFO") as logs:
            bundle = compact_conversation(
                messages,
                llm=DummySummaryLLM(),
                trigger="manual",
                preserved_tail_count=1,
            )

        payload = find_telemetry_payload(
            parse_memory_telemetry_logs(logs.output),
            event="compaction.run",
            status="ok",
        )
        self.assertEqual(bundle.compacted_source_count, 2)
        self.assertEqual(payload["preserved_tail_count"], 1)
        self.assertEqual(payload["compacted_source_count"], 2)
        self.assertFalse(payload["used_session_memory"])

        with self.assertLogs("app.compaction", level="WARNING") as error_logs:
            with self.assertRaisesRegex(ValueError, "empty active transcript slice"):
                compact_conversation([], llm=DummySummaryLLM())

        error_payload = find_telemetry_payload(
            parse_memory_telemetry_logs(error_logs.output),
            event="compaction.run",
            status="error",
        )
        self.assertIn("empty active transcript slice", str(error_payload["error"]))

    def test_extractor_behavior_and_failure_emit_telemetry(self) -> None:
        transcript_messages = [
            {
                "role": "user",
                "type": "message",
                "markdown": "Please keep future updates concise and call me Kay.",
            },
            {
                "role": "assistant",
                "type": "message",
                "markdown": "Understood. I will keep updates concise and call you Kay.",
            },
        ]

        with self.assertLogs("app.memory.extraction", level="INFO") as logs:
            saved = persist_durable_turn_memories(
                self.settings,
                agent_name="project_task_agent",
                memory_scope="user",
                state={"user_id": "user-123", "thread_id": "web:thread-1"},
                transcript_messages=transcript_messages,
            )

        payloads = parse_memory_telemetry_logs(logs.output)
        self.assertGreaterEqual(len(saved), 1)
        self.assertGreaterEqual(find_telemetry_payload(payloads, event="extraction.extract", status="ok")["candidate_count"], 1)
        self.assertEqual(find_telemetry_payload(payloads, event="extraction.persist", status="ok")["saved_count"], len(saved))

        with self.assertLogs("app.memory.extraction", level="WARNING") as error_logs:
            with self.assertRaisesRegex(ValueError, "Unsupported agent memory scope"):
                persist_durable_turn_memories(
                    self.settings,
                    agent_name="project_task_agent",
                    memory_scope="bogus",
                    state={"user_id": "user-123", "thread_id": "web:thread-1"},
                    transcript_messages=transcript_messages,
                )

        error_payload = find_telemetry_payload(
            parse_memory_telemetry_logs(error_logs.output),
            event="extraction.persist",
            status="error",
        )
        self.assertIn("Unsupported agent memory scope", str(error_payload["error"]))

    def test_snapshot_keep_merge_replace_and_failure_emit_telemetry(self) -> None:
        project_root = self.runtime_root / "memory" / "project-scope"
        user_root = self.runtime_root / "memory" / "user-scope"
        snapshot_root = resolve_long_term_memory_snapshot_dir(project_root, "default")
        upsert_long_term_memory(
            snapshot_root,
            {
                "memory_id": "project/overview",
                "name": "Project Overview",
                "description": "Shared roadmap context.",
                "memory_type": "project",
                "content_markdown": "Snapshot baseline.",
            },
        )
        upsert_long_term_memory(
            user_root,
            {
                "memory_id": "project/overview",
                "name": "Project Overview",
                "description": "Shared roadmap context.",
                "memory_type": "project",
                "content_markdown": "User-added note.",
            },
        )

        with self.assertLogs("app.memory.snapshots", level="INFO") as logs:
            pending = get_pending_long_term_memory_snapshot(project_root, user_root)
            keep_summary = apply_long_term_memory_snapshot(user_root, project_root, action="keep")
            merge_summary = apply_long_term_memory_snapshot(user_root, project_root, action="merge")
            replace_summary = apply_long_term_memory_snapshot(user_root, project_root, action="replace")

        payloads = parse_memory_telemetry_logs(logs.output)
        self.assertIsNotNone(pending)
        self.assertEqual(keep_summary.action, "keep")
        self.assertEqual(merge_summary.action, "merge")
        self.assertEqual(replace_summary.action, "replace")
        self.assertEqual(find_telemetry_payload(payloads, event="snapshots.pending", status="ok")["snapshot_id"], "default")
        self.assertEqual(find_telemetry_payload(payloads, event="snapshots.apply", status="ok", action="keep")["created_count"], 0)
        self.assertGreaterEqual(find_telemetry_payload(payloads, event="snapshots.apply", status="ok", action="merge")["updated_count"], 1)
        self.assertGreaterEqual(find_telemetry_payload(payloads, event="snapshots.apply", status="ok", action="replace")["deleted_count"], 1)

        with self.assertLogs("app.memory.snapshots", level="WARNING") as error_logs:
            with self.assertRaisesRegex(Exception, "No project-provided memory snapshot is available"):
                apply_long_term_memory_snapshot(
                    self.runtime_root / "memory" / "missing-user",
                    self.runtime_root / "memory" / "missing-project",
                    action="merge",
                )

        error_payload = find_telemetry_payload(
            parse_memory_telemetry_logs(error_logs.output),
            event="snapshots.apply",
            status="error",
        )
        self.assertIn("No project-provided memory snapshot is available", str(error_payload["error"]))


if __name__ == "__main__":
    unittest.main()
