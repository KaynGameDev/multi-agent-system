from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.compaction import (
    ContinuationSummaryDraft,
    compact_conversation,
    format_continuation_summary,
)
from app.rehydration import RUNTIME_REHYDRATION_METADATA_KEY
from app.session_memory import SessionMemoryRecord
from interfaces.web.conversations import TRANSCRIPT_TYPE_COMPACT_BOUNDARY, WebConversationStore
from interfaces.web.server import WebServer
from tests.common import make_settings


class StructuredSummaryLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return ContinuationSummaryDraft(
            draft_analysis="This analysis should never survive formatting.",
            durable_summary="<think>hidden</think>\nThe user wants to keep moving on the same topic.",
            open_loops=["Confirm the next implementation step."],
            user_preferences=["Keep the reply concise."],
            assistant_commitments=["Carry forward the current implementation context."],
            attachment_notes=["spec-notes.md"],
        )


class ExplodingSummaryLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        raise AssertionError("Fresh summarization should not run when session memory is usable.")


class RecordingSummaryLLM:
    def __init__(self) -> None:
        self.requests = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        self.requests.append(list(messages))
        return ContinuationSummaryDraft(
            durable_summary="Keep the durable implementation context.",
        )


class RecordingGraph:
    def __init__(self, reply_text: str = "Recorded reply.") -> None:
        self.reply_text = reply_text
        self.last_state = None

    def invoke(self, initial_state, config=None):
        self.last_state = dict(initial_state)
        return {
            **initial_state,
            "route": "general_chat_agent",
            "route_reason": "Recorded route.",
            "skill_resolution_diagnostics": [],
            "agent_selection_diagnostics": [],
            "selection_warnings": [],
            "messages": [AIMessage(content=self.reply_text)],
        }


class MissingCheckpointStore:
    def has_checkpoint(self, _thread_id: str) -> bool:
        return False

    def delete_thread(self, _thread_id: str) -> None:
        return None


class CompactionTests(unittest.TestCase):
    def test_format_continuation_summary_strips_draft_analysis(self) -> None:
        formatted = format_continuation_summary(
            ContinuationSummaryDraft(
                draft_analysis="private reasoning",
                durable_summary="<draft_analysis>remove me</draft_analysis>\n<think>hide</think>\nDurable summary\nKeep the current plan.",
                open_loops=["<think>hidden</think>Confirm the next step."],
                assistant_commitments=["Preserve the active implementation context."],
            ),
            attachments=["diagram.png"],
        )

        self.assertIn("## Continuation Summary", formatted)
        self.assertIn("Keep the current plan.", formatted)
        self.assertIn("Confirm the next step.", formatted)
        self.assertIn("diagram.png", formatted)
        self.assertNotIn("private reasoning", formatted)
        self.assertNotIn("hide", formatted)
        self.assertNotIn("draft analysis", formatted.lower())

    def test_compact_conversation_builds_boundary_summary_and_preserved_tail(self) -> None:
        messages = [
            {
                "id": "u1",
                "role": "user",
                "type": "message",
                "markdown": "First request",
                "created_at": "2026-04-16T00:00:00+00:00",
            },
            {
                "id": "a1",
                "role": "assistant",
                "type": "message",
                "markdown": "First answer",
                "created_at": "2026-04-16T00:00:01+00:00",
            },
            {
                "id": "u2",
                "role": "user",
                "type": "message",
                "markdown": "Latest follow-up",
                "created_at": "2026-04-16T00:00:02+00:00",
            },
        ]

        bundle = compact_conversation(
            messages,
            llm=StructuredSummaryLLM(),
            trigger="manual",
            preserved_tail_count=1,
            attachments=["spec-notes.md"],
        )

        self.assertEqual(bundle.active_slice_start, 0)
        self.assertEqual(bundle.compacted_source_count, 2)
        self.assertEqual(bundle.boundary_message["type"], TRANSCRIPT_TYPE_COMPACT_BOUNDARY)
        self.assertEqual(bundle.boundary_message["metadata"]["trigger"], "manual")
        self.assertEqual(bundle.boundary_message["metadata"]["preservedTail"]["count"], 1)
        self.assertEqual(bundle.boundary_message["metadata"]["preservedTail"]["messageIds"], ["u2"])
        self.assertEqual(bundle.summary_message["role"], "assistant")
        self.assertIn("## Continuation Summary", bundle.summary_message["markdown"])
        self.assertIn("Keep the reply concise.", bundle.summary_message["markdown"])
        self.assertEqual(bundle.attachments, [{"label": "spec-notes.md"}])
        self.assertEqual([message["id"] for message in bundle.preserved_tail_messages], ["u2"])
        self.assertEqual(
            [message["type"] for message in bundle.compacted_messages],
            [TRANSCRIPT_TYPE_COMPACT_BOUNDARY, "message", "message"],
        )

    def test_compact_conversation_sends_source_slice_once_to_summarizer(self) -> None:
        messages = [
            {
                "id": "u1",
                "role": "user",
                "type": "message",
                "markdown": "First request",
                "created_at": "2026-04-16T00:00:00+00:00",
            },
            {
                "id": "a1",
                "role": "assistant",
                "type": "message",
                "markdown": "First answer",
                "created_at": "2026-04-16T00:00:01+00:00",
            },
        ]
        llm = RecordingSummaryLLM()

        bundle = compact_conversation(
            messages,
            llm=llm,
        )

        self.assertIn("## Continuation Summary", bundle.summary_message["markdown"])
        self.assertEqual(len(llm.requests), 1)
        request_messages = llm.requests[0]
        self.assertEqual(len(request_messages), 1)
        self.assertTrue(isinstance(request_messages[0], SystemMessage))
        request_content = request_messages[0].content
        self.assertEqual(request_content.count("- user: First request"), 1)
        self.assertEqual(request_content.count("- assistant: First answer"), 1)

    def test_compact_conversation_carries_runtime_rehydration_state_into_summary_metadata(self) -> None:
        messages = [
            {
                "id": "u1",
                "role": "user",
                "type": "message",
                "markdown": "Open the architecture guide",
                "created_at": "2026-04-16T00:00:00+00:00",
            },
            {
                "id": "a1",
                "role": "assistant",
                "type": "message",
                "markdown": "I opened the architecture guide.",
                "created_at": "2026-04-16T00:00:01+00:00",
                "metadata": {
                    RUNTIME_REHYDRATION_METADATA_KEY: {
                        "context_paths": ["knowledge/Docs/00_Shared/ArchitectureOverview.md"],
                        "tool_result": {
                            "tool_name": "read_knowledge_document",
                            "tool_id": "knowledge.read_document",
                            "status": "ok",
                            "payload": {
                                "ok": True,
                                "document": {
                                    "path": "knowledge/Docs/00_Shared/ArchitectureOverview.md",
                                },
                                "content": "Architecture excerpt",
                            },
                        },
                    }
                },
            },
        ]

        bundle = compact_conversation(
            messages,
            llm=StructuredSummaryLLM(),
        )

        rehydration_state = bundle.summary_message["metadata"][RUNTIME_REHYDRATION_METADATA_KEY]
        self.assertEqual(
            rehydration_state["context_paths"],
            ["knowledge/Docs/00_Shared/ArchitectureOverview.md"],
        )
        self.assertEqual(
            rehydration_state["recent_file_reads"],
            ["knowledge/Docs/00_Shared/ArchitectureOverview.md"],
        )

    def test_compact_conversation_prefers_session_memory_when_recent_tail_fits(self) -> None:
        messages = [
            {
                "id": "u1",
                "role": "user",
                "type": "message",
                "markdown": "Open the architecture guide",
                "created_at": "2026-04-16T00:00:00+00:00",
            },
            {
                "id": "a1",
                "role": "assistant",
                "type": "message",
                "markdown": "I opened the architecture guide.",
                "created_at": "2026-04-16T00:00:01+00:00",
            },
            {
                "id": "u2",
                "role": "user",
                "type": "message",
                "markdown": "Keep comparing it to the release checklist.",
                "created_at": "2026-04-16T00:00:02+00:00",
            },
        ]
        session_memory = SessionMemoryRecord(
            thread_id="web:test",
            updated_at="2026-04-16T00:00:01+00:00",
            last_message_id="a1",
            last_message_created_at="2026-04-16T00:00:01+00:00",
            covered_message_count=2,
            covered_tokens=320,
            summary_markdown="## Continuation Summary\nThe user is working on the architecture guide.",
            source="update",
        )

        bundle = compact_conversation(
            messages,
            llm=ExplodingSummaryLLM(),
            preserved_tail_count=1,
            session_memory=session_memory,
        )

        self.assertTrue(bundle.used_session_memory)
        self.assertEqual(bundle.summary_message["markdown"], session_memory.summary_markdown)
        self.assertEqual(bundle.summary_message["metadata"]["compaction"]["source"], "session_memory")
        self.assertEqual(
            bundle.summary_message["metadata"]["compaction"]["sessionMemory"]["lastMessageId"],
            "a1",
        )
        self.assertEqual([message["id"] for message in bundle.preserved_tail_messages], ["u2"])

    def test_compact_conversation_falls_back_when_session_memory_is_stale(self) -> None:
        messages = [
            {
                "id": "u1",
                "role": "user",
                "type": "message",
                "markdown": "Open the architecture guide",
                "created_at": "2026-04-16T00:00:00+00:00",
            },
            {
                "id": "a1",
                "role": "assistant",
                "type": "message",
                "markdown": "I opened the architecture guide.",
                "created_at": "2026-04-16T00:00:01+00:00",
            },
            {
                "id": "u2",
                "role": "user",
                "type": "message",
                "markdown": "Keep comparing it to the release checklist.",
                "created_at": "2026-04-16T00:00:02+00:00",
            },
        ]
        session_memory = SessionMemoryRecord(
            thread_id="web:test",
            updated_at="2026-04-16T00:00:01+00:00",
            last_message_id="a1",
            last_message_created_at="2026-04-16T00:00:01+00:00",
            covered_message_count=2,
            covered_tokens=320,
            summary_markdown="## Continuation Summary\nThe user is working on the architecture guide.",
            source="update",
        )

        bundle = compact_conversation(
            messages,
            llm=StructuredSummaryLLM(),
            preserved_tail_count=0,
            session_memory=session_memory,
        )

        self.assertFalse(bundle.used_session_memory)
        self.assertEqual(bundle.summary_message["metadata"]["compaction"]["source"], "fresh_summary")
        self.assertIn("## Continuation Summary", bundle.summary_message["markdown"])

    def test_compact_conversation_preserves_whole_tool_exchange_in_tail(self) -> None:
        messages = [
            HumanMessage(content="Find the architecture guide", id="u1"),
            AIMessage(
                content="",
                id="a_tool",
                tool_calls=[{"id": "call_old", "name": "read_knowledge_document", "args": {"document": "Guide"}}],
            ),
            ToolMessage(
                content='{"ok": true, "document": {"title": "Architecture Guide"}, "content": "short"}',
                tool_call_id="call_old",
                id="t_old",
            ),
            HumanMessage(content="What about the release checklist?", id="u2"),
        ]

        bundle = compact_conversation(
            messages,
            llm=StructuredSummaryLLM(),
            preserved_tail_count=2,
        )

        self.assertEqual(bundle.compacted_source_count, 1)
        self.assertEqual(
            [message["id"] for message in bundle.preserved_tail_messages],
            ["a_tool", "t_old", "u2"],
        )
        self.assertEqual(
            bundle.boundary_message["metadata"]["preservedTail"]["messageIds"],
            ["a_tool", "t_old", "u2"],
        )

    def test_compact_conversation_uses_session_memory_when_recent_tail_expands_to_tool_pair(self) -> None:
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
        session_memory = SessionMemoryRecord(
            thread_id="web:test",
            updated_at="2026-04-16T00:00:01+00:00",
            last_message_id="a1",
            last_message_created_at="2026-04-16T00:00:01+00:00",
            covered_message_count=2,
            covered_tokens=320,
            summary_markdown="## Continuation Summary\nThe user is working on the architecture guide.",
            source="update",
        )

        bundle = compact_conversation(
            messages,
            llm=ExplodingSummaryLLM(),
            preserved_tail_count=1,
            session_memory=session_memory,
        )

        self.assertTrue(bundle.used_session_memory)
        self.assertEqual(bundle.summary_message["markdown"], session_memory.summary_markdown)
        self.assertEqual(
            [message["id"] for message in bundle.preserved_tail_messages],
            ["a_tool", "t_read"],
        )

    def test_compact_conversation_replaces_only_active_slice_after_existing_boundary(self) -> None:
        messages = [
            {
                "id": "boundary-old",
                "role": "system",
                "type": TRANSCRIPT_TYPE_COMPACT_BOUNDARY,
                "markdown": "",
                "created_at": "2026-04-16T00:00:00+00:00",
                "metadata": {"trigger": "manual", "preTokens": 400},
            },
            {
                "id": "summary-old",
                "role": "assistant",
                "type": "message",
                "markdown": "Old continuation summary",
                "created_at": "2026-04-16T00:00:00+00:00",
            },
            {
                "id": "u1",
                "role": "user",
                "type": "message",
                "markdown": "Newer request",
                "created_at": "2026-04-16T00:00:02+00:00",
            },
            {
                "id": "a1",
                "role": "assistant",
                "type": "message",
                "markdown": "Newest reply",
                "created_at": "2026-04-16T00:00:03+00:00",
            },
        ]

        bundle = compact_conversation(
            messages,
            llm=StructuredSummaryLLM(),
            preserved_tail_count=1,
        )

        self.assertEqual(bundle.active_slice_start, 1)
        self.assertEqual([message["id"] for message in bundle.prefix_messages], ["boundary-old"])
        self.assertEqual(bundle.compacted_messages[0]["id"], "boundary-old")
        self.assertEqual(bundle.compacted_messages[-1]["id"], "a1")
        self.assertEqual(bundle.compacted_messages[-2]["type"], "message")
        self.assertEqual(bundle.compacted_messages[-3]["type"], TRANSCRIPT_TYPE_COMPACT_BOUNDARY)

    def test_manual_compaction_persists_and_conversation_continues_after_reload(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        storage_path = root / "web_conversations.json"
        settings = make_settings(root / "runtime")

        store = WebConversationStore(storage_path)
        conversation = store.create_conversation(title="Compact me")
        conversation_id = conversation["conversation_id"]
        store.append_message(conversation_id, role="user", markdown="Old request")
        store.append_message(conversation_id, role="assistant", markdown="Old answer")
        store.append_message(conversation_id, role="user", markdown="Recent request")
        store.append_transcript_message(
            conversation_id,
            role="assistant",
            message_type="message",
            markdown="Recent answer",
            metadata={
                RUNTIME_REHYDRATION_METADATA_KEY: {
                    "context_paths": ["knowledge/Docs/00_Shared/ArchitectureOverview.md"],
                    "pending_action": {
                        "id": "pending-doc",
                        "status": "awaiting_confirmation",
                        "summary": "Review the opened knowledge document.",
                    },
                    "tool_result": {
                        "tool_name": "read_knowledge_document",
                        "tool_id": "knowledge.read_document",
                        "status": "ok",
                        "payload": {
                            "ok": True,
                            "document": {
                                "name": "ArchitectureOverview",
                                "title": "Architecture Overview",
                                "path": "knowledge/Docs/00_Shared/ArchitectureOverview.md",
                            },
                            "content": "Architecture excerpt",
                        },
                    },
                }
            },
        )

        full_conversation = store.get_full_conversation(conversation_id)
        bundle = compact_conversation(
            full_conversation["messages"],
            llm=StructuredSummaryLLM(),
            preserved_tail_count=1,
        )
        store.replace_transcript(
            conversation_id,
            messages=bundle.compacted_messages,
        )

        reloaded_store = WebConversationStore(storage_path)
        compacted = reloaded_store.get_full_conversation(conversation_id)
        self.assertEqual(
            [message["type"] for message in compacted["messages"]],
            [TRANSCRIPT_TYPE_COMPACT_BOUNDARY, "message", "message"],
        )

        graph = RecordingGraph(reply_text="Post-compaction reply.")
        server = WebServer(
            agent_graph=graph,
            settings=settings,
            conversation_store=reloaded_store,
            checkpoint_store=MissingCheckpointStore(),
        )
        client = TestClient(server.app)

        response = client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"message": "Continue from here", "display_name": "Tester", "email": "tester@example.com"},
        )

        self.assertEqual(response.status_code, 200)
        resumed_messages = graph.last_state["messages"]
        self.assertEqual(len(resumed_messages), 4)
        self.assertTrue(isinstance(resumed_messages[0], SystemMessage))
        self.assertTrue(isinstance(resumed_messages[1], AIMessage))
        self.assertTrue(isinstance(resumed_messages[2], AIMessage))
        self.assertTrue(isinstance(resumed_messages[3], HumanMessage))
        self.assertIn("## Continuation Summary", resumed_messages[1].content)
        self.assertEqual(resumed_messages[2].content, "Recent answer")
        self.assertEqual(resumed_messages[3].content, "Continue from here")
        rebuilt_contents = [getattr(message, "content", "") for message in resumed_messages]
        self.assertTrue(all("Old request" not in content for content in rebuilt_contents))
        self.assertTrue(all("Old answer" not in content for content in rebuilt_contents))
        self.assertEqual(
            graph.last_state["context_paths"],
            ["knowledge/Docs/00_Shared/ArchitectureOverview.md"],
        )
        self.assertEqual(
            graph.last_state["recent_file_reads"],
            ["knowledge/Docs/00_Shared/ArchitectureOverview.md"],
        )
        self.assertEqual(graph.last_state["pending_action"]["id"], "pending-doc")
        self.assertEqual(
            graph.last_state["tool_result"]["payload"]["document"]["path"],
            "knowledge/Docs/00_Shared/ArchitectureOverview.md",
        )


if __name__ == "__main__":
    unittest.main()
