from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.model_request import (
    ModelRequestReductionConfig,
    ModelRequestReducerHooks,
    build_model_request_messages,
    get_messages_after_compact_boundary,
)
from interfaces.web.conversations import TRANSCRIPT_TYPE_COMPACT_BOUNDARY


class ModelRequestTests(unittest.TestCase):
    def test_get_messages_after_compact_boundary_returns_only_tail(self) -> None:
        messages = [
            HumanMessage(content="old user"),
            AIMessage(content="old assistant"),
            SystemMessage(
                content="",
                additional_kwargs={"transcript_type": TRANSCRIPT_TYPE_COMPACT_BOUNDARY},
            ),
            HumanMessage(content="new user"),
            AIMessage(content="new assistant"),
        ]

        tail = get_messages_after_compact_boundary(messages)

        self.assertEqual(len(tail), 2)
        self.assertEqual([message.content for message in tail], ["new user", "new assistant"])

    def test_build_model_request_messages_uses_projected_tail_and_system_prompt(self) -> None:
        messages = [
            HumanMessage(content="discard me"),
            SystemMessage(
                content="",
                additional_kwargs={"transcript_type": TRANSCRIPT_TYPE_COMPACT_BOUNDARY},
            ),
            HumanMessage(content="keep me"),
        ]

        built = build_model_request_messages(
            system_prompt="You are helpful.",
            transcript_messages=messages,
        )

        self.assertEqual(len(built), 2)
        self.assertTrue(isinstance(built[0], SystemMessage))
        self.assertEqual(built[0].content, "You are helpful.")
        self.assertEqual(built[1].content, "keep me")

    def test_reducer_hooks_run_in_expected_order(self) -> None:
        calls: list[tuple[str, list[str]]] = []

        def make_reducer(label: str):
            def reducer(messages):
                calls.append((label, [getattr(message, "content", "") for message in messages]))
                return list(messages)

            return reducer

        built = build_model_request_messages(
            system_prompt="Reducer test.",
            transcript_messages=[
                HumanMessage(content="before"),
                SystemMessage(
                    content="",
                    additional_kwargs={"transcript_type": TRANSCRIPT_TYPE_COMPACT_BOUNDARY},
                ),
                HumanMessage(content="after"),
            ],
            reducer_hooks=ModelRequestReducerHooks(
                snip=make_reducer("snip"),
                microcompact=make_reducer("microcompact"),
                collapse=make_reducer("collapse"),
                auto_compact=make_reducer("auto_compact"),
            ),
        )

        self.assertEqual([label for label, _ in calls], ["snip", "microcompact", "collapse", "auto_compact"])
        self.assertTrue(all(contents == ["after"] for _, contents in calls))
        self.assertEqual([message.content for message in built], ["Reducer test.", "after"])

    def test_microcompact_compresses_old_large_tool_results(self) -> None:
        built = build_model_request_messages(
            transcript_messages=[
                HumanMessage(content="Find the architecture guide"),
                AIMessage(
                    content="",
                    additional_kwargs={"created_at": "2026-04-16T00:00:01+00:00"},
                    tool_calls=[{"id": "call_old", "name": "read_knowledge_document", "args": {"document": "Guide"}}],
                ),
                ToolMessage(
                    content='{"ok": true, "document": {"title": "Architecture Guide", "path": "knowledge/Guide.md"}, "content": "%s"}' % ("A" * 1200),
                    tool_call_id="call_old",
                    additional_kwargs={"created_at": "2026-04-16T00:00:02+00:00"},
                ),
                HumanMessage(content="What about the release checklist?"),
                AIMessage(
                    content="",
                    additional_kwargs={"created_at": "2026-04-16T00:00:03+00:00"},
                    tool_calls=[{"id": "call_recent", "name": "read_knowledge_document", "args": {"document": "Checklist"}}],
                ),
                ToolMessage(
                    content='{"ok": true, "document": {"title": "Release Checklist"}, "content": "short"}',
                    tool_call_id="call_recent",
                    additional_kwargs={"created_at": "2026-04-16T00:00:04+00:00"},
                ),
            ],
        )

        self.assertEqual(len(built), 5)
        self.assertTrue(isinstance(built[1], SystemMessage))
        self.assertIn("Earlier tool result `read_knowledge_document` was compressed", built[1].content)
        self.assertIn("Architecture Guide", built[1].content)
        self.assertEqual(built[2].content, "What about the release checklist?")
        self.assertTrue(isinstance(built[3], AIMessage))
        self.assertTrue(isinstance(built[4], ToolMessage))
        self.assertEqual(built[4].tool_call_id, "call_recent")

    def test_cold_cache_clearing_drops_old_reduced_tool_context_when_enabled(self) -> None:
        built = build_model_request_messages(
            transcript_messages=[
                HumanMessage(content="Start"),
                SystemMessage(
                    content="Earlier tool result `read_knowledge_document` was compressed.",
                    additional_kwargs={
                        "created_at": "2026-04-16T00:00:00+00:00",
                        "cheap_context_reduction": {"kind": "microcompact_tool_result"},
                    },
                ),
                HumanMessage(content="Continue"),
                AIMessage(
                    content="Reply",
                    additional_kwargs={"created_at": "2026-04-16T00:10:00+00:00"},
                ),
                HumanMessage(content="Latest"),
            ],
            reduction_config=ModelRequestReductionConfig(
                cold_cache_clear_after_seconds=60,
                cold_cache_min_following_messages=2,
            ),
        )

        self.assertEqual([message.content for message in built], ["Start", "Continue", "Reply", "Latest"])

    def test_old_tool_invocation_stub_is_dropped_after_microcompact(self) -> None:
        built = build_model_request_messages(
            transcript_messages=[
                HumanMessage(content="Open the guide"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "call_old", "name": "read_knowledge_document", "args": {"document": "Guide"}}],
                ),
                ToolMessage(
                    content='{"ok": true, "document": {"title": "Guide"}, "content": "%s"}' % ("B" * 1000),
                    tool_call_id="call_old",
                ),
                HumanMessage(content="What next?"),
            ],
            reduction_config=ModelRequestReductionConfig(
                microcompact_tool_result_threshold_chars=100,
                preserve_recent_tool_results=0,
            ),
        )

        self.assertEqual(len(built), 3)
        self.assertEqual([type(message).__name__ for message in built], ["HumanMessage", "SystemMessage", "HumanMessage"])


if __name__ == "__main__":
    unittest.main()
