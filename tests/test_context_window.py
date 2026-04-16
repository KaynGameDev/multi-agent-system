from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.context_window import (
    ContextWindowThresholdOverrides,
    USAGE_BASELINE_STAGE_BEFORE_MESSAGE,
    evaluate_context_window,
    evaluate_context_window_for_transcript,
    estimate_transcript_message_tokens,
    format_context_window_status,
    get_context_window_for_model,
    should_auto_compact,
    token_count_with_estimation,
)
from app.model_request import ModelRequestReductionConfig, project_transcript_messages
from interfaces.web.conversations import TRANSCRIPT_TYPE_COMPACT_BOUNDARY


class ContextWindowTests(unittest.TestCase):
    def test_get_context_window_for_known_models(self) -> None:
        self.assertEqual(get_context_window_for_model("gpt-5-mini"), 400_000)
        self.assertEqual(get_context_window_for_model("gemini-3-flash-preview"), 1_048_576)
        self.assertEqual(get_context_window_for_model("MiniMax-M2.7-highspeed"), 204_800)

    def test_get_context_window_for_unknown_model_uses_fallback(self) -> None:
        self.assertEqual(get_context_window_for_model("unknown-model"), 128_000)

    def test_token_count_uses_usage_baseline_and_estimates_unsent_tail(self) -> None:
        messages = [
            {
                "id": "user-1",
                "role": "user",
                "type": "message",
                "markdown": "hello",
                "created_at": "2026-04-16T00:00:00+00:00",
            },
            {
                "id": "assistant-1",
                "role": "assistant",
                "type": "message",
                "markdown": "Rendered after tool execution.",
                "created_at": "2026-04-16T00:00:01+00:00",
                "usage": {"input_tokens": 300, "output_tokens": 40, "total_tokens": 340},
                "metadata": {"usage_baseline_stage": USAGE_BASELINE_STAGE_BEFORE_MESSAGE},
            },
            {
                "id": "user-2",
                "role": "user",
                "type": "message",
                "markdown": "follow up question",
                "created_at": "2026-04-16T00:00:02+00:00",
            },
        ]

        estimate = token_count_with_estimation(messages)
        expected_tail = (
            estimate_transcript_message_tokens(messages[1])
            + estimate_transcript_message_tokens(messages[2])
        )

        self.assertTrue(estimate.used_baseline_usage)
        self.assertEqual(estimate.baseline_message_id, "assistant-1")
        self.assertEqual(estimate.baseline_tokens, 300)
        self.assertEqual(estimate.estimated_tail_tokens, expected_tail)
        self.assertEqual(estimate.estimated_total_tokens, 300 + expected_tail)

    def test_token_count_without_usage_estimates_all_messages(self) -> None:
        messages = [
            {
                "id": "user-1",
                "role": "user",
                "type": "message",
                "markdown": "hello",
                "created_at": "2026-04-16T00:00:00+00:00",
            },
            {
                "id": "assistant-1",
                "role": "assistant",
                "type": "message",
                "markdown": "plain reply",
                "created_at": "2026-04-16T00:00:01+00:00",
            },
        ]

        estimate = token_count_with_estimation(messages)
        expected_total = sum(estimate_transcript_message_tokens(message) for message in messages)

        self.assertFalse(estimate.used_baseline_usage)
        self.assertIsNone(estimate.baseline_message_id)
        self.assertEqual(estimate.baseline_tokens, 0)
        self.assertEqual(estimate.estimated_tail_tokens, expected_total)
        self.assertEqual(estimate.estimated_total_tokens, expected_total)

    def test_context_window_snapshot_reports_remaining_percentage_and_threshold_level(self) -> None:
        messages = [
            {
                "id": "assistant-1",
                "role": "assistant",
                "type": "message",
                "markdown": "This baseline is already heavy.",
                "created_at": "2026-04-16T00:00:01+00:00",
                "usage": {"input_tokens": 875, "output_tokens": 25, "total_tokens": 900},
            }
        ]

        snapshot = evaluate_context_window(
            messages,
            model="gpt-5-mini",
            threshold_overrides=ContextWindowThresholdOverrides(
                effective_window=1_000,
                warning_threshold=700,
                auto_compact_threshold=850,
                hard_block_threshold=950,
            ),
        )
        status_line = format_context_window_status(snapshot)

        self.assertEqual(snapshot.used_tokens, 900)
        self.assertEqual(snapshot.remaining_tokens, 100)
        self.assertAlmostEqual(snapshot.remaining_percentage, 10.0)
        self.assertEqual(snapshot.decision.level, "auto_compact")
        self.assertIn("900/1000", status_line)
        self.assertIn("% remaining", status_line)

    def test_context_window_uses_error_state_when_auto_compact_is_unavailable(self) -> None:
        messages = [
            {
                "id": "assistant-1",
                "role": "assistant",
                "type": "message",
                "markdown": "High-token baseline",
                "created_at": "2026-04-16T00:00:01+00:00",
                "usage": {"input_tokens": 760, "output_tokens": 20, "total_tokens": 780},
            }
        ]

        snapshot = evaluate_context_window(
            messages,
            model="gpt-5-mini",
            threshold_overrides=ContextWindowThresholdOverrides(
                effective_window=1_000,
                warning_threshold=600,
                auto_compact_threshold=700,
                hard_block_threshold=900,
            ),
            auto_compact_enabled=False,
        )

        self.assertEqual(snapshot.decision.level, "error")
        self.assertTrue(snapshot.decision.should_block)
        self.assertFalse(snapshot.decision.auto_compact_available)

    def test_should_auto_compact_respects_failure_breaker(self) -> None:
        messages = [
            {
                "id": "assistant-1",
                "role": "assistant",
                "type": "message",
                "markdown": "High-token baseline",
                "created_at": "2026-04-16T00:00:01+00:00",
                "usage": {"input_tokens": 760, "output_tokens": 20, "total_tokens": 780},
            }
        ]
        thresholds = ContextWindowThresholdOverrides(
            effective_window=1_000,
            warning_threshold=600,
            auto_compact_threshold=700,
            hard_block_threshold=900,
        )

        self.assertTrue(
            should_auto_compact(
                messages,
                model="gpt-5-mini",
                threshold_overrides=thresholds,
            )
        )
        self.assertFalse(
            should_auto_compact(
                messages,
                model="gpt-5-mini",
                threshold_overrides=thresholds,
                auto_compact_failure_count=1,
                auto_compact_failure_limit=1,
            )
        )

    def test_evaluate_context_window_for_transcript_ignores_archived_pre_boundary_usage(self) -> None:
        messages = [
            {
                "id": "assistant-old",
                "role": "assistant",
                "type": "message",
                "markdown": "Old heavy answer",
                "created_at": "2026-04-16T00:00:01+00:00",
                "usage": {"input_tokens": 760, "output_tokens": 20, "total_tokens": 780},
            },
            {
                "id": "boundary-1",
                "role": "system",
                "type": TRANSCRIPT_TYPE_COMPACT_BOUNDARY,
                "markdown": "",
                "created_at": "2026-04-16T00:00:02+00:00",
                "metadata": {"trigger": "manual", "preTokens": 780},
            },
            {
                "id": "assistant-summary",
                "role": "assistant",
                "type": "message",
                "markdown": "## Continuation Summary\nKeep working on the release checklist.",
                "created_at": "2026-04-16T00:00:03+00:00",
            },
            {
                "id": "user-new",
                "role": "user",
                "type": "message",
                "markdown": "What should we do next?",
                "created_at": "2026-04-16T00:00:04+00:00",
            },
        ]

        projected_messages = project_transcript_messages(messages)
        snapshot = evaluate_context_window_for_transcript(
            messages,
            model="gpt-5-mini",
            threshold_overrides=ContextWindowThresholdOverrides(
                effective_window=1_000,
                warning_threshold=600,
                auto_compact_threshold=700,
                hard_block_threshold=950,
            ),
            auto_compact_enabled=False,
        )

        self.assertEqual(snapshot.used_tokens, token_count_with_estimation(projected_messages).estimated_total_tokens)
        self.assertFalse(snapshot.estimate.used_baseline_usage)
        self.assertEqual(snapshot.decision.level, "ok")

    def test_evaluate_context_window_for_transcript_matches_reduced_projected_slice(self) -> None:
        messages = [
            HumanMessage(content="Find the architecture guide"),
            AIMessage(
                content="",
                additional_kwargs={"created_at": "2026-04-16T00:00:01+00:00"},
                tool_calls=[{"id": "call_old", "name": "read_knowledge_document", "args": {"document": "Guide"}}],
            ),
            ToolMessage(
                content='{"ok": true, "document": {"title": "Architecture Guide"}, "content": "%s"}' % ("A" * 1200),
                tool_call_id="call_old",
                additional_kwargs={"created_at": "2026-04-16T00:00:02+00:00"},
            ),
            HumanMessage(content="What next?"),
        ]
        reduction_config = ModelRequestReductionConfig(
            microcompact_tool_result_threshold_chars=100,
            preserve_recent_tool_results=0,
        )

        raw_total = token_count_with_estimation(messages).estimated_total_tokens
        projected_messages = project_transcript_messages(
            messages,
            reduction_config=reduction_config,
        )
        snapshot = evaluate_context_window_for_transcript(
            messages,
            model="gpt-5-mini",
            reduction_config=reduction_config,
        )

        self.assertEqual(snapshot.used_tokens, token_count_with_estimation(projected_messages).estimated_total_tokens)
        self.assertLess(snapshot.used_tokens, raw_total)


if __name__ == "__main__":
    unittest.main()
