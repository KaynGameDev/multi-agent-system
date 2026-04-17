from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.project_task.agent import build_project_task_prompt
from app.graph import build_default_agent_registrations
from app.memory.agent_scope import resolve_agent_memory_context
from tests.common import make_settings


def invoke_memory_tool(tool_map: dict[str, object], *, state: dict, name: str, args: dict | None = None) -> dict:
    tool = tool_map[name]
    tool_func = getattr(tool, "func", None)
    if not callable(tool_func):
        raise AssertionError(f"Tool `{name}` is missing a callable func.")
    return tool_func(**dict(args or {}), state=state)


class AgentMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.runtime_root = Path(self.tempdir.name) / "runtime"
        self.settings = make_settings(self.runtime_root)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_resolve_agent_memory_context_supports_local_user_and_project_scopes(self) -> None:
        state = {
            "user_id": "User-123",
            "target_market_slug": "indonesia-main",
            "target_game_slug": "f4buyu",
            "target_feature_slug": "memory-subsystem",
            "thread_id": "web:thread-1",
        }

        user_context = resolve_agent_memory_context(
            self.settings,
            agent_name="project_task_agent",
            memory_scope="user",
            state=state,
        )
        project_context = resolve_agent_memory_context(
            self.settings,
            agent_name="project_task_agent",
            memory_scope="project",
            state=state,
        )
        local_context = resolve_agent_memory_context(
            self.settings,
            agent_name="project_task_agent",
            memory_scope="local",
            state=state,
        )

        self.assertEqual(user_context.scope_key, "user-123")
        self.assertTrue(str(user_context.root_dir).endswith("/agents/project_task_agent/users/user-123"))
        self.assertEqual(project_context.scope_key, "indonesia-main/f4buyu/memory-subsystem")
        self.assertTrue(
            str(project_context.root_dir).endswith(
                "/agents/project_task_agent/projects/indonesia-main/f4buyu/memory-subsystem"
            )
        )
        self.assertEqual(local_context.scope_key, "local")
        self.assertTrue(str(local_context.root_dir).endswith("/agents/project_task_agent/local"))

    def test_project_task_prompt_includes_scoped_memory_instructions(self) -> None:
        state = {
            "interface_name": "web",
            "user_id": "user-123",
            "user_sheet_name": "Tester",
            "user_google_name": "",
            "user_job_title": "",
            "user_mapped_slack_name": "Tester",
            "thread_id": "web:thread-1",
            "messages": [],
        }
        expected_context = resolve_agent_memory_context(
            self.settings,
            agent_name="project_task_agent",
            memory_scope="user",
            state=state,
        )

        prompt = build_project_task_prompt(
            state,
            settings=self.settings,
            agent_name="project_task_agent",
            tool_ids=(),
            memory_scope="user",
        )

        self.assertIn("# Memory Scope", prompt)
        self.assertIn(str(expected_context.root_dir), prompt)
        self.assertIn("Use only the memory tools", prompt)

    def test_default_registrations_attach_user_scoped_memory_to_project_task_agent(self) -> None:
        registrations = build_default_agent_registrations(settings=self.settings)
        project_task_registration = next(
            registration for registration in registrations if registration.name == "project_task_agent"
        )

        self.assertEqual(project_task_registration.memory_scope, "user")
        self.assertIn("memory.list", project_task_registration.tool_ids)
        self.assertIn("memory.write", project_task_registration.tool_ids)
        self.assertTrue(any(tool.name == "write_agent_memory" for tool in project_task_registration.tools))

    def test_project_task_agent_memory_tools_are_user_scoped_and_isolated(self) -> None:
        registrations = build_default_agent_registrations(settings=self.settings)
        project_task_registration = next(
            registration for registration in registrations if registration.name == "project_task_agent"
        )
        memory_tools = tuple(
            tool
            for tool in project_task_registration.tools
            if tool.name in {"list_agent_memories", "read_agent_memory", "write_agent_memory", "delete_agent_memory"}
        )
        tool_map = {tool.name: tool for tool in memory_tools}
        user_state = {
            "user_id": "User-123",
            "thread_id": "web:thread-1",
            "channel_id": "web",
        }

        write_payload = invoke_memory_tool(
            tool_map,
            state=user_state,
            name="write_agent_memory",
            args={
                "memory_id": "preferences/task-view",
                "name": "Task View Preference",
                "description": "Preferred task response layout.",
                "memory_type": "user",
                "content": "Prefer grouped due-date summaries.",
            },
        )
        read_payload = invoke_memory_tool(
            tool_map,
            state=user_state,
            name="read_agent_memory",
            args={"memory_id": "preferences/task-view"},
        )
        list_payload = invoke_memory_tool(
            tool_map,
            state=user_state,
            name="list_agent_memories",
        )
        other_user_list_payload = invoke_memory_tool(
            tool_map,
            state={**user_state, "user_id": "other-user"},
            name="list_agent_memories",
        )
        delete_payload = invoke_memory_tool(
            tool_map,
            state=user_state,
            name="delete_agent_memory",
            args={"memory_id": "preferences/task-view"},
        )

        self.assertTrue(write_payload["ok"])
        self.assertTrue(write_payload["path_scoped"])
        self.assertIn("/agents/project_task_agent/users/user-123", write_payload["memory_root"])
        self.assertEqual(
            write_payload["memory"]["relative_path"],
            "topics/preferences/task-view.md",
        )
        self.assertTrue(read_payload["ok"])
        self.assertEqual(read_payload["memory"]["content_markdown"], "Prefer grouped due-date summaries.")
        self.assertTrue(list_payload["ok"])
        self.assertEqual(list_payload["count"], 1)
        self.assertEqual(list_payload["entries"][0]["memory_id"], "preferences/task-view")
        self.assertTrue(other_user_list_payload["ok"])
        self.assertEqual(other_user_list_payload["count"], 0)
        self.assertTrue(delete_payload["ok"])
        self.assertTrue(delete_payload["deleted"])

    def test_project_task_agent_memory_tools_reject_path_escape_memory_ids(self) -> None:
        registrations = build_default_agent_registrations(settings=self.settings)
        project_task_registration = next(
            registration for registration in registrations if registration.name == "project_task_agent"
        )
        tool_map = {
            tool.name: tool
            for tool in project_task_registration.tools
            if tool.name == "write_agent_memory"
        }

        payload = invoke_memory_tool(
            tool_map,
            state={
                "user_id": "user-123",
                "thread_id": "web:thread-1",
            },
            name="write_agent_memory",
            args={
                "memory_id": "../escape",
                "name": "Escape Attempt",
                "description": "Should fail.",
                "memory_type": "user",
                "content": "bad",
            },
        )

        self.assertFalse(payload["ok"])
        self.assertIn("Invalid long-term memory id", payload["error"])


if __name__ == "__main__":
    unittest.main()
