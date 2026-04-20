from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from app.graph import build_graph
from app.knowledge_status import build_knowledge_base_status_prompt_context
from app.skills import SkillRegistry
from interfaces.web.server import WebServer
from tests.common import build_registration, make_settings


class StaticBuilderNode:
    def __call__(self, _state):
        return {"messages": [AIMessage(content="builder reply")]}


class RecordingGraph:
    def __init__(self) -> None:
        self.last_state = None

    def invoke(self, initial_state, config=None):
        del config
        self.last_state = dict(initial_state)
        return {
            **initial_state,
            "route": "knowledge_base_builder_agent",
            "route_reason": "Single-agent runtime: all fresh user turns route to `knowledge_base_builder_agent`.",
            "skill_resolution_diagnostics": [],
            "agent_selection_diagnostics": [],
            "selection_warnings": [],
            "messages": [AIMessage(content="builder reply")],
        }


class SingleAgentRuntimeTests(unittest.TestCase):
    def test_graph_routes_fresh_turns_to_builder_in_single_agent_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            registrations = (
                build_registration(
                    "knowledge_base_builder_agent",
                    namespace="knowledge_base_builder",
                    is_general_assistant=True,
                    build_node=lambda _llm=None, skill_registry=None, pending_action_router=None: StaticBuilderNode(),
                ),
            )
            registry = SkillRegistry(registrations, project_root=root)
            graph = build_graph(
                object(),
                agent_registrations=registrations,
                default_route="knowledge_base_builder_agent",
                skill_registry=registry,
            )

            final_state = graph.invoke(
                {
                    "messages": [HumanMessage(content="帮我推进知识库")],
                    "requested_skill_ids": [],
                    "context_paths": [],
                }
            )

            self.assertEqual(final_state["route"], "knowledge_base_builder_agent")
            self.assertEqual(final_state["requested_agent"], "knowledge_base_builder_agent")
            self.assertIn("Single-agent runtime", final_state["route_reason"])

    def test_runtime_status_prompt_reports_missing_canonical_layers(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            knowledge_root = root / "knowledge"
            docs_root = knowledge_root / "Docs"
            (docs_root / "00_Shared" / "Standards").mkdir(parents=True, exist_ok=True)
            (docs_root / "30_Review").mkdir(parents=True, exist_ok=True)
            (docs_root / "50_Templates").mkdir(parents=True, exist_ok=True)

            (docs_root / "00_Shared" / "Standards" / "KB_V1_Status.md").write_text(
                "# KB V1 状态\n\n## 当前阶段\n- 仍在内容迁移早期。\n\n## 阻塞项\n- 缺少 GameLine 文档。\n\n## 下一步\n- 先补第一批 canonical 文档。\n",
                encoding="utf-8",
            )
            (docs_root / "30_Review" / "Open_Questions.md").write_text("# Open Questions\n", encoding="utf-8")
            (docs_root / "30_Review" / "Decision_Backlog.md").write_text("# Decision Backlog\n", encoding="utf-8")
            (docs_root / "30_Review" / "Migration_Checker.md").write_text("# Migration Checker\n", encoding="utf-8")
            (docs_root / "50_Templates" / "TEMPLATE_FEATURE_SPEC.md").write_text("# Template\n", encoding="utf-8")

            settings = replace(make_settings(root / "runtime"), knowledge_base_dir=str(knowledge_root))
            prompt = build_knowledge_base_status_prompt_context(settings=settings)

            self.assertIn("运行时知识库状态扫描", prompt)
            self.assertIn("`10_GameLines/` 还没有已填充的 canonical 文档", prompt)
            self.assertIn("`20_Deployments/` 还没有已填充的 canonical 文档", prompt)
            self.assertIn("Open_Questions.md", prompt)

    def test_web_server_defaults_new_turns_to_builder_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            settings = make_settings(root / "runtime")
            graph = RecordingGraph()
            server = WebServer(agent_graph=graph, settings=settings)
            client = TestClient(server.app)

            conversation = client.post("/api/conversations", json={"title": "KBB"}).json()
            response = client.post(
                f"/api/conversations/{conversation['conversation_id']}/messages",
                json={"message": "继续推进知识库", "display_name": "Tester", "email": "tester@example.com"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(graph.last_state["requested_agent"], "knowledge_base_builder_agent")
            transcript = client.get(f"/api/conversations/{conversation['conversation_id']}").json()
            self.assertNotIn("mode", transcript)


if __name__ == "__main__":
    unittest.main()
