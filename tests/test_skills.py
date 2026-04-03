from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agents.general_chat.agent import build_general_chat_prompt
from app.skills import SkillRegistry
from tests.common import build_registration, write_skill


class SkillRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_agent_local_skill_overrides_project_shared_skill(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
            build_registration("alpha_agent", namespace="alpha"),
        )
        write_skill(
            self.root,
            ".jade/skills/shared-doc",
            frontmatter={
                "name": "Shared Doc",
                "description": "Project shared base skill.",
                "available_to_agents": ["alpha_agent"],
            },
            body="# Shared Doc\n\nProject shared instructions.",
        )
        agent_local_path = write_skill(
            self.root,
            "agents/alpha/Skills/shared-doc",
            frontmatter={
                "name": "Shared Doc Override",
                "description": "Agent-local override.",
            },
            body="# Shared Doc Override\n\nAgent-local instructions.",
        )

        registry = SkillRegistry(registrations, project_root=self.root)
        resolution = registry.resolve_skill("shared-doc")

        self.assertIsNotNone(resolution.effective_definition)
        self.assertEqual(resolution.effective_definition.source_path.resolve(), agent_local_path.resolve())
        self.assertEqual(resolution.effective_definition.scope, "agent_local")

    def test_same_scope_duplicates_are_conflicts(self) -> None:
        registrations = (build_registration("alpha_agent", namespace="alpha"),)
        write_skill(
            self.root,
            ".jade/skills/duplicate-one",
            frontmatter={
                "skill_id": "duplicate-skill",
                "name": "Duplicate One",
                "description": "First duplicate.",
                "available_to_agents": ["alpha_agent"],
            },
            body="# Duplicate One\n\nFirst duplicate.",
        )
        write_skill(
            self.root,
            ".jade/skills/duplicate-two",
            frontmatter={
                "skill_id": "duplicate-skill",
                "name": "Duplicate Two",
                "description": "Second duplicate.",
                "available_to_agents": ["alpha_agent"],
            },
            body="# Duplicate Two\n\nSecond duplicate.",
        )

        registry = SkillRegistry(registrations, project_root=self.root)
        resolution = registry.resolve_skill("duplicate-skill")

        self.assertIsNone(resolution.effective_definition)
        self.assertTrue(any(item["kind"] == "conflict" for item in resolution.diagnostics))

    def test_path_scoped_skill_wins_only_when_context_matches(self) -> None:
        registrations = (build_registration("alpha_agent", namespace="alpha"),)
        default_path = write_skill(
            self.root,
            ".jade/skills/audit-default",
            frontmatter={
                "skill_id": "audit-skill",
                "name": "Audit Default",
                "description": "Default audit behavior.",
                "available_to_agents": ["alpha_agent"],
            },
            body="# Audit Default\n\nDefault audit behavior.",
        )
        specific_path = write_skill(
            self.root,
            ".jade/skills/audit-specific",
            frontmatter={
                "skill_id": "audit-skill",
                "name": "Audit Specific",
                "description": "Specific audit behavior.",
                "available_to_agents": ["alpha_agent"],
                "path_patterns": ["src/*"],
            },
            body="# Audit Specific\n\nSpecific audit behavior.",
        )

        registry = SkillRegistry(registrations, project_root=self.root)
        specific_resolution = registry.resolve_skill("audit-skill", context_paths=["src/main.py"])
        default_resolution = registry.resolve_skill("audit-skill", context_paths=["docs/readme.md"])

        self.assertEqual(specific_resolution.effective_definition.source_path.resolve(), specific_path.resolve())
        self.assertEqual(default_resolution.effective_definition.source_path.resolve(), default_path.resolve())

    def test_legacy_skill_normalizes_from_existing_markdown(self) -> None:
        registrations = (
            build_registration("knowledge_base_builder_agent", namespace="knowledge_base_builder"),
        )
        write_skill(
            self.root,
            "agents/knowledge_base_builder/Skills/Elicit-Knowledge",
            body="""
# Skill: Elicit Knowledge

## 目的

用于主持一轮通用的知识抽取过程，把团队成员脑中的隐性知识逐步引导出来，并整理成结构化内容。
            """,
        )

        registry = SkillRegistry(registrations, project_root=self.root)
        resolution = registry.resolve_skill("elicit-knowledge")

        self.assertIsNotNone(resolution.effective_definition)
        self.assertEqual(resolution.effective_definition.skill_id, "elicit-knowledge")
        self.assertIn("知识抽取过程", resolution.effective_definition.description)
        self.assertEqual(
            resolution.effective_definition.available_to_agents,
            ("knowledge_base_builder_agent",),
        )

    def test_prompt_builder_injects_selected_skill_body(self) -> None:
        registrations = (
            build_registration("general_chat_agent", namespace="general_chat", is_general_assistant=True),
        )
        write_skill(
            self.root,
            ".jade/skills/greeting-enhancer",
            frontmatter={
                "name": "Greeting Enhancer",
                "description": "Makes greetings warmer.",
                "available_to_agents": ["general_chat_agent"],
            },
            body="# Greeting Enhancer\n\nAlways greet the user warmly and concisely.",
        )
        registry = SkillRegistry(registrations, project_root=self.root)

        prompt = build_general_chat_prompt(
            {
                "interface_name": "web",
                "resolved_skill_ids": ["greeting-enhancer"],
                "context_paths": [],
            },
            skill_registry=registry,
            agent_name="general_chat_agent",
        )

        self.assertIn("Greeting Enhancer", prompt)
        self.assertIn("Always greet the user warmly and concisely.", prompt)


if __name__ == "__main__":
    unittest.main()
