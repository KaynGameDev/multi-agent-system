from __future__ import annotations

from pathlib import Path
from typing import Any

from app.agent_registry import AgentRegistration
from app.config import Settings


def build_registration(
    name: str,
    *,
    namespace: str = "",
    selection_order: int = 100,
    is_general_assistant: bool = False,
    matcher=None,
    build_node=None,
    tools=(),
    tool_ids=(),
) -> AgentRegistration:
    node_builder = build_node or (lambda _llm=None, skill_registry=None: None)
    return AgentRegistration(
        name=name,
        description=f"Registration for {name}",
        build_node=node_builder,
        tools=tuple(tools),
        tool_ids=tuple(tool_ids),
        selection_order=selection_order,
        is_general_assistant=is_general_assistant,
        skill_namespace=namespace,
        matcher=matcher,
    )


def make_settings(runtime_dir: Path) -> Settings:
    return Settings(
        slack_enabled=False,
        slack_bot_token="",
        slack_app_token="",
        web_enabled=True,
        web_host="127.0.0.1",
        web_port=8000,
        google_api_key="test-google-api-key",
        gemini_model="gemini-test",
        gemini_temperature=0.2,
        google_application_credentials="/tmp/credentials.json",
        jade_project_sheet_id="sheet-id",
        project_sheet_range="Tasks!A1:Z",
        project_sheet_cache_ttl_seconds=30,
        slack_thinking_reaction="eyes",
        project_lookup_keywords=("task", "deadline", "assignee"),
        knowledge_base_dir="knowledge",
        knowledge_file_types=(".md",),
        jade_project_skills_dir=".jade/skills",
        knowledge_google_sheets_catalog_path="knowledge/AI/Rules/google_sheets_catalog.json",
        knowledge_google_sheets_cache_ttl_seconds=120,
        conversion_work_dir=str(runtime_dir),
        langgraph_checkpoint_db_path="",
        gemini_http_trust_env=False,
    )


def write_skill(
    root: Path,
    relative_dir: str,
    *,
    body: str,
    frontmatter: dict[str, Any] | None = None,
) -> Path:
    skill_dir = root / relative_dir
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"

    if frontmatter:
        lines = ["---"]
        for key, value in frontmatter.items():
            if isinstance(value, (list, tuple)):
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {item}")
                continue
            if isinstance(value, bool):
                normalized_value = "true" if value else "false"
            else:
                normalized_value = value
            lines.append(f"{key}: {normalized_value}")
        lines.append("---")
        content = "\n".join(lines) + "\n\n" + body.strip() + "\n"
    else:
        content = body.strip() + "\n"

    skill_path.write_text(content, encoding="utf-8")
    return skill_path
