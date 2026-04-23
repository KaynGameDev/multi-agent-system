from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from mas_platform.errors import ValidationError
from mas_platform.registry import resolve_repo_root


def scaffold_package(
    repo_root: str | Path,
    *,
    kind: str,
    agent_id: str,
    owner: str,
    description: str | None = None,
) -> Path:
    resolved_root = resolve_repo_root(repo_root)
    if not agent_id.isidentifier():
        raise ValidationError("agent_id must be a valid Python identifier.")
    if kind not in {"agent", "group"}:
        raise ValidationError("kind must be 'agent' or 'group'.")

    agents_dir = resolved_root / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    init_path = agents_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("", encoding="utf-8")

    package_root = agents_dir / agent_id
    if package_root.exists():
        raise ValidationError(f"Package already exists: {package_root}")

    package_root.mkdir(parents=True)
    (package_root / "tests").mkdir()
    (package_root / "tests" / "__init__.py").write_text("", encoding="utf-8")

    resolved_description = description or (
        f"Sample {'agent group' if kind == 'group' else 'agent'} scaffold for {agent_id}."
    )
    entrypoint = f"agents.{agent_id}.agent:build_app"

    (package_root / "manifest.yaml").write_text(
        dedent(
            f"""\
            id: {agent_id}
            version: 0.1.0
            kind: {"agent_group" if kind == "group" else "agent"}
            runtime: adk
            entrypoint: {entrypoint}
            owner: {owner}
            description: {resolved_description}
            tags:
              - scaffold
            required_secrets: []
            capabilities: []
            test_paths:
              - tests
            """
        ),
        encoding="utf-8",
    )
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "README.md").write_text(
        dedent(
            f"""\
            # {agent_id}

            {resolved_description}

            Generated with `mas scaffold {kind} {agent_id}`.

            Keep implementation details inside this folder. If you need reusable
            helpers, prefer importing from the repo's public `shared/` package
            instead of reaching into another agent's private files.
            """
        ),
        encoding="utf-8",
    )
    (package_root / "prompts.py").write_text(
        dedent(
            f"""\
            SYSTEM_PROMPT = {resolved_description!r}
            """
        ),
        encoding="utf-8",
    )
    (package_root / "tools.py").write_text(
        dedent(
            """\
            TOOLS = []
            """
        ),
        encoding="utf-8",
    )

    agent_source = (
        _group_agent_source(agent_id, resolved_description)
        if kind == "group"
        else _single_agent_source(agent_id, resolved_description)
    )
    (package_root / "agent.py").write_text(agent_source, encoding="utf-8")
    (package_root / "tests" / "test_smoke.py").write_text(
        _test_source(agent_id),
        encoding="utf-8",
    )
    return package_root


def _single_agent_source(agent_id: str, description: str) -> str:
    return dedent(
        f"""\
        from __future__ import annotations

        import sys
        from pathlib import Path
        from typing import AsyncGenerator

        from google.adk.agents import BaseAgent
        from google.adk.agents.invocation_context import InvocationContext
        from google.adk.apps import App

        REPO_ROOT = Path(__file__).resolve().parents[2]
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        from shared.adk_utils import build_text_event, extract_user_text

        from .prompts import SYSTEM_PROMPT
        from .tools import TOOLS


        class {agent_id.title().replace("_", "")}Agent(BaseAgent):
            description = SYSTEM_PROMPT

            async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator:
                user_text = extract_user_text(context) or "hello"
                yield build_text_event(
                    context,
                    author=self.name,
                    text=f"{{self.description}} Reply: {{user_text}}",
                )


        def build_app() -> App:
            return App(
                name="{agent_id}",
                root_agent={agent_id.title().replace("_", "")}Agent(
                    name="{agent_id}",
                    description=SYSTEM_PROMPT,
                ),
            )


        app = build_app()
        root_agent = app.root_agent
        """
    )


def _group_agent_source(agent_id: str, description: str) -> str:
    return dedent(
        f"""\
        from __future__ import annotations

        import sys
        from pathlib import Path
        from typing import AsyncGenerator

        from google.adk.agents import BaseAgent, SequentialAgent
        from google.adk.agents.invocation_context import InvocationContext
        from google.adk.apps import App

        REPO_ROOT = Path(__file__).resolve().parents[2]
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        from shared.adk_utils import build_text_event, extract_user_text

        from .prompts import SYSTEM_PROMPT


        class IntakeAgent(BaseAgent):
            async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator:
                context.session.state["request_text"] = extract_user_text(context) or "hello"
                yield build_text_event(
                    context,
                    author=self.name,
                    text="Captured the incoming request.",
                )


        class SummaryAgent(BaseAgent):
            async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator:
                request_text = str(context.session.state.get("request_text", "")).strip()
                yield build_text_event(
                    context,
                    author=self.name,
                    text=f"{SYSTEM_PROMPT} Summary: {{request_text}}",
                )


        def build_app() -> App:
            return App(
                name="{agent_id}",
                root_agent=SequentialAgent(
                    name="{agent_id}",
                    description=SYSTEM_PROMPT,
                    sub_agents=[
                        IntakeAgent(name="intake"),
                        SummaryAgent(name="summary"),
                    ],
                ),
            )


        app = build_app()
        root_agent = app.root_agent
        """
    )


def _test_source(agent_id: str) -> str:
    return dedent(
        f"""\
        from pathlib import Path

        from mas_platform.runtime import run_package


        def test_{agent_id}_smoke() -> None:
            repo_root = Path(__file__).resolve().parents[3]
            result = run_package(repo_root, agent_id="{agent_id}", message="hello")
            assert result.event_lines
        """
    )
