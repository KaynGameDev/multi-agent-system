from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from mas_platform.loader import load_app
from mas_platform.errors import MasPlatformError
from mas_platform.registry import load_registry


@dataclass(slots=True)
class RunResult:
    agent_id: str
    app_name: str
    session_id: str
    event_lines: list[str]


def event_to_text_line(event) -> str | None:
    content = getattr(event, "content", None)
    if content is None:
        return None
    parts = getattr(content, "parts", None) or []
    fragments = [part.text for part in parts if getattr(part, "text", None)]
    if not fragments:
        return None
    text = "\n".join(fragment for fragment in fragments if fragment).strip()
    if not text:
        return None
    author = getattr(event, "author", "agent")
    return f"[{author}] {text}"


async def _run_package_async(
    repo_root: str | Path,
    *,
    agent_id: str,
    message: str,
    user_id: str,
    session_id: str | None,
) -> RunResult:
    registry = load_registry(repo_root)
    if agent_id not in registry:
        raise MasPlatformError(f"Unknown agent id '{agent_id}'.")
    package = registry[agent_id]
    app = load_app(package, repo_root=repo_root)
    session_service = InMemorySessionService()
    resolved_session_id = session_id or f"{agent_id}-{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name=app.name,
        user_id=user_id,
        session_id=resolved_session_id,
    )
    runner = Runner(
        app=app,
        session_service=session_service,
    )
    new_message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=message)],
    )
    event_lines: list[str] = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=resolved_session_id,
        new_message=new_message,
    ):
        line = event_to_text_line(event)
        if line:
            event_lines.append(line)
    return RunResult(
        agent_id=agent_id,
        app_name=app.name,
        session_id=resolved_session_id,
        event_lines=event_lines,
    )


def run_package(
    repo_root: str | Path,
    *,
    agent_id: str,
    message: str,
    user_id: str = "local_user",
    session_id: str | None = None,
) -> RunResult:
    return asyncio.run(
        _run_package_async(
            repo_root,
            agent_id=agent_id,
            message=message,
            user_id=user_id,
            session_id=session_id,
        )
    )
