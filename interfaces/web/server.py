from __future__ import annotations

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.checkpoints import GraphCheckpointStore
from app.config import Settings
from app.identity import build_user_identity_context
from app.messages import extract_final_text
from app.paths import resolve_project_path
from interfaces.web.conversations import (
    ConversationNotFoundError,
    WebConversationStore,
    transcript_to_langchain_messages,
)
from tools.document_conversion import ConversionSessionStore

logger = logging.getLogger(__name__)


def format_web_chat_url(host: str, port: int) -> str:
    normalized_host = (host or "").strip() or "127.0.0.1"
    if normalized_host in {"0.0.0.0", "::", "[::]"}:
        normalized_host = "127.0.0.1"
    return f"http://{normalized_host}:{port}"


class ConversationCreateRequest(BaseModel):
    title: str = "New chat"


class ConversationRenameRequest(BaseModel):
    title: str = "New chat"


class WebMessageRequest(BaseModel):
    message: str = Field(min_length=1)
    display_name: str = ""
    email: str = ""


class WebServer:
    def __init__(
        self,
        agent_graph,
        settings: Settings,
        conversation_store: WebConversationStore | None = None,
        conversion_store: ConversionSessionStore | None = None,
        checkpoint_store: GraphCheckpointStore | None = None,
    ) -> None:
        self.agent_graph = agent_graph
        self.settings = settings
        self.conversation_store = conversation_store or WebConversationStore(
            self._resolve_path(settings.conversion_work_dir) / "web_conversations.json"
        )
        self.conversion_store = conversion_store or ConversionSessionStore(self._resolve_path(settings.conversion_work_dir))
        self.checkpoint_store = checkpoint_store
        self.static_dir = Path(__file__).resolve().parent / "static"
        self.app = self._build_app()
        self._server: uvicorn.Server | None = None

    def start(self) -> None:
        config = uvicorn.Config(
            self.app,
            host=self.settings.web_host,
            port=self.settings.web_port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        print(f"🌐 Jade Agent web chat is listening on {format_web_chat_url(self.settings.web_host, self.settings.web_port)}")
        self._server.run()

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Jade Agent Web Chat")

        @app.middleware("http")
        async def disable_frontend_caching(request: Request, call_next):
            response = await call_next(request)
            if request.url.path == "/" or request.url.path.startswith("/static/"):
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            return response

        app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")

        @app.get("/")
        def index() -> HTMLResponse:
            template = (self.static_dir / "index.html").read_text(encoding="utf-8")
            html = (
                template
                .replace("{{ APP_CSS_HREF }}", self._versioned_static_path("app.css"))
                .replace("{{ APP_JS_HREF }}", self._versioned_static_path("app.js"))
            )
            return HTMLResponse(content=html)

        @app.get("/api/health")
        def health() -> dict[str, bool]:
            return {"ok": True}

        @app.get("/api/conversations")
        def list_conversations() -> dict[str, list[dict]]:
            return {"conversations": self.conversation_store.list_conversations()}

        @app.post("/api/conversations")
        def create_conversation(payload: ConversationCreateRequest | None = None) -> dict:
            title = (payload.title if payload is not None else "New chat").strip() or "New chat"
            return self.conversation_store.create_conversation(title=title)

        @app.get("/api/conversations/{conversation_id}")
        def get_conversation(conversation_id: str) -> dict:
            try:
                return self.conversation_store.get_conversation(conversation_id)
            except ConversationNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Conversation not found.") from exc

        @app.patch("/api/conversations/{conversation_id}")
        def rename_conversation(conversation_id: str, payload: ConversationRenameRequest) -> dict:
            try:
                return self.conversation_store.rename_conversation(
                    conversation_id,
                    title=payload.title,
                )
            except ConversationNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Conversation not found.") from exc

        @app.delete("/api/conversations/{conversation_id}")
        def delete_conversation(conversation_id: str) -> dict[str, object]:
            thread_id = f"web:{conversation_id}"
            try:
                self.conversation_store.delete_conversation(conversation_id)
            except ConversationNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Conversation not found.") from exc

            try:
                self.conversion_store.delete_sessions_by_thread(thread_id)
            except Exception:
                logger.warning("Failed to delete conversion sessions for thread=%s", thread_id, exc_info=True)

            if self.checkpoint_store is not None:
                try:
                    self.checkpoint_store.delete_thread(thread_id)
                except Exception:
                    logger.warning("Failed to delete checkpoint thread=%s", thread_id, exc_info=True)

            return {
                "deleted": True,
                "conversation_id": conversation_id,
            }

        @app.post("/api/conversations/{conversation_id}/messages")
        def send_message(conversation_id: str, payload: WebMessageRequest) -> dict:
            try:
                self.conversation_store.get_conversation(conversation_id)
            except ConversationNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Conversation not found.") from exc

            user_message = payload.message.strip()
            conversation = self.conversation_store.append_message(
                conversation_id,
                role="user",
                markdown=user_message,
            )
            thread_id = f"web:{conversation_id}"
            active_session = self.conversion_store.get_active_session_by_thread(thread_id)
            seed_messages = self._build_seed_messages(
                thread_id=thread_id,
                conversation=conversation,
                latest_user_message=user_message,
            )

            user_context = build_user_identity_context(
                slack_display_name=payload.display_name,
                slack_real_name=payload.display_name,
                email=payload.email,
            )
            initial_state = {
                "messages": seed_messages,
                "interface_name": "web",
                "thread_id": thread_id,
                "user_id": payload.email.strip() or payload.display_name.strip() or thread_id,
                "channel_id": conversation_id,
                "requested_agent": "",
                "requested_skill_ids": [],
                "uploaded_files": [],
                "context_paths": [],
                "conversion_session_id": active_session.session_id if active_session is not None else "",
                **user_context,
            }

            logger.debug(
                "Invoking web graph conversation=%s thread=%s email=%s display_name=%s",
                conversation_id,
                thread_id,
                payload.email.strip(),
                payload.display_name.strip(),
            )

            try:
                final_state = self.agent_graph.invoke(
                    initial_state,
                    config={"configurable": {"thread_id": thread_id}},
                )
                assistant_text = extract_final_text(final_state)
                route = str(final_state.get("route", "")).strip()
                route_reason = str(final_state.get("route_reason", "")).strip()
                skill_resolution_diagnostics = final_state.get("skill_resolution_diagnostics", [])
                agent_selection_diagnostics = final_state.get("agent_selection_diagnostics", [])
                selection_warnings = final_state.get("selection_warnings", [])
            except Exception as exc:
                logger.exception("Failed while processing web conversation=%s", conversation_id)
                assistant_text = f"I hit an error while processing that request: {exc}"
                route = ""
                route_reason = "Request failed before a route completed."
                skill_resolution_diagnostics = []
                agent_selection_diagnostics = []
                selection_warnings = []

            conversation = self.conversation_store.append_message(
                conversation_id,
                role="assistant",
                markdown=assistant_text,
            )
            return {
                **conversation,
                "assistant_message": conversation["messages"][-1],
                "route": route,
                "route_reason": route_reason,
                "skill_resolution_diagnostics": skill_resolution_diagnostics,
                "agent_selection_diagnostics": agent_selection_diagnostics,
                "selection_warnings": selection_warnings,
            }

        return app

    def _versioned_static_path(self, relative_path: str) -> str:
        asset_path = self.static_dir / relative_path
        version = asset_path.stat().st_mtime_ns
        return f"/static/{relative_path}?v={version}"

    def _resolve_path(self, configured_value: str) -> Path:
        return resolve_project_path(configured_value, self.settings.conversion_work_dir)

    def _build_seed_messages(
        self,
        *,
        thread_id: str,
        conversation: dict,
        latest_user_message: str,
    ) -> list:
        transcript_messages = conversation.get("messages", [])
        has_prior_transcript = isinstance(transcript_messages, list) and len(transcript_messages) > 1
        if self.checkpoint_store is None:
            if has_prior_transcript:
                return transcript_to_langchain_messages(transcript_messages)
            return [HumanMessage(content=latest_user_message)]

        try:
            if self.checkpoint_store.has_checkpoint(thread_id):
                return [HumanMessage(content=latest_user_message)]
        except Exception:
            logger.warning("Failed to read checkpoint thread=%s; rebuilding from transcript.", thread_id, exc_info=True)
            self._reset_checkpoint_thread(thread_id)
            if has_prior_transcript:
                return transcript_to_langchain_messages(transcript_messages)
            return [HumanMessage(content=latest_user_message)]

        if has_prior_transcript:
            return transcript_to_langchain_messages(transcript_messages)
        return [HumanMessage(content=latest_user_message)]

    def _reset_checkpoint_thread(self, thread_id: str) -> None:
        if self.checkpoint_store is None:
            return
        try:
            self.checkpoint_store.delete_thread(thread_id)
        except Exception:
            logger.warning("Failed to reset checkpoint thread=%s", thread_id, exc_info=True)
