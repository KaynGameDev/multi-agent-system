from __future__ import annotations

import logging
import secrets
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.checkpoints import GraphCheckpointStore
from app.config import Settings
from app.context_window import (
    ContextWindowThresholdOverrides,
    USAGE_BASELINE_STAGE_BEFORE_MESSAGE,
    evaluate_context_window_for_transcript,
    format_context_window_status,
    serialize_context_window_snapshot,
)
from app.identity import build_user_identity_context
from app.messages import extract_final_text
from app.memory.agent_scope import resolve_agent_memory_context
from app.memory.consolidation import (
    consolidate_long_term_memory,
    should_schedule_long_term_memory_consolidation,
)
from app.memory.consolidation_background import (
    BackgroundMemoryConsolidator,
    MemoryConsolidationTarget,
)
from app.memory.extraction import persist_durable_turn_memories, turn_has_direct_memory_write
from app.paths import resolve_project_path
from app.reactive_recovery import (
    ReactiveRecoverySignal,
    build_reactive_recovery_detail,
    detect_reactive_recovery_signal_from_exception,
    detect_reactive_recovery_signal_from_final_state,
)
from app.rehydration import (
    RUNTIME_REHYDRATION_METADATA_KEY,
    build_runtime_rehydration_state,
    extract_runtime_rehydration_state_from_transcript,
    merge_runtime_rehydration_state,
)
from app.session_memory import (
    DEFAULT_SESSION_MEMORY_BACKGROUND_MIN_TURNS,
    SessionMemoryStore,
    build_session_memory_record,
    count_session_memory_refresh_activity,
    should_schedule_background_session_memory_refresh,
)
from app.session_memory_background import BackgroundSessionMemoryUpdater, SessionMemoryRefreshTarget
from interfaces.web.conversations import (
    ConversationNotFoundError,
    TRANSCRIPT_TYPE_MESSAGE,
    WebConversationStore,
    transcript_to_langchain_messages,
)
from tools.document_conversion import ConversionSessionStore

logger = logging.getLogger(__name__)

KNOWLEDGE_BUILD_REQUESTED_AGENT = "knowledge_base_builder_agent"

def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )


def format_web_chat_url(host: str, port: int) -> str:
    normalized_host = (host or "").strip() or "127.0.0.1"
    if normalized_host in {"0.0.0.0", "::", "[::]"}:
        normalized_host = "127.0.0.1"
    return f"http://{normalized_host}:{port}"


def format_model_label(provider: str, model: str) -> str:
    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model or "").strip()
    if not normalized_model:
        return "Model unavailable"

    if normalized_provider == "google" and normalized_model == "gemini-3-flash-preview":
        return "Gemini Flash 3"
    if normalized_provider == "minimax" and normalized_model == "MiniMax-M2.7-highspeed":
        return "MiniMax M2.7 Highspeed"
    if normalized_provider == "openai" and normalized_model == "gpt-5-mini":
        return "GPT-5 Mini"
    return normalized_model


class ConversationCreateRequest(BaseModel):
    title: str = "New chat"


class ConversationRenameRequest(BaseModel):
    title: str = "New chat"


class WebMessageRequest(BaseModel):
    message: str = Field(min_length=1)
    display_name: str = ""
    email: str = ""


class WebLoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class WebServer:
    def __init__(
        self,
        agent_graph,
        settings: Settings,
        conversation_store: WebConversationStore | None = None,
        conversion_store: ConversionSessionStore | None = None,
        checkpoint_store: GraphCheckpointStore | None = None,
        session_memory_updater: BackgroundSessionMemoryUpdater | None = None,
        memory_consolidator: BackgroundMemoryConsolidator | None = None,
    ) -> None:
        self.agent_graph = agent_graph
        self.settings = settings
        self.conversation_store = conversation_store or WebConversationStore(
            self._resolve_path(settings.conversion_work_dir) / "web_conversations.json"
        )
        self.session_memory_store = SessionMemoryStore(
            self._resolve_path(settings.conversion_work_dir) / "session_memory.json"
        )
        self.session_memory_updater = session_memory_updater or BackgroundSessionMemoryUpdater(
            self._refresh_session_memory_in_background,
        )
        self.memory_consolidator = memory_consolidator or BackgroundMemoryConsolidator(
            self._consolidate_memory_in_background,
            debounce_seconds=self.settings.memory_consolidation_debounce_seconds,
        )
        self.conversion_store = conversion_store or ConversionSessionStore(self._resolve_path(settings.conversion_work_dir))
        self.checkpoint_store = checkpoint_store
        self.static_dir = Path(__file__).resolve().parent / "static"
        self._auto_compact_failures: dict[str, int] = {}
        self._auth_credentials = {
            credential.username: credential.password
            for credential in self.settings.web_auth_credentials
        }
        from app.graph import build_default_agent_registrations

        self._agent_memory_scopes = {
            registration.name: registration.memory_scope
            for registration in build_default_agent_registrations(settings=self.settings)
            if registration.memory_scope is not None
        }
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
        self.session_memory_updater.close()
        self.memory_consolidator.close()
        if self._server is not None:
            self._server.should_exit = True

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Jade Agent Web Chat")
        limiter = Limiter(key_func=get_remote_address)

        if self.settings.web_allowed_hosts:
            app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=list(self.settings.web_allowed_hosts),
            )

        if self.settings.web_auth_enabled:
            app.add_middleware(
                SessionMiddleware,
                secret_key=self.settings.web_auth_session_secret,
                session_cookie="jade_web_session",
                same_site="lax",
                https_only=self.settings.web_auth_cookie_secure,
                max_age=self.settings.web_auth_session_max_age_seconds,
            )

        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        app.add_middleware(SlowAPIMiddleware)

        @app.middleware("http")
        async def disable_frontend_caching(request: Request, call_next):
            response = await call_next(request)
            if request.url.path in {"/", "/login"} or request.url.path.startswith("/static/"):
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            return response

        app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")

        @app.get("/")
        def index(request: Request) -> Response:
            auth_redirect = self._build_auth_redirect(request)
            if auth_redirect is not None:
                return auth_redirect

            template = (self.static_dir / "index.html").read_text(encoding="utf-8")
            html = (
                template
                .replace("{{ LOGO_HREF }}", self._versioned_static_path("branding/jade-logo-512.png"))
                .replace("{{ FAVICON_PNG_HREF }}", self._versioned_static_path("branding/favicon-32.png"))
                .replace("{{ APPLE_TOUCH_ICON_HREF }}", self._versioned_static_path("branding/apple-touch-icon.png"))
                .replace("{{ FAVICON_ICO_HREF }}", self._versioned_static_path("branding/favicon.ico"))
                .replace("{{ CHAT_MODEL_LABEL }}", format_model_label(self.settings.llm_provider, self.settings.llm_model))
                .replace("{{ APP_CSS_HREF }}", self._versioned_static_path("app.css"))
                .replace("{{ APP_JS_HREF }}", self._versioned_static_path("app.js"))
            )
            return HTMLResponse(content=html)

        @app.get("/login")
        def login_page(request: Request) -> Response:
            if not self.settings.web_auth_enabled:
                return RedirectResponse(url="/", status_code=303)
            if self._get_authenticated_user(request):
                return RedirectResponse(url="/", status_code=303)

            template = (self.static_dir / "login.html").read_text(encoding="utf-8")
            html = (
                template
                .replace("{{ LOGO_HREF }}", self._versioned_static_path("branding/jade-logo-512.png"))
                .replace("{{ FAVICON_PNG_HREF }}", self._versioned_static_path("branding/favicon-32.png"))
                .replace("{{ APPLE_TOUCH_ICON_HREF }}", self._versioned_static_path("branding/apple-touch-icon.png"))
                .replace("{{ FAVICON_ICO_HREF }}", self._versioned_static_path("branding/favicon.ico"))
                .replace("{{ LOGIN_CSS_HREF }}", self._versioned_static_path("login.css"))
                .replace("{{ LOGIN_JS_HREF }}", self._versioned_static_path("login.js"))
            )
            return HTMLResponse(content=html)

        @app.get("/favicon.ico")
        def favicon() -> RedirectResponse:
            return RedirectResponse(url=self._versioned_static_path("branding/favicon.ico"), status_code=307)

        @app.get("/api/health")
        def health() -> dict[str, bool]:
            return {"ok": True}

        @app.get("/api/auth/session")
        def auth_session(request: Request) -> dict[str, object]:
            username = self._get_authenticated_user(request)
            if not self.settings.web_auth_enabled:
                return {
                    "enabled": False,
                    "authenticated": True,
                    "username": "",
                }
            return {
                "enabled": True,
                "authenticated": bool(username),
                "username": username,
            }

        @app.post("/api/auth/login")
        @limiter.limit("5/minute")
        def login(request: Request, payload: WebLoginRequest) -> dict[str, object]:
            if not self.settings.web_auth_enabled:
                raise HTTPException(status_code=400, detail="Web authentication is not enabled.")

            normalized_username = payload.username.strip()
            expected_password = self._auth_credentials.get(normalized_username)
            if expected_password is None or not secrets.compare_digest(expected_password, payload.password):
                raise HTTPException(status_code=401, detail="Invalid username or password.")

            request.session.clear()
            request.session["web_auth_user"] = normalized_username
            return {
                "enabled": True,
                "authenticated": True,
                "username": normalized_username,
            }

        @app.post("/api/auth/logout")
        def logout(request: Request) -> dict[str, object]:
            if self.settings.web_auth_enabled:
                request.session.clear()
            return {
                "enabled": bool(self.settings.web_auth_enabled),
                "authenticated": False,
                "username": "",
            }

        @app.get("/api/conversations")
        def list_conversations(request: Request) -> dict[str, list[dict]]:
            self._require_api_auth(request)
            return {"conversations": self.conversation_store.list_conversations()}

        @app.post("/api/conversations")
        @limiter.limit("20/minute")
        def create_conversation(request: Request, payload: ConversationCreateRequest | None = None) -> dict:
            self._require_api_auth(request)
            title = (payload.title if payload is not None else "New chat").strip() or "New chat"
            return self.conversation_store.create_conversation(title=title)

        @app.get("/api/conversations/{conversation_id}")
        def get_conversation(request: Request, conversation_id: str) -> dict:
            self._require_api_auth(request)
            try:
                return self._build_public_conversation_payload(conversation_id)
            except ConversationNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Conversation not found.") from exc

        @app.patch("/api/conversations/{conversation_id}")
        def rename_conversation(request: Request, conversation_id: str, payload: ConversationRenameRequest) -> dict:
            self._require_api_auth(request)
            try:
                return self.conversation_store.rename_conversation(
                    conversation_id,
                    title=payload.title,
                )
            except ConversationNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Conversation not found.") from exc

        @app.delete("/api/conversations/{conversation_id}")
        def delete_conversation(request: Request, conversation_id: str) -> dict[str, object]:
            self._require_api_auth(request)
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
            self._clear_auto_compact_failures(thread_id)
            self._delete_session_memory(thread_id)

            return {
                "deleted": True,
                "conversation_id": conversation_id,
            }

        @app.post("/api/conversations/{conversation_id}/messages")
        @limiter.limit("10/minute")
        def send_message(request: Request, conversation_id: str, payload: WebMessageRequest) -> dict:
            self._require_api_auth(request)
            try:
                self._build_public_conversation_payload(conversation_id)
            except ConversationNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Conversation not found.") from exc

            user_message = payload.message.strip()
            self.conversation_store.append_message(
                conversation_id,
                role="user",
                markdown=user_message,
            )
            full_conversation = self.conversation_store.get_full_conversation(conversation_id)
            thread_id = f"web:{conversation_id}"
            context_snapshot = self._evaluate_context_window_snapshot(
                thread_id=thread_id,
                messages=full_conversation.get("messages", []),
            )
            context_compaction: dict[str, Any] = {
                "attempted": False,
                "applied": False,
                "failure_count": self._get_auto_compact_failure_count(thread_id),
            }
            limit_recovery: dict[str, Any] = {
                "attempted": False,
                "recovered": False,
                "reason": "",
                "retry_count": 0,
            }
            if not context_snapshot.decision.should_auto_compact:
                self._clear_auto_compact_failures(thread_id)
            elif context_snapshot.decision.auto_compact_available:
                attempted_auto_compaction = self._attempt_auto_compaction(
                    conversation_id=conversation_id,
                    thread_id=thread_id,
                    conversation=full_conversation,
                )
                if attempted_auto_compaction is None:
                    blocked_snapshot = self._evaluate_context_window_snapshot(
                        thread_id=thread_id,
                        messages=full_conversation.get("messages", []),
                    )
                    return self._build_context_window_block_response(
                        conversation_id=conversation_id,
                        snapshot=blocked_snapshot,
                        detail=self._build_context_window_block_detail(
                            blocked_snapshot,
                            auto_compaction_failed=True,
                        ),
                        context_compaction={
                            "attempted": True,
                            "applied": False,
                            "failure_count": self._get_auto_compact_failure_count(thread_id),
                        },
                    )
                full_conversation = attempted_auto_compaction["conversation"]
                context_snapshot = attempted_auto_compaction["context_snapshot"]
                context_compaction = attempted_auto_compaction["context_compaction"]
            elif context_snapshot.decision.should_block:
                return self._build_context_window_block_response(
                    conversation_id=conversation_id,
                    snapshot=context_snapshot,
                    detail=self._build_context_window_block_detail(context_snapshot),
                    context_compaction=context_compaction,
                )

            if context_snapshot.decision.should_block:
                return self._build_context_window_block_response(
                    conversation_id=conversation_id,
                    snapshot=context_snapshot,
                    detail=self._build_context_window_block_detail(
                        context_snapshot,
                        after_auto_compaction=context_compaction.get("applied", False),
                    ),
                    context_compaction=context_compaction,
                )

            active_session = self.conversion_store.get_active_session_by_thread(thread_id)
            invoke_result = self._invoke_with_reactive_recovery(
                conversation_id=conversation_id,
                thread_id=thread_id,
                payload=payload,
                latest_user_message=user_message,
                conversation=full_conversation,
                context_snapshot=context_snapshot,
                context_compaction=context_compaction,
                active_session=active_session,
            )
            if isinstance(invoke_result, JSONResponse):
                return invoke_result

            assistant_text = invoke_result["assistant_text"]
            route = invoke_result["route"]
            route_reason = invoke_result["route_reason"]
            skill_resolution_diagnostics = invoke_result["skill_resolution_diagnostics"]
            agent_selection_diagnostics = invoke_result["agent_selection_diagnostics"]
            selection_warnings = invoke_result["selection_warnings"]
            assistant_usage = invoke_result["assistant_usage"]
            assistant_metadata = invoke_result["assistant_metadata"]
            context_snapshot = invoke_result["context_snapshot"]
            context_compaction = invoke_result["context_compaction"]
            limit_recovery = invoke_result["limit_recovery"]

            self.conversation_store.append_transcript_message(
                conversation_id,
                role="assistant",
                message_type=TRANSCRIPT_TYPE_MESSAGE,
                markdown=assistant_text,
                usage=assistant_usage,
                metadata=assistant_metadata,
            )
            updated_conversation = self.conversation_store.get_full_conversation(conversation_id)
            memory_state = self._build_web_memory_runtime_state(
                conversation_id=conversation_id,
                thread_id=thread_id,
                payload=payload,
                conversation=updated_conversation,
                route=route,
                active_session=active_session,
            )
            self._extract_durable_memories_after_turn(
                conversation_id=conversation_id,
                thread_id=thread_id,
                route=route,
                state=memory_state,
                conversation=updated_conversation,
                assistant_metadata=assistant_metadata,
            )
            self._schedule_memory_consolidation(
                route=route,
                state=memory_state,
            )
            self._schedule_session_memory_refresh(
                conversation_id=conversation_id,
                thread_id=thread_id,
                conversation=updated_conversation,
                force_refresh=bool(context_compaction.get("applied")),
            )
            conversation = self._build_public_conversation_payload(conversation_id)
            return {
                **conversation,
                "assistant_message": conversation["messages"][-1],
                "blocked": False,
                "context_window": serialize_context_window_snapshot(context_snapshot),
                "context_compaction": context_compaction,
                "limit_recovery": limit_recovery,
                "route": route,
                "route_reason": route_reason,
                "skill_resolution_diagnostics": skill_resolution_diagnostics,
                "agent_selection_diagnostics": agent_selection_diagnostics,
                "selection_warnings": selection_warnings,
            }

        return app

    def _build_web_initial_state(
        self,
        *,
        conversation_id: str,
        thread_id: str,
        payload: WebMessageRequest,
        latest_user_message: str,
        conversation: dict[str, Any],
        active_session: Any | None,
    ) -> dict[str, Any]:
        seed_messages = self._build_seed_messages(
            thread_id=thread_id,
            conversation=conversation,
            latest_user_message=latest_user_message,
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
            "requested_agent": KNOWLEDGE_BUILD_REQUESTED_AGENT,
            "requested_skill_ids": [],
            "uploaded_files": [],
            "context_paths": [],
            "conversion_session_id": active_session.session_id if active_session is not None else "",
            **user_context,
        }
        requested_agent = self._resolve_requested_agent_from_transcript(conversation)
        if requested_agent:
            initial_state["requested_agent"] = requested_agent
        return self._apply_transcript_rehydration(
            initial_state,
            conversation=conversation,
        )

    def _invoke_with_reactive_recovery(
        self,
        *,
        conversation_id: str,
        thread_id: str,
        payload: WebMessageRequest,
        latest_user_message: str,
        conversation: dict[str, Any],
        context_snapshot: Any,
        context_compaction: dict[str, Any],
        active_session: Any | None,
    ) -> dict[str, Any] | JSONResponse:
        limit_recovery: dict[str, Any] = {
            "attempted": False,
            "recovered": False,
            "reason": "",
            "retry_count": 0,
        }
        current_conversation = conversation
        current_context_snapshot = context_snapshot
        current_context_compaction = dict(context_compaction)

        for attempt_index in range(2):
            initial_state = self._build_web_initial_state(
                conversation_id=conversation_id,
                thread_id=thread_id,
                payload=payload,
                latest_user_message=latest_user_message,
                conversation=current_conversation,
                active_session=active_session,
            )
            logger.debug(
                "Invoking web graph conversation=%s thread=%s email=%s display_name=%s attempt=%s",
                conversation_id,
                thread_id,
                payload.email.strip(),
                payload.display_name.strip(),
                attempt_index + 1,
            )
            logger.info(
                "Context window thread=%s model=%s %s",
                thread_id,
                self.settings.llm_model,
                format_context_window_status(current_context_snapshot),
            )

            try:
                final_state = self.agent_graph.invoke(
                    initial_state,
                    config={"configurable": {"thread_id": thread_id}},
                )
            except Exception as exc:
                signal = detect_reactive_recovery_signal_from_exception(exc)
                if signal is not None and attempt_index == 0:
                    recovery_result = self._attempt_reactive_limit_recovery(
                        conversation_id=conversation_id,
                        thread_id=thread_id,
                        conversation=current_conversation,
                        signal=signal,
                    )
                    limit_recovery.update(
                        {
                            "attempted": True,
                            "reason": signal.kind,
                            "retry_count": 1,
                        }
                    )
                    if recovery_result is None:
                        return self._build_context_window_block_response(
                            conversation_id=conversation_id,
                            snapshot=current_context_snapshot,
                            detail=build_reactive_recovery_detail(
                                signal,
                                compaction_failed=True,
                            ),
                            context_compaction=current_context_compaction,
                            limit_recovery=limit_recovery,
                        )
                    current_conversation = recovery_result["conversation"]
                    current_context_snapshot = recovery_result["context_snapshot"]
                    current_context_compaction = recovery_result["context_compaction"]
                    continue

                if signal is not None:
                    return self._build_context_window_block_response(
                        conversation_id=conversation_id,
                        snapshot=current_context_snapshot,
                        detail=build_reactive_recovery_detail(
                            signal,
                            retry_exhausted=bool(limit_recovery["attempted"]),
                        ),
                        context_compaction=current_context_compaction,
                        limit_recovery=limit_recovery,
                    )

                logger.exception("Failed while processing web conversation=%s", conversation_id)
                return self._build_unhandled_invoke_error_response(
                    conversation_id=conversation_id,
                    snapshot=current_context_snapshot,
                    context_compaction=current_context_compaction,
                    limit_recovery=limit_recovery,
                )

            signal = detect_reactive_recovery_signal_from_final_state(final_state)
            if signal is not None and attempt_index == 0:
                recovery_result = self._attempt_reactive_limit_recovery(
                    conversation_id=conversation_id,
                    thread_id=thread_id,
                    conversation=current_conversation,
                    signal=signal,
                )
                limit_recovery.update(
                    {
                        "attempted": True,
                        "reason": signal.kind,
                        "retry_count": 1,
                    }
                )
                if recovery_result is None:
                    return self._build_context_window_block_response(
                        conversation_id=conversation_id,
                        snapshot=current_context_snapshot,
                        detail=build_reactive_recovery_detail(
                            signal,
                            compaction_failed=True,
                        ),
                        context_compaction=current_context_compaction,
                        limit_recovery=limit_recovery,
                    )
                current_conversation = recovery_result["conversation"]
                current_context_snapshot = recovery_result["context_snapshot"]
                current_context_compaction = recovery_result["context_compaction"]
                continue

            if signal is not None:
                return self._build_context_window_block_response(
                    conversation_id=conversation_id,
                    snapshot=current_context_snapshot,
                    detail=build_reactive_recovery_detail(
                        signal,
                        retry_exhausted=bool(limit_recovery["attempted"]),
                    ),
                    context_compaction=current_context_compaction,
                    limit_recovery=limit_recovery,
                )

            assistant_text = extract_final_text(final_state)
            route = str(final_state.get("route", "")).strip()
            route_reason = str(final_state.get("route_reason", "")).strip()
            skill_resolution_diagnostics = final_state.get("skill_resolution_diagnostics", [])
            agent_selection_diagnostics = final_state.get("agent_selection_diagnostics", [])
            selection_warnings = final_state.get("selection_warnings", [])
            assistant_usage, assistant_metadata = self._extract_transcript_usage_metadata(final_state)
            assistant_metadata = self._augment_transcript_metadata_with_runtime_state(
                assistant_metadata,
                final_state,
            )
            if limit_recovery["attempted"]:
                limit_recovery["recovered"] = True
            return {
                "assistant_text": assistant_text,
                "route": route,
                "route_reason": route_reason,
                "skill_resolution_diagnostics": skill_resolution_diagnostics,
                "agent_selection_diagnostics": agent_selection_diagnostics,
                "selection_warnings": selection_warnings,
                "assistant_usage": assistant_usage,
                "assistant_metadata": assistant_metadata,
                "context_snapshot": current_context_snapshot,
                "context_compaction": current_context_compaction,
                "limit_recovery": limit_recovery,
            }

        unreachable = ReactiveRecoverySignal(kind="prompt_too_long", detail="retry exhausted")
        return self._build_context_window_block_response(
            conversation_id=conversation_id,
            snapshot=current_context_snapshot,
            detail=build_reactive_recovery_detail(unreachable, retry_exhausted=True),
            context_compaction=current_context_compaction,
            limit_recovery=limit_recovery,
        )

    def _context_window_threshold_overrides(self) -> ContextWindowThresholdOverrides | None:
        if not any(
            value is not None
            for value in (
                self.settings.context_window_effective_window,
                self.settings.context_window_warning_threshold,
                self.settings.context_window_auto_compact_threshold,
                self.settings.context_window_hard_block_threshold,
            )
        ):
            return None
        return ContextWindowThresholdOverrides(
            effective_window=self.settings.context_window_effective_window,
            warning_threshold=self.settings.context_window_warning_threshold,
            auto_compact_threshold=self.settings.context_window_auto_compact_threshold,
            hard_block_threshold=self.settings.context_window_hard_block_threshold,
        )

    def _evaluate_context_window_snapshot(self, thread_id: str, messages: list[dict[str, Any]]) -> Any:
        return evaluate_context_window_for_transcript(
            messages,
            model=self.settings.llm_model,
            threshold_overrides=self._context_window_threshold_overrides(),
            auto_compact_enabled=self.settings.context_window_auto_compact_enabled,
            auto_compact_failure_count=self._get_auto_compact_failure_count(thread_id),
            auto_compact_failure_limit=self.settings.context_window_auto_compact_failure_limit,
        )

    def _attempt_auto_compaction(
        self,
        *,
        conversation_id: str,
        thread_id: str,
        conversation: dict[str, Any],
    ) -> dict[str, Any] | None:
        try:
            return self._apply_compaction_to_conversation(
                conversation_id=conversation_id,
                thread_id=thread_id,
                conversation=conversation,
                trigger="auto",
                boundary_trigger="auto",
                record_auto_compact_failure=True,
            )
        except Exception:
            failure_count = self._record_auto_compact_failure(thread_id)
            logger.warning(
                "Auto compaction failed conversation=%s thread=%s failures=%s",
                conversation_id,
                thread_id,
                failure_count,
                exc_info=True,
            )
            return None

    def _build_context_window_block_response(
        self,
        *,
        conversation_id: str,
        snapshot: Any,
        detail: str,
        context_compaction: dict[str, Any],
        limit_recovery: dict[str, Any] | None = None,
    ) -> JSONResponse:
        conversation = self._build_public_conversation_payload(conversation_id)
        payload = {
            **conversation,
            "assistant_message": None,
            "blocked": True,
            "detail": detail,
            "context_window": serialize_context_window_snapshot(snapshot),
            "context_compaction": context_compaction,
            "limit_recovery": limit_recovery or {
                "attempted": False,
                "recovered": False,
                "reason": "",
                "retry_count": 0,
            },
            "route": "",
            "route_reason": detail,
            "skill_resolution_diagnostics": [],
            "agent_selection_diagnostics": [],
            "selection_warnings": [detail],
        }
        return JSONResponse(status_code=409, content=payload)

    def _build_unhandled_invoke_error_response(
        self,
        *,
        conversation_id: str,
        snapshot: Any,
        context_compaction: dict[str, Any],
        limit_recovery: dict[str, Any] | None = None,
        detail: str = "I hit an error while processing that request. Please try again.",
    ) -> JSONResponse:
        conversation = self._build_public_conversation_payload(conversation_id)
        payload = {
            **conversation,
            "assistant_message": None,
            "blocked": False,
            "error": True,
            "detail": detail,
            "context_window": serialize_context_window_snapshot(snapshot),
            "context_compaction": context_compaction,
            "limit_recovery": limit_recovery or {
                "attempted": False,
                "recovered": False,
                "reason": "",
                "retry_count": 0,
            },
            "route": "",
            "route_reason": "",
            "skill_resolution_diagnostics": [],
            "agent_selection_diagnostics": [],
            "selection_warnings": [],
        }
        return JSONResponse(status_code=500, content=payload)

    def _build_context_window_block_detail(
        self,
        snapshot: Any,
        *,
        auto_compaction_failed: bool = False,
        after_auto_compaction: bool = False,
    ) -> str:
        usage_summary = (
            f"Context usage is {snapshot.used_tokens}/{snapshot.thresholds.effective_window} tokens "
            f"({snapshot.remaining_percentage:.1f}% remaining)."
        )
        if after_auto_compaction:
            return (
                f"{usage_summary} I auto-compacted the conversation, but the projected request is still too large. "
                "Please compact the conversation manually before continuing."
            )
        if auto_compaction_failed:
            if snapshot.decision.auto_compact_breaker_open:
                return (
                    f"{usage_summary} Automatic compaction is now paused for this conversation after repeated failures. "
                    "Please compact the conversation manually before continuing."
                )
            return (
                f"{usage_summary} I couldn't auto-compact this conversation safely, so I stopped before sending the "
                "next model request. Please compact the conversation manually and try again."
            )
        if snapshot.decision.auto_compact_breaker_open:
            return (
                f"{usage_summary} Automatic compaction is temporarily paused after repeated failures. "
                "Please compact the conversation manually before sending another request."
            )
        if not snapshot.decision.auto_compact_enabled:
            return (
                f"{usage_summary} Automatic compaction is disabled, so please compact the conversation manually "
                "before sending another request."
            )
        if snapshot.decision.should_hard_block:
            return (
                f"{usage_summary} This conversation is already at the hard context limit. "
                "Please compact it manually before continuing."
            )
        return (
            f"{usage_summary} This conversation needs manual compaction before I can send another model request."
        )

    def _get_auto_compact_failure_count(self, thread_id: str) -> int:
        return max(int(self._auto_compact_failures.get(thread_id, 0) or 0), 0)

    def _record_auto_compact_failure(self, thread_id: str) -> int:
        failure_count = self._get_auto_compact_failure_count(thread_id) + 1
        self._auto_compact_failures[thread_id] = failure_count
        return failure_count

    def _clear_auto_compact_failures(self, thread_id: str) -> None:
        self._auto_compact_failures.pop(thread_id, None)

    def _attempt_reactive_limit_recovery(
        self,
        *,
        conversation_id: str,
        thread_id: str,
        conversation: dict[str, Any],
        signal: ReactiveRecoverySignal,
    ) -> dict[str, Any] | None:
        try:
            return self._apply_compaction_to_conversation(
                conversation_id=conversation_id,
                thread_id=thread_id,
                conversation=conversation,
                trigger="reactive_recovery",
                boundary_trigger=f"reactive_recovery:{signal.kind}",
                preserved_tail_count=max(self.settings.context_window_auto_compact_preserved_tail_count, 1),
                record_auto_compact_failure=False,
                extra_context_compaction_fields={"reason": signal.kind},
            )
        except Exception:
            logger.warning(
                "Reactive recovery compaction failed conversation=%s thread=%s reason=%s",
                conversation_id,
                thread_id,
                signal.kind,
                exc_info=True,
            )
            return None

    def _apply_compaction_to_conversation(
        self,
        *,
        conversation_id: str,
        thread_id: str,
        conversation: dict[str, Any],
        trigger: str,
        boundary_trigger: str,
        preserved_tail_count: int | None = None,
        record_auto_compact_failure: bool,
        extra_context_compaction_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from app.compaction import compact_conversation

        session_memory = self.session_memory_store.get(thread_id) if self.settings.session_memory_enabled else None
        bundle = compact_conversation(
            conversation.get("messages", []),
            trigger=boundary_trigger,
            preserved_tail_count=(
                self.settings.context_window_auto_compact_preserved_tail_count
                if preserved_tail_count is None
                else preserved_tail_count
            ),
            session_memory=session_memory,
        )
        self._reset_checkpoint_thread(thread_id, raise_on_failure=True)
        self._cancel_session_memory_refresh(thread_id)
        self.conversation_store.replace_transcript(
            conversation_id,
            messages=bundle.compacted_messages,
        )
        if record_auto_compact_failure:
            self._clear_auto_compact_failures(thread_id)

        compacted_conversation = self.conversation_store.get_full_conversation(conversation_id)
        self._schedule_session_memory_refresh(
            conversation_id=conversation_id,
            thread_id=thread_id,
            conversation=compacted_conversation,
            force_refresh=True,
        )
        compacted_snapshot = self._evaluate_context_window_snapshot(
            thread_id=thread_id,
            messages=compacted_conversation.get("messages", []),
        )
        logger.info(
            "%s compaction applied thread=%s %s",
            trigger,
            thread_id,
            format_context_window_status(compacted_snapshot),
        )
        context_compaction = {
            "attempted": True,
            "applied": True,
            "failure_count": 0 if record_auto_compact_failure else self._get_auto_compact_failure_count(thread_id),
            "trigger": trigger,
            "compacted_source_count": bundle.compacted_source_count,
            "preserved_tail_count": len(bundle.preserved_tail_messages),
            "summary_message_id": bundle.summary_message["id"],
            "used_session_memory": bundle.used_session_memory,
        }
        if extra_context_compaction_fields:
            context_compaction.update(extra_context_compaction_fields)
        return {
            "conversation": compacted_conversation,
            "context_snapshot": compacted_snapshot,
            "context_compaction": context_compaction,
        }

    def _apply_transcript_rehydration(
        self,
        initial_state: dict[str, Any],
        *,
        conversation: dict[str, Any],
    ) -> dict[str, Any]:
        transcript_messages = conversation.get("messages", [])
        if not isinstance(transcript_messages, list):
            return dict(initial_state)

        runtime_rehydration_state = extract_runtime_rehydration_state_from_transcript(transcript_messages)
        if not runtime_rehydration_state:
            return dict(initial_state)

        merged_state = merge_runtime_rehydration_state(initial_state, runtime_rehydration_state)
        logger.info(
            "Applied transcript rehydration thread=%s fields=%s recent_file_reads=%s",
            str(merged_state.get("thread_id", "")).strip(),
            sorted(runtime_rehydration_state.keys()),
            runtime_rehydration_state.get("recent_file_reads", []),
        )
        requested_agent = self._resolve_requested_agent_from_transcript(conversation)
        merged_state["requested_agent"] = requested_agent
        return merged_state

    def _build_public_conversation_payload(self, conversation_id: str) -> dict[str, Any]:
        public_conversation = self.conversation_store.get_conversation(conversation_id)
        return public_conversation

    def _resolve_requested_agent_from_transcript(self, conversation: dict[str, Any]) -> str:
        return KNOWLEDGE_BUILD_REQUESTED_AGENT

    def _augment_transcript_metadata_with_runtime_state(
        self,
        metadata: dict[str, object] | None,
        final_state: dict[str, Any],
    ) -> dict[str, object] | None:
        runtime_rehydration_state = build_runtime_rehydration_state(final_state)
        if not runtime_rehydration_state:
            return metadata

        merged_metadata = dict(metadata or {})
        merged_metadata[RUNTIME_REHYDRATION_METADATA_KEY] = runtime_rehydration_state
        return merged_metadata

    def _schedule_session_memory_refresh(
        self,
        *,
        conversation_id: str,
        thread_id: str,
        conversation: dict[str, Any],
        force_refresh: bool = False,
    ) -> None:
        if not self.settings.session_memory_enabled:
            return

        transcript_messages = conversation.get("messages", [])
        if not isinstance(transcript_messages, list):
            return

        existing_record = self.session_memory_store.get(thread_id)
        if not force_refresh and not should_schedule_background_session_memory_refresh(
            transcript_messages,
            existing_record,
            initialize_threshold_tokens=self.settings.session_memory_initialize_threshold_tokens,
            update_growth_threshold_tokens=self.settings.session_memory_update_growth_threshold_tokens,
            min_turns=DEFAULT_SESSION_MEMORY_BACKGROUND_MIN_TURNS,
        ):
            return

        self.session_memory_updater.schedule(
            SessionMemoryRefreshTarget(
                conversation_id=conversation_id,
                thread_id=thread_id,
                allowed_session_file_path=str(self.session_memory_store.resolve_session_file_path(thread_id)),
                force_refresh=force_refresh,
            )
        )

    def _extract_durable_memories_after_turn(
        self,
        *,
        conversation_id: str,
        thread_id: str,
        route: str,
        state: dict[str, Any],
        conversation: dict[str, Any],
        assistant_metadata: dict[str, object] | None,
    ) -> None:
        if not self.settings.long_term_memory_enabled:
            return

        memory_scope = self._agent_memory_scopes.get(str(route or "").strip())
        if memory_scope is None:
            return
        if turn_has_direct_memory_write(assistant_metadata if isinstance(assistant_metadata, dict) else None):
            return

        transcript_messages = conversation.get("messages", [])
        if not isinstance(transcript_messages, list):
            return

        try:
            persisted = persist_durable_turn_memories(
                self.settings,
                agent_name=str(route or "").strip(),
                memory_scope=memory_scope,
                state=state,
                transcript_messages=transcript_messages,
            )
        except Exception:
            logger.warning(
                "Automatic durable memory extraction failed conversation=%s thread=%s route=%s",
                conversation_id,
                thread_id,
                route,
                exc_info=True,
            )
            return

        if persisted:
            logger.info(
                "Persisted durable memories conversation=%s thread=%s route=%s count=%s ids=%s",
                conversation_id,
                thread_id,
                route,
                len(persisted),
                [memory.memory_id for memory in persisted],
            )

    def _schedule_memory_consolidation(
        self,
        *,
        route: str,
        state: dict[str, Any],
    ) -> None:
        if not self.settings.memory_consolidation_enabled:
            return

        normalized_route = str(route or "").strip()
        memory_scope = self._agent_memory_scopes.get(normalized_route)
        if memory_scope is None:
            return

        context = resolve_agent_memory_context(
            self.settings,
            agent_name=normalized_route,
            memory_scope=memory_scope,
            state=state,
        )
        if not should_schedule_long_term_memory_consolidation(
            context.root_dir,
            min_entries=self.settings.memory_consolidation_min_entries,
        ):
            return

        self.memory_consolidator.schedule(
            MemoryConsolidationTarget(
                root_dir=str(context.root_dir),
                agent_name=context.agent_name,
                memory_scope=context.scope,
                scope_key=context.scope_key,
            )
        )

    def _build_web_memory_runtime_state(
        self,
        *,
        conversation_id: str,
        thread_id: str,
        payload: WebMessageRequest,
        conversation: dict[str, Any],
        route: str,
        active_session: Any | None,
    ) -> dict[str, Any]:
        user_context = build_user_identity_context(
            slack_display_name=payload.display_name,
            slack_real_name=payload.display_name,
            email=payload.email,
        )
        base_state = {
            "interface_name": "web",
            "thread_id": thread_id,
            "channel_id": conversation_id,
            "requested_agent": str(route or "").strip(),
            "user_id": payload.email.strip() or payload.display_name.strip() or thread_id,
            "conversion_session_id": active_session.session_id if active_session is not None else "",
            **user_context,
        }
        merged_state = self._apply_transcript_rehydration(
            base_state,
            conversation=conversation,
        )
        merged_state["requested_agent"] = str(route or "").strip()
        return merged_state

    def _consolidate_memory_in_background(
        self,
        target: MemoryConsolidationTarget,
    ) -> None:
        if not self.settings.memory_consolidation_enabled:
            return

        summary = consolidate_long_term_memory(
            target.root_dir,
            min_entries=self.settings.memory_consolidation_min_entries,
        )
        if summary.updated_memory_ids or summary.deleted_memory_ids:
            logger.info(
                "Consolidated long-term memory root=%s agent=%s scope=%s scope_key=%s updated=%s deleted=%s noisy_groups=%s duplicate_groups=%s",
                target.root_dir,
                target.agent_name,
                target.memory_scope,
                target.scope_key,
                summary.updated_memory_ids,
                summary.deleted_memory_ids,
                summary.noisy_group_count,
                summary.duplicate_group_count,
            )

    def _refresh_session_memory_in_background(
        self,
        target: SessionMemoryRefreshTarget,
    ) -> None:
        if not self.settings.session_memory_enabled:
            return

        try:
            conversation = self.conversation_store.get_full_conversation(target.conversation_id)
        except ConversationNotFoundError:
            logger.info(
                "Skipped background session memory refresh for deleted conversation=%s thread=%s",
                target.conversation_id,
                target.thread_id,
            )
            return

        transcript_messages = conversation.get("messages", [])
        if not isinstance(transcript_messages, list):
            return

        thread_id = target.thread_id
        existing_record = self.session_memory_store.get(thread_id)
        activity = count_session_memory_refresh_activity(transcript_messages, existing_record)
        updated_record = build_session_memory_record(
            thread_id,
            transcript_messages,
            session_memory=existing_record,
            initialize_threshold_tokens=self.settings.session_memory_initialize_threshold_tokens,
            update_growth_threshold_tokens=self.settings.session_memory_update_growth_threshold_tokens,
            force_refresh=(
                target.force_refresh
                or activity.turn_count >= DEFAULT_SESSION_MEMORY_BACKGROUND_MIN_TURNS
                or activity.tool_activity_count > 0
            ),
        )
        if updated_record is None:
            return

        self.session_memory_store.upsert_scoped(
            updated_record,
            allowed_thread_id=thread_id,
            allowed_session_file_path=target.allowed_session_file_path,
        )
        logger.info(
            "Updated session memory in background thread=%s source=%s covered_messages=%s covered_tokens=%s",
            thread_id,
            updated_record.source,
            updated_record.covered_message_count,
            updated_record.covered_tokens,
        )

    def _cancel_session_memory_refresh(self, thread_id: str) -> None:
        self.session_memory_updater.cancel(thread_id)

    def _delete_session_memory(self, thread_id: str) -> None:
        self._cancel_session_memory_refresh(thread_id)
        try:
            self.session_memory_store.delete(thread_id)
        except Exception:
            logger.warning("Failed to delete session memory thread=%s", thread_id, exc_info=True)

    def _versioned_static_path(self, relative_path: str) -> str:
        asset_path = self.static_dir / relative_path
        version = asset_path.stat().st_mtime_ns
        return f"/static/{relative_path}?v={version}"

    def _build_auth_redirect(self, request: Request) -> RedirectResponse | None:
        if self.settings.web_auth_enabled and not self._get_authenticated_user(request):
            return RedirectResponse(url="/login", status_code=303)
        return None

    def _require_api_auth(self, request: Request) -> None:
        if self.settings.web_auth_enabled and not self._get_authenticated_user(request):
            raise HTTPException(status_code=401, detail="Authentication required.")

    def _get_authenticated_user(self, request: Request) -> str:
        if not self.settings.web_auth_enabled:
            return ""

        username = str(request.session.get("web_auth_user", "")).strip()
        if username and username in self._auth_credentials:
            return username

        if username:
            request.session.clear()
        return ""

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

    def _reset_checkpoint_thread(self, thread_id: str, *, raise_on_failure: bool = False) -> bool:
        if self.checkpoint_store is None:
            return True
        try:
            self.checkpoint_store.delete_thread(thread_id)
        except Exception as exc:
            if raise_on_failure:
                raise RuntimeError(f"Failed to reset checkpoint thread={thread_id}") from exc
            logger.warning("Failed to reset checkpoint thread=%s", thread_id, exc_info=True)
            return False
        return True

    def _extract_transcript_usage_metadata(self, final_state: dict) -> tuple[dict[str, int] | None, dict[str, object] | None]:
        messages = final_state.get("messages") or []
        if not isinstance(messages, list):
            return None, None

        usage_message_index: int | None = None
        usage_metadata: dict[str, int] | None = None
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            raw_usage = getattr(message, "usage_metadata", None)
            if not isinstance(raw_usage, dict):
                continue
            normalized_usage = {
                str(key): value
                for key, value in raw_usage.items()
                if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
            }
            if not normalized_usage:
                continue
            usage_message_index = index
            usage_metadata = normalized_usage
            break

        if usage_message_index is None or usage_metadata is None:
            return None, None

        metadata: dict[str, object] = {}
        if usage_message_index != len(messages) - 1:
            metadata["usage_baseline_stage"] = USAGE_BASELINE_STAGE_BEFORE_MESSAGE
        return usage_metadata, metadata or None
