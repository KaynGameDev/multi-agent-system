from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.config import DEFAULT_KNOWLEDGE_BASE_DIR, load_settings
from app.contracts import build_assistant_response
from app.paths import resolve_project_path
from app.prompt_loader import join_prompt_layers, load_prompt_sections, load_shared_instruction_text
from app.skill_runtime import build_skill_prompt_context
from app.skills import SkillRegistry
from agents.document_conversion.rendering import (
    build_targeted_questions,
    classify_conversion_failure,
    is_retryable_conversion_failure,
    render_conversion_response,
)
from app.language import detect_response_language
from app.pending_actions import (
    build_pending_action,
    get_pending_action,
    is_pending_action_active,
    resolve_pending_action_reply,
    update_pending_action,
)
from app.state import AgentState
from tools.document_conversion import (
    DEFAULT_CONVERSION_WORK_DIR,
    OPTIONAL_MODULE_NAMES,
    UPLOAD_ONLY_FALLBACK_TEXT,
    append_answer_to_session,
    append_questions_to_session,
    build_default_fact_rows,
    build_missing_required_fields,
    ensure_company_scaffolding,
    ensure_game_shared_scaffolding,
    ingest_google_document_references,
    ingest_uploaded_files,
    load_existing_package_context,
    load_shared_context,
    normalize_module_name,
    normalize_slug,
    publish_conversion_package,
    stage_conversion_package,
    ConversionSessionRecord,
    ConversionSessionStore,
    build_conversion_package_relative_path,
)
from tools.conversion_google_sources import GoogleDocumentReference, extract_google_document_references
from tools.conversion_retrieval import build_retrieved_source_bundle

logger = logging.getLogger(__name__)
EXTRACT_DRAFT_MAX_ATTEMPTS = 3
EXTRACT_DRAFT_RETRY_BASE_DELAY_SECONDS = 0.75


class ConversionTerminologyItem(BaseModel):
    term_id: str = ""
    canonical_zh: str = ""
    canonical_en: str = ""
    aliases: list[str] = Field(default_factory=list)
    definition: str = ""


class ConversionEntityItem(BaseModel):
    entity_id: str = ""
    name_zh: str = ""
    name_en: str = ""
    description: str = ""


class ConversionRuleItem(BaseModel):
    rule_id: str = ""
    title_zh: str = ""
    title_en: str = ""
    description: str = ""
    condition: str = ""


class ConversionFactItem(BaseModel):
    module: str = "core"
    subject_type: str = ""
    subject_id: str = ""
    attribute: str = ""
    value_zh: str = ""
    value_en: str = ""
    value_raw: str = ""
    unit: str = ""
    condition: str = ""
    confidence: float = 0.7


class ConversionModuleItem(BaseModel):
    name: str = ""
    content: str = ""


class ConversionDraftPayload(BaseModel):
    game_name: str = ""
    game_slug: str = ""
    market_name: str = ""
    market_slug: str = ""
    feature_name: str = ""
    feature_slug: str = ""
    overview: str = ""
    terminology: list[ConversionTerminologyItem] = Field(default_factory=list)
    entities: list[ConversionEntityItem] = Field(default_factory=list)
    rules: list[ConversionRuleItem] = Field(default_factory=list)
    config_overview: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    facts: list[ConversionFactItem] = Field(default_factory=list)
    modules: list[ConversionModuleItem] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
PROMPT_PATH = "agents/document_conversion/AGENT.md"


class DocumentConversionAgentNode:
    def __init__(self, llm, settings=None, *, skill_registry: SkillRegistry | None = None, agent_name: str = "") -> None:
        self.llm = llm
        self.extractor = llm.with_structured_output(ConversionDraftPayload)
        self.settings = settings or load_settings()
        self.skill_registry = skill_registry
        self.agent_name = agent_name
        self.store = ConversionSessionStore(self._resolve_path(self.settings.conversion_work_dir, DEFAULT_CONVERSION_WORK_DIR))
        self.knowledge_root = self._resolve_path(
            self.settings.knowledge_base_dir,
            DEFAULT_KNOWLEDGE_BASE_DIR,
        )

    def __call__(self, state: AgentState) -> dict:
        try:
            result = self._run_turn(state)
            updates = {
                "messages": [AIMessage(content=result["content"])],
                "conversion_session_id": result["session"].session_id if result["session"] else "",
                "target_game_slug": result["session"].game_slug if result["session"] else "",
                "target_market_slug": result["session"].market_slug if result["session"] else "",
                "target_feature_slug": result["session"].feature_slug if result["session"] else "",
                "conversion_status": result["session"].status if result["session"] else "",
                "missing_required_fields": result["session"].missing_required_fields if result["session"] else [],
                "approval_state": result["session"].approval_state if result["session"] else "",
                "pending_action": result.get("pending_action"),
                "execution_contract": result.get("execution_contract"),
                "assistant_response": result.get("assistant_response"),
                "tool_invocation": result.get("tool_invocation"),
                "tool_result": result.get("tool_result"),
            }
            return updates
        except Exception as exc:
            self._mark_session_failed(state)
            preferred_language = resolve_preferred_language(state)
            failure_kind = classify_conversion_failure(exc)
            logger.exception(
                "Document conversion failed thread=%s channel=%s user=%s conversion_session_id=%s failure_kind=%s",
                str(state.get("thread_id", "")).strip(),
                str(state.get("channel_id", "")).strip(),
                str(state.get("user_id", "")).strip(),
                str(state.get("conversion_session_id", "")).strip(),
                failure_kind,
            )
            return {
                "messages": [
                    AIMessage(
                        content=render_conversion_response(
                            self.llm,
                            response_kind="failure",
                            preferred_language=preferred_language,
                            exc=exc,
                        )
                    )
                ],
                "conversion_status": "failed",
            }

    def _run_turn(self, state: AgentState) -> dict[str, Any]:
        ensure_company_scaffolding(self.knowledge_root)

        thread_id = str(state.get("thread_id", "")).strip()
        channel_id = str(state.get("channel_id", "")).strip()
        user_id = str(state.get("user_id", "")).strip()
        uploaded_files = state.get("uploaded_files")
        latest_text = get_latest_user_text(state)
        google_document_references = extract_google_document_references(latest_text)
        preferred_language = resolve_preferred_language(state)
        session = self._resolve_session(
            state=state,
            thread_id=thread_id,
            channel_id=channel_id,
            user_id=user_id,
            uploaded_files=uploaded_files if isinstance(uploaded_files, list) else [],
            google_document_references=google_document_references,
        )

        if session is None:
            content = render_conversion_response(
                self.llm,
                response_kind="missing_session",
                preferred_language=preferred_language,
            )
            return {
                "content": content,
                "assistant_response": build_assistant_response(kind="text", content=content),
                "session": None,
            }
        preferred_language = resolve_preferred_language(state, session)

        pending_action = get_pending_action(state)
        if pending_action and pending_action.get("requested_by_agent") == self.agent_name and is_pending_action_active(pending_action):
            pending_action_result = self._build_pending_action_response(
                state,
                session=session,
                pending_action=pending_action,
                preferred_language=preferred_language,
            )
            if pending_action_result is not None:
                return pending_action_result

        logger.debug(
            "Processing conversion session session=%s thread=%s channel=%s user=%s status=%s uploads=%s",
            session.session_id,
            thread_id,
            channel_id,
            user_id,
            session.status,
            len(uploaded_files) if isinstance(uploaded_files, list) else 0,
        )

        author = (
            str(state.get("user_sheet_name", "")).strip()
            or str(state.get("user_display_name", "")).strip()
            or user_id
        )

        ingested, skipped, download_failures = ingest_uploaded_files(
            self.store,
            session,
            uploaded_files if isinstance(uploaded_files, list) else [],
            slack_bot_token=self.settings.slack_bot_token,
            author=author,
        )
        google_ingested, google_access_failures = ingest_google_document_references(
            self.store,
            session,
            google_document_references,
            author=author,
        )
        session = self.store.get_session(session.session_id) or session
        source_access_failures = download_failures + google_access_failures

        if ingested or google_ingested:
            logger.debug(
                "Ingested conversion sources session=%s file_sources=%s google_sources=%s skipped=%s source_access_failures=%s",
                session.session_id,
                [source.original_name for source in ingested],
                [source.original_name for source in google_ingested],
                skipped,
                source_access_failures,
            )
        elif skipped or source_access_failures:
            logger.debug(
                "Conversion ingestion issues session=%s skipped=%s source_access_failures=%s",
                session.session_id,
                skipped,
                source_access_failures,
            )

        if source_access_failures and not self.store.list_sources(session.session_id):
            session = self.store.update_session(
                session.session_id,
                status="needs_info",
                missing_required_fields=["provenance"],
            )
            content = render_conversion_response(
                self.llm,
                response_kind="download_failure",
                preferred_language=preferred_language,
                download_failures=source_access_failures,
            )
            return {
                "content": content,
                "assistant_response": build_assistant_response(kind="text", content=content),
                "session": session,
            }

        if latest_text and latest_text != UPLOAD_ONLY_FALLBACK_TEXT:
            session = append_answer_to_session(self.store, session, latest_text)

        sources = self.store.list_sources(session.session_id)
        if not sources and skipped:
            skipped_text = ", ".join(f"`{name}`" for name in skipped)
            session = self.store.update_session(
                session.session_id,
                status="needs_info",
                missing_required_fields=["provenance"],
            )
            content = render_conversion_response(
                self.llm,
                response_kind="unsupported_files",
                preferred_language=preferred_language,
                skipped_text=skipped_text,
            )
            return {
                "content": content,
                "assistant_response": build_assistant_response(kind="await_confirmation", content=content),
                "session": session,
            }

        if not sources:
            session = self.store.update_session(
                session.session_id,
                status="needs_info",
                missing_required_fields=["provenance"],
            )
            content = render_conversion_response(
                self.llm,
                response_kind="missing_source",
                preferred_language=preferred_language,
            )
            return {
                "content": content,
                "assistant_response": build_assistant_response(kind="await_confirmation", content=content),
                "session": session,
            }

        initial_context = load_shared_context(self.knowledge_root)
        source_bundle = build_retrieved_source_bundle(
            sources,
            shared_context=initial_context,
            existing_package_context="",
            answer_history=session.answer_history,
            latest_user_text=latest_text,
        )
        first_pass = self._extract_draft(source_bundle, state)
        first_pass = normalize_draft_payload(first_pass)

        game_slug = str(first_pass.get("game_slug", "")).strip()
        market_slug = str(first_pass.get("market_slug", "")).strip()
        feature_slug = str(first_pass.get("feature_slug", "")).strip()

        if game_slug:
            ensure_game_shared_scaffolding(self.knowledge_root, game_slug)

        second_pass_context = load_shared_context(self.knowledge_root, game_slug=game_slug)
        existing_package_context = load_existing_package_context(
            self.knowledge_root,
            game_slug,
            market_slug,
            feature_slug,
        )
        if second_pass_context or existing_package_context:
            source_bundle = build_retrieved_source_bundle(
                sources,
                shared_context=second_pass_context,
                existing_package_context=existing_package_context,
                answer_history=session.answer_history,
                latest_user_text=latest_text,
                game_slug=game_slug,
                market_slug=market_slug,
                feature_slug=feature_slug,
            )
            draft_payload = normalize_draft_payload(self._extract_draft(source_bundle, state))
        else:
            draft_payload = first_pass

        conflicts = list(draft_payload.get("conflicts", []))
        conflicts.extend(detect_session_conflicts(session, draft_payload))
        if conflicts:
            session = self.store.update_session(
                session.session_id,
                status="blocked_conflict",
                game_slug=str(draft_payload.get("game_slug", "")).strip(),
                market_slug=str(draft_payload.get("market_slug", "")).strip(),
                feature_slug=str(draft_payload.get("feature_slug", "")).strip(),
                draft_payload=draft_payload,
                missing_required_fields=[],
                approval_state="pending",
            )
            content = render_conversion_response(
                self.llm,
                response_kind="conflict",
                preferred_language=preferred_language,
                conflicts=conflicts,
                skipped_files=skipped,
            )
            return {
                "content": content,
                "assistant_response": build_assistant_response(kind="await_confirmation", content=content),
                "session": session,
            }

        missing_required_fields = build_missing_required_fields(draft_payload, sources)
        if missing_required_fields:
            questions = build_targeted_questions(missing_required_fields, preferred_language)
            next_status = "blocked_unknown_target" if any(
                field in {"game_slug", "market_slug", "feature_slug"} for field in missing_required_fields
            ) else "needs_info"
            session = self.store.update_session(
                session.session_id,
                status=next_status,
                game_slug=str(draft_payload.get("game_slug", "")).strip(),
                market_slug=str(draft_payload.get("market_slug", "")).strip(),
                feature_slug=str(draft_payload.get("feature_slug", "")).strip(),
                draft_payload=draft_payload,
                missing_required_fields=missing_required_fields,
                approval_state="pending",
            )
            session = append_questions_to_session(self.store, session, questions)
            logger.debug(
                "Conversion session requires more info session=%s status=%s missing_fields=%s",
                session.session_id,
                next_status,
                missing_required_fields,
            )
            content = render_conversion_response(
                self.llm,
                response_kind="needs_info",
                preferred_language=preferred_language,
                session=session,
                questions=questions,
                skipped_files=skipped,
            )
            return {
                "content": content,
                "assistant_response": build_assistant_response(kind="await_confirmation", content=content),
                "session": session,
            }

        stage_result = stage_conversion_package(
            self.store,
            session,
            draft_payload,
            sources,
            knowledge_root=self.knowledge_root,
        )
        session = self.store.update_session(
            session.session_id,
            status="ready_for_approval",
            game_slug=str(draft_payload.get("game_slug", "")).strip(),
            market_slug=str(draft_payload.get("market_slug", "")).strip(),
            feature_slug=str(draft_payload.get("feature_slug", "")).strip(),
            draft_payload=draft_payload,
            missing_required_fields=[],
            approval_state="pending",
            staged_package_path=str(stage_result.package_path),
        )
        logger.debug(
            "Conversion session ready for approval session=%s package=%s populated_modules=%s missing_optional_modules=%s",
            session.session_id,
            session.staged_package_path,
            stage_result.populated_modules,
            stage_result.missing_optional_modules,
        )
        pending_action = build_conversion_pending_action(
            state=state,
            session=session,
            source_count=len(sources),
        )
        content = render_conversion_response(
            self.llm,
            response_kind="ready_for_approval",
            preferred_language=preferred_language,
            session=session,
            sources=sources,
            draft_payload=draft_payload,
            populated_modules=stage_result.populated_modules,
            missing_optional_modules=stage_result.missing_optional_modules,
            skipped_files=skipped,
            download_failures=source_access_failures,
        )
        return {
            "content": content,
            "assistant_response": build_assistant_response(
                kind="await_confirmation",
                content=content,
                pending_action=pending_action,
            ),
            "session": session,
            "pending_action": pending_action,
            "execution_contract": None,
        }

    def _build_pending_action_response(
        self,
        state: AgentState,
        *,
        session: ConversionSessionRecord,
        pending_action: dict[str, Any],
        preferred_language: str,
    ) -> dict[str, Any] | None:
        latest_text = get_latest_user_text(state)
        resolution = resolve_pending_action_reply(pending_action, latest_text)
        contract = resolution["contract"]
        validation = resolution["validation"]

        if validation.get("runtime_action") == "cancel":
            session = self.store.update_session(
                session.session_id,
                status="cancelled",
                approval_state="cancelled",
                missing_required_fields=[],
            )
            content = render_conversion_response(
                self.llm,
                response_kind="cancelled",
                preferred_language=preferred_language,
            )
            return {
                "content": content,
                "assistant_response": build_assistant_response(kind="text", content=content),
                "session": session,
                "pending_action": None,
                "execution_contract": None,
            }

        normalized_scope = validation.get("normalized_scope") or {}
        next_status = str(validation.get("next_status", "ask_clarification")).strip() or "ask_clarification"
        updated_action = update_pending_action(
            pending_action,
            status=next_status,
            target_scope=normalized_scope or None,
            metadata_updates={"last_contract": dict(contract)},
        )

        if not validation.get("valid"):
            content = build_conversion_pending_action_clarification(
                pending_action=updated_action,
                validation=validation,
                preferred_language=preferred_language,
            )
            return {
                "content": content,
                "assistant_response": build_assistant_response(
                    kind="await_confirmation",
                    content=content,
                    pending_action=updated_action,
                ),
                "session": session,
                "pending_action": updated_action,
                "execution_contract": None,
            }

        runtime_action = str(validation.get("runtime_action", "")).strip()
        if runtime_action == "request_revision":
            content = build_conversion_pending_action_revision_response(
                pending_action=updated_action,
                validation=validation,
                preferred_language=preferred_language,
            )
            return {
                "content": content,
                "assistant_response": build_assistant_response(
                    kind="await_confirmation",
                    content=content,
                    pending_action=updated_action,
                ),
                "session": session,
                "pending_action": updated_action,
                "execution_contract": None,
            }

        if runtime_action != "execute":
            content = build_conversion_pending_action_clarification(
                pending_action=updated_action,
                validation=validation,
                preferred_language=preferred_language,
            )
            return {
                "content": content,
                "assistant_response": build_assistant_response(
                    kind="await_confirmation",
                    content=content,
                    pending_action=updated_action,
                ),
                "session": session,
                "pending_action": updated_action,
                "execution_contract": None,
            }

        if session.status != "ready_for_approval" or not session.draft_payload:
            missing = ", ".join(session.missing_required_fields) or (
                "草稿尚未暂存" if preferred_language == "zh" else "the draft has not been staged yet"
            )
            content = render_conversion_response(
                self.llm,
                response_kind="not_ready_for_publish",
                preferred_language=preferred_language,
                missing=missing,
            )
            return {
                "content": content,
                "assistant_response": build_assistant_response(
                    kind="await_confirmation",
                    content=content,
                    pending_action=updated_action,
                    execution_contract=contract,
                ),
                "session": session,
                "pending_action": updated_action,
                "execution_contract": None,
            }

        relative_package_path = publish_conversion_package(
            self.store,
            session,
            knowledge_root=self.knowledge_root,
        )
        session = self.store.get_session(session.session_id) or session
        content = render_conversion_response(
            self.llm,
            response_kind="published",
            preferred_language=preferred_language,
            relative_package_path=relative_package_path,
            source_count=len(self.store.list_sources(session.session_id)),
        )
        return {
            "content": content,
            "assistant_response": build_assistant_response(
                kind="execute",
                content=content,
                execution_contract=contract,
            ),
            "session": session,
            "pending_action": None,
            "execution_contract": contract,
        }

    def _resolve_path(self, configured_value: str, default_value: str) -> Path:
        return resolve_project_path(configured_value, default_value)

    def _resolve_session(
        self,
        *,
        state: AgentState,
        thread_id: str,
        channel_id: str,
        user_id: str,
        uploaded_files: list[dict[str, Any]],
        google_document_references: list[GoogleDocumentReference],
    ) -> ConversionSessionRecord | None:
        session_id = str(state.get("conversion_session_id", "")).strip()
        if session_id:
            session = self.store.get_session(session_id)
            if session is not None:
                return session

        if thread_id:
            session = self.store.get_active_session_by_thread(thread_id)
            if session is not None:
                return session

        if uploaded_files or google_document_references:
            return self.store.create_session(
                thread_id=thread_id or channel_id or user_id or "conversion-thread",
                channel_id=channel_id or "unknown-channel",
                user_id=user_id or "unknown-user",
            )
        return None

    def _mark_session_failed(self, state: AgentState) -> None:
        session = self._find_existing_session(state)
        if session is None:
            return
        try:
            self.store.update_session(
                session.session_id,
                status="failed",
                approval_state="failed",
                missing_required_fields=[],
            )
        except Exception:
            pass

    def _find_existing_session(self, state: AgentState) -> ConversionSessionRecord | None:
        session_id = str(state.get("conversion_session_id", "")).strip()
        if session_id:
            session = self.store.get_session(session_id)
            if session is not None:
                return session

        thread_id = str(state.get("thread_id", "")).strip()
        if thread_id:
            return self.store.get_active_session_by_thread(thread_id)
        return None

    def _extract_draft(self, source_bundle: str, state: AgentState) -> dict[str, Any]:
        messages = [
            SystemMessage(
                content=build_conversion_extractor_prompt(
                    self.skill_registry,
                    agent_name=self.agent_name,
                    state=state,
                )
            ),
            HumanMessage(content=source_bundle),
        ]
        last_error: Exception | None = None

        for attempt in range(1, EXTRACT_DRAFT_MAX_ATTEMPTS + 1):
            try:
                result = self.extractor.invoke(messages)
                if isinstance(result, BaseModel):
                    return result.model_dump()
                if hasattr(result, "model_dump"):
                    return result.model_dump()
                if isinstance(result, dict):
                    return result
                raise RuntimeError("Conversion extractor returned an unexpected payload type.")
            except Exception as exc:
                last_error = exc
                if attempt >= EXTRACT_DRAFT_MAX_ATTEMPTS or not is_retryable_conversion_failure(exc):
                    raise
                delay_seconds = EXTRACT_DRAFT_RETRY_BASE_DELAY_SECONDS * attempt
                logger.warning(
                    "Conversion extractor disconnected; retrying attempt=%s/%s source_bundle_chars=%s error=%s",
                    attempt + 1,
                    EXTRACT_DRAFT_MAX_ATTEMPTS,
                    len(source_bundle),
                    exc,
                )
                time.sleep(delay_seconds)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Conversion extractor failed without returning a payload.")


def build_conversion_pending_action(
    *,
    state: AgentState,
    session: ConversionSessionRecord,
    source_count: int,
) -> dict[str, Any] | None:
    relative_package_path = build_conversion_package_relative_path(
        session.game_slug,
        session.market_slug,
        session.feature_slug,
    )
    summary = f"Publish staged conversion package for `{relative_package_path}`."
    metadata = {
        "conversion_session_id": session.session_id,
        "staged_package_path": session.staged_package_path,
        "relative_package_path": relative_package_path,
        "source_count": source_count,
        "game_slug": session.game_slug,
        "market_slug": session.market_slug,
        "feature_slug": session.feature_slug,
    }
    return build_pending_action(
        session_id=str(session.session_id or state.get("thread_id", "")).strip(),
        action_type="publish_conversion_package",
        requested_by_agent="document_conversion_agent",
        summary=summary,
        target_scope={"files": [session.staged_package_path]} if session.staged_package_path else {},
        risk_level="high",
        requires_explicit_approval=True,
        metadata=metadata,
    )


def build_conversion_pending_action_clarification(
    *,
    pending_action: dict[str, Any],
    validation: dict[str, Any],
    preferred_language: str,
) -> str:
    summary = str(pending_action.get("summary", "")).strip()
    reason = str(validation.get("reason", "")).strip()
    if preferred_language == "zh":
        parts = [f"我还不能确定是否发布这个转换结果：{summary}"]
        if reason:
            parts.append(f"原因：{reason}")
        parts.append("请回复 `approve` 继续发布，或回复 `cancel` 取消。")
        return "\n\n".join(parts).strip()

    parts = [f"I still can't determine whether to publish this conversion result: {summary}"]
    if reason:
        parts.append(f"Reason: {reason}")
    parts.append("Reply `approve` to publish, or `cancel` to stop.")
    return "\n\n".join(parts).strip()


def build_conversion_pending_action_revision_response(
    *,
    pending_action: dict[str, Any],
    validation: dict[str, Any],
    preferred_language: str,
) -> str:
    metadata = pending_action.get("metadata") if isinstance(pending_action.get("metadata"), dict) else {}
    staged_package_path = str(metadata.get("staged_package_path", "")).strip()
    relative_package_path = str(metadata.get("relative_package_path", "")).strip()
    summary = str(pending_action.get("summary", "")).strip()
    reason = str(validation.get("reason", "")).strip()
    package_label = relative_package_path or staged_package_path or summary

    if preferred_language == "zh":
        parts = [f"这个转换结果已经暂存，尚未发布：{package_label}"]
        if reason:
            parts.append(f"原因：{reason}")
        parts.append("如果可以继续发布，请回复 `approve`；如果要停止，请回复 `cancel`。")
        return "\n\n".join(parts).strip()

    parts = [f"This conversion result is staged but not published yet: {package_label}"]
    if reason:
        parts.append(f"Reason: {reason}")
    parts.append("Reply `approve` to publish, or `cancel` to stop.")
    return "\n\n".join(parts).strip()


def build_conversion_extractor_prompt(
    skill_registry: SkillRegistry | None = None,
    *,
    agent_name: str = "",
    state: AgentState | None = None,
) -> str:
    sections = load_prompt_sections(
        PROMPT_PATH,
        required_sections=(
            "extractor_role",
            "extractor_responsibilities",
            "extractor_boundaries",
        ),
    )
    skill_prompt = build_skill_prompt_context(
        state,
        skill_registry=skill_registry,
        agent_name=agent_name,
    )
    return join_prompt_layers(
        load_shared_instruction_text(),
        sections["extractor_role"],
        sections["extractor_responsibilities"],
        skill_prompt,
        sections["extractor_boundaries"],
    )


def normalize_draft_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["game_slug"] = normalize_slug(str(normalized.get("game_slug", "")).strip())
    normalized["market_slug"] = normalize_slug(str(normalized.get("market_slug", "")).strip())
    normalized["feature_slug"] = normalize_slug(str(normalized.get("feature_slug", "")).strip())

    if not str(normalized.get("feature_name", "")).strip() and normalized.get("feature_slug"):
        normalized["feature_name"] = str(normalized["feature_slug"]).replace("-", " ").title()
    if not str(normalized.get("game_name", "")).strip() and normalized.get("game_slug"):
        normalized["game_name"] = str(normalized["game_slug"]).replace("-", " ").title()
    if not str(normalized.get("market_name", "")).strip() and normalized.get("market_slug"):
        normalized["market_name"] = str(normalized["market_slug"]).replace("-", " ").title()

    for key in ("terminology", "entities", "rules", "facts", "modules", "conflicts", "config_overview", "open_questions", "assumptions"):
        value = normalized.get(key)
        if not isinstance(value, list):
            normalized[key] = []

    normalized_modules: list[dict[str, str]] = []
    for item in normalized["modules"]:
        if not isinstance(item, dict):
            continue
        name = normalize_module_name(str(item.get("name", "")))
        content = str(item.get("content", "")).strip()
        if name in OPTIONAL_MODULE_NAMES and content:
            normalized_modules.append({"name": name, "content": content})
    normalized["modules"] = normalized_modules

    if not normalized["open_questions"] and not normalized["assumptions"]:
        normalized["assumptions"] = [
            "No additional assumptions or unresolved questions were identified from the current source bundle."
        ]

    normalized["facts"] = build_default_fact_rows(normalized, [])
    return normalized


def detect_session_conflicts(session: ConversionSessionRecord, draft_payload: dict[str, Any]) -> list[str]:
    conflicts: list[str] = []
    for field_name in ("game_slug", "market_slug", "feature_slug"):
        existing_value = getattr(session, field_name)
        new_value = str(draft_payload.get(field_name, "")).strip()
        if existing_value and new_value and existing_value != new_value:
            conflicts.append(
                f"{field_name} changed from `{existing_value}` to `{new_value}` within the same conversion session."
            )
    return conflicts


def get_latest_user_text(state: AgentState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return stringify_message_content(message.content)
    return ""


def stringify_message_content(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(part.strip() for part in parts if part.strip()).strip()
    return str(content).strip()


def resolve_preferred_language(
    state: AgentState,
    session: ConversionSessionRecord | None = None,
) -> str:
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, HumanMessage):
            continue
        text = stringify_message_content(message.content)
        if not text or text == UPLOAD_ONLY_FALLBACK_TEXT:
            continue
        return detect_response_language(text)

    if session is not None:
        for text in reversed(session.answer_history):
            cleaned = str(text).strip()
            if not cleaned or cleaned == UPLOAD_ONLY_FALLBACK_TEXT:
                continue
            return detect_response_language(cleaned)

    latest_text = get_latest_user_text(state)
    if latest_text:
        return detect_response_language(latest_text)
    return "en"
