from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from core.config import DEFAULT_KNOWLEDGE_BASE_DIR, load_settings
from core.document_conversion_rendering import (
    classify_conversion_failure,
    build_targeted_questions,
    render_conversion_response,
)
from core.language import detect_response_language
from core.state import AgentState
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
    load_source_bundle_text,
    normalize_module_name,
    normalize_slug,
    publish_conversion_package,
    stage_conversion_package,
    ConversionSessionRecord,
    ConversionSessionStore,
)
from tools.conversion_google_sources import GoogleDocumentReference, extract_google_document_references

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
logger = logging.getLogger(__name__)

APPROVAL_PATTERNS = (
    r"^\s*approve\s*$",
    r"^\s*approved\s*$",
    r"^\s*publish\s*$",
    r"^\s*go ahead\s*$",
    r"^\s*looks good\s*$",
    r"^\s*批准\s*$",
    r"^\s*通过\s*$",
    r"^\s*发布\s*$",
)
CANCEL_PATTERNS = (
    r"^\s*cancel\s*$",
    r"^\s*stop\s*$",
    r"^\s*discard\s*$",
    r"^\s*取消\s*$",
    r"^\s*停止\s*$",
    r"^\s*放弃\s*$",
    r"^\s*算了\s*$",
)


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


CONVERSION_EXTRACTOR_PROMPT = (
    "You convert internal game design documents into canonical AI-friendly knowledge packages. "
    "Work only from the provided source bundle, user clarifications, shared company/game context, and existing approved package context. "
    "Do not invent undocumented behavior. If information is missing, leave fields empty. "
    "Return lowercase ASCII kebab-case slugs for game_slug, market_slug, and feature_slug whenever the source bundle makes them clear. "
    "Preserve important Chinese wording in the Chinese fields when present, and add concise English normalization when confident. "
    "For modules, only populate optional module content for config, economy, localization, ui, analytics, or qa when there is enough evidence. "
    "If the source bundle contains contradictions, list them in conflicts."
)


class DocumentConversionAgentNode:
    def __init__(self, llm, settings=None) -> None:
        self.llm = llm
        self.extractor = llm.with_structured_output(ConversionDraftPayload)
        self.settings = settings or load_settings()
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
            return {
                "content": render_conversion_response(
                    self.llm,
                    response_kind="missing_session",
                    preferred_language=preferred_language,
                ),
                "session": None,
            }
        preferred_language = resolve_preferred_language(state, session)

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
            return {
                "content": render_conversion_response(
                    self.llm,
                    response_kind="download_failure",
                    preferred_language=preferred_language,
                    download_failures=source_access_failures,
                ),
                "session": session,
            }

        if is_cancel_intent(latest_text):
            session = self.store.update_session(
                session.session_id,
                status="cancelled",
                approval_state="cancelled",
                missing_required_fields=[],
            )
            return {
                "content": render_conversion_response(
                    self.llm,
                    response_kind="cancelled",
                    preferred_language=preferred_language,
                ),
                "session": session,
            }

        if latest_text and latest_text != UPLOAD_ONLY_FALLBACK_TEXT and not is_approval_intent(latest_text):
            session = append_answer_to_session(self.store, session, latest_text)

        if is_approval_intent(latest_text):
            if session.status != "ready_for_approval" or not session.draft_payload:
                missing = ", ".join(session.missing_required_fields) or (
                    "草稿尚未暂存" if preferred_language == "zh" else "the draft has not been staged yet"
                )
                return {
                    "content": render_conversion_response(
                        self.llm,
                        response_kind="not_ready_for_publish",
                        preferred_language=preferred_language,
                        missing=missing,
                    ),
                    "session": session,
                }

            relative_package_path = publish_conversion_package(
                self.store,
                session,
                knowledge_root=self.knowledge_root,
            )
            session = self.store.get_session(session.session_id) or session
            return {
                "content": render_conversion_response(
                    self.llm,
                    response_kind="published",
                    preferred_language=preferred_language,
                    relative_package_path=relative_package_path,
                    source_count=len(self.store.list_sources(session.session_id)),
                ),
                "session": session,
            }

        sources = self.store.list_sources(session.session_id)
        if not sources and skipped:
            skipped_text = ", ".join(f"`{name}`" for name in skipped)
            session = self.store.update_session(
                session.session_id,
                status="needs_info",
                missing_required_fields=["provenance"],
            )
            return {
                "content": render_conversion_response(
                    self.llm,
                    response_kind="unsupported_files",
                    preferred_language=preferred_language,
                    skipped_text=skipped_text,
                ),
                "session": session,
            }

        if not sources:
            session = self.store.update_session(
                session.session_id,
                status="needs_info",
                missing_required_fields=["provenance"],
            )
            return {
                "content": render_conversion_response(
                    self.llm,
                    response_kind="missing_source",
                    preferred_language=preferred_language,
                ),
                "session": session,
            }

        initial_context = load_shared_context(self.knowledge_root)
        source_bundle = load_source_bundle_text(
            sources,
            shared_context=initial_context,
            existing_package_context="",
            answer_history=session.answer_history,
        )
        first_pass = self._extract_draft(source_bundle)
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
            source_bundle = load_source_bundle_text(
                sources,
                shared_context=second_pass_context,
                existing_package_context=existing_package_context,
                answer_history=session.answer_history,
            )
            draft_payload = normalize_draft_payload(self._extract_draft(source_bundle))
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
            return {
                "content": render_conversion_response(
                    self.llm,
                    response_kind="conflict",
                    preferred_language=preferred_language,
                    conflicts=conflicts,
                    skipped_files=skipped,
                ),
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
            return {
                "content": render_conversion_response(
                    self.llm,
                    response_kind="needs_info",
                    preferred_language=preferred_language,
                    session=session,
                    questions=questions,
                    skipped_files=skipped,
                ),
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
        return {
            "content": render_conversion_response(
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
            ),
            "session": session,
        }

    def _resolve_path(self, configured_value: str, default_value: str) -> Path:
        configured_path = Path(configured_value or default_value).expanduser()
        if configured_path.is_absolute():
            return configured_path.resolve()
        return (PROJECT_ROOT / configured_path).resolve()

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

    def _extract_draft(self, source_bundle: str) -> dict[str, Any]:
        result = self.extractor.invoke(
            [
                SystemMessage(content=CONVERSION_EXTRACTOR_PROMPT),
                HumanMessage(content=source_bundle),
            ]
        )
        if isinstance(result, BaseModel):
            return result.model_dump()
        if hasattr(result, "model_dump"):
            return result.model_dump()
        if isinstance(result, dict):
            return result
        raise RuntimeError("Conversion extractor returned an unexpected payload type.")


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
        if is_approval_intent(text) or is_cancel_intent(text):
            continue
        return detect_response_language(text)

    if session is not None:
        for text in reversed(session.answer_history):
            cleaned = str(text).strip()
            if not cleaned or cleaned == UPLOAD_ONLY_FALLBACK_TEXT:
                continue
            if is_approval_intent(cleaned) or is_cancel_intent(cleaned):
                continue
            return detect_response_language(cleaned)

    latest_text = get_latest_user_text(state)
    if latest_text:
        return detect_response_language(latest_text)
    return "en"


def is_approval_intent(text: str) -> bool:
    normalized = text.strip().lower()
    return bool(normalized) and any(re.match(pattern, normalized) for pattern in APPROVAL_PATTERNS)


def is_cancel_intent(text: str) -> bool:
    normalized = text.strip().lower()
    return bool(normalized) and any(re.match(pattern, normalized) for pattern in CANCEL_PATTERNS)
