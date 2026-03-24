from __future__ import annotations

import csv
import json
import re
import shutil
import sqlite3
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.config import DEFAULT_KNOWLEDGE_BASE_DIR, load_settings
from tools.knowledge_base import load_knowledge_document


SUPPORTED_CONVERSION_FILE_TYPES = (".md", ".txt", ".csv", ".tsv", ".xlsx", ".xlsm")
CONVERSION_ACTIVE_STATUSES = {
    "collecting_source",
    "needs_info",
    "ready_for_approval",
    "blocked_unknown_target",
    "blocked_conflict",
}
CONVERSION_TERMINAL_STATUSES = {"published", "cancelled", "failed"}
OPTIONAL_MODULE_NAMES = ("config", "economy", "localization", "ui", "analytics", "qa")
DEFAULT_CONVERSION_WORK_DIR = "data/conversion"
UPLOAD_ONLY_FALLBACK_TEXT = "Please convert the uploaded document into the canonical knowledge format."

MANIFEST_FIELDNAMES = (
    "package_id",
    "game_slug",
    "market_slug",
    "feature_slug",
    "package_status",
    "inherits_company",
    "inherits_game_shared",
    "approved_revision",
    "completeness_state",
)
FACT_FIELDNAMES = (
    "fact_id",
    "fact_status",
    "supersedes_fact_id",
    "module",
    "subject_type",
    "subject_id",
    "attribute",
    "value_zh",
    "value_en",
    "value_raw",
    "unit",
    "condition",
    "market",
    "source_doc_id",
    "source_locator",
    "confidence",
)
SOURCE_FIELDNAMES = (
    "source_doc_id",
    "upload_ts",
    "slack_file_id",
    "original_name",
    "source_type",
    "author",
    "coverage",
    "raw_path",
    "notes",
)


@dataclass(frozen=True)
class ConversionSourceRecord:
    source_doc_id: str
    session_id: str
    slack_file_id: str
    original_name: str
    source_type: str
    author: str
    coverage: str
    raw_path: str
    notes: str
    upload_ts: str


@dataclass(frozen=True)
class ConversionSessionRecord:
    session_id: str
    thread_id: str
    channel_id: str
    user_id: str
    status: str
    game_slug: str
    market_slug: str
    feature_slug: str
    active_source_ids: list[str]
    missing_required_fields: list[str]
    approval_state: str
    staged_package_path: str
    draft_payload: dict[str, Any]
    answer_history: list[str]
    question_history: list[str]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class StageResult:
    package_path: Path
    populated_modules: list[str]
    missing_optional_modules: list[str]


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def serialize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def deserialize_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def get_conversion_work_root() -> Path:
    settings = load_settings()
    configured_root = Path(getattr(settings, "conversion_work_dir", DEFAULT_CONVERSION_WORK_DIR)).expanduser()
    if configured_root.is_absolute():
        return configured_root.resolve()
    return (Path(__file__).resolve().parent.parent / configured_root).resolve()


def get_knowledge_root() -> Path:
    settings = load_settings()
    configured_root = Path(settings.knowledge_base_dir or DEFAULT_KNOWLEDGE_BASE_DIR).expanduser()
    if configured_root.is_absolute():
        return configured_root.resolve()
    return (Path(__file__).resolve().parent.parent / configured_root).resolve()


def normalize_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_value.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered)
    return lowered.strip("-")


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", value.strip(), flags=re.UNICODE)
    return cleaned.strip("._") or "upload"


def normalize_module_name(value: str) -> str:
    return normalize_slug(value).replace("-", "_")


def is_supported_conversion_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_CONVERSION_FILE_TYPES


def build_conversion_store() -> "ConversionSessionStore":
    return ConversionSessionStore(get_conversion_work_root())


def is_active_conversion_status(status: str) -> bool:
    return status in CONVERSION_ACTIVE_STATUSES


def build_conversion_package_relative_path(game_slug: str, market_slug: str, feature_slug: str) -> str:
    return f"games/{game_slug}/{market_slug}/{feature_slug}"


class ConversionSessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.uploads_dir = self.root / "uploads"
        self.staging_dir = self.root / "staging"
        self.database_path = self.root / "sessions.sqlite3"
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversion_sessions (
                    session_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    game_slug TEXT NOT NULL,
                    market_slug TEXT NOT NULL,
                    feature_slug TEXT NOT NULL,
                    active_source_ids TEXT NOT NULL,
                    missing_required_fields TEXT NOT NULL,
                    approval_state TEXT NOT NULL,
                    staged_package_path TEXT NOT NULL,
                    draft_payload TEXT NOT NULL,
                    answer_history TEXT NOT NULL,
                    question_history TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversion_sessions_thread
                ON conversion_sessions(thread_id, updated_at)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversion_sources (
                    source_doc_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    slack_file_id TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    author TEXT NOT NULL,
                    coverage TEXT NOT NULL,
                    raw_path TEXT NOT NULL,
                    notes TEXT NOT NULL,
                    upload_ts TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversion_sources_session
                ON conversion_sources(session_id, upload_ts)
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def create_session(self, *, thread_id: str, channel_id: str, user_id: str) -> ConversionSessionRecord:
        session_id = uuid4().hex
        timestamp = utc_now_iso()
        record = ConversionSessionRecord(
            session_id=session_id,
            thread_id=thread_id,
            channel_id=channel_id,
            user_id=user_id,
            status="collecting_source",
            game_slug="",
            market_slug="",
            feature_slug="",
            active_source_ids=[],
            missing_required_fields=[],
            approval_state="pending",
            staged_package_path="",
            draft_payload={},
            answer_history=[],
            question_history=[],
            created_at=timestamp,
            updated_at=timestamp,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversion_sessions (
                    session_id,
                    thread_id,
                    channel_id,
                    user_id,
                    status,
                    game_slug,
                    market_slug,
                    feature_slug,
                    active_source_ids,
                    missing_required_fields,
                    approval_state,
                    staged_package_path,
                    draft_payload,
                    answer_history,
                    question_history,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.thread_id,
                    record.channel_id,
                    record.user_id,
                    record.status,
                    record.game_slug,
                    record.market_slug,
                    record.feature_slug,
                    serialize_json(record.active_source_ids),
                    serialize_json(record.missing_required_fields),
                    record.approval_state,
                    record.staged_package_path,
                    serialize_json(record.draft_payload),
                    serialize_json(record.answer_history),
                    serialize_json(record.question_history),
                    record.created_at,
                    record.updated_at,
                ),
            )
            connection.commit()
        return record

    def get_session(self, session_id: str) -> ConversionSessionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversion_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._row_to_session(row)

    def get_active_session_by_thread(self, thread_id: str) -> ConversionSessionRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM conversion_sessions
                WHERE thread_id = ?
                ORDER BY updated_at DESC
                """,
                (thread_id,),
            ).fetchone()
        session = self._row_to_session(row)
        if session is None or not is_active_conversion_status(session.status):
            return None
        return session

    def has_active_session(self, thread_id: str) -> bool:
        return self.get_active_session_by_thread(thread_id) is not None

    def update_session(self, session_id: str, **changes: Any) -> ConversionSessionRecord:
        session = self.get_session(session_id)
        if session is None:
            raise RuntimeError(f"Unknown conversion session: {session_id}")

        merged = {
            "thread_id": session.thread_id,
            "channel_id": session.channel_id,
            "user_id": session.user_id,
            "status": session.status,
            "game_slug": session.game_slug,
            "market_slug": session.market_slug,
            "feature_slug": session.feature_slug,
            "active_source_ids": list(session.active_source_ids),
            "missing_required_fields": list(session.missing_required_fields),
            "approval_state": session.approval_state,
            "staged_package_path": session.staged_package_path,
            "draft_payload": dict(session.draft_payload),
            "answer_history": list(session.answer_history),
            "question_history": list(session.question_history),
            "created_at": session.created_at,
            "updated_at": utc_now_iso(),
        }
        merged.update(changes)

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE conversion_sessions
                SET thread_id = ?,
                    channel_id = ?,
                    user_id = ?,
                    status = ?,
                    game_slug = ?,
                    market_slug = ?,
                    feature_slug = ?,
                    active_source_ids = ?,
                    missing_required_fields = ?,
                    approval_state = ?,
                    staged_package_path = ?,
                    draft_payload = ?,
                    answer_history = ?,
                    question_history = ?,
                    created_at = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (
                    merged["thread_id"],
                    merged["channel_id"],
                    merged["user_id"],
                    merged["status"],
                    merged["game_slug"],
                    merged["market_slug"],
                    merged["feature_slug"],
                    serialize_json(merged["active_source_ids"]),
                    serialize_json(merged["missing_required_fields"]),
                    merged["approval_state"],
                    merged["staged_package_path"],
                    serialize_json(merged["draft_payload"]),
                    serialize_json(merged["answer_history"]),
                    serialize_json(merged["question_history"]),
                    merged["created_at"],
                    merged["updated_at"],
                    session_id,
                ),
            )
            connection.commit()

        updated = self.get_session(session_id)
        if updated is None:
            raise RuntimeError(f"Failed to reload conversion session: {session_id}")
        return updated

    def add_source(
        self,
        session_id: str,
        *,
        slack_file_id: str,
        original_name: str,
        source_type: str,
        author: str,
        coverage: str,
        raw_path: str,
        notes: str = "",
    ) -> ConversionSourceRecord:
        timestamp = utc_now_iso()
        source_doc_id = uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversion_sources (
                    source_doc_id,
                    session_id,
                    slack_file_id,
                    original_name,
                    source_type,
                    author,
                    coverage,
                    raw_path,
                    notes,
                    upload_ts
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_doc_id,
                    session_id,
                    slack_file_id,
                    original_name,
                    source_type,
                    author,
                    coverage,
                    raw_path,
                    notes,
                    timestamp,
                ),
            )
            connection.commit()

        session = self.get_session(session_id)
        if session is None:
            raise RuntimeError(f"Unknown conversion session while adding source: {session_id}")
        active_source_ids = list(session.active_source_ids)
        active_source_ids.append(source_doc_id)
        self.update_session(session_id, active_source_ids=active_source_ids)
        return ConversionSourceRecord(
            source_doc_id=source_doc_id,
            session_id=session_id,
            slack_file_id=slack_file_id,
            original_name=original_name,
            source_type=source_type,
            author=author,
            coverage=coverage,
            raw_path=raw_path,
            notes=notes,
            upload_ts=timestamp,
        )

    def list_sources(self, session_id: str) -> list[ConversionSourceRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM conversion_sources
                WHERE session_id = ?
                ORDER BY upload_ts ASC
                """,
                (session_id,),
            ).fetchall()
        return [self._row_to_source(row) for row in rows]

    def _row_to_session(self, row: sqlite3.Row | None) -> ConversionSessionRecord | None:
        if row is None:
            return None
        return ConversionSessionRecord(
            session_id=str(row["session_id"]),
            thread_id=str(row["thread_id"]),
            channel_id=str(row["channel_id"]),
            user_id=str(row["user_id"]),
            status=str(row["status"]),
            game_slug=str(row["game_slug"]),
            market_slug=str(row["market_slug"]),
            feature_slug=str(row["feature_slug"]),
            active_source_ids=deserialize_json(row["active_source_ids"], []),
            missing_required_fields=deserialize_json(row["missing_required_fields"], []),
            approval_state=str(row["approval_state"]),
            staged_package_path=str(row["staged_package_path"]),
            draft_payload=deserialize_json(row["draft_payload"], {}),
            answer_history=deserialize_json(row["answer_history"], []),
            question_history=deserialize_json(row["question_history"], []),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _row_to_source(self, row: sqlite3.Row) -> ConversionSourceRecord:
        return ConversionSourceRecord(
            source_doc_id=str(row["source_doc_id"]),
            session_id=str(row["session_id"]),
            slack_file_id=str(row["slack_file_id"]),
            original_name=str(row["original_name"]),
            source_type=str(row["source_type"]),
            author=str(row["author"]),
            coverage=str(row["coverage"]),
            raw_path=str(row["raw_path"]),
            notes=str(row["notes"]),
            upload_ts=str(row["upload_ts"]),
        )


def append_answer_to_session(store: ConversionSessionStore, session: ConversionSessionRecord, text: str) -> ConversionSessionRecord:
    cleaned = text.strip()
    if not cleaned or cleaned == UPLOAD_ONLY_FALLBACK_TEXT:
        return session

    answers = list(session.answer_history)
    if answers and answers[-1].strip() == cleaned:
        return session
    answers.append(cleaned)
    return store.update_session(session.session_id, answer_history=answers)


def append_questions_to_session(
    store: ConversionSessionStore,
    session: ConversionSessionRecord,
    questions: list[str],
) -> ConversionSessionRecord:
    history = list(session.question_history)
    history.extend(question for question in questions if question.strip())
    return store.update_session(session.session_id, question_history=history)


def ensure_company_scaffolding(knowledge_root: Path) -> None:
    company_root = knowledge_root / "company"
    company_root.mkdir(parents=True, exist_ok=True)

    terminology_path = company_root / "terminology.tsv"
    if not terminology_path.exists():
        write_tsv(
            terminology_path,
            ("term_id", "canonical_zh", "canonical_en", "aliases", "definition", "approved_wording", "notes"),
            [],
        )

    wording_path = company_root / "wording.md"
    if not wording_path.exists():
        wording_path.write_text(
            "# Company Wording\n\nUse this file for company-level approved wording and reusable phrasing.\n",
            encoding="utf-8",
        )

    concepts_path = company_root / "concepts.md"
    if not concepts_path.exists():
        concepts_path.write_text(
            "# Company Concepts\n\nUse this file for company-level product concepts shared across games.\n",
            encoding="utf-8",
        )


def ensure_game_shared_scaffolding(knowledge_root: Path, game_slug: str) -> None:
    if not game_slug:
        return

    shared_root = knowledge_root / "games" / game_slug / "shared"
    shared_root.mkdir(parents=True, exist_ok=True)

    glossary_path = shared_root / "glossary.tsv"
    if not glossary_path.exists():
        write_tsv(
            glossary_path,
            ("term_id", "canonical_zh", "canonical_en", "aliases", "definition", "related_company_term_id", "notes"),
            [],
        )

    basics_path = shared_root / "game_basics.md"
    if not basics_path.exists():
        basics_path.write_text(
            f"# {game_slug} Shared Basics\n\nUse this file for cross-market knowledge shared by the game.\n",
            encoding="utf-8",
        )

    entities_path = shared_root / "shared_entities.tsv"
    if not entities_path.exists():
        write_tsv(
            entities_path,
            ("entity_id", "canonical_name", "description", "notes"),
            [],
        )

    economy_path = shared_root / "shared_economy.md"
    if not economy_path.exists():
        economy_path.write_text(
            f"# {game_slug} Shared Economy\n\nUse this file for cross-market economy definitions and shared units.\n",
            encoding="utf-8",
        )


def build_upload_directory(store: ConversionSessionStore, session_id: str) -> Path:
    path = store.uploads_dir / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def persist_uploaded_file(
    store: ConversionSessionStore,
    session_id: str,
    uploaded_file: dict[str, Any],
    *,
    slack_bot_token: str,
) -> Path:
    upload_dir = build_upload_directory(store, session_id)

    source_name = str(uploaded_file.get("name") or uploaded_file.get("title") or uploaded_file.get("id") or "upload")
    safe_name = sanitize_filename(source_name)
    destination_path = upload_dir / f"{uuid4().hex}_{safe_name}"

    local_path = str(uploaded_file.get("local_path", "")).strip()
    if local_path:
        shutil.copyfile(local_path, destination_path)
        return destination_path

    download_url = str(uploaded_file.get("url_private_download") or uploaded_file.get("url_private") or "").strip()
    if not download_url:
        raise RuntimeError(f"Uploaded file is missing a downloadable URL: {source_name}")

    request = urllib.request.Request(download_url)
    request.add_header("Authorization", f"Bearer {slack_bot_token}")
    try:
        with urllib.request.urlopen(request) as response, destination_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to download Slack file {source_name}: {exc}") from exc
    return destination_path


def ingest_uploaded_files(
    store: ConversionSessionStore,
    session: ConversionSessionRecord,
    uploaded_files: list[dict[str, Any]],
    *,
    slack_bot_token: str,
    author: str,
) -> tuple[list[ConversionSourceRecord], list[str]]:
    ingested: list[ConversionSourceRecord] = []
    skipped: list[str] = []
    existing = {source.slack_file_id for source in store.list_sources(session.session_id)}

    for uploaded_file in uploaded_files:
        original_name = str(uploaded_file.get("name") or uploaded_file.get("title") or "").strip() or "upload"
        if not is_supported_conversion_file(original_name):
            skipped.append(original_name)
            continue

        slack_file_id = str(uploaded_file.get("id") or original_name).strip()
        if slack_file_id in existing:
            continue

        raw_path = persist_uploaded_file(
            store,
            session.session_id,
            uploaded_file,
            slack_bot_token=slack_bot_token,
        )
        ingested.append(
            store.add_source(
                session.session_id,
                slack_file_id=slack_file_id,
                original_name=original_name,
                source_type=Path(original_name).suffix.lower().lstrip(".") or "file",
                author=author,
                coverage="full",
                raw_path=str(raw_path),
                notes="",
            )
        )
        existing.add(slack_file_id)

    return ingested, skipped


def load_source_bundle_text(
    sources: list[ConversionSourceRecord],
    *,
    shared_context: str,
    existing_package_context: str,
    answer_history: list[str],
) -> str:
    parts: list[str] = []
    if shared_context.strip():
        parts.append("## Shared Context\n" + shared_context.strip())
    if existing_package_context.strip():
        parts.append("## Existing Approved Package\n" + existing_package_context.strip())
    if answer_history:
        answers = "\n".join(f"- {item}" for item in answer_history if item.strip())
        if answers.strip():
            parts.append("## User Clarifications\n" + answers)

    for source in sources:
        source_path = Path(source.raw_path)
        try:
            content = load_knowledge_document(source_path)
        except Exception:
            content = source_path.read_text(encoding="utf-8", errors="replace")
        parts.append(
            "\n".join(
                [
                    f"## Source Document: {source.original_name}",
                    f"Source ID: {source.source_doc_id}",
                    content.strip(),
                ]
            ).strip()
        )

    return "\n\n".join(part for part in parts if part.strip()).strip()


def load_shared_context(knowledge_root: Path, game_slug: str = "") -> str:
    documents: list[str] = []
    company_root = knowledge_root / "company"
    for relative_path in ("terminology.tsv", "wording.md", "concepts.md"):
        path = company_root / relative_path
        if path.exists():
            documents.append(f"# {path.relative_to(knowledge_root).as_posix()}\n{path.read_text(encoding='utf-8', errors='replace').strip()}")

    if game_slug:
        shared_root = knowledge_root / "games" / game_slug / "shared"
        for relative_path in ("glossary.tsv", "game_basics.md", "shared_entities.tsv", "shared_economy.md"):
            path = shared_root / relative_path
            if path.exists():
                if path.suffix.lower() == ".tsv":
                    content = path.read_text(encoding="utf-8", errors="replace").strip()
                else:
                    content = path.read_text(encoding="utf-8", errors="replace").strip()
                documents.append(f"# {path.relative_to(knowledge_root).as_posix()}\n{content}")
    return "\n\n".join(documents).strip()


def load_existing_package_context(knowledge_root: Path, game_slug: str, market_slug: str, feature_slug: str) -> str:
    if not game_slug or not market_slug or not feature_slug:
        return ""
    package_root = knowledge_root / "games" / game_slug / market_slug / feature_slug
    if not package_root.exists():
        return ""

    documents: list[str] = []
    for file_name in (
        "README.md",
        "core.md",
        "facts.tsv",
        "sources.tsv",
        "config.md",
        "economy.md",
        "localization.md",
        "ui.md",
        "analytics.md",
        "qa.md",
    ):
        path = package_root / file_name
        if not path.exists():
            continue
        documents.append(f"# {file_name}\n{path.read_text(encoding='utf-8', errors='replace').strip()}")
    return "\n\n".join(documents).strip()


def build_missing_required_fields(draft_payload: dict[str, Any], sources: list[ConversionSourceRecord]) -> list[str]:
    missing: list[str] = []

    if not draft_payload.get("game_slug"):
        missing.append("game_slug")
    if not draft_payload.get("market_slug"):
        missing.append("market_slug")
    if not draft_payload.get("feature_slug"):
        missing.append("feature_slug")
    if not str(draft_payload.get("overview", "")).strip():
        missing.append("overview")
    if not isinstance(draft_payload.get("terminology"), list) or not draft_payload.get("terminology"):
        missing.append("terminology")
    if not isinstance(draft_payload.get("entities"), list) or not draft_payload.get("entities"):
        missing.append("entities")
    if not isinstance(draft_payload.get("rules"), list) or not draft_payload.get("rules"):
        missing.append("rules")
    if not isinstance(draft_payload.get("config_overview"), list) or not draft_payload.get("config_overview"):
        missing.append("config_overview")
    if not sources:
        missing.append("provenance")
    return missing


def build_targeted_questions(missing_fields: list[str]) -> list[str]:
    question_map = {
        "game_slug": "Which game is this document for? Please give the canonical game name and, if possible, the English slug you want.",
        "market_slug": "Which market or package variant does this feature belong to?",
        "feature_slug": "What should this feature be called in the canonical docs?",
        "overview": "What is the feature goal and user-facing purpose in 1-3 sentences?",
        "terminology": "Which product terms, feature names, rewards, or UI labels must be standardized in the converted package?",
        "entities": "What are the key entities or configurable items for this feature?",
        "rules": "What are the core business rules, reward rules, or progression rules?",
        "config_overview": "Which server-side or client-side config knobs, limits, or switches control this feature?",
        "provenance": "Please upload at least one supported source file for this conversion session.",
    }
    questions: list[str] = []
    for field_name in missing_fields:
        question = question_map.get(field_name)
        if question:
            questions.append(question)
        if len(questions) >= 5:
            break
    return questions


def list_populated_modules(draft_payload: dict[str, Any]) -> list[str]:
    modules = draft_payload.get("modules", [])
    if not isinstance(modules, list):
        return []
    populated: list[str] = []
    for module in modules:
        if not isinstance(module, dict):
            continue
        name = normalize_module_name(str(module.get("name", "")))
        content = str(module.get("content", "")).strip()
        if name in OPTIONAL_MODULE_NAMES and content:
            populated.append(name)
    return sorted(set(populated))


def build_default_fact_rows(draft_payload: dict[str, Any], sources: list[ConversionSourceRecord]) -> list[dict[str, str]]:
    source_doc_id = sources[-1].source_doc_id if sources else ""
    source_locator = sources[-1].original_name if sources else ""
    market = str(draft_payload.get("market_slug", "")).strip()
    rows: list[dict[str, str]] = []

    def append_fact(
        *,
        module: str,
        subject_type: str,
        subject_id: str,
        attribute: str,
        value_zh: str = "",
        value_en: str = "",
        value_raw: str = "",
        unit: str = "",
        condition: str = "",
        confidence: str = "0.7",
    ) -> None:
        rows.append(
            {
                "fact_id": uuid4().hex,
                "fact_status": "active",
                "supersedes_fact_id": "",
                "module": module,
                "subject_type": subject_type,
                "subject_id": subject_id,
                "attribute": attribute,
                "value_zh": value_zh,
                "value_en": value_en,
                "value_raw": value_raw,
                "unit": unit,
                "condition": condition,
                "market": market,
                "source_doc_id": source_doc_id,
                "source_locator": source_locator,
                "confidence": confidence,
            }
        )

    overview = str(draft_payload.get("overview", "")).strip()
    if overview:
        append_fact(
            module="core",
            subject_type="feature",
            subject_id=str(draft_payload.get("feature_slug", "")).strip() or "main",
            attribute="overview",
            value_raw=overview,
        )

    for item in draft_payload.get("terminology", []):
        if not isinstance(item, dict):
            continue
        append_fact(
            module="core",
            subject_type="term",
            subject_id=str(item.get("term_id", "")).strip() or uuid4().hex[:8],
            attribute="definition",
            value_zh=str(item.get("canonical_zh", "")).strip(),
            value_en=str(item.get("canonical_en", "")).strip(),
            value_raw=str(item.get("definition", "")).strip(),
        )

    for item in draft_payload.get("entities", []):
        if not isinstance(item, dict):
            continue
        append_fact(
            module="core",
            subject_type="entity",
            subject_id=str(item.get("entity_id", "")).strip() or uuid4().hex[:8],
            attribute="description",
            value_raw=str(item.get("description", "")).strip(),
            value_zh=str(item.get("name_zh", "")).strip(),
            value_en=str(item.get("name_en", "")).strip(),
        )

    for item in draft_payload.get("rules", []):
        if not isinstance(item, dict):
            continue
        append_fact(
            module="core",
            subject_type="rule",
            subject_id=str(item.get("rule_id", "")).strip() or uuid4().hex[:8],
            attribute="rule",
            value_raw=str(item.get("description", "")).strip(),
            condition=str(item.get("condition", "")).strip(),
            value_zh=str(item.get("title_zh", "")).strip(),
            value_en=str(item.get("title_en", "")).strip(),
        )

    for index, item in enumerate(draft_payload.get("config_overview", []), start=1):
        value = str(item).strip()
        if not value:
            continue
        append_fact(
            module="config",
            subject_type="config",
            subject_id=f"config_{index}",
            attribute="overview",
            value_raw=value,
        )

    for item in draft_payload.get("facts", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "fact_id": uuid4().hex,
                "fact_status": "active",
                "supersedes_fact_id": "",
                "module": str(item.get("module", "core")).strip() or "core",
                "subject_type": str(item.get("subject_type", "fact")).strip() or "fact",
                "subject_id": str(item.get("subject_id", "")).strip() or uuid4().hex[:8],
                "attribute": str(item.get("attribute", "")).strip(),
                "value_zh": str(item.get("value_zh", "")).strip(),
                "value_en": str(item.get("value_en", "")).strip(),
                "value_raw": str(item.get("value_raw", "")).strip(),
                "unit": str(item.get("unit", "")).strip(),
                "condition": str(item.get("condition", "")).strip(),
                "market": market,
                "source_doc_id": source_doc_id,
                "source_locator": source_locator,
                "confidence": str(item.get("confidence", "0.7")).strip() or "0.7",
            }
        )

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        key = (
            row["module"],
            row["subject_type"],
            row["subject_id"],
            row["attribute"],
            row["value_raw"] or row["value_en"] or row["value_zh"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def render_readme_markdown(
    draft_payload: dict[str, Any],
    *,
    relative_package_path: str,
    sources: list[ConversionSourceRecord],
    populated_modules: list[str],
    missing_optional_modules: list[str],
) -> str:
    lines = [
        f"# {draft_payload.get('feature_name') or draft_payload.get('feature_slug') or 'Feature Package'}",
        "",
        f"- Game: `{draft_payload.get('game_slug', '')}`",
        f"- Market: `{draft_payload.get('market_slug', '')}`",
        f"- Feature: `{draft_payload.get('feature_slug', '')}`",
        f"- Package Path: `{relative_package_path}`",
        f"- Source Count: `{len(sources)}`",
        f"- Populated Modules: `{', '.join(populated_modules) or 'core only'}`",
        f"- Missing Optional Modules: `{', '.join(missing_optional_modules) or 'none'}`",
        "",
        "## Summary",
        "",
        str(draft_payload.get("overview", "")).strip() or "No summary available.",
        "",
        "## Sources",
        "",
    ]

    if sources:
        for source in sources:
            lines.append(
                f"- `{source.original_name}` ({source.source_type}) from `{source.upload_ts}`"
            )
    else:
        lines.append("- No sources attached.")

    return "\n".join(lines).strip() + "\n"


def render_core_markdown(draft_payload: dict[str, Any]) -> str:
    lines = [
        f"# {draft_payload.get('feature_name') or draft_payload.get('feature_slug') or 'Feature'}",
        "",
        "## Overview",
        "",
        str(draft_payload.get("overview", "")).strip() or "No overview available yet.",
        "",
        "## Terminology",
        "",
    ]

    terminology = draft_payload.get("terminology", [])
    if isinstance(terminology, list) and terminology:
        for item in terminology:
            if not isinstance(item, dict):
                continue
            label = str(item.get("canonical_zh") or item.get("canonical_en") or item.get("term_id") or "term").strip()
            definition = str(item.get("definition", "")).strip()
            aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
            alias_text = ", ".join(str(alias).strip() for alias in aliases if str(alias).strip())
            english = str(item.get("canonical_en", "")).strip()
            extra_parts = [part for part in (english, alias_text) if part]
            if extra_parts:
                lines.append(f"- `{item.get('term_id') or normalize_slug(label) or 'term'}`: {label} ({'; '.join(extra_parts)})")
            else:
                lines.append(f"- `{item.get('term_id') or normalize_slug(label) or 'term'}`: {label}")
            if definition:
                lines.append(f"  {definition}")
    else:
        lines.append("- No terminology captured yet.")

    lines.extend(["", "## Entities", ""])
    entities = draft_payload.get("entities", [])
    if isinstance(entities, list) and entities:
        for item in entities:
            if not isinstance(item, dict):
                continue
            label = str(item.get("name_zh") or item.get("name_en") or item.get("entity_id") or "entity").strip()
            description = str(item.get("description", "")).strip()
            lines.append(f"- `{item.get('entity_id') or normalize_slug(label) or 'entity'}`: {label}")
            if description:
                lines.append(f"  {description}")
    else:
        lines.append("- No entities captured yet.")

    lines.extend(["", "## Rules", ""])
    rules = draft_payload.get("rules", [])
    if isinstance(rules, list) and rules:
        for index, item in enumerate(rules, start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title_zh") or item.get("title_en") or item.get("rule_id") or f"rule_{index}").strip()
            description = str(item.get("description", "")).strip()
            condition = str(item.get("condition", "")).strip()
            lines.append(f"{index}. {title}")
            if description:
                lines.append(f"   {description}")
            if condition:
                lines.append(f"   Condition: {condition}")
    else:
        lines.append("1. No rules captured yet.")

    lines.extend(["", "## Config Overview", ""])
    config_overview = draft_payload.get("config_overview", [])
    if isinstance(config_overview, list) and config_overview:
        for item in config_overview:
            value = str(item).strip()
            if value:
                lines.append(f"- {value}")
    else:
        lines.append("- No config overview captured yet.")

    lines.extend(["", "## Open Questions / Assumptions", ""])
    open_questions = draft_payload.get("open_questions", [])
    assumptions = draft_payload.get("assumptions", [])
    if isinstance(open_questions, list) and open_questions:
        lines.append("### Open Questions")
        lines.append("")
        for item in open_questions:
            value = str(item).strip()
            if value:
                lines.append(f"- {value}")
        lines.append("")
    if isinstance(assumptions, list) and assumptions:
        lines.append("### Assumptions")
        lines.append("")
        for item in assumptions:
            value = str(item).strip()
            if value:
                lines.append(f"- {value}")
        lines.append("")
    if (not isinstance(open_questions, list) or not open_questions) and (
        not isinstance(assumptions, list) or not assumptions
    ):
        lines.append("- No open questions or assumptions recorded.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_optional_module_markdown(module_name: str, title: str, content: str) -> str:
    body = content.strip() or "No content recorded."
    return f"# {title}\n\n{body}\n"


def write_tsv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [{str(key): str(value or "") for key, value in row.items()} for row in reader]


def read_manifest_revision(path: Path) -> int:
    rows = read_tsv(path)
    if not rows:
        return 0
    try:
        return int(rows[0].get("approved_revision", "0") or "0")
    except ValueError:
        return 0


def stage_conversion_package(
    store: ConversionSessionStore,
    session: ConversionSessionRecord,
    draft_payload: dict[str, Any],
    sources: list[ConversionSourceRecord],
    *,
    knowledge_root: Path,
) -> StageResult:
    stage_root = store.staging_dir / session.session_id
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    game_slug = str(draft_payload.get("game_slug", "")).strip()
    market_slug = str(draft_payload.get("market_slug", "")).strip()
    feature_slug = str(draft_payload.get("feature_slug", "")).strip()
    relative_package_path = build_conversion_package_relative_path(game_slug, market_slug, feature_slug)
    package_path = stage_root / relative_package_path
    package_path.mkdir(parents=True, exist_ok=True)

    populated_modules = list_populated_modules(draft_payload)
    missing_optional_modules = [name for name in OPTIONAL_MODULE_NAMES if name not in populated_modules]

    revision = read_manifest_revision(knowledge_root / relative_package_path / "manifest.tsv")
    manifest_rows = [
        {
            "package_id": f"{game_slug}:{market_slug}:{feature_slug}",
            "game_slug": game_slug,
            "market_slug": market_slug,
            "feature_slug": feature_slug,
            "package_status": "draft",
            "inherits_company": "true",
            "inherits_game_shared": "true",
            "approved_revision": str(revision + 1),
            "completeness_state": "ready_for_approval",
        }
    ]
    write_tsv(package_path / "manifest.tsv", MANIFEST_FIELDNAMES, manifest_rows)
    write_tsv(
        package_path / "sources.tsv",
        SOURCE_FIELDNAMES,
        [
            {
                "source_doc_id": source.source_doc_id,
                "upload_ts": source.upload_ts,
                "slack_file_id": source.slack_file_id,
                "original_name": source.original_name,
                "source_type": source.source_type,
                "author": source.author,
                "coverage": source.coverage,
                "raw_path": source.raw_path,
                "notes": source.notes,
            }
            for source in sources
        ],
    )

    facts = build_default_fact_rows(draft_payload, sources)
    write_tsv(package_path / "facts.tsv", FACT_FIELDNAMES, facts)
    (package_path / "README.md").write_text(
        render_readme_markdown(
            draft_payload,
            relative_package_path=relative_package_path,
            sources=sources,
            populated_modules=populated_modules,
            missing_optional_modules=missing_optional_modules,
        ),
        encoding="utf-8",
    )
    (package_path / "core.md").write_text(render_core_markdown(draft_payload), encoding="utf-8")

    module_title_map = {
        "config": "Config",
        "economy": "Economy",
        "localization": "Localization",
        "ui": "UI",
        "analytics": "Analytics",
        "qa": "QA",
    }
    modules = draft_payload.get("modules", [])
    if isinstance(modules, list):
        for module in modules:
            if not isinstance(module, dict):
                continue
            module_name = normalize_module_name(str(module.get("name", "")))
            content = str(module.get("content", "")).strip()
            if module_name not in OPTIONAL_MODULE_NAMES or not content:
                continue
            (package_path / f"{module_name}.md").write_text(
                render_optional_module_markdown(module_name, module_title_map[module_name], content),
                encoding="utf-8",
            )

    return StageResult(
        package_path=package_path,
        populated_modules=populated_modules,
        missing_optional_modules=missing_optional_modules,
    )


def _fact_identity(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("module", ""),
        row.get("subject_type", ""),
        row.get("subject_id", ""),
        row.get("attribute", ""),
        row.get("market", ""),
    )


def _fact_value_signature(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("value_zh", ""),
        row.get("value_en", ""),
        row.get("value_raw", ""),
        row.get("unit", ""),
        row.get("condition", ""),
    )


def merge_fact_rows(existing_rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    merged = list(existing_rows)
    for new_row in new_rows:
        new_row = {field: str(new_row.get(field, "")) for field in FACT_FIELDNAMES}
        target_key = _fact_identity(new_row)
        active_match: dict[str, str] | None = None
        for existing_row in reversed(merged):
            if existing_row.get("fact_status") != "active":
                continue
            if _fact_identity(existing_row) == target_key:
                active_match = existing_row
                break

        if active_match is None:
            merged.append(new_row)
            continue

        if _fact_value_signature(active_match) == _fact_value_signature(new_row):
            continue

        active_match["fact_status"] = "superseded"
        new_row["supersedes_fact_id"] = active_match.get("fact_id", "")
        merged.append(new_row)

    return merged


def append_source_rows(existing_rows: list[dict[str, str]], new_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    merged = list(existing_rows)
    seen = {row.get("source_doc_id", "") for row in existing_rows}
    for row in new_rows:
        source_id = row.get("source_doc_id", "")
        if source_id in seen:
            continue
        merged.append(row)
        seen.add(source_id)
    return merged


def publish_conversion_package(
    store: ConversionSessionStore,
    session: ConversionSessionRecord,
    *,
    knowledge_root: Path,
) -> str:
    draft_payload = dict(session.draft_payload)
    if not draft_payload:
        raise RuntimeError("No staged conversion draft is available for publishing.")

    game_slug = str(draft_payload.get("game_slug", "")).strip()
    market_slug = str(draft_payload.get("market_slug", "")).strip()
    feature_slug = str(draft_payload.get("feature_slug", "")).strip()
    relative_package_path = build_conversion_package_relative_path(game_slug, market_slug, feature_slug)
    package_root = knowledge_root / relative_package_path
    package_root.mkdir(parents=True, exist_ok=True)

    ensure_company_scaffolding(knowledge_root)
    ensure_game_shared_scaffolding(knowledge_root, game_slug)

    sources = store.list_sources(session.session_id)
    new_source_rows = [
        {
            "source_doc_id": source.source_doc_id,
            "upload_ts": source.upload_ts,
            "slack_file_id": source.slack_file_id,
            "original_name": source.original_name,
            "source_type": source.source_type,
            "author": source.author,
            "coverage": source.coverage,
            "raw_path": source.raw_path,
            "notes": source.notes,
        }
        for source in sources
    ]
    merged_sources = append_source_rows(read_tsv(package_root / "sources.tsv"), new_source_rows)
    write_tsv(package_root / "sources.tsv", SOURCE_FIELDNAMES, merged_sources)

    new_facts = build_default_fact_rows(draft_payload, sources)
    merged_facts = merge_fact_rows(read_tsv(package_root / "facts.tsv"), new_facts)
    write_tsv(package_root / "facts.tsv", FACT_FIELDNAMES, merged_facts)

    previous_revision = read_manifest_revision(package_root / "manifest.tsv")
    manifest_rows = [
        {
            "package_id": f"{game_slug}:{market_slug}:{feature_slug}",
            "game_slug": game_slug,
            "market_slug": market_slug,
            "feature_slug": feature_slug,
            "package_status": "approved",
            "inherits_company": "true",
            "inherits_game_shared": "true",
            "approved_revision": str(previous_revision + 1),
            "completeness_state": "published",
        }
    ]
    write_tsv(package_root / "manifest.tsv", MANIFEST_FIELDNAMES, manifest_rows)

    populated_modules = list_populated_modules(draft_payload)
    existing_optional_modules = [name for name in OPTIONAL_MODULE_NAMES if (package_root / f"{name}.md").exists()]
    all_populated_modules = sorted(set(populated_modules + existing_optional_modules))
    missing_optional_modules = [name for name in OPTIONAL_MODULE_NAMES if name not in all_populated_modules]

    (package_root / "README.md").write_text(
        render_readme_markdown(
            draft_payload,
            relative_package_path=relative_package_path,
            sources=sources,
            populated_modules=all_populated_modules,
            missing_optional_modules=missing_optional_modules,
        ),
        encoding="utf-8",
    )
    (package_root / "core.md").write_text(render_core_markdown(draft_payload), encoding="utf-8")

    module_title_map = {
        "config": "Config",
        "economy": "Economy",
        "localization": "Localization",
        "ui": "UI",
        "analytics": "Analytics",
        "qa": "QA",
    }
    modules = draft_payload.get("modules", [])
    if isinstance(modules, list):
        for module in modules:
            if not isinstance(module, dict):
                continue
            module_name = normalize_module_name(str(module.get("name", "")))
            content = str(module.get("content", "")).strip()
            if module_name not in OPTIONAL_MODULE_NAMES or not content:
                continue
            (package_root / f"{module_name}.md").write_text(
                render_optional_module_markdown(module_name, module_title_map[module_name], content),
                encoding="utf-8",
            )

    store.update_session(
        session.session_id,
        status="published",
        approval_state="approved",
        staged_package_path=str(package_root),
    )
    return relative_package_path
