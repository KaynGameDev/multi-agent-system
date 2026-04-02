from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from tools.document_conversion import ConversionSessionRecord, build_conversion_package_relative_path

logger = logging.getLogger(__name__)

CONVERSION_RENDERER_PROMPT = """You are a response formatter for a document-conversion workflow.
You do not decide workflow state. You only turn the provided JSON payload into the final user-facing reply.

Rules:
- Match the language requested by `preferred_language`. If it is `zh`, reply in Chinese. If it is `en`, reply in English.
- Keep technical literals exact. Do not alter paths, slugs, field names, filenames, counts, IDs, or command words.
- Any item listed in `render_rules.must_include_verbatim` must appear exactly as written.
- Do not invent workflow status, facts, or instructions that are not present in the payload.
- Keep the reply concise and Slack-friendly. Short paragraphs and flat lists are fine.
- Return only the final reply text.

Example input:
{
  "response_kind": "needs_info",
  "preferred_language": "en",
  "payload": {
    "target_path": "games/buyudalouandou/?/lucky-bundle",
    "missing_fields": ["market_slug"],
    "questions": ["Which market or package variant does this feature belong to?"]
  },
  "render_rules": {
    "must_include_verbatim": ["`market_slug`"]
  }
}

Example output:
I need a bit more information before I can stage the canonical package.

- Current target: `games/buyudalouandou/?/lucky-bundle`
- Missing required fields: `market_slug`

Please reply in this thread with:
1. Which market or package variant does this feature belong to?

Example input:
{
  "response_kind": "ready_for_approval",
  "preferred_language": "zh",
  "payload": {
    "target_path": "games/buyudalouandou/indonesia/weekly-activity",
    "source_count": 1,
    "populated_modules": ["config"],
    "missing_optional_modules": ["economy", "ui"],
    "major_facts": ["周常活动通过每周目标和奖励提升玩家活跃与付费。"]
  },
  "render_rules": {
    "must_include_verbatim": ["`games/buyudalouandou/indonesia/weekly-activity`", "`approve`", "`cancel`"]
  }
}

Example output:
规范知识包已准备好，等待确认。

- 目标路径: `games/buyudalouandou/indonesia/weekly-activity`
- 来源文件数: `1`
- 已填充模块: `config`
- 缺失的可选模块: `economy, ui`

主要提取事实：
1. 周常活动通过每周目标和奖励提升玩家活跃与付费。

请回复 `approve` 发布该知识包，或回复 `cancel` 丢弃本次会话。
"""


def render_conversion_response(
    llm,
    *,
    response_kind: str,
    preferred_language: str = "en",
    **context: Any,
) -> str:
    fallback = render_conversion_response_fallback(
        response_kind,
        preferred_language=preferred_language,
        **context,
    )
    if llm is None or not hasattr(llm, "invoke"):
        return fallback

    payload = build_render_payload(
        response_kind,
        preferred_language=preferred_language,
        **context,
    )
    try:
        response = llm.invoke(
            [
                SystemMessage(content=CONVERSION_RENDERER_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
            ]
        )
    except Exception:
        logger.debug(
            "LLM conversion renderer failed response_kind=%s preferred_language=%s",
            response_kind,
            preferred_language,
            exc_info=True,
        )
        return fallback

    rendered = extract_rendered_text(response)
    if not rendered:
        return fallback
    if not validate_rendered_text(rendered, payload.get("render_rules", {})):
        logger.debug(
            "LLM conversion renderer returned invalid output response_kind=%s preferred_language=%s",
            response_kind,
            preferred_language,
        )
        return fallback
    return rendered


def render_conversion_response_fallback(
    response_kind: str,
    *,
    preferred_language: str = "en",
    **context: Any,
) -> str:
    if response_kind == "needs_info":
        return build_needs_info_response(
            context["session"],
            context["questions"],
            context.get("skipped_files", []),
            preferred_language,
        )
    if response_kind == "download_failure":
        return build_download_failure_response(context["download_failures"], preferred_language)
    if response_kind == "failure":
        return build_conversion_failure_response(context["exc"], preferred_language)
    if response_kind == "conflict":
        return build_conflict_response(
            context["conflicts"],
            context.get("skipped_files", []),
            preferred_language,
        )
    if response_kind == "ready_for_approval":
        return build_ready_for_approval_response(
            context["session"],
            sources=context["sources"],
            draft_payload=context["draft_payload"],
            populated_modules=context["populated_modules"],
            missing_optional_modules=context["missing_optional_modules"],
            skipped_files=context.get("skipped_files", []),
            download_failures=context.get("download_failures"),
            preferred_language=preferred_language,
        )
    if response_kind == "missing_session":
        return build_missing_session_response(preferred_language)
    if response_kind == "cancelled":
        return build_cancelled_response(preferred_language)
    if response_kind == "not_ready_for_publish":
        return build_not_ready_for_publish_response(context["missing"], preferred_language)
    if response_kind == "published":
        return build_published_response(
            context["relative_package_path"],
            context["source_count"],
            preferred_language,
        )
    if response_kind == "unsupported_files":
        return build_unsupported_files_response(context["skipped_text"], preferred_language)
    if response_kind == "missing_source":
        return build_missing_source_response(preferred_language)
    raise ValueError(f"Unsupported conversion response kind: {response_kind}")


def build_render_payload(
    response_kind: str,
    *,
    preferred_language: str = "en",
    **context: Any,
) -> dict[str, Any]:
    if response_kind == "needs_info":
        session: ConversionSessionRecord = context["session"]
        payload = {
            "target_path": (
                build_conversion_package_relative_path(
                    session.game_slug or "?",
                    session.market_slug or "?",
                    session.feature_slug or "?",
                )
                if session.game_slug or session.market_slug or session.feature_slug
                else ""
            ),
            "missing_fields": list(session.missing_required_fields),
            "skipped_files": list(context.get("skipped_files", [])),
            "questions": list(context["questions"]),
        }
        rules = {
            "must_include_verbatim": [f"`{field}`" for field in session.missing_required_fields],
        }
    elif response_kind == "download_failure":
        payload = {
            "download_failures": list(context["download_failures"]),
            "recovery_options": [
                "re-upload the file and try again",
                "paste the relevant content into the thread",
                "share the Google Doc or Sheet with the bot service account if the source is hosted on Google Docs",
                "export locally and upload again",
            ],
        }
        rules = {"must_include_verbatim": []}
    elif response_kind == "failure":
        exc = context["exc"]
        payload = {
            "failure_kind": classify_conversion_failure(exc),
            "error_detail": str(exc),
        }
        rules = {"must_include_verbatim": [f"`{exc}`"] if classify_conversion_failure(exc) == "proxy_transport" else []}
    elif response_kind == "conflict":
        payload = {
            "conflicts": list(context["conflicts"]),
            "skipped_files": list(context.get("skipped_files", [])),
        }
        rules = {"must_include_verbatim": []}
    elif response_kind == "ready_for_approval":
        session = context["session"]
        relative_path = build_conversion_package_relative_path(
            session.game_slug,
            session.market_slug,
            session.feature_slug,
        )
        payload = {
            "target_path": relative_path,
            "source_count": len(context["sources"]),
            "populated_modules": list(context["populated_modules"]),
            "missing_optional_modules": list(context["missing_optional_modules"]),
            "skipped_files": list(context.get("skipped_files", [])),
            "download_failures": list(context.get("download_failures") or []),
            "major_facts": build_major_fact_summary(context["draft_payload"]),
            "control_commands": ["approve", "cancel"],
        }
        rules = {
            "must_include_verbatim": [f"`{relative_path}`", "`approve`", "`cancel`"],
        }
    elif response_kind == "missing_session":
        payload = {
            "accepted_types": [".md", ".txt", ".csv", ".tsv", ".xlsx", ".xlsm"],
            "accepted_online_sources": ["Google Docs URL", "Google Sheets URL"],
        }
        rules = {"must_include_verbatim": []}
    elif response_kind == "cancelled":
        payload = {"status": "cancelled"}
        rules = {"must_include_verbatim": []}
    elif response_kind == "not_ready_for_publish":
        payload = {
            "missing": context["missing"],
            "control_commands": ["approve", "cancel"],
        }
        rules = {"must_include_verbatim": []}
    elif response_kind == "published":
        payload = {
            "target_path": context["relative_package_path"],
            "source_count": context["source_count"],
        }
        rules = {"must_include_verbatim": [f"`{context['relative_package_path']}`"]}
    elif response_kind == "unsupported_files":
        payload = {
            "skipped_files": [context["skipped_text"]],
            "accepted_types": [".md", ".txt", ".csv", ".tsv", ".xlsx", ".xlsm"],
            "accepted_online_sources": ["Google Docs URL", "Google Sheets URL"],
        }
        rules = {"must_include_verbatim": []}
    elif response_kind == "missing_source":
        payload = {
            "accepted_types": [".md", ".txt", ".csv", ".tsv", ".xlsx", ".xlsm"],
            "accepted_online_sources": ["Google Docs URL", "Google Sheets URL"],
        }
        rules = {"must_include_verbatim": []}
    else:
        raise ValueError(f"Unsupported conversion response kind: {response_kind}")

    return {
        "response_kind": response_kind,
        "preferred_language": preferred_language,
        "payload": payload,
        "render_rules": rules,
    }


def extract_rendered_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text.strip())
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def validate_rendered_text(text: str, render_rules: dict[str, Any]) -> bool:
    required = render_rules.get("must_include_verbatim")
    if not isinstance(required, list):
        return True
    return all(isinstance(item, str) and item in text for item in required)


def build_targeted_questions(missing_fields: list[str], preferred_language: str = "en") -> list[str]:
    if preferred_language == "zh":
        question_map = {
            "game_slug": "这份文档对应哪个游戏？请给出规范游戏名，如果可以的话也请提供你希望使用的英文 slug。",
            "market_slug": "这个功能属于哪个市场或包体版本？",
            "feature_slug": "这个功能在规范文档里应该叫什么？",
            "overview": "请用 1-3 句话说明这个功能的目标和面向用户的用途。",
            "terminology": "这个功能里有哪些产品术语、奖励名、功能名或 UI 文案需要统一？",
            "entities": "这个功能里的关键实体或可配置项有哪些？",
            "rules": "这个功能的核心业务规则、奖励规则或成长规则是什么？",
            "config_overview": "这个功能由哪些服务端或客户端配置项、限制或开关控制？",
            "provenance": "请至少上传一个支持的源文件，或直接贴一个可访问的 Google Docs / Sheets 链接来继续这个转换会话。",
        }
    else:
        question_map = {
            "game_slug": "Which game is this document for? Please give the canonical game name and, if possible, the English slug you want.",
            "market_slug": "Which market or package variant does this feature belong to?",
            "feature_slug": "What should this feature be called in the canonical docs?",
            "overview": "What is the feature goal and user-facing purpose in 1-3 sentences?",
            "terminology": "Which product terms, feature names, rewards, or UI labels must be standardized in the converted package?",
            "entities": "What are the key entities or configurable items for this feature?",
            "rules": "What are the core business rules, reward rules, or progression rules?",
            "config_overview": "Which server-side or client-side config knobs, limits, or switches control this feature?",
            "provenance": "Please upload at least one supported source file, or paste an accessible Google Docs / Sheets URL for this conversion session.",
        }

    questions: list[str] = []
    for field_name in missing_fields:
        question = question_map.get(field_name)
        if question:
            questions.append(question)
        if len(questions) >= 5:
            break
    return questions


def build_needs_info_response(
    session: ConversionSessionRecord,
    questions: list[str],
    skipped_files: list[str],
    preferred_language: str = "en",
) -> str:
    if preferred_language == "zh":
        lines = ["在暂存规范知识包之前，我还需要一些信息。"]
    else:
        lines = ["I need a bit more information before I can stage the canonical package."]

    if session.game_slug or session.market_slug or session.feature_slug:
        lines.extend(
            [
                "",
                (
                    f"- 当前目标: `{build_conversion_package_relative_path(session.game_slug or '?', session.market_slug or '?', session.feature_slug or '?')}`"
                    if preferred_language == "zh"
                    else f"- Current target: `{build_conversion_package_relative_path(session.game_slug or '?', session.market_slug or '?', session.feature_slug or '?')}`"
                ),
            ]
        )

    if session.missing_required_fields:
        label = "缺少必填字段" if preferred_language == "zh" else "Missing required fields"
        lines.append(f"- {label}: `{', '.join(session.missing_required_fields)}`")

    if skipped_files:
        label = "已跳过的不支持文件" if preferred_language == "zh" else "Skipped unsupported files"
        lines.append(f"- {label}: `{', '.join(skipped_files)}`")

    if questions:
        prompt = "请直接在这个线程回复：" if preferred_language == "zh" else "Please reply in this thread with:"
        lines.extend(["", prompt])
        for index, question in enumerate(questions, start=1):
            lines.append(f"{index}. {question}")

    return "\n".join(lines).strip()


def build_download_failure_response(download_failures: list[str], preferred_language: str = "en") -> str:
    if preferred_language == "zh":
        lines = [
            "我无法访问你提供的源文档，所以现在还不能开始转换。",
            "",
            "访问错误：",
        ]
    else:
        lines = [
            "I couldn't access one or more source documents, so I can't start the conversion yet.",
            "",
            "Access errors:",
        ]
    for index, item in enumerate(download_failures, start=1):
        lines.append(f"{index}. {item}")
    if preferred_language == "zh":
        lines.extend(
            [
                "",
                "你可以尝试：",
                "1. 重新上传文件后再试。",
                "2. 直接把相关内容粘贴到线程里。",
                "3. 如果是 Google Docs / Sheets，请确认已经把文档共享给 bot 使用的服务账号。",
                "4. 如果 Slack 文件下载仍然失败，先在本地导出表格再重新上传。",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Please try one of these:",
                "1. Re-upload the file and try again.",
                "2. Paste the relevant content into the thread.",
                "3. If this is a Google Doc or Sheet, make sure it is shared with the bot's service account.",
                "4. Export the sheet locally and upload it again if Slack file download continues to fail.",
            ]
        )
    return "\n".join(lines).strip()


def build_conversion_failure_response(exc: Exception, preferred_language: str = "en") -> str:
    failure_kind = classify_conversion_failure(exc)
    if failure_kind == "proxy_transport":
        if preferred_language == "zh":
            return "\n".join(
                [
                    "我无法连接 Gemini API 来处理这个转换会话，因为 HTTP 代理返回了 `503 Service Unavailable`。",
                    "",
                    "请检查以下任一项：",
                    "1. 如果这个 bot 应该直连外网，请设置 `GEMINI_HTTP_TRUST_ENV=false` 后重启。",
                    "2. 如果必须走代理，请修复 `HTTPS_PROXY` / `ALL_PROXY` 到 Gemini 的出口。",
                    "3. 网络恢复后再重试这个转换。",
                    "",
                    f"错误详情：`{exc}`",
                ]
            ).strip()
        return "\n".join(
            [
                "I couldn't reach the Gemini API for this conversion session because the HTTP proxy returned `503 Service Unavailable`.",
                "",
                "Please check one of these:",
                "1. If this bot should connect directly, set `GEMINI_HTTP_TRUST_ENV=false` and restart it.",
                "2. If a proxy is required, fix `HTTPS_PROXY` / `ALL_PROXY` for outbound Gemini traffic.",
                "3. Retry the conversion after the network path is healthy.",
                "",
                f"Error detail: `{exc}`",
            ]
        ).strip()

    if failure_kind == "transport_disconnect":
        if preferred_language == "zh":
            return "\n".join(
                [
                    "我已经读取到源文档，但在把内容发送给 Gemini 做结构化提取时，连接在返回结果前中断了。",
                    "",
                    "这通常是临时性的模型网络问题，或者源内容过大导致上游提前断开，而不是文档权限问题。",
                    "",
                    "你可以这样处理：",
                    "1. 直接再重试一次转换。",
                    "2. 如果反复发生，请先缩小文档范围，或把最关键的 sheet / 段落单独发来。",
                    "3. 如果环境必须经过代理，请继续检查 Gemini 的外网链路是否稳定。",
                    "",
                    f"错误详情：`{exc}`",
                ]
            ).strip()
        return "\n".join(
            [
                "I could read the source document, but the connection dropped while Gemini was generating the structured extraction result.",
                "",
                "This usually means a transient model/network issue, or that the source bundle was large enough for the upstream request to get cut off. It is not the same as a document-permission failure.",
                "",
                "Please try one of these:",
                "1. Retry the conversion once.",
                "2. If it keeps happening, narrow the document or send the most important sheet/tab first.",
                "3. If this environment must use a proxy, keep checking that outbound Gemini traffic is stable.",
                "",
                f"Error detail: `{exc}`",
            ]
        ).strip()

    if preferred_language == "zh":
        return f"处理这个转换会话时出错：{exc}"
    return f"I hit an error while processing this conversion session: {exc}"


def classify_conversion_failure(exc: Exception) -> str:
    if is_proxy_transport_error(exc):
        return "proxy_transport"
    if is_transport_disconnect_error(exc):
        return "transport_disconnect"
    return "unexpected"


def is_retryable_conversion_failure(exc: Exception) -> bool:
    return is_transport_disconnect_error(exc)


def is_proxy_transport_error(exc: Exception) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        error_type = type(current).__name__.lower()
        error_text = str(current).lower()
        if "proxyerror" in error_type:
            return True
        if "tunnel connection failed" in error_text:
            return True
        current = current.__cause__ or current.__context__

    return False


def is_transport_disconnect_error(exc: Exception) -> bool:
    disconnect_markers = (
        "server disconnected without sending a response",
        "connection reset by peer",
        "connection aborted",
        "remote end closed connection without response",
        "connection closed before full response",
        "connection terminated unexpectedly",
        "read timed out",
        "timed out",
        "eof occurred in violation of protocol",
    )
    disconnect_types = (
        "remoteprotocolerror",
        "readtimeout",
        "connecttimeout",
        "readerror",
        "connecterror",
        "apiconnectionerror",
        "serversdisconnectederror",
        "serversdisconnected",
    )

    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        error_type = type(current).__name__.lower()
        error_text = str(current).lower()
        if any(marker in error_type for marker in disconnect_types):
            return True
        if any(marker in error_text for marker in disconnect_markers):
            return True
        current = current.__cause__ or current.__context__

    return False


def build_conflict_response(
    conflicts: list[str],
    skipped_files: list[str],
    preferred_language: str = "en",
) -> str:
    if preferred_language == "zh":
        lines = [
            "我在这个转换会话里发现了冲突信息，因此在澄清之前不会发布。",
            "",
        ]
    else:
        lines = [
            "I found conflicting information in this conversion session, so I'm blocking publication until the conflict is clarified.",
            "",
        ]
    for index, item in enumerate(conflicts, start=1):
        lines.append(f"{index}. {item}")
    if skipped_files:
        lines.extend(
            [
                "",
                (
                    f"已跳过的不支持文件：`{', '.join(skipped_files)}`"
                    if preferred_language == "zh"
                    else f"Skipped unsupported files: `{', '.join(skipped_files)}`"
                ),
            ]
        )
    return "\n".join(lines).strip()


def build_ready_for_approval_response(
    session: ConversionSessionRecord,
    *,
    sources: list,
    draft_payload: dict[str, Any],
    populated_modules: list[str],
    missing_optional_modules: list[str],
    skipped_files: list[str],
    download_failures: list[str] | None = None,
    preferred_language: str = "en",
) -> str:
    relative_path = build_conversion_package_relative_path(
        session.game_slug,
        session.market_slug,
        session.feature_slug,
    )
    populated_summary = ", ".join(populated_modules) or ("仅 core" if preferred_language == "zh" else "core only")
    missing_summary = ", ".join(missing_optional_modules) or ("无" if preferred_language == "zh" else "none")
    if preferred_language == "zh":
        lines = [
            "规范知识包已准备好，等待确认。",
            "",
            f"- 目标路径: `{relative_path}`",
            f"- 来源文件数: `{len(sources)}`",
            f"- 已填充模块: `{populated_summary}`",
            f"- 缺失的可选模块: `{missing_summary}`",
        ]
    else:
        lines = [
            "The canonical package is ready for approval.",
            "",
            f"- Target path: `{relative_path}`",
            f"- Sources: `{len(sources)}`",
            f"- Populated modules: `{populated_summary}`",
            f"- Missing optional modules: `{missing_summary}`",
        ]

    if skipped_files:
        label = "已跳过的不支持文件" if preferred_language == "zh" else "Skipped unsupported files"
        lines.append(f"- {label}: `{', '.join(skipped_files)}`")
    if download_failures:
        label = "下载失败" if preferred_language == "zh" else "Failed downloads"
        lines.append(f"- {label}: `{'; '.join(download_failures)}`")

    major_facts = build_major_fact_summary(draft_payload)
    if major_facts:
        heading = "主要提取事实：" if preferred_language == "zh" else "Major extracted facts:"
        lines.extend(["", heading])
        for index, item in enumerate(major_facts, start=1):
            lines.append(f"{index}. {item}")

    instruction = (
        "请回复 `approve`（或“批准”）发布该知识包，或回复 `cancel`（或“取消”）丢弃本次会话。"
        if preferred_language == "zh"
        else 'Reply with `approve` to publish this package, or `cancel` to discard the session.'
    )
    lines.extend(["", instruction])
    return "\n".join(lines).strip()


def build_major_fact_summary(draft_payload: dict[str, Any]) -> list[str]:
    summary: list[str] = []
    overview = str(draft_payload.get("overview", "")).strip()
    if overview:
        summary.append(overview)

    rules = draft_payload.get("rules", [])
    if isinstance(rules, list):
        for item in rules[:2]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title_zh") or item.get("title_en") or item.get("rule_id") or "").strip()
            description = str(item.get("description", "")).strip()
            if title and description:
                summary.append(f"{title}: {description}")
            elif description:
                summary.append(description)

    config_overview = draft_payload.get("config_overview", [])
    if isinstance(config_overview, list):
        for item in config_overview[:2]:
            value = str(item).strip()
            if value:
                summary.append(value)
    return summary[:5]


def build_missing_session_response(preferred_language: str = "en") -> str:
    if preferred_language == "zh":
        return (
            "我没有找到这个线程里的活动转换会话。请先上传一个支持的文档：`.md`、`.txt`、`.csv`、`.tsv`、`.xlsx`、`.xlsm`，"
            "或者直接贴一个可访问的 Google Docs / Sheets 链接。"
        )
    return (
        "I couldn't find an active conversion session for this thread. "
        "Upload a supported document first: `.md`, `.txt`, `.csv`, `.tsv`, `.xlsx`, or `.xlsm`, "
        "or paste an accessible Google Docs / Sheets URL."
    )


def build_cancelled_response(preferred_language: str = "en") -> str:
    if preferred_language == "zh":
        return "这个转换会话已取消。准备好后重新上传新文档即可开始。"
    return "This conversion session has been cancelled. Upload a new document when you want to start again."


def build_not_ready_for_publish_response(missing: str, preferred_language: str = "en") -> str:
    if preferred_language == "zh":
        return f"这个转换还不能发布。我当前仍然卡在：{missing}。"
    return f"This conversion is not ready to publish yet. I'm still blocked on: {missing}."


def build_published_response(relative_package_path: str, source_count: int, preferred_language: str = "en") -> str:
    if preferred_language == "zh":
        return (
            "已发布规范知识包。\n\n"
            f"- 路径: `{relative_package_path}`\n"
            f"- 来源文件: `{source_count}`\n"
            "知识 agent 现在可以读取这个知识包。"
        )
    return (
        "Published the canonical package.\n\n"
        f"- Path: `{relative_package_path}`\n"
        f"- Sources: `{source_count}`\n"
        "The package is now available to the knowledge agent."
    )


def build_unsupported_files_response(skipped_text: str, preferred_language: str = "en") -> str:
    if preferred_language == "zh":
        return (
            "我无法开始转换，因为上传的文件类型不受支持。\n\n"
            f"- 已跳过: {skipped_text}\n"
            "- 支持类型: `.md`, `.txt`, `.csv`, `.tsv`, `.xlsx`, `.xlsm`\n"
            "- 或直接贴一个可访问的 Google Docs / Sheets 链接"
        )
    return (
        "I couldn't start the conversion because the uploaded files are not supported.\n\n"
        f"- Skipped: {skipped_text}\n"
        "- Accepted types: `.md`, `.txt`, `.csv`, `.tsv`, `.xlsx`, `.xlsm`\n"
        "- Or paste an accessible Google Docs / Sheets URL"
    )


def build_missing_source_response(preferred_language: str = "en") -> str:
    if preferred_language == "zh":
        return (
            "在构建规范知识包之前，我至少需要一个支持的源文件。\n\n"
            "- 支持类型: `.md`, `.txt`, `.csv`, `.tsv`, `.xlsx`, `.xlsm`\n"
            "- 也支持可访问的 Google Docs / Sheets 链接"
        )
    return (
        "I need at least one supported source file before I can build the canonical package.\n\n"
        "- Accepted types: `.md`, `.txt`, `.csv`, `.tsv`, `.xlsx`, `.xlsm`\n"
        "- Also accepted: accessible Google Docs / Sheets URLs"
    )
