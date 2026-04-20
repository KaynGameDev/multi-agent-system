from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from app.config import Settings, load_settings
from app.paths import resolve_project_path

_LAYER_DIRECTORIES = (
    ("00_Shared", "Shared"),
    ("10_GameLines", "GameLine"),
    ("20_Deployments", "Deployment"),
    ("30_Review", "Review"),
    ("40_Legacy", "Legacy"),
    ("50_Templates", "Templates"),
)
_SECTION_PATTERN = re.compile(r"^(#{2,6})\s+(.*)$")


@dataclass(frozen=True)
class KnowledgeBaseStatusSnapshot:
    knowledge_root: Path
    docs_root: Path
    total_markdown_docs: int
    layer_counts: dict[str, int]
    template_names: tuple[str, ...]
    current_stage: str
    blockers: tuple[str, ...]
    next_steps: tuple[str, ...]
    tracker_states: dict[str, str]
    priority_gaps: tuple[str, ...]


def build_knowledge_base_status_prompt_context(settings: Settings | None = None) -> str:
    snapshot = scan_knowledge_base_status(settings=settings)
    return render_knowledge_base_status_prompt(snapshot)


def scan_knowledge_base_status(settings: Settings | None = None) -> KnowledgeBaseStatusSnapshot:
    resolved_settings = settings or load_settings()
    knowledge_root = resolve_project_path(resolved_settings.knowledge_base_dir)
    docs_root = knowledge_root / "Docs"

    all_markdown_docs = _list_markdown_files(docs_root)
    layer_counts = {
        label: len(_list_markdown_files(docs_root / directory_name))
        for directory_name, label in _LAYER_DIRECTORIES
    }
    template_names = tuple(path.name for path in _list_markdown_files(docs_root / "50_Templates"))

    status_doc = docs_root / "00_Shared" / "Standards" / "KB_V1_Status.md"
    status_text = _read_text_if_exists(status_doc)
    current_stage = _pick_first_content_line(
        _extract_section_lines(status_text, "当前阶段"),
        fallback="未在 `KB_V1_Status.md` 中读到明确阶段说明。",
    )
    blockers = _extract_section_points(status_text, "阻塞项")
    next_steps = _extract_section_points(status_text, "下一步")

    tracker_paths = {
        "Decision_Backlog.md": docs_root / "30_Review" / "Decision_Backlog.md",
        "Migration_Checker.md": docs_root / "30_Review" / "Migration_Checker.md",
        "Open_Questions.md": docs_root / "30_Review" / "Open_Questions.md",
    }
    tracker_states = {
        name: _describe_tracker_state(path)
        for name, path in tracker_paths.items()
    }

    priority_gaps: list[str] = []
    if layer_counts.get("GameLine", 0) == 0:
        priority_gaps.append("`10_GameLines/` 还没有已填充的 canonical 文档。")
    if layer_counts.get("Deployment", 0) == 0:
        priority_gaps.append("`20_Deployments/` 还没有已填充的 canonical 文档。")
    if tracker_states.get("Open_Questions.md", "").startswith("占位"):
        priority_gaps.append("`Open_Questions.md` 仍是占位状态，缺少真实待确认问题清单。")
    if tracker_states.get("Decision_Backlog.md", "").startswith("占位"):
        priority_gaps.append("`Decision_Backlog.md` 仍是占位状态，缺少真实决策积压记录。")
    if tracker_states.get("Migration_Checker.md", "").startswith("占位"):
        priority_gaps.append("`Migration_Checker.md` 仍是占位状态，缺少真实迁移检查清单。")
    if not priority_gaps:
        priority_gaps.append("继续补齐可审查的 canonical 文档，并同步更新状态登记。")

    return KnowledgeBaseStatusSnapshot(
        knowledge_root=knowledge_root,
        docs_root=docs_root,
        total_markdown_docs=len(all_markdown_docs),
        layer_counts=layer_counts,
        template_names=template_names,
        current_stage=current_stage,
        blockers=blockers,
        next_steps=next_steps,
        tracker_states=tracker_states,
        priority_gaps=tuple(priority_gaps[:5]),
    )


def render_knowledge_base_status_prompt(snapshot: KnowledgeBaseStatusSnapshot) -> str:
    docs_root = _display_path(snapshot.docs_root)
    template_summary = (
        "、".join(f"`{name}`" for name in snapshot.template_names)
        if snapshot.template_names
        else "未发现模板文件"
    )
    blockers = snapshot.blockers or ("未在状态文档中读到明确阻塞项。",)
    next_steps = snapshot.next_steps or ("先确认当前最缺的知识空位，再推进抽取或落稿。",)

    lines = [
        "## 运行时知识库状态扫描",
        "以下内容由当前仓库实时扫描生成，只能当作仓库证据；没有扫描到的事实必须标记为待确认。",
        f"- 扫描目录：`{docs_root}`",
        f"- Markdown 文档总数：{snapshot.total_markdown_docs}",
        (
            "- 层级分布："
            f"Shared={snapshot.layer_counts.get('Shared', 0)}，"
            f"GameLine={snapshot.layer_counts.get('GameLine', 0)}，"
            f"Deployment={snapshot.layer_counts.get('Deployment', 0)}，"
            f"Legacy={snapshot.layer_counts.get('Legacy', 0)}，"
            f"Review={snapshot.layer_counts.get('Review', 0)}，"
            f"Templates={snapshot.layer_counts.get('Templates', 0)}"
        ),
        f"- 当前模板：{template_summary}",
        "",
        "### 当前阶段",
        f"- {snapshot.current_stage}",
        "",
        "### 当前阻塞",
        *[f"- {item}" for item in blockers[:4]],
        "",
        "### Review 跟踪器现状",
        *[
            f"- `{name}`：{state}"
            for name, state in snapshot.tracker_states.items()
        ],
        "",
        "### 当前最优先补齐的空位",
        *[f"- {item}" for item in snapshot.priority_gaps],
        "",
        "### 下一步提醒",
        *[f"- {item}" for item in next_steps[:3]],
        "",
        "### 运行时要求",
        "- 每次用户发言后，先参考这份状态扫描判断当前最值得推进的知识缺口。",
        "- 如果用户提供的是知识素材，先判断它补的是哪个缺口，再继续抽取、归纳、定落点。",
        "- 即使用户的问题比较局部，也要在回答末尾给出最贴近当前状态缺口的下一步建议。",
    ]
    return "\n".join(lines)


def _list_markdown_files(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(root.rglob("*.md")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return files


def _read_text_if_exists(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _extract_section_lines(content: str, heading: str) -> list[str]:
    if not content.strip():
        return []

    lines = content.splitlines()
    collected: list[str] = []
    collecting = False
    heading_level = 0

    for line in lines:
        stripped = line.strip()
        match = _SECTION_PATTERN.match(stripped)
        if match is not None:
            level = len(match.group(1))
            title = match.group(2).strip()
            if collecting and level <= heading_level:
                break
            if title == heading:
                collecting = True
                heading_level = level
                continue
        if collecting:
            collected.append(line.rstrip())
    return collected


def _extract_section_points(content: str, heading: str) -> tuple[str, ...]:
    raw_lines = _extract_section_lines(content, heading)
    points: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        cleaned = re.sub(r"^[-*]\s*", "", stripped).strip()
        if not cleaned:
            continue
        points.append(cleaned)
    return tuple(points)


def _pick_first_content_line(lines: list[str], *, fallback: str) -> str:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        return re.sub(r"^[-*]\s*", "", stripped).strip()
    return fallback


def _describe_tracker_state(path: Path) -> str:
    content = _read_text_if_exists(path)
    if not content.strip():
        return "缺失"

    meaningful_lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not meaningful_lines:
        return "占位文件，尚未填入实质内容"
    if len(meaningful_lines) <= 2:
        return "占位文件，只有极少量内容"
    return f"已有内容（{len(meaningful_lines)} 行有效内容）"


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(resolve_project_path(".")).as_posix()
    except ValueError:
        return str(path)
