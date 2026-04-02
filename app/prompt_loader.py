from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from app.paths import resolve_project_path


_SECTION_HEADING_PATTERN = re.compile(r"^##\s+(?P<name>[A-Za-z0-9_ -]+)\s*$")
_TEMPLATE_VARIABLE_PATTERN = re.compile(r"{{\s*(?P<name>[A-Za-z0-9_]+)\s*}}")
JADE_PROMPT_PATH = Path("JADE.md")
JADE_RULES_DIR = Path(".jade/rules")


def load_prompt_text(relative_path: str | Path, **variables: str) -> str:
    path, raw_text = _read_prompt_source(relative_path)
    return render_prompt_template(raw_text.strip(), variables, source_path=path)


def load_prompt_sections(
    relative_path: str | Path,
    *,
    required_sections: Sequence[str] = (),
    **variables: str,
) -> dict[str, str]:
    path, raw_text = _read_prompt_source(relative_path)
    sections = _parse_prompt_sections(raw_text)
    normalized_required = tuple(_normalize_section_name(name) for name in required_sections)
    missing = [name for name in normalized_required if name not in sections]
    if missing:
        raise RuntimeError(
            f"Prompt file {path} is missing required sections: {', '.join(missing)}"
        )
    return {
        name: render_prompt_template(content, variables, source_path=path).strip()
        for name, content in sections.items()
    }


def load_optional_prompt_text(relative_path: str | Path, **variables: str) -> str | None:
    path = resolve_project_path(relative_path)
    if not path.is_file():
        return None
    raw_text = path.read_text(encoding="utf-8")
    return render_prompt_template(raw_text.strip(), variables, source_path=path)


def load_shared_instruction_text(
    *,
    include_repo_prompt: bool = True,
    rule_names: Sequence[str] | None = None,
    **variables: str,
) -> str:
    layers: list[str] = []
    if include_repo_prompt:
        jade_prompt = load_optional_prompt_text(JADE_PROMPT_PATH, **variables)
        if jade_prompt:
            layers.append(jade_prompt)
    for rule_path in _resolve_shared_rule_paths(rule_names):
        layers.append(load_prompt_text(rule_path, **variables))
    return join_prompt_layers(*layers)


def join_prompt_layers(*layers: str) -> str:
    return "\n\n".join(
        layer.strip()
        for layer in layers
        if isinstance(layer, str) and layer.strip()
    )


def render_prompt_template(
    text: str,
    variables: dict[str, str],
    *,
    source_path: Path,
) -> str:
    def replace(match: re.Match[str]) -> str:
        variable_name = match.group("name")
        if variable_name not in variables:
            raise RuntimeError(
                f"Prompt file {source_path} references unknown template variable: {variable_name}"
            )
        return str(variables[variable_name]).strip()

    return _TEMPLATE_VARIABLE_PATTERN.sub(replace, text)


def _read_prompt_source(relative_path: str | Path) -> tuple[Path, str]:
    path = resolve_project_path(relative_path)
    if not path.is_file():
        raise RuntimeError(f"Prompt file not found: {path}")
    return path, path.read_text(encoding="utf-8")


def _resolve_shared_rule_paths(rule_names: Sequence[str] | None) -> tuple[Path, ...]:
    rules_dir = resolve_project_path(JADE_RULES_DIR)
    if rule_names is None:
        if not rules_dir.is_dir():
            return ()
        return tuple(sorted(path for path in rules_dir.glob("*.md") if path.is_file()))

    paths: list[Path] = []
    missing: list[str] = []
    for rule_name in rule_names:
        rule_name_text = str(rule_name)
        path = _resolve_rule_path(rules_dir, rule_name_text)
        if path is None:
            missing.append(str(rules_dir / (rule_name_text if rule_name_text.endswith(".md") else f"{rule_name_text}.md")))
            continue
        paths.append(path)

    if missing:
        raise RuntimeError(f"Shared prompt rule files not found: {', '.join(missing)}")
    return tuple(paths)


def _resolve_rule_path(rules_dir: Path, rule_name: str) -> Path | None:
    candidate_names = [rule_name]
    if not rule_name.endswith(".md"):
        candidate_names.append(f"{rule_name}.md")

    for candidate_name in candidate_names:
        candidate_path = rules_dir / candidate_name
        if candidate_path.is_file():
            return candidate_path

    normalized_candidates = {name.casefold() for name in candidate_names}
    for path in sorted(rules_dir.glob("*.md")):
        if path.name.casefold() in normalized_candidates:
            return path
    return None


def _parse_prompt_sections(raw_text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_section_name: str | None = None

    for line in raw_text.splitlines():
        match = _SECTION_HEADING_PATTERN.match(line.strip())
        if match:
            current_section_name = _normalize_section_name(match.group("name"))
            sections.setdefault(current_section_name, [])
            continue
        if current_section_name is None:
            continue
        sections[current_section_name].append(line)

    return {
        name: "\n".join(lines).strip()
        for name, lines in sections.items()
    }


def _normalize_section_name(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower())
