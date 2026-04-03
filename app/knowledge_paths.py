from __future__ import annotations

import re
import unicodedata
from pathlib import Path


KNOWN_GAME_LINE_DIRECTORY_NAMES = {
    "buyudaluandou": "BuYuDaLuanDou",
    "2-player-fishing": "BuYuDaLuanDou",
    "two-player-fishing": "BuYuDaLuanDou",
    "f4buyu": "F4BuYu",
    "4-player-fishing": "F4BuYu",
    "four-player-fishing": "F4BuYu",
    "xybuyu": "XYBuYu",
    "west-journey-fishing": "XYBuYu",
}

KNOWN_DEPLOYMENT_DIRECTORY_NAMES = {
    "indonesiamain": "IndonesiaMain",
    "indonesia-main": "IndonesiaMain",
    "indonesiasub": "IndonesiaSub",
    "indonesia-sub": "IndonesiaSub",
    "malaysiasub": "MalaysiaSub",
    "malaysia-sub": "MalaysiaSub",
    "thailandsub": "ThailandSub",
    "thailand-sub": "ThailandSub",
}

SHARED_CATEGORY_DIRECTORY_NAMES = {
    "glossary": "Glossary",
    "design": "DesignPrinciples",
    "design_principle": "DesignPrinciples",
    "design_principles": "DesignPrinciples",
    "naming": "Naming",
    "standard": "Standards",
    "standards": "Standards",
    "common_system": "CommonSystems",
    "common_systems": "CommonSystems",
    "system": "CommonSystems",
    "systems": "CommonSystems",
}

GAME_LINE_CATEGORY_DIRECTORY_NAMES = {
    "line_overview": "LineOverview",
    "overview": "LineOverview",
    "shared_design": "SharedDesign",
    "design": "SharedDesign",
    "shared_system": "SharedSystems",
    "shared_systems": "SharedSystems",
    "system": "SharedSystems",
    "systems": "SharedSystems",
    "shared_terminology": "SharedTerminology",
    "terminology": "SharedTerminology",
    "glossary": "SharedTerminology",
}

DEPLOYMENT_CATEGORY_DIRECTORY_NAMES = {
    "master_gdd": "MasterGDD",
    "mastergdd": "MasterGDD",
    "review": "Review",
    "feature": "Features",
    "features": "Features",
}


def normalize_slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_value.lower().strip()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered)
    return lowered.strip("-")


def build_pascal_case_name(value: str) -> str:
    parts = [part for part in re.split(r"[^a-zA-Z0-9]+", value) if part]
    if not parts:
        return "?"

    rendered_parts: list[str] = []
    for part in parts:
        if part.isdigit():
            rendered_parts.append(part)
            continue
        if len(part) <= 2 and any(char.isalpha() for char in part):
            rendered_parts.append(part.upper())
            continue
        rendered_parts.append(part[:1].upper() + part[1:])
    return "".join(rendered_parts) or "?"


def build_game_line_directory_name(game_slug: str) -> str:
    normalized = normalize_slug(game_slug)
    if not normalized:
        return "?"
    return KNOWN_GAME_LINE_DIRECTORY_NAMES.get(normalized, build_pascal_case_name(normalized))


def build_deployment_directory_name(market_slug: str) -> str:
    normalized = normalize_slug(market_slug)
    if not normalized:
        return "?"
    return KNOWN_DEPLOYMENT_DIRECTORY_NAMES.get(normalized, build_pascal_case_name(normalized))


def build_conversion_package_relative_path(game_slug: str, market_slug: str, feature_slug: str) -> str:
    deployment_dir = build_deployment_directory_name(market_slug)
    game_line_dir = build_game_line_directory_name(game_slug)
    return f"Docs/20_Deployments/{deployment_dir}/{game_line_dir}/Features/{feature_slug or '?'}"


def build_shared_docs_root(knowledge_root: Path) -> Path:
    return knowledge_root / "Docs" / "00_Shared"


def build_game_line_root(knowledge_root: Path, game_slug: str) -> Path:
    return knowledge_root / "Docs" / "10_GameLines" / build_game_line_directory_name(game_slug)


def build_deployment_game_root(knowledge_root: Path, game_slug: str, market_slug: str) -> Path:
    return (
        knowledge_root
        / "Docs"
        / "20_Deployments"
        / build_deployment_directory_name(market_slug)
        / build_game_line_directory_name(game_slug)
    )


def normalize_knowledge_category(value: str) -> str:
    return normalize_slug(value).replace("-", "_")


def sanitize_markdown_filename(value: str, *, default_name: str = "README.md") -> str:
    raw_name = str(value or "").strip()
    if not raw_name:
        return default_name

    basename = raw_name.replace("\\", "/").split("/")[-1].strip()
    basename = basename.strip(". ")
    basename = re.sub(r"\s+", "_", basename)
    basename = basename or default_name.rsplit(".", 1)[0]
    if not basename.lower().endswith(".md"):
        basename = f"{basename}.md"
    return basename or default_name


def build_knowledge_markdown_relative_path(
    *,
    layer: str,
    category: str,
    filename: str = "README.md",
    game_slug: str = "",
    market_slug: str = "",
    feature_slug: str = "",
    legacy_bucket: str = "",
) -> str:
    normalized_layer = normalize_knowledge_category(layer)
    normalized_category = normalize_knowledge_category(category)

    if normalized_layer == "shared":
        category_dir = SHARED_CATEGORY_DIRECTORY_NAMES.get(normalized_category)
        if not category_dir:
            raise ValueError(
                "Unsupported shared category. Use one of: glossary, design_principles, naming, standards, common_systems."
            )
        return (Path("Docs") / "00_Shared" / category_dir / sanitize_markdown_filename(filename)).as_posix()

    if normalized_layer == "game_line":
        if not str(game_slug).strip():
            raise ValueError("game_slug is required for game_line documents.")
        category_dir = GAME_LINE_CATEGORY_DIRECTORY_NAMES.get(normalized_category)
        if not category_dir:
            raise ValueError(
                "Unsupported game_line category. Use one of: line_overview, shared_design, shared_systems, shared_terminology."
            )
        return (
            Path("Docs")
            / "10_GameLines"
            / build_game_line_directory_name(game_slug)
            / category_dir
            / sanitize_markdown_filename(filename)
        ).as_posix()

    if normalized_layer == "deployment":
        if not str(game_slug).strip():
            raise ValueError("game_slug is required for deployment documents.")
        if not str(market_slug).strip():
            raise ValueError("market_slug is required for deployment documents.")
        category_dir = DEPLOYMENT_CATEGORY_DIRECTORY_NAMES.get(normalized_category)
        if not category_dir:
            raise ValueError("Unsupported deployment category. Use one of: master_gdd, review, feature.")

        base_path = (
            Path("Docs")
            / "20_Deployments"
            / build_deployment_directory_name(market_slug)
            / build_game_line_directory_name(game_slug)
        )
        if category_dir == "Features":
            normalized_feature_slug = normalize_slug(feature_slug)
            if not normalized_feature_slug:
                raise ValueError("feature_slug is required when category is feature.")
            return (base_path / "Features" / normalized_feature_slug / "README.md").as_posix()
        return (base_path / category_dir / sanitize_markdown_filename(filename)).as_posix()

    if normalized_layer == "legacy":
        bucket_name = ""
        if str(market_slug).strip():
            bucket_name = build_deployment_directory_name(market_slug)
        elif str(legacy_bucket).strip():
            bucket_name = build_pascal_case_name(normalize_slug(legacy_bucket))
        else:
            bucket_name = "General"
        return (Path("Docs") / "40_Legacy" / bucket_name / sanitize_markdown_filename(filename)).as_posix()

    raise ValueError("Unsupported layer. Use one of: shared, game_line, deployment, legacy.")
