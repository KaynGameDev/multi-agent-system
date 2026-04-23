from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from mas_platform.errors import MasPlatformError
from mas_platform.registry import load_registry, resolve_repo_root
from mas_platform.runtime import run_package
from mas_platform.scaffold import scaffold_package
from mas_platform.validator import validate_package, validate_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mas", description="Shared ADK agent monorepo CLI.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the monorepo root. Defaults to the current directory.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List published packages.")

    validate_parser = subparsers.add_parser("validate", help="Validate one package or the entire repo.")
    validate_parser.add_argument("agent_id", nargs="?", help="Optional package id to validate.")

    run_parser = subparsers.add_parser("run", help="Run one package against a single prompt.")
    run_parser.add_argument("agent_id", help="Package id to run.")
    run_parser.add_argument("--message", default="hello", help="User message sent to the app.")
    run_parser.add_argument("--user-id", default="local_user", help="User id for the ADK session.")
    run_parser.add_argument("--session-id", default=None, help="Optional session id.")

    test_parser = subparsers.add_parser("test", help="Run only the selected package tests.")
    test_parser.add_argument("agent_id", help="Package id to test.")

    scaffold_parser = subparsers.add_parser("scaffold", help="Create a new package scaffold.")
    scaffold_parser.add_argument("kind", choices=["agent", "group"], help="Package kind.")
    scaffold_parser.add_argument("agent_id", help="New package id.")
    scaffold_parser.add_argument("--owner", default="team", help="Owner recorded in the manifest.")
    scaffold_parser.add_argument("--description", default=None, help="Optional package description.")

    return parser


def command_list(repo_root: Path) -> int:
    registry = load_registry(repo_root)
    for package in registry.values():
        manifest = package.manifest
        print(
            "\t".join(
                [
                    manifest.id,
                    manifest.kind,
                    manifest.version,
                    manifest.owner,
                    manifest.description,
                ]
            )
        )
    return 0


def command_validate(repo_root: Path, agent_id: str | None) -> int:
    if agent_id:
        registry = load_registry(repo_root)
        if agent_id not in registry:
            raise MasPlatformError(f"Unknown agent id '{agent_id}'.")
        reports = [validate_package(registry[agent_id], repo_root=repo_root)]
    else:
        reports = validate_registry(repo_root)

    exit_code = 0
    for report in reports:
        status = "OK" if report.ok else "ERROR"
        print(f"{status}\t{report.package.id}")
        for warning in report.warnings:
            print(f"  warning: {warning}")
        for error in report.errors:
            print(f"  error: {error}")
        if not report.ok:
            exit_code = 1
    return exit_code


def command_run(
    repo_root: Path,
    *,
    agent_id: str,
    message: str,
    user_id: str,
    session_id: str | None,
) -> int:
    registry = load_registry(repo_root)
    if agent_id not in registry:
        raise MasPlatformError(f"Unknown agent id '{agent_id}'.")
    result = run_package(
        repo_root,
        agent_id=agent_id,
        message=message,
        user_id=user_id,
        session_id=session_id,
    )
    print(f"app={result.app_name}\tsession={result.session_id}")
    for line in result.event_lines:
        print(line)
    return 0


def command_test(repo_root: Path, *, agent_id: str) -> int:
    registry = load_registry(repo_root)
    if agent_id not in registry:
        raise MasPlatformError(f"Unknown agent id '{agent_id}'.")

    package = registry[agent_id]
    report = validate_package(package, repo_root=repo_root)
    if not report.ok:
        raise MasPlatformError(
            "Package must validate before running tests:\n" + "\n".join(report.errors)
        )

    test_paths = [str(path) for path in package.effective_test_paths()]
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *test_paths],
        cwd=repo_root,
        check=False,
    )
    return completed.returncode


def command_scaffold(
    repo_root: Path,
    *,
    kind: str,
    agent_id: str,
    owner: str,
    description: str | None,
) -> int:
    package_root = scaffold_package(
        repo_root,
        kind=kind,
        agent_id=agent_id,
        owner=owner,
        description=description,
    )
    print(package_root)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = resolve_repo_root(args.repo_root)
    load_dotenv(repo_root / ".env", override=False)

    try:
        if args.command == "list":
            return command_list(repo_root)
        if args.command == "validate":
            return command_validate(repo_root, args.agent_id)
        if args.command == "run":
            return command_run(
                repo_root,
                agent_id=args.agent_id,
                message=args.message,
                user_id=args.user_id,
                session_id=args.session_id,
            )
        if args.command == "test":
            return command_test(repo_root, agent_id=args.agent_id)
        if args.command == "scaffold":
            return command_scaffold(
                repo_root,
                kind=args.kind,
                agent_id=args.agent_id,
                owner=args.owner,
                description=args.description,
            )
    except MasPlatformError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    parser.error(f"Unsupported command: {args.command}")
    return 2


def run_cli() -> None:
    raise SystemExit(main())
