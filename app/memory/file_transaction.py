from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile


@dataclass(frozen=True)
class RootFileTransactionOperation:
    path: Path
    content: str | None = None
    delete: bool = False


@dataclass(frozen=True)
class RootFileTransactionSnapshot:
    path: Path
    existed: bool
    content_bytes: bytes | None = None


class RootFileTransaction:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self._operations: dict[Path, RootFileTransactionOperation] = {}

    def write_text(self, path: str | Path, content: str) -> None:
        resolved_path = self._resolve_path(path)
        self._operations[resolved_path] = RootFileTransactionOperation(
            path=resolved_path,
            content=str(content),
        )

    def delete(self, path: str | Path) -> None:
        resolved_path = self._resolve_path(path)
        self._operations[resolved_path] = RootFileTransactionOperation(
            path=resolved_path,
            delete=True,
        )

    def commit(self) -> None:
        snapshots = [
            _snapshot_root_file_transaction_path(operation.path)
            for operation in self._operations.values()
        ]
        try:
            for operation in self._operations.values():
                _apply_root_file_transaction_operation(operation)
        except Exception:
            _rollback_root_file_transaction(self.root_dir, snapshots)
            raise

        for operation in self._operations.values():
            if operation.delete:
                _prune_empty_parent_directories(operation.path.parent, stop_at=self.root_dir)

    def _resolve_path(self, path: str | Path) -> Path:
        raw_path = Path(path).expanduser()
        resolved_path = (raw_path if raw_path.is_absolute() else self.root_dir / raw_path).resolve()
        try:
            resolved_path.relative_to(self.root_dir)
        except ValueError as exc:
            raise ValueError(
                f"Root file transaction path must stay under {self.root_dir}: {path}"
            ) from exc
        return resolved_path


def _snapshot_root_file_transaction_path(path: Path) -> RootFileTransactionSnapshot:
    if not path.exists():
        return RootFileTransactionSnapshot(path=path, existed=False)
    return RootFileTransactionSnapshot(
        path=path,
        existed=True,
        content_bytes=path.read_bytes(),
    )


def _apply_root_file_transaction_operation(operation: RootFileTransactionOperation) -> None:
    if operation.delete:
        if operation.path.exists():
            operation.path.unlink()
        return

    _write_bytes_atomically(
        operation.path,
        str(operation.content or "").encode("utf-8"),
    )


def _rollback_root_file_transaction(
    root_dir: Path,
    snapshots: list[RootFileTransactionSnapshot],
) -> None:
    for snapshot in snapshots:
        if snapshot.existed:
            _write_bytes_atomically(snapshot.path, snapshot.content_bytes or b"")
            continue
        if snapshot.path.exists():
            snapshot.path.unlink()
        _prune_empty_parent_directories(snapshot.path.parent, stop_at=root_dir)


def _write_bytes_atomically(path: Path, content_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb",
            dir=str(path.parent),
            delete=False,
        ) as temp_file:
            temp_file.write(content_bytes)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_path = Path(temp_file.name)
        temp_path.replace(path)
        _fsync_directory(path.parent)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def _fsync_directory(directory: Path) -> None:
    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)
    except (OSError, NotImplementedError):
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _prune_empty_parent_directories(start_dir: Path, *, stop_at: Path) -> None:
    current = start_dir.resolve()
    resolved_stop_at = stop_at.resolve()
    while current != resolved_stop_at and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent
