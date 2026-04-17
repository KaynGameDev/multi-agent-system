from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Timer
from time import monotonic, sleep
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_CONSOLIDATION_DEBOUNCE_SECONDS = 5.0
DEFAULT_MEMORY_CONSOLIDATION_WAIT_POLL_SECONDS = 0.01


@dataclass(frozen=True)
class MemoryConsolidationTarget:
    root_dir: str
    agent_name: str = ""
    memory_scope: str = ""
    scope_key: str = ""


class BackgroundMemoryConsolidator:
    def __init__(
        self,
        consolidate_callback: Callable[[MemoryConsolidationTarget], None],
        *,
        debounce_seconds: float = DEFAULT_MEMORY_CONSOLIDATION_DEBOUNCE_SECONDS,
    ) -> None:
        self._consolidate_callback = consolidate_callback
        self._debounce_seconds = max(float(debounce_seconds or 0.0), 0.0)
        self._lock = Lock()
        self._timers: dict[str, Timer] = {}
        self._pending_targets: dict[str, MemoryConsolidationTarget] = {}
        self._active_roots: set[str] = set()
        self._closed = False

    def schedule(self, target: MemoryConsolidationTarget) -> None:
        normalized_root = str(target.root_dir or "").strip()
        if not normalized_root:
            return

        with self._lock:
            if self._closed:
                return
            existing_timer = self._timers.pop(normalized_root, None)
            if existing_timer is not None:
                existing_timer.cancel()
            self._pending_targets[normalized_root] = target
            timer = Timer(self._debounce_seconds, self._run_pending_target, args=(normalized_root,))
            timer.daemon = True
            self._timers[normalized_root] = timer
            timer.start()

    def cancel(self, root_dir: str) -> None:
        normalized_root = str(root_dir or "").strip()
        if not normalized_root:
            return
        with self._lock:
            timer = self._timers.pop(normalized_root, None)
            if timer is not None:
                timer.cancel()
            self._pending_targets.pop(normalized_root, None)

    def flush(self, root_dir: str | None = None) -> None:
        if root_dir is None:
            with self._lock:
                pending_roots = list(self._pending_targets.keys())
            for pending_root in pending_roots:
                self.flush(pending_root)
            return

        normalized_root = str(root_dir or "").strip()
        if not normalized_root:
            return

        with self._lock:
            timer = self._timers.pop(normalized_root, None)
            target = self._pending_targets.pop(normalized_root, None)
        if timer is not None:
            timer.cancel()
        if target is not None:
            self._run_target(target)

    def wait_for_idle(self, root_dir: str | None = None, *, timeout: float = 1.0) -> bool:
        deadline = monotonic() + max(float(timeout or 0.0), 0.0)
        normalized_root = str(root_dir or "").strip()
        while monotonic() <= deadline:
            with self._lock:
                if normalized_root:
                    is_idle = (
                        normalized_root not in self._pending_targets
                        and normalized_root not in self._active_roots
                    )
                else:
                    is_idle = not self._pending_targets and not self._active_roots
            if is_idle:
                return True
            sleep(DEFAULT_MEMORY_CONSOLIDATION_WAIT_POLL_SECONDS)
        return False

    def close(self) -> None:
        with self._lock:
            self._closed = True
            timers = list(self._timers.values())
            self._timers.clear()
            self._pending_targets.clear()
        for timer in timers:
            timer.cancel()

    def _run_pending_target(self, root_dir: str) -> None:
        with self._lock:
            self._timers.pop(root_dir, None)
            target = self._pending_targets.pop(root_dir, None)
        if target is not None:
            self._run_target(target)

    def _run_target(self, target: MemoryConsolidationTarget) -> None:
        normalized_root = str(target.root_dir or "").strip()
        if not normalized_root:
            return

        with self._lock:
            if self._closed:
                return
            self._active_roots.add(normalized_root)

        try:
            self._consolidate_callback(target)
        except Exception:
            logger.warning(
                "Background memory consolidation failed root=%s agent=%s scope=%s scope_key=%s",
                Path(target.root_dir).expanduser().resolve(),
                target.agent_name,
                target.memory_scope,
                target.scope_key,
                exc_info=True,
            )
        finally:
            with self._lock:
                self._active_roots.discard(normalized_root)
