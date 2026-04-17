from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock, Timer
from time import monotonic, sleep
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_SESSION_MEMORY_UPDATE_DEBOUNCE_SECONDS = 0.1
DEFAULT_SESSION_MEMORY_WAIT_POLL_SECONDS = 0.01


@dataclass(frozen=True)
class SessionMemoryRefreshTarget:
    conversation_id: str
    thread_id: str
    allowed_session_file_path: str


class BackgroundSessionMemoryUpdater:
    def __init__(
        self,
        refresh_callback: Callable[[SessionMemoryRefreshTarget], None],
        *,
        debounce_seconds: float = DEFAULT_SESSION_MEMORY_UPDATE_DEBOUNCE_SECONDS,
    ) -> None:
        self._refresh_callback = refresh_callback
        self._debounce_seconds = max(float(debounce_seconds or 0.0), 0.0)
        self._lock = Lock()
        self._timers: dict[str, Timer] = {}
        self._pending_targets: dict[str, SessionMemoryRefreshTarget] = {}
        self._active_threads: set[str] = set()
        self._closed = False

    def schedule(self, target: SessionMemoryRefreshTarget) -> None:
        normalized_thread_id = str(target.thread_id or "").strip()
        if not normalized_thread_id:
            return

        with self._lock:
            if self._closed:
                return
            existing_timer = self._timers.pop(normalized_thread_id, None)
            if existing_timer is not None:
                existing_timer.cancel()
            self._pending_targets[normalized_thread_id] = target
            timer = Timer(self._debounce_seconds, self._run_pending_target, args=(normalized_thread_id,))
            timer.daemon = True
            self._timers[normalized_thread_id] = timer
            timer.start()

    def cancel(self, thread_id: str) -> None:
        normalized_thread_id = str(thread_id or "").strip()
        if not normalized_thread_id:
            return
        with self._lock:
            timer = self._timers.pop(normalized_thread_id, None)
            if timer is not None:
                timer.cancel()
            self._pending_targets.pop(normalized_thread_id, None)

    def flush(self, thread_id: str | None = None) -> None:
        if thread_id is None:
            with self._lock:
                pending_thread_ids = list(self._pending_targets.keys())
            for pending_thread_id in pending_thread_ids:
                self.flush(pending_thread_id)
            return

        normalized_thread_id = str(thread_id or "").strip()
        if not normalized_thread_id:
            return

        with self._lock:
            timer = self._timers.pop(normalized_thread_id, None)
            target = self._pending_targets.pop(normalized_thread_id, None)
        if timer is not None:
            timer.cancel()
        if target is not None:
            self._run_target(target)

    def wait_for_idle(self, thread_id: str | None = None, *, timeout: float = 1.0) -> bool:
        deadline = monotonic() + max(float(timeout or 0.0), 0.0)
        normalized_thread_id = str(thread_id or "").strip()
        while monotonic() <= deadline:
            with self._lock:
                if normalized_thread_id:
                    is_idle = (
                        normalized_thread_id not in self._pending_targets
                        and normalized_thread_id not in self._active_threads
                    )
                else:
                    is_idle = not self._pending_targets and not self._active_threads
            if is_idle:
                return True
            sleep(DEFAULT_SESSION_MEMORY_WAIT_POLL_SECONDS)
        return False

    def close(self) -> None:
        with self._lock:
            self._closed = True
            timers = list(self._timers.values())
            self._timers.clear()
            self._pending_targets.clear()
        for timer in timers:
            timer.cancel()

    def _run_pending_target(self, thread_id: str) -> None:
        with self._lock:
            self._timers.pop(thread_id, None)
            target = self._pending_targets.pop(thread_id, None)
        if target is not None:
            self._run_target(target)

    def _run_target(self, target: SessionMemoryRefreshTarget) -> None:
        normalized_thread_id = str(target.thread_id or "").strip()
        if not normalized_thread_id:
            return

        with self._lock:
            if self._closed:
                return
            self._active_threads.add(normalized_thread_id)

        try:
            self._refresh_callback(target)
        except Exception:
            logger.warning(
                "Background session memory refresh failed thread=%s conversation=%s file=%s",
                target.thread_id,
                target.conversation_id,
                Path(target.allowed_session_file_path).expanduser().resolve(),
                exc_info=True,
            )
        finally:
            with self._lock:
                self._active_threads.discard(normalized_thread_id)
