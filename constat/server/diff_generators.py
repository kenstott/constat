# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Diff generator registry for heartbeat-driven background work."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from constat.server.session_manager import ManagedSession, SessionManager

logger = logging.getLogger(__name__)


class DiffGenerator(ABC):
    """Base class for heartbeat-triggered diff generators.

    Each generator can specify a ``interval_seconds`` to sleep between runs.
    The heartbeat loop calls ``should_run`` on every beat; the base class
    enforces the cooldown automatically — subclasses only need to override
    ``_should_run`` for their data-change logic.
    """

    name: str
    interval_seconds: float = 0  # 0 = run on every eligible heartbeat

    def __init__(self) -> None:
        # Per-session last-run timestamps: session_id → monotonic time
        self._last_run: dict[str, float] = {}

    def should_run(self, managed: ManagedSession, since: str | None) -> bool:
        """Check cooldown, then delegate to subclass ``_should_run``."""
        if self.interval_seconds > 0:
            last = self._last_run.get(managed.session_id, 0)
            if (time.monotonic() - last) < self.interval_seconds:
                return False
        return self._should_run(managed, since)

    @abstractmethod
    def _should_run(self, managed: ManagedSession, since: str | None) -> bool:
        """Return True if data changed and this generator should fire."""

    def run(self, managed: ManagedSession, session_manager: SessionManager, since: str | None) -> None:
        """Record run time and delegate to ``_run``."""
        self._last_run[managed.session_id] = time.monotonic()
        self._run(managed, session_manager, since)

    @abstractmethod
    def _run(self, managed: ManagedSession, session_manager: SessionManager, since: str | None) -> None:
        """Execute the generator. Should be non-blocking (fire-and-forget)."""


class EntityDiffGenerator(DiffGenerator):
    """Triggers NER when document sources change."""

    name = "entities"

    def _should_run(self, managed: ManagedSession, since: str | None) -> bool:
        # Skip if NER already running — let it complete
        if managed._ner_thread and managed._ner_thread.is_alive():
            return False
        # First heartbeat (since=None) — run if docs exist
        if since is None:
            return bool(managed._file_refs)
        # Subsequent — run if file_refs updated since last heartbeat
        for ref in managed._file_refs:
            if (ref.get("last_refreshed") or ref.get("added_at", "")) > since:
                return True
        return False

    def _run(self, managed: ManagedSession, session_manager: SessionManager, since: str | None) -> None:
        session_manager.refresh_entities_async(managed.session_id)
