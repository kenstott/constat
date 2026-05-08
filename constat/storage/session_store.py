# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Session ID persistence for REPL and UI clients.

Stores and retrieves session IDs per user, allowing session continuity
across restarts. New session IDs are only generated on explicit request
(e.g., "new query" command) or when no stored session exists.
"""

import uuid
from pathlib import Path
from typing import Optional


class SessionStore:
    """Persists session IDs per user.

    Session IDs are stored in {base_dir}/{user_id}/session_id
    """

    def __init__(self, user_id: str = "default", base_dir: Optional[Path] = None):
        """Initialize session store.

        Args:
            user_id: User identifier for scoping session storage
            base_dir: Base directory for storage. Defaults to .constat/
        """
        self.user_id = user_id
        self.base_dir = base_dir or Path(".constat")
        self._session_file = self.base_dir / user_id / "session_id"

    def get_or_create(self) -> str:
        """Get existing session ID or create a new one.

        Returns:
            Session ID (existing if available, new if not)
        """
        existing = self.get()
        if existing:
            return existing
        return self.create_new()

    def get(self) -> Optional[str]:
        """Get stored session ID.

        Returns:
            Session ID if exists, None otherwise
        """
        if self._session_file.exists():
            try:
                session_id = self._session_file.read_text().strip()
                if session_id:
                    return session_id
            except (IOError, OSError):
                pass
        return None

    def create_new(self) -> str:
        """Create and store a new session ID.

        Returns:
            Newly generated session ID
        """
        session_id = str(uuid.uuid4())
        self.store(session_id)
        return session_id

    def store(self, session_id: str) -> None:
        """Store a session ID.

        Args:
            session_id: Session ID to store
        """
        self._session_file.parent.mkdir(parents=True, exist_ok=True)
        self._session_file.write_text(session_id)

    def clear(self) -> None:
        """Clear stored session ID."""
        if self._session_file.exists():
            self._session_file.unlink()
