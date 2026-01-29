# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""User roles for customizing system prompts.

Roles are optional. If ~/.constat/roles.yaml exists, users can switch
between defined roles. Each role adds a prompt to the system prompt.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import yaml
import logging

logger = logging.getLogger(__name__)

ROLES_FILE = Path.home() / ".constat" / "roles.yaml"


@dataclass
class Role:
    """A user-defined role."""
    name: str
    prompt: str


class RoleManager:
    """Manages user roles loaded from ~/.constat/roles.yaml."""

    def __init__(self):
        self._roles: dict[str, Role] = {}
        self._active_role: Optional[str] = None
        self._load_roles()

    def _load_roles(self) -> None:
        """Load roles from YAML file if it exists."""
        if not ROLES_FILE.exists():
            logger.debug(f"No roles file at {ROLES_FILE}")
            return

        try:
            with open(ROLES_FILE, "r") as f:
                data = yaml.safe_load(f) or {}

            for name, config in data.items():
                if isinstance(config, dict) and "prompt" in config:
                    self._roles[name] = Role(
                        name=name,
                        prompt=config["prompt"].strip(),
                    )
                    logger.debug(f"Loaded role: {name}")

            logger.info(f"Loaded {len(self._roles)} roles from {ROLES_FILE}")
        except Exception as e:
            logger.warning(f"Failed to load roles from {ROLES_FILE}: {e}")

    def reload(self) -> None:
        """Reload roles from file."""
        self._roles.clear()
        self._load_roles()

    def list_roles(self) -> list[str]:
        """Get list of available role names."""
        return list(self._roles.keys())

    def get_role(self, name: str) -> Optional[Role]:
        """Get a role by name."""
        return self._roles.get(name)

    def set_active_role(self, name: Optional[str]) -> bool:
        """Set the active role.

        Args:
            name: Role name, or None to clear active role

        Returns:
            True if role was set successfully, False if role not found
        """
        if name is None or name.lower() == "none":
            self._active_role = None
            return True

        if name not in self._roles:
            return False

        self._active_role = name
        return True

    @property
    def active_role(self) -> Optional[Role]:
        """Get the currently active role."""
        if self._active_role:
            return self._roles.get(self._active_role)
        return None

    @property
    def active_role_name(self) -> Optional[str]:
        """Get the name of the currently active role."""
        return self._active_role

    def get_role_prompt(self) -> str:
        """Get the prompt for the active role, or empty string if none."""
        role = self.active_role
        return role.prompt if role else ""

    @property
    def has_roles(self) -> bool:
        """Check if any roles are defined."""
        return len(self._roles) > 0

    @property
    def roles_file_path(self) -> Path:
        """Get the path to the roles file."""
        return ROLES_FILE
