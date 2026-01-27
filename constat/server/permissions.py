# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""User permissions management.

Permissions are stored in .constat/permissions.yaml and define:
- admin: boolean - allows managing projects from UI
- projects: list of project filenames the user can access

Example permissions.yaml:
```yaml
users:
  kennethstott@gmail.com:
    admin: true
    projects: []  # Empty means all projects (for admins)

  analyst@company.com:
    admin: false
    projects:
      - sales-analytics.yaml
      - hr-reporting.yaml

# Default permissions for users not explicitly listed
default:
  admin: false
  projects: []  # Empty with admin=false means no project access
```
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


class UserPermissions:
    """User permissions data."""

    def __init__(
        self,
        user_id: str,
        email: Optional[str] = None,
        admin: bool = False,
        projects: Optional[list[str]] = None,
    ):
        self.user_id = user_id
        self.email = email
        self.admin = admin
        self.projects = projects or []

    def can_access_project(self, project_filename: str) -> bool:
        """Check if user can access a specific project."""
        # Admins can access all projects
        if self.admin:
            return True
        # Check if project is in user's allowed list
        return project_filename in self.projects

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "admin": self.admin,
            "projects": self.projects,
        }


class PermissionsStore:
    """Manages user permissions from YAML file."""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize permissions store.

        Args:
            base_dir: Base directory for .constat. Defaults to current directory.
        """
        self.base_dir = Path(base_dir) if base_dir else Path(".constat")
        self.file_path = self.base_dir / "permissions.yaml"
        self._data: Optional[dict] = None

    def _load(self) -> dict:
        """Load permissions from YAML file."""
        if self._data is not None:
            return self._data

        if not self.file_path.exists():
            # Create default permissions file
            self._data = self._create_default()
            return self._data

        try:
            with open(self.file_path) as f:
                self._data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load permissions: {e}")
            self._data = {"users": {}, "default": {"admin": False, "projects": []}}

        return self._data

    def _create_default(self) -> dict:
        """Create default permissions file."""
        default_data = {
            "users": {
                "kennethstott@gmail.com": {
                    "admin": True,
                    "projects": [],  # Empty means all projects for admins
                },
            },
            "default": {
                "admin": False,
                "projects": [],  # No project access by default
            },
        }

        # Ensure directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Write default file
        try:
            with open(self.file_path, "w") as f:
                yaml.dump(default_data, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Created default permissions file: {self.file_path}")
        except Exception as e:
            logger.error(f"Failed to create permissions file: {e}")

        return default_data

    def _save(self) -> None:
        """Save permissions to YAML file."""
        if self._data is None:
            return

        self.base_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.file_path, "w") as f:
                yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.error(f"Failed to save permissions: {e}")

    def reload(self) -> None:
        """Force reload from file."""
        self._data = None
        self._load()

    def get_user_permissions(self, email: str, user_id: str = "") -> UserPermissions:
        """Get permissions for a user by email.

        Args:
            email: User's email address
            user_id: Firebase user ID (for reference)

        Returns:
            UserPermissions object
        """
        data = self._load()
        users = data.get("users", {})
        default = data.get("default", {"admin": False, "projects": []})

        # Look up by email
        user_data = users.get(email, default)

        return UserPermissions(
            user_id=user_id,
            email=email,
            admin=user_data.get("admin", False),
            projects=user_data.get("projects", []),
        )

    def set_user_permissions(
        self,
        email: str,
        admin: Optional[bool] = None,
        projects: Optional[list[str]] = None,
    ) -> UserPermissions:
        """Set permissions for a user.

        Args:
            email: User's email address
            admin: Whether user has admin rights
            projects: List of project filenames user can access

        Returns:
            Updated UserPermissions object
        """
        data = self._load()

        if "users" not in data:
            data["users"] = {}

        if email not in data["users"]:
            data["users"][email] = {"admin": False, "projects": []}

        if admin is not None:
            data["users"][email]["admin"] = admin
        if projects is not None:
            data["users"][email]["projects"] = projects

        self._save()

        return self.get_user_permissions(email)

    def list_users(self) -> list[dict[str, Any]]:
        """List all users with explicit permissions.

        Returns:
            List of user permission dicts
        """
        data = self._load()
        users = data.get("users", {})

        return [
            {
                "email": email,
                "admin": perms.get("admin", False),
                "projects": perms.get("projects", []),
            }
            for email, perms in users.items()
        ]


# Global instance for convenience
_permissions_store: Optional[PermissionsStore] = None


def get_permissions_store() -> PermissionsStore:
    """Get or create the global permissions store."""
    global _permissions_store
    if _permissions_store is None:
        _permissions_store = PermissionsStore()
    return _permissions_store