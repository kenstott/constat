# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""User permissions management.

Permissions are defined in the config.yaml under the 'permissions' section:

```yaml
permissions:
  users:
    kennethstott@gmail.com:
      admin: true
      projects: []
      databases: []
      documents: []
      apis: []

    analyst@company.com:
      admin: false
      projects:
        - sales-analytics.yaml
      databases:
        - sales
        - inventory
      documents: []
      apis: []

  default:
    admin: false
    projects: []
    databases: []
    documents: []
    apis: []
```
"""

import logging
from typing import Any, Optional

from constat.server.config import ServerConfig, UserPermissions as ConfigUserPermissions

logger = logging.getLogger(__name__)


class UserPermissions:
    """User permissions data for API responses."""

    def __init__(
        self,
        user_id: str,
        email: Optional[str] = None,
        admin: bool = False,
        projects: Optional[list[str]] = None,
        databases: Optional[list[str]] = None,
        documents: Optional[list[str]] = None,
        apis: Optional[list[str]] = None,
    ):
        self.user_id = user_id
        self.email = email
        self.admin = admin
        self.projects = projects or []
        self.databases = databases or []
        self.documents = documents or []
        self.apis = apis or []

    def can_access_project(self, project_filename: str) -> bool:
        """Check if user can access a specific project."""
        if self.admin:
            return True
        return project_filename in self.projects

    def can_access_database(self, db_name: str) -> bool:
        """Check if user can access a specific database."""
        if self.admin:
            return True
        return db_name in self.databases

    def can_access_document(self, doc_name: str) -> bool:
        """Check if user can access a specific document."""
        if self.admin:
            return True
        return doc_name in self.documents

    def can_access_api(self, api_name: str) -> bool:
        """Check if user can access a specific API."""
        if self.admin:
            return True
        return api_name in self.apis

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "admin": self.admin,
            "projects": self.projects,
            "databases": self.databases,
            "documents": self.documents,
            "apis": self.apis,
        }

    @classmethod
    def from_config(
        cls,
        config_perms: ConfigUserPermissions,
        user_id: str = "",
        email: Optional[str] = None,
    ) -> "UserPermissions":
        """Create from config UserPermissions model."""
        return cls(
            user_id=user_id,
            email=email,
            admin=config_perms.admin,
            projects=config_perms.projects,
            databases=config_perms.databases,
            documents=config_perms.documents,
            apis=config_perms.apis,
        )


def get_user_permissions(
    server_config: ServerConfig,
    email: str,
    user_id: str = "",
) -> UserPermissions:
    """Get permissions for a user from server config.

    Args:
        server_config: Server configuration containing permissions
        email: User's email address
        user_id: Firebase user ID (for reference)

    Returns:
        UserPermissions object
    """
    config_perms = server_config.permissions.get_user_permissions(email)
    return UserPermissions.from_config(config_perms, user_id=user_id, email=email)


def list_all_permissions(server_config: ServerConfig) -> list[dict[str, Any]]:
    """List all users with explicit permissions.

    Args:
        server_config: Server configuration containing permissions

    Returns:
        List of user permission dicts
    """
    result = []
    for email, perms in server_config.permissions.users.items():
        result.append({
            "email": email,
            "admin": perms.admin,
            "projects": perms.projects,
            "databases": perms.databases,
            "documents": perms.documents,
            "apis": perms.apis,
        })
    return result