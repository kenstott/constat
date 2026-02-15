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
      domains: []
      databases: []
      documents: []
      apis: []

    analyst@company.com:
      admin: false
      domains:
        - sales-analytics.yaml
      databases:
        - sales
        - inventory
      documents: []
      apis: []

  default:
    admin: false
    domains: []
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
        domains: Optional[list[str]] = None,
        databases: Optional[list[str]] = None,
        documents: Optional[list[str]] = None,
        apis: Optional[list[str]] = None,
        # Backwards compatibility
        projects: Optional[list[str]] = None,
    ):
        self.user_id = user_id
        self.email = email
        self.admin = admin
        self.domains = domains or projects or []
        self.databases = databases or []
        self.documents = documents or []
        self.apis = apis or []

    @property
    def projects(self) -> list[str]:
        """Backwards compatibility alias for domains."""
        return self.domains

    @projects.setter
    def projects(self, value: list[str]) -> None:
        """Backwards compatibility alias for domains."""
        self.domains = value

    def can_access_domain(self, domain_filename: str) -> bool:
        """Check if user can access a specific domain."""
        if self.admin:
            return True
        return domain_filename in self.domains

    # Backwards compatibility alias
    can_access_project = can_access_domain

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
            "domains": self.domains,
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
            domains=config_perms.domains,
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
            "domains": perms.domains,
            "databases": perms.databases,
            "documents": perms.documents,
            "apis": perms.apis,
        })
    return result


def compute_effective_permissions(
    user_perms: Optional[UserPermissions],
    config: "Config",
    active_domains: Optional[list[str]] = None,
    permissions_configured: bool = True,
) -> dict[str, Optional[set[str]]]:
    """Compute effective allowed resources by merging permissions and active domains.

    Rules:
    - If no permissions configured (permissions_configured=False) → no filtering
    - If user is admin → no filtering
    - Otherwise, merge explicit permissions + active domain resources

    Args:
        user_perms: User's base permissions (None if no permissions configured)
        config: Config with domain definitions
        active_domains: Currently active domain IDs (if any)
        permissions_configured: Whether permissions are configured at all

    Returns:
        Dict with allowed_databases, allowed_apis, allowed_documents.
        None values mean no filtering.
    """

    # No permissions configured = everything available
    if not permissions_configured or user_perms is None:
        return {
            "allowed_databases": None,
            "allowed_apis": None,
            "allowed_documents": None,
        }

    # Admins see everything
    if user_perms.admin:
        return {
            "allowed_databases": None,
            "allowed_apis": None,
            "allowed_documents": None,
        }

    # Start with explicit permissions
    allowed_databases = set(user_perms.databases)
    allowed_apis = set(user_perms.apis)
    allowed_documents = set(user_perms.documents)

    # Add resources from active domains the user has access to
    active_domains = active_domains or []
    for domain_id in active_domains:
        # Check user has access to this domain
        if not user_perms.can_access_domain(domain_id):
            continue

        # Load domain and add all its resources
        domain = config.load_domain(domain_id)
        if domain:
            allowed_databases.update(domain.databases.keys())
            allowed_apis.update(domain.apis.keys())
            allowed_documents.update(domain.documents.keys())

    return {
        "allowed_databases": allowed_databases,
        "allowed_apis": allowed_apis,
        "allowed_documents": allowed_documents,
    }
