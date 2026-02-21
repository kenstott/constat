# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""User permissions management.

Permissions are defined in the config.yaml under the 'permissions' section:

```yaml
permissions:
  users:
    8TgdzQHw7EbTHSJY9osIuCElbGF2:
      persona: platform_admin
      domains: []
      databases: []
      documents: []
      apis: []

    xK9mPqR2wYnZaB4cD7eF1gH3iJ5:
      persona: domain_user
      domains:
        - sales-analytics.yaml
      databases:
        - sales
        - inventory
      documents: []
      apis: []

  default:
    persona: viewer
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
        persona: str = "viewer",
        domains: Optional[list[str]] = None,
        databases: Optional[list[str]] = None,
        documents: Optional[list[str]] = None,
        apis: Optional[list[str]] = None,
    ):
        self.user_id = user_id
        self.email = email
        self.persona = persona
        self.domains = domains or []
        self.databases = databases or []
        self.documents = documents or []
        self.apis = apis or []

    @property
    def is_admin(self) -> bool:
        """Derived admin status from persona."""
        return self.persona == "platform_admin"

    def can_access_domain(self, domain_filename: str) -> bool:
        """Check if user can access a specific domain."""
        if self.is_admin:
            return True
        return domain_filename in self.domains

    def can_access_database(self, db_name: str) -> bool:
        """Check if user can access a specific database."""
        if self.is_admin:
            return True
        return db_name in self.databases

    def can_access_document(self, doc_name: str) -> bool:
        """Check if user can access a specific document."""
        if self.is_admin:
            return True
        return doc_name in self.documents

    def can_access_api(self, api_name: str) -> bool:
        """Check if user can access a specific API."""
        if self.is_admin:
            return True
        return api_name in self.apis

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "admin": self.is_admin,
            "persona": self.persona,
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
            persona=config_perms.persona,
            domains=config_perms.domains,
            databases=config_perms.databases,
            documents=config_perms.documents,
            apis=config_perms.apis,
        )


def get_user_permissions(
    server_config: ServerConfig,
    user_id: str,
    email: str = "",
) -> UserPermissions:
    """Get permissions for a user from server config.

    Args:
        server_config: Server configuration containing permissions
        user_id: Stable user identifier (Firebase UID, etc.)
        email: User's email address (informational, not used for lookup)

    Returns:
        UserPermissions object
    """
    config_perms = server_config.permissions.get_user_permissions(user_id=user_id)
    return UserPermissions.from_config(config_perms, user_id=user_id, email=email)


def list_all_permissions(server_config: ServerConfig) -> list[dict[str, Any]]:
    """List all users with explicit permissions.

    Args:
        server_config: Server configuration containing permissions

    Returns:
        List of user permission dicts
    """
    result = []
    for user_id, perms in server_config.permissions.users.items():
        result.append({
            "user_id": user_id,
            "admin": perms.persona == "platform_admin",
            "persona": perms.persona,
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
    """Compute effective allowed resources using least-privilege intersection.

    Rules:
    - If no permissions configured (permissions_configured=False) -> no filtering
    - If user is admin -> no filtering
    - Otherwise, start with user's global permissions, then INTERSECT with
      domain-level restrictions (if a domain has its own permissions.yaml).
    - If a domain has no permissions.yaml, all of that domain's resources are
      available (no additional restriction from the domain side).
    - Final allowed = resources the user has global permission for AND that
      aren't restricted by domain-scoped permissions.

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
    if user_perms.is_admin:
        return {
            "allowed_databases": None,
            "allowed_apis": None,
            "allowed_documents": None,
        }

    # Start with user's global permissions
    allowed_databases = set(user_perms.databases)
    allowed_apis = set(user_perms.apis)
    allowed_documents = set(user_perms.documents)

    # Process active domains
    active_domains = active_domains or []
    for domain_id in active_domains:
        # Check user has access to this domain
        if not user_perms.can_access_domain(domain_id):
            continue

        domain = config.load_domain(domain_id)
        if not domain:
            continue

        domain_databases = set(domain.databases.keys())
        domain_apis = set(domain.apis.keys())
        domain_documents = set(domain.documents.keys())

        # Check if domain has its own permissions.yaml
        if domain.permissions is not None:
            # Domain has scoped permissions — intersect with domain restrictions
            domain_perms = domain.permissions.get_user_permissions(user_id=user_perms.user_id)
            domain_allowed_dbs = set(domain_perms.databases)
            domain_allowed_apis = set(domain_perms.apis)
            domain_allowed_docs = set(domain_perms.documents)

            # User gets domain resources that are in BOTH their global permissions
            # and the domain's allowed list (least privilege)
            allowed_databases.update(domain_databases & domain_allowed_dbs)
            allowed_apis.update(domain_apis & domain_allowed_apis)
            allowed_documents.update(domain_documents & domain_allowed_docs)
        else:
            # No domain permissions — all domain resources available
            # but still intersected with user's global permissions implicitly
            # (user must have global permission OR domain grants access)
            allowed_databases.update(domain_databases)
            allowed_apis.update(domain_apis)
            allowed_documents.update(domain_documents)

    return {
        "allowed_databases": allowed_databases,
        "allowed_apis": allowed_apis,
        "allowed_documents": allowed_documents,
    }
