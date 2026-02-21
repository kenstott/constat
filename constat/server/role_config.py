# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Role-based visibility and write permission configuration."""

import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RoleDefinition(BaseModel):
    """Definition of a single platform role."""

    description: str = ""
    visibility: dict[str, bool] = Field(default_factory=dict)
    writes: dict[str, bool] = Field(default_factory=dict)
    feedback: dict[str, bool] = Field(default_factory=dict)


class RolesConfig(BaseModel):
    """Container for all role definitions."""

    roles: dict[str, RoleDefinition] = Field(default_factory=dict)

    def get_role(self, role_name: str) -> RoleDefinition:
        """Get a role definition by name. Returns empty definition for unknown roles."""
        return self.roles.get(role_name, RoleDefinition())

    def can_see(self, role_name: str, section: str) -> bool:
        """Check if a role has visibility for a section."""
        return self.get_role(role_name).visibility.get(section, False)

    def can_write(self, role_name: str, resource: str) -> bool:
        """Check if a role has write permission for a resource."""
        return self.get_role(role_name).writes.get(resource, False)


def load_roles_config(path: str | Path | None = None) -> RolesConfig:
    """Load role definitions from YAML.

    Args:
        path: Path to roles.yaml. Defaults to constat/server/roles.yaml.

    Returns:
        Parsed RolesConfig.

    Raises:
        FileNotFoundError: If the roles file doesn't exist.
    """
    if path is None:
        path = Path(__file__).parent / "roles.yaml"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Roles config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return RolesConfig.model_validate(data or {})


def require_write(resource: str):
    """FastAPI dependency factory that checks role write permission.

    Usage:
        @router.post("/glossary", dependencies=[Depends(require_write("glossary"))])

    Raises:
        HTTPException 403 if the user's role lacks write permission.
    """
    from constat.server.auth import CurrentUserId, CurrentUserEmail

    async def _check_write(
        request: Request,
        user_id: CurrentUserId,
        email: CurrentUserEmail,
    ) -> None:
        from constat.server.permissions import get_user_permissions

        roles_config: RolesConfig | None = getattr(request.app.state, "roles_config", None)
        if roles_config is None:
            return  # No role config loaded — allow (backwards compat)

        server_config = request.app.state.server_config
        perms = get_user_permissions(server_config, email=email or "", user_id=user_id)

        if not roles_config.can_write(perms.role, resource):
            raise HTTPException(
                status_code=403,
                detail=f"Role '{perms.role}' does not have write access to '{resource}'",
            )

    return _check_write
