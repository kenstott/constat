# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Persona-based visibility and write permission configuration."""

import logging
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PersonaDefinition(BaseModel):
    """Definition of a single platform persona."""

    description: str = ""
    visibility: dict[str, bool] = Field(default_factory=dict)
    writes: dict[str, bool] = Field(default_factory=dict)
    feedback: dict[str, bool] = Field(default_factory=dict)


class PersonasConfig(BaseModel):
    """Container for all persona definitions."""

    personas: dict[str, PersonaDefinition] = Field(default_factory=dict)

    def get_persona(self, persona_name: str) -> PersonaDefinition:
        """Get a persona definition by name. Returns empty definition for unknown personas."""
        return self.personas.get(persona_name, PersonaDefinition())

    def can_see(self, persona_name: str, section: str) -> bool:
        """Check if a persona has visibility for a section."""
        return self.get_persona(persona_name).visibility.get(section, False)

    def can_write(self, persona_name: str, resource: str) -> bool:
        """Check if a persona has write permission for a resource."""
        return self.get_persona(persona_name).writes.get(resource, False)


def load_personas_config(path: str | Path | None = None) -> PersonasConfig:
    """Load persona definitions from YAML.

    Args:
        path: Path to personas.yaml. Defaults to constat/server/personas.yaml.

    Returns:
        Parsed PersonasConfig.

    Raises:
        FileNotFoundError: If the personas file doesn't exist.
    """
    if path is None:
        path = Path(__file__).parent / "personas.yaml"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Personas config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return PersonasConfig.model_validate(data or {})


def require_write(resource: str):
    """FastAPI dependency factory that checks persona write permission.

    Usage:
        @router.post("/glossary", dependencies=[Depends(require_write("glossary"))])

    Raises:
        HTTPException 403 if the user's persona lacks write permission.
    """
    from constat.server.auth import CurrentUserId, CurrentUserEmail

    async def _check_write(
        request: Request,
        user_id: CurrentUserId,
        email: CurrentUserEmail,
    ) -> None:
        from constat.server.permissions import get_user_permissions

        server_config = request.app.state.server_config

        if server_config.auth_disabled:
            return

        personas_config: PersonasConfig | None = getattr(request.app.state, "personas_config", None)
        if personas_config is None:
            return  # No persona config loaded — allow (backwards compat)

        perms = get_user_permissions(server_config, user_id=user_id, email=email or "")

        if not personas_config.can_write(perms.persona, resource):
            raise HTTPException(
                status_code=403,
                detail=f"Persona '{perms.persona}' does not have write access to '{resource}'",
            )

    return _check_write
