# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tier management REST endpoints.

Provides promote, remove, and create actions for config items across tiers.
Each config item (fact, glossary term, relationship, learning rule, right)
lives at a specific tier. These endpoints move items between tiers.

Promotion rules:
- session → user: any authenticated user
- user → user_domain: any authenticated user (for their active domains)
- user → system: admin only
- user_domain → system_domain: admin or domain owner
- system_domain → system: admin only

Remove: deletes the item from its current tier (lower-tier value resurfaces).
Create: adds a new item at the specified tier.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from constat.core.tiered_config import ConfigSource
from constat.server.auth import CurrentUserId, CurrentUserEmail
from constat.server.permissions import get_user_permissions
from constat.server.role_config import require_write
from constat.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Sections that support tier management
MANAGEABLE_SECTIONS = {
    "facts", "glossary", "relationships", "learnings", "rights",
}

# Promotion paths: source_tier → target_tier
PROMOTION_MAP = {
    ConfigSource.SESSION: ConfigSource.USER,
    ConfigSource.USER: ConfigSource.SYSTEM,
    ConfigSource.USER_DOMAIN: ConfigSource.SYSTEM_DOMAIN,
    ConfigSource.SYSTEM_DOMAIN: ConfigSource.SYSTEM,
}


def get_session_manager(request: Request) -> SessionManager:
    """Dependency to get session manager from app state."""
    return request.app.state.session_manager


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TierItemRequest(BaseModel):
    """Request body for promote/remove/create operations."""
    section: str = Field(description="Config section: facts, glossary, relationships, learnings, rights")
    key: str = Field(description="Item key within the section")
    value: Any = Field(default=None, description="Item value (required for create, optional for promote)")
    target_tier: Optional[str] = Field(default=None, description="Target tier for promote/create (system, system_domain, user, user_domain, session)")


class TierItemResponse(BaseModel):
    """Response for tier management operations."""
    status: str = Field(description="ok or error")
    section: str
    key: str
    tier: str = Field(description="Tier the item now lives at")
    re_resolved: bool = Field(default=True, description="Whether config was re-resolved")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tier(tier_str: str) -> ConfigSource:
    """Parse a tier string into ConfigSource enum."""
    try:
        return ConfigSource(tier_str)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid tier: {tier_str}. Must be one of: {[s.value for s in ConfigSource]}")


def _get_tier_yaml_path(
    tier: ConfigSource,
    server_config: Any,
    user_id: str,
    config_dir: Optional[str],
    domain_name: Optional[str] = None,
) -> Path:
    """Get the YAML config file path for a given tier."""
    if tier == ConfigSource.SYSTEM:
        if not config_dir:
            raise HTTPException(status_code=400, detail="No config_dir configured for system tier")
        return Path(config_dir) / "config.yaml"

    elif tier == ConfigSource.SYSTEM_DOMAIN:
        if not config_dir or not domain_name:
            raise HTTPException(status_code=400, detail="config_dir and domain_name required for system_domain tier")
        return Path(config_dir) / "domains" / domain_name / "config.yaml"

    elif tier == ConfigSource.USER:
        return server_config.data_dir / user_id / "config.yaml"

    elif tier == ConfigSource.USER_DOMAIN:
        if not domain_name:
            raise HTTPException(status_code=400, detail="domain_name required for user_domain tier")
        return server_config.data_dir / user_id / "domains" / domain_name / "config.yaml"

    elif tier == ConfigSource.SESSION:
        raise HTTPException(status_code=400, detail="Session tier items are in-memory only and cannot be persisted via tier management")

    raise HTTPException(status_code=400, detail=f"Unsupported tier: {tier.value}")


def _load_tier_yaml(path: Path) -> dict:
    """Load a tier YAML file, returning empty dict if not found."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _save_tier_yaml(path: Path, data: dict) -> None:
    """Save data to a tier YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _check_promotion_permission(
    current_tier: ConfigSource,
    target_tier: ConfigSource,
    is_admin: bool,
    is_domain_owner: bool,
) -> None:
    """Check if the user has permission to promote to target tier."""
    if target_tier in (ConfigSource.SYSTEM, ConfigSource.SYSTEM_DOMAIN):
        if target_tier == ConfigSource.SYSTEM and not is_admin:
            raise HTTPException(status_code=403, detail="Admin access required to promote to system tier")
        if target_tier == ConfigSource.SYSTEM_DOMAIN and not (is_admin or is_domain_owner):
            raise HTTPException(status_code=403, detail="Admin or domain owner access required to promote to system_domain tier")


def _check_remove_permission(
    item_tier: ConfigSource,
    is_admin: bool,
    is_domain_owner: bool,
) -> None:
    """Check if the user has permission to remove from this tier."""
    if item_tier == ConfigSource.SYSTEM and not is_admin:
        raise HTTPException(status_code=403, detail="Admin access required to remove from system tier")
    if item_tier == ConfigSource.SYSTEM_DOMAIN and not (is_admin or is_domain_owner):
        raise HTTPException(status_code=403, detail="Admin or domain owner access required to remove from system_domain tier")


def _is_domain_owner(config: Any, domain_name: str, email: str) -> bool:
    """Check if user is the owner of a domain."""
    domain = config.load_domain(domain_name)
    if not domain:
        return False
    return domain.owner and domain.owner == email


# ---------------------------------------------------------------------------
# GET resolved config
# ---------------------------------------------------------------------------

@router.get("/{session_id}/config/resolved")
async def get_resolved_config(
    session_id: str,
    user_id: CurrentUserId,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict:
    """Get the fully resolved config with tier attribution.

    Returns all manageable sections with their values and the tier
    each value came from. Used by the UI to render tier badges.

    Args:
        session_id: Session ID
        user_id: Authenticated user ID
        session_manager: Injected session manager

    Returns:
        Dict with sections and attribution
    """
    managed = session_manager.get_session(session_id)
    rc = managed.resolved_config

    if not rc:
        # Resolve now if not yet done
        rc = session_manager.resolve_config(session_id)

    if not rc:
        raise HTTPException(status_code=500, detail="Could not resolve config")

    # Build response with attribution for each item
    result: dict[str, Any] = {
        "active_domains": rc.active_domains,
        "sections": {},
    }

    for section in MANAGEABLE_SECTIONS:
        section_data = getattr(rc, section, {})
        items = {}
        for key, value in (section_data or {}).items():
            tier_source = rc._attribution.get(f"{section}.{key}")
            items[key] = {
                "value": value,
                "tier": tier_source.value if tier_source else None,
            }
        result["sections"][section] = items

    return result


# ---------------------------------------------------------------------------
# POST promote
# ---------------------------------------------------------------------------

@router.post("/{session_id}/config/promote", response_model=TierItemResponse, dependencies=[Depends(require_write("tier_promote"))])
async def promote_item(
    session_id: str,
    body: TierItemRequest,
    request: Request,
    user_id: CurrentUserId,
    email: str = Depends(CurrentUserEmail),
    session_manager: SessionManager = Depends(get_session_manager),
) -> TierItemResponse:
    """Promote a config item to a higher tier.

    Copies the item's value to the target tier's YAML file.
    The item remains at the source tier as well (higher tier wins on merge).

    Args:
        session_id: Session ID
        body: Promote request with section, key, and optional target_tier
        request: FastAPI request
        user_id: Authenticated user ID
        email: Authenticated user email
        session_manager: Injected session manager

    Returns:
        Promotion result
    """
    if body.section not in MANAGEABLE_SECTIONS:
        raise HTTPException(status_code=400, detail=f"Section '{body.section}' is not manageable. Must be one of: {sorted(MANAGEABLE_SECTIONS)}")

    managed = session_manager.get_session(session_id)
    rc = managed.resolved_config
    if not rc:
        raise HTTPException(status_code=400, detail="No resolved config available")

    # Determine current tier of the item
    current_tier_source = rc._attribution.get(f"{body.section}.{body.key}")
    if not current_tier_source:
        raise HTTPException(status_code=404, detail=f"Item '{body.section}.{body.key}' not found in resolved config")

    # Determine target tier
    if body.target_tier:
        target_tier = _parse_tier(body.target_tier)
    else:
        # Default: promote one level up
        target_tier = PROMOTION_MAP.get(current_tier_source)
        if not target_tier:
            raise HTTPException(status_code=400, detail=f"Cannot promote from {current_tier_source.value} (already at highest tier or no promotion path)")

    # Permission check
    server_config = request.app.state.server_config
    perms = get_user_permissions(server_config, email=email or "", user_id=user_id)
    config = request.app.state.config

    # Determine domain name from active domains (use first if multiple)
    domain_name = managed.active_domains[0] if managed.active_domains else None
    is_owner = _is_domain_owner(config, domain_name, email) if domain_name and email else False

    _check_promotion_permission(current_tier_source, target_tier, perms.admin, is_owner)

    # Get the item value
    section_data = getattr(rc, body.section, {})
    value = body.value if body.value is not None else section_data.get(body.key)
    if value is None:
        raise HTTPException(status_code=404, detail=f"No value found for '{body.section}.{body.key}'")

    # Write to target tier YAML
    yaml_path = _get_tier_yaml_path(target_tier, server_config, managed.user_id, config.config_dir, domain_name)
    tier_data = _load_tier_yaml(yaml_path)
    if body.section not in tier_data:
        tier_data[body.section] = {}
    tier_data[body.section][body.key] = value
    _save_tier_yaml(yaml_path, tier_data)

    logger.info(f"Promoted {body.section}.{body.key} to {target_tier.value} (path={yaml_path})")

    # Re-resolve config
    session_manager.resolve_config(session_id)

    # Trigger entity rebuild if glossary or relationships changed
    if body.section in ("glossary", "relationships"):
        session_manager.refresh_entities_async(session_id)

    return TierItemResponse(
        status="ok",
        section=body.section,
        key=body.key,
        tier=target_tier.value,
    )


# ---------------------------------------------------------------------------
# DELETE remove
# ---------------------------------------------------------------------------

@router.post("/{session_id}/config/remove", response_model=TierItemResponse, dependencies=[Depends(require_write("tier_promote"))])
async def remove_item(
    session_id: str,
    body: TierItemRequest,
    request: Request,
    user_id: CurrentUserId,
    email: str = Depends(CurrentUserEmail),
    session_manager: SessionManager = Depends(get_session_manager),
) -> TierItemResponse:
    """Remove a config item from its current tier.

    Sets the key to null in the tier's YAML file (null-deletion).
    If a lower tier has a value for this key, it will resurface after re-resolve.

    Args:
        session_id: Session ID
        body: Remove request with section and key
        request: FastAPI request
        user_id: Authenticated user ID
        email: Authenticated user email
        session_manager: Injected session manager

    Returns:
        Removal result
    """
    if body.section not in MANAGEABLE_SECTIONS:
        raise HTTPException(status_code=400, detail=f"Section '{body.section}' is not manageable. Must be one of: {sorted(MANAGEABLE_SECTIONS)}")

    managed = session_manager.get_session(session_id)
    rc = managed.resolved_config
    if not rc:
        raise HTTPException(status_code=400, detail="No resolved config available")

    # Determine which tier the item is at
    current_tier_source = rc._attribution.get(f"{body.section}.{body.key}")
    if not current_tier_source:
        raise HTTPException(status_code=404, detail=f"Item '{body.section}.{body.key}' not found in resolved config")

    # Use explicit target_tier if given, otherwise remove from current tier
    remove_tier = _parse_tier(body.target_tier) if body.target_tier else current_tier_source

    # Permission check
    server_config = request.app.state.server_config
    perms = get_user_permissions(server_config, email=email or "", user_id=user_id)
    config = request.app.state.config

    domain_name = managed.active_domains[0] if managed.active_domains else None
    is_owner = _is_domain_owner(config, domain_name, email) if domain_name and email else False

    _check_remove_permission(remove_tier, perms.admin, is_owner)

    # Write null to the tier's YAML (null-deletion semantics)
    yaml_path = _get_tier_yaml_path(remove_tier, server_config, managed.user_id, config.config_dir, domain_name)
    tier_data = _load_tier_yaml(yaml_path)
    if body.section not in tier_data:
        tier_data[body.section] = {}
    tier_data[body.section][body.key] = None
    _save_tier_yaml(yaml_path, tier_data)

    logger.info(f"Removed {body.section}.{body.key} from {remove_tier.value} (null-deletion at {yaml_path})")

    # Re-resolve config
    session_manager.resolve_config(session_id)

    # Trigger entity rebuild if glossary or relationships changed
    if body.section in ("glossary", "relationships"):
        session_manager.refresh_entities_async(session_id)

    return TierItemResponse(
        status="ok",
        section=body.section,
        key=body.key,
        tier=remove_tier.value,
    )


# ---------------------------------------------------------------------------
# POST create
# ---------------------------------------------------------------------------

@router.post("/{session_id}/config/create", response_model=TierItemResponse, dependencies=[Depends(require_write("tier_promote"))])
async def create_item(
    session_id: str,
    body: TierItemRequest,
    request: Request,
    user_id: CurrentUserId,
    email: str = Depends(CurrentUserEmail),
    session_manager: SessionManager = Depends(get_session_manager),
) -> TierItemResponse:
    """Create a new config item at a specific tier.

    Writes the item to the tier's YAML file. Defaults to user tier.

    Args:
        session_id: Session ID
        body: Create request with section, key, value, and optional target_tier
        request: FastAPI request
        user_id: Authenticated user ID
        email: Authenticated user email
        session_manager: Injected session manager

    Returns:
        Creation result
    """
    if body.section not in MANAGEABLE_SECTIONS:
        raise HTTPException(status_code=400, detail=f"Section '{body.section}' is not manageable. Must be one of: {sorted(MANAGEABLE_SECTIONS)}")

    if body.value is None:
        raise HTTPException(status_code=400, detail="value is required for create")

    managed = session_manager.get_session(session_id)

    # Default to user tier
    target_tier = _parse_tier(body.target_tier) if body.target_tier else ConfigSource.USER

    # Permission check
    server_config = request.app.state.server_config
    perms = get_user_permissions(server_config, email=email or "", user_id=user_id)
    config = request.app.state.config

    domain_name = managed.active_domains[0] if managed.active_domains else None
    is_owner = _is_domain_owner(config, domain_name, email) if domain_name and email else False

    _check_promotion_permission(ConfigSource.USER, target_tier, perms.admin, is_owner)

    # Write to target tier YAML
    yaml_path = _get_tier_yaml_path(target_tier, server_config, managed.user_id, config.config_dir, domain_name)
    tier_data = _load_tier_yaml(yaml_path)
    if body.section not in tier_data:
        tier_data[body.section] = {}
    tier_data[body.section][body.key] = body.value
    _save_tier_yaml(yaml_path, tier_data)

    logger.info(f"Created {body.section}.{body.key} at {target_tier.value} (path={yaml_path})")

    # Re-resolve config
    session_manager.resolve_config(session_id)

    # Trigger entity rebuild if glossary or relationships changed
    if body.section in ("glossary", "relationships"):
        session_manager.refresh_entities_async(session_id)

    return TierItemResponse(
        status="ok",
        section=body.section,
        key=body.key,
        tier=target_tier.value,
    )