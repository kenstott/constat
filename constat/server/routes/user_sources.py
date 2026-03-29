"""User source management API — list, remove, update user-scoped sources."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException

from constat.server.config import user_vault_dir

logger = logging.getLogger(__name__)

router = APIRouter(tags=["user-sources"])

_CONSTAT_DIR = Path(".constat")


def _load_user_config(user_id: str) -> dict:
    config_path = user_vault_dir(_CONSTAT_DIR, user_id) / "config.yaml"
    if not config_path.exists():
        return {}
    return yaml.safe_load(config_path.read_text()) or {}


def _save_user_config(user_id: str, config: dict) -> None:
    config_path = user_vault_dir(_CONSTAT_DIR, user_id) / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


@router.get("/users/{user_id}/sources")
async def list_user_sources(user_id: str) -> dict:
    """List all user-scoped sources from .constat/{user_id}/config.yaml."""
    config = _load_user_config(user_id)
    return {
        "databases": {k: v for k, v in config.get("databases", {}).items() if v.get("source") == "user"},
        "documents": {k: v for k, v in config.get("documents", {}).items() if v.get("source") == "user"},
        "apis": {k: v for k, v in config.get("apis", {}).items() if v.get("source") == "user"},
    }


@router.delete("/users/{user_id}/sources/{source_type}/{name}")
async def remove_user_source(user_id: str, source_type: str, name: str) -> dict:
    """Remove a user-scoped source from config.yaml."""
    if source_type not in ("databases", "documents", "apis"):
        raise HTTPException(status_code=400, detail=f"Invalid source type: {source_type}")

    config = _load_user_config(user_id)
    section = config.get(source_type, {})
    if name not in section:
        raise HTTPException(status_code=404, detail=f"Source '{name}' not found in {source_type}")

    del section[name]
    config[source_type] = section
    _save_user_config(user_id, config)
    return {"status": "removed", "name": name, "source_type": source_type}


@router.put("/users/{user_id}/sources/{source_type}/{name}")
async def update_user_source(user_id: str, source_type: str, name: str, body: dict) -> dict:
    """Update a user-scoped source (promote/demote, change fields)."""
    if source_type not in ("databases", "documents", "apis"):
        raise HTTPException(status_code=400, detail=f"Invalid source type: {source_type}")

    config = _load_user_config(user_id)
    section = config.get(source_type, {})

    if name not in section:
        # Promote: add new entry
        section[name] = body
    else:
        # Update existing
        section[name].update(body)

    section[name]["source"] = "user"
    config[source_type] = section
    _save_user_config(user_id, config)
    return {"status": "updated", "name": name, "source_type": source_type}
