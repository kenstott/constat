# Copyright (c) 2025 Kenneth Stott
# Canary: 26fbd04c-ad24-4380-927d-55629a423904
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""OAuth token management — separate token storage from source config."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from constat.core.paths import user_vault_dir

logger = logging.getLogger(__name__)

_CONSTAT_DIR = Path(".constat")


def load_user_tokens(user_id: str) -> dict:
    """Load OAuth tokens from .constat/{user_id}.vault/tokens.yaml."""
    path = user_vault_dir(_CONSTAT_DIR, user_id) / "tokens.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("tokens", {})


def save_user_tokens(user_id: str, tokens: dict) -> None:
    """Save OAuth tokens to .constat/{user_id}.vault/tokens.yaml."""
    path = user_vault_dir(_CONSTAT_DIR, user_id) / "tokens.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump({"tokens": tokens}, default_flow_style=False, sort_keys=False))


def inject_oauth_tokens(user_id: str, documents: dict, server_config=None) -> dict:
    """Resolve oauth2_token_ref entries by loading tokens.yaml.

    Injects server's OAuth client credentials from server_config.
    Modifies documents dict in place and returns it.
    """
    tokens = load_user_tokens(user_id)
    if not tokens:
        return documents
    for name, doc in documents.items():
        ref = doc.get("oauth2_token_ref")
        if not ref or ref not in tokens:
            continue
        token_entry = tokens[ref]
        provider = token_entry.get("provider", "")
        if server_config and provider == "google":
            doc["oauth2_client_id"] = getattr(server_config, "google_email_client_id", "")
            doc["oauth2_client_secret"] = token_entry["refresh_token"]
            doc["password"] = getattr(server_config, "google_email_client_secret", "")
        elif server_config and provider == "microsoft":
            doc["oauth2_client_id"] = getattr(server_config, "microsoft_email_client_id", "")
            doc["oauth2_client_secret"] = token_entry["refresh_token"]
            doc["oauth2_tenant_id"] = token_entry.get("tenant_id") or getattr(server_config, "microsoft_email_tenant_id", "")
        else:
            # Fallback: inject refresh token directly
            doc["oauth2_client_secret"] = token_entry.get("refresh_token", "")
        doc["auth_type"] = "oauth2_refresh"
    return documents


def migrate_email_tokens(user_id: str) -> int:
    """Extract inline OAuth tokens from user config to tokens.yaml.

    Returns number of tokens migrated.
    """
    config_path = user_vault_dir(_CONSTAT_DIR, user_id) / "config.yaml"
    if not config_path.exists():
        return 0

    config = yaml.safe_load(config_path.read_text()) or {}
    documents = config.get("documents", {})
    tokens = load_user_tokens(user_id)
    count = 0

    for name, doc in documents.items():
        auth_type = doc.get("auth_type", "")
        has_secret = doc.get("oauth2_client_secret")
        if auth_type not in ("oauth2_refresh", "oauth2") or not has_secret:
            continue
        if doc.get("oauth2_token_ref"):
            continue  # Already migrated

        tokens[name] = {
            "provider": "microsoft" if doc.get("oauth2_tenant_id") else "google",
            "email": doc.get("username", ""),
            "refresh_token": doc["oauth2_client_secret"],
            "tenant_id": doc.get("oauth2_tenant_id"),
            "scopes": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        doc.pop("oauth2_client_secret", None)
        doc.pop("oauth2_client_id", None)
        doc["oauth2_token_ref"] = name
        count += 1

    if count > 0:
        save_user_tokens(user_id, tokens)
        config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
        logger.info(f"Migrated {count} email OAuth tokens for user {user_id}")

    return count


def migrate_user_config_source_tags(user_id: str) -> int:
    """Rename source='session' to source='user' in user config.

    Returns number of entries changed.
    """
    config_path = user_vault_dir(_CONSTAT_DIR, user_id) / "config.yaml"
    if not config_path.exists():
        return 0

    config = yaml.safe_load(config_path.read_text()) or {}
    count = 0

    for section in ("databases", "documents", "apis"):
        for name, entry in config.get(section, {}).items():
            if isinstance(entry, dict) and entry.get("source") == "session":
                entry["source"] = "user"
                count += 1

    if count > 0:
        config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))
        logger.info(f"Migrated {count} source tags session->user for user {user_id}")

    return count
