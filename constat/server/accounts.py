# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Personal account management: load, save, encrypt, convert to DocumentConfig."""

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

VALID_TYPES = {"imap", "drive", "calendar", "sharepoint"}
VALID_PROVIDERS = {"google", "microsoft"}
VALID_AUTH_TYPES = {"oauth2_refresh"}


@dataclass
class PersonalAccount:
    """A user-connected personal resource (email, drive, calendar, SharePoint)."""

    name: str
    type: str  # imap, drive, calendar, sharepoint
    provider: str  # google, microsoft
    display_name: str
    email: str
    auth_type: str = "oauth2_refresh"
    refresh_token: str = ""  # encrypted at rest
    created_at: str = ""
    active: bool = True
    oauth2_tenant_id: str | None = None
    options: dict = field(default_factory=dict)


def validate_account(account: PersonalAccount) -> None:
    """Validate a PersonalAccount, raising ValueError for invalid fields."""
    if not account.name:
        raise ValueError("Account name is required")
    if not account.name.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"Account name must be alphanumeric with hyphens/underscores: {account.name}")
    if account.type not in VALID_TYPES:
        raise ValueError(f"Invalid account type: {account.type} (must be one of {VALID_TYPES})")
    if account.provider not in VALID_PROVIDERS:
        raise ValueError(f"Invalid provider: {account.provider} (must be one of {VALID_PROVIDERS})")
    if not account.display_name:
        raise ValueError("Display name is required")
    if not account.email:
        raise ValueError("Email is required")
    if account.auth_type not in VALID_AUTH_TYPES:
        raise ValueError(f"Invalid auth_type: {account.auth_type} (must be one of {VALID_AUTH_TYPES})")
    # SharePoint requires microsoft provider
    if account.type == "sharepoint" and account.provider != "microsoft":
        raise ValueError("SharePoint accounts require microsoft provider")
    # Microsoft accounts need tenant_id for non-common tenants
    if account.provider == "microsoft" and account.type == "sharepoint" and not account.options.get("site_url"):
        raise ValueError("SharePoint accounts require site_url in options")


def _accounts_path(user_id: str, data_dir: Path | None = None) -> Path:
    """Get the path to a user's accounts.yaml file."""
    if data_dir is None:
        data_dir = Path(".constat")
    return data_dir / user_id / "accounts.yaml"


def load_user_accounts(user_id: str, data_dir: Path | None = None) -> dict[str, PersonalAccount]:
    """Load user accounts from .constat/{user_id}/accounts.yaml.

    Returns empty dict if file does not exist.
    Raises on parse errors (no silent swallowing).
    """
    path = _accounts_path(user_id, data_dir)
    if not path.exists():
        return {}

    raw = yaml.safe_load(path.read_text())
    if not raw or "accounts" not in raw:
        return {}

    accounts: dict[str, PersonalAccount] = {}
    for name, data in raw["accounts"].items():
        accounts[name] = PersonalAccount(
            name=name,
            type=data["type"],
            provider=data["provider"],
            display_name=data["display_name"],
            email=data["email"],
            auth_type=data.get("auth_type", "oauth2_refresh"),
            refresh_token=data.get("refresh_token", ""),
            created_at=data.get("created_at", ""),
            active=data.get("active", True),
            oauth2_tenant_id=data.get("oauth2_tenant_id"),
            options=data.get("options", {}),
        )
    return accounts


def save_user_accounts(
    user_id: str,
    accounts: dict[str, PersonalAccount],
    data_dir: Path | None = None,
) -> None:
    """Save user accounts to YAML."""
    path = _accounts_path(user_id, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized: dict = {"accounts": {}}
    for name, acct in accounts.items():
        entry: dict = {
            "type": acct.type,
            "provider": acct.provider,
            "display_name": acct.display_name,
            "email": acct.email,
            "auth_type": acct.auth_type,
            "refresh_token": acct.refresh_token,
            "created_at": acct.created_at,
            "active": acct.active,
            "options": acct.options,
        }
        if acct.oauth2_tenant_id is not None:
            entry["oauth2_tenant_id"] = acct.oauth2_tenant_id
        serialized["accounts"][name] = entry

    path.write_text(yaml.dump(serialized, default_flow_style=False, sort_keys=False))


def _derive_key(secret: str, user_id: str) -> bytes:
    """Derive a Fernet key from server secret + user_id via PBKDF2."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=user_id.encode(),
        iterations=100_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(secret.encode()))


def encrypt_token(token: str, secret: str, user_id: str) -> str:
    """Encrypt refresh token with Fernet (PBKDF2 key from secret+user_id)."""
    from cryptography.fernet import Fernet

    key = _derive_key(secret, user_id)
    return Fernet(key).encrypt(token.encode()).decode()


def decrypt_token(encrypted: str, secret: str, user_id: str) -> str:
    """Decrypt refresh token."""
    from cryptography.fernet import Fernet

    key = _derive_key(secret, user_id)
    return Fernet(key).decrypt(encrypted.encode()).decode()


def account_to_document_config(account: PersonalAccount, server_config) -> dict:
    """Convert a PersonalAccount to DocumentConfig kwargs dict.

    Maps account type+provider to the right config fields for DocumentConfig.
    """
    base: dict = {
        "type": _doc_type(account),
        "description": account.display_name,
        "auth_type": account.auth_type,
    }

    # Set OAuth client credentials from server config
    client_id, client_secret = _get_oauth_credentials(account.provider, server_config)
    base["oauth2_client_id"] = client_id
    base["oauth2_client_secret"] = client_secret

    # Decrypt refresh token if encryption secret is available
    if server_config.account_encryption_secret and account.refresh_token:
        # The refresh token stored in the account is the encrypted form;
        # pass it through as oauth2 token cache or similar mechanism
        pass

    # Provider-specific URL and settings
    if account.provider == "google":
        _apply_google_config(account, base)
    elif account.provider == "microsoft":
        _apply_microsoft_config(account, base, server_config)

    # Merge account options into base
    for key, value in account.options.items():
        if key not in base:
            base[key] = value

    return base


def _doc_type(account: PersonalAccount) -> str:
    """Map account type to DocumentConfig type string."""
    if account.type == "imap":
        return "auto"  # IMAP is detected from URL scheme
    return account.type


def _get_oauth_credentials(provider: str, server_config) -> tuple[Optional[str], Optional[str]]:
    """Get OAuth client_id and client_secret for a provider from server config.

    Prefers generalized oauth fields, falls back to email-specific fields.
    """
    if provider == "google":
        client_id = server_config.google_oauth_client_id or server_config.google_email_client_id
        client_secret = server_config.google_oauth_client_secret or server_config.google_email_client_secret
        return client_id, client_secret
    elif provider == "microsoft":
        client_id = server_config.microsoft_oauth_client_id or server_config.microsoft_email_client_id
        client_secret = server_config.microsoft_oauth_client_secret or server_config.microsoft_email_client_secret
        return client_id, client_secret
    return None, None


def _apply_google_config(account: PersonalAccount, base: dict) -> None:
    """Apply Google-specific config fields."""
    if account.type == "imap":
        base["url"] = "imaps://imap.gmail.com:993"
        base["username"] = account.email


def _apply_microsoft_config(account: PersonalAccount, base: dict, server_config) -> None:
    """Apply Microsoft-specific config fields."""
    tenant = account.oauth2_tenant_id or getattr(server_config, "microsoft_oauth_tenant_id", None) or "common"
    base["oauth2_tenant_id"] = tenant

    if account.type == "imap":
        base["url"] = "imaps://outlook.office365.com:993"
        base["username"] = account.email
    elif account.type == "sharepoint":
        base["site_url"] = account.options.get("site_url")


def now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()
