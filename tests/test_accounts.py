# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for personal account management (accounts.py)."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from constat.server.accounts import (
    PersonalAccount,
    account_to_document_config,
    decrypt_token,
    encrypt_token,
    load_user_accounts,
    now_iso,
    save_user_accounts,
    validate_account,
)


# --- Token encryption/decryption ---


class TestTokenEncryption:
    def test_round_trip(self):
        token = "ya29.a0AfH6SMA_refresh_token_value"
        secret = "my-server-secret-key-1234"
        user_id = "user123"

        encrypted = encrypt_token(token, secret, user_id)
        assert encrypted != token
        decrypted = decrypt_token(encrypted, secret, user_id)
        assert decrypted == token

    def test_different_users_different_ciphertext(self):
        token = "same-token"
        secret = "same-secret"

        enc1 = encrypt_token(token, secret, "user-a")
        enc2 = encrypt_token(token, secret, "user-b")
        assert enc1 != enc2

        # Each decrypts correctly with its own user_id
        assert decrypt_token(enc1, secret, "user-a") == token
        assert decrypt_token(enc2, secret, "user-b") == token

    def test_wrong_secret_fails(self):
        token = "my-token"
        encrypted = encrypt_token(token, "correct-secret", "user1")
        with pytest.raises(Exception):
            decrypt_token(encrypted, "wrong-secret", "user1")

    def test_wrong_user_fails(self):
        token = "my-token"
        encrypted = encrypt_token(token, "secret", "user1")
        with pytest.raises(Exception):
            decrypt_token(encrypted, "secret", "user2")

    def test_empty_token(self):
        encrypted = encrypt_token("", "secret", "user1")
        assert decrypt_token(encrypted, "secret", "user1") == ""


# --- Load/save accounts YAML ---


class TestLoadSaveAccounts:
    def test_round_trip(self, tmp_path):
        accounts = {
            "my-gmail": PersonalAccount(
                name="my-gmail",
                type="imap",
                provider="google",
                display_name="Gmail (test@gmail.com)",
                email="test@gmail.com",
                refresh_token="encrypted-token-1",
                created_at="2026-03-24T10:00:00Z",
                active=True,
                options={"mailbox": "INBOX", "max_messages": 500},
            ),
            "work-cal": PersonalAccount(
                name="work-cal",
                type="calendar",
                provider="microsoft",
                display_name="Outlook Calendar",
                email="test@company.com",
                refresh_token="encrypted-token-2",
                created_at="2026-03-24T10:10:00Z",
                active=False,
                oauth2_tenant_id="abc-123",
                options={"calendar_id": "primary"},
            ),
        }

        save_user_accounts("testuser", accounts, tmp_path)
        loaded = load_user_accounts("testuser", tmp_path)

        assert len(loaded) == 2
        assert "my-gmail" in loaded
        assert "work-cal" in loaded

        gmail = loaded["my-gmail"]
        assert gmail.type == "imap"
        assert gmail.provider == "google"
        assert gmail.display_name == "Gmail (test@gmail.com)"
        assert gmail.email == "test@gmail.com"
        assert gmail.active is True
        assert gmail.options["mailbox"] == "INBOX"

        cal = loaded["work-cal"]
        assert cal.type == "calendar"
        assert cal.provider == "microsoft"
        assert cal.active is False
        assert cal.oauth2_tenant_id == "abc-123"

    def test_load_nonexistent_returns_empty(self, tmp_path):
        result = load_user_accounts("nonexistent", tmp_path)
        assert result == {}

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "emptyuser" / "accounts.yaml"
        path.parent.mkdir(parents=True)
        path.write_text("")
        result = load_user_accounts("emptyuser", tmp_path)
        assert result == {}

    def test_save_creates_directory(self, tmp_path):
        accounts = {
            "test": PersonalAccount(
                name="test",
                type="imap",
                provider="google",
                display_name="Test",
                email="test@test.com",
            ),
        }
        save_user_accounts("newuser", accounts, tmp_path)
        assert (tmp_path / "newuser" / "accounts.yaml").exists()

    def test_tenant_id_preserved(self, tmp_path):
        accounts = {
            "ms-account": PersonalAccount(
                name="ms-account",
                type="email",
                provider="microsoft",
                display_name="Outlook",
                email="test@company.com",
                oauth2_tenant_id="tenant-xyz",
            ),
        }
        save_user_accounts("user1", accounts, tmp_path)
        loaded = load_user_accounts("user1", tmp_path)
        assert loaded["ms-account"].oauth2_tenant_id == "tenant-xyz"


# --- account_to_document_config ---


def _mock_server_config(**overrides):
    """Create a mock server config with defaults."""
    cfg = MagicMock()
    cfg.google_oauth_client_id = overrides.get("google_oauth_client_id", "google-client-id")
    cfg.google_oauth_client_secret = overrides.get("google_oauth_client_secret", "google-secret")
    cfg.google_email_client_id = overrides.get("google_email_client_id", None)
    cfg.google_email_client_secret = overrides.get("google_email_client_secret", None)
    cfg.microsoft_oauth_client_id = overrides.get("microsoft_oauth_client_id", "ms-client-id")
    cfg.microsoft_oauth_client_secret = overrides.get("microsoft_oauth_client_secret", "ms-secret")
    cfg.microsoft_email_client_id = overrides.get("microsoft_email_client_id", None)
    cfg.microsoft_email_client_secret = overrides.get("microsoft_email_client_secret", None)
    cfg.microsoft_oauth_tenant_id = overrides.get("microsoft_oauth_tenant_id", "common")
    cfg.account_encryption_secret = overrides.get("account_encryption_secret", None)
    return cfg


class TestAccountToDocumentConfig:
    def test_imap_google(self):
        account = PersonalAccount(
            name="gmail",
            type="imap",
            provider="google",
            display_name="Gmail",
            email="user@gmail.com",
            options={"mailbox": "INBOX", "max_messages": 100},
        )
        result = account_to_document_config(account, _mock_server_config())

        assert result["url"] == "imaps://imap.gmail.com:993"
        assert result["username"] == "user@gmail.com"
        assert result["auth_type"] == "oauth2_refresh"
        assert result["oauth2_client_id"] == "google-client-id"
        assert result["oauth2_client_secret"] == "google-secret"
        assert result["description"] == "Gmail"
        assert result["mailbox"] == "INBOX"
        assert result["max_messages"] == 100

    def test_imap_microsoft(self):
        account = PersonalAccount(
            name="outlook",
            type="imap",
            provider="microsoft",
            display_name="Outlook",
            email="user@company.com",
            oauth2_tenant_id="tenant-abc",
        )
        result = account_to_document_config(account, _mock_server_config())

        assert result["url"] == "imaps://outlook.office365.com:993"
        assert result["username"] == "user@company.com"
        assert result["oauth2_tenant_id"] == "tenant-abc"
        assert result["oauth2_client_id"] == "ms-client-id"

    def test_drive_google(self):
        account = PersonalAccount(
            name="gdrive",
            type="drive",
            provider="google",
            display_name="Google Drive",
            email="user@gmail.com",
            options={"folder_id": "abc123"},
        )
        result = account_to_document_config(account, _mock_server_config())

        assert result["type"] == "drive"
        assert result["oauth2_client_id"] == "google-client-id"
        assert result["folder_id"] == "abc123"

    def test_calendar_microsoft(self):
        account = PersonalAccount(
            name="cal",
            type="calendar",
            provider="microsoft",
            display_name="Calendar",
            email="user@company.com",
            oauth2_tenant_id="tenant-xyz",
            options={"calendar_id": "primary"},
        )
        result = account_to_document_config(account, _mock_server_config())

        assert result["type"] == "calendar"
        assert result["oauth2_tenant_id"] == "tenant-xyz"
        assert result["calendar_id"] == "primary"

    def test_sharepoint_microsoft(self):
        account = PersonalAccount(
            name="sp",
            type="sharepoint",
            provider="microsoft",
            display_name="SharePoint",
            email="user@company.com",
            oauth2_tenant_id="tenant-xyz",
            options={"site_url": "https://contoso.sharepoint.com/sites/analytics"},
        )
        result = account_to_document_config(account, _mock_server_config())

        assert result["type"] == "sharepoint"
        assert result["site_url"] == "https://contoso.sharepoint.com/sites/analytics"
        assert result["oauth2_tenant_id"] == "tenant-xyz"

    def test_fallback_to_email_credentials(self):
        """When generalized OAuth creds are not set, falls back to email-specific."""
        cfg = _mock_server_config(
            google_oauth_client_id=None,
            google_oauth_client_secret=None,
            google_email_client_id="email-cid",
            google_email_client_secret="email-secret",
        )
        account = PersonalAccount(
            name="gmail",
            type="imap",
            provider="google",
            display_name="Gmail",
            email="user@gmail.com",
        )
        result = account_to_document_config(account, cfg)
        assert result["oauth2_client_id"] == "email-cid"
        assert result["oauth2_client_secret"] == "email-secret"

    def test_microsoft_tenant_fallback(self):
        """Uses server config tenant when account has no tenant_id."""
        account = PersonalAccount(
            name="outlook",
            type="imap",
            provider="microsoft",
            display_name="Outlook",
            email="user@company.com",
            oauth2_tenant_id=None,
        )
        cfg = _mock_server_config(microsoft_oauth_tenant_id="server-tenant")
        result = account_to_document_config(account, cfg)
        assert result["oauth2_tenant_id"] == "server-tenant"


# --- Validation ---


class TestValidation:
    def test_valid_imap_google(self):
        acct = PersonalAccount(
            name="my-gmail",
            type="imap",
            provider="google",
            display_name="Gmail",
            email="test@gmail.com",
        )
        validate_account(acct)  # Should not raise

    def test_missing_name(self):
        acct = PersonalAccount(
            name="",
            type="imap",
            provider="google",
            display_name="Gmail",
            email="test@gmail.com",
        )
        with pytest.raises(ValueError, match="name is required"):
            validate_account(acct)

    def test_invalid_name_chars(self):
        acct = PersonalAccount(
            name="my gmail!",
            type="imap",
            provider="google",
            display_name="Gmail",
            email="test@gmail.com",
        )
        with pytest.raises(ValueError, match="alphanumeric"):
            validate_account(acct)

    def test_invalid_type(self):
        acct = PersonalAccount(
            name="test",
            type="ftp",
            provider="google",
            display_name="Test",
            email="test@test.com",
        )
        with pytest.raises(ValueError, match="Invalid account type"):
            validate_account(acct)

    def test_invalid_provider(self):
        acct = PersonalAccount(
            name="test",
            type="imap",
            provider="yahoo",
            display_name="Test",
            email="test@test.com",
        )
        with pytest.raises(ValueError, match="Invalid provider"):
            validate_account(acct)

    def test_missing_display_name(self):
        acct = PersonalAccount(
            name="test",
            type="imap",
            provider="google",
            display_name="",
            email="test@test.com",
        )
        with pytest.raises(ValueError, match="Display name"):
            validate_account(acct)

    def test_missing_email(self):
        acct = PersonalAccount(
            name="test",
            type="imap",
            provider="google",
            display_name="Test",
            email="",
        )
        with pytest.raises(ValueError, match="Email"):
            validate_account(acct)

    def test_sharepoint_requires_microsoft(self):
        acct = PersonalAccount(
            name="test",
            type="sharepoint",
            provider="google",
            display_name="Test",
            email="test@test.com",
        )
        with pytest.raises(ValueError, match="microsoft"):
            validate_account(acct)

    def test_sharepoint_requires_site_url(self):
        acct = PersonalAccount(
            name="test",
            type="sharepoint",
            provider="microsoft",
            display_name="Test",
            email="test@test.com",
            options={},
        )
        with pytest.raises(ValueError, match="site_url"):
            validate_account(acct)

    def test_invalid_auth_type(self):
        acct = PersonalAccount(
            name="test",
            type="imap",
            provider="google",
            display_name="Test",
            email="test@test.com",
            auth_type="basic",
        )
        with pytest.raises(ValueError, match="Invalid auth_type"):
            validate_account(acct)


# --- Active/inactive filtering ---


class TestActiveFiltering:
    def test_filter_active_accounts(self, tmp_path):
        accounts = {
            "active-one": PersonalAccount(
                name="active-one",
                type="imap",
                provider="google",
                display_name="Active",
                email="a@test.com",
                active=True,
            ),
            "inactive-one": PersonalAccount(
                name="inactive-one",
                type="drive",
                provider="google",
                display_name="Inactive",
                email="b@test.com",
                active=False,
            ),
            "active-two": PersonalAccount(
                name="active-two",
                type="calendar",
                provider="microsoft",
                display_name="Active 2",
                email="c@test.com",
                active=True,
                oauth2_tenant_id="t1",
            ),
        }
        save_user_accounts("filteruser", accounts, tmp_path)
        loaded = load_user_accounts("filteruser", tmp_path)

        active = {n: a for n, a in loaded.items() if a.active}
        inactive = {n: a for n, a in loaded.items() if not a.active}

        assert len(active) == 2
        assert "active-one" in active
        assert "active-two" in active
        assert len(inactive) == 1
        assert "inactive-one" in inactive


# --- now_iso ---


class TestNowIso:
    def test_returns_iso_string(self):
        result = now_iso()
        assert "T" in result
        assert "+" in result or "Z" in result
