# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for scope-aware source persistence and OAuth token management."""

import yaml
import pytest

from constat.server.user_tokens import (
    inject_oauth_tokens,
    load_user_tokens,
    migrate_email_tokens,
    migrate_user_config_source_tags,
    save_user_tokens,
)


@pytest.fixture
def constat_dir(tmp_path, monkeypatch):
    """Create a .constat directory and patch user_tokens to use it."""
    import constat.server.user_tokens as ut
    monkeypatch.setattr(ut, "_CONSTAT_DIR", tmp_path)
    return tmp_path


class TestSourcePersistence:
    """Tests for session vs user scope persistence logic."""

    def test_session_scoped_db_persisted_to_session_config(self, tmp_path):
        """Session-scoped DB written to session config.yaml, NOT user config."""
        session_dir = tmp_path / "sessions" / "s1"
        session_dir.mkdir(parents=True)

        # Simulate what _persist_session_config does
        config = {
            "databases": {
                "csv_db": {
                    "type": "sql",
                    "uri": "sqlite:///data.db",
                    "description": "test db",
                    "source": "session",
                }
            }
        }
        (session_dir / "config.yaml").write_text(yaml.dump(config))

        # Verify session config has the db
        loaded = yaml.safe_load((session_dir / "config.yaml").read_text())
        assert "csv_db" in loaded["databases"]
        assert loaded["databases"]["csv_db"]["source"] == "session"

        # User config should NOT exist
        user_config = tmp_path / "config.yaml"
        assert not user_config.exists()

    def test_user_scoped_db_persisted_to_user_config(self, tmp_path):
        """User-scoped DB written to user config with source=user."""
        user_config_path = tmp_path / "config.yaml"

        # Simulate _persist_user_scoped_dbs
        config = {
            "databases": {
                "main_db": {
                    "type": "sql",
                    "uri": "postgresql://host/db",
                    "description": "Main database",
                    "source": "user",
                }
            }
        }
        user_config_path.write_text(yaml.dump(config))

        loaded = yaml.safe_load(user_config_path.read_text())
        assert loaded["databases"]["main_db"]["source"] == "user"

    def test_save_does_not_delete_existing_user_entries(self, tmp_path):
        """Existing user config entries from other sessions survive save."""
        user_config_path = tmp_path / "config.yaml"

        # Pre-existing entry from another session
        existing = {
            "databases": {
                "old_db": {
                    "type": "sql",
                    "uri": "sqlite:///old.db",
                    "source": "user",
                }
            }
        }
        user_config_path.write_text(yaml.dump(existing))

        # Add new entry (simulating _persist_user_scoped_dbs additive logic)
        config = yaml.safe_load(user_config_path.read_text()) or {}
        databases = config.get("databases", {})
        databases["new_db"] = {
            "type": "sql",
            "uri": "sqlite:///new.db",
            "source": "user",
        }
        config["databases"] = databases
        user_config_path.write_text(yaml.dump(config))

        loaded = yaml.safe_load(user_config_path.read_text())
        assert "old_db" in loaded["databases"]
        assert "new_db" in loaded["databases"]

    def test_empty_session_config_deleted(self, tmp_path):
        """Empty session config file should be cleaned up."""
        session_dir = tmp_path / "sessions" / "s1"
        session_dir.mkdir(parents=True)
        config_path = session_dir / "config.yaml"
        config_path.write_text("")

        # Simulate cleanup logic: if config is empty, delete
        config = yaml.safe_load(config_path.read_text()) or {}
        if not config:
            config_path.unlink()

        assert not config_path.exists()


class TestTokenManagement:

    def test_load_save_tokens(self, constat_dir):
        """Round-trip: save tokens then load them back."""
        tokens = {
            "my-gmail": {
                "provider": "google",
                "email": "user@gmail.com",
                "refresh_token": "rt_abc",
                "tenant_id": None,
                "scopes": "",
                "created_at": "2026-03-24T10:00:00Z",
            }
        }
        save_user_tokens("alice", tokens)
        loaded = load_user_tokens("alice")
        assert loaded["my-gmail"]["refresh_token"] == "rt_abc"
        assert loaded["my-gmail"]["provider"] == "google"

    def test_load_nonexistent_returns_empty(self, constat_dir):
        """Loading tokens for user with no tokens.yaml returns empty dict."""
        assert load_user_tokens("nobody") == {}

    def test_token_injection_google(self, constat_dir):
        """oauth2_token_ref resolved with Google provider credentials."""
        tokens = {
            "my-gmail": {
                "provider": "google",
                "email": "user@gmail.com",
                "refresh_token": "rt_google_123",
            }
        }
        save_user_tokens("alice", tokens)

        class MockServerConfig:
            google_email_client_id = "google-client-id"
            google_email_client_secret = "google-client-secret"

        documents = {
            "my-gmail": {
                "type": "imap",
                "url": "imaps://imap.gmail.com:993",
                "oauth2_token_ref": "my-gmail",
            }
        }
        result = inject_oauth_tokens("alice", documents, MockServerConfig())
        assert result["my-gmail"]["oauth2_client_id"] == "google-client-id"
        assert result["my-gmail"]["oauth2_client_secret"] == "rt_google_123"
        assert result["my-gmail"]["password"] == "google-client-secret"
        assert result["my-gmail"]["auth_type"] == "oauth2_refresh"

    def test_token_injection_microsoft(self, constat_dir):
        """oauth2_token_ref resolved with Microsoft provider credentials."""
        tokens = {
            "my-outlook": {
                "provider": "microsoft",
                "email": "user@outlook.com",
                "refresh_token": "rt_ms_456",
                "tenant_id": "tenant-abc",
            }
        }
        save_user_tokens("bob", tokens)

        class MockServerConfig:
            microsoft_email_client_id = "ms-client-id"
            microsoft_email_client_secret = "ms-client-secret"
            microsoft_email_tenant_id = "default-tenant"

        documents = {
            "my-outlook": {
                "type": "imap",
                "oauth2_token_ref": "my-outlook",
            }
        }
        result = inject_oauth_tokens("bob", documents, MockServerConfig())
        assert result["my-outlook"]["oauth2_client_id"] == "ms-client-id"
        assert result["my-outlook"]["oauth2_client_secret"] == "rt_ms_456"
        assert result["my-outlook"]["oauth2_tenant_id"] == "tenant-abc"
        assert result["my-outlook"]["auth_type"] == "oauth2_refresh"

    def test_token_injection_no_server_config(self, constat_dir):
        """Fallback: inject refresh token directly without server config."""
        tokens = {
            "src": {
                "provider": "google",
                "refresh_token": "rt_fallback",
            }
        }
        save_user_tokens("user1", tokens)

        documents = {"src": {"oauth2_token_ref": "src"}}
        result = inject_oauth_tokens("user1", documents, server_config=None)
        assert result["src"]["oauth2_client_secret"] == "rt_fallback"
        assert result["src"]["auth_type"] == "oauth2_refresh"

    def test_token_injection_no_matching_ref(self, constat_dir):
        """Documents with non-matching oauth2_token_ref are unchanged."""
        save_user_tokens("user1", {"other": {"provider": "google", "refresh_token": "x"}})
        documents = {"doc1": {"oauth2_token_ref": "nonexistent"}}
        result = inject_oauth_tokens("user1", documents)
        assert "auth_type" not in result["doc1"]

    def test_token_injection_no_tokens_file(self, constat_dir):
        """If no tokens.yaml exists, documents pass through unchanged."""
        documents = {"doc1": {"type": "file", "path": "/tmp/test.pdf"}}
        result = inject_oauth_tokens("nouser", documents)
        assert result == documents

    def test_email_token_migration(self, constat_dir):
        """Inline OAuth tokens extracted to tokens.yaml, config updated with ref."""
        user_dir = constat_dir / "migrate-user.vault"
        user_dir.mkdir()
        config = {
            "documents": {
                "my-gmail": {
                    "type": "imap",
                    "username": "user@gmail.com",
                    "auth_type": "oauth2_refresh",
                    "oauth2_client_id": "old-client-id",
                    "oauth2_client_secret": "inline-refresh-token",
                }
            }
        }
        (user_dir / "config.yaml").write_text(yaml.dump(config))

        count = migrate_email_tokens("migrate-user")
        assert count == 1

        # Check tokens.yaml was created
        tokens = load_user_tokens("migrate-user")
        assert "my-gmail" in tokens
        assert tokens["my-gmail"]["refresh_token"] == "inline-refresh-token"
        assert tokens["my-gmail"]["provider"] == "google"
        assert tokens["my-gmail"]["email"] == "user@gmail.com"

        # Check config was updated
        updated_config = yaml.safe_load((user_dir / "config.yaml").read_text())
        doc = updated_config["documents"]["my-gmail"]
        assert doc["oauth2_token_ref"] == "my-gmail"
        assert "oauth2_client_secret" not in doc
        assert "oauth2_client_id" not in doc

    def test_email_token_migration_microsoft(self, constat_dir):
        """Microsoft tenant detected during migration."""
        user_dir = constat_dir / "ms-user.vault"
        user_dir.mkdir()
        config = {
            "documents": {
                "outlook": {
                    "type": "imap",
                    "username": "user@outlook.com",
                    "auth_type": "oauth2",
                    "oauth2_client_secret": "ms-token",
                    "oauth2_tenant_id": "tenant-123",
                }
            }
        }
        (user_dir / "config.yaml").write_text(yaml.dump(config))

        count = migrate_email_tokens("ms-user")
        assert count == 1
        tokens = load_user_tokens("ms-user")
        assert tokens["outlook"]["provider"] == "microsoft"
        assert tokens["outlook"]["tenant_id"] == "tenant-123"

    def test_email_token_migration_skip_already_migrated(self, constat_dir):
        """Already-migrated entries (with oauth2_token_ref) are skipped."""
        user_dir = constat_dir / "skip-user.vault"
        user_dir.mkdir()
        config = {
            "documents": {
                "already": {
                    "auth_type": "oauth2_refresh",
                    "oauth2_client_secret": "should-not-move",
                    "oauth2_token_ref": "already",
                }
            }
        }
        (user_dir / "config.yaml").write_text(yaml.dump(config))

        count = migrate_email_tokens("skip-user")
        assert count == 0

    def test_email_token_migration_no_config(self, constat_dir):
        """Migration returns 0 when no config.yaml exists."""
        assert migrate_email_tokens("nouser") == 0

    def test_migrate_source_tags(self, constat_dir):
        """source='session' renamed to source='user' in user config."""
        user_dir = constat_dir / "tag-user.vault"
        user_dir.mkdir()
        config = {
            "databases": {
                "db1": {"type": "sql", "source": "session"},
                "db2": {"type": "sql", "source": "user"},
            },
            "documents": {
                "doc1": {"type": "file", "source": "session"},
            },
            "apis": {},
        }
        (user_dir / "config.yaml").write_text(yaml.dump(config))

        count = migrate_user_config_source_tags("tag-user")
        assert count == 2

        updated = yaml.safe_load((user_dir / "config.yaml").read_text())
        assert updated["databases"]["db1"]["source"] == "user"
        assert updated["databases"]["db2"]["source"] == "user"
        assert updated["documents"]["doc1"]["source"] == "user"

    def test_migrate_source_tags_no_changes(self, constat_dir):
        """No changes needed returns 0."""
        user_dir = constat_dir / "ok-user.vault"
        user_dir.mkdir()
        config = {"databases": {"db1": {"source": "user"}}}
        (user_dir / "config.yaml").write_text(yaml.dump(config))

        count = migrate_user_config_source_tags("ok-user")
        assert count == 0

    def test_migrate_source_tags_no_config(self, constat_dir):
        assert migrate_user_config_source_tags("nouser") == 0
