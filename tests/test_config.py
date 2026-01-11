"""Tests for configuration loading and model tiering."""

import os
import tempfile
from pathlib import Path

import pytest

from constat.core.config import (
    Config, LLMConfig, LLMTiersConfig, DatabaseConfig,
    DatabaseCredentials, UserConfig
)


class TestLLMConfig:
    """Tests for LLM configuration."""

    def test_default_model(self):
        """Test default model when no tiers configured."""
        config = LLMConfig(model="claude-sonnet-4-20250514")

        assert config.get_model("default") == "claude-sonnet-4-20250514"
        assert config.get_model("planning") == "claude-sonnet-4-20250514"
        assert config.get_model("codegen") == "claude-sonnet-4-20250514"
        assert config.get_model("simple") == "claude-sonnet-4-20250514"
        assert config.get_model("unknown") == "claude-sonnet-4-20250514"

    def test_tiered_models(self):
        """Test model selection with tiers configured."""
        tiers = LLMTiersConfig(
            planning="claude-opus-4-20250514",
            codegen="claude-sonnet-4-20250514",
            simple="claude-3-5-haiku-20241022",
        )
        config = LLMConfig(model="claude-sonnet-4-20250514", tiers=tiers)

        assert config.get_model("default") == "claude-sonnet-4-20250514"
        assert config.get_model("planning") == "claude-opus-4-20250514"
        assert config.get_model("codegen") == "claude-sonnet-4-20250514"
        assert config.get_model("simple") == "claude-3-5-haiku-20241022"

    def test_unknown_tier_returns_default(self):
        """Test that unknown tier falls back to default model."""
        tiers = LLMTiersConfig(
            planning="claude-opus-4-20250514",
            codegen="claude-sonnet-4-20250514",
            simple="claude-3-5-haiku-20241022",
        )
        config = LLMConfig(model="claude-sonnet-4-20250514", tiers=tiers)

        assert config.get_model("unknown_tier") == "claude-sonnet-4-20250514"
        assert config.get_model("invalid") == "claude-sonnet-4-20250514"


class TestLLMTiersConfig:
    """Tests for LLM tiers configuration."""

    def test_tier_defaults(self):
        """Test tier default values."""
        tiers = LLMTiersConfig()

        assert tiers.planning == "claude-sonnet-4-20250514"
        assert tiers.codegen == "claude-sonnet-4-20250514"
        assert tiers.simple == "claude-3-5-haiku-20241022"

    def test_custom_tier_values(self):
        """Test custom tier configuration."""
        tiers = LLMTiersConfig(
            planning="model-a",
            codegen="model-b",
            simple="model-c",
        )

        assert tiers.planning == "model-a"
        assert tiers.codegen == "model-b"
        assert tiers.simple == "model-c"


class TestDatabaseConfig:
    """Tests for database configuration."""

    def test_database_with_description(self):
        """Test database config with description."""
        db = DatabaseConfig(
            name="chinook",
            uri="sqlite:///./data/chinook.db",
            description="Digital music store selling tracks and albums online",
        )

        assert db.name == "chinook"
        assert db.description == "Digital music store selling tracks and albums online"

    def test_database_description_defaults_empty(self):
        """Test that description defaults to empty string."""
        db = DatabaseConfig(name="test", uri="sqlite:///test.db")

        assert db.description == ""


class TestConfigFromYaml:
    """Tests for loading config from YAML files."""

    def test_load_config_with_tiers(self):
        """Test loading config with model tiers from YAML."""
        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key
  tiers:
    planning: claude-opus-4-20250514
    codegen: claude-sonnet-4-20250514
    simple: claude-3-5-haiku-20241022

databases:
  - name: test_db
    uri: sqlite:///test.db
    description: Test database
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = Config.from_yaml(f.name)

                assert config.llm.tiers is not None
                assert config.llm.get_model("planning") == "claude-opus-4-20250514"
                assert config.llm.get_model("codegen") == "claude-sonnet-4-20250514"
                assert config.llm.get_model("simple") == "claude-3-5-haiku-20241022"
            finally:
                os.unlink(f.name)

    def test_load_config_without_tiers(self):
        """Test loading config without model tiers defaults correctly."""
        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key

databases:
  - name: test_db
    uri: sqlite:///test.db
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = Config.from_yaml(f.name)

                assert config.llm.tiers is None
                # All tiers should return default model when tiers not configured
                assert config.llm.get_model("planning") == "claude-sonnet-4-20250514"
                assert config.llm.get_model("codegen") == "claude-sonnet-4-20250514"
                assert config.llm.get_model("simple") == "claude-sonnet-4-20250514"
            finally:
                os.unlink(f.name)

    def test_load_config_with_database_descriptions(self):
        """Test loading config with database descriptions."""
        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key

databases_description: |
  Each database represents a different company.

databases:
  - name: company_a
    uri: sqlite:///a.db
    description: Company A's sales data
  - name: company_b
    uri: sqlite:///b.db
    description: Company B's inventory data
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = Config.from_yaml(f.name)

                assert "Each database represents a different company" in config.databases_description
                assert len(config.databases) == 2
                assert config.databases[0].description == "Company A's sales data"
                assert config.databases[1].description == "Company B's inventory data"
            finally:
                os.unlink(f.name)

    def test_load_config_env_var_substitution(self):
        """Test environment variable substitution in config."""
        os.environ["TEST_API_KEY"] = "my-secret-key"

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${TEST_API_KEY}

databases: []
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = Config.from_yaml(f.name)

                assert config.llm.api_key == "my-secret-key"
            finally:
                os.unlink(f.name)
                del os.environ["TEST_API_KEY"]

    def test_load_config_missing_env_var_raises(self):
        """Test that missing environment variable raises error."""
        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${NONEXISTENT_VAR_FOR_TEST}

databases: []
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                with pytest.raises(ValueError, match="Environment variable not set"):
                    Config.from_yaml(f.name)
            finally:
                os.unlink(f.name)


class TestDatabaseCredentials:
    """Tests for database credential handling."""

    def test_credentials_is_complete(self):
        """Test credential completeness check."""
        incomplete = DatabaseCredentials(username="user")
        assert not incomplete.is_complete()

        incomplete2 = DatabaseCredentials(password="pass")
        assert not incomplete2.is_complete()

        complete = DatabaseCredentials(username="user", password="pass")
        assert complete.is_complete()

    def test_database_config_with_credentials(self):
        """Test database config with inline credentials."""
        db = DatabaseConfig(
            name="test_db",
            uri="postgresql://localhost:5432/test",
            username="admin",
            password="secret123"
        )

        uri = db.get_connection_uri()
        assert "admin" in uri
        assert "secret123" in uri
        assert "localhost:5432" in uri

    def test_database_config_uri_passthrough(self):
        """Test that URI is returned as-is when no credentials."""
        db = DatabaseConfig(
            name="test_db",
            uri="sqlite:///./data/test.db"
        )

        assert db.get_connection_uri() == "sqlite:///./data/test.db"

    def test_database_config_user_credentials_override(self):
        """Test that user credentials override config credentials."""
        db = DatabaseConfig(
            name="test_db",
            uri="postgresql://localhost:5432/test",
            username="config_user",
            password="config_pass",
            requires_user_credentials=True
        )

        user_creds = DatabaseCredentials(username="session_user", password="session_pass")
        uri = db.get_connection_uri(user_credentials=user_creds)

        assert "session_user" in uri
        assert "session_pass" in uri
        assert "config_user" not in uri

    def test_database_config_requires_user_credentials(self):
        """Test that missing user credentials raises error when required."""
        db = DatabaseConfig(
            name="secure_db",
            uri="postgresql://localhost:5432/secure",
            requires_user_credentials=True
        )

        with pytest.raises(ValueError, match="requires user credentials"):
            db.get_connection_uri()

    def test_credential_injection_with_special_chars(self):
        """Test that special characters in credentials are properly escaped."""
        db = DatabaseConfig(
            name="test_db",
            uri="postgresql://localhost:5432/test",
            username="user@domain",
            password="p@ss:word/123"
        )

        uri = db.get_connection_uri()
        # Special characters should be URL-encoded
        assert "user%40domain" in uri
        assert "p%40ss%3Aword%2F123" in uri


class TestUserConfig:
    """Tests for user config and merging."""

    def test_user_config_creation(self):
        """Test creating user config with credentials."""
        user_config = UserConfig(
            database_credentials={
                "db1": DatabaseCredentials(username="user1", password="pass1"),
                "db2": DatabaseCredentials(username="user2", password="pass2"),
            }
        )

        assert len(user_config.database_credentials) == 2
        assert user_config.database_credentials["db1"].username == "user1"

    def test_config_merge_user_credentials(self):
        """Test merging user credentials into config."""
        base_config = Config(
            databases=[
                DatabaseConfig(name="db1", uri="postgresql://localhost/db1"),
                DatabaseConfig(name="db2", uri="postgresql://localhost/db2"),
            ]
        )

        user_config = UserConfig(
            database_credentials={
                "db1": DatabaseCredentials(username="alice", password="secret"),
            }
        )

        merged = base_config.merge_user_config(user_config)

        # User credentials should be accessible
        creds = merged.get_database_credentials("db1")
        assert creds is not None
        assert creds.username == "alice"
        assert creds.password == "secret"

        # Non-provided credentials should be None
        assert merged.get_database_credentials("db2") is None

    def test_config_merge_database_overrides(self):
        """Test merging database config overrides."""
        base_config = Config(
            databases=[
                DatabaseConfig(
                    name="db1",
                    uri="postgresql://localhost/db1",
                    description="Base description"
                ),
            ]
        )

        user_config = UserConfig(
            databases=[
                DatabaseConfig(
                    name="db1",
                    uri="postgresql://localhost/db1",
                    username="custom_user",
                    password="custom_pass"
                ),
            ]
        )

        merged = base_config.merge_user_config(user_config)

        db = merged.get_database_config("db1")
        assert db is not None
        assert db.username == "custom_user"
        assert db.password == "custom_pass"
        # Base description should be preserved
        assert db.description == "Base description"

    def test_config_merge_add_new_database(self):
        """Test that user config can add new databases."""
        base_config = Config(
            databases=[
                DatabaseConfig(name="db1", uri="sqlite:///db1.db"),
            ]
        )

        user_config = UserConfig(
            databases=[
                DatabaseConfig(name="db2", uri="sqlite:///db2.db", description="User DB"),
            ]
        )

        merged = base_config.merge_user_config(user_config)

        assert len(merged.databases) == 2
        db2 = merged.get_database_config("db2")
        assert db2 is not None
        assert db2.description == "User DB"

    def test_get_database_credentials_priority(self):
        """Test credential lookup priority: user > config."""
        config = Config(
            databases=[
                DatabaseConfig(
                    name="db1",
                    uri="postgresql://localhost/db1",
                    username="config_user",
                    password="config_pass"
                ),
            ]
        )

        # Without user credentials, should use config
        creds = config.get_database_credentials("db1")
        assert creds.username == "config_user"

        # With user credentials, should use those instead
        user_config = UserConfig(
            database_credentials={
                "db1": DatabaseCredentials(username="user_user", password="user_pass"),
            }
        )
        merged = config.merge_user_config(user_config)
        creds = merged.get_database_credentials("db1")
        assert creds.username == "user_user"
