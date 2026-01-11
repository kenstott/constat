"""Tests for configuration loading and model tiering."""

import os
import tempfile
from pathlib import Path

import pytest

from constat.core.config import (
    Config, LLMConfig, LLMTiersConfig, TierConfig, DatabaseConfig,
    DatabaseCredentials
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

    def test_tier_with_provider_override(self):
        """Test tier with provider override."""
        tiers = LLMTiersConfig(
            planning="claude-opus-4-20250514",
            codegen="claude-sonnet-4-20250514",
            simple=TierConfig(provider="ollama", model="llama3.2:3b"),
        )

        # String tiers should be normalized to TierConfig
        planning_config = tiers.get_tier_config("planning")
        assert planning_config.provider is None  # Uses default
        assert planning_config.model == "claude-opus-4-20250514"

        # TierConfig tier should have provider override
        simple_config = tiers.get_tier_config("simple")
        assert simple_config.provider == "ollama"
        assert simple_config.model == "llama3.2:3b"

    def test_get_tier_config_normalizes_strings(self):
        """Test that get_tier_config normalizes string to TierConfig."""
        tiers = LLMTiersConfig(
            planning="my-model",
        )

        tier_config = tiers.get_tier_config("planning")
        assert isinstance(tier_config, TierConfig)
        assert tier_config.model == "my-model"
        assert tier_config.provider is None


class TestTierConfig:
    """Tests for TierConfig model."""

    def test_tier_config_model_only(self):
        """TierConfig with just model."""
        config = TierConfig(model="gpt-4")
        assert config.model == "gpt-4"
        assert config.provider is None
        assert config.base_url is None

    def test_tier_config_with_provider(self):
        """TierConfig with provider override."""
        config = TierConfig(provider="ollama", model="llama3.2:3b")
        assert config.provider == "ollama"
        assert config.model == "llama3.2:3b"

    def test_tier_config_with_base_url(self):
        """TierConfig with custom base URL."""
        config = TierConfig(
            provider="ollama",
            model="llama3.2:3b",
            base_url="http://192.168.1.100:11434/v1",
        )
        assert config.base_url == "http://192.168.1.100:11434/v1"


class TestLLMConfigTierConfig:
    """Tests for LLMConfig.get_tier_config method."""

    def test_get_tier_config_default(self):
        """get_tier_config returns default when no tiers configured."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
        )

        tier_config = config.get_tier_config("planning")
        assert tier_config.provider is None
        assert tier_config.model == "claude-sonnet-4-20250514"

    def test_get_tier_config_with_tiers(self):
        """get_tier_config returns correct config for each tier."""
        tiers = LLMTiersConfig(
            planning="claude-opus-4-20250514",
            codegen="claude-sonnet-4-20250514",
            simple=TierConfig(provider="ollama", model="llama3.2:3b"),
        )
        config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            tiers=tiers,
        )

        planning = config.get_tier_config("planning")
        assert planning.provider is None
        assert planning.model == "claude-opus-4-20250514"

        simple = config.get_tier_config("simple")
        assert simple.provider == "ollama"
        assert simple.model == "llama3.2:3b"

    def test_get_tier_config_unknown_tier_returns_default(self):
        """Unknown tier returns default config."""
        tiers = LLMTiersConfig()
        config = LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            tiers=tiers,
        )

        tier_config = config.get_tier_config("unknown")
        assert tier_config.provider is None
        assert tier_config.model == "claude-sonnet-4-20250514"


class TestDatabaseConfig:
    """Tests for database configuration."""

    def test_database_with_description(self):
        """Test database config with description."""
        db = DatabaseConfig(
            uri="sqlite:///./data/chinook.db",
            description="Digital music store selling tracks and albums online",
        )

        assert db.description == "Digital music store selling tracks and albums online"

    def test_database_description_defaults_empty(self):
        """Test that description defaults to empty string."""
        db = DatabaseConfig(uri="sqlite:///test.db")

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
  test_db:
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

    def test_load_config_with_provider_override_tiers(self):
        """Test loading config with provider override in tiers from YAML."""
        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key
  tiers:
    planning: claude-opus-4-20250514
    codegen: claude-sonnet-4-20250514
    simple:
      provider: ollama
      model: llama3.2:3b

databases:
  test_db:
    uri: sqlite:///test.db
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = Config.from_yaml(f.name)

                assert config.llm.tiers is not None

                # Model tiers (string format)
                assert config.llm.get_model("planning") == "claude-opus-4-20250514"
                assert config.llm.get_model("codegen") == "claude-sonnet-4-20250514"

                # Provider override tier
                simple_config = config.llm.get_tier_config("simple")
                assert simple_config.provider == "ollama"
                assert simple_config.model == "llama3.2:3b"
            finally:
                os.unlink(f.name)

    def test_load_config_with_base_url_in_tier(self):
        """Test loading config with base_url in tier config from YAML."""
        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key
  tiers:
    planning: claude-opus-4-20250514
    simple:
      provider: ollama
      model: llama3.2:3b
      base_url: http://192.168.1.100:11434/v1

databases: {}
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()

            try:
                config = Config.from_yaml(f.name)

                simple_config = config.llm.get_tier_config("simple")
                assert simple_config.provider == "ollama"
                assert simple_config.model == "llama3.2:3b"
                assert simple_config.base_url == "http://192.168.1.100:11434/v1"
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
  test_db:
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
  company_a:
    uri: sqlite:///a.db
    description: Company A's sales data
  company_b:
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
                assert config.databases["company_a"].description == "Company A's sales data"
                assert config.databases["company_b"].description == "Company B's inventory data"
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

databases: {}
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

databases: {}
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
        db = DatabaseConfig(uri="sqlite:///./data/test.db")

        assert db.get_connection_uri() == "sqlite:///./data/test.db"

    def test_credential_injection_with_special_chars(self):
        """Test that special characters in credentials are properly escaped."""
        db = DatabaseConfig(
            uri="postgresql://localhost:5432/test",
            username="user@domain",
            password="p@ss:word/123"
        )

        uri = db.get_connection_uri()
        # Special characters should be URL-encoded
        assert "user%40domain" in uri
        assert "p%40ss%3Aword%2F123" in uri


class TestUserConfigMerging:
    """Tests for user config merging."""

    def test_merge_user_credentials(self):
        """Test merging user credentials into config."""
        engine_yaml = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key

databases:
  db1:
    uri: postgresql://localhost/db1
  db2:
    uri: postgresql://localhost/db2
"""
        user_yaml = """
databases:
  db1:
    username: alice
    password: secret
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as engine_f:
            engine_f.write(engine_yaml)
            engine_f.flush()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as user_f:
                user_f.write(user_yaml)
                user_f.flush()

                try:
                    config = Config.from_yaml(engine_f.name, user_config_path=user_f.name)

                    # User credentials should be merged
                    assert config.databases["db1"].username == "alice"
                    assert config.databases["db1"].password == "secret"

                    # db2 should not have credentials
                    assert config.databases["db2"].username is None
                finally:
                    os.unlink(engine_f.name)
                    os.unlink(user_f.name)

    def test_merge_preserves_engine_values(self):
        """Test that engine values are preserved when user config doesn't override."""
        engine_yaml = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key

databases:
  db1:
    uri: postgresql://localhost/db1
    description: Engine description
"""
        user_yaml = """
databases:
  db1:
    username: alice
    password: secret
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as engine_f:
            engine_f.write(engine_yaml)
            engine_f.flush()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as user_f:
                user_f.write(user_yaml)
                user_f.flush()

                try:
                    config = Config.from_yaml(engine_f.name, user_config_path=user_f.name)

                    # Engine description should be preserved
                    assert config.databases["db1"].description == "Engine description"
                    # User credentials should be merged
                    assert config.databases["db1"].username == "alice"
                finally:
                    os.unlink(engine_f.name)
                    os.unlink(user_f.name)

    def test_user_can_add_new_databases(self):
        """Test that user config can add new databases."""
        engine_yaml = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key

databases:
  db1:
    uri: sqlite:///db1.db
"""
        user_yaml = """
databases:
  db2:
    uri: sqlite:///db2.db
    description: User's personal database
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as engine_f:
            engine_f.write(engine_yaml)
            engine_f.flush()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as user_f:
                user_f.write(user_yaml)
                user_f.flush()

                try:
                    config = Config.from_yaml(engine_f.name, user_config_path=user_f.name)

                    assert len(config.databases) == 2
                    assert "db1" in config.databases
                    assert "db2" in config.databases
                    assert config.databases["db2"].description == "User's personal database"
                finally:
                    os.unlink(engine_f.name)
                    os.unlink(user_f.name)

    def test_merge_with_dict(self):
        """Test merging with dict instead of file."""
        engine_yaml = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key

databases:
  db1:
    uri: postgresql://localhost/db1
"""
        user_config = {
            "databases": {
                "db1": {"username": "bob", "password": "pass123"}
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(engine_yaml)
            f.flush()

            try:
                config = Config.from_yaml(f.name, user_config=user_config)

                assert config.databases["db1"].username == "bob"
                assert config.databases["db1"].password == "pass123"
            finally:
                os.unlink(f.name)


class TestConfigHelpers:
    """Tests for config helper methods."""

    def test_get_database(self):
        """Test get_database helper."""
        config = Config(
            databases={
                "main": DatabaseConfig(uri="sqlite:///main.db"),
                "analytics": DatabaseConfig(uri="sqlite:///analytics.db"),
            }
        )

        assert config.get_database("main") is not None
        assert config.get_database("main").uri == "sqlite:///main.db"
        assert config.get_database("nonexistent") is None
