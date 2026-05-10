# Copyright (c) 2025 Kenneth Stott
# Canary: 1dcb97c6-8791-4f5a-a805-4b8312a18447
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for tiered config resolution and tombstone behavior."""

from __future__ import annotations

from constat.core.config import Config, DomainConfig


class TestTieredConfigNerStopList:
    """Tests for ner_stop_list in tiered config resolution."""

    def _make_config(self, ner_stop_list=None, system_prompt="", domains=None):
        from unittest.mock import Mock
        config = Mock(spec=Config)
        config.databases = {}
        config.apis = {}
        config.documents = {}
        config.facts = {}
        config.relationships = {}
        config.rights = {}
        config.skills = None
        config.system_prompt = system_prompt
        config.databases_description = ""
        config.ner_stop_list = ner_stop_list or []
        config.llm = None
        config.domains = domains or {}

        def _load_domain(name):
            return config.domains.get(name)
        config.load_domain = _load_domain
        return config

    def _make_domain(self, ner_stop_list=None, system_prompt="", databases=None, apis=None, documents=None):
        from unittest.mock import Mock
        domain = Mock(spec=DomainConfig)
        domain.databases = databases or {}
        domain.apis = apis or {}
        domain.documents = documents or {}
        domain.relationships = {}
        domain.rights = {}
        domain.facts = {}
        domain.learnings = {}
        domain.system_prompt = system_prompt
        domain.databases_description = ""
        domain.ner_stop_list = ner_stop_list or []
        domain.task_routing = None
        domain.path = None
        domain.aliases = {}
        return domain

    def test_ner_stop_list_from_system_config(self):
        """System-level ner_stop_list appears in resolved config."""
        from constat.core.tiered_config import TieredConfigLoader
        config = self._make_config(ner_stop_list=["the", "a", "an"])
        loader = TieredConfigLoader(config=config)
        resolved = loader.resolve()
        assert resolved.ner_stop_list == ["the", "a", "an"]

    def test_ner_stop_list_from_domain_overrides_system(self):
        """Domain ner_stop_list replaces system-level (last tier wins for lists)."""
        from constat.core.tiered_config import TieredConfigLoader
        domain = self._make_domain(ner_stop_list=["stop1", "stop2"])
        config = self._make_config(
            ner_stop_list=["the", "a"],
            domains={"sales": domain},
        )
        loader = TieredConfigLoader(config=config, domain_names=["sales"])
        resolved = loader.resolve()
        assert resolved.ner_stop_list == ["stop1", "stop2"]

    def test_ner_stop_list_empty_when_not_configured(self):
        """ner_stop_list defaults to empty when not set anywhere."""
        from constat.core.tiered_config import TieredConfigLoader
        config = self._make_config()
        loader = TieredConfigLoader(config=config)
        resolved = loader.resolve()
        assert resolved.ner_stop_list == []

    def test_system_prompt_from_multi_domain_resolution(self):
        """system_prompt from last domain wins (sorted order)."""
        from constat.core.tiered_config import TieredConfigLoader
        domain_a = self._make_domain(system_prompt="Prompt A")
        domain_b = self._make_domain(system_prompt="Prompt B")
        config = self._make_config(
            system_prompt="System prompt",
            domains={"a-domain": domain_a, "b-domain": domain_b},
        )
        loader = TieredConfigLoader(config=config, domain_names=["a-domain", "b-domain"])
        resolved = loader.resolve()
        # b-domain sorts after a-domain, so its prompt wins
        assert resolved.system_prompt == "Prompt B"

    def test_facts_round_trip(self):
        """Domain facts appear in resolved config."""
        from constat.core.tiered_config import TieredConfigLoader
        domain = self._make_domain()
        domain.facts = {"fiscal_year_start": "April 1"}
        config = self._make_config(domains={"sales": domain})
        loader = TieredConfigLoader(config=config, domain_names=["sales"])
        resolved = loader.resolve()
        assert resolved.facts == {"fiscal_year_start": "April 1"}


class TestConfigTombstones:
    """Tests for null-tombstone soft-delete of config-seeded elements."""

    def _make_config(self, facts=None, learnings=None, domains=None):
        from unittest.mock import Mock
        config = Mock(spec=Config)
        config.databases = {}
        config.apis = {}
        config.documents = {}
        config.facts = facts or {}
        config.relationships = {}
        config.rights = {}
        config.skills = None
        config.system_prompt = ""
        config.databases_description = ""
        config.ner_stop_list = []
        config.llm = None
        config.domains = domains or {}

        def _load_domain(name):
            return config.domains.get(name)
        config.load_domain = _load_domain
        return config

    def _make_domain(self, facts=None, learnings=None):
        from unittest.mock import Mock
        domain = Mock(spec=DomainConfig)
        domain.databases = {}
        domain.apis = {}
        domain.documents = {}
        domain.relationships = {}
        domain.rights = {}
        domain.facts = facts or {}
        domain.learnings = learnings or {}
        domain.system_prompt = ""
        domain.databases_description = ""
        domain.ner_stop_list = []
        domain.task_routing = None
        domain.path = None
        domain.aliases = {}
        return domain

    def _make_session_manager(self, config, tmp_path):
        """Create a minimal SessionManager with a mock ServerConfig."""
        from unittest.mock import Mock
        from constat.server.config import ServerConfig
        from constat.server.session_manager import SessionManager

        server_config = Mock(spec=ServerConfig)
        server_config.data_dir = tmp_path
        server_config.max_concurrent_sessions = 10
        return SessionManager(config=config, server_config=server_config)

    def _make_managed_session(self, session_id, user_id, resolved_config):
        """Create a minimal ManagedSession with resolved config."""
        from unittest.mock import Mock
        from constat.server.session_manager import ManagedSession

        managed = Mock(spec=ManagedSession)
        managed.session_id = session_id
        managed.user_id = user_id
        managed.resolved_config = resolved_config
        return managed

    def test_tombstone_writes_null_to_user_yaml(self, tmp_path):
        """Tombstone for SYSTEM_DOMAIN item writes null to user config."""
        import yaml
        from constat.core.tiered_config import TieredConfigLoader, ConfigSource

        domain = self._make_domain(facts={"revenue": "Total income"})
        config = self._make_config(domains={"sales": domain})

        loader = TieredConfigLoader(
            config=config, user_id="alice", base_dir=tmp_path,
            domain_names=["sales"],
        )
        resolved = loader.resolve()
        assert resolved.facts["revenue"] == "Total income"
        # Attribution may be at "facts" level (whole section from domain)
        assert resolved._attribution.get("facts") == ConfigSource.SYSTEM_DOMAIN

        sm = self._make_session_manager(config, tmp_path)
        managed = self._make_managed_session("s1", "alice", resolved)
        sm._sessions["s1"] = managed

        assert sm.write_config_tombstone("s1", "facts", "revenue") is True

        user_yaml = yaml.safe_load((tmp_path / "alice.vault" / "config.yaml").read_text())
        assert user_yaml["facts"]["revenue"] is None

    def test_tombstone_skips_user_created_items(self, tmp_path):
        """Tombstone is not written for USER-tier items."""
        import yaml
        from constat.core.tiered_config import TieredConfigLoader, ConfigSource

        # System-level fact so attribution exists at leaf, then user overrides
        config = self._make_config(facts={"my_fact": "system_val"})
        # Write a user-tier override
        user_dir = tmp_path / "alice.vault"
        user_dir.mkdir(parents=True)
        (user_dir / "config.yaml").write_text(yaml.dump({"facts": {"my_fact": "hello"}}))

        loader = TieredConfigLoader(
            config=config, user_id="alice", base_dir=tmp_path,
        )
        resolved = loader.resolve()
        assert resolved._attribution["facts.my_fact"] == ConfigSource.USER

        sm = self._make_session_manager(config, tmp_path)
        managed = self._make_managed_session("s1", "alice", resolved)
        sm._sessions["s1"] = managed

        assert sm.write_config_tombstone("s1", "facts", "my_fact") is False

        # User config should be unchanged
        user_yaml = yaml.safe_load((user_dir / "config.yaml").read_text())
        assert user_yaml["facts"]["my_fact"] == "hello"

    def test_tombstone_nested_key(self, tmp_path):
        """Dotted key 'rules.my_rule' produces nested null in learnings."""
        import yaml
        from constat.core.tiered_config import TieredConfigLoader, ConfigSource

        domain = self._make_domain(learnings={"rules": {"my_rule": "Always use fiscal year"}})
        config = self._make_config(domains={"sales": domain})

        loader = TieredConfigLoader(
            config=config, user_id="alice", base_dir=tmp_path,
            domain_names=["sales"],
        )
        resolved = loader.resolve()
        # Attribution may be at parent level since whole learnings came from domain
        source = resolved._attribution.get("learnings.rules.my_rule") or resolved._attribution.get("learnings")
        assert source == ConfigSource.SYSTEM_DOMAIN

        sm = self._make_session_manager(config, tmp_path)
        managed = self._make_managed_session("s1", "alice", resolved)
        sm._sessions["s1"] = managed

        assert sm.write_config_tombstone("s1", "learnings", "rules.my_rule") is True

        user_yaml = yaml.safe_load((tmp_path / "alice.vault" / "config.yaml").read_text())
        assert user_yaml["learnings"]["rules"]["my_rule"] is None

    def test_tombstone_prevents_reseeding(self, tmp_path):
        """After tombstone, resolve() no longer includes the deleted key."""
        import yaml
        from constat.core.tiered_config import TieredConfigLoader

        domain = self._make_domain(facts={"fiscal_year": "April 1", "currency": "USD"})
        config = self._make_config(domains={"sales": domain})

        # Write tombstone for fiscal_year
        user_dir = tmp_path / "alice.vault"
        user_dir.mkdir(parents=True)
        (user_dir / "config.yaml").write_text(yaml.dump({"facts": {"fiscal_year": None}}))

        loader = TieredConfigLoader(
            config=config, user_id="alice", base_dir=tmp_path,
            domain_names=["sales"],
        )
        resolved = loader.resolve()
        assert "fiscal_year" not in resolved.facts
        assert resolved.facts["currency"] == "USD"

    def test_tombstone_preserves_existing_config(self, tmp_path):
        """Writing a tombstone preserves other keys in user config."""
        import yaml
        from constat.core.tiered_config import TieredConfigLoader, ConfigSource

        domain = self._make_domain(facts={"revenue": "Total income"})
        config = self._make_config(domains={"sales": domain})

        # Pre-populate user config with other data
        user_dir = tmp_path / "alice.vault"
        user_dir.mkdir(parents=True)
        (user_dir / "config.yaml").write_text(yaml.dump({
            "facts": {"my_fact": "hello"},
            "relationships": {"custom_rel": "My relationship"},
        }))

        loader = TieredConfigLoader(
            config=config, user_id="alice", base_dir=tmp_path,
            domain_names=["sales"],
        )
        resolved = loader.resolve()

        sm = self._make_session_manager(config, tmp_path)
        managed = self._make_managed_session("s1", "alice", resolved)
        sm._sessions["s1"] = managed

        sm.write_config_tombstone("s1", "facts", "revenue")

        user_yaml = yaml.safe_load((user_dir / "config.yaml").read_text())
        assert user_yaml["facts"]["my_fact"] == "hello"
        assert user_yaml["relationships"]["custom_rel"] == "My relationship"
        assert user_yaml["facts"]["revenue"] is None
