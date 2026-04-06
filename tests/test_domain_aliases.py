# Copyright (c) 2025 Kenneth Stott
# Canary: f7752321-ff48-4e82-af1e-1add2900ed18
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for domain resource aliasing (manual + auto)."""

from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from constat.core.config import (
    Config, DomainConfig, DatabaseConfig, APIConfig, DocumentConfig,
    EntityResolutionConfig,
)


class TestDomainConfigAliases:
    """Test that DomainConfig accepts the aliases field."""

    def test_aliases_default_empty(self):
        domain = DomainConfig(name="test")
        assert domain.aliases == {}

    def test_aliases_parsed(self):
        domain = DomainConfig(
            name="Regional Sales",
            aliases={
                "shared-sales": {
                    "databases": {"sales": "sales_emea"},
                    "apis": {"crm": "crm_emea"},
                }
            },
        )
        assert domain.aliases["shared-sales"]["databases"]["sales"] == "sales_emea"
        assert domain.aliases["shared-sales"]["apis"]["crm"] == "crm_emea"


class TestLoadDomainsAliasing:
    """Test _load_domains_into_session alias resolution."""

    def _make_managed(self, config_databases=None, domains=None):
        """Build a minimal ManagedSession mock for alias testing."""
        config = MagicMock(spec=Config)
        config.databases = config_databases or {}
        config.apis = {}
        config.documents = {}
        config.domains = domains or {}

        def _load_domain(filename):
            return config.domains.get(filename)
        config.load_domain = _load_domain

        session = MagicMock()
        session.config = config
        session.schema_manager = None  # skip DB loading
        session.doc_tools = MagicMock()
        session.doc_tools._vector_store = None
        session.skill_manager = MagicMock()
        session.skill_manager.add_domain_skills = MagicMock()

        managed = MagicMock()
        managed.session = session
        managed.active_domains = []
        managed._domain_databases = set()
        return managed

    def test_no_conflict_no_alias(self):
        """Two domains with unique keys — no aliasing needed."""
        from constat.server.routes.sessions import _load_domains_into_session

        domain_a = DomainConfig(
            name="A",
            databases={"db_a": DatabaseConfig(uri="sqlite:///a.db")},
        )
        domain_b = DomainConfig(
            name="B",
            databases={"db_b": DatabaseConfig(uri="sqlite:///b.db")},
        )
        managed = self._make_managed(domains={"a.yaml": domain_a, "b.yaml": domain_b})

        loaded, conflicts = _load_domains_into_session(managed, ["a.yaml", "b.yaml"])

        assert set(loaded) == {"a.yaml", "b.yaml"}
        assert conflicts == []
        # Resources registered with original names
        call_args = managed.session.add_domain_resources.call_args_list
        all_db_keys = set()
        for call in call_args:
            all_db_keys.update(call.kwargs.get("databases", call[1].get("databases", {}) if len(call[1]) > 1 else {}).keys())
        assert "db_a" in all_db_keys
        assert "db_b" in all_db_keys

    def test_auto_alias_on_conflict(self):
        """Two domains with same key — second gets auto-aliased."""
        from constat.server.routes.sessions import _load_domains_into_session

        domain_a = DomainConfig(
            name="Sales EMEA",
            databases={"sales": DatabaseConfig(uri="sqlite:///emea.db")},
        )
        domain_b = DomainConfig(
            name="Sales APAC",
            databases={"sales": DatabaseConfig(uri="sqlite:///apac.db")},
        )
        managed = self._make_managed(domains={
            "sales-emea.yaml": domain_a,
            "sales-apac.yaml": domain_b,
        })

        loaded, conflicts = _load_domains_into_session(managed, ["sales-emea.yaml", "sales-apac.yaml"])

        # Both domains loaded (no rejection)
        assert set(loaded) == {"sales-emea.yaml", "sales-apac.yaml"}
        assert conflicts == []

        # Check that add_domain_resources was called with aliased name for second domain
        call_args = managed.session.add_domain_resources.call_args_list
        all_db_keys = set()
        for call in call_args:
            dbs = call.kwargs.get("databases", {})
            all_db_keys.update(dbs.keys())
        assert "sales" in all_db_keys
        assert "sales-apac--sales" in all_db_keys

    def test_manual_alias(self):
        """Parent domain manually aliases a child domain's key."""
        from constat.server.routes.sessions import _load_domains_into_session

        parent = DomainConfig(
            name="Regional",
            aliases={
                "shared-sales.yaml": {
                    "databases": {"sales": "sales_emea"},
                }
            },
        )
        child = DomainConfig(
            name="Shared Sales",
            databases={"sales": DatabaseConfig(uri="sqlite:///sales.db")},
        )
        managed = self._make_managed(domains={
            "regional.yaml": parent,
            "shared-sales.yaml": child,
        })

        loaded, conflicts = _load_domains_into_session(managed, ["regional.yaml", "shared-sales.yaml"])

        assert set(loaded) == {"regional.yaml", "shared-sales.yaml"}
        assert conflicts == []

        # Check child's "sales" was renamed to "sales_emea"
        call_args = managed.session.add_domain_resources.call_args_list
        all_db_keys = set()
        for call in call_args:
            dbs = call.kwargs.get("databases", {})
            all_db_keys.update(dbs.keys())
        assert "sales_emea" in all_db_keys
        assert "sales" not in all_db_keys

    def test_manual_alias_prevents_auto_alias(self):
        """Manual alias resolves what would be an auto-alias conflict."""
        from constat.server.routes.sessions import _load_domains_into_session

        parent = DomainConfig(
            name="Regional",
            databases={"sales": DatabaseConfig(uri="sqlite:///regional.db")},
            aliases={
                "shared-sales.yaml": {
                    "databases": {"sales": "sales_shared"},
                }
            },
        )
        child = DomainConfig(
            name="Shared Sales",
            databases={"sales": DatabaseConfig(uri="sqlite:///shared.db")},
        )
        managed = self._make_managed(domains={
            "regional.yaml": parent,
            "shared-sales.yaml": child,
        })

        loaded, conflicts = _load_domains_into_session(managed, ["regional.yaml", "shared-sales.yaml"])

        assert set(loaded) == {"regional.yaml", "shared-sales.yaml"}
        # Both loaded, child's sales renamed to sales_shared
        call_args = managed.session.add_domain_resources.call_args_list
        all_db_keys = set()
        for call in call_args:
            dbs = call.kwargs.get("databases", {})
            all_db_keys.update(dbs.keys())
        assert "sales" in all_db_keys  # parent's original
        assert "sales_shared" in all_db_keys  # child's manual alias
        assert not any("--" in k for k in all_db_keys)  # no auto-alias needed

    def test_auto_alias_apis_and_documents(self):
        """Auto-aliasing works for APIs and documents too."""
        from constat.server.routes.sessions import _load_domains_into_session

        domain_a = DomainConfig(
            name="A",
            apis={"crm": APIConfig(url="https://a.example.com")},
            documents={"glossary": DocumentConfig(path="./a.md")},
        )
        domain_b = DomainConfig(
            name="B",
            apis={"crm": APIConfig(url="https://b.example.com")},
            documents={"glossary": DocumentConfig(path="./b.md")},
        )
        managed = self._make_managed(domains={"a.yaml": domain_a, "b.yaml": domain_b})

        loaded, conflicts = _load_domains_into_session(managed, ["a.yaml", "b.yaml"])

        assert set(loaded) == {"a.yaml", "b.yaml"}
        call_args = managed.session.add_domain_resources.call_args_list
        all_api_keys = set()
        all_doc_keys = set()
        for call in call_args:
            all_api_keys.update(call.kwargs.get("apis", {}).keys())
            all_doc_keys.update(call.kwargs.get("documents", {}).keys())
        assert "crm" in all_api_keys
        assert "b--crm" in all_api_keys
        assert "glossary" in all_doc_keys
        assert "b--glossary" in all_doc_keys

    def test_config_db_conflict_auto_aliases(self):
        """Domain key conflicting with base config key gets auto-aliased."""
        from constat.server.routes.sessions import _load_domains_into_session

        domain = DomainConfig(
            name="Sales",
            databases={"main": DatabaseConfig(uri="sqlite:///sales.db")},
        )
        config_dbs = {"main": DatabaseConfig(uri="sqlite:///main.db")}
        managed = self._make_managed(
            config_databases=config_dbs,
            domains={"sales.yaml": domain},
        )

        loaded, conflicts = _load_domains_into_session(managed, ["sales.yaml"])

        assert loaded == ["sales.yaml"]
        call_args = managed.session.add_domain_resources.call_args_list
        db_keys = set()
        for call in call_args:
            db_keys.update(call.kwargs.get("databases", {}).keys())
        assert "sales--main" in db_keys
        assert "main" not in db_keys  # original not used (it's in config)

    def test_alias_map_stored_on_managed(self):
        """Alias mappings are stored on managed session for entity resolution."""
        from constat.server.routes.sessions import _load_domains_into_session

        domain_a = DomainConfig(
            name="A",
            databases={"sales": DatabaseConfig(uri="sqlite:///a.db")},
        )
        domain_b = DomainConfig(
            name="B",
            databases={"sales": DatabaseConfig(uri="sqlite:///b.db")},
        )
        managed = self._make_managed(domains={
            "a.yaml": domain_a,
            "b.yaml": domain_b,
        })

        _load_domains_into_session(managed, ["a.yaml", "b.yaml"])

        # Second domain should have alias mapping
        alias_map = managed._domain_alias_map
        assert "b.yaml" in alias_map
        assert alias_map["b.yaml"]["databases"]["sales"] == "b--sales"


class TestEntityResolutionAliasing:
    """Test that entity resolution source refs get aliased."""

    def test_er_source_aliased(self):
        """Entity resolution config source field gets remapped via alias map."""
        from constat.core.config import EntityResolutionConfig

        er = EntityResolutionConfig(entity_type="COUNTRY", source="sales", table="countries", name_column="name")

        # Simulate what session_manager does
        db_aliases = {"sales": "sales_emea"}
        api_aliases = {}

        if er.source in db_aliases:
            er = er.model_copy(update={"source": db_aliases[er.source]})
        elif er.source in api_aliases:
            er = er.model_copy(update={"source": api_aliases[er.source]})

        assert er.source == "sales_emea"
        assert er.entity_type == "COUNTRY"  # unchanged

    def test_er_source_not_aliased_when_no_match(self):
        """Entity resolution source unchanged when not in alias map."""
        from constat.core.config import EntityResolutionConfig

        er = EntityResolutionConfig(entity_type="PRODUCT", source="inventory", table="products", name_column="name")

        db_aliases = {"sales": "sales_emea"}
        if er.source in db_aliases:
            er = er.model_copy(update={"source": db_aliases[er.source]})

        assert er.source == "inventory"  # unchanged
