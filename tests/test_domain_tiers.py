# Copyright (c) 2025 Kenneth Stott
# Canary: a84db0ee-f52c-4c76-b262-72a5dbc2fecf
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for constat.core.domain_tiers."""

from unittest.mock import MagicMock

import pytest

from constat.core.domain_tiers import get_domain_tier


@pytest.fixture
def config():
    """Minimal Config mock with projects dict."""
    cfg = MagicMock()
    cfg.projects = {}
    return cfg


def _make_domain(tier: str):
    d = MagicMock()
    d.tier = tier
    return d


class TestGetDomainTier:
    def test_none_domain_returns_user(self, config):
        assert get_domain_tier(None, config, "alice") == "user"

    def test_domain_equals_user_id_returns_user(self, config):
        assert get_domain_tier("alice", config, "alice") == "user"

    def test_base_domain_returns_system(self, config):
        assert get_domain_tier("__base__", config, "alice") == "system"

    def test_configured_system_tier_returns_system(self, config):
        config.projects = {"sales": _make_domain("system")}
        assert get_domain_tier("sales", config, "alice") == "system"

    def test_configured_shared_tier_returns_system(self, config):
        config.projects = {"hr": _make_domain("shared")}
        assert get_domain_tier("hr", config, "alice") == "system"

    def test_configured_user_tier_returns_user(self, config):
        config.projects = {"personal": _make_domain("user")}
        assert get_domain_tier("personal", config, "alice") == "user"

    def test_unknown_domain_returns_user(self, config):
        assert get_domain_tier("unknown", config, "alice") == "user"

    def test_empty_projects_unknown_domain_returns_user(self, config):
        config.projects = {}
        assert get_domain_tier("anything", config, "bob") == "user"
