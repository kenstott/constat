# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for role configuration and permission gating."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from constat.server.role_config import (
    RoleDefinition,
    RolesConfig,
    load_roles_config,
    require_write,
)
from constat.server.config import UserPermissions


# ---------------------------------------------------------------------------
# load_roles_config
# ---------------------------------------------------------------------------

class TestLoadRolesConfig:
    def test_loads_all_roles(self):
        config = load_roles_config()
        assert set(config.roles.keys()) == {
            "platform_admin",
            "domain_builder",
            "sme",
            "domain_user",
            "viewer",
        }

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_roles_config(tmp_path / "nonexistent.yaml")

    def test_custom_path(self, tmp_path):
        p = tmp_path / "roles.yaml"
        p.write_text("roles:\n  custom:\n    description: test\n    visibility: {x: true}\n    writes: {}\n    feedback: {}\n")
        config = load_roles_config(p)
        assert "custom" in config.roles


# ---------------------------------------------------------------------------
# RolesConfig.can_see / can_write
# ---------------------------------------------------------------------------

class TestCanSee:
    @pytest.fixture(autouse=True)
    def _config(self):
        self.config = load_roles_config()

    @pytest.mark.parametrize("section", [
        "results", "databases", "apis", "documents", "system_prompt",
        "roles", "skills", "learnings", "code", "inference_code", "facts", "glossary",
    ])
    def test_platform_admin_sees_all(self, section):
        assert self.config.can_see("platform_admin", section)

    @pytest.mark.parametrize("section", [
        "results", "databases", "apis", "documents", "system_prompt",
        "roles", "skills", "learnings", "code", "inference_code", "facts", "glossary",
    ])
    def test_domain_builder_sees_all(self, section):
        assert self.config.can_see("domain_builder", section)

    def test_sme_sees_results_learnings_facts_glossary(self):
        for s in ("results", "learnings", "facts", "glossary"):
            assert self.config.can_see("sme", s)

    def test_sme_cannot_see_databases(self):
        assert not self.config.can_see("sme", "databases")

    def test_domain_user_sees_results_only(self):
        assert self.config.can_see("domain_user", "results")
        assert not self.config.can_see("domain_user", "glossary")

    def test_viewer_sees_results_only(self):
        assert self.config.can_see("viewer", "results")
        assert not self.config.can_see("viewer", "skills")

    def test_unknown_role_sees_nothing(self):
        assert not self.config.can_see("nonexistent", "results")


class TestCanWrite:
    @pytest.fixture(autouse=True)
    def _config(self):
        self.config = load_roles_config()

    @pytest.mark.parametrize("resource", [
        "sources", "glossary", "skills", "roles", "facts",
        "learnings", "system_prompt", "tier_promote",
    ])
    def test_platform_admin_writes_all(self, resource):
        assert self.config.can_write("platform_admin", resource)

    def test_sme_can_write_glossary_facts_learnings(self):
        for r in ("glossary", "facts", "learnings"):
            assert self.config.can_write("sme", r)

    def test_sme_cannot_write_skills(self):
        assert not self.config.can_write("sme", "skills")

    def test_domain_user_cannot_write_anything(self):
        assert not self.config.can_write("domain_user", "glossary")
        assert not self.config.can_write("domain_user", "sources")

    def test_viewer_cannot_write_anything(self):
        assert not self.config.can_write("viewer", "glossary")
        assert not self.config.can_write("viewer", "tier_promote")


# ---------------------------------------------------------------------------
# UserPermissions admin → role validator
# ---------------------------------------------------------------------------

class TestUserPermissionsValidator:
    def test_admin_true_default_role_upgrades(self):
        p = UserPermissions(admin=True)
        assert p.role == "platform_admin"

    def test_admin_true_explicit_role_preserved(self):
        p = UserPermissions(admin=True, role="domain_builder")
        assert p.role == "domain_builder"

    def test_admin_false_default_role_stays_viewer(self):
        p = UserPermissions(admin=False)
        assert p.role == "viewer"

    def test_non_admin_explicit_role_preserved(self):
        p = UserPermissions(admin=False, role="sme")
        assert p.role == "sme"


# ---------------------------------------------------------------------------
# require_write dependency
# ---------------------------------------------------------------------------

class TestRequireWrite:
    @pytest.mark.asyncio
    async def test_allows_admin_write(self):
        dep = require_write("glossary")
        config = load_roles_config()

        request = MagicMock()
        request.app.state.roles_config = config
        request.app.state.server_config = MagicMock()

        with patch("constat.server.permissions.get_user_permissions") as mock_perms:
            mock_perms.return_value = MagicMock(role="platform_admin")
            # Should not raise
            await dep(request=request, user_id="uid", email="admin@test.com")

    @pytest.mark.asyncio
    async def test_blocks_viewer_write(self):
        dep = require_write("glossary")
        config = load_roles_config()

        request = MagicMock()
        request.app.state.roles_config = config
        request.app.state.server_config = MagicMock()

        with patch("constat.server.permissions.get_user_permissions") as mock_perms:
            mock_perms.return_value = MagicMock(role="viewer")
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await dep(request=request, user_id="uid", email="viewer@test.com")
            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_allows_when_no_roles_config(self):
        dep = require_write("glossary")

        request = MagicMock()
        request.app.state.roles_config = None

        # Should not raise — backwards compat
        await dep(request=request, user_id="uid", email="anyone@test.com")

    @pytest.mark.asyncio
    async def test_sme_can_write_glossary(self):
        dep = require_write("glossary")
        config = load_roles_config()

        request = MagicMock()
        request.app.state.roles_config = config
        request.app.state.server_config = MagicMock()

        with patch("constat.server.permissions.get_user_permissions") as mock_perms:
            mock_perms.return_value = MagicMock(role="sme")
            await dep(request=request, user_id="uid", email="sme@test.com")

    @pytest.mark.asyncio
    async def test_sme_cannot_write_skills(self):
        dep = require_write("skills")
        config = load_roles_config()

        request = MagicMock()
        request.app.state.roles_config = config
        request.app.state.server_config = MagicMock()

        with patch("constat.server.permissions.get_user_permissions") as mock_perms:
            mock_perms.return_value = MagicMock(role="sme")
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await dep(request=request, user_id="uid", email="sme@test.com")
            assert exc_info.value.status_code == 403
