# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for persona configuration and permission gating."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from constat.server.persona_config import (
    PersonaDefinition,
    PersonasConfig,
    load_personas_config,
    require_write,
)
from constat.server.config import UserPermissions


# ---------------------------------------------------------------------------
# load_personas_config
# ---------------------------------------------------------------------------

class TestLoadPersonasConfig:
    def test_loads_all_personas(self):
        config = load_personas_config()
        assert set(config.personas.keys()) == {
            "platform_admin",
            "domain_builder",
            "sme",
            "domain_user",
            "viewer",
        }

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_personas_config(tmp_path / "nonexistent.yaml")

    def test_custom_path(self, tmp_path):
        p = tmp_path / "personas.yaml"
        p.write_text("personas:\n  custom:\n    description: test\n    visibility: {x: true}\n    writes: {}\n    feedback: {}\n")
        config = load_personas_config(p)
        assert "custom" in config.personas


# ---------------------------------------------------------------------------
# PersonasConfig.can_see / can_write
# ---------------------------------------------------------------------------

class TestCanSee:
    @pytest.fixture(autouse=True)
    def _config(self):
        self.config = load_personas_config()

    @pytest.mark.parametrize("section", [
        "results", "databases", "apis", "documents", "system_prompt",
        "agents", "skills", "learnings", "code", "inference_code", "facts", "glossary",
    ])
    def test_platform_admin_sees_all(self, section):
        assert self.config.can_see("platform_admin", section)

    @pytest.mark.parametrize("section", [
        "results", "databases", "apis", "documents", "system_prompt",
        "agents", "skills", "learnings", "code", "inference_code", "facts", "glossary",
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

    def test_unknown_persona_sees_nothing(self):
        assert not self.config.can_see("nonexistent", "results")


class TestCanWrite:
    @pytest.fixture(autouse=True)
    def _config(self):
        self.config = load_personas_config()

    @pytest.mark.parametrize("resource", [
        "sources", "glossary", "skills", "agents", "facts",
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
# UserPermissions admin → persona validator
# ---------------------------------------------------------------------------

class TestUserPermissionsValidator:
    def test_platform_admin_persona_implies_admin(self):
        p = UserPermissions(persona="platform_admin")
        assert p.persona == "platform_admin"

    def test_default_persona_is_viewer(self):
        p = UserPermissions()
        assert p.persona == "viewer"

    def test_explicit_persona_preserved(self):
        p = UserPermissions(persona="sme")
        assert p.persona == "sme"

    def test_domain_builder_persona(self):
        p = UserPermissions(persona="domain_builder")
        assert p.persona == "domain_builder"


# ---------------------------------------------------------------------------
# require_write dependency
# ---------------------------------------------------------------------------

class TestRequireWrite:
    @pytest.mark.asyncio
    async def test_allows_admin_write(self):
        dep = require_write("glossary")
        config = load_personas_config()

        request = MagicMock()
        request.app.state.personas_config = config
        request.app.state.server_config = MagicMock(auth_disabled=False)

        with patch("constat.server.permissions.get_user_permissions") as mock_perms:
            mock_perms.return_value = MagicMock(persona="platform_admin")
            # Should not raise
            await dep(request=request, user_id="uid", email="admin@test.com")

    @pytest.mark.asyncio
    async def test_blocks_viewer_write(self):
        dep = require_write("glossary")
        config = load_personas_config()

        request = MagicMock()
        request.app.state.personas_config = config
        request.app.state.server_config = MagicMock(auth_disabled=False)

        with patch("constat.server.permissions.get_user_permissions") as mock_perms:
            mock_perms.return_value = MagicMock(persona="viewer")
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await dep(request=request, user_id="uid", email="viewer@test.com")
            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_allows_when_no_personas_config(self):
        dep = require_write("glossary")

        request = MagicMock()
        request.app.state.personas_config = None
        request.app.state.server_config = MagicMock(auth_disabled=False)

        # Should not raise — backwards compat
        await dep(request=request, user_id="uid", email="anyone@test.com")

    @pytest.mark.asyncio
    async def test_sme_can_write_glossary(self):
        dep = require_write("glossary")
        config = load_personas_config()

        request = MagicMock()
        request.app.state.personas_config = config
        request.app.state.server_config = MagicMock(auth_disabled=False)

        with patch("constat.server.permissions.get_user_permissions") as mock_perms:
            mock_perms.return_value = MagicMock(persona="sme")
            await dep(request=request, user_id="uid", email="sme@test.com")

    @pytest.mark.asyncio
    async def test_sme_cannot_write_skills(self):
        dep = require_write("skills")
        config = load_personas_config()

        request = MagicMock()
        request.app.state.personas_config = config
        request.app.state.server_config = MagicMock(auth_disabled=False)

        with patch("constat.server.permissions.get_user_permissions") as mock_perms:
            mock_perms.return_value = MagicMock(persona="sme")
            from fastapi import HTTPException
            with pytest.raises(HTTPException) as exc_info:
                await dep(request=request, user_id="uid", email="sme@test.com")
            assert exc_info.value.status_code == 403
