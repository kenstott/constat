# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for vault status and creation endpoints."""

from __future__ import annotations
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_request(data_dir: Path, vault_encrypt: bool) -> MagicMock:
    """Build a mock FastAPI Request with server_config."""
    server_config = MagicMock()
    server_config.data_dir = data_dir
    server_config.vault_encrypt = vault_encrypt
    request = MagicMock()
    request.app.state.server_config = server_config
    return request


# ---------------------------------------------------------------------------
# vault_status
# ---------------------------------------------------------------------------

class TestVaultStatus:
    @pytest.mark.asyncio
    async def test_vault_encrypt_disabled_always_has_vault(self, tmp_path):
        from constat.server.routes.vault import vault_status
        request = _make_request(tmp_path / ".constat", vault_encrypt=False)
        result = await vault_status(request, "user1")
        assert result == {"has_vault": True}

    @pytest.mark.asyncio
    async def test_no_salt_file_returns_false(self, tmp_path):
        from constat.server.routes.vault import vault_status
        request = _make_request(tmp_path, vault_encrypt=True)
        result = await vault_status(request, "user1")
        assert result == {"has_vault": False}

    @pytest.mark.asyncio
    async def test_salt_file_exists_returns_true(self, tmp_path):
        from constat.server.routes.vault import vault_status
        from constat.core.paths import user_vault_dir
        from constat.server.vault import UserVault

        user_dir = user_vault_dir(tmp_path, "user1")
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / UserVault.SALT_FILE).write_bytes(b"fakesalt" * 4)

        request = _make_request(tmp_path, vault_encrypt=True)
        result = await vault_status(request, "user1")
        assert result == {"has_vault": True}


# ---------------------------------------------------------------------------
# create_vault
# ---------------------------------------------------------------------------

class TestCreateVault:
    @pytest.mark.asyncio
    async def test_vault_encrypt_disabled_returns_ok(self, tmp_path):
        from constat.server.routes.vault import create_vault, CreateVaultRequest
        request = _make_request(tmp_path / ".constat", vault_encrypt=False)
        result = await create_vault(request, "user1", CreateVaultRequest(password="secret"))
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_creates_salt_file(self, tmp_path):
        from constat.server.routes.vault import create_vault, CreateVaultRequest
        from constat.core.paths import user_vault_dir
        from constat.server.vault import UserVault

        request = _make_request(tmp_path, vault_encrypt=True)
        result = await create_vault(request, "user2", CreateVaultRequest(password="mypassword"))
        assert result == {"status": "ok"}

        user_dir = user_vault_dir(tmp_path, "user2")
        assert (user_dir / UserVault.SALT_FILE).exists()
        assert (user_dir / UserVault.SALT_FILE).stat().st_size == 32

    @pytest.mark.asyncio
    async def test_409_if_vault_already_exists(self, tmp_path):
        from fastapi import HTTPException
        from constat.server.routes.vault import create_vault, CreateVaultRequest
        from constat.core.paths import user_vault_dir
        from constat.server.vault import UserVault

        user_dir = user_vault_dir(tmp_path, "user3")
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / UserVault.SALT_FILE).write_bytes(b"x" * 32)

        request = _make_request(tmp_path, vault_encrypt=True)
        with pytest.raises(HTTPException) as exc_info:
            await create_vault(request, "user3", CreateVaultRequest(password="pw"))
        assert exc_info.value.status_code == 409
