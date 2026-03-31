# Copyright (c) 2025 Kenneth Stott
# Canary: 427b9030-675b-47a4-9af2-ebe4f834039f
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for constat.server.vault — AES-256-GCM file-level encryption."""

import pytest

from constat.server.vault import UserVault


@pytest.fixture
def user_dir(tmp_path):
    return tmp_path / "user1"


@pytest.fixture
def db_content():
    return b"DUCKDB_MAGIC_BYTES" + b"\x00" * 1024


def _write_plain(user_dir, content):
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "vectors.duckdb").write_bytes(content)


class TestEncryptMode:
    def test_create_unlock_roundtrip(self, user_dir, db_content):
        """Create vault, write DB, lock, then unlock — content intact."""
        vault = UserVault(user_dir, encrypt=True)
        vault.create(b"secret-key-material")
        _write_plain(user_dir, db_content)
        vault.lock()

        assert not vault.plain_path.exists()
        assert vault.enc_path.exists()
        assert vault.salt_path.exists()

        vault2 = UserVault(user_dir, encrypt=True)
        path = vault2.unlock(b"secret-key-material")
        assert path == vault.plain_path
        assert path.read_bytes() == db_content

    def test_lock_removes_plaintext(self, user_dir, db_content):
        vault = UserVault(user_dir, encrypt=True)
        vault.create(b"key")
        _write_plain(user_dir, db_content)
        vault.lock()

        assert not vault.plain_path.exists()
        assert vault.enc_path.exists()

    def test_wrong_key_fails(self, user_dir, db_content):
        vault = UserVault(user_dir, encrypt=True)
        vault.create(b"correct-key")
        _write_plain(user_dir, db_content)
        vault.lock()

        vault2 = UserVault(user_dir, encrypt=True)
        with pytest.raises(Exception):  # InvalidTag from cryptography
            vault2.unlock(b"wrong-key")

    def test_lock_without_unlock_raises(self, user_dir):
        user_dir.mkdir(parents=True, exist_ok=True)
        vault = UserVault(user_dir, encrypt=True)
        with pytest.raises(RuntimeError, match="Vault not unlocked"):
            vault.lock()


class TestNoEncryptMode:
    def test_passthrough_create(self, user_dir, db_content):
        user_dir.mkdir(parents=True, exist_ok=True)
        vault = UserVault(user_dir, encrypt=False)
        path = vault.create(b"ignored")
        assert path == vault.plain_path

    def test_passthrough_unlock(self, user_dir, db_content):
        user_dir.mkdir(parents=True, exist_ok=True)
        vault = UserVault(user_dir, encrypt=False)
        path = vault.unlock(b"ignored")
        assert path == vault.plain_path

    def test_passthrough_lock(self, user_dir, db_content):
        vault = UserVault(user_dir, encrypt=False)
        _write_plain(user_dir, db_content)
        vault.lock()  # no-op
        assert vault.plain_path.exists()  # plaintext NOT removed


class TestServerConfig:
    def test_vault_encrypt_default(self):
        from constat.server.config import ServerConfig

        cfg = ServerConfig()
        assert cfg.vault_encrypt is False

    def test_vault_encrypt_yaml(self):
        from constat.server.config import ServerConfig

        cfg = ServerConfig.from_yaml_data({"vault_encrypt": True})
        assert cfg.vault_encrypt is True

    def test_vault_encrypt_env(self, monkeypatch):
        from constat.server.config import ServerConfig

        monkeypatch.setenv("CONSTAT_VAULT_ENCRYPT", "true")
        cfg = ServerConfig()
        assert cfg.vault_encrypt is True
