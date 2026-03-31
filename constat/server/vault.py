# Copyright (c) 2025 Kenneth Stott
# Canary: 95d62f3b-483c-45db-9e9c-1fa575b849e4
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""AES-256-GCM file-level encryption for per-user DuckDB vector stores."""

import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


_INFO = b"constat-vault"
_NONCE_LEN = 12


def _derive_key(key_material: bytes, salt: bytes) -> bytes:
    """Derive a 256-bit key from key_material + salt via HKDF-SHA256."""
    return HKDF(
        algorithm=SHA256(),
        length=32,
        salt=salt,
        info=_INFO,
    ).derive(key_material)


class UserVault:
    """Encrypt/decrypt a user's vectors.duckdb file with AES-256-GCM.

    When encrypt=False, all operations are no-ops that return the plain path.
    """

    SALT_FILE = ".vault_salt"
    PLAIN_NAME = "vectors.duckdb"
    ENC_NAME = "vectors.duckdb.enc"

    def __init__(self, user_dir: Path, encrypt: bool = False) -> None:
        self.user_dir = user_dir
        self._encrypt = encrypt
        self._key: bytes | None = None

    @property
    def plain_path(self) -> Path:
        return self.user_dir / self.PLAIN_NAME

    @property
    def enc_path(self) -> Path:
        return self.user_dir / self.ENC_NAME

    @property
    def salt_path(self) -> Path:
        return self.user_dir / self.SALT_FILE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, key_material: bytes) -> Path:
        """Create vault: generate salt, derive key. Returns plain_path.

        Does NOT encrypt yet — caller writes the DB first, then calls lock().
        """
        if not self._encrypt:
            return self.plain_path

        self.user_dir.mkdir(parents=True, exist_ok=True)
        salt = os.urandom(32)
        self.salt_path.write_bytes(salt)
        self._key = _derive_key(key_material, salt)
        return self.plain_path

    def unlock(self, key_material: bytes) -> Path:
        """Decrypt vectors.duckdb.enc -> vectors.duckdb, return plain path."""
        if not self._encrypt:
            return self.plain_path

        salt = self.salt_path.read_bytes()
        self._key = _derive_key(key_material, salt)
        self._decrypt_db()
        return self.plain_path

    def lock(self) -> None:
        """Encrypt vectors.duckdb -> vectors.duckdb.enc, remove plaintext."""
        if not self._encrypt:
            return

        if self._key is None:
            raise RuntimeError("Vault not unlocked — cannot lock without key")

        self._encrypt_db()
        self.plain_path.unlink()
        self._key = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _encrypt_db(self) -> None:
        plaintext = self.plain_path.read_bytes()
        nonce = os.urandom(_NONCE_LEN)
        aesgcm = AESGCM(self._key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        self.enc_path.write_bytes(nonce + ciphertext)

    def _decrypt_db(self) -> None:
        raw = self.enc_path.read_bytes()
        nonce = raw[:_NONCE_LEN]
        ciphertext = raw[_NONCE_LEN:]
        aesgcm = AESGCM(self._key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        self.plain_path.write_bytes(plaintext)
