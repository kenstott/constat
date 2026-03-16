# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Server-local username/password authentication (no Firebase required)."""

import hashlib
import os
import secrets
import time


def hash_password(password: str) -> str:
    """Hash a password with scrypt. Returns 'scrypt:salt_hex:hash_hex'."""
    salt = os.urandom(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return f"scrypt:{salt.hex()}:{h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Verify password against stored scrypt hash."""
    _, salt_hex, hash_hex = stored.split(":")
    salt = bytes.fromhex(salt_hex)
    h = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return secrets.compare_digest(h.hex(), hash_hex)


# In-memory token store: token -> (user_id, email, expiry_monotonic)
_local_tokens: dict[str, tuple[str, str, float]] = {}
_TOKEN_TTL = 86400  # 24h
_TOKEN_MAX = 500


def create_local_token(user_id: str, email: str) -> str:
    """Create an opaque session token for a local user."""
    token = secrets.token_urlsafe(32)
    now = time.monotonic()
    if len(_local_tokens) >= _TOKEN_MAX:
        oldest = min(_local_tokens, key=lambda k: _local_tokens[k][2])
        del _local_tokens[oldest]
    _local_tokens[token] = (user_id, email, now + _TOKEN_TTL)
    return token


def validate_local_token(token: str) -> tuple[str, str] | None:
    """Validate an opaque token. Returns (user_id, email) or None."""
    entry = _local_tokens.get(token)
    if entry is None:
        return None
    user_id, email, expiry = entry
    if time.monotonic() > expiry:
        del _local_tokens[token]
        return None
    return (user_id, email)
