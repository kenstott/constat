# Copyright (c) 2025 Kenneth Stott
# Canary: c444bffe-251d-43de-ad6d-0646f6f7ce0a
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Sensitive path detection and value masking for deployment diffs."""

SENSITIVE_KEYS = {
    "uri", "password", "api_key", "auth_token", "api_token",
    "oauth2_client_secret", "smtp_password", "key", "secret",
    "aws_secret_access_key", "aws_session_token", "admin_token",
    "password_hash", "firebase_api_key", "notion_token",
}


def is_sensitive_path(path: str) -> bool:
    """Check if a dot-delimited config path contains sensitive keys."""
    parts = path.split(".")
    return any(part in SENSITIVE_KEYS for part in parts)


def mask_value(value) -> str:
    """Mask a sensitive value for display."""
    return "***"
