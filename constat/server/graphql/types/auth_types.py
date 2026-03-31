# Copyright (c) 2025 Kenneth Stott
# Canary: 176dfd8c-a794-483f-ab4f-c8d66757d045
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Strawberry GraphQL types for authentication."""

from __future__ import annotations

from typing import Optional

import strawberry
from strawberry.scalars import JSON


@strawberry.type
class AuthPayload:
    token: str
    user_id: str
    email: str
    vault_unlocked: bool | None = None


@strawberry.type
class PasskeyOptions:
    """JSON options for WebAuthn ceremony."""
    options_json: JSON


@strawberry.type
class ModelRouteInfoType:
    provider: str
    model: str


@strawberry.type
class ServerConfigType:
    databases: list[str]
    apis: list[str]
    documents: list[str]
    llm_provider: str
    llm_model: str
    execution_timeout: int
    task_routing: JSON


@strawberry.type
class UserPermissionsType:
    user_id: str
    email: str | None
    admin: bool
    persona: str
    domains: list[str]
    databases: list[str]
    documents: list[str]
    apis: list[str]
    visibility: JSON
    writes: JSON
    feedback: JSON
