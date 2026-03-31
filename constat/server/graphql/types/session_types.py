# Copyright (c) 2025 Kenneth Stott
# Canary: 7fff21de-c4f6-4ca1-a087-cd1e4cb1343a
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Strawberry GraphQL types for sessions."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

import strawberry


@strawberry.enum
class SessionStatusEnum(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@strawberry.type
class SessionType:
    session_id: str
    user_id: str
    status: SessionStatusEnum
    created_at: datetime
    last_activity: datetime
    current_query: Optional[str] = None
    summary: Optional[str] = None
    active_domains: list[str] = strawberry.field(default_factory=list)
    tables_count: int = 0
    artifacts_count: int = 0
    shared_with: list[str] = strawberry.field(default_factory=list)
    is_public: bool = False


@strawberry.type
class SessionListType:
    sessions: list[SessionType]
    total: int


@strawberry.type
class ShareResponseType:
    status: str
    share_url: str


@strawberry.type
class TogglePublicResponseType:
    status: str
    public: bool
    share_url: str
