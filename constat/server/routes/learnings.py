# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Learnings and configuration REST endpoints."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from constat.core.config import Config
from constat.server.config import ServerConfig
from constat.server.models import (
    ConfigResponse,
    LearningCreateRequest,
    LearningInfo,
    LearningListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_config(request: Request) -> Config:
    """Dependency to get config from app state."""
    return request.app.state.config


def get_server_config(request: Request) -> ServerConfig:
    """Dependency to get server config from app state."""
    return request.app.state.server_config


# In-memory learnings store (would use LearningStore in production)
_learnings: list[dict[str, Any]] = []


@router.get("/learnings", response_model=LearningListResponse)
async def list_learnings(
    category: str | None = None,
    config: Config = Depends(get_config),
) -> LearningListResponse:
    """Get all captured learnings.

    Args:
        category: Optional category filter

    Returns:
        List of learnings
    """
    # Try to get from LearningStore if available
    try:
        from constat.storage.learnings import LearningStore
        store = LearningStore()
        learnings_data = store.get_learnings(category=category)

        return LearningListResponse(
            learnings=[
                LearningInfo(
                    id=l.get("id", str(uuid.uuid4())),
                    content=l["content"],
                    category=l.get("category", "user_correction"),
                    source=l.get("source", "explicit_command"),
                    context=l.get("context"),
                    applied_count=l.get("applied_count", 0),
                    created_at=datetime.fromisoformat(l["created_at"]) if l.get("created_at") else datetime.now(timezone.utc),
                )
                for l in learnings_data
            ]
        )
    except Exception as e:
        logger.warning(f"Could not load from LearningStore: {e}")

    # Fall back to in-memory store
    filtered = _learnings
    if category:
        filtered = [l for l in _learnings if l.get("category") == category]

    return LearningListResponse(
        learnings=[
            LearningInfo(
                id=l["id"],
                content=l["content"],
                category=l.get("category", "user_correction"),
                source=l.get("source", "explicit_command"),
                context=l.get("context"),
                applied_count=l.get("applied_count", 0),
                created_at=datetime.fromisoformat(l["created_at"]),
            )
            for l in filtered
        ]
    )


@router.post("/learnings", response_model=LearningInfo)
async def add_learning(
    body: LearningCreateRequest,
    config: Config = Depends(get_config),
) -> LearningInfo:
    """Add a new learning.

    Args:
        body: Learning content and category

    Returns:
        Created learning
    """
    now = datetime.now(timezone.utc)
    learning_id = str(uuid.uuid4())

    learning = {
        "id": learning_id,
        "content": body.content,
        "category": body.category,
        "source": "explicit_command",
        "context": None,
        "applied_count": 0,
        "created_at": now.isoformat(),
    }

    # Try to persist to LearningStore
    try:
        from constat.storage.learnings import LearningStore
        store = LearningStore()
        store.add_learning(
            content=body.content,
            category=body.category,
            source="explicit_command",
        )
    except Exception as e:
        logger.warning(f"Could not persist to LearningStore: {e}")

    # Also store in memory
    _learnings.append(learning)

    return LearningInfo(
        id=learning_id,
        content=body.content,
        category=body.category,
        source="explicit_command",
        context=None,
        applied_count=0,
        created_at=now,
    )


@router.delete("/learnings/{learning_id}")
async def delete_learning(
    learning_id: str,
    config: Config = Depends(get_config),
) -> dict:
    """Delete a learning.

    Args:
        learning_id: Learning ID to delete

    Returns:
        Deletion confirmation

    Raises:
        404: Learning not found
    """
    global _learnings

    # Try to delete from LearningStore
    try:
        from constat.storage.learnings import LearningStore
        store = LearningStore()
        if store.delete_learning(learning_id):
            return {"status": "deleted", "id": learning_id}
    except Exception as e:
        logger.warning(f"Could not delete from LearningStore: {e}")

    # Try in-memory store
    original_len = len(_learnings)
    _learnings = [l for l in _learnings if l["id"] != learning_id]

    if len(_learnings) == original_len:
        raise HTTPException(status_code=404, detail=f"Learning not found: {learning_id}")

    return {"status": "deleted", "id": learning_id}


@router.get("/config", response_model=ConfigResponse)
async def get_config_sanitized(
    config: Config = Depends(get_config),
) -> ConfigResponse:
    """Get current configuration (sanitized).

    Returns config without sensitive data like API keys.

    Returns:
        Sanitized configuration
    """
    return ConfigResponse(
        databases=list(config.databases.keys()),
        apis=list(config.apis.keys()),
        documents=list(config.documents.keys()),
        llm_provider=config.llm.provider,
        llm_model=config.llm.model,
        execution_timeout=config.execution.timeout_seconds,
    )