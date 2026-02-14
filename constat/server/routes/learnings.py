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
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from constat.core.config import Config
from constat.server.auth import CurrentUserId
from constat.server.config import ServerConfig
from constat.server.models import (
    ConfigResponse,
    LearningCreateRequest,
    LearningInfo,
    LearningListResponse,
    ProjectDetailResponse,
    ProjectInfo,
    ProjectListResponse,
    RuleCreateRequest,
    RuleInfo,
    RuleUpdateRequest,
)
from constat.storage.learnings import LearningCategory, LearningSource

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
    user_id: CurrentUserId,
    category: str | None = None,
    _config: Config = Depends(get_config),
) -> LearningListResponse:
    """Get all captured learnings for the authenticated user.

    Args:
        user_id: Authenticated user ID
        category: Optional category filter
        config: Injected application config

    Returns:
        List of learnings
    """
    logger.info(f"[LEARNINGS] Fetching learnings for user_id={user_id}")
    # Try to get from LearningStore if available
    try:
        from constat.storage.learnings import LearningStore
        store = LearningStore(user_id=user_id)
        logger.info(f"[LEARNINGS] LearningStore file_path={store.file_path}, exists={store.file_path.exists()}")
        cat_enum = LearningCategory(category) if category else None
        learnings_data = store.list_raw_learnings(category=cat_enum, limit=100)
        rules_data = store.list_rules(category=cat_enum, limit=50)
        logger.info(f"[LEARNINGS] Loaded {len(learnings_data)} learnings, {len(rules_data)} rules")

        return LearningListResponse(
            learnings=[
                LearningInfo(
                    id=l.get("id", str(uuid.uuid4())),
                    content=l.get("correction", ""),  # YAML uses 'correction' field
                    category=l.get("category", LearningCategory.USER_CORRECTION.value),
                    source=l.get("source", LearningSource.EXPLICIT_COMMAND.value),
                    context=l.get("context"),
                    applied_count=l.get("applied_count", 0),
                    created_at=datetime.fromisoformat(l["created"]) if l.get("created") else datetime.now(timezone.utc),
                )
                for l in learnings_data
            ],
            rules=[
                RuleInfo(
                    id=r.get("id", ""),
                    summary=r.get("summary", ""),
                    category=r.get("category", LearningCategory.USER_CORRECTION.value),
                    confidence=r.get("confidence", 0.0),
                    source_count=len(r.get("source_learnings", [])),
                    tags=r.get("tags", []),
                )
                for r in rules_data
            ],
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
                category=l.get("category", LearningCategory.USER_CORRECTION.value),
                source=l.get("source", LearningSource.EXPLICIT_COMMAND.value),
                context=l.get("context"),
                applied_count=l.get("applied_count", 0),
                created_at=datetime.fromisoformat(l["created_at"]),
            )
            for l in filtered
        ],
        rules=[],
    )


@router.post("/learnings", response_model=LearningInfo)
async def add_learning(
    body: LearningCreateRequest,
    user_id: CurrentUserId,
    _config: Config = Depends(get_config),
) -> LearningInfo:
    """Add a new learning for the authenticated user.

    Args:
        body: Learning content and category
        user_id: Authenticated user ID
        config: Injected application config

    Returns:
        Created learning
    """
    now = datetime.now(timezone.utc)
    learning_id = str(uuid.uuid4())

    learning = {
        "id": learning_id,
        "content": body.content,
        "category": body.category,
        "source": LearningSource.EXPLICIT_COMMAND.value,
        "context": None,
        "applied_count": 0,
        "created_at": now.isoformat(),
    }

    # Try to persist to LearningStore
    try:
        from constat.storage.learnings import LearningStore
        store = LearningStore(user_id=user_id)
        store.save_learning(
            correction=body.content,
            category=LearningCategory(body.category),
            context={},
            source=LearningSource.EXPLICIT_COMMAND,
        )
    except Exception as e:
        logger.warning(f"Could not persist to LearningStore: {e}")

    # Also store in memory
    _learnings.append(learning)

    return LearningInfo(
        id=learning_id,
        content=body.content,
        category=body.category,
        source=LearningSource.EXPLICIT_COMMAND.value,
        context=None,
        applied_count=0,
        created_at=now,
    )


@router.delete("/learnings/{learning_id}")
async def delete_learning(
    learning_id: str,
    user_id: CurrentUserId,
    _config: Config = Depends(get_config),
) -> dict:
    """Delete a learning for the authenticated user.

    Args:
        learning_id: Learning ID to delete
        user_id: Authenticated user ID
        config: Injected application config

    Returns:
        Deletion confirmation

    Raises:
        404: Learning not found
    """
    global _learnings

    # Try to delete from LearningStore
    try:
        from constat.storage.learnings import LearningStore
        store = LearningStore(user_id=user_id)
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


@router.post("/learnings/compact")
async def compact_learnings(
    user_id: CurrentUserId,
    config: Config = Depends(get_config),
) -> dict:
    """Compact similar learnings into rules using LLM.

    This analyzes pending learnings and groups similar ones into rules.

    Returns:
        Compaction results with counts of rules created/strengthened
    """
    try:
        from constat.storage.learnings import LearningStore
        from constat.learning.compactor import LearningCompactor
        from constat.providers import TaskRouter

        store = LearningStore(user_id=user_id)
        stats = store.get_stats()
        unpromoted = stats.get("unpromoted", 0)

        if unpromoted < 2:
            return {
                "status": "skipped",
                "message": f"Not enough learnings to compact ({unpromoted} pending, need at least 2)",
                "rules_created": 0,
                "learnings_archived": 0,
            }

        # Create LLM router for compaction
        llm = TaskRouter(config.llm)

        compactor = LearningCompactor(store, llm)
        result = compactor.compact(dry_run=False)

        return {
            "status": "success",
            "rules_created": result.rules_created,
            "rules_strengthened": result.rules_strengthened,
            "rules_merged": result.rules_merged,
            "learnings_archived": result.learnings_archived,
            "groups_found": result.groups_found,
            "skipped_low_confidence": result.skipped_low_confidence,
            "errors": result.errors,
        }
    except ImportError as e:
        logger.warning(f"Compactor not available: {e}")
        return {
            "status": "error",
            "message": "Learning compactor not available",
            "rules_created": 0,
            "learnings_archived": 0,
        }
    except Exception as e:
        logger.error(f"Error compacting learnings: {e}")
        return {
            "status": "error",
            "message": str(e),
            "rules_created": 0,
            "learnings_archived": 0,
        }


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


# ============================================================================
# Project Endpoints
# ============================================================================


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(
    config: Config = Depends(get_config),
) -> ProjectListResponse:
    """List available projects from the projects directory.

    Projects are YAML files defining reusable collections of databases,
    APIs, and documents. The projects_path in config determines where
    to look for project files.

    Returns:
        List of available projects
    """
    project_infos = config.list_projects()
    return ProjectListResponse(
        projects=[ProjectInfo(**p) for p in project_infos]
    )


@router.get("/projects/{filename}", response_model=ProjectDetailResponse)
async def get_project(
    filename: str,
    config: Config = Depends(get_config),
) -> ProjectDetailResponse:
    """Get details for a specific project.

    Args:
        filename: Project YAML filename (e.g., 'sales-analytics.yaml')
        config: Injected application config

    Returns:
        Project details including data source names

    Raises:
        404: Project not found
    """
    project = config.load_project(filename)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {filename}")

    return ProjectDetailResponse(
        filename=filename,
        name=project.name,
        description=project.description,
        databases=list(project.databases.keys()),
        apis=list(project.apis.keys()),
        documents=list(project.documents.keys()),
    )


@router.get("/projects/{filename}/content")
async def get_project_content(
    filename: str,
    config: Config = Depends(get_config),
) -> dict:
    """Get the raw YAML content of a project file.

    Args:
        filename: Project YAML filename
        config: Injected application config

    Returns:
        Dict with 'content' (YAML string) and 'path' (full file path)

    Raises:
        404: Project not found
    """
    # noinspection DuplicatedCode
    project = config.load_project(filename)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {filename}")

    if not project.source_path:
        raise HTTPException(status_code=400, detail="Project has no source path (inline config)")

    project_path = Path(project.source_path)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail=f"Project file not found: {project_path}")

    content = project_path.read_text()
    return {
        "content": content,
        "path": str(project_path),
        "filename": filename,
    }


@router.put("/projects/{filename}/content")
async def update_project_content(
    filename: str,
    body: dict,
    config: Config = Depends(get_config),
) -> dict:
    """Update the YAML content of a project file.

    Args:
        filename: Project YAML filename
        body: Dict with 'content' (new YAML string)
        config: Injected application config

    Returns:
        Status confirmation

    Raises:
        404: Project not found
        400: Invalid YAML
    """
    # noinspection DuplicatedCode
    project = config.load_project(filename)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {filename}")

    if not project.source_path:
        raise HTTPException(status_code=400, detail="Project has no source path (inline config)")

    project_path = Path(project.source_path)
    if not project_path.exists():
        raise HTTPException(status_code=404, detail=f"Project file not found: {project_path}")

    content = body.get("content")
    if not content:
        raise HTTPException(status_code=400, detail="Content is required")

    # Validate YAML before saving
    import yaml
    try:
        yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    # Write the file
    project_path.write_text(content)

    return {
        "status": "saved",
        "filename": filename,
        "path": str(project_path),
    }


# ============================================================================
# Rule Endpoints
# ============================================================================


@router.post("/rules", response_model=RuleInfo)
async def add_rule(
    body: RuleCreateRequest,
    user_id: CurrentUserId,
    _config: Config = Depends(get_config),
) -> RuleInfo:
    """Add a new rule directly.

    Args:
        body: Rule content and metadata
        user_id: Authenticated user ID
        config: Injected application config

    Returns:
        Created rule
    """
    try:
        from constat.storage.learnings import LearningStore

        store = LearningStore(user_id=user_id)

        # Map string category to enum
        try:
            category = LearningCategory(body.category)
        except ValueError:
            category = LearningCategory.USER_CORRECTION

        rule_id = store.save_rule(
            summary=body.summary,
            category=category,
            confidence=body.confidence,
            source_learnings=[],  # User-created rules have no source learnings
            tags=body.tags,
        )

        return RuleInfo(
            id=rule_id,
            summary=body.summary,
            category=body.category,
            confidence=body.confidence,
            source_count=0,
            tags=body.tags,
        )
    except Exception as e:
        logger.error(f"Error creating rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/rules/{rule_id}", response_model=RuleInfo)
async def update_rule(
    rule_id: str,
    body: RuleUpdateRequest,
    user_id: CurrentUserId,
    _config: Config = Depends(get_config),
) -> RuleInfo:
    """Update an existing rule.

    Args:
        rule_id: Rule ID to update
        body: Fields to update
        user_id: Authenticated user ID
        config: Injected application config

    Returns:
        Updated rule

    Raises:
        404: Rule not found
    """
    try:
        from constat.storage.learnings import LearningStore

        store = LearningStore(user_id=user_id)

        # Check if rule exists
        rules = store.list_rules()
        existing = next((r for r in rules if r["id"] == rule_id), None)
        if not existing:
            raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

        # Update rule
        success = store.update_rule(
            rule_id=rule_id,
            summary=body.summary,
            tags=body.tags,
            confidence=body.confidence,
        )

        if not success:
            raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")

        # Fetch updated rule
        rules = store.list_rules()
        updated = next((r for r in rules if r["id"] == rule_id), None)

        return RuleInfo(
            id=rule_id,
            summary=updated["summary"] if updated else body.summary or existing["summary"],
            category=updated["category"] if updated else existing["category"],
            confidence=updated["confidence"] if updated else body.confidence or existing["confidence"],
            source_count=len(updated.get("source_learnings", [])) if updated else existing.get("source_count", 0),
            tags=updated["tags"] if updated else body.tags or existing.get("tags", []),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rules/{rule_id}")
async def delete_rule(
    rule_id: str,
    user_id: CurrentUserId,
    _config: Config = Depends(get_config),
) -> dict:
    """Delete a rule.

    Args:
        rule_id: Rule ID to delete
        user_id: Authenticated user ID
        config: Injected application config

    Returns:
        Deletion confirmation

    Raises:
        404: Rule not found
    """
    try:
        from constat.storage.learnings import LearningStore

        store = LearningStore(user_id=user_id)
        if store.delete_rule(rule_id):
            return {"status": "deleted", "id": rule_id}
        raise HTTPException(status_code=404, detail=f"Rule not found: {rule_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))