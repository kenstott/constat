# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Fact and output endpoints."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from constat.server.models import (
    FactInfo,
    FactListResponse,
)
from constat.server.routes.data import get_session_manager
from constat.server.session_manager import SessionManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{session_id}/facts", response_model=FactListResponse)
async def list_facts(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> FactListResponse:
    """List all resolved facts in the session.

    Facts are values extracted from user queries or resolved during execution.

    Args:
        session_id: Session ID

    Returns:
        List of resolved facts

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    try:
        all_facts = managed.session.fact_resolver.get_all_facts()

        # Get persisted fact names from FactStore
        from constat.storage.facts import FactStore
        fact_store = FactStore(user_id=managed.user_id)
        persisted_fact_names = set(fact_store.list_facts().keys())

        facts_list = []

        # Add config facts first (core facts)
        config_facts = managed.session.config.facts or {}
        for name, value in config_facts.items():
            facts_list.append(FactInfo(
                name=name,
                value=value,
                source="config",
                reasoning=None,
                confidence=1.0,
                is_persisted=False,  # Config facts are always available, not user-persisted
                role_id=None,
            ))

        # Add session facts
        for name, fact in all_facts.items():
            # Skip if already added from config (config takes precedence for display)
            if name in config_facts:
                continue
            facts_list.append(FactInfo(
                name=name,
                value=fact.value,
                source=fact.source.value if hasattr(fact.source, "value") else str(fact.source),
                reasoning=fact.reasoning,
                confidence=getattr(fact, "confidence", None),
                is_persisted=name in persisted_fact_names,
                role_id=getattr(fact, "role_id", None),
            ))

        return FactListResponse(facts=facts_list)
    except Exception as e:
        logger.error(f"Error listing facts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/proof-tree")
async def get_proof_tree(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Get the proof tree for auditable mode execution.

    The proof tree shows how facts were resolved and combined
    to produce the final answer.

    Args:
        session_id: Session ID

    Returns:
        Proof tree structure

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    try:
        # Get all facts with their provenance
        all_facts = managed.session.fact_resolver.get_all_facts()

        # Build proof tree structure
        nodes = []
        for name, fact in all_facts.items():
            node = {
                "name": name,
                "value": fact.value,
                "source": fact.source.value if hasattr(fact.source, "value") else str(fact.source),
                "reasoning": fact.reasoning,
                "dependencies": getattr(fact, "dependencies", []),
            }
            nodes.append(node)

        return {
            "facts": nodes,
            "execution_trace": [],  # Could be populated from datastore logs
        }

    except Exception as e:
        logger.error(f"Error getting proof tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/output")
async def get_output(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Get the final output/answer from the session.

    Args:
        session_id: Session ID

    Returns:
        Final output with any suggestions

    Raises:
        404: Session not found
    """
    managed = session_manager.get_session(session_id)

    # Get output from scratchpad or last execution
    output = ""
    suggestions = []

    if managed.session.scratchpad:
        # Try to get synthesized output from scratchpad
        recent = managed.session.scratchpad.get_recent_context(max_steps=1)
        if recent:
            output = recent

    return {
        "output": output,
        "suggestions": suggestions,
        "current_query": managed.current_query,
    }


# ============================================================================
# Fact Action Endpoints
# ============================================================================


@router.post("/{session_id}/facts")
async def add_fact(
    session_id: str,
    body: dict[str, Any],
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Add a new fact to the session.

    Args:
        session_id: Session ID
        body: Request body with name, value, and optional persist flag

    Returns:
        Created fact

    Raises:
        400: Missing name or value
    """
    managed = session_manager.get_session(session_id)

    if "name" not in body:
        raise HTTPException(status_code=400, detail="Missing 'name' in request body")
    if "value" not in body:
        raise HTTPException(status_code=400, detail="Missing 'value' in request body")

    try:
        fact_name = body["name"]
        fact_value = body["value"]
        persist = body.get("persist", False)

        # Add the fact via fact_resolver
        from constat.execution.fact_resolver import FactSource
        managed.session.fact_resolver.add_user_fact(
            fact_name=fact_name,
            value=fact_value,
            source=FactSource.USER_PROVIDED,
            reasoning="Added via UI",
        )

        # Optionally persist to FactStore
        is_persisted = False
        if persist:
            from constat.storage.facts import FactStore
            fact_store = FactStore(user_id=managed.user_id)
            fact_store.save_fact(
                name=fact_name,
                value=fact_value,
                description="Added via UI",
            )
            is_persisted = True

        return {
            "status": "created",
            "fact": {
                "name": fact_name,
                "value": fact_value,
                "source": FactSource.USER_PROVIDED.value,
                "is_persisted": is_persisted,
            },
        }

    except Exception as e:
        logger.error(f"Error adding fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/facts/{fact_name}/persist")
async def persist_fact(
    session_id: str,
    fact_name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Cache a fact for future use.

    Args:
        session_id: Session ID
        fact_name: Name of the fact to persist

    Returns:
        Confirmation

    Raises:
        404: Session or fact not found
    """
    managed = session_manager.get_session(session_id)

    try:
        all_facts = managed.session.fact_resolver.get_all_facts()
        if fact_name not in all_facts:
            raise HTTPException(status_code=404, detail=f"Fact not found: {fact_name}")

        # Persist the fact
        if hasattr(managed.session.fact_resolver, "persist_fact"):
            managed.session.fact_resolver.persist_fact(fact_name)

        return {"status": "persisted", "fact_name": fact_name}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error persisting fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{session_id}/facts/{fact_name}/forget")
async def forget_fact(
    session_id: str,
    fact_name: str,
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Forget a fact (removes from both session and persistent storage).

    Args:
        session_id: Session ID
        fact_name: Name of the fact to forget

    Returns:
        Confirmation with what was deleted

    Raises:
        404: Session or fact not found
    """
    managed = session_manager.get_session(session_id)

    try:
        all_facts = managed.session.fact_resolver.get_all_facts()
        if fact_name not in all_facts:
            raise HTTPException(status_code=404, detail=f"Fact not found: {fact_name}")

        deleted_persistent = False
        deleted_session = False

        # Delete from persistent storage (facts.yaml) if exists
        from constat.storage.facts import FactStore
        fact_store = FactStore(user_id=managed.user_id)
        if fact_store.delete_fact(fact_name):
            deleted_persistent = True

        # Remove from session cache
        if hasattr(managed.session.fact_resolver, "_cache"):
            if fact_name in managed.session.fact_resolver._cache:
                managed.session.fact_resolver._cache.pop(fact_name, None)
                deleted_session = True

        return {
            "status": "forgotten",
            "fact_name": fact_name,
            "deleted_persistent": deleted_persistent,
            "deleted_session": deleted_session,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forgetting fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{session_id}/facts/{fact_name}")
async def edit_fact(
    session_id: str,
    fact_name: str,
    body: dict[str, Any],
    session_manager: SessionManager = Depends(get_session_manager),
) -> dict[str, Any]:
    """Edit a fact value.

    Args:
        session_id: Session ID
        fact_name: Name of the fact to edit
        body: Request body with new value

    Returns:
        Updated fact

    Raises:
        404: Session or fact not found
        400: Missing value in request
    """
    managed = session_manager.get_session(session_id)

    if "value" not in body:
        raise HTTPException(status_code=400, detail="Missing 'value' in request body")

    try:
        all_facts = managed.session.fact_resolver.get_all_facts()
        if fact_name not in all_facts:
            raise HTTPException(status_code=404, detail=f"Fact not found: {fact_name}")

        # Edit the fact by updating the cache directly
        from constat.execution.fact_resolver import FactSource
        managed.session.fact_resolver.add_user_fact(
            fact_name=fact_name,
            value=body["value"],
            source=FactSource.USER_PROVIDED,
            reasoning="Edited via UI",
        )

        return {
            "status": "updated",
            "fact_name": fact_name,
            "new_value": body["value"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error editing fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))
