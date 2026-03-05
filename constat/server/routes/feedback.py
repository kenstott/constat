# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Feedback endpoints — answer flagging and glossary suggestions."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from constat.server.auth import CurrentUserId, CurrentUserEmail
from constat.server.persona_config import (
    PersonasConfig,
    require_feedback,
    require_write,
)
from constat.server.session_manager import SessionManager
from constat.storage.learnings import LearningCategory, LearningSource, LearningStore

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class FlagRequest(BaseModel):
    query_text: str
    answer_summary: str
    message: str
    glossary_term: str | None = None
    suggested_definition: str | None = None


class FlagResponse(BaseModel):
    learning_id: str
    glossary_suggestion_id: str | None = None


class GlossarySuggestion(BaseModel):
    learning_id: str
    term: str
    suggested_definition: str
    message: str
    created: str
    user_id: str


class GlossarySuggestionsResponse(BaseModel):
    suggestions: list[GlossarySuggestion]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


def _get_learning_store(user_id: str) -> LearningStore:
    return LearningStore(user_id=user_id)


# ---------------------------------------------------------------------------
# Flag answer
# ---------------------------------------------------------------------------


@router.post(
    "/{session_id}/feedback/flag",
    response_model=FlagResponse,
    dependencies=[Depends(require_feedback("flag_answers"))],
)
async def flag_answer(
    session_id: str,
    body: FlagRequest,
    request: Request,
    user_id: CurrentUserId,
    email: CurrentUserEmail,
    sm: SessionManager = Depends(_get_session_manager),
) -> FlagResponse:
    """Flag an answer as incorrect and optionally suggest a glossary correction."""
    # Validate session exists
    managed = sm.get_session(session_id)

    store = _get_learning_store(user_id)

    # Save the correction learning
    learning_id = store.save_learning(
        category=LearningCategory.USER_CORRECTION,
        context={
            "query_text": body.query_text,
            "answer_summary": body.answer_summary,
            "session_id": session_id,
        },
        correction=body.message,
        source=LearningSource.EXPLICIT_COMMAND,
    )

    glossary_suggestion_id: str | None = None

    # Handle glossary suggestion if provided
    if body.glossary_term and body.suggested_definition:
        # Check if persona has auto_approve — apply directly
        personas_config: PersonasConfig | None = getattr(
            request.app.state, "personas_config", None
        )
        server_config = request.app.state.server_config
        auto_approve = False

        if personas_config and not server_config.auth_disabled:
            from constat.server.permissions import get_user_permissions

            perms = get_user_permissions(
                server_config, user_id=user_id, email=email or ""
            )
            auto_approve = personas_config.can_feedback(perms.persona, "auto_approve")

        if auto_approve:
            # SME: directly update glossary
            try:
                vs = managed.session.doc_tools._vector_store
                vs.relational.update_glossary_term(
                    name=body.glossary_term,
                    session_id=session_id,
                    updates={"definition": body.suggested_definition},
                    user_id=user_id,
                )
                logger.info(
                    f"Auto-approved glossary update: {body.glossary_term} by {user_id}"
                )
            except Exception as e:
                logger.warning(f"Failed to auto-approve glossary update: {e}")
        else:
            # Save as pending glossary suggestion
            glossary_suggestion_id = store.save_learning(
                category=LearningCategory.GLOSSARY_REFINEMENT,
                context={
                    "term": body.glossary_term,
                    "suggested_definition": body.suggested_definition,
                    "status": "pending",
                    "flagged_by": user_id,
                    "session_id": session_id,
                    "query_text": body.query_text,
                },
                correction=body.message,
                source=LearningSource.EXPLICIT_COMMAND,
            )

    return FlagResponse(
        learning_id=learning_id,
        glossary_suggestion_id=glossary_suggestion_id,
    )


# ---------------------------------------------------------------------------
# Glossary suggestions — list / approve / reject
# ---------------------------------------------------------------------------


@router.get(
    "/{session_id}/feedback/glossary-suggestions",
    response_model=GlossarySuggestionsResponse,
)
async def list_glossary_suggestions(
    session_id: str,
    user_id: CurrentUserId,
    sm: SessionManager = Depends(_get_session_manager),
) -> GlossarySuggestionsResponse:
    """List pending glossary suggestions for this user."""
    sm.get_session(session_id)  # validate session

    store = _get_learning_store(user_id)
    raw = store.list_raw_learnings(
        category=LearningCategory.GLOSSARY_REFINEMENT, limit=100
    )

    suggestions = []
    for item in raw:
        ctx = item.get("context", {})
        if ctx.get("status") != "pending":
            continue
        suggestions.append(
            GlossarySuggestion(
                learning_id=item["id"],
                term=ctx.get("term", ""),
                suggested_definition=ctx.get("suggested_definition", ""),
                message=item.get("correction", ""),
                created=item.get("created", ""),
                user_id=ctx.get("flagged_by", user_id),
            )
        )

    return GlossarySuggestionsResponse(suggestions=suggestions)


@router.post(
    "/{session_id}/feedback/glossary-suggestions/{learning_id}/approve",
    dependencies=[Depends(require_write("glossary"))],
)
async def approve_glossary_suggestion(
    session_id: str,
    learning_id: str,
    user_id: CurrentUserId,
    sm: SessionManager = Depends(_get_session_manager),
) -> dict[str, str]:
    """Approve a pending glossary suggestion — updates the glossary term."""
    managed = sm.get_session(session_id)
    store = _get_learning_store(user_id)

    # Find the learning
    raw = store.list_raw_learnings(
        category=LearningCategory.GLOSSARY_REFINEMENT, limit=200
    )
    target = next((r for r in raw if r["id"] == learning_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    ctx = target.get("context", {})
    if ctx.get("status") != "pending":
        raise HTTPException(status_code=400, detail="Suggestion is not pending")

    term = ctx.get("term", "")
    definition = ctx.get("suggested_definition", "")

    # Update glossary
    try:
        vs = managed.session.doc_tools._vector_store
        vs.relational.update_glossary_term(
            name=term,
            session_id=session_id,
            updates={"definition": definition},
            user_id=user_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update glossary: {e}"
        )

    # Mark as approved
    store.update_learning_context(learning_id, {"status": "approved"})

    return {"status": "approved", "learning_id": learning_id}


@router.post(
    "/{session_id}/feedback/glossary-suggestions/{learning_id}/reject",
    dependencies=[Depends(require_write("glossary"))],
)
async def reject_glossary_suggestion(
    session_id: str,
    learning_id: str,
    user_id: CurrentUserId,
    sm: SessionManager = Depends(_get_session_manager),
) -> dict[str, str]:
    """Reject a pending glossary suggestion."""
    sm.get_session(session_id)  # validate session
    store = _get_learning_store(user_id)

    raw = store.list_raw_learnings(
        category=LearningCategory.GLOSSARY_REFINEMENT, limit=200
    )
    target = next((r for r in raw if r["id"] == learning_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Suggestion not found")

    ctx = target.get("context", {})
    if ctx.get("status") != "pending":
        raise HTTPException(status_code=400, detail="Suggestion is not pending")

    store.update_learning_context(learning_id, {"status": "rejected"})

    return {"status": "rejected", "learning_id": learning_id}
