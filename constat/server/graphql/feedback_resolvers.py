# Copyright (c) 2025 Kenneth Stott
# Canary: d5b03166-1de3-42a5-9231-6812652cc564
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for feedback, flagging, and glossary suggestions (Phase 9)."""

from __future__ import annotations

import logging

import strawberry

from constat.server.graphql.session_context import GqlInfo as Info
from constat.server.graphql.types import (
    FlagAnswerInput,
    FlagAnswerResultType,
    GlossarySuggestionType,
    SuggestionActionResultType,
)

logger = logging.getLogger(__name__)


def _require_auth(info: Info) -> str:
    user_id = info.context.user_id
    if not user_id:
        raise ValueError("Authentication required")
    return user_id


def _get_learning_store(user_id: str):
    from constat.storage.learnings import LearningStore

    return LearningStore(user_id=user_id)


@strawberry.type
class Query:
    @strawberry.field
    async def glossary_suggestions(
        self, info: Info, session_id: str
    ) -> list[GlossarySuggestionType]:
        user_id = _require_auth(info)
        session_manager = info.context.session_manager
        session_manager.get_session(session_id)  # validate session

        from constat.storage.learnings import LearningCategory

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
                GlossarySuggestionType(
                    learning_id=item["id"],
                    term=ctx.get("term", ""),
                    suggested_definition=ctx.get("suggested_definition", ""),
                    message=item.get("correction", ""),
                    created=item.get("created", ""),
                    user_id=ctx.get("flagged_by", user_id),
                )
            )

        return suggestions


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def flag_answer(
        self, info: Info, input: FlagAnswerInput
    ) -> FlagAnswerResultType:
        user_id = _require_auth(info)
        session_manager = info.context.session_manager
        managed = session_manager.get_session(input.session_id)

        from constat.storage.learnings import (
            LearningCategory,
            LearningSource,
            LearningStore,
        )

        store = LearningStore(user_id=user_id)

        learning_id = store.save_learning(
            category=LearningCategory.USER_CORRECTION,
            context={
                "query_text": input.query_text,
                "answer_summary": input.answer_summary,
                "session_id": input.session_id,
            },
            correction=input.message,
            source=LearningSource.EXPLICIT_COMMAND,
        )

        glossary_suggestion_id: str | None = None

        if input.glossary_term and input.suggested_definition:
            server_config = info.context.server_config
            personas_config = getattr(
                info.context.request.app.state, "personas_config", None
            )
            auto_approve = False

            if personas_config and not server_config.auth_disabled:
                from constat.server.permissions import get_user_permissions

                perms = get_user_permissions(
                    server_config, user_id=user_id, email=""
                )
                auto_approve = personas_config.can_feedback(perms.persona, "auto_approve")

            if auto_approve:
                try:
                    vs = managed.session.doc_tools._vector_store
                    vs.relational.update_glossary_term(
                        name=input.glossary_term,
                        session_id=input.session_id,
                        updates={"definition": input.suggested_definition},
                        user_id=user_id,
                    )
                    logger.info(
                        f"Auto-approved glossary update: {input.glossary_term} by {user_id}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to auto-approve glossary update: {e}")
            else:
                glossary_suggestion_id = store.save_learning(
                    category=LearningCategory.GLOSSARY_REFINEMENT,
                    context={
                        "term": input.glossary_term,
                        "suggested_definition": input.suggested_definition,
                        "status": "pending",
                        "flagged_by": user_id,
                        "session_id": input.session_id,
                        "query_text": input.query_text,
                    },
                    correction=input.message,
                    source=LearningSource.EXPLICIT_COMMAND,
                )

        return FlagAnswerResultType(
            learning_id=learning_id,
            glossary_suggestion_id=glossary_suggestion_id,
        )

    @strawberry.mutation
    async def approve_glossary_suggestion(
        self, info: Info, session_id: str, learning_id: str
    ) -> SuggestionActionResultType:
        user_id = _require_auth(info)
        session_manager = info.context.session_manager
        managed = session_manager.get_session(session_id)

        from constat.storage.learnings import LearningCategory, LearningStore

        store = LearningStore(user_id=user_id)
        raw = store.list_raw_learnings(
            category=LearningCategory.GLOSSARY_REFINEMENT, limit=200
        )
        target = next((r for r in raw if r["id"] == learning_id), None)
        if not target:
            raise ValueError("Suggestion not found")

        ctx = target.get("context", {})
        if ctx.get("status") != "pending":
            raise ValueError("Suggestion is not pending")

        term = ctx.get("term", "")
        definition = ctx.get("suggested_definition", "")

        try:
            vs = managed.session.doc_tools._vector_store
            vs.relational.update_glossary_term(
                name=term,
                session_id=session_id,
                updates={"definition": definition},
                user_id=user_id,
            )
        except Exception as e:
            raise ValueError(f"Failed to update glossary: {e}")

        store.update_learning_context(learning_id, {"status": "approved"})

        return SuggestionActionResultType(status="approved", learning_id=learning_id)

    @strawberry.mutation
    async def reject_glossary_suggestion(
        self, info: Info, session_id: str, learning_id: str
    ) -> SuggestionActionResultType:
        user_id = _require_auth(info)
        session_manager = info.context.session_manager
        session_manager.get_session(session_id)  # validate session

        from constat.storage.learnings import LearningCategory, LearningStore

        store = LearningStore(user_id=user_id)
        raw = store.list_raw_learnings(
            category=LearningCategory.GLOSSARY_REFINEMENT, limit=200
        )
        target = next((r for r in raw if r["id"] == learning_id), None)
        if not target:
            raise ValueError("Suggestion not found")

        ctx = target.get("context", {})
        if ctx.get("status") != "pending":
            raise ValueError("Suggestion is not pending")

        store.update_learning_context(learning_id, {"status": "rejected"})

        return SuggestionActionResultType(status="rejected", learning_id=learning_id)
