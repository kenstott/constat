# Copyright (c) 2025 Kenneth Stott
# Canary: 5045755d-efab-4745-b0c5-38344f9e8b51
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for GraphQL feedback resolvers (Phase 9)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_context(user_id="test-user"):
    from constat.server.graphql.session_context import GraphQLContext

    mock_sm = MagicMock()
    mock_server_config = MagicMock()
    mock_server_config.auth_disabled = True
    mock_server_config.data_dir = Path("/tmp/test-graphql-feedback")

    ctx = GraphQLContext(
        session_manager=mock_sm,
        server_config=mock_server_config,
        user_id=user_id,
        config=None,
    )
    mock_request = MagicMock()
    mock_request.app.state.personas_config = None
    ctx.request = mock_request
    return ctx


# ============================================================================
# Schema stitching tests
# ============================================================================


class TestFeedbackSchemaStitching:
    def _get_sdl(self):
        from constat.server.graphql import schema
        return schema.as_str()

    def test_glossary_suggestions_query(self):
        assert "glossarySuggestions" in self._get_sdl()

    def test_flag_answer_mutation(self):
        assert "flagAnswer" in self._get_sdl()

    def test_approve_glossary_suggestion_mutation(self):
        assert "approveGlossarySuggestion" in self._get_sdl()

    def test_reject_glossary_suggestion_mutation(self):
        assert "rejectGlossarySuggestion" in self._get_sdl()

    def test_flag_answer_result_type(self):
        assert "FlagAnswerResultType" in self._get_sdl()

    def test_glossary_suggestion_type(self):
        assert "GlossarySuggestionType" in self._get_sdl()

    def test_suggestion_action_result_type(self):
        assert "SuggestionActionResultType" in self._get_sdl()


# ============================================================================
# Query resolver tests
# ============================================================================


class TestGlossarySuggestionsQuery:
    @pytest.mark.asyncio
    async def test_glossary_suggestions_returns_pending(self):
        from constat.server.graphql.feedback_resolvers import Query

        ctx = _make_context()
        mock_managed = MagicMock()
        ctx.session_manager.get_session.return_value = mock_managed

        mock_store = MagicMock()
        mock_store.list_raw_learnings.return_value = [
            {
                "id": "l1",
                "context": {
                    "term": "Revenue",
                    "suggested_definition": "Total sales",
                    "status": "pending",
                    "flagged_by": "user1",
                },
                "correction": "Wrong definition",
                "created": "2026-01-01",
            }
        ]

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx
            result = await Query().glossary_suggestions(info, session_id="sess1")

        assert len(result) == 1
        assert result[0].term == "Revenue"
        assert result[0].learning_id == "l1"

    @pytest.mark.asyncio
    async def test_glossary_suggestions_filters_non_pending(self):
        from constat.server.graphql.feedback_resolvers import Query

        ctx = _make_context()
        mock_managed = MagicMock()
        ctx.session_manager.get_session.return_value = mock_managed

        mock_store = MagicMock()
        mock_store.list_raw_learnings.return_value = [
            {
                "id": "l1",
                "context": {"term": "A", "status": "approved"},
                "correction": "msg",
                "created": "2026-01-01",
            },
            {
                "id": "l2",
                "context": {"term": "B", "status": "pending"},
                "correction": "msg",
                "created": "2026-01-01",
            },
        ]

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx
            result = await Query().glossary_suggestions(info, session_id="sess1")

        assert len(result) == 1
        assert result[0].learning_id == "l2"

    @pytest.mark.asyncio
    async def test_glossary_suggestions_requires_auth(self):
        from constat.server.graphql.feedback_resolvers import Query

        ctx = _make_context(user_id=None)
        info = MagicMock()
        info.context = ctx

        with pytest.raises((ValueError, Exception)):
            await Query().glossary_suggestions(info, session_id="sess1")


# ============================================================================
# Mutation resolver tests
# ============================================================================


class TestFlagAnswerMutation:
    @pytest.mark.asyncio
    async def test_flag_answer_basic(self):
        from constat.server.graphql.feedback_resolvers import Mutation
        from constat.server.graphql.types import FlagAnswerInput

        ctx = _make_context()
        mock_managed = MagicMock()
        ctx.session_manager.get_session.return_value = mock_managed

        mock_store = MagicMock()
        mock_store.save_learning.return_value = "learning-id-1"

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx
            inp = FlagAnswerInput(
                session_id="sess1",
                query_text="What is revenue?",
                answer_summary="Revenue is X",
                message="This is wrong",
            )
            result = await Mutation().flag_answer(info, input=inp)

        assert result.learning_id == "learning-id-1"
        assert result.glossary_suggestion_id is None

    @pytest.mark.asyncio
    async def test_flag_answer_with_glossary_suggestion(self):
        from constat.server.graphql.feedback_resolvers import Mutation
        from constat.server.graphql.types import FlagAnswerInput

        ctx = _make_context()
        mock_managed = MagicMock()
        ctx.session_manager.get_session.return_value = mock_managed

        mock_store = MagicMock()
        mock_store.save_learning.side_effect = ["learning-id-1", "gloss-sugg-id-1"]

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx
            inp = FlagAnswerInput(
                session_id="sess1",
                query_text="What is revenue?",
                answer_summary="Revenue is X",
                message="This is wrong",
                glossary_term="Revenue",
                suggested_definition="Total sales",
            )
            result = await Mutation().flag_answer(info, input=inp)

        assert result.learning_id == "learning-id-1"
        assert result.glossary_suggestion_id == "gloss-sugg-id-1"
        assert mock_store.save_learning.call_count == 2

    @pytest.mark.asyncio
    async def test_flag_answer_requires_auth(self):
        from constat.server.graphql.feedback_resolvers import Mutation
        from constat.server.graphql.types import FlagAnswerInput

        ctx = _make_context(user_id=None)
        info = MagicMock()
        info.context = ctx

        inp = FlagAnswerInput(
            session_id="sess1",
            query_text="q",
            answer_summary="a",
            message="m",
        )
        with pytest.raises((ValueError, Exception)):
            await Mutation().flag_answer(info, input=inp)


class TestApproveGlossarySuggestionMutation:
    @pytest.mark.asyncio
    async def test_approve_suggestion_success(self):
        from constat.server.graphql.feedback_resolvers import Mutation

        ctx = _make_context()
        mock_managed = MagicMock()
        ctx.session_manager.get_session.return_value = mock_managed

        mock_store = MagicMock()
        mock_store.list_raw_learnings.return_value = [
            {
                "id": "l1",
                "context": {
                    "term": "Revenue",
                    "suggested_definition": "Total sales",
                    "status": "pending",
                },
            }
        ]

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx
            result = await Mutation().approve_glossary_suggestion(
                info, session_id="sess1", learning_id="l1"
            )

        assert result.status == "approved"
        assert result.learning_id == "l1"
        mock_store.update_learning_context.assert_called_once_with("l1", {"status": "approved"})

    @pytest.mark.asyncio
    async def test_approve_suggestion_not_found(self):
        from constat.server.graphql.feedback_resolvers import Mutation

        ctx = _make_context()
        mock_managed = MagicMock()
        ctx.session_manager.get_session.return_value = mock_managed

        mock_store = MagicMock()
        mock_store.list_raw_learnings.return_value = []

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx

            with pytest.raises(ValueError, match="not found"):
                await Mutation().approve_glossary_suggestion(
                    info, session_id="sess1", learning_id="bad-id"
                )

    @pytest.mark.asyncio
    async def test_approve_suggestion_not_pending(self):
        from constat.server.graphql.feedback_resolvers import Mutation

        ctx = _make_context()
        mock_managed = MagicMock()
        ctx.session_manager.get_session.return_value = mock_managed

        mock_store = MagicMock()
        mock_store.list_raw_learnings.return_value = [
            {
                "id": "l1",
                "context": {"term": "Revenue", "status": "approved"},
            }
        ]

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx

            with pytest.raises(ValueError, match="not pending"):
                await Mutation().approve_glossary_suggestion(
                    info, session_id="sess1", learning_id="l1"
                )

    @pytest.mark.asyncio
    async def test_approve_suggestion_requires_auth(self):
        from constat.server.graphql.feedback_resolvers import Mutation

        ctx = _make_context(user_id=None)
        info = MagicMock()
        info.context = ctx

        with pytest.raises((ValueError, Exception)):
            await Mutation().approve_glossary_suggestion(
                info, session_id="sess1", learning_id="l1"
            )


class TestRejectGlossarySuggestionMutation:
    @pytest.mark.asyncio
    async def test_reject_suggestion_success(self):
        from constat.server.graphql.feedback_resolvers import Mutation

        ctx = _make_context()
        mock_managed = MagicMock()
        ctx.session_manager.get_session.return_value = mock_managed

        mock_store = MagicMock()
        mock_store.list_raw_learnings.return_value = [
            {
                "id": "l1",
                "context": {"term": "Revenue", "status": "pending"},
            }
        ]

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx
            result = await Mutation().reject_glossary_suggestion(
                info, session_id="sess1", learning_id="l1"
            )

        assert result.status == "rejected"
        assert result.learning_id == "l1"
        mock_store.update_learning_context.assert_called_once_with("l1", {"status": "rejected"})

    @pytest.mark.asyncio
    async def test_reject_suggestion_not_found(self):
        from constat.server.graphql.feedback_resolvers import Mutation

        ctx = _make_context()
        mock_managed = MagicMock()
        ctx.session_manager.get_session.return_value = mock_managed

        mock_store = MagicMock()
        mock_store.list_raw_learnings.return_value = []

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx

            with pytest.raises(ValueError, match="not found"):
                await Mutation().reject_glossary_suggestion(
                    info, session_id="sess1", learning_id="bad-id"
                )

    @pytest.mark.asyncio
    async def test_reject_suggestion_requires_auth(self):
        from constat.server.graphql.feedback_resolvers import Mutation

        ctx = _make_context(user_id=None)
        info = MagicMock()
        info.context = ctx

        with pytest.raises((ValueError, Exception)):
            await Mutation().reject_glossary_suggestion(
                info, session_id="sess1", learning_id="l1"
            )
