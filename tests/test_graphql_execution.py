# Copyright (c) 2025 Kenneth Stott
# Canary: d2b75348-e9d4-4a5b-9539-7632494372da
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for GraphQL execution resolvers (Phase 7)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_context(user_id="test-user"):
    from constat.server.graphql.session_context import GraphQLContext

    mock_sm = MagicMock()
    mock_server_config = MagicMock()
    mock_server_config.auth_disabled = True
    mock_server_config.base_url = "http://localhost:3000"
    mock_server_config.data_dir = Path("/tmp/test-graphql-execution")

    ctx = GraphQLContext(
        session_manager=mock_sm,
        server_config=mock_server_config,
        user_id=user_id,
    )
    mock_request = MagicMock()
    ctx.request = mock_request
    return ctx


def _make_managed_session(session_id="test-session", status="idle"):
    from constat.server.models import SessionStatus
    from constat.server.session_manager import ManagedSession

    mock_session = MagicMock()
    mock_session._event_handlers = []
    mock_session.plan = None
    mock_session.session_config = MagicMock()
    mock_session.session_config.require_approval = False
    mock_session.session_config.ask_clarifications = False
    mock_session.datastore = None
    mock_session.schema_manager = None

    managed = MagicMock(spec=ManagedSession)
    managed.session_id = session_id
    managed.user_id = "test-user"
    managed.session = mock_session
    managed.status = SessionStatus(status) if isinstance(status, str) else status
    managed.current_query = None
    managed.execution_id = None
    managed.approval_event = None
    managed.approval_response = None
    managed.clarification_event = None
    managed.clarification_response = None
    managed.event_queue = asyncio.Queue()
    managed._entity_rebuild_event = None
    managed._glossary_generating = False
    managed.active_domains = []
    return managed


# ============================================================================
# Schema stitching tests
# ============================================================================


class TestExecutionSchemaStitching:
    """Verify all execution operations appear in the SDL."""

    def _get_sdl(self):
        from constat.server.graphql import schema
        return schema.as_str()

    # Mutations
    def test_submit_query_mutation(self):
        assert "submitQuery(" in self._get_sdl()

    def test_cancel_execution_mutation(self):
        assert "cancelExecution(" in self._get_sdl()

    def test_approve_plan_mutation(self):
        assert "approvePlan(" in self._get_sdl()

    def test_answer_clarification_mutation(self):
        assert "answerClarification(" in self._get_sdl()

    def test_skip_clarification_mutation(self):
        assert "skipClarification(" in self._get_sdl()

    def test_replan_from_mutation(self):
        assert "replanFrom(" in self._get_sdl()

    def test_edit_objective_mutation(self):
        assert "editObjective(" in self._get_sdl()

    def test_delete_objective_mutation(self):
        assert "deleteObjective(" in self._get_sdl()

    def test_request_autocomplete_mutation(self):
        assert "requestAutocomplete(" in self._get_sdl()

    def test_heartbeat_mutation(self):
        assert "heartbeat(" in self._get_sdl()

    # Query
    def test_execution_plan_query(self):
        assert "executionPlan(" in self._get_sdl()

    # Subscription
    def test_query_execution_subscription(self):
        assert "queryExecution(" in self._get_sdl()


# ============================================================================
# Type tests
# ============================================================================


class TestExecutionTypes:
    def _get_sdl(self):
        from constat.server.graphql import schema
        return schema.as_str()

    def test_execution_event_type_fields(self):
        sdl = self._get_sdl()
        assert "ExecutionEventType" in sdl
        assert "eventType" in sdl
        assert "sessionId" in sdl
        assert "stepNumber" in sdl
        assert "timestamp" in sdl

    def test_query_submission_type_fields(self):
        sdl = self._get_sdl()
        assert "QuerySubmissionType" in sdl
        assert "executionId" in sdl
        assert "status" in sdl

    def test_execution_plan_type_fields(self):
        sdl = self._get_sdl()
        assert "ExecutionPlanType" in sdl
        assert "problem" in sdl
        assert "steps" in sdl
        assert "isComplete" in sdl

    def test_autocomplete_result_type_fields(self):
        sdl = self._get_sdl()
        assert "AutocompleteResultType" in sdl
        assert "requestId" in sdl
        assert "items" in sdl


# ============================================================================
# Resolver unit tests
# ============================================================================


class TestExecutionPlanQuery:
    @pytest.mark.asyncio
    async def test_execution_plan_not_found(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = None

        result = await schema.execute(
            '{ executionPlan(sessionId: "nonexistent") { problem isComplete } }',
            context_value=ctx,
        )
        assert result.errors is not None
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_execution_plan_no_plan(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        managed = _make_managed_session()
        managed.session.plan = None
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ executionPlan(sessionId: "test-session") { problem isComplete } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_execution_plan_with_plan(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        managed = _make_managed_session()

        mock_step = MagicMock()
        mock_step.number = 1
        mock_step.goal = "Compute total sales"
        mock_step.status = MagicMock()
        mock_step.status.value = "completed"
        mock_step.expected_inputs = []
        mock_step.expected_outputs = ["total_sales"]
        mock_step.depends_on = []
        mock_step.code = "SELECT SUM(amount) FROM sales"
        mock_step.domain = "sales"
        mock_step.result = None

        mock_plan = MagicMock()
        mock_plan.problem = "What are total sales?"
        mock_plan.steps = [mock_step]
        mock_plan.current_step = 1
        mock_plan.completed_steps = [1]
        mock_plan.failed_steps = []
        mock_plan.is_complete = True

        managed.session.plan = mock_plan
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ executionPlan(sessionId: "test-session") { problem isComplete completedSteps } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["executionPlan"]["problem"] == "What are total sales?"
        assert result.data["executionPlan"]["isComplete"] is True
        assert result.data["executionPlan"]["completedSteps"] == [1]


class TestCancelExecution:
    @pytest.mark.asyncio
    async def test_cancel_execution_not_found(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = None

        result = await schema.execute(
            'mutation { cancelExecution(sessionId: "nonexistent") { status } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_cancel_execution_sets_flag(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        managed = _make_managed_session()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { cancelExecution(sessionId: "test-session") { status message } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["cancelExecution"]["status"] == "cancelling"


class TestApprovePlan:
    @pytest.mark.asyncio
    async def test_approve_plan_no_event(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        managed = _make_managed_session()
        managed.approval_event = None
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { approvePlan(sessionId: "test-session", input: { approved: true }) { status } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_approve_plan_sets_response(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        managed = _make_managed_session()
        managed.approval_event = asyncio.Event()
        managed.approval_response = None
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { approvePlan(sessionId: "test-session", input: { approved: true }) { status message } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["approvePlan"]["status"] == "approved"
        assert managed.approval_response["approved"] is True

    @pytest.mark.asyncio
    async def test_reject_plan(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        managed = _make_managed_session()
        managed.approval_event = asyncio.Event()
        managed.approval_response = None
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { approvePlan(sessionId: "test-session", input: { approved: false, feedback: "Not good" }) { status } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["approvePlan"]["status"] == "rejected"
        assert managed.approval_response["approved"] is False


class TestSkipClarification:
    @pytest.mark.asyncio
    async def test_skip_clarification_not_found(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = None

        result = await schema.execute(
            'mutation { skipClarification(sessionId: "nonexistent") { status } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_skip_clarification_sets_response(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        managed = _make_managed_session()
        managed.clarification_event = asyncio.Event()
        managed.clarification_response = None
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { skipClarification(sessionId: "test-session") { status } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["skipClarification"]["status"] == "ok"
        assert managed.clarification_response["skip"] is True


class TestRequestAutocomplete:
    @pytest.mark.asyncio
    async def test_autocomplete_not_found(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = None

        result = await schema.execute(
            'mutation { requestAutocomplete(sessionId: "nonexistent", context: "table", prefix: "") { requestId items { label value } } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_autocomplete_table_context(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        managed = _make_managed_session()
        mock_datastore = MagicMock()
        mock_datastore.list_tables.return_value = [
            {"name": "sales_data", "description": "Sales", "row_count": 100},
            {"name": "hr_data", "description": "HR", "row_count": 50},
        ]
        managed.session.datastore = mock_datastore
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { requestAutocomplete(sessionId: "test-session", context: "table", prefix: "sales") { requestId items { label value } } }',
            context_value=ctx,
        )
        assert result.errors is None
        items = result.data["requestAutocomplete"]["items"]
        assert len(items) == 1
        assert items[0]["label"] == "sales_data"

    @pytest.mark.asyncio
    async def test_autocomplete_empty_prefix(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        managed = _make_managed_session()
        mock_datastore = MagicMock()
        mock_datastore.list_tables.return_value = [
            {"name": "sales_data", "row_count": 10},
            {"name": "hr_data", "row_count": 5},
        ]
        managed.session.datastore = mock_datastore
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { requestAutocomplete(sessionId: "test-session", context: "table", prefix: "") { requestId items { label } } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert len(result.data["requestAutocomplete"]["items"]) == 2


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.process_heartbeat.return_value = "2026-03-30T00:00:00Z"

        result = await schema.execute(
            'mutation { heartbeat(sessionId: "test-session") { status message } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["heartbeat"]["status"] == "ok"
        assert result.data["heartbeat"]["message"] == "2026-03-30T00:00:00Z"


# ============================================================================
# Pub/sub tests
# ============================================================================


class TestExecutionPubSub:
    def test_subscribe_execution_creates_queue(self):
        from constat.server.session_manager import SessionManager

        sm = MagicMock(spec=SessionManager)
        sm._execution_subscribers = {}

        # Use real method
        SessionManager.subscribe_execution(sm, "session-1")
        assert "session-1" in sm._execution_subscribers
        assert len(sm._execution_subscribers["session-1"]) == 1

    def test_unsubscribe_execution_removes_queue(self):
        from constat.server.session_manager import SessionManager

        sm = MagicMock(spec=SessionManager)
        sm._execution_subscribers = {}

        queue = SessionManager.subscribe_execution(sm, "session-1")
        assert len(sm._execution_subscribers["session-1"]) == 1

        SessionManager.unsubscribe_execution(sm, "session-1", queue)
        assert len(sm._execution_subscribers["session-1"]) == 0

    def test_publish_execution_event_delivers_to_queues(self):
        from constat.server.session_manager import SessionManager

        sm = MagicMock(spec=SessionManager)
        sm._execution_subscribers = {}

        queue1 = SessionManager.subscribe_execution(sm, "session-1")
        queue2 = SessionManager.subscribe_execution(sm, "session-1")

        event = {"event_type": "step_start", "session_id": "session-1", "step_number": 1, "data": {}}
        SessionManager.publish_execution_event(sm, "session-1", event)

        assert not queue1.empty()
        assert not queue2.empty()
        assert queue1.get_nowait() == event
        assert queue2.get_nowait() == event

    def test_publish_execution_event_no_subscribers(self):
        from constat.server.session_manager import SessionManager

        sm = MagicMock(spec=SessionManager)
        sm._execution_subscribers = {}

        # Should not raise
        SessionManager.publish_execution_event(sm, "no-such-session", {"event_type": "test"})

    def test_publish_after_unsubscribe_not_received(self):
        from constat.server.session_manager import SessionManager

        sm = MagicMock(spec=SessionManager)
        sm._execution_subscribers = {}

        queue = SessionManager.subscribe_execution(sm, "session-1")
        SessionManager.unsubscribe_execution(sm, "session-1", queue)

        SessionManager.publish_execution_event(sm, "session-1", {"event_type": "test"})
        assert queue.empty()
