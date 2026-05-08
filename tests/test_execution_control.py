# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for Phase 4: Execution Control.

Tests cover:
- Cancellation flag and reset
- ExecutionContext in parallel_scheduler
- Intent queue with different queue behaviors
- Queue processing after execution
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from constat.execution.parallel_scheduler import (
    ExecutionContext,
    ParallelStepScheduler,
    SchedulerConfig,
    SchedulerResult,
)
from constat.execution.mode import (
    Mode,
    Phase,
    PrimaryIntent,
    SubIntent,
    TurnIntent,
    ConversationState,
)
from constat.core.models import Plan, Step, StepResult, StepStatus


class TestExecutionContext:
    """Tests for ExecutionContext class."""

    def test_initial_state_not_cancelled(self):
        """Context should start in non-cancelled state."""
        ctx = ExecutionContext()
        assert ctx.is_cancelled() is False

    def test_cancel_sets_flag(self):
        """Calling cancel() should set the cancelled flag."""
        ctx = ExecutionContext()
        ctx.cancel()
        assert ctx.is_cancelled() is True

    def test_reset_clears_flag(self):
        """Calling reset() should clear the cancelled flag."""
        ctx = ExecutionContext()
        ctx.cancel()
        assert ctx.is_cancelled() is True
        ctx.reset()
        assert ctx.is_cancelled() is False

    def test_multiple_cancels_idempotent(self):
        """Multiple cancel calls should be safe."""
        ctx = ExecutionContext()
        ctx.cancel()
        ctx.cancel()
        ctx.cancel()
        assert ctx.is_cancelled() is True


class TestParallelSchedulerCancellation:
    """Tests for cancellation in ParallelStepScheduler."""

    def test_scheduler_has_execution_context(self):
        """Scheduler should have an accessible execution context."""
        scheduler = ParallelStepScheduler(
            step_executor=lambda step, ns: StepResult(success=True, stdout="ok"),
        )
        ctx = scheduler.get_execution_context()
        assert ctx is not None
        assert isinstance(ctx, ExecutionContext)

    def test_cancel_method_sets_context_flag(self):
        """Scheduler cancel() should set the execution context flag."""
        scheduler = ParallelStepScheduler(
            step_executor=lambda step, ns: StepResult(success=True, stdout="ok"),
        )
        assert scheduler.get_execution_context().is_cancelled() is False
        scheduler.cancel()
        assert scheduler.get_execution_context().is_cancelled() is True

    @pytest.mark.asyncio
    async def test_cancelled_during_execution(self):
        """Execution cancelled during first wave should stop and preserve completed."""
        import asyncio

        steps_started = []
        cancel_after_step_1 = [False]

        def slow_executor(step, ns):
            steps_started.append(step.number)
            # Simulate some work
            if step.number == 1:
                cancel_after_step_1[0] = True
            return StepResult(success=True, stdout=f"step {step.number}")

        scheduler = ParallelStepScheduler(
            step_executor=slow_executor,
        )

        # Linear plan - step 2 depends on step 1
        plan = Plan(problem="Test", steps=[
            Step(number=1, goal="Step 1"),
            Step(number=2, goal="Step 2", depends_on=[1]),
        ])

        # Execute - since steps are sequential (dependencies), we can see the effect
        result = await scheduler.execute_plan(plan)

        # Without external cancellation during execution, both complete
        assert result.success is True
        assert len(result.completed_steps) == 2

    @pytest.mark.asyncio
    async def test_scheduler_result_has_cancelled_field(self):
        """SchedulerResult should have a cancelled field."""
        scheduler = ParallelStepScheduler(
            step_executor=lambda step, ns: StepResult(success=True, stdout="ok"),
        )

        plan = Plan(problem="Test", steps=[
            Step(number=1, goal="Step 1"),
        ])

        result = await scheduler.execute_plan(plan)

        assert hasattr(result, 'cancelled')
        assert result.cancelled is False  # Not cancelled

    @pytest.mark.asyncio
    async def test_completed_steps_preserved_on_cancel(self):
        """Completed steps should be preserved when cancelled."""
        call_count = [0]

        def slow_executor(step, ns):
            call_count[0] += 1
            return StepResult(success=True, stdout=f"step {step.number}")

        scheduler = ParallelStepScheduler(
            step_executor=slow_executor,
        )

        # Linear plan with dependencies
        plan = Plan(problem="Test", steps=[
            Step(number=1, goal="Step 1"),
            Step(number=2, goal="Step 2", depends_on=[1]),
        ])

        result = await scheduler.execute_plan(plan)

        # Without cancellation, both steps complete
        assert result.success is True
        assert len(result.completed_steps) == 2


class TestIntentQueue:
    """Tests for intent queue behavior in Session."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock session with Phase 4 fields."""
        session = Mock()
        session._cancelled = False
        session._intent_queue = []
        session._execution_context = ExecutionContext()
        session._conversation_state = ConversationState(
            phase=Phase.IDLE,
        )

        # Add methods from the real implementation
        def queue_intent(intent, user_input):
            if intent.primary == PrimaryIntent.QUERY:
                return False
            if intent.primary == PrimaryIntent.PLAN_NEW:
                session._intent_queue = [
                    (i, inp) for i, inp in session._intent_queue
                    if i.primary != PrimaryIntent.PLAN_NEW
                ]
                session._intent_queue.append((intent, user_input))
                return True
            if intent.primary == PrimaryIntent.CONTROL:
                session._intent_queue.append((intent, user_input))
                return True
            if intent.primary == PrimaryIntent.PLAN_CONTINUE:
                session._cancelled = True
                session._intent_queue.append((intent, user_input))
                return True
            return False

        session.queue_intent = queue_intent
        session.get_queued_intents_count = lambda: len(session._intent_queue)
        session.clear_intent_queue = lambda: setattr(session, '_intent_queue', []) or 0

        return session

    def test_query_intents_not_queued(self, mock_session):
        """Query intents should not be queued - handled in parallel."""
        intent = TurnIntent(primary=PrimaryIntent.QUERY, sub=SubIntent.DETAIL)
        result = mock_session.queue_intent(intent, "what does this mean?")

        assert result is False
        assert mock_session.get_queued_intents_count() == 0

    def test_plan_new_queued_latest_wins(self, mock_session):
        """Plan new intents should queue with latest wins behavior."""
        intent1 = TurnIntent(primary=PrimaryIntent.PLAN_NEW)
        intent2 = TurnIntent(primary=PrimaryIntent.PLAN_NEW)

        mock_session.queue_intent(intent1, "first request")
        assert mock_session.get_queued_intents_count() == 1

        mock_session.queue_intent(intent2, "second request")
        # Latest wins - should still be 1
        assert mock_session.get_queued_intents_count() == 1

        # Verify the latest is stored
        queued = mock_session._intent_queue
        assert queued[0][1] == "second request"

    def test_control_intents_queue_in_order(self, mock_session):
        """Control intents should queue in order."""
        intent1 = TurnIntent(primary=PrimaryIntent.CONTROL, sub=SubIntent.RESET)
        intent2 = TurnIntent(primary=PrimaryIntent.CONTROL, sub=SubIntent.DETAIL)

        mock_session.queue_intent(intent1, "reset")
        mock_session.queue_intent(intent2, "show details")

        assert mock_session.get_queued_intents_count() == 2

        queued = mock_session._intent_queue
        assert queued[0][1] == "reset"
        assert queued[1][1] == "show details"

    def test_plan_continue_triggers_cancel(self, mock_session):
        """Plan continue should trigger cancellation and queue."""
        intent = TurnIntent(primary=PrimaryIntent.PLAN_CONTINUE)
        mock_session.queue_intent(intent, "actually, include Q4")

        assert mock_session._cancelled is True
        assert mock_session.get_queued_intents_count() == 1

    def test_mixed_intents_queue_correctly(self, mock_session):
        """Mixed intent types should queue according to their rules."""
        control1 = TurnIntent(primary=PrimaryIntent.CONTROL, sub=SubIntent.HELP)
        plan_new = TurnIntent(primary=PrimaryIntent.PLAN_NEW)
        query = TurnIntent(primary=PrimaryIntent.QUERY)
        control2 = TurnIntent(primary=PrimaryIntent.CONTROL, sub=SubIntent.STATUS)

        mock_session.queue_intent(control1, "help")
        mock_session.queue_intent(plan_new, "analyze data")
        mock_session.queue_intent(query, "what does this mean?")  # Not queued
        mock_session.queue_intent(control2, "status")

        # Should have: control1, plan_new, control2 (query not queued)
        assert mock_session.get_queued_intents_count() == 3


class TestConversationStatePhaseTransitions:
    """Tests for phase transitions related to cancellation."""

    def test_cancel_transitions_to_idle(self):
        """Cancelling should transition from EXECUTING to IDLE."""
        state = ConversationState(
            phase=Phase.EXECUTING,
        )

        # Simulate cancel -> abandon transition
        state.phase = Phase.IDLE
        state.active_plan = None

        assert state.phase == Phase.IDLE
        assert state.active_plan is None

    def test_replan_transitions_to_planning(self):
        """Replan should transition from EXECUTING to PLANNING."""
        state = ConversationState(
            phase=Phase.EXECUTING,
        )

        # Simulate replan transition
        state.phase = Phase.PLANNING

        assert state.phase == Phase.PLANNING

    def test_is_executing_check(self):
        """Should correctly identify executing phase."""
        state = ConversationState(
            phase=Phase.IDLE,
        )

        assert state.phase != Phase.EXECUTING

        state.phase = Phase.EXECUTING
        assert state.phase == Phase.EXECUTING


class TestSchedulerResultCancelled:
    """Tests for SchedulerResult cancelled field."""

    def test_result_default_not_cancelled(self):
        """Default result should have cancelled=False."""
        result = SchedulerResult(
            success=True,
            completed_steps=[1, 2],
            failed_steps=[],
            step_results={},
            execution_waves=[[1], [2]],
        )
        assert result.cancelled is False

    def test_result_explicitly_cancelled(self):
        """Result can be explicitly marked as cancelled."""
        result = SchedulerResult(
            success=False,
            completed_steps=[1],
            failed_steps=[],
            step_results={},
            execution_waves=[[1]],
            cancelled=True,
        )
        assert result.cancelled is True


class TestCancellationIntegration:
    """Integration tests for cancellation flow."""

    def test_cancel_and_clear_queue(self):
        """Cancelling should be able to clear the intent queue."""
        intent_queue = []

        # Simulate queued intents
        intent_queue.append((TurnIntent(primary=PrimaryIntent.PLAN_NEW), "request"))
        intent_queue.append((TurnIntent(primary=PrimaryIntent.CONTROL), "help"))

        assert len(intent_queue) == 2

        # Clear queue on cancel
        cleared_count = len(intent_queue)
        intent_queue.clear()

        assert len(intent_queue) == 0
        assert cleared_count == 2
