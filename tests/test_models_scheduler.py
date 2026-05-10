# Copyright (c) 2025 Kenneth Stott
# Canary: 38d1ca9b-5a69-4d97-adce-b96f15d7e23a
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Component tests for ParallelStepScheduler execution."""

from __future__ import annotations
import pytest
from constat.core.models import Plan, Step, StepResult


class TestParallelStepSchedulerComponent:
    """Component tests for ParallelStepScheduler execution."""

    @pytest.mark.asyncio
    async def test_parallel_execution_timing_proves_concurrency(self):
        """Component: Verify parallel execution by measuring actual timing."""
        import time
        from constat.execution.parallel_scheduler import ParallelStepScheduler, SchedulerConfig

        DELAY_MS = 100
        NUM_PARALLEL_STEPS = 3

        execution_log = []

        def slow_step_executor(step: Step, namespace: dict) -> StepResult:
            """Step executor that takes DELAY_MS to complete."""
            execution_log.append(("start", step.number, time.time()))
            time.sleep(DELAY_MS / 1000)
            execution_log.append(("end", step.number, time.time()))
            return StepResult(
                success=True,
                stdout=f"Step {step.number} done",
                variables={f"result_{step.number}": step.number * 10},
                duration_ms=DELAY_MS,
            )

        plan = Plan(
            problem="Test parallel execution",
            steps=[
                Step(number=1, goal="Task A", depends_on=[]),
                Step(number=2, goal="Task B", depends_on=[]),
                Step(number=3, goal="Task C", depends_on=[]),
                Step(number=4, goal="Aggregate", depends_on=[1, 2, 3]),
            ],
        )

        scheduler = ParallelStepScheduler(
            step_executor=slow_step_executor,
            config=SchedulerConfig(max_concurrent_steps=10),
        )

        start = time.time()
        result = await scheduler.execute_plan(plan)
        total_time = (time.time() - start) * 1000

        assert result.success
        assert len(result.completed_steps) == 4

        assert len(result.execution_waves) == 2
        assert set(result.execution_waves[0]) == {1, 2, 3}
        assert result.execution_waves[1] == [4]

        expected_sequential = NUM_PARALLEL_STEPS * DELAY_MS + DELAY_MS  # 400ms
        expected_parallel = 2 * DELAY_MS + 50  # ~250ms with overhead

        print(f"\nTotal time: {total_time:.0f}ms")
        print(f"Expected sequential: {expected_sequential}ms")
        print(f"Expected parallel: ~{expected_parallel}ms")

        assert total_time < expected_sequential * 0.7, (
            f"Parallel execution took {total_time:.0f}ms, expected <{expected_sequential * 0.7:.0f}ms. "
            f"Steps may not be running in parallel!"
        )

    @pytest.mark.asyncio
    async def test_dependency_ordering_enforced(self):
        """Component: Verify dependent steps wait for prerequisites."""
        import time
        from constat.execution.parallel_scheduler import ParallelStepScheduler

        completion_order = []

        def tracking_executor(step: Step, namespace: dict) -> StepResult:
            """Track when each step completes."""
            time.sleep(0.05)
            completion_order.append(step.number)
            return StepResult(
                success=True,
                stdout=f"Step {step.number}",
                variables={f"out_{step.number}": True},
            )

        plan = Plan(
            problem="Test dependency ordering",
            steps=[
                Step(number=1, goal="First", depends_on=[]),
                Step(number=2, goal="Second", depends_on=[1]),
                Step(number=3, goal="Third", depends_on=[2]),
            ],
        )

        scheduler = ParallelStepScheduler(step_executor=tracking_executor)
        result = await scheduler.execute_plan(plan)

        assert result.success
        assert completion_order == [1, 2, 3]
        assert result.execution_waves == [[1], [2], [3]]

    @pytest.mark.asyncio
    async def test_namespace_flows_between_steps(self):
        """Component: Verify step outputs are available to dependent steps."""
        from constat.execution.parallel_scheduler import ParallelStepScheduler

        received_values = {}

        def namespace_checking_executor(step: Step, namespace: dict) -> StepResult:
            """Check namespace and produce output."""
            received_values[step.number] = dict(namespace)

            if step.number == 1:
                return StepResult(
                    success=True,
                    stdout="Step 1",
                    variables={"value_a": 100},
                )
            elif step.number == 2:
                return StepResult(
                    success=True,
                    stdout="Step 2",
                    variables={"value_b": 200},
                )
            elif step.number == 3:
                total = namespace.get("value_a", 0) + namespace.get("value_b", 0)
                return StepResult(
                    success=True,
                    stdout=f"Total: {total}",
                    variables={"total": total},
                )

            return StepResult(success=True, stdout="")

        plan = Plan(
            problem="Test namespace flow",
            steps=[
                Step(number=1, goal="Produce A", depends_on=[]),
                Step(number=2, goal="Produce B", depends_on=[]),
                Step(number=3, goal="Sum A+B", depends_on=[1, 2]),
            ],
        )

        scheduler = ParallelStepScheduler(step_executor=namespace_checking_executor)
        result = await scheduler.execute_plan(plan)

        assert result.success

        assert "value_a" in received_values[3]
        assert "value_b" in received_values[3]
        assert received_values[3]["value_a"] == 100
        assert received_values[3]["value_b"] == 200

        assert result.step_results[3].variables.get("total") == 300

    @pytest.mark.asyncio
    async def test_fail_fast_stops_execution(self):
        """Component: Verify fail_fast stops on first failure."""
        from constat.execution.parallel_scheduler import ParallelStepScheduler, SchedulerConfig

        executed_steps = []

        def failing_executor(step: Step, namespace: dict) -> StepResult:
            executed_steps.append(step.number)
            if step.number == 2:
                return StepResult(success=False, stdout="", error="Step 2 failed")
            return StepResult(success=True, stdout=f"Step {step.number}")

        plan = Plan(
            problem="Test fail fast",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B (fails)", depends_on=[]),
                Step(number=3, goal="C", depends_on=[1, 2]),
            ],
        )

        scheduler = ParallelStepScheduler(
            step_executor=failing_executor,
            config=SchedulerConfig(fail_fast=True),
        )
        result = await scheduler.execute_plan(plan)

        assert not result.success
        assert 2 in result.failed_steps
        assert 3 not in executed_steps

    @pytest.mark.asyncio
    async def test_employee_count_scenario(self):
        """Component: Test the employee count across companies scenario."""
        import time
        from constat.execution.parallel_scheduler import ParallelStepScheduler

        DELAY_MS = 50
        step_start_times = {}

        def company_count_executor(step: Step, namespace: dict) -> StepResult:
            step_start_times[step.number] = time.time()
            time.sleep(DELAY_MS / 1000)

            if step.number <= 5:
                return StepResult(
                    success=True,
                    stdout=f"Company {step.number}: 100 employees",
                    variables={f"count_{step.number}": 100},
                )
            else:
                total = sum(
                    namespace.get(f"count_{i}", 0)
                    for i in range(1, 6)
                )
                return StepResult(
                    success=True,
                    stdout=f"Total: {total} employees",
                    variables={"total": total},
                )

        plan = Plan(
            problem="Get total employees across all companies",
            steps=[
                Step(number=1, goal="Get count from Company A", depends_on=[]),
                Step(number=2, goal="Get count from Company B", depends_on=[]),
                Step(number=3, goal="Get count from Company C", depends_on=[]),
                Step(number=4, goal="Get count from Company D", depends_on=[]),
                Step(number=5, goal="Get count from Company E", depends_on=[]),
                Step(number=6, goal="Compute total", depends_on=[1, 2, 3, 4, 5]),
            ],
        )

        scheduler = ParallelStepScheduler(step_executor=company_count_executor)

        start = time.time()
        result = await scheduler.execute_plan(plan)
        total_time = (time.time() - start) * 1000

        assert result.success
        assert len(result.completed_steps) == 6

        assert len(result.execution_waves) == 2
        assert set(result.execution_waves[0]) == {1, 2, 3, 4, 5}
        assert result.execution_waves[1] == [6]

        assert result.step_results[6].variables.get("total") == 500

        print(f"\nEmployee count scenario: {total_time:.0f}ms")
        assert total_time < 200, f"Expected <200ms, got {total_time:.0f}ms"

        start_times = [step_start_times[i] for i in range(1, 6)]
        start_spread = max(start_times) - min(start_times)
        print(f"Start time spread for parallel steps: {start_spread*1000:.0f}ms")
        assert start_spread < 0.05, "Parallel steps should start within 50ms of each other"

    def test_sync_wrapper_works(self):
        """Test synchronous wrapper for execute_plan."""
        from constat.execution.parallel_scheduler import ParallelStepScheduler

        def simple_executor(step: Step, namespace: dict) -> StepResult:
            return StepResult(
                success=True,
                stdout=f"Step {step.number}",
                variables={f"v{step.number}": step.number},
            )

        plan = Plan(
            problem="Test sync",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[1]),
            ],
        )

        scheduler = ParallelStepScheduler(step_executor=simple_executor)
        result = scheduler.execute_plan_sync(plan)

        assert result.success
        assert len(result.completed_steps) == 2
