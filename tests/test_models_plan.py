# Copyright (c) 2025 Kenneth Stott
# Canary: 38d1ca9b-5a69-4d97-adce-b96f15d7e23a
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for Plan, Step, and StepResult models."""

from __future__ import annotations
import pytest
from constat.core.models import Plan, Step, StepResult, StepStatus, StepType


class TestStep:
    """Tests for Step model."""

    def test_step_creation(self):
        """Test creating a step."""
        step = Step(
            number=1,
            goal="Load customer data",
            expected_inputs=["config"],
            expected_outputs=["customers_df"],
        )

        assert step.number == 1
        assert step.goal == "Load customer data"
        assert step.expected_inputs == ["config"]
        assert step.expected_outputs == ["customers_df"]
        assert step.step_type == StepType.PYTHON
        assert step.status == StepStatus.PENDING
        assert step.code is None
        assert step.result is None

    def test_step_defaults(self):
        """Test step defaults."""
        step = Step(number=1, goal="Test step")

        assert step.expected_inputs == []
        assert step.expected_outputs == []
        assert step.step_type == StepType.PYTHON

    def test_step_depends_on_field(self):
        """Test that Step has depends_on field."""
        step = Step(
            number=1,
            goal="Test step",
            depends_on=[2, 3],
        )
        assert step.depends_on == [2, 3]

    def test_step_depends_on_defaults_empty(self):
        """Test depends_on defaults to empty list."""
        step = Step(number=1, goal="Test step")
        assert step.depends_on == []


class TestStepResult:
    """Tests for StepResult model."""

    def test_successful_result(self):
        """Test successful step result."""
        result = StepResult(
            success=True,
            stdout="Loaded 100 rows",
            attempts=1,
            duration_ms=500,
        )

        assert result.success
        assert result.stdout == "Loaded 100 rows"
        assert result.error is None
        assert result.attempts == 1
        assert result.duration_ms == 500

    def test_failed_result(self):
        """Test failed step result."""
        result = StepResult(
            success=False,
            stdout="",
            error="Column not found",
            attempts=3,
            duration_ms=1500,
        )

        assert not result.success
        assert result.error == "Column not found"
        assert result.attempts == 3


class TestPlan:
    """Tests for Plan model."""

    @pytest.fixture
    def sample_plan(self):
        """Create a sample plan for testing."""
        return Plan(
            problem="Analyze sales data",
            steps=[
                Step(number=1, goal="Load data", expected_outputs=["sales_df"]),
                Step(number=2, goal="Calculate totals", expected_inputs=["sales_df"], expected_outputs=["totals"]),
                Step(number=3, goal="Generate report", expected_inputs=["totals"]),
            ],
        )

    def test_plan_creation(self, sample_plan):
        """Test plan creation."""
        assert sample_plan.problem == "Analyze sales data"
        assert len(sample_plan.steps) == 3
        assert sample_plan.current_step == 0
        assert sample_plan.completed_steps == []

    def test_is_complete_initially_false(self, sample_plan):
        """Test is_complete is False initially."""
        assert not sample_plan.is_complete

    def test_next_step(self, sample_plan):
        """Test getting next step."""
        next_step = sample_plan.next_step
        assert next_step is not None
        assert next_step.number == 1

    def test_get_step(self, sample_plan):
        """Test getting step by number."""
        step = sample_plan.get_step(2)
        assert step is not None
        assert step.goal == "Calculate totals"

        assert sample_plan.get_step(99) is None

    def test_mark_step_completed(self, sample_plan):
        """Test marking a step as completed."""
        result = StepResult(success=True, stdout="Done", attempts=1, duration_ms=100)
        sample_plan.mark_step_completed(1, result)

        assert 1 in sample_plan.completed_steps
        step = sample_plan.get_step(1)
        assert step.status == StepStatus.COMPLETED
        assert step.result == result

    def test_mark_step_failed(self, sample_plan):
        """Test marking a step as failed."""
        result = StepResult(success=False, stdout="", error="Error", attempts=3, duration_ms=500)
        sample_plan.mark_step_failed(2, result)

        assert 2 in sample_plan.failed_steps
        step = sample_plan.get_step(2)
        assert step.status == StepStatus.FAILED

    def test_is_complete_when_all_done(self, sample_plan):
        """Test is_complete when all steps are done."""
        result = StepResult(success=True, stdout="Done", attempts=1, duration_ms=100)

        sample_plan.mark_step_completed(1, result)
        assert not sample_plan.is_complete

        sample_plan.mark_step_completed(2, result)
        assert not sample_plan.is_complete

        sample_plan.mark_step_completed(3, result)
        assert sample_plan.is_complete

    def test_next_step_skips_completed(self, sample_plan):
        """Test next_step skips completed steps."""
        result = StepResult(success=True, stdout="Done", attempts=1, duration_ms=100)
        sample_plan.mark_step_completed(1, result)

        next_step = sample_plan.next_step
        assert next_step.number == 2


class TestPlanDependencies:
    """Tests for Plan dependency analysis and parallel execution support."""

    def test_infer_dependencies_from_inputs_outputs(self):
        """Test inferring dependencies from input/output overlap."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="Load A", expected_outputs=["data_a"]),
                Step(number=2, goal="Load B", expected_outputs=["data_b"]),
                Step(number=3, goal="Combine", expected_inputs=["data_a", "data_b"], expected_outputs=["combined"]),
                Step(number=4, goal="Report", expected_inputs=["combined"]),
            ],
        )

        plan.infer_dependencies()

        step3 = plan.get_step(3)
        assert 1 in step3.depends_on
        assert 2 in step3.depends_on

        step4 = plan.get_step(4)
        assert 3 in step4.depends_on

        assert plan.get_step(1).depends_on == []
        assert plan.get_step(2).depends_on == []

    def test_get_dependency_graph(self):
        """Test getting dependency graph as adjacency list."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[]),
                Step(number=3, goal="C", depends_on=[1, 2]),
            ],
        )

        graph = plan.get_dependency_graph()

        assert graph == {1: [], 2: [], 3: [1, 2]}

    def test_get_runnable_steps_initial(self):
        """Test getting runnable steps when none are completed."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[]),
                Step(number=3, goal="C", depends_on=[1, 2]),
            ],
        )

        runnable = plan.get_runnable_steps()

        runnable_numbers = [s.number for s in runnable]
        assert 1 in runnable_numbers
        assert 2 in runnable_numbers
        assert 3 not in runnable_numbers

    def test_get_runnable_steps_after_completion(self):
        """Test getting runnable steps after some are completed."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[]),
                Step(number=3, goal="C", depends_on=[1, 2]),
            ],
        )

        result = StepResult(success=True, stdout="Done", attempts=1, duration_ms=100)
        plan.mark_step_completed(1, result)
        plan.mark_step_completed(2, result)

        runnable = plan.get_runnable_steps()

        runnable_numbers = [s.number for s in runnable]
        assert 3 in runnable_numbers
        assert 1 not in runnable_numbers
        assert 2 not in runnable_numbers

    def test_get_runnable_steps_partial_completion(self):
        """Test getting runnable steps with partial completion."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[]),
                Step(number=3, goal="C", depends_on=[1, 2]),
            ],
        )

        result = StepResult(success=True, stdout="Done", attempts=1, duration_ms=100)
        plan.mark_step_completed(1, result)

        runnable = plan.get_runnable_steps()

        runnable_numbers = [s.number for s in runnable]
        assert 2 in runnable_numbers
        assert 3 not in runnable_numbers

    def test_get_execution_order_waves(self):
        """Test getting execution order as waves."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[]),
                Step(number=3, goal="C", depends_on=[1]),
                Step(number=4, goal="D", depends_on=[2]),
                Step(number=5, goal="E", depends_on=[3, 4]),
            ],
        )

        waves = plan.get_execution_order()

        assert len(waves) == 3
        assert set(waves[0]) == {1, 2}
        assert set(waves[1]) == {3, 4}
        assert set(waves[2]) == {5}

    def test_get_execution_order_sequential(self):
        """Test execution order for fully sequential plan."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[1]),
                Step(number=3, goal="C", depends_on=[2]),
            ],
        )

        waves = plan.get_execution_order()

        assert waves == [[1], [2], [3]]

    def test_get_execution_order_fully_parallel(self):
        """Test execution order for fully parallel plan."""
        plan = Plan(
            problem="Test",
            steps=[
                Step(number=1, goal="A", depends_on=[]),
                Step(number=2, goal="B", depends_on=[]),
                Step(number=3, goal="C", depends_on=[]),
                Step(number=4, goal="D", depends_on=[]),
            ],
        )

        waves = plan.get_execution_order()

        assert len(waves) == 1
        assert set(waves[0]) == {1, 2, 3, 4}

    def test_employee_count_example(self):
        """Test the employee count across companies example from requirements."""
        plan = Plan(
            problem="Get total employees across all companies",
            steps=[
                Step(number=1, goal="Get employee count from Company A", depends_on=[], expected_outputs=["count_a"]),
                Step(number=2, goal="Get employee count from Company B", depends_on=[], expected_outputs=["count_b"]),
                Step(number=3, goal="Get employee count from Company C", depends_on=[], expected_outputs=["count_c"]),
                Step(number=4, goal="Get employee count from Company D", depends_on=[], expected_outputs=["count_d"]),
                Step(number=5, goal="Get employee count from Company E", depends_on=[], expected_outputs=["count_e"]),
                Step(number=6, goal="Compute total sum", depends_on=[1, 2, 3, 4, 5], expected_inputs=["count_a", "count_b", "count_c", "count_d", "count_e"]),
            ],
        )

        waves = plan.get_execution_order()

        assert len(waves) == 2
        assert set(waves[0]) == {1, 2, 3, 4, 5}
        assert waves[1] == [6]
