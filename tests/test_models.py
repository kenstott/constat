"""Tests for core data models."""

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
