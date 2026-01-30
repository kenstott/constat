# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for plan approval workflow.

These tests verify the approval mechanism without requiring API calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from constat.execution.mode import (
    PlanApproval,
    PlanApprovalRequest,
    PlanApprovalResponse,
)


class TestPlanApprovalModels:
    """Tests for approval model classes."""

    def test_plan_approval_enum_values(self):
        """Test PlanApproval enum has expected values."""
        assert PlanApproval.APPROVE.value == "approve"
        assert PlanApproval.REJECT.value == "reject"
        assert PlanApproval.SUGGEST.value == "suggest"

    def test_plan_approval_response_approve(self):
        """Test creating an approval response."""
        response = PlanApprovalResponse.approve()
        assert response.decision == PlanApproval.APPROVE
        assert response.suggestion is None
        assert response.reason is None

    def test_plan_approval_response_reject(self):
        """Test creating a rejection response."""
        response = PlanApprovalResponse.reject("Too complex")
        assert response.decision == PlanApproval.REJECT
        assert response.reason == "Too complex"
        assert response.suggestion is None

    def test_plan_approval_response_reject_no_reason(self):
        """Test rejection without reason."""
        response = PlanApprovalResponse.reject()
        assert response.decision == PlanApproval.REJECT
        assert response.reason is None

    def test_plan_approval_response_suggest(self):
        """Test creating a suggestion response."""
        response = PlanApprovalResponse.suggest("Add error handling step")
        assert response.decision == PlanApproval.SUGGEST
        assert response.suggestion == "Add error handling step"
        assert response.reason is None

    def test_plan_approval_request_creation(self):
        """Test creating an approval request."""
        steps = [
            {"number": 1, "goal": "Load data", "inputs": [], "outputs": ["data"]},
            {"number": 2, "goal": "Process data", "inputs": ["data"], "outputs": ["result"]},
        ]
        request = PlanApprovalRequest(
            problem="Analyze customer data",
            steps=steps,
            reasoning="Two-step approach for data analysis",
        )
        assert request.problem == "Analyze customer data"
        assert len(request.steps) == 2

    def test_plan_approval_request_format_for_display(self):
        """Test formatting approval request for display."""
        steps = [
            {"number": 1, "goal": "Load data"},
            {"number": 2, "goal": "Process data"},
        ]
        request = PlanApprovalRequest(
            problem="Test problem",
            steps=steps,
            reasoning="Step-by-step approach",
        )
        display = request.format_for_display()
        assert "Load data" in display
        assert "Process data" in display
        assert "Step-by-step approach" in display


class TestSessionApprovalIntegration:
    """Tests for Session integration with approval workflow."""

    @pytest.fixture
    def mock_session_deps(self):
        """Create mock dependencies for Session."""
        # Mock config
        config = MagicMock()
        config.databases = {"test": MagicMock()}
        config.model_dump.return_value = {}
        config.execution = MagicMock()
        config.execution.timeout_seconds = 60
        config.execution.allowed_imports = None
        config.llm = MagicMock()
        config.system_prompt = None

        return config

    def test_session_config_defaults(self):
        """Test SessionConfig default values."""
        from constat.session import SessionConfig

        config = SessionConfig()
        assert config.require_approval is True
        assert config.max_replan_attempts == 3
        assert config.auto_approve is False

    def test_session_config_auto_approve(self):
        """Test SessionConfig with auto_approve."""
        from constat.session import SessionConfig

        config = SessionConfig(auto_approve=True)
        assert config.auto_approve is True

    def test_plan_approval_response_is_dataclass(self):
        """Test that PlanApprovalResponse is properly structured."""
        response = PlanApprovalResponse(
            decision=PlanApproval.APPROVE,
            suggestion=None,
            reason=None,
        )
        assert response.decision == PlanApproval.APPROVE

    def test_approval_callback_type(self):
        """Test that approval callback has correct signature."""
        from constat.session import ApprovalCallback

        def my_callback(request: PlanApprovalRequest) -> PlanApprovalResponse:
            return PlanApprovalResponse.approve()

        # This should type-check correctly
        callback: ApprovalCallback = my_callback
        assert callable(callback)


class TestFeedbackDisplayApproval:
    """Tests for FeedbackDisplay approval methods."""

    @pytest.fixture
    def mock_console(self):
        """Create a mock console."""
        return MagicMock()

    def test_show_replan_notice(self, mock_console):
        """Test displaying replan notice."""
        from constat.repl.feedback import FeedbackDisplay

        display = FeedbackDisplay(console=mock_console)
        display.show_replan_notice(attempt=2, max_attempts=3)

        mock_console.print.assert_called()
        call_args = str(mock_console.print.call_args)
        assert "2" in call_args
        assert "3" in call_args


class TestApprovalWorkflow:
    """Integration tests for complete approval workflow."""

    def test_approval_request_to_response_flow(self):
        """Test the complete flow from request to response."""
        # Create a request
        steps = [{"number": 1, "goal": "Test step"}]
        request = PlanApprovalRequest(
            problem="Test problem",
            steps=steps,
            reasoning="Test plan reasoning",
        )

        # Simulate user approval
        response = PlanApprovalResponse.approve()

        assert response.decision == PlanApproval.APPROVE

    def test_suggestion_workflow(self):
        """Test the suggestion workflow."""
        # Initial request
        steps = [{"number": 1, "goal": "Basic step"}]
        request = PlanApprovalRequest(
            problem="Test problem",
            steps=steps,
            reasoning="Initial plan",
        )

        # User suggests changes
        response = PlanApprovalResponse.suggest("Add validation step")
        assert response.decision == PlanApproval.SUGGEST
        assert response.suggestion == "Add validation step"

        # System would replan and create new request
        new_steps = [
            {"number": 1, "goal": "Validate data"},
            {"number": 2, "goal": "Basic step"},
        ]
        new_request = PlanApprovalRequest(
            problem="Test problem",
            steps=new_steps,
            reasoning="Revised plan with validation",
        )

        # User approves revised plan
        final_response = PlanApprovalResponse.approve()
        assert final_response.decision == PlanApproval.APPROVE

    def test_rejection_workflow(self):
        """Test the rejection workflow."""
        request = PlanApprovalRequest(
            problem="Delete all data",
            steps=[{"number": 1, "goal": "Delete everything"}],
            reasoning="Deletion plan",
        )

        # User rejects
        response = PlanApprovalResponse.reject("This would delete important data")
        assert response.decision == PlanApproval.REJECT
        assert response.reason == "This would delete important data"
