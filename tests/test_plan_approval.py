"""Tests for plan approval workflow.

These tests verify the approval mechanism without requiring API calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from constat.execution.mode import (
    ExecutionMode,
    PlanApproval,
    PlanApprovalRequest,
    PlanApprovalResponse,
    suggest_mode,
    ModeSelection,
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
            mode=ExecutionMode.EXPLORATORY,
            mode_reasoning="Query suggests exploratory analysis",
            steps=steps,
            reasoning="Two-step approach for data analysis",
        )
        assert request.problem == "Analyze customer data"
        assert request.mode == ExecutionMode.EXPLORATORY
        assert len(request.steps) == 2

    def test_plan_approval_request_format_for_display(self):
        """Test formatting approval request for display."""
        steps = [
            {"number": 1, "goal": "Load data"},
            {"number": 2, "goal": "Process data"},
        ]
        request = PlanApprovalRequest(
            problem="Test problem",
            mode=ExecutionMode.AUDITABLE,
            mode_reasoning="Compliance query",
            steps=steps,
            reasoning="Step-by-step approach",
        )
        display = request.format_for_display()
        assert "AUDITABLE" in display
        assert "Load data" in display
        assert "Process data" in display
        assert "Step-by-step approach" in display


class TestModeSelection:
    """Tests for mode selection and suggest_mode."""

    def test_suggest_mode_auditable_keywords(self):
        """Test that auditable keywords trigger AUDITABLE mode."""
        queries = [
            "Why is this customer flagged as high risk?",
            "Prove that the transaction is compliant",
            "Explain why this loan was rejected",
            "What is the reasoning behind this classification?",
        ]
        for query in queries:
            selection = suggest_mode(query)
            assert selection.mode == ExecutionMode.AUDITABLE, f"Query '{query}' should be AUDITABLE"
            assert len(selection.matched_keywords) > 0

    def test_suggest_mode_exploratory_keywords(self):
        """Test that exploratory keywords trigger EXPLORATORY mode."""
        queries = [
            "Show me a dashboard of sales",
            "Create a report of customer trends",
            "Display a chart of revenue by month",
            "Build an overview of the data",
        ]
        for query in queries:
            selection = suggest_mode(query)
            assert selection.mode == ExecutionMode.EXPLORATORY, f"Query '{query}' should be EXPLORATORY"
            assert len(selection.matched_keywords) > 0

    def test_suggest_mode_default_auditable(self):
        """Test that ambiguous queries default to AUDITABLE."""
        selection = suggest_mode("Get customer data")
        assert selection.mode == ExecutionMode.AUDITABLE
        assert selection.confidence == 0.5

    def test_suggest_mode_confidence_increases_with_matches(self):
        """Test that confidence increases with more keyword matches."""
        # Single keyword
        selection1 = suggest_mode("why")
        # Multiple keywords
        selection2 = suggest_mode("why prove justify explain")

        assert selection2.confidence > selection1.confidence

    def test_mode_selection_reasoning(self):
        """Test that mode selection includes reasoning."""
        selection = suggest_mode("Show me a dashboard")
        assert "exploratory" in selection.reasoning.lower()
        assert "dashboard" in str(selection.matched_keywords)


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

    def test_show_mode_selection_exploratory(self, mock_console):
        """Test displaying exploratory mode selection."""
        from constat.feedback import FeedbackDisplay

        display = FeedbackDisplay(console=mock_console)
        display.show_mode_selection(ExecutionMode.EXPLORATORY, "Query suggests analysis")

        mock_console.print.assert_called()

    def test_show_mode_selection_auditable(self, mock_console):
        """Test displaying auditable mode selection."""
        from constat.feedback import FeedbackDisplay

        display = FeedbackDisplay(console=mock_console)
        display.show_mode_selection(ExecutionMode.AUDITABLE, "Compliance required")

        mock_console.print.assert_called()

    def test_show_replan_notice(self, mock_console):
        """Test displaying replan notice."""
        from constat.feedback import FeedbackDisplay

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
            mode=ExecutionMode.EXPLORATORY,
            mode_reasoning="Test reasoning",
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
            mode=ExecutionMode.EXPLORATORY,
            mode_reasoning="Test",
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
            mode=ExecutionMode.EXPLORATORY,
            mode_reasoning="Test",
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
            mode=ExecutionMode.EXPLORATORY,
            mode_reasoning="Exploratory",
            steps=[{"number": 1, "goal": "Delete everything"}],
            reasoning="Deletion plan",
        )

        # User rejects
        response = PlanApprovalResponse.reject("This would delete important data")
        assert response.decision == PlanApproval.REJECT
        assert response.reason == "This would delete important data"


class TestModeSystemPrompts:
    """Tests for mode-specific system prompts."""

    def test_exploratory_prompt_exists(self):
        """Test that exploratory mode has a system prompt."""
        from constat.execution.mode import MODE_SYSTEM_PROMPTS

        assert ExecutionMode.EXPLORATORY in MODE_SYSTEM_PROMPTS
        prompt = MODE_SYSTEM_PROMPTS[ExecutionMode.EXPLORATORY]
        assert "data analyst" in prompt.lower()

    def test_auditable_prompt_exists(self):
        """Test that auditable mode has a system prompt."""
        from constat.execution.mode import MODE_SYSTEM_PROMPTS

        assert ExecutionMode.AUDITABLE in MODE_SYSTEM_PROMPTS
        prompt = MODE_SYSTEM_PROMPTS[ExecutionMode.AUDITABLE]
        assert "audit" in prompt.lower() or "proof" in prompt.lower()

    def test_get_mode_system_prompt(self):
        """Test getting mode system prompt."""
        from constat.execution.mode import get_mode_system_prompt

        exploratory_prompt = get_mode_system_prompt(ExecutionMode.EXPLORATORY)
        auditable_prompt = get_mode_system_prompt(ExecutionMode.AUDITABLE)

        assert exploratory_prompt != auditable_prompt
        assert len(exploratory_prompt) > 0
        assert len(auditable_prompt) > 0


class TestDomainPresets:
    """Tests for domain-specific configuration presets."""

    def test_financial_domain_enforces_auditable(self):
        """Test that financial domain enforces auditable mode."""
        from constat.execution.mode import get_domain_preset

        config = get_domain_preset("financial")
        assert config.default_mode == ExecutionMode.AUDITABLE
        assert config.allow_mode_override is False
        assert config.min_confidence >= 0.8

    def test_healthcare_domain_enforces_auditable(self):
        """Test that healthcare domain enforces auditable mode."""
        from constat.execution.mode import get_domain_preset

        config = get_domain_preset("healthcare")
        assert config.default_mode == ExecutionMode.AUDITABLE
        assert config.allow_mode_override is False
        assert config.require_provenance is True

    def test_analytics_domain_allows_exploratory(self):
        """Test that analytics domain allows exploratory mode."""
        from constat.execution.mode import get_domain_preset

        config = get_domain_preset("analytics")
        assert config.default_mode == ExecutionMode.EXPLORATORY
        assert config.allow_mode_override is True

    def test_unknown_domain_returns_default(self):
        """Test that unknown domain returns default config."""
        from constat.execution.mode import get_domain_preset, ExecutionConfig

        config = get_domain_preset("unknown_domain")
        default = ExecutionConfig()
        assert config.default_mode == default.default_mode
