# Copyright (c) 2025 Kenneth Stott
# Canary: ec4e337d-b4ad-4dd8-b15f-6c8be0b85a48
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for REPL display/output formatting — StatusLine, FeedbackDisplay, /remember."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from rich.console import Console

from constat.repl import InteractiveREPL
from constat.core.config import Config


@pytest.fixture
def mock_config():
    """Create a minimal mock Config."""
    config = Mock(spec=Config)
    config.databases = {}
    config.llm = Mock()
    config.execution = Mock()
    config.execution.timeout_seconds = 60
    config.execution.allowed_imports = []
    config.system_prompt = ""
    config.model_dump = Mock(return_value={})
    config.databases_description = None
    config.documents = None
    config.apis = None
    config.context_preload = None
    return config


@pytest.fixture
def mock_console():
    """Create a mock Rich Console that captures output."""
    console = Mock(spec=Console)
    console.print = Mock()
    return console


@pytest.fixture
def repl(mock_config, mock_console):
    """Create an InteractiveREPL with mocked dependencies."""
    with patch('constat.repl.interactive._core.FeedbackDisplay'), \
         patch('constat.repl.interactive._core.FactStore') as mock_fact_store_class, \
         patch('constat.repl.interactive._core.Session') as mock_session_class, \
         patch('constat.repl.interactive._core.LearningStore') as mock_learning_store_class:
        mock_fact_store = Mock()
        mock_fact_store.list_facts.return_value = {}
        mock_fact_store_class.return_value = mock_fact_store

        mock_learning_store = Mock()
        mock_learning_store.get_stats.return_value = {"unpromoted": 0}
        mock_learning_store_class.return_value = mock_learning_store

        mock_session = Mock()
        mock_session.session_id = None
        mock_session.datastore = None
        mock_session_class.return_value = mock_session

        repl = InteractiveREPL(
            config=mock_config,
            verbose=False,
            console=mock_console,
        )
    return repl


class TestRememberCommand:
    """Tests for /remember command - promoting session facts to persistent storage."""

    def test_remember_session_fact(self, repl, mock_console):
        """Test /remember <fact-name> persists a session fact."""
        from constat.execution.fact_resolver import Fact, FactSource

        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {
            "churn_rate()": Fact(
                name="churn_rate",
                value=0.032,
                source=FactSource.DATABASE,
                source_name="analytics_db",
                query="SELECT AVG(churned) FROM customers",
                resolved_at=datetime.now(),
            ),
        }

        repl._remember_fact("churn_rate")

        repl.api.fact_store.save_fact.assert_called_once()
        call_args = repl.api.fact_store.save_fact.call_args
        assert call_args.kwargs["name"] == "churn_rate"
        assert call_args.kwargs["value"] == 0.032
        assert "database" in call_args.kwargs["context"].lower()

        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Remembered" in calls_str
        assert "churn_rate" in calls_str

    def test_remember_session_fact_with_rename(self, repl, mock_console):
        """Test /remember <fact-name> as <new-name> persists with new name."""
        from constat.execution.fact_resolver import Fact, FactSource

        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {
            "enterprise_churn()": Fact(
                name="enterprise_churn",
                value=0.015,
                source=FactSource.DATABASE,
                resolved_at=datetime.now(),
            ),
        }

        repl._remember_fact("enterprise_churn as baseline_churn")

        repl.api.fact_store.save_fact.assert_called_once()
        call_args = repl.api.fact_store.save_fact.call_args
        assert call_args.kwargs["name"] == "baseline_churn"
        assert call_args.kwargs["value"] == 0.015

        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Renamed from" in calls_str

    def test_remember_fact_by_name_property(self, repl, mock_console):
        """Test /remember matches by fact.name property, not just cache key."""
        from constat.execution.fact_resolver import Fact, FactSource

        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {
            "customer_ltv(segment=enterprise)": Fact(
                name="customer_ltv",
                value=50000,
                source=FactSource.DATABASE,
                resolved_at=datetime.now(),
            ),
        }

        repl._remember_fact("customer_ltv")

        repl.api.fact_store.save_fact.assert_called_once()
        assert repl.api.fact_store.save_fact.call_args.kwargs["value"] == 50000

    def test_remember_fact_not_found_shows_error(self, repl, mock_console):
        """Test /remember with non-existent fact shows helpful error."""
        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {}
        repl.api.session.fact_resolver.add_user_facts_from_text.return_value = []
        repl.display = Mock()

        repl._remember_fact("nonexistent_fact")

        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "No session fact named" in calls_str or "nonexistent_fact" in calls_str

    def test_remember_falls_back_to_text_extraction(self, repl, mock_console):
        """Test /remember falls back to LLM extraction for natural language."""
        from constat.execution.fact_resolver import Fact, FactSource

        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {}

        mock_fact = Fact(
            name="user_role",
            value="CFO",
            source=FactSource.USER_PROVIDED,
            description="User's role",
        )
        repl.api.session.fact_resolver.add_user_facts_from_text.return_value = [mock_fact]
        repl.display = Mock()

        repl._remember_fact("my role is CFO")

        repl.api.session.fact_resolver.add_user_facts_from_text.assert_called_once_with("my role is CFO")

        repl.api.fact_store.save_fact.assert_called_once()
        assert repl.api.fact_store.save_fact.call_args.kwargs["name"] == "user_role"
        assert repl.api.fact_store.save_fact.call_args.kwargs["value"] == "CFO"

    def test_remember_preserves_provenance(self, repl, mock_console):
        """Test /remember preserves full provenance in context."""
        from constat.execution.fact_resolver import Fact, FactSource

        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {
            "revenue()": Fact(
                name="revenue",
                value=1000000,
                source=FactSource.DATABASE,
                source_name="sales_db",
                query="SELECT SUM(amount) FROM sales",
                reasoning="Aggregated from sales table",
                resolved_at=datetime(2024, 1, 15, 10, 30, 0),
            ),
        }

        repl._remember_fact("revenue")

        call_args = repl.api.fact_store.save_fact.call_args
        context = call_args.kwargs["context"]
        assert "database" in context.lower()
        assert "sales_db" in context
        assert "SELECT SUM(amount)" in context

    def test_remember_no_args_shows_usage(self, repl, mock_console):
        """Test /remember without arguments shows usage."""
        repl._remember_fact("")

        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Usage" in calls_str
        assert "/remember" in calls_str

    def test_remember_handles_table_facts(self, repl, mock_console):
        """Test /remember handles facts with table references."""
        from constat.execution.fact_resolver import Fact, FactSource

        repl.api._session = Mock()
        repl.api.session.fact_resolver = Mock()
        repl.api.session.fact_resolver.get_all_facts.return_value = {
            "top_customers()": Fact(
                name="top_customers",
                value=None,
                source=FactSource.DATABASE,
                table_name="_top_customers",
                row_count=100,
                resolved_at=datetime.now(),
            ),
        }

        repl._remember_fact("top_customers")

        repl.api.fact_store.save_fact.assert_called_once()
        calls = [str(call) for call in mock_console.print.call_args_list]
        calls_str = " ".join(calls)
        assert "Remembered" in calls_str


class TestStatusLine:
    """Tests for StatusLine class."""

    def test_status_line_initial_state(self):
        """Test StatusLine initial render."""
        from constat.repl.feedback import StatusLine

        status = StatusLine()

        rendered = status.render()
        assert "Ready" in rendered

    def test_status_line_planning_phase(self):
        """Test StatusLine renders planning phase correctly."""
        from constat.repl.feedback import StatusLine
        from constat.execution.mode import Phase

        status = StatusLine()
        status._phase = Phase.PLANNING
        status._plan_name = "Analyze revenue"

        rendered = status.render()
        assert "Planning" in rendered
        assert "Analyze revenue" in rendered

    def test_status_line_executing_phase(self):
        """Test StatusLine renders executing phase correctly."""
        from constat.repl.feedback import StatusLine
        from constat.execution.mode import Phase

        status = StatusLine()
        status._phase = Phase.EXECUTING
        status._step_current = 2
        status._step_total = 5
        status._step_description = "Loading data"

        rendered = status.render()
        assert "Executing" in rendered
        assert "Step 2/5" in rendered
        assert "Loading data" in rendered

    def test_status_line_failed_phase(self):
        """Test StatusLine renders failed phase correctly."""
        from constat.repl.feedback import StatusLine
        from constat.execution.mode import Phase

        status = StatusLine()
        status._phase = Phase.FAILED
        status._step_current = 3
        status._error_message = "Connection timeout"

        rendered = status.render()
        assert "Failed" in rendered
        assert "step 3" in rendered
        assert "Connection timeout" in rendered
        assert "retry/replan/abandon" in rendered

    def test_status_line_awaiting_approval_phase(self):
        """Test StatusLine renders awaiting_approval phase correctly."""
        from constat.repl.feedback import StatusLine
        from constat.execution.mode import Phase

        status = StatusLine()
        status._phase = Phase.AWAITING_APPROVAL
        status._plan_name = "Revenue analysis"
        status._step_total = 4

        rendered = status.render()
        assert "Awaiting approval" in rendered
        assert "(4 steps)" in rendered
        assert "y/n/suggest" in rendered

    def test_status_line_update_from_state(self):
        """Test StatusLine.update() from ConversationState."""
        from constat.repl.feedback import StatusLine
        from constat.execution.mode import Phase, ConversationState

        status = StatusLine()
        state = ConversationState(
            phase=Phase.EXECUTING,
            failure_context="Test error",
        )

        status.update(state)

        assert status._phase == Phase.EXECUTING
        assert status._error_message == "Test error"

    def test_status_line_queue_indicator(self):
        """Test StatusLine shows queue indicator during execution."""
        from constat.repl.feedback import StatusLine
        from constat.execution.mode import Phase

        status = StatusLine()
        status._phase = Phase.EXECUTING
        status._queue_count = 2

        rendered = status.render()
        assert "Queued: 2" in rendered

    def test_status_line_spinner_advance(self):
        """Test StatusLine.advance_spinner() cycles through frames."""
        from constat.repl.feedback import StatusLine

        status = StatusLine()
        initial_frame = status._spinner_frame

        status.advance_spinner()
        assert status._spinner_frame == initial_frame + 1

    def test_status_line_status_message(self):
        """Test StatusLine renders status message when set."""
        from constat.repl.feedback import StatusLine

        status = StatusLine()
        status.set_status_message("Analyzing your question...")

        rendered = status.render()
        assert "Analyzing your question..." in rendered
        assert "Ready" not in rendered

        status.set_status_message(None)
        rendered = status.render()
        assert "Analyzing" not in rendered
        assert "Ready" in rendered


class TestFailureRecovery:
    """Tests for failure recovery prompts."""

    def test_failure_recovery_retry(self, mock_console):
        """Test failure recovery prompt returns 'retry' for retry input."""
        from constat.repl.feedback import FeedbackDisplay

        display = FeedbackDisplay(console=mock_console)

        mock_console.input = Mock(return_value="retry")

        result = display.request_failure_recovery("Connection error", "step 2")

        assert result == "retry"

    def test_failure_recovery_replan(self, mock_console):
        """Test failure recovery prompt returns 'replan' for replan input."""
        from constat.repl.feedback import FeedbackDisplay

        display = FeedbackDisplay(console=mock_console)

        mock_console.input = Mock(return_value="replan")

        result = display.request_failure_recovery("API error")

        assert result == "replan"

    def test_failure_recovery_abandon(self, mock_console):
        """Test failure recovery prompt returns 'abandon' for abandon input."""
        from constat.repl.feedback import FeedbackDisplay

        display = FeedbackDisplay(console=mock_console)

        mock_console.input = Mock(return_value="abandon")

        result = display.request_failure_recovery("Unknown error")

        assert result == "abandon"

    def test_failure_recovery_numeric_shortcuts(self, mock_console):
        """Test failure recovery accepts numeric shortcuts."""
        from constat.repl.feedback import FeedbackDisplay

        display = FeedbackDisplay(console=mock_console)

        mock_console.input = Mock(return_value="1")
        assert display.request_failure_recovery("Error") == "retry"

        mock_console.input = Mock(return_value="2")
        assert display.request_failure_recovery("Error") == "replan"

        mock_console.input = Mock(return_value="3")
        assert display.request_failure_recovery("Error") == "abandon"

    def test_failure_recovery_keyboard_interrupt(self, mock_console):
        """Test failure recovery handles KeyboardInterrupt."""
        from constat.repl.feedback import FeedbackDisplay

        display = FeedbackDisplay(console=mock_console)

        mock_console.input = Mock(side_effect=KeyboardInterrupt())

        result = display.request_failure_recovery("Error")

        assert result == "abandon"

    def test_failure_recovery_empty_input(self, mock_console):
        """Test failure recovery treats empty input as abandon."""
        from constat.repl.feedback import FeedbackDisplay

        display = FeedbackDisplay(console=mock_console)

        mock_console.input = Mock(return_value="")

        result = display.request_failure_recovery("Error")

        assert result == "abandon"
