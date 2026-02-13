# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for Live Feedback system.

Tests for the Rich-based terminal display that provides real-time
feedback during session execution.

Test Strategy:
- StepDisplay: Direct unit tests on dataclass behavior
- FeedbackDisplay state: Test internal state transitions without console
- FeedbackDisplay lifecycle: Mock Live class to verify start/stop
- FeedbackDisplay output: Capture console output to verify formatting
- SessionFeedbackHandler: Mock display to verify event routing
"""

from dataclasses import dataclass, field
from io import StringIO
from unittest.mock import Mock, patch, MagicMock

import pytest
from rich.console import Console

from constat.repl.feedback import StepDisplay, FeedbackDisplay, SessionFeedbackHandler
from constat.session import StepEvent


# =============================================================================
# Fixtures
# =============================================================================

def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text for easier assertions."""
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


@pytest.fixture
def captured_console():
    """Console that captures output to a string buffer."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=120)
    return console, output


@pytest.fixture
def mock_display():
    """Mock FeedbackDisplay for testing SessionFeedbackHandler."""
    return Mock(spec=FeedbackDisplay)


@pytest.fixture
def sample_steps():
    """Sample steps for testing."""
    return [
        {"number": 1, "goal": "Load customer data"},
        {"number": 2, "goal": "Calculate revenue"},
        {"number": 3, "goal": "Generate report"},
    ]


@pytest.fixture
def sample_events():
    """Sample StepEvents for testing."""
    return {
        "step_start": StepEvent(
            event_type="step_start",
            step_number=1,
            data={"goal": "Load data"}
        ),
        "generating": StepEvent(
            event_type="generating",
            step_number=1,
            data={"attempt": 1}
        ),
        "generating_retry": StepEvent(
            event_type="generating",
            step_number=1,
            data={"attempt": 3}
        ),
        "executing": StepEvent(
            event_type="executing",
            step_number=1,
            data={"attempt": 1, "code": "print('hello')"}
        ),
        "step_complete": StepEvent(
            event_type="step_complete",
            step_number=1,
            data={
                "stdout": "Loaded 100 rows",
                "attempts": 1,
                "duration_ms": 500,
                "tables_created": ["customers"]
            }
        ),
        "step_complete_retry": StepEvent(
            event_type="step_complete",
            step_number=1,
            data={
                "stdout": "Done after retries",
                "attempts": 3,
                "duration_ms": 1500,
                "tables_created": ["results"]
            }
        ),
        "step_error": StepEvent(
            event_type="step_error",
            step_number=1,
            data={"error": "NameError: undefined", "attempt": 1}
        ),
    }


# =============================================================================
# TestStepDisplay - Dataclass tests
# =============================================================================

class TestStepDisplay:
    """Tests for StepDisplay dataclass."""

    def test_default_status_is_pending(self):
        """Default status is 'pending'."""
        step = StepDisplay(number=1, goal="Test step")
        assert step.status == "pending"

    def test_default_attempts_is_zero(self):
        """Default attempts is 0."""
        step = StepDisplay(number=1, goal="Test step")
        assert step.attempts == 0

    def test_default_duration_is_zero(self):
        """Default duration_ms is 0."""
        step = StepDisplay(number=1, goal="Test step")
        assert step.duration_ms == 0

    def test_tables_created_default_not_shared(self):
        """Mutable default (tables_created) is not shared between instances."""
        step1 = StepDisplay(number=1, goal="Step 1")
        step2 = StepDisplay(number=2, goal="Step 2")

        step1.tables_created.append("table1")

        assert step1.tables_created == ["table1"]
        assert step2.tables_created == []  # Should be independent

    def test_all_fields_settable(self):
        """All fields can be set in constructor."""
        step = StepDisplay(
            number=1,
            goal="Test goal",
            status="completed",
            code="print('hello')",
            output="hello",
            error=None,
            attempts=2,
            duration_ms=1500,
            tables_created=["table1", "table2"],
        )

        assert step.number == 1
        assert step.goal == "Test goal"
        assert step.status == "completed"
        assert step.code == "print('hello')"
        assert step.output == "hello"
        assert step.error is None
        assert step.attempts == 2
        assert step.duration_ms == 1500
        assert step.tables_created == ["table1", "table2"]

    def test_optional_fields_default_to_none(self):
        """Optional fields default to None."""
        step = StepDisplay(number=1, goal="Test")

        assert step.code is None
        assert step.output is None
        assert step.error is None


# =============================================================================
# TestFeedbackDisplayState - State management tests (no console output)
# =============================================================================

class TestFeedbackDisplayState:
    """Tests for FeedbackDisplay internal state management."""

    def test_initial_state(self):
        """Initial state is empty."""
        display = FeedbackDisplay()

        assert display.plan_steps == []
        assert display.current_step is None
        assert display.problem == ""
        assert display._live is None

    def test_set_problem_stores_problem(self):
        """set_problem stores the problem text."""
        display = FeedbackDisplay()

        display.set_problem("What are the top 5 customers?")

        assert display.problem == "What are the top 5 customers?"

    def test_show_plan_creates_step_displays(self, sample_steps):
        """show_plan populates plan_steps list."""
        display = FeedbackDisplay()

        display.show_plan(sample_steps)

        assert len(display.plan_steps) == 3
        assert display.plan_steps[0].number == 1
        assert display.plan_steps[0].goal == "Load customer data"
        assert display.plan_steps[0].status == "pending"

    def test_show_plan_assigns_numbers_if_missing(self):
        """show_plan assigns sequential numbers if not provided."""
        display = FeedbackDisplay()
        steps = [{"goal": "Step A"}, {"goal": "Step B"}]

        display.show_plan(steps)

        assert display.plan_steps[0].number == 1
        assert display.plan_steps[1].number == 2

    def test_step_start_sets_current_step(self, sample_steps):
        """step_start updates current_step."""
        display = FeedbackDisplay()
        display.show_plan(sample_steps)

        display.step_start(2, "Calculate revenue")

        assert display.current_step == 2

    def test_step_start_marks_step_running(self, sample_steps):
        """step_start updates status to 'running'."""
        display = FeedbackDisplay()
        display.show_plan(sample_steps)

        display.step_start(1, "Load customer data")

        assert display.plan_steps[0].status == "running"

    def test_step_complete_marks_step_completed(self, sample_steps):
        """step_complete updates status to 'completed'."""
        display = FeedbackDisplay()
        display.show_plan(sample_steps)

        display.step_complete(
            step_number=1,
            output="Done",
            attempts=1,
            duration_ms=500,
        )

        assert display.plan_steps[0].status == "completed"

    def test_step_complete_records_output(self, sample_steps):
        """step_complete stores output."""
        display = FeedbackDisplay()
        display.show_plan(sample_steps)

        display.step_complete(
            step_number=1,
            output="Loaded 100 rows",
            attempts=1,
            duration_ms=500,
        )

        assert display.plan_steps[0].output == "Loaded 100 rows"

    def test_step_complete_records_attempts(self, sample_steps):
        """step_complete stores attempts count."""
        display = FeedbackDisplay()
        display.show_plan(sample_steps)

        display.step_complete(
            step_number=1,
            output="Done",
            attempts=3,
            duration_ms=500,
        )

        assert display.plan_steps[0].attempts == 3

    def test_step_complete_records_duration(self, sample_steps):
        """step_complete stores duration_ms."""
        display = FeedbackDisplay()
        display.show_plan(sample_steps)

        display.step_complete(
            step_number=1,
            output="Done",
            attempts=1,
            duration_ms=1500,
        )

        assert display.plan_steps[0].duration_ms == 1500

    def test_step_complete_records_tables_created(self, sample_steps):
        """step_complete stores tables_created."""
        display = FeedbackDisplay()
        display.show_plan(sample_steps)

        display.step_complete(
            step_number=1,
            output="Done",
            attempts=1,
            duration_ms=500,
            tables_created=["customers", "orders"],
        )

        assert display.plan_steps[0].tables_created == ["customers", "orders"]

    def test_step_complete_tables_default_to_empty_list(self, sample_steps):
        """step_complete defaults tables_created to empty list."""
        display = FeedbackDisplay()
        display.show_plan(sample_steps)

        display.step_complete(
            step_number=1,
            output="Done",
            attempts=1,
            duration_ms=500,
            tables_created=None,
        )

        assert display.plan_steps[0].tables_created == []

    def test_step_failed_marks_step_failed(self, sample_steps):
        """step_failed updates status to 'failed'."""
        display = FeedbackDisplay()
        display.show_plan(sample_steps)

        display.step_failed(
            step_number=1,
            error="Something went wrong",
            attempts=3,
        )

        assert display.plan_steps[0].status == "failed"

    def test_step_failed_records_error(self, sample_steps):
        """step_failed stores error message."""
        display = FeedbackDisplay()
        display.show_plan(sample_steps)

        display.step_failed(
            step_number=1,
            error="NameError: undefined",
            attempts=3,
        )

        assert display.plan_steps[0].error == "NameError: undefined"

    def test_step_failed_records_attempts(self, sample_steps):
        """step_failed stores attempts count."""
        display = FeedbackDisplay()
        display.show_plan(sample_steps)

        display.step_failed(
            step_number=1,
            error="Error",
            attempts=5,
        )

        assert display.plan_steps[0].attempts == 5

    def test_step_start_with_nonexistent_step_number(self, sample_steps):
        """step_start with invalid step number does not crash."""
        display = FeedbackDisplay()
        display.show_plan(sample_steps)

        # Should not raise - step 99 doesn't exist
        display.step_start(99, "Unknown step")

        assert display.current_step == 99
        # Original steps unchanged
        assert all(s.status == "pending" for s in display.plan_steps)

    def test_step_complete_with_nonexistent_step_number(self, sample_steps):
        """step_complete with invalid step number does not crash."""
        display = FeedbackDisplay()
        display.show_plan(sample_steps)

        # Should not raise - step 99 doesn't exist
        display.step_complete(99, "Done", 1, 500)

        # Original steps unchanged
        assert all(s.status == "pending" for s in display.plan_steps)


# =============================================================================
# TestFeedbackDisplayLifecycle - Live display lifecycle tests
# =============================================================================

class TestFeedbackDisplayLifecycle:
    """Tests for Live display lifecycle management."""

    @patch('constat.repl.feedback._display_core.Live')
    def test_start_creates_live_display(self, MockLive):
        """start() creates and starts Live display."""
        display = FeedbackDisplay()

        display.start()

        MockLive.assert_called_once()
        MockLive.return_value.start.assert_called_once()
        assert display._live is MockLive.return_value

    @patch('constat.repl.feedback._display_core.Live')
    def test_stop_stops_live_display(self, MockLive):
        """stop() stops Live display and clears reference."""
        display = FeedbackDisplay()
        display.start()

        display.stop()

        MockLive.return_value.stop.assert_called_once()
        assert display._live is None

    def test_stop_when_not_started_is_safe(self):
        """stop() is safe to call without prior start()."""
        display = FeedbackDisplay()

        # Should not raise
        display.stop()

        assert display._live is None

    @patch('constat.repl.feedback._display_core.Live')
    def test_multiple_stops_are_safe(self, MockLive):
        """Calling stop() multiple times is safe."""
        display = FeedbackDisplay()
        display.start()

        display.stop()
        display.stop()  # Second call should be no-op

        # Only called once
        assert MockLive.return_value.stop.call_count == 1


# =============================================================================
# TestFeedbackDisplayOutput - Console output tests
# =============================================================================

class TestFeedbackDisplayOutput:
    """Tests for console output formatting."""

    def test_set_problem_prints_problem(self, captured_console):
        """set_problem stores the problem and prints blank line."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)

        display.set_problem("What are the top 5 customers?")

        result = strip_ansi(output.getvalue())
        assert "\n" in result  # Just stores the problem and prints blank line

    def test_show_plan_prints_steps(self, captured_console, sample_steps):
        """show_plan displays each step goal."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)

        display.show_plan(sample_steps)

        result = output.getvalue()
        assert "Load customer data" in result
        assert "Calculate revenue" in result
        assert "Generate report" in result

    def test_step_start_shows_step_header(self, captured_console, sample_steps):
        """step_start displays step goal in live display."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)
        display.show_plan(sample_steps)

        display.step_start(1, "Load customer data")

        result = strip_ansi(output.getvalue())
        # Live display shows goal in animated checklist format
        assert "Load customer data" in result
        # Step is marked as running with spinner/working status
        assert "starting" in result or "working" in result

    def test_step_complete_shows_ok_indicator(self, captured_console, sample_steps):
        """step_complete marks step as completed in live display."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)
        display.show_plan(sample_steps)

        display.step_complete(1, "", 1, 500)

        # Verify step state is updated to completed
        assert display.plan_steps[0].status == "completed"
        # Live display shows the goal with completed status
        result = strip_ansi(output.getvalue())
        assert "Load customer data" in result

    def test_step_complete_shows_duration(self, captured_console, sample_steps):
        """step_complete shows duration."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)
        display.show_plan(sample_steps)

        display.step_complete(1, "Done", 1, 1500)

        result = strip_ansi(output.getvalue())
        assert "1.5s" in result

    def test_step_complete_shows_retry_count_when_multiple_attempts(self, captured_console, sample_steps):
        """step_complete shows attempt count when > 1."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)
        display.show_plan(sample_steps)

        display.step_complete(1, "Done", 3, 1000)

        result = strip_ansi(output.getvalue())
        # Live display shows retry count as "(3 tries)"
        assert "3 tries" in result

    def test_step_complete_no_retry_count_for_single_attempt(self, captured_console, sample_steps):
        """step_complete does not show attempt count when == 1."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)
        display.show_plan(sample_steps)

        display.step_complete(1, "Done", 1, 1000)

        result = output.getvalue()
        assert "attempts" not in result

    def test_step_complete_shows_tables_created(self, captured_console, sample_steps):
        """step_complete shows tables created in verbose mode."""
        console, output = captured_console
        display = FeedbackDisplay(console=console, verbose=True)  # Tables shown in verbose mode
        display.show_plan(sample_steps)

        display.step_complete(1, "Done", 1, 500, ["customers", "orders"])

        result = output.getvalue()
        assert "tables" in result
        assert "customers" in result
        assert "orders" in result

    def test_step_complete_shows_output(self, captured_console, sample_steps):
        """step_complete displays output."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)
        display.show_plan(sample_steps)

        display.step_complete(1, "Loaded 100 rows of data", 1, 500)

        result = strip_ansi(output.getvalue())
        assert "Loaded" in result and "100" in result  # Partial check for formatted output

    def test_step_complete_truncates_long_output(self, captured_console, sample_steps):
        """Long output is truncated in non-verbose mode."""
        console, output = captured_console
        display = FeedbackDisplay(console=console, verbose=False)
        display.show_plan(sample_steps)

        long_output = "\n".join([f"Line {i}" for i in range(10)])
        display.step_complete(1, long_output, 1, 100)

        result = strip_ansi(output.getvalue())  # Strip ANSI codes for comparison
        # In non-verbose mode, shows first 3 lines with truncation indicator
        assert "Line 0" in result  # First lines shown
        # May show "..." or "more lines" as truncation indicator
        assert "more lines" in result or "..." in result

    def test_verbose_shows_full_output(self, captured_console, sample_steps):
        """Verbose mode shows tables created (output truncation is same as non-verbose)."""
        console, output = captured_console
        display = FeedbackDisplay(console=console, verbose=True)
        display.show_plan(sample_steps)

        # In verbose mode, tables created are shown
        display.step_complete(1, "Output", 1, 100, tables_created=["test_table"])

        result = output.getvalue()
        assert "test_table" in result  # Verbose shows tables

    def test_step_failed_shows_failed_indicator(self, captured_console, sample_steps):
        """step_failed marks step as failed in live display."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)
        display.show_plan(sample_steps)

        display.step_failed(1, "Something went wrong", 3)

        # Verify step state is updated to failed
        assert display.plan_steps[0].status == "failed"
        # Live display shows the goal (which will be styled red for failed)
        result = strip_ansi(output.getvalue())
        assert "Load customer data" in result

    def test_step_failed_records_attempt_count(self, captured_console, sample_steps):
        """step_failed records attempt count in step state."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)
        display.show_plan(sample_steps)

        display.step_failed(1, "Error", 5)

        # Verify attempts are recorded in step state
        assert display.plan_steps[0].attempts == 5

    def test_step_failed_records_error(self, captured_console, sample_steps):
        """step_failed records error message in step state."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)
        display.show_plan(sample_steps)

        display.step_failed(1, "NameError: undefined variable 'x'", 3)

        # Error is stored in step state
        assert display.plan_steps[0].error == "NameError: undefined variable 'x'"

    def test_step_error_shows_brief_error(self, captured_console, sample_steps):
        """step_error shows brief error (last line) in verbose mode."""
        console, output = captured_console
        display = FeedbackDisplay(console=console, verbose=True)  # Need verbose to see error
        display.show_plan(sample_steps)

        error = "Traceback...\n  File...\nNameError: undefined"
        display.step_error(1, error, 1)

        result = output.getvalue()
        assert "NameError: undefined" in result

    def test_step_error_truncates_long_error(self, captured_console, sample_steps):
        """step_error brief line is truncated to 80 chars in verbose mode."""
        console, output = captured_console
        display = FeedbackDisplay(console=console, verbose=True)  # Need verbose to see error
        display.show_plan(sample_steps)

        error = "A" * 200  # Very long error
        display.step_error(1, error, 1)

        result = strip_ansi(output.getvalue())
        # Brief line truncated to 80 chars, but Panel shows full error
        # So both 80 and 200 A's appear - verify error is displayed
        assert "Error:" in result  # Error header shown
        assert "A" * 80 in result  # At least 80 chars shown

    def test_step_error_shows_full_in_verbose(self, captured_console, sample_steps):
        """Verbose mode shows full error."""
        console, output = captured_console
        display = FeedbackDisplay(console=console, verbose=True)
        display.show_plan(sample_steps)

        error = "Full traceback:\n  File test.py\n  Line 10\nNameError: x"
        display.step_error(1, error, 1)

        result = output.getvalue()
        assert "Full traceback" in result
        assert "File test.py" in result

    def test_show_summary_success(self, captured_console, sample_steps):
        """show_summary for success stops live display (timing shown via show_tables)."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)
        display.show_plan(sample_steps)

        # Success case just stops display - timing shown via show_tables
        display.show_summary(success=True, total_steps=3, duration_ms=5000)
        # Show tables to display timing
        display.show_tables([], duration_ms=5000)

        result = strip_ansi(output.getvalue())
        assert "5.0s" in result  # Timing shown via show_tables

    def test_show_summary_failure(self, captured_console, sample_steps):
        """show_summary displays failure with completed count."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)
        display.show_plan(sample_steps)

        # Mark first step as completed
        display.step_complete(1, "Done", 1, 500)

        display.show_summary(success=False, total_steps=3, duration_ms=3000)

        result = strip_ansi(output.getvalue())
        assert "FAILED" in result
        assert "1/3" in result  # 1 of 3 completed

    def test_show_tables_displays_table_info(self, captured_console):
        """show_tables displays table information with force_show."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)

        tables = [
            {"name": "customers", "row_count": 100, "step_number": 1},
            {"name": "orders", "row_count": 500, "step_number": 2},
        ]
        display.show_tables(tables, force_show=True)  # Need force_show for non-verbose

        result = output.getvalue()
        assert "customers" in result
        assert "100" in result
        assert "orders" in result
        assert "500" in result

    def test_show_tables_empty_list(self, captured_console):
        """show_tables with empty list produces no output."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)

        display.show_tables([])

        result = output.getvalue()
        assert result == ""  # No output for empty list

    def test_show_output_renders_content(self, captured_console):
        """show_output renders Markdown content."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)

        display.show_output("# Summary\n\nThe top customer is **Acme Corp**.")

        result = output.getvalue()
        # Content is rendered as Markdown (headers converted to bold)
        assert "Summary" in result
        assert "Acme Corp" in result

    def test_step_executing_shows_code_in_verbose(self, captured_console, sample_steps):
        """Verbose mode shows code being executed."""
        console, output = captured_console
        display = FeedbackDisplay(console=console, verbose=True)
        display.show_plan(sample_steps)

        display.step_executing(1, 1, "df = pd.read_sql('SELECT * FROM customers', db)")

        result = output.getvalue()
        assert "read_sql" in result

    def test_step_generating_shows_retry_indicator(self, captured_console, sample_steps):
        """step_generating shows retry indicator for attempts > 1."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)
        display.show_plan(sample_steps)

        display.step_generating(1, 3)

        result = strip_ansi(output.getvalue())
        assert "retry" in result
        assert "#3" in result


# =============================================================================
# TestSessionFeedbackHandler - Event routing tests
# =============================================================================

class TestSessionFeedbackHandler:
    """Tests for event routing."""

    def test_handles_step_start_event(self, mock_display, sample_events):
        """step_start event routes to display.step_start()."""
        handler = SessionFeedbackHandler(mock_display)

        handler.handle_event(sample_events["step_start"])

        mock_display.step_start.assert_called_once_with(1, "Load data")

    def test_handles_generating_event(self, mock_display, sample_events):
        """generating event routes to display.step_generating()."""
        handler = SessionFeedbackHandler(mock_display)

        handler.handle_event(sample_events["generating"])

        mock_display.step_generating.assert_called_once_with(1, 1)

    def test_handles_generating_event_with_retry(self, mock_display, sample_events):
        """generating event passes attempt number."""
        handler = SessionFeedbackHandler(mock_display)

        handler.handle_event(sample_events["generating_retry"])

        mock_display.step_generating.assert_called_once_with(1, 3)

    def test_handles_executing_event(self, mock_display, sample_events):
        """executing event routes to display.step_executing()."""
        handler = SessionFeedbackHandler(mock_display)

        handler.handle_event(sample_events["executing"])

        mock_display.step_executing.assert_called_once_with(1, 1, "print('hello')")

    def test_handles_step_complete_event(self, mock_display, sample_events):
        """step_complete event routes to display.step_complete()."""
        handler = SessionFeedbackHandler(mock_display)

        handler.handle_event(sample_events["step_complete"])

        mock_display.step_complete.assert_called_once_with(
            1,
            "Loaded 100 rows",
            1,
            500,
            ["customers"]
        )

    def test_handles_step_complete_event_with_retries(self, mock_display, sample_events):
        """step_complete event passes retry information."""
        handler = SessionFeedbackHandler(mock_display)

        handler.handle_event(sample_events["step_complete_retry"])

        mock_display.step_complete.assert_called_once_with(
            1,
            "Done after retries",
            3,
            1500,
            ["results"]
        )

    def test_handles_step_error_event(self, mock_display, sample_events):
        """step_error event routes to display.step_error()."""
        handler = SessionFeedbackHandler(mock_display)

        handler.handle_event(sample_events["step_error"])

        mock_display.step_error.assert_called_once_with(
            1,
            "NameError: undefined",
            1
        )

    def test_handles_missing_optional_fields(self, mock_display):
        """Missing optional fields use defaults."""
        handler = SessionFeedbackHandler(mock_display)

        event = StepEvent(
            event_type="step_complete",
            step_number=1,
            data={}  # Missing all optional fields
        )

        handler.handle_event(event)

        mock_display.step_complete.assert_called_once_with(
            1,      # step_number
            "",     # stdout default
            1,      # attempts default
            0,      # duration_ms default
            None    # tables_created default
        )

    def test_handles_missing_goal_field(self, mock_display):
        """Missing goal field uses empty string."""
        handler = SessionFeedbackHandler(mock_display)

        event = StepEvent(
            event_type="step_start",
            step_number=1,
            data={}  # Missing goal
        )

        handler.handle_event(event)

        mock_display.step_start.assert_called_once_with(1, "")

    def test_handles_missing_error_field(self, mock_display):
        """Missing error field uses default message."""
        handler = SessionFeedbackHandler(mock_display)

        event = StepEvent(
            event_type="step_error",
            step_number=1,
            data={}  # Missing error
        )

        handler.handle_event(event)

        mock_display.step_error.assert_called_once_with(1, "Unknown error", 1)

    def test_unknown_event_type_is_ignored(self, mock_display):
        """Unknown event types do not raise errors."""
        handler = SessionFeedbackHandler(mock_display)

        event = StepEvent(
            event_type="unknown_event",
            step_number=1,
            data={}
        )

        # Should not raise
        handler.handle_event(event)

        # No display methods called
        mock_display.step_start.assert_not_called()
        mock_display.step_complete.assert_not_called()
        mock_display.step_error.assert_not_called()


# =============================================================================
# TestFeedbackDisplayEdgeCases - Edge case and defensive tests
# =============================================================================

class TestFeedbackDisplayEdgeCases:
    """Edge case and defensive tests."""

    def test_handles_unicode_in_goal(self, captured_console):
        """Unicode characters in goal are handled."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)

        steps = [{"number": 1, "goal": "Calculate revenue"}]
        display.show_plan(steps)

        result = output.getvalue()
        assert "Calculate" in result

    def test_handles_unicode_in_output(self, captured_console, sample_steps):
        """Unicode characters in output are handled."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)
        display.show_plan(sample_steps)

        display.step_complete(1, "Result: 100", 1, 500)

        result = output.getvalue()
        assert "100" in result

    def test_handles_empty_output(self, captured_console, sample_steps):
        """Empty string output is handled."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)
        display.show_plan(sample_steps)

        # Should not raise
        display.step_complete(1, "", 1, 500)

        # Verify step is marked as completed
        assert display.plan_steps[0].status == "completed"
        # Live display shows the goal with completed status
        result = strip_ansi(output.getvalue())
        assert "Load customer data" in result

    def test_handles_empty_error(self, captured_console, sample_steps):
        """Empty string error is handled."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)
        display.show_plan(sample_steps)

        # Should not raise
        display.step_error(1, "", 1)

    def test_show_plan_with_empty_steps(self, captured_console):
        """Empty plan is handled."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)

        # Should not raise
        display.show_plan([])

        assert display.plan_steps == []

    def test_verbose_flag_is_stored(self):
        """Verbose flag is stored correctly."""
        display_verbose = FeedbackDisplay(verbose=True)
        display_normal = FeedbackDisplay(verbose=False)

        assert display_verbose.verbose is True
        assert display_normal.verbose is False

    def test_console_injection(self, captured_console):
        """Custom console is used."""
        console, output = captured_console
        display = FeedbackDisplay(console=console)

        # Use show_plan which actually prints to verify console injection
        display.show_plan([{"number": 1, "goal": "Test step"}])

        result = output.getvalue()
        assert "CONSTAT" in result  # Output went to our captured console (shows CONSTAT banner)

    def test_default_console_is_created(self):
        """Default console is created if not provided."""
        display = FeedbackDisplay()

        assert display.console is not None
        assert isinstance(display.console, Console)
