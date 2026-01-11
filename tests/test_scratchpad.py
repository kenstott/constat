"""Tests for Scratchpad."""

import pytest
from constat.execution.scratchpad import Scratchpad


class TestScratchpad:
    """Tests for Scratchpad state management."""

    def test_create_with_context(self):
        """Test creating scratchpad with initial context."""
        pad = Scratchpad(initial_context="This is the problem statement")
        assert "problem statement" in pad.get_context()

    def test_create_empty(self):
        """Test creating empty scratchpad."""
        pad = Scratchpad()
        assert pad.get_context() == ""

    def test_add_step_result(self):
        """Test adding step results."""
        pad = Scratchpad()
        pad.add_step_result(
            step_number=1,
            goal="Load data",
            result="Loaded 100 rows from customers table"
        )

        result = pad.get_step_result(1)
        assert result is not None
        assert "100 rows" in result

    def test_add_step_result_with_tables(self):
        """Test step result with tables created."""
        pad = Scratchpad()
        pad.add_step_result(
            step_number=1,
            goal="Load data",
            result="Loaded data",
            tables_created=["customers", "orders"]
        )

        result = pad.get_step_result(1)
        assert "customers" in result
        assert "orders" in result

    def test_add_note(self):
        """Test adding notes."""
        pad = Scratchpad()
        pad.add_note("This is important")
        pad.add_note("This is also important")

        markdown = pad.to_markdown()
        assert "## Notes" in markdown
        assert "This is important" in markdown
        assert "This is also important" in markdown

    def test_to_markdown(self):
        """Test exporting to markdown."""
        pad = Scratchpad(initial_context="Problem: Analyze sales")
        pad.add_step_result(1, "Load data", "Loaded 50 rows")
        pad.add_step_result(2, "Calculate totals", "Total: $1000")

        markdown = pad.to_markdown()

        assert "## Context" in markdown
        assert "## Step 1: Load data" in markdown
        assert "## Step 2: Calculate totals" in markdown

    def test_from_markdown(self):
        """Test loading from markdown."""
        markdown = """## Context
Problem: Analyze sales

## Step 1: Load data
Loaded 50 rows

## Step 2: Calculate totals
Total: $1000

## Notes
Important observation"""

        pad = Scratchpad.from_markdown(markdown)

        assert "Analyze sales" in pad.get_context()
        assert "50 rows" in pad.get_step_result(1)
        assert "$1000" in pad.get_step_result(2)

    def test_get_recent_context(self):
        """Test getting recent context for LLM."""
        pad = Scratchpad(initial_context="Problem")

        # Add many steps
        for i in range(10):
            pad.add_step_result(i + 1, f"Step {i + 1}", f"Result {i + 1}")

        # Get recent context with limit
        recent = pad.get_recent_context(max_steps=3)

        # Should include context
        assert "## Context" in recent

        # Should include last 3 steps
        assert "Step 8" in recent
        assert "Step 9" in recent
        assert "Step 10" in recent

        # Should NOT include earlier steps
        assert "Step 1:" not in recent
        assert "Step 5:" not in recent

    def test_str_representation(self):
        """Test string representation."""
        pad = Scratchpad(initial_context="Test")
        assert str(pad) == "## Context\nTest"

    def test_repr(self):
        """Test repr."""
        pad = Scratchpad(initial_context="Test")
        pad.add_step_result(1, "Step 1", "Result")
        assert "2 sections" in repr(pad)
