"""Tests for the prompt logger module."""

import json
import tempfile
from pathlib import Path

import pytest

from constat.providers.prompt_logger import PromptLogger, PromptLogEntry


class TestPromptLogEntry:
    """Test PromptLogEntry dataclass."""

    def test_to_dict(self):
        """Entry should serialize to dict correctly."""
        entry = PromptLogEntry(
            timestamp="2025-01-16T12:00:00",
            task_type="sql_generation",
            model="claude-3-5-sonnet",
            provider="anthropic",
            system_prompt_chars=1000,
            user_message_chars=500,
            total_chars=1500,
            estimated_tokens=375,
        )

        d = entry.to_dict()
        assert d["task_type"] == "sql_generation"
        assert d["total_chars"] == 1500
        assert d["estimated_tokens"] == 375


class TestPromptLogger:
    """Test the PromptLogger class."""

    @pytest.fixture
    def temp_log_path(self):
        """Create a temporary log file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_prompt_log.jsonl"

    def test_log_creates_entry(self, temp_log_path):
        """Logging should create an entry."""
        logger = PromptLogger(log_path=temp_log_path)

        entry = logger.log(
            task_type="planning",
            model="claude-3-5-sonnet",
            provider="anthropic",
            system_prompt="You are a planner.",
            user_message="Plan this query: show sales",
        )

        assert entry.task_type == "planning"
        assert entry.system_prompt_chars == len("You are a planner.")
        assert entry.user_message_chars == len("Plan this query: show sales")
        assert entry.estimated_tokens > 0

    def test_log_writes_to_file(self, temp_log_path):
        """Logged entries should be written to file."""
        logger = PromptLogger(log_path=temp_log_path)

        logger.log(
            task_type="sql_generation",
            model="claude-3-5-sonnet",
            provider="anthropic",
            system_prompt="Generate SQL",
            user_message="Get customers",
        )

        assert temp_log_path.exists()
        with open(temp_log_path) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["task_type"] == "sql_generation"

    def test_get_stats(self, temp_log_path):
        """Should compute statistics from log."""
        logger = PromptLogger(log_path=temp_log_path)

        # Log multiple entries
        logger.log(
            task_type="planning",
            model="claude-3-5-sonnet",
            provider="anthropic",
            system_prompt="A" * 1000,
            user_message="B" * 500,
        )
        logger.log(
            task_type="sql_generation",
            model="claude-3-5-sonnet",
            provider="anthropic",
            system_prompt="A" * 2000,
            user_message="B" * 1000,
        )

        stats = logger.get_stats()
        assert stats["count"] == 2
        assert stats["avg_total_chars"] == (1500 + 3000) / 2
        assert stats["max_total_chars"] == 3000
        assert stats["min_total_chars"] == 1500

    def test_find_large_prompts(self, temp_log_path):
        """Should find prompts exceeding threshold."""
        logger = PromptLogger(log_path=temp_log_path)

        # Small prompt
        logger.log(
            task_type="planning",
            model="claude-3-5-sonnet",
            provider="anthropic",
            system_prompt="Small",
            user_message="Query",
        )
        # Large prompt
        logger.log(
            task_type="sql_generation",
            model="claude-3-5-sonnet",
            provider="anthropic",
            system_prompt="A" * 10000,
            user_message="B" * 5000,
        )

        large = logger.find_large_prompts(threshold_chars=1000)
        assert len(large) == 1
        assert large[0]["total_chars"] == 15000

    def test_disabled_logger(self, temp_log_path):
        """Disabled logger should not write to file."""
        logger = PromptLogger(log_path=temp_log_path, enabled=False)

        logger.log(
            task_type="planning",
            model="claude-3-5-sonnet",
            provider="anthropic",
            system_prompt="Test",
            user_message="Query",
        )

        assert not temp_log_path.exists()

    def test_concepts_detected_logged(self, temp_log_path):
        """Should log detected concepts."""
        logger = PromptLogger(log_path=temp_log_path)

        logger.log(
            task_type="step",
            model="claude-3-5-sonnet",
            provider="anthropic",
            system_prompt="Execute step",
            user_message="Create a dashboard",
            concepts_detected=["dashboard_layout", "visualization"],
            injected_sections=2,
        )

        with open(temp_log_path) as f:
            data = json.loads(f.readline())
            assert data["concepts_detected"] == ["dashboard_layout", "visualization"]
            assert data["injected_sections"] == 2
