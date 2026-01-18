"""Prompt logging for tracking LLM prompt sizes and costs.

This module provides instrumentation for analyzing prompt design efficiency.
It logs prompt characteristics to help identify bloated or sub-optimal prompts.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class PromptLogEntry:
    """A single prompt log entry."""

    timestamp: str
    task_type: str
    model: str
    provider: str

    # Size metrics
    system_prompt_chars: int
    user_message_chars: int
    total_chars: int

    # Estimated tokens (chars / 4 is rough approximation)
    estimated_tokens: int

    # Actual tokens if available from provider
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # Additional context
    query_preview: str = ""  # First 100 chars of user message
    concepts_detected: list[str] = field(default_factory=list)
    injected_sections: int = 0

    # Full prompt text (for debugging)
    system_prompt_full: str = ""
    user_message_full: str = ""

    # Performance
    response_time_ms: Optional[int] = None
    success: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PromptLogger:
    """Logger for tracking prompt sizes and characteristics.

    Writes to a JSONL file for easy analysis with tools like DuckDB, jq, or pandas.

    Usage:
        logger = PromptLogger()

        # Log a prompt
        logger.log(
            task_type="sql_generation",
            model="claude-3-5-sonnet-20241022",
            provider="anthropic",
            system_prompt=system_prompt,
            user_message=user_message,
            concepts_detected=["api_filtering", "visualization"],
        )

        # Get statistics
        stats = logger.get_stats()
        print(f"Average prompt size: {stats['avg_total_chars']}")

        # Find bloated prompts
        bloated = logger.find_large_prompts(threshold_chars=10000)
    """

    DEFAULT_LOG_PATH = Path.cwd() / ".constat" / "prompt_log.jsonl"

    def __init__(self, log_path: Optional[Path] = None, enabled: bool = True):
        """Initialize the prompt logger.

        Args:
            log_path: Path to the log file. Defaults to ~/.constat/prompt_log.jsonl
            enabled: Whether logging is enabled. Disable for tests.
        """
        self.log_path = log_path or self.DEFAULT_LOG_PATH
        self.enabled = enabled

        # Ensure directory exists
        if self.enabled:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        task_type: str,
        model: str,
        provider: str,
        system_prompt: str,
        user_message: str,
        concepts_detected: Optional[list[str]] = None,
        injected_sections: int = 0,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        response_time_ms: Optional[int] = None,
        success: bool = True,
    ) -> PromptLogEntry:
        """Log a prompt execution.

        Args:
            task_type: Type of task (planning, sql_generation, etc.)
            model: Model used (e.g., claude-3-5-sonnet-20241022)
            provider: Provider name (anthropic, openai, etc.)
            system_prompt: The full system prompt sent
            user_message: The user message sent
            concepts_detected: List of concepts detected by ConceptDetector
            injected_sections: Number of prompt sections injected
            input_tokens: Actual input tokens from provider response
            output_tokens: Actual output tokens from provider response
            response_time_ms: Response time in milliseconds
            success: Whether the call succeeded

        Returns:
            The logged entry
        """
        system_chars = len(system_prompt)
        user_chars = len(user_message)
        total_chars = system_chars + user_chars

        entry = PromptLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_type=task_type,
            model=model,
            provider=provider,
            system_prompt_chars=system_chars,
            user_message_chars=user_chars,
            total_chars=total_chars,
            estimated_tokens=total_chars // 4,  # Rough estimate
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            query_preview=user_message[:100].replace("\n", " "),
            concepts_detected=concepts_detected or [],
            injected_sections=injected_sections,
            system_prompt_full=system_prompt,
            user_message_full=user_message,
            response_time_ms=response_time_ms,
            success=success,
        )

        if self.enabled:
            self._write_entry(entry)

        return entry

    def _write_entry(self, entry: PromptLogEntry) -> None:
        """Write an entry to the log file."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def get_stats(self, last_n: Optional[int] = None) -> dict:
        """Get statistics from the log.

        Args:
            last_n: Only consider last N entries. None for all.

        Returns:
            Dictionary with statistics
        """
        entries = self._read_entries(last_n)
        if not entries:
            return {
                "count": 0,
                "avg_total_chars": 0,
                "avg_estimated_tokens": 0,
                "max_total_chars": 0,
                "min_total_chars": 0,
            }

        total_chars = [e["total_chars"] for e in entries]
        estimated_tokens = [e["estimated_tokens"] for e in entries]

        return {
            "count": len(entries),
            "avg_total_chars": sum(total_chars) / len(total_chars),
            "avg_estimated_tokens": sum(estimated_tokens) / len(estimated_tokens),
            "max_total_chars": max(total_chars),
            "min_total_chars": min(total_chars),
            "by_task_type": self._stats_by_field(entries, "task_type"),
        }

    def _stats_by_field(self, entries: list[dict], field: str) -> dict:
        """Get average chars by a grouping field."""
        groups: dict[str, list[int]] = {}
        for e in entries:
            key = e.get(field, "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(e["total_chars"])

        return {k: sum(v) / len(v) for k, v in groups.items()}

    def find_large_prompts(
        self,
        threshold_chars: int = 10000,
        last_n: Optional[int] = 100,
    ) -> list[dict]:
        """Find prompts exceeding a size threshold.

        Args:
            threshold_chars: Character threshold for "large" prompts
            last_n: Only consider last N entries

        Returns:
            List of entries exceeding threshold
        """
        entries = self._read_entries(last_n)
        return [e for e in entries if e["total_chars"] > threshold_chars]

    def _read_entries(self, last_n: Optional[int] = None) -> list[dict]:
        """Read entries from the log file."""
        if not self.log_path.exists():
            return []

        entries = []
        with open(self.log_path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        if last_n:
            entries = entries[-last_n:]

        return entries


# Global logger instance (can be disabled for tests)
_prompt_logger: Optional[PromptLogger] = None


def get_prompt_logger() -> PromptLogger:
    """Get the global prompt logger instance."""
    global _prompt_logger
    if _prompt_logger is None:
        _prompt_logger = PromptLogger()
    return _prompt_logger


def log_prompt(
    task_type: str,
    model: str,
    provider: str,
    system_prompt: str,
    user_message: str,
    **kwargs,
) -> PromptLogEntry:
    """Convenience function to log a prompt using the global logger."""
    return get_prompt_logger().log(
        task_type=task_type,
        model=model,
        provider=provider,
        system_prompt=system_prompt,
        user_message=user_message,
        **kwargs,
    )
