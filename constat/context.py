# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Context size estimation and compaction utilities.

Provides tools to estimate token usage of session context and
compact it when it grows too large for LLM context windows.
"""

import json
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


# Rough estimate: 1 token ≈ 4 characters for English text
CHARS_PER_TOKEN = 4

# Default thresholds
DEFAULT_WARNING_TOKENS = 50_000  # Warn when context exceeds this
DEFAULT_MAX_TOKENS = 100_000    # Suggest compaction above this
DEFAULT_MAX_TABLE_ROWS = 1000   # Sample tables larger than this
DEFAULT_MAX_NARRATIVE_CHARS = 2000  # Truncate narratives longer than this


@dataclass
class ContextStats:
    """Statistics about context size."""
    total_tokens: int
    scratchpad_tokens: int
    state_tokens: int
    table_metadata_tokens: int
    artifact_tokens: int

    # Breakdown details
    scratchpad_entries: int
    state_variables: int
    tables: int
    artifacts: int

    # Largest items
    largest_scratchpad_entry: Optional[tuple[int, int]] = None  # (step_number, tokens)
    largest_table: Optional[tuple[str, int, int]] = None  # (name, rows, tokens)
    largest_state: Optional[tuple[str, int]] = None  # (key, tokens)

    @property
    def is_warning(self) -> bool:
        """Return True if context size is in warning zone."""
        return self.total_tokens >= DEFAULT_WARNING_TOKENS

    @property
    def is_critical(self) -> bool:
        """Return True if context size is critical."""
        return self.total_tokens >= DEFAULT_MAX_TOKENS

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Context size: ~{self.total_tokens:,} tokens",
            f"  Scratchpad: {self.scratchpad_tokens:,} tokens ({self.scratchpad_entries} entries)",
            f"  State vars: {self.state_tokens:,} tokens ({self.state_variables} variables)",
            f"  Tables:     {self.table_metadata_tokens:,} tokens ({self.tables} tables)",
            f"  Artifacts:  {self.artifact_tokens:,} tokens ({self.artifacts} artifacts)",
        ]

        if self.largest_scratchpad_entry:
            step, tokens = self.largest_scratchpad_entry
            lines.append(f"  Largest scratchpad: Step {step} ({tokens:,} tokens)")

        if self.largest_table:
            name, rows, tokens = self.largest_table
            lines.append(f"  Largest table: {name} ({rows:,} rows, {tokens:,} tokens)")

        if self.largest_state:
            key, tokens = self.largest_state
            lines.append(f"  Largest state var: {key} ({tokens:,} tokens)")

        return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string."""
    if not text:
        return 0
    return len(text) // CHARS_PER_TOKEN


def estimate_json_tokens(obj: Any) -> int:
    """Estimate token count for a JSON-serializable object."""
    try:
        return estimate_tokens(json.dumps(obj))
    except (TypeError, ValueError):
        return estimate_tokens(str(obj))


def estimate_dataframe_tokens(df: pd.DataFrame, sample_size: int = 10) -> int:
    """
    Estimate tokens for a DataFrame's representation in context.

    We don't include full data, just schema and sample rows.
    """
    if df is None or df.empty:
        return 0

    # Schema info
    schema_str = f"Columns: {', '.join(df.columns)}\n"
    schema_str += f"Shape: {df.shape}\n"

    # Sample rows
    sample = df.head(sample_size)
    sample_str = sample.to_string()

    return estimate_tokens(schema_str + sample_str)


class ContextEstimator:
    """Estimates context size from datastore contents."""

    def __init__(self, datastore):
        """
        Initialize with a DataStore instance.

        Args:
            datastore: DataStore instance to analyze
        """
        self.datastore = datastore

    def estimate(self) -> ContextStats:
        """
        Estimate total context size.

        Returns:
            ContextStats with detailed breakdown
        """
        # Scratchpad analysis
        scratchpad = self.datastore.get_scratchpad()
        scratchpad_tokens = 0
        largest_scratchpad = None

        for entry in scratchpad:
            entry_text = f"## Step {entry['step_number']}: {entry['goal']}\n{entry['narrative']}"
            if entry.get('code'):
                entry_text += f"\n```python\n{entry['code']}\n```"

            tokens = estimate_tokens(entry_text)
            scratchpad_tokens += tokens

            if largest_scratchpad is None or tokens > largest_scratchpad[1]:
                largest_scratchpad = (entry['step_number'], tokens)

        # State variables
        state = self.datastore.get_all_state()
        state_tokens = 0
        largest_state = None

        for key, value in state.items():
            tokens = estimate_json_tokens({key: value})
            state_tokens += tokens

            if largest_state is None or tokens > largest_state[1]:
                largest_state = (key, tokens)

        # Tables metadata (we include schema info in context)
        tables = self.datastore.list_tables()
        table_tokens = 0
        largest_table = None

        for table in tables:
            # Schema info
            schema = self.datastore.get_table_schema(table['name'])
            schema_text = f"Table: {table['name']} ({table['row_count']} rows)\n"
            if schema:
                schema_text += "Columns: " + ", ".join(
                    f"{c['name']} ({c['type']})" for c in schema
                )

            tokens = estimate_tokens(schema_text)
            table_tokens += tokens

            if largest_table is None or table['row_count'] > largest_table[1]:
                largest_table = (table['name'], table['row_count'], tokens)

        # Artifacts (we include metadata, not full content in prompts)
        artifacts = self.datastore.list_artifacts(include_content=False)
        artifact_tokens = 0

        for artifact in artifacts:
            artifact_text = f"Artifact: {artifact['name']} ({artifact['type']})"
            if artifact.get('title'):
                artifact_text += f" - {artifact['title']}"
            artifact_tokens += estimate_tokens(artifact_text)

        total = scratchpad_tokens + state_tokens + table_tokens + artifact_tokens

        return ContextStats(
            total_tokens=total,
            scratchpad_tokens=scratchpad_tokens,
            state_tokens=state_tokens,
            table_metadata_tokens=table_tokens,
            artifact_tokens=artifact_tokens,
            scratchpad_entries=len(scratchpad),
            state_variables=len(state),
            tables=len(tables),
            artifacts=len(artifacts),
            largest_scratchpad_entry=largest_scratchpad,
            largest_table=largest_table,
            largest_state=largest_state,
        )


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    success: bool
    message: str
    tokens_before: int
    tokens_after: int

    # What was compacted
    scratchpad_entries_summarized: int = 0
    tables_sampled: int = 0
    state_vars_cleared: int = 0
    artifacts_cleared: int = 0

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [self.message]

        if self.tokens_saved > 0:
            lines.append(f"Tokens saved: ~{self.tokens_saved:,}")
            lines.append(f"  Before: {self.tokens_before:,}")
            lines.append(f"  After:  {self.tokens_after:,}")

        actions = []
        if self.scratchpad_entries_summarized > 0:
            actions.append(f"Summarized {self.scratchpad_entries_summarized} scratchpad entries")
        if self.tables_sampled > 0:
            actions.append(f"Sampled {self.tables_sampled} large tables")
        if self.state_vars_cleared > 0:
            actions.append(f"Cleared {self.state_vars_cleared} state variables")
        if self.artifacts_cleared > 0:
            actions.append(f"Cleared {self.artifacts_cleared} artifacts")

        if actions:
            lines.append("Actions taken:")
            for action in actions:
                lines.append(f"  - {action}")

        return "\n".join(lines)


class ContextCompactor:
    """Compacts context by summarizing, sampling, and clearing old data."""

    def __init__(
        self,
        datastore,
        llm_provider=None,
        max_narrative_chars: int = DEFAULT_MAX_NARRATIVE_CHARS,
        max_table_rows: int = DEFAULT_MAX_TABLE_ROWS,
    ):
        """
        Initialize compactor.

        Args:
            datastore: DataStore instance to compact
            llm_provider: Optional LLM provider for summarization
            max_narrative_chars: Max chars for scratchpad narratives
            max_table_rows: Max rows to keep in tables
        """
        self.datastore = datastore
        self.llm = llm_provider
        self.max_narrative_chars = max_narrative_chars
        self.max_table_rows = max_table_rows

    def compact(
        self,
        summarize_scratchpad: bool = True,
        sample_tables: bool = True,
        clear_old_state: bool = False,
        keep_recent_steps: int = 3,
    ) -> CompactionResult:
        """
        Compact the context to reduce token usage.

        Args:
            summarize_scratchpad: Truncate old scratchpad narratives
            sample_tables: Sample large tables
            clear_old_state: Clear state vars from old steps
            keep_recent_steps: Number of recent steps to keep intact

        Returns:
            CompactionResult with details
        """
        estimator = ContextEstimator(self.datastore)
        tokens_before = estimator.estimate().total_tokens

        entries_summarized = 0
        tables_sampled = 0
        state_cleared = 0

        # Get current step count
        scratchpad = self.datastore.get_scratchpad()
        max_step = max((e['step_number'] for e in scratchpad), default=0)
        cutoff_step = max(1, max_step - keep_recent_steps)

        # Summarize old scratchpad entries
        if summarize_scratchpad:
            for entry in scratchpad:
                if entry['step_number'] < cutoff_step:
                    narrative = entry.get('narrative', '')
                    if len(narrative) > self.max_narrative_chars:
                        # Truncate the narrative
                        truncated = narrative[:self.max_narrative_chars] + "\n\n[... truncated for context ...]"

                        # Update in datastore
                        self.datastore.add_scratchpad_entry(
                            step_number=entry['step_number'],
                            goal=entry['goal'],
                            narrative=truncated,
                            tables_created=entry.get('tables_created'),
                            code=None,  # Clear code from old steps
                        )
                        entries_summarized += 1

        # Sample large tables
        if sample_tables:
            tables = self.datastore.list_tables()
            for table in tables:
                if table['row_count'] > self.max_table_rows:
                    df = self.datastore.load_dataframe(table['name'])
                    if df is not None:
                        # Sample down to max_table_rows
                        sampled = df.sample(n=min(self.max_table_rows, len(df)), random_state=42)
                        self.datastore.save_dataframe(
                            table['name'],
                            sampled,
                            step_number=table.get('step_number', 0),
                            description=f"{table.get('description', '')} [sampled from {table['row_count']} rows]",
                        )
                        tables_sampled += 1

        # Clear old state variables
        if clear_old_state:
            # Get state with step numbers
            with self.datastore.engine.connect() as conn:
                from sqlalchemy import text
                rows = conn.execute(
                    text("SELECT key, step_number FROM _constat_state WHERE step_number < :cutoff"),
                    {"cutoff": cutoff_step}
                ).fetchall()

            for key, _ in rows:
                with self.datastore.engine.begin() as conn:
                    from sqlalchemy import text
                    conn.execute(
                        text("DELETE FROM _constat_state WHERE key = :key"),
                        {"key": key}
                    )
                    state_cleared += 1

        # Estimate tokens after
        tokens_after = estimator.estimate().total_tokens

        return CompactionResult(
            success=True,
            message="Context compacted successfully",
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            scratchpad_entries_summarized=entries_summarized,
            tables_sampled=tables_sampled,
            state_vars_cleared=state_cleared,
        )

    def clear_all(self) -> CompactionResult:
        """
        Clear all context (full reset).

        Use with caution - this removes all session state.
        """
        estimator = ContextEstimator(self.datastore)
        tokens_before = estimator.estimate().total_tokens

        scratchpad = self.datastore.get_scratchpad()
        tables = self.datastore.list_tables()
        state = self.datastore.get_all_state()
        artifacts = self.datastore.list_artifacts()

        # Clear scratchpad
        with self.datastore.engine.begin() as conn:
            from sqlalchemy import text
            conn.execute(text("DELETE FROM _constat_scratchpad"))

        # Clear state
        with self.datastore.engine.begin() as conn:
            from sqlalchemy import text
            conn.execute(text("DELETE FROM _constat_state"))

        # Drop all user tables
        for table in tables:
            self.datastore.drop_table(table['name'])

        # Clear artifacts
        with self.datastore.engine.begin() as conn:
            from sqlalchemy import text
            conn.execute(text("DELETE FROM _constat_artifacts"))

        # Clear plan steps
        with self.datastore.engine.begin() as conn:
            from sqlalchemy import text
            conn.execute(text("DELETE FROM _constat_plan_steps"))

        return CompactionResult(
            success=True,
            message="All context cleared",
            tokens_before=tokens_before,
            tokens_after=0,
            scratchpad_entries_summarized=len(scratchpad),
            tables_sampled=len(tables),
            state_vars_cleared=len(state),
            artifacts_cleared=len(artifacts),
        )
