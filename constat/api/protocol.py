# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Protocol definition for ConstatAPI.

Defines the public API contract that REPL/UI can import from.
"""

from typing import Callable, Optional, Protocol, Any

from constat.api.types import (
    CorrectionDetection,
    ContextCompactionResult,
    ContextStats,
    DisplayOverrides,
    Fact,
    FollowUpResult,
    Learning,
    LearningCompactionResult,
    ReplayResult,
    ResumeResult,
    Rule,
    SessionState,
    SolveResult,
    SummarizeResult,
)
from constat.execution.mode import PlanApprovalRequest, PlanApprovalResponse

# Type alias for event callbacks
EventCallback = Callable[[str, dict[str, Any]], None]


class ConstatAPI(Protocol):
    """Protocol defining the public ConstatAPI interface.

    This is the clean boundary that REPL/UI should import from.
    All business logic is encapsulated behind this interface.
    """

    # -------------------------------------------------------------------------
    # Core Operations
    # -------------------------------------------------------------------------

    def solve(
        self,
        problem: str,
        *,
        require_approval: bool = True,
    ) -> SolveResult:
        """Solve a new problem, generating and executing a plan.

        Args:
            problem: The problem or question to solve
            require_approval: If True, wait for plan approval before execution

        Returns:
            SolveResult with answer, artifacts, and suggestions
        """
        ...

    def follow_up(self, question: str) -> FollowUpResult:
        """Ask a follow-up question in the current session context.

        Args:
            question: The follow-up question

        Returns:
            FollowUpResult with answer and any new artifacts
        """
        ...

    def resume(self, session_id: str) -> ResumeResult:
        """Resume a previous session.

        Args:
            session_id: ID of the session to resume

        Returns:
            ResumeResult with session state information
        """
        ...

    def replay(self, plan_id: str) -> ReplayResult:
        """Replay a saved plan.

        Args:
            plan_id: ID of the plan to replay

        Returns:
            ReplayResult with execution information
        """
        ...

    # -------------------------------------------------------------------------
    # State and Context
    # -------------------------------------------------------------------------

    def get_state(self) -> SessionState:
        """Get current session state.

        Returns:
            SessionState snapshot
        """
        ...

    def get_context_stats(self) -> ContextStats:
        """Get context token usage statistics.

        Returns:
            ContextStats with token breakdown
        """
        ...

    def compact_context(self) -> ContextCompactionResult:
        """Compact session context to reduce token usage.

        Returns:
            ContextCompactionResult with compaction details
        """
        ...

    def reset_context(self) -> None:
        """Reset session context, clearing plan and datastore."""
        ...

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def set_approval_callback(
        self,
        callback: Callable[[PlanApprovalRequest], PlanApprovalResponse],
    ) -> None:
        """Set callback for plan approval requests.

        Args:
            callback: Function to call when plan approval is needed
        """
        ...

    def on_event(self, callback: EventCallback) -> None:
        """Register callback for session events.

        Args:
            callback: Function(event_type, data) called on events
        """
        ...

    # -------------------------------------------------------------------------
    # Facts
    # -------------------------------------------------------------------------

    def get_facts(self) -> dict[str, Fact]:
        """Get all persistent facts.

        Returns:
            Dict mapping fact names to Fact objects
        """
        ...

    def remember_fact(
        self,
        name: str,
        value: Any,
        description: str = "",
        context: str = "",
    ) -> Fact:
        """Remember a persistent fact.

        Args:
            name: Fact name (snake_case recommended)
            value: Fact value
            description: Human-readable description
            context: Creation context

        Returns:
            The created Fact
        """
        ...

    def forget_fact(self, name: str) -> bool:
        """Forget a persistent fact.

        Args:
            name: Fact name to forget

        Returns:
            True if fact was deleted, False if not found
        """
        ...

    def extract_facts_from_text(self, text: str) -> list[dict[str, Any]]:
        """Extract potential facts from natural language text.

        Args:
            text: Text to analyze

        Returns:
            List of extracted fact candidates
        """
        ...

    # -------------------------------------------------------------------------
    # Learnings
    # -------------------------------------------------------------------------

    def get_learnings(
        self,
        category: Optional[str] = None,
        limit: int = 50,
    ) -> list[Learning]:
        """Get raw learnings.

        Args:
            category: Filter by category (None for all)
            limit: Maximum number to return

        Returns:
            List of Learning objects, newest first
        """
        ...

    def get_rules(
        self,
        category: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> list[Rule]:
        """Get compacted rules.

        Args:
            category: Filter by category (None for all)
            min_confidence: Minimum confidence threshold

        Returns:
            List of Rule objects
        """
        ...

    def save_correction(
        self,
        category: str,
        context: dict[str, Any],
        correction: str,
    ) -> str:
        """Save a correction as a learning.

        Args:
            category: Learning category
            context: Contextual information
            correction: The correction text

        Returns:
            Learning ID
        """
        ...

    def compact_learnings(self, dry_run: bool = False) -> LearningCompactionResult:
        """Compact learnings into rules.

        Args:
            dry_run: If True, analyze but don't create rules

        Returns:
            LearningCompactionResult with compaction details
        """
        ...

    def forget_learning(self, learning_id: str) -> bool:
        """Delete a learning.

        Args:
            learning_id: Learning ID to delete

        Returns:
            True if deleted, False if not found
        """
        ...

    # -------------------------------------------------------------------------
    # Detection
    # -------------------------------------------------------------------------

    def detect_display_overrides(self, text: str) -> DisplayOverrides:
        """Detect display preference overrides in natural language.

        Args:
            text: User input text

        Returns:
            DisplayOverrides with persistent and single-turn settings
        """
        ...

    def detect_nl_correction(self, text: str) -> CorrectionDetection:
        """Detect if text contains a correction pattern.

        Args:
            text: User input text

        Returns:
            CorrectionDetection with type and matched text if detected
        """
        ...

    def save_nl_correction(
        self,
        original_text: str,
        correction_type: str,
        matched_text: str,
    ) -> str:
        """Save a detected natural language correction.

        Args:
            original_text: The original user input
            correction_type: Type of correction detected
            matched_text: The matched correction text

        Returns:
            Learning ID
        """
        ...

    # -------------------------------------------------------------------------
    # Summarization
    # -------------------------------------------------------------------------

    def summarize_plan(self) -> SummarizeResult:
        """Summarize the current execution plan.

        Returns:
            SummarizeResult with plan summary
        """
        ...

    def summarize_session(self) -> SummarizeResult:
        """Summarize the current session state.

        Returns:
            SummarizeResult with session summary
        """
        ...

    def summarize_facts(self) -> SummarizeResult:
        """Summarize all cached facts.

        Returns:
            SummarizeResult with facts summary
        """
        ...

    def summarize_table(self, table_name: str) -> SummarizeResult:
        """Summarize a specific table's contents.

        Args:
            table_name: Name of the table to summarize

        Returns:
            SummarizeResult with table summary
        """
        ...
