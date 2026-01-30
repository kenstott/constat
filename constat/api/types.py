# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Frozen dataclasses for ConstatAPI return types.

All types are immutable value objects to ensure clean API boundaries.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


# -----------------------------------------------------------------------------
# Detection Results
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DisplayOverrides:
    """Display preference overrides detected from natural language."""
    persistent: dict[str, bool] = field(default_factory=dict)
    single_turn: dict[str, bool] = field(default_factory=dict)

    @property
    def has_overrides(self) -> bool:
        """Return True if any overrides were detected."""
        return bool(self.persistent or self.single_turn)


@dataclass(frozen=True)
class CorrectionDetection:
    """Result of natural language correction detection."""
    detected: bool
    correction_type: Optional[str] = None
    matched_text: Optional[str] = None


# -----------------------------------------------------------------------------
# Fact Types
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Fact:
    """A persistent fact stored for the user."""
    name: str
    value: Any
    description: str
    context: str
    created: datetime


# -----------------------------------------------------------------------------
# Learning Types
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Learning:
    """A raw learning/correction captured from user interaction."""
    id: str
    category: str
    correction: str
    context: dict[str, Any]
    source: str
    created: datetime
    applied_count: int
    promoted_to: Optional[str] = None


@dataclass(frozen=True)
class Rule:
    """A compacted rule derived from multiple learnings."""
    id: str
    category: str
    summary: str
    confidence: float
    source_learnings: tuple[str, ...]
    tags: tuple[str, ...]
    created: datetime
    applied_count: int


@dataclass(frozen=True)
class LearningCompactionResult:
    """Result of learning compaction operation."""
    rules_created: int
    rules_strengthened: int
    rules_merged: int
    learnings_archived: int
    learnings_expired: int
    groups_found: int
    skipped_low_confidence: int
    errors: tuple[str, ...]


# -----------------------------------------------------------------------------
# Plan Types
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SavedPlan:
    """A saved execution plan."""
    plan_id: str
    name: str
    description: str
    problem: str
    created: datetime
    tags: tuple[str, ...]


# -----------------------------------------------------------------------------
# Session State Types
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionState:
    """Current session state snapshot."""
    session_id: Optional[str]
    has_plan: bool
    plan_goal: Optional[str]
    plan_steps: int
    completed_steps: int
    tables: tuple[str, ...]
    facts_count: int
    mode: Optional[str]


@dataclass(frozen=True)
class ContextStats:
    """Statistics about context token usage."""
    total_tokens: int
    scratchpad_tokens: int
    state_tokens: int
    table_metadata_tokens: int
    artifact_tokens: int
    scratchpad_entries: int
    state_variables: int
    tables: int
    artifacts: int
    is_warning: bool
    is_critical: bool


@dataclass(frozen=True)
class ContextCompactionResult:
    """Result of context compaction operation."""
    original_tokens: int
    compacted_tokens: int
    tokens_saved: int
    entries_removed: int
    tables_sampled: int


# -----------------------------------------------------------------------------
# Summarization Results
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SummarizeResult:
    """Result of a summarization operation."""
    success: bool
    summary: Optional[str] = None
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# Core Operation Results
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class StepInfo:
    """Information about a plan step."""
    number: int
    description: str
    status: str
    code: Optional[str] = None


@dataclass(frozen=True)
class ArtifactInfo:
    """Information about a generated artifact."""
    id: str
    name: str
    artifact_type: str
    step_number: int


@dataclass(frozen=True)
class SolveResult:
    """Result of a solve() operation."""
    success: bool
    answer: Optional[str] = None
    plan_goal: Optional[str] = None
    steps: tuple[StepInfo, ...] = ()
    artifacts: tuple[ArtifactInfo, ...] = ()
    tables_created: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()
    error: Optional[str] = None
    raw_output: Optional[str] = None


@dataclass(frozen=True)
class FollowUpResult:
    """Result of a follow_up() operation."""
    success: bool
    answer: Optional[str] = None
    steps: tuple[StepInfo, ...] = ()
    artifacts: tuple[ArtifactInfo, ...] = ()
    tables_created: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()
    error: Optional[str] = None
    raw_output: Optional[str] = None


@dataclass(frozen=True)
class ResumeResult:
    """Result of a resume() operation."""
    success: bool
    session_id: Optional[str] = None
    plan_goal: Optional[str] = None
    completed_steps: int = 0
    total_steps: int = 0
    error: Optional[str] = None


@dataclass(frozen=True)
class ReplayResult:
    """Result of a replay() operation."""
    success: bool
    steps_executed: int = 0
    total_steps: int = 0
    artifacts: tuple[ArtifactInfo, ...] = ()
    error: Optional[str] = None
