# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Execution phase and plan approval handling.

All queries run in exploratory mode by default (fast, conversational, best-effort).
Use the /prove command to verify claims from the conversation with auditable proofs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Mode(Enum):
    """Execution mode - kept for backwards compatibility.

    Note: Mode selection is no longer used. All queries run exploratory by default.
    Use /prove to generate auditable proofs when needed.
    """

    PROOF = "proof"
    """Fact resolver for traceable, defensible conclusions."""

    EXPLORATORY = "exploratory"
    """Multi-step planner for exploration and analysis."""


class Phase(Enum):
    """Task lifecycle state tracking phase transitions."""

    IDLE = "idle"
    """No active plan, waiting for user input."""

    PLANNING = "planning"
    """Generating or revising a plan."""

    AWAITING_APPROVAL = "awaiting_approval"
    """Plan generated, waiting for user approval."""

    EXECUTING = "executing"
    """Plan approved, execution in progress."""

    FAILED = "failed"
    """Execution failed, awaiting user decision (retry/replan/abandon)."""


class PrimaryIntent(Enum):
    """Primary intent determines the code path for a user turn."""

    QUERY = "query"
    """Answer from knowledge or current context. No approval required."""

    PLAN_NEW = "plan_new"
    """Start planning a new task. Requires approval before execution."""

    PLAN_CONTINUE = "plan_continue"
    """Refine or extend the active plan. No approval required (still planning)."""

    CONTROL = "control"
    """System/session commands. No approval required."""


class SubIntent(Enum):
    """Sub-intent refines behavior within a primary intent handler."""

    # Query sub-intents
    DETAIL = "detail"
    """Drill down, explain further."""

    PROVENANCE = "provenance"
    """Show proof chain / how we got here."""

    SUMMARY = "summary"
    """Condense results."""

    LOOKUP = "lookup"
    """Simple fact retrieval."""

    # Plan new sub-intents
    COMPARE = "compare"
    """Evaluate alternatives."""

    PREDICT = "predict"
    """What-if / forecast."""

    # Plan continue sub-intents
    CORRECTION = "correction"
    """User providing a reusable correction/rule (always use X, never do Y)."""

    # Control sub-intents (session management)
    RESET = "reset"
    """Clear session state (/reset)."""

    REDO_CMD = "redo_cmd"
    """Re-execute last plan (/redo)."""

    HELP = "help"
    """Show available commands (/help)."""

    STATUS = "status"
    """Show current state (/status)."""

    EXIT = "exit"
    """End session (/exit, /quit)."""

    # Control sub-intents (execution management)
    CANCEL = "cancel"
    """Stop execution entirely."""

    REPLAN = "replan"
    """Stop and revise the plan."""

    PROVE = "prove"
    """Generate proof for conversation claims (/prove)."""


@dataclass
class TurnIntent:
    """Classified intent for a single user turn."""

    primary: PrimaryIntent
    """Primary intent determines the code path."""

    sub: Optional[SubIntent] = None
    """Optional sub-intent refines behavior within the primary handler."""

    target: Optional[str] = None
    """What to modify, drill into, etc."""


@dataclass
class ConversationState:
    """Combined state tracking for conversation flow."""

    phase: Phase
    """Current task lifecycle phase."""

    active_plan: Optional[Any] = None
    """Current plan if any (Plan object when available)."""

    session_facts: dict[str, Any] = field(default_factory=dict)
    """Accumulated facts from session."""

    failure_context: Optional[str] = None
    """Error details when phase == FAILED."""

    def can_execute(self) -> bool:
        """Check if we can transition to executing."""
        return self.phase == Phase.AWAITING_APPROVAL

    def can_retry(self) -> bool:
        """Check if we can retry execution."""
        return self.phase == Phase.FAILED

    def is_planning(self) -> bool:
        """Check if we're in a planning-related phase."""
        return self.phase in (Phase.PLANNING, Phase.AWAITING_APPROVAL)


@dataclass
class ExecutionConfig:
    """Configuration for execution behavior."""

    # Proof verification settings
    min_confidence: float = 0.0  # Minimum confidence to accept a fact
    require_provenance: bool = True  # All facts must have source
    max_resolution_depth: int = 5  # Max recursive fact resolution

    # Step execution settings
    max_steps: int = 20  # Maximum plan steps
    step_timeout_seconds: int = 60  # Per-step timeout


class PlanApproval(Enum):
    """User response to plan approval request."""

    APPROVE = "approve"
    """Execute the plan as-is."""

    REJECT = "reject"
    """Cancel execution, do not proceed."""

    SUGGEST = "suggest"
    """User wants to suggest changes before execution."""

    COMMAND = "command"
    """User entered a slash command - pass back to REPL for handling."""


@dataclass
class PlanApprovalRequest:
    """Request for user approval of a generated plan."""
    problem: str
    steps: list[dict]
    reasoning: str

    def format_for_display(self) -> str:
        """Format the approval request for display."""
        lines = ["Plan:"]
        for step in self.steps:
            lines.append(f"  {step.get('number', '?')}. {step.get('goal', 'Unknown')}")

        if self.reasoning:
            lines.extend(["", "Reasoning:", f"  {self.reasoning}"])

        return "\n".join(lines)


@dataclass
class PlanApprovalResponse:
    """User's response to plan approval request."""
    decision: PlanApproval
    suggestion: Optional[str] = None
    reason: Optional[str] = None
    command: Optional[str] = None
    deleted_steps: Optional[list[int]] = None
    edited_steps: Optional[list[dict]] = None  # List of {"number": int, "goal": str}

    @classmethod
    def approve(cls, deleted_steps: Optional[list[int]] = None) -> "PlanApprovalResponse":
        """Create an approval response."""
        return cls(decision=PlanApproval.APPROVE, deleted_steps=deleted_steps)

    @classmethod
    def reject(cls, reason: Optional[str] = None) -> "PlanApprovalResponse":
        """Create a rejection response."""
        return cls(decision=PlanApproval.REJECT, reason=reason)

    @classmethod
    def suggest(cls, suggestion: str, edited_steps: Optional[list[dict]] = None) -> "PlanApprovalResponse":
        """Create a suggestion response with optional edited plan."""
        return cls(decision=PlanApproval.SUGGEST, suggestion=suggestion, edited_steps=edited_steps)

    @classmethod
    def pass_command(cls, command: str) -> "PlanApprovalResponse":
        """Create a command pass-through response."""
        return cls(decision=PlanApproval.COMMAND, command=command)
