"""Execution mode selection for traceability guarantees.

Two execution modes with different purposes:

PROOF (Fact resolver):
- Generates a plan with assumed facts, resolves lazily
- Trace = formal derivation chain (proof)
- Plans must be complete and self-contained
- Every conclusion has a provenance chain
- Good for: decisions, compliance, defensible conclusions
- Audit answer: "X is true BECAUSE Y AND Z, where Y came from..."

EXPLORATORY (Multi-step planner):
- Generates a plan, executes steps sequentially
- Trace = narrative log of what was done
- Plans can reference facts/data from previous plans
- Session builds up a working context
- Good for: dashboards, reports, data exploration
- Audit answer: "I ran these queries and computed this"

The LLM can suggest a mode based on the query, or the user can specify.
For regulated domains, PROOF should be the default.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Mode(Enum):
    """Execution mode determines traceability guarantees."""

    PROOF = "proof"
    """Fact resolver for traceable, defensible conclusions.

    - Plan is generated with assumed facts
    - Facts are resolved lazily with provenance
    - Trace is a formal derivation chain
    - Every conclusion has a proof
    - Plans are self-contained (no implicit dependencies on prior session state)
    """

    EXPLORATORY = "exploratory"
    """Multi-step planner for exploration and analysis.

    - Plan is executed step-by-step
    - Each step can run arbitrary Python/SQL
    - Trace is a narrative log
    - Fast, flexible, good for building things
    - Plans can reference facts/data from previous plans
    - Session builds up a working context
    """


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

    # Control sub-intents (session management)
    MODE_SWITCH = "mode_switch"
    """Change execution mode (/proof, /explore)."""

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

    mode: Mode
    """Current execution mode (proof or exploratory)."""

    phase: Phase
    """Current task lifecycle phase."""

    active_plan: Optional[Any] = None
    """Current plan if any (Plan object when available)."""

    session_facts: dict[str, Any] = field(default_factory=dict)
    """Accumulated facts (used in exploratory mode)."""

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


# Keywords that suggest proof mode is needed
PROOF_KEYWORDS = [
    # Verification keywords
    "verify",
    "validate",
    "confirm",
    "check",
    "prove",
    "justify",
    # Reasoning keywords
    "why",
    "because",
    "reasoning",
    "evidence",
    # Decision/classification keywords
    "conclude",
    "determine",
    "classify",
    "approve",
    "reject",
    "flag",
    "qualify",
    "eligibility",
    # Compliance/audit keywords
    "compliance",
    "audit",
    "regulation",
    "defensible",
    "risk",
    # Certainty keywords
    "certain",
    "accurate",
    "correct",
    "true",
    "false",
]

# Keywords that suggest exploratory mode (data analysis)
EXPLORATORY_KEYWORDS = [
    "show",
    "display",
    "dashboard",
    "report",
    "chart",
    "graph",
    "visualize",
    "explore",
    "analyze",
    "summarize",
    "list",
    "compare",
    "trend",
    "overview",
    "build",
    "create",
]


@dataclass
class ModeSelection:
    """Result of mode selection with reasoning."""
    mode: Mode
    confidence: float
    reasoning: str
    matched_keywords: list[str] | None = None


def suggest_mode(query: str, default: Mode = Mode.PROOF) -> ModeSelection:
    """
    Suggest an execution mode based on query analysis.

    Uses keyword matching as a heuristic. For production use,
    the LLM should make this decision with full context.

    Priority: PROOF > EXPLORATORY
    - PROOF: Verification with fact-based provenance
    - EXPLORATORY: Data analysis with multi-step execution

    Args:
        query: The user's natural language query
        default: Default mode if unclear (PROOF for safety)

    Returns:
        ModeSelection with mode, confidence, and reasoning
    """
    import re
    query_lower = query.lower()

    # Helper to check word boundaries for certain keywords
    def match_with_boundary(kw: str, text: str) -> bool:
        # Keywords that should match as whole words only (not as substrings)
        boundary_keywords = {"correct", "true", "false", "risk", "flag"}
        if kw in boundary_keywords:
            return bool(re.search(rf'\b{re.escape(kw)}\b', text))
        return kw in text

    # Patterns that indicate action requests, not verification
    action_patterns = [
        r'\bverify\s+(?:you|i|we|that\s+you|that\s+i|that\s+we)\s+(?:are|am|is|were|was|have|has|had)\b',
        r'\bcheck\s+(?:you|i|we|that\s+you|that\s+i|that\s+we)\s+(?:are|am|is|were|was|have|has|had)\b',
        r'\bconfirm\s+(?:you|i|we|that\s+you|that\s+i|that\s+we)\s+(?:are|am|is|were|was|have|has|had)\b',
        r'\bmake\s+sure\b',
        r'\bensure\b',
        r'\btry\s+(?:the|this|it|again)\b',
    ]

    is_action_request = any(re.search(p, query_lower) for p in action_patterns)

    proof_matches = [kw for kw in PROOF_KEYWORDS if match_with_boundary(kw, query_lower)]
    exploratory_matches = [kw for kw in EXPLORATORY_KEYWORDS if match_with_boundary(kw, query_lower)]

    proof_score = len(proof_matches)
    exploratory_score = len(exploratory_matches)

    # If this looks like an action request, discount proof keywords
    if is_action_request:
        proof_score = 0
        exploratory_score = max(exploratory_score, 1)

    if proof_score > exploratory_score:
        return ModeSelection(
            mode=Mode.PROOF,
            confidence=min(0.9, 0.5 + proof_score * 0.1),
            reasoning=f"Query suggests need for proof-based reasoning (matched: {proof_matches})",
            matched_keywords=proof_matches,
        )
    elif exploratory_score > proof_score:
        return ModeSelection(
            mode=Mode.EXPLORATORY,
            confidence=min(0.9, 0.5 + exploratory_score * 0.1),
            reasoning=f"Query suggests exploratory analysis (matched: {exploratory_matches})",
            matched_keywords=exploratory_matches,
        )
    else:
        return ModeSelection(
            mode=default,
            confidence=0.5,
            reasoning=f"Unclear intent, defaulting to {default.value} mode for safety",
            matched_keywords=[],
        )


# System prompts for each mode
MODE_SYSTEM_PROMPTS = {
    Mode.EXPLORATORY: """You are a data analyst assistant.

Execute the user's request by creating a step-by-step plan and running each step.
Each step can query databases, transform data, create visualizations, etc.

Your trace should document WHAT you did at each step.
Focus on getting results efficiently.""",

    Mode.PROOF: """You are a reasoning assistant for auditable decisions.

For the user's request, identify the key facts needed to reach a conclusion.
Express your reasoning as a set of facts and rules that derive the answer.

Each fact must have:
- A source (database query, business rule, domain knowledge)
- A confidence level (1.0 for database facts, lower for heuristics)
- A clear derivation chain

Your trace must document WHY each conclusion is true, not just what you did.
Every conclusion must be defensible with a formal proof.""",
}


def get_mode_system_prompt(mode: Mode) -> str:
    """Get the system prompt addition for an execution mode."""
    return MODE_SYSTEM_PROMPTS.get(mode, "")


@dataclass
class ExecutionConfig:
    """Configuration for execution behavior."""

    # Default execution mode
    default_mode: Mode = Mode.PROOF

    # Allow LLM to override mode selection
    allow_mode_override: bool = True

    # For proof mode
    min_confidence: float = 0.0  # Minimum confidence to accept a fact
    require_provenance: bool = True  # All facts must have source
    max_resolution_depth: int = 5  # Max recursive fact resolution

    # For exploratory mode
    max_steps: int = 20  # Maximum plan steps
    step_timeout_seconds: int = 60  # Per-step timeout


# Domain presets for common use cases
DOMAIN_PRESETS = {
    "financial": ExecutionConfig(
        default_mode=Mode.PROOF,
        allow_mode_override=False,
        min_confidence=0.8,
        require_provenance=True,
    ),
    "healthcare": ExecutionConfig(
        default_mode=Mode.PROOF,
        allow_mode_override=False,
        min_confidence=0.9,
        require_provenance=True,
    ),
    "compliance": ExecutionConfig(
        default_mode=Mode.PROOF,
        allow_mode_override=False,
        min_confidence=0.85,
        require_provenance=True,
    ),
    "analytics": ExecutionConfig(
        default_mode=Mode.EXPLORATORY,
        allow_mode_override=True,
        min_confidence=0.0,
        require_provenance=False,
    ),
    "reporting": ExecutionConfig(
        default_mode=Mode.EXPLORATORY,
        allow_mode_override=True,
    ),
}


def get_domain_preset(domain: str) -> ExecutionConfig:
    """Get execution config for a domain preset."""
    return DOMAIN_PRESETS.get(domain, ExecutionConfig())


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
    mode: Mode
    mode_reasoning: str
    steps: list[dict]
    reasoning: str

    def format_for_display(self) -> str:
        """Format the approval request for display."""
        lines = [
            f"Mode: {self.mode.value.upper()}",
            f"  {self.mode_reasoning}",
            "",
            "Plan:",
        ]
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
    target_mode: Optional[Mode] = None

    @classmethod
    def approve(cls) -> "PlanApprovalResponse":
        """Create an approval response."""
        return cls(decision=PlanApproval.APPROVE)

    @classmethod
    def reject(cls, reason: Optional[str] = None) -> "PlanApprovalResponse":
        """Create a rejection response."""
        return cls(decision=PlanApproval.REJECT, reason=reason)

    @classmethod
    def suggest(cls, suggestion: str) -> "PlanApprovalResponse":
        """Create a suggestion response."""
        return cls(decision=PlanApproval.SUGGEST, suggestion=suggestion)

    @classmethod
    def pass_command(cls, command: str) -> "PlanApprovalResponse":
        """Create a command pass-through response."""
        return cls(decision=PlanApproval.COMMAND, command=command)
