"""Execution mode selection for traceability guarantees.

Three execution modes with different purposes:

KNOWLEDGE (Document lookup + LLM synthesis):
- Searches configured documents for relevant content
- LLM synthesizes explanation from documents + world knowledge
- No code execution, no planning needed
- Good for: explanations, definitions, process descriptions, policy lookups
- Audit answer: "Based on [document], the process works as follows..."

EXPLORATORY (Multi-step planner):
- Generates a plan, executes steps sequentially
- Trace = narrative log of what was done
- Good for: dashboards, reports, data exploration
- Audit answer: "I ran these queries and computed this"

AUDITABLE (Fact resolver):
- Generates a plan with assumed facts, resolves lazily
- Trace = formal derivation chain (proof)
- Good for: decisions, compliance, defensible conclusions
- Audit answer: "X is true BECAUSE Y AND Z, where Y came from..."

The LLM can suggest a mode based on the query, or the user can specify.
For regulated domains, AUDITABLE should be the default.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ExecutionMode(Enum):
    """Execution mode determines traceability guarantees."""

    KNOWLEDGE = "knowledge"
    """Document lookup and LLM synthesis for explanations.

    - Searches configured documents for relevant content
    - LLM synthesizes explanation from documents + world knowledge
    - No code execution, no multi-step planning
    - Shows sources consulted
    """

    EXPLORATORY = "exploratory"
    """Multi-step planner for exploration and analysis.

    - Plan is executed step-by-step
    - Each step can run arbitrary Python/SQL
    - Trace is a narrative log
    - Fast, flexible, good for building things
    """

    AUDITABLE = "auditable"
    """Fact resolver for traceable, defensible conclusions.

    - Plan is generated with assumed facts
    - Facts are resolved lazily with provenance
    - Trace is a formal derivation chain
    - Every conclusion has a proof
    """


# Keywords that suggest knowledge/explanation mode (no data analysis needed)
KNOWLEDGE_KEYWORDS = [
    # Explanation requests
    "explain",
    "describe",
    "what is",
    "what are",
    "how does",
    "how do",
    "tell me about",
    "definition of",
    "define",
    # Process/procedure questions
    "process",
    "procedure",
    "workflow",
    "steps for",
    "how to",
    # Policy/rule lookups
    "policy",
    "rule",
    "guideline",
    "requirement",
    # Overview requests (without data)
    "overview of",
    "introduction to",
    "background on",
]

# Keywords that suggest auditable mode is needed
AUDITABLE_KEYWORDS = [
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
    mode: ExecutionMode
    confidence: float
    reasoning: str

    # Keywords that influenced the decision
    matched_keywords: list[str]


def suggest_mode(query: str, default: ExecutionMode = ExecutionMode.AUDITABLE) -> ModeSelection:
    """
    Suggest an execution mode based on query analysis.

    Uses keyword matching as a heuristic. For production use,
    the LLM should make this decision with full context.

    Priority: KNOWLEDGE > EXPLORATORY > AUDITABLE
    - KNOWLEDGE: Pure explanation/lookup, no data analysis
    - EXPLORATORY: Data analysis with multi-step execution
    - AUDITABLE: Verification with fact-based provenance

    Args:
        query: The user's natural language query
        default: Default mode if unclear (AUDITABLE for safety)

    Returns:
        ModeSelection with mode, confidence, and reasoning
    """
    query_lower = query.lower()

    knowledge_matches = [kw for kw in KNOWLEDGE_KEYWORDS if kw in query_lower]
    auditable_matches = [kw for kw in AUDITABLE_KEYWORDS if kw in query_lower]
    exploratory_matches = [kw for kw in EXPLORATORY_KEYWORDS if kw in query_lower]

    knowledge_score = len(knowledge_matches)
    auditable_score = len(auditable_matches)
    exploratory_score = len(exploratory_matches)

    # Knowledge mode takes priority when it has matches and no strong
    # data analysis signals (exploratory keywords suggest actual data work)
    if knowledge_score > 0 and knowledge_score >= exploratory_score and auditable_score == 0:
        return ModeSelection(
            mode=ExecutionMode.KNOWLEDGE,
            confidence=min(0.9, 0.5 + knowledge_score * 0.1),
            reasoning=f"Query is an explanation/knowledge request (matched: {knowledge_matches})",
            matched_keywords=knowledge_matches,
        )
    elif auditable_score > exploratory_score:
        return ModeSelection(
            mode=ExecutionMode.AUDITABLE,
            confidence=min(0.9, 0.5 + auditable_score * 0.1),
            reasoning=f"Query suggests need for auditable reasoning (matched: {auditable_matches})",
            matched_keywords=auditable_matches,
        )
    elif exploratory_score > auditable_score:
        return ModeSelection(
            mode=ExecutionMode.EXPLORATORY,
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
    ExecutionMode.KNOWLEDGE: """You are a knowledgeable assistant providing explanations.

Answer the user's question by synthesizing information from available documents
and your general knowledge. Focus on clear, accurate explanations.

When referencing documents:
- Cite the specific document by name
- Quote relevant sections when helpful
- Distinguish between document content and general knowledge

Keep explanations concise but complete. If you don't have enough information,
say so rather than guessing.""",

    ExecutionMode.EXPLORATORY: """You are a data analyst assistant.

Execute the user's request by creating a step-by-step plan and running each step.
Each step can query databases, transform data, create visualizations, etc.

Your trace should document WHAT you did at each step.
Focus on getting results efficiently.""",

    ExecutionMode.AUDITABLE: """You are a reasoning assistant for auditable decisions.

For the user's request, identify the key facts needed to reach a conclusion.
Express your reasoning as a set of facts and rules that derive the answer.

Each fact must have:
- A source (database query, business rule, domain knowledge)
- A confidence level (1.0 for database facts, lower for heuristics)
- A clear derivation chain

Your trace must document WHY each conclusion is true, not just what you did.
Every conclusion must be defensible with a formal proof.""",
}


def get_mode_system_prompt(mode: ExecutionMode) -> str:
    """Get the system prompt addition for an execution mode."""
    return MODE_SYSTEM_PROMPTS.get(mode, "")


@dataclass
class ExecutionConfig:
    """Configuration for execution behavior."""

    # Default execution mode
    default_mode: ExecutionMode = ExecutionMode.AUDITABLE

    # Allow LLM to override mode selection
    allow_mode_override: bool = True

    # For auditable mode
    min_confidence: float = 0.0  # Minimum confidence to accept a fact
    require_provenance: bool = True  # All facts must have source
    max_resolution_depth: int = 5  # Max recursive fact resolution

    # For exploratory mode
    max_steps: int = 20  # Maximum plan steps
    step_timeout_seconds: int = 60  # Per-step timeout


# Domain presets for common use cases
DOMAIN_PRESETS = {
    "financial": ExecutionConfig(
        default_mode=ExecutionMode.AUDITABLE,
        allow_mode_override=False,  # Always auditable
        min_confidence=0.8,
        require_provenance=True,
    ),
    "healthcare": ExecutionConfig(
        default_mode=ExecutionMode.AUDITABLE,
        allow_mode_override=False,
        min_confidence=0.9,
        require_provenance=True,
    ),
    "compliance": ExecutionConfig(
        default_mode=ExecutionMode.AUDITABLE,
        allow_mode_override=False,
        min_confidence=0.85,
        require_provenance=True,
    ),
    "analytics": ExecutionConfig(
        default_mode=ExecutionMode.EXPLORATORY,
        allow_mode_override=True,
        min_confidence=0.0,
        require_provenance=False,
    ),
    "reporting": ExecutionConfig(
        default_mode=ExecutionMode.EXPLORATORY,
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

    MODE_SWITCH = "mode_switch"
    """User wants to switch execution mode (e.g., exploratory <-> auditable)."""


@dataclass
class PlanApprovalRequest:
    """Request for user approval of a generated plan.

    This is sent to the approval callback with all context needed
    for the user to make an informed decision.
    """
    problem: str
    mode: ExecutionMode
    mode_reasoning: str
    steps: list[dict]  # List of step dicts with number, goal, inputs, outputs
    reasoning: str  # Planner's reasoning for this approach

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
    suggestion: Optional[str] = None  # Feedback if decision is SUGGEST
    reason: Optional[str] = None  # Optional explanation for rejection
    command: Optional[str] = None  # Slash command if decision is COMMAND
    target_mode: Optional[ExecutionMode] = None  # Target mode if decision is MODE_SWITCH

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

    @classmethod
    def switch_mode(cls, target_mode: ExecutionMode) -> "PlanApprovalResponse":
        """Create a mode switch response."""
        return cls(decision=PlanApproval.MODE_SWITCH, target_mode=target_mode)
