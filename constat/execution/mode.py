"""Execution mode selection for traceability guarantees.

Two fundamental modes with different traceability properties:

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


# Keywords that suggest auditable mode is needed
AUDITABLE_KEYWORDS = [
    "why",
    "prove",
    "justify",
    "explain",
    "compliance",
    "audit",
    "regulation",
    "defensible",
    "because",
    "reasoning",
    "evidence",
    "conclude",
    "determine",
    "classify",
    "approve",
    "reject",
    "flag",
    "risk",
    "eligibility",
    "qualify",
]

# Keywords that suggest exploratory mode
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

    Args:
        query: The user's natural language query
        default: Default mode if unclear (AUDITABLE for safety)

    Returns:
        ModeSelection with mode, confidence, and reasoning
    """
    query_lower = query.lower()

    auditable_matches = [kw for kw in AUDITABLE_KEYWORDS if kw in query_lower]
    exploratory_matches = [kw for kw in EXPLORATORY_KEYWORDS if kw in query_lower]

    auditable_score = len(auditable_matches)
    exploratory_score = len(exploratory_matches)

    if auditable_score > exploratory_score:
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
    return MODE_SYSTEM_PROMPTS[mode]


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
