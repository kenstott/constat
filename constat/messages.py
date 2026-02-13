# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Unified message format for REPL and UI.

This module provides structured message types that can be formatted
differently by each client:
- REPL: Formats to Rich Text for terminal display
- UI: Formats to Markdown for web display

Both clients consume the same message structure from the session/server.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from constat.prompts import load_yaml

# Vera's personality loaded from YAML (single source of truth)
_personality = load_yaml("vera_personality.yaml")
RELIABLE_ADJECTIVES = _personality["reliable_adjectives"]
HONEST_ADJECTIVES = _personality["honest_adjectives"]
STARTER_SUGGESTIONS = _personality["starter_suggestions"]
TAGLINES = _personality["taglines"]


def get_vera_adjectives() -> tuple[str, str]:
    """Return a random pair of (reliable, honest) adjectives for Vera's intro."""
    return (
        random.choice(RELIABLE_ADJECTIVES),
        random.choice(HONEST_ADJECTIVES),
    )


def get_vera_tagline() -> str:
    """Return a random tagline for Vera's intro."""
    return random.choice(TAGLINES)


def get_starter_suggestions(count: int = 4) -> list[str]:
    """Return a random sample of starter suggestions."""
    return random.sample(STARTER_SUGGESTIONS, min(count, len(STARTER_SUGGESTIONS)))


class MessageType(str, Enum):
    """Types of messages that can be displayed."""
    WELCOME = "welcome"
    SYSTEM = "system"
    USER = "user"
    STEP_STATUS = "step_status"
    PLAN = "plan"
    CLARIFICATION = "clarification"
    INSIGHT = "insight"
    ERROR = "error"
    SUGGESTIONS = "suggestions"


@dataclass
class WelcomeMessage:
    """Welcome message from Vera."""
    reliable_adjective: str
    honest_adjective: str
    tagline: str = ""
    suggestions: list[str] = field(default_factory=list)

    @classmethod
    def create(cls) -> "WelcomeMessage":
        """Create a welcome message with random adjectives, tagline, and suggestions."""
        reliable, honest = get_vera_adjectives()
        return cls(
            reliable_adjective=reliable,
            honest_adjective=honest,
            tagline=get_vera_tagline(),
            suggestions=get_starter_suggestions(),
        )

    def to_markdown(self) -> str:
        """Format for UI display (Markdown)."""
        suggestions_text = "\n".join(
            f"{i}. {s}" for i, s in enumerate(self.suggestions, 1)
        )
        return f"""Hi, I'm **Vera**, your {self.reliable_adjective} and {self.honest_adjective} data analyst.

_{self.tagline}_

**Try asking:**
{suggestions_text}"""

    def to_plain(self) -> str:
        """Format as plain text."""
        suggestions_text = "\n".join(
            f"  {i}. {s}" for i, s in enumerate(self.suggestions, 1)
        )
        return f"""Hi, I'm Vera, your {self.reliable_adjective} and {self.honest_adjective} data analyst.

{self.tagline}

Try asking:
{suggestions_text}"""


@dataclass
class StepStatusMessage:
    """Status update for a step during execution."""
    step_number: int
    total_steps: int
    status: str  # "start", "generating", "executing", "complete", "error", "retry"
    goal: str
    output: Optional[str] = None
    error: Optional[str] = None
    attempt: int = 1
    max_attempts: int = 3

    def to_markdown(self) -> str:
        """Format for UI display."""
        if self.status == "start":
            return f"Step {self.step_number}: {self.goal}..."
        elif self.status == "generating":
            attempt_str = f" (attempt {self.attempt})" if self.attempt > 1 else ""
            return f"Step {self.step_number}: Generating code{attempt_str}..."
        elif self.status == "executing":
            return f"Step {self.step_number}: Executing..."
        elif self.status == "complete":
            output_str = f"\n\n{self.output}" if self.output else ""
            return f"Step {self.step_number}: ✓ {self.goal}{output_str}"
        elif self.status == "error":
            return f"Step {self.step_number}: ❌ {self.error}"
        elif self.status == "retry":
            return f"Step {self.step_number}: Retrying (attempt {self.attempt})..."
        return f"Step {self.step_number}: {self.goal}"


@dataclass
class InsightMessage:
    """Final insight/synthesis message."""
    findings: str
    observations: Optional[str] = None
    suggested_next_steps: list[str] = field(default_factory=list)
    tables_created: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Format for UI display."""
        parts = [self.findings]

        if self.observations:
            parts.append(f"\n\n**Observations**\n{self.observations}")

        if self.suggested_next_steps:
            steps = "\n".join(f"- {s}" for s in self.suggested_next_steps)
            parts.append(f"\n\n**Suggested Next Steps**\n{steps}")

        return "".join(parts)


@dataclass
class SuggestionsMessage:
    """Follow-up suggestions after a query."""
    suggestions: list[str]

    def to_markdown(self) -> str:
        """Format for UI display."""
        if not self.suggestions:
            return ""
        items = "\n".join(f"{i}. {s}" for i, s in enumerate(self.suggestions, 1))
        return f"**You might also ask:**\n{items}"


@dataclass
class ClarificationMessage:
    """Clarification request with questions."""
    original_question: str
    ambiguity_reason: str
    questions: list[dict]  # List of {text: str, suggestions: list[str]}

    def to_markdown(self) -> str:
        """Format for UI display."""
        parts = ["**Clarification needed**", f"_{self.ambiguity_reason}_", ""]

        for i, q in enumerate(self.questions, 1):
            parts.append(f"**Q{i}:** {q['text']}")
            if q.get("suggestions"):
                for j, s in enumerate(q["suggestions"], 1):
                    parts.append(f"  {j}. {s}")
            parts.append("")

        return "\n".join(parts)


@dataclass
class ErrorMessage:
    """Error message."""
    error: str
    details: Optional[str] = None
    recoverable: bool = True

    def to_markdown(self) -> str:
        """Format for UI display."""
        msg = f"**Error:** {self.error}"
        if self.details:
            msg += f"\n\n{self.details}"
        return msg