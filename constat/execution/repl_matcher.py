"""REPL command matching via natural language exemplars.

This module provides fast, pre-LLM matching of user input against
known REPL commands using simple token similarity. This allows
natural language shortcuts like "start over" to match "/reset".

The matcher runs BEFORE LLM intent classification for efficiency.
At 80% similarity threshold, approximate matches are accepted.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ReplCommand:
    """A REPL command with its natural language exemplars."""
    command: str
    exemplars: list[str]
    description: str
    min_tokens: int = 1  # Minimum tokens in input to consider matching


# REPL commands with natural language exemplars
# Order matters: more specific commands should come first
REPL_COMMANDS: list[ReplCommand] = [
    # Session control
    ReplCommand(
        command="/reset",
        exemplars=[
            "start over",
            "new topic",
            "fresh start",
            "clear everything",
            "begin again",
            "start fresh",
            "reset session",
            "clear session",
        ],
        description="Clear session and start fresh",
        min_tokens=2,
    ),

    # Mode switching
    ReplCommand(
        command="/mode exploratory",
        exemplars=[
            "explore this",
            "switch to exploratory",
            "exploratory mode",
            "let me explore",
            "data exploration",
            "explore the data",
        ],
        description="Switch to exploratory mode",
        min_tokens=2,
    ),
    ReplCommand(
        command="/mode auditable",
        exemplars=[
            "verify this",
            "audit mode",
            "auditable mode",
            "formal verification",
            "prove this",
            "formal derivation",
            "switch to auditable",
        ],
        description="Switch to auditable mode",
        min_tokens=2,
    ),
    ReplCommand(
        command="/mode knowledge",
        exemplars=[
            "knowledge mode",
            "explain this",
            "just explain",
            "switch to knowledge",
        ],
        description="Switch to knowledge mode",
        min_tokens=2,
    ),

    # Derivation/provenance
    ReplCommand(
        command="/provenance",
        exemplars=[
            "show derivation",
            "how did you get this",
            "show your work",
            "trace this",
            "show the proof",
            "derivation chain",
            "audit trail",
        ],
        description="Show derivation chain",
        min_tokens=2,
    ),

    # Redo operations
    ReplCommand(
        command="/redo",
        exemplars=[
            "run it again",
            "redo this",
            "try again",
            "rerun",
            "do it again",
        ],
        description="Re-run the previous analysis",
        min_tokens=2,
    ),

    # Help
    ReplCommand(
        command="/help",
        exemplars=[
            "show help",
            "what can you do",
            "list commands",
            "available commands",
        ],
        description="Show available commands",
        min_tokens=2,
    ),
]


def tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase words, filtering stopwords."""
    # Common stopwords to ignore (keep "this" since it's meaningful in exemplars)
    stopwords = {
        "a", "an", "the", "to", "in", "on", "at", "for", "of", "and", "or",
        "is", "are", "was", "were", "be", "been", "being",
        "i", "me", "my", "you", "your", "it", "its",
        "that", "these", "those",
        "can", "could", "would", "should", "will",
        "please", "just", "now", "then",
    }

    words = text.lower().split()
    return {w.strip(".,!?;:'\"") for w in words if w.strip(".,!?;:'\"") not in stopwords}


def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def token_overlap_ratio(user_tokens: set[str], exemplar_tokens: set[str]) -> float:
    """Calculate what fraction of exemplar tokens appear in user input.

    This is more lenient than Jaccard - if user says "please start over now",
    we care that "start" and "over" are present, not that extra words exist.
    """
    if not exemplar_tokens:
        return 0.0
    matched = len(user_tokens & exemplar_tokens)
    return matched / len(exemplar_tokens)


@dataclass
class MatchResult:
    """Result of REPL command matching."""
    command: str
    description: str
    confidence: float
    matched_exemplar: str


def match_repl_command(
    user_input: str,
    threshold: float = 0.80,
) -> Optional[MatchResult]:
    """Match user input against REPL command exemplars.

    Args:
        user_input: The user's natural language input
        threshold: Minimum similarity score (0.0-1.0) to accept a match

    Returns:
        MatchResult if a command matches above threshold, None otherwise
    """
    user_tokens = tokenize(user_input)

    if not user_tokens:
        return None

    best_match: Optional[MatchResult] = None
    best_score = 0.0

    for cmd in REPL_COMMANDS:
        # Skip if input is too short
        if len(user_tokens) < cmd.min_tokens:
            continue

        for exemplar in cmd.exemplars:
            exemplar_tokens = tokenize(exemplar)
            if not exemplar_tokens:
                continue

            # Use token overlap ratio (more lenient than Jaccard)
            # This handles "please start over now" matching "start over"
            overlap = token_overlap_ratio(user_tokens, exemplar_tokens)

            # Combine with Jaccard for balanced scoring
            # This penalizes inputs with many extra tokens
            jaccard = jaccard_similarity(user_tokens, exemplar_tokens)

            # Weight: 60% overlap (all exemplar tokens present), 40% Jaccard (length similarity)
            score = (0.6 * overlap) + (0.4 * jaccard)

            if score > best_score and score >= threshold:
                best_score = score
                best_match = MatchResult(
                    command=cmd.command,
                    description=cmd.description,
                    confidence=score,
                    matched_exemplar=exemplar,
                )

    return best_match


def get_all_commands() -> list[dict]:
    """Get all REPL commands with their exemplars for help display."""
    return [
        {
            "command": cmd.command,
            "description": cmd.description,
            "exemplars": cmd.exemplars,
        }
        for cmd in REPL_COMMANDS
    ]
