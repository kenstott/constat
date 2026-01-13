"""Smart suggestion system for REPL input."""

from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import random

if TYPE_CHECKING:
    from constat.session import Session

try:
    from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
    from prompt_toolkit.document import Document
    from prompt_toolkit.buffer import Buffer
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False
    AutoSuggest = object
    Suggestion = None
    Document = None
    Buffer = None


@dataclass
class SuggestionConfig:
    """Configuration for suggestion behavior."""
    starter_prompt: str = "What questions can you answer for me?"
    show_schema_questions: bool = True
    max_suggestions: int = 10


@dataclass
class SuggestionContext:
    """Context for generating suggestions."""
    tables: list[str] = field(default_factory=list)
    columns: dict[str, list[str]] = field(default_factory=dict)  # table -> columns
    recent_queries: list[str] = field(default_factory=list)
    last_answer: Optional[str] = None
    has_session: bool = False


class SmartSuggester(AutoSuggest if PROMPT_TOOLKIT_AVAILABLE else object):
    """
    Context-aware suggestion generator for the REPL.

    Generates suggestions based on:
    - Available schema (tables, columns)
    - Session history
    - Common analytical patterns
    - Previous answer context
    """

    # Common analytical question patterns
    PATTERNS = [
        "What is the total {metric} by {dimension}?",
        "Which {entity} has the highest {metric}?",
        "Show me the trend of {metric} over time",
        "Compare {metric} across different {dimension}",
        "What are the top 10 {entity} by {metric}?",
        "How has {metric} changed in the last {period}?",
        "What is the average {metric} per {dimension}?",
        "Find {entity} where {condition}",
        "Break down {metric} by {dimension}",
        "What percentage of {entity} have {condition}?",
    ]

    # Generic starter questions when no schema available
    STARTER_QUESTIONS = [
        "What data sources are available?",
        "Show me a summary of the data",
        "What tables or datasets can I query?",
        "What are the main entities in this dataset?",
        "Give me an overview of the schema",
    ]

    # Follow-up patterns based on previous answer
    FOLLOWUP_PATTERNS = [
        "Tell me more about {topic}",
        "Break this down by {dimension}",
        "Why is {topic} so high/low?",
        "Show me the underlying data",
        "What's driving this trend?",
        "Compare this to the previous period",
        "Which factors contribute most to {topic}?",
    ]

    def __init__(
        self,
        session: Optional["Session"] = None,
        config: Optional[SuggestionConfig] = None,
    ):
        self.session = session
        self.config = config or SuggestionConfig()
        self._context = SuggestionContext()
        self._suggestion_index = 0
        self._cached_suggestions: list[str] = []

    def update_context(self, session: Optional["Session"] = None) -> None:
        """Update suggestion context from session state."""
        if session:
            self.session = session

        if not self.session:
            self._context = SuggestionContext()
            return

        self._context.has_session = bool(self.session.session_id)

        # Get table information from datastore
        if self.session.datastore:
            tables = self.session.datastore.list_tables()
            self._context.tables = [t["name"] for t in tables]

            # Try to get column info for each table
            for table_info in tables:
                table_name = table_info["name"]
                try:
                    # Query schema information
                    schema_df = self.session.datastore.query(
                        f"SELECT * FROM {table_name} LIMIT 0"
                    )
                    self._context.columns[table_name] = list(schema_df.columns)
                except Exception:
                    pass

        # Get recent queries from history
        if hasattr(self.session, 'history') and self.session.history:
            try:
                recent = self.session.history.get_recent_queries(limit=5)
                self._context.recent_queries = [q.query for q in recent if hasattr(q, 'query')]
            except Exception:
                pass

        # Regenerate suggestions with new context
        self._cached_suggestions = self._generate_suggestions()
        self._suggestion_index = 0

    def _generate_suggestions(self) -> list[str]:
        """Generate a list of contextual suggestions."""
        suggestions = []

        # Always start with the configured starter prompt
        starter = self.config.starter_prompt

        if not self._context.has_session:
            # No session - use starter + generic questions
            return [starter] + self.STARTER_QUESTIONS.copy()

        if not self._context.tables:
            # Session but no tables yet - use starter + generic starters
            return [
                starter,
                "What questions can I ask about this data?",
                "Show me what data is available",
                "Summarize the main insights",
            ] + self.STARTER_QUESTIONS[:2]

        # Start with configured starter prompt
        suggestions.append(starter)

        # Generate schema-aware suggestions if enabled
        if self.config.show_schema_questions:
            for table in self._context.tables[:3]:  # Limit to first 3 tables
                columns = self._context.columns.get(table, [])

                # Find likely metric and dimension columns
                metrics = [c for c in columns if any(
                    kw in c.lower() for kw in
                    ['amount', 'total', 'count', 'sum', 'value', 'price', 'revenue', 'cost', 'qty', 'quantity']
                )]
                dimensions = [c for c in columns if any(
                    kw in c.lower() for kw in
                    ['name', 'type', 'category', 'status', 'region', 'date', 'id', 'customer', 'product']
                )]

                # Generate questions using patterns
                if metrics and dimensions:
                    metric = metrics[0]
                    dimension = dimensions[0]
                    suggestions.extend([
                        f"What is the total {metric} by {dimension}?",
                        f"Which {table} has the highest {metric}?",
                        f"Show me {table} breakdown by {dimension}",
                    ])
                elif columns:
                    # Use first few columns
                    col = columns[0] if columns else table
                    suggestions.extend([
                        f"Show me all {table}",
                        f"What are the unique values in {col}?",
                        f"Count {table} by {col}" if len(columns) > 1 else f"Count all {table}",
                    ])

        # Add some generic analytical questions
        suggestions.extend([
            "What are the key insights from this data?",
            "Are there any anomalies or outliers?",
            "What trends do you see?",
        ])

        # Add follow-up suggestions if we have a last answer
        if self._context.last_answer:
            suggestions.insert(0, "Tell me more about this")
            suggestions.insert(1, "Break this down further")

        # Limit to configured max
        return suggestions[:self.config.max_suggestions]

    def get_suggestion(self, buffer: "Buffer", document: "Document") -> Optional["Suggestion"]:
        """
        Get a suggestion for the current input.

        Called by prompt_toolkit to provide auto-suggestions.
        """
        if not PROMPT_TOOLKIT_AVAILABLE:
            return None

        text = document.text

        # Only suggest when input is empty or very short
        if len(text) > 20:
            return None

        # Update context if needed
        if not self._cached_suggestions:
            self._cached_suggestions = self._generate_suggestions()

        if not self._cached_suggestions:
            return None

        # If empty input, show first suggestion
        if not text:
            suggestion = self._cached_suggestions[self._suggestion_index % len(self._cached_suggestions)]
            return Suggestion(suggestion)

        # If partial input, find matching suggestion
        text_lower = text.lower()
        for suggestion in self._cached_suggestions:
            if suggestion.lower().startswith(text_lower):
                # Return the remaining part of the suggestion
                return Suggestion(suggestion[len(text):])

        return None

    def next_suggestion(self) -> Optional[str]:
        """Cycle to the next suggestion (for Tab-Tab behavior)."""
        if not self._cached_suggestions:
            return None

        self._suggestion_index = (self._suggestion_index + 1) % len(self._cached_suggestions)
        return self._cached_suggestions[self._suggestion_index]

    def get_current_suggestion(self) -> Optional[str]:
        """Get the current full suggestion."""
        if not self._cached_suggestions:
            self._cached_suggestions = self._generate_suggestions()

        if not self._cached_suggestions:
            return None

        return self._cached_suggestions[self._suggestion_index % len(self._cached_suggestions)]

    def set_last_answer(self, answer: str) -> None:
        """Update context with the last answer for follow-up suggestions."""
        self._context.last_answer = answer
        # Regenerate suggestions with new context
        self._cached_suggestions = self._generate_suggestions()
        self._suggestion_index = 0


def create_suggester(
    session: Optional["Session"] = None,
    config: Optional[SuggestionConfig] = None,
) -> Optional[SmartSuggester]:
    """Create a SmartSuggester if prompt_toolkit is available."""
    if not PROMPT_TOOLKIT_AVAILABLE:
        return None
    return SmartSuggester(session, config)
