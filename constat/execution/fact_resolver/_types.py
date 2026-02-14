# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Enums, dataclasses, constants, and utility functions for fact resolution."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)


class FactSource(Enum):
    """Where a fact was resolved from."""
    CACHE = "cache"
    DATABASE = "database"
    DOCUMENT = "document"  # From a reference document
    API = "api"  # From REST or GraphQL API
    LLM_KNOWLEDGE = "llm_knowledge"
    LLM_HEURISTIC = "llm_heuristic"
    RULE = "rule"  # Derived via a registered rule function
    SUB_PLAN = "sub_plan"  # Required a mini-plan to derive (legacy)
    DERIVED = "derived"  # Computed from other resolved facts
    USER_PROVIDED = "user_provided"
    CONFIG = "config"  # From system prompt / config
    UNRESOLVED = "unresolved"


class Tier2Strategy(Enum):
    """Result of Tier 2 LLM assessment for unresolved facts."""
    DERIVABLE = "derivable"  # Can be computed from 2+ inputs
    KNOWN = "known"  # LLM can provide directly (general knowledge)
    USER_REQUIRED = "user_required"  # Needs human input


# Sources that represent ground truth (raw data from sources)
# These can be reused in audit mode since data is immutable
GROUND_TRUTH_SOURCES = {
    FactSource.DATABASE,
    FactSource.DOCUMENT,
    FactSource.API,
    FactSource.USER_PROVIDED,
    FactSource.CONFIG,
}

# Sources that represent derived/computed facts
# These must be re-derived independently in audit mode
DERIVED_SOURCES = {
    FactSource.RULE,
    FactSource.SUB_PLAN,
    FactSource.DERIVED,
    FactSource.LLM_KNOWLEDGE,
    FactSource.LLM_HEURISTIC,
}


def format_source_attribution(
    source_type: Union[FactSource, str],
    source_name: Optional[str] = None,
    entity: Optional[str] = None,
    params: Optional[list[str]] = None,
) -> str:
    """Format source attribution consistently across the codebase.

    This provides a common format for displaying where data came from:
    - Database: db_name.table_name (e.g., "chinook.employees")
    - GraphQL: api_name.query_path (e.g., "catfacts.breeds")
    - REST: api_name.resource[params] (e.g., "countries./v3.1/all[fields,lang]")
    - User: "user" (values from user input)
    - Cache: "cache"
    - Knowledge: "knowledge"
    - Derived: "derived"

    Args:
        source_type: The FactSource enum or string source type
        source_name: Name of the source (db name, api name, etc.)
        entity: The specific entity accessed (table name, query path, endpoint)
        params: Optional list of parameter names for REST APIs

    Returns:
        Formatted source attribution string
    """
    import re

    # Normalize source_type to string
    if isinstance(source_type, FactSource):
        source_str = source_type.value
    else:
        source_str = str(source_type).lower()

    # Handle special cases
    if source_str in ("user", "user_provided", "embedded"):
        return "user"
    if source_str == "cache":
        return "cache"
    if source_str in ("llm_knowledge", "knowledge"):
        return "knowledge"
    if source_str == "derived":
        return "derived"
    if source_str == "document":
        if source_name and entity:
            return f"{source_name}.{entity}"
        return source_name or "document"

    # Database format: db_name.table_name
    if source_str == "database":
        if source_name and entity:
            return f"{source_name}.{entity}"
        return source_name or "database"

    # API format depends on type
    if source_str == "api":
        if not source_name:
            return "api"

        if entity:
            # Check if it's a GraphQL query (contains { })
            if "{" in entity:
                # Extract query path with dot notation
                # e.g., "{ countries { currencies } }" -> "countries.currencies"
                # Remove query wrapper and extract field names
                clean = entity.strip()
                if clean.startswith("{"):
                    clean = clean[1:]
                if clean.endswith("}"):
                    clean = clean[:-1]
                # Extract field names (words before { or at leaf level)
                fields = re.findall(r'(\w+)\s*(?:\{|$)', clean)
                if fields:
                    query_path = ".".join(fields)
                    return f"{source_name}.{query_path}"
                return f"{source_name}.query"

            # REST endpoint: extract resource name and params
            # e.g., "GET /v3.1/all" or "/users/{id}/orders?limit=10"
            endpoint = entity
            # Remove HTTP method prefix
            endpoint = re.sub(r'^(GET|POST|PUT|DELETE|PATCH)\s+', '', endpoint)
            # Extract path (before query string)
            path = endpoint.split("?")[0]
            # Extract path variable names from {name} patterns
            path_vars = re.findall(r'\{(\w+)\}', path)
            # Extract query param names
            query_str = endpoint.split("?")[1] if "?" in endpoint else ""
            query_params = re.findall(r'(\w+)=', query_str)

            # Combine params (use provided list or extract from endpoint)
            all_params = params or (path_vars + query_params)

            if all_params:
                return f"{source_name}.{path}[{','.join(all_params)}]"
            return f"{source_name}.{path}"

        return source_name

    # Default: just return source name or type
    return source_name or source_str


@dataclass
class AuditContext:
    """Context for audit mode from exploratory session.

    Enables independent re-derivation while reusing ground truth data.
    The audit mode will:
    1. Re-interpret the question independently
    2. Determine what conclusion to prove
    3. Reuse ground truth facts (why re-query same data?)
    4. Re-derive all computed facts independently
    5. Compare final answer with exploratory's result
    """
    # The question to re-interpret and prove
    original_question: str
    follow_ups: list[str] = field(default_factory=list)

    # Ground truth facts with their retrieval methods (reusable)
    ground_truth_facts: dict[str, "Fact"] = field(default_factory=dict)

    # Exploratory's answer (for comparison AFTER audit derives its own)
    exploratory_answer: Any = None

    # Optional hints (audit can use or ignore)
    relevant_tables: list[str] = field(default_factory=list)
    relevant_columns: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_exploratory_session(
        cls,
        original_question: str,
        follow_ups: list[str],
        facts: list["Fact"],
        exploratory_answer: Any,
        relevant_tables: Optional[list[str]] = None,
        relevant_columns: Optional[dict[str, list[str]]] = None,
    ) -> "AuditContext":
        """Build audit context from exploratory session state."""
        # Extract only ground truth facts
        ground_truth = {}
        for fact in facts:
            if fact.source in GROUND_TRUTH_SOURCES:
                ground_truth[fact.name] = fact

        return cls(
            original_question=original_question,
            follow_ups=follow_ups,
            ground_truth_facts=ground_truth,
            exploratory_answer=exploratory_answer,
            relevant_tables=relevant_tables or [],
            relevant_columns=relevant_columns or {},
        )


@dataclass
class Fact:
    """A resolved fact with provenance."""
    name: str
    value: Any
    confidence: float = 1.0
    source: FactSource = FactSource.DATABASE

    # Provenance chain - facts this was derived from
    because: list["Fact"] = field(default_factory=list)

    # Additional metadata
    description: Optional[str] = None  # Human-friendly description of what this fact represents
    source_name: Optional[str] = None  # Specific source (database name, document path, API name)
    query: Optional[str] = None  # SQL query if from database
    api_endpoint: Optional[str] = None  # REST endpoint or GraphQL query name if from API
    rule_name: Optional[str] = None  # Rule function name if derived
    reasoning: Optional[str] = None  # LLM explanation if from knowledge
    resolved_at: datetime = field(default_factory=datetime.now)

    # Table reference (for large array values stored in datastore)
    table_name: Optional[str] = None  # Name of table in datastore if value is stored there
    row_count: Optional[int] = None  # Number of rows if stored as table

    # Creation context - detailed information about how this fact was created
    # Contains: code from plan, user prompt, SQL query, or other creation details
    context: Optional[str] = None

    # Role provenance - which role created this fact (metadata, not access control)
    # None = created in shared context, "financial-analyst" = created by that role
    # All facts are globally accessible regardless of role_id
    role_id: Optional[str] = None

    @property
    def is_resolved(self) -> bool:
        return self.source != FactSource.UNRESOLVED

    @property
    def is_table_reference(self) -> bool:
        """True if value is stored as a table in datastore."""
        return self.table_name is not None

    @property
    def display_value(self) -> str:
        """Human-readable value for display (concise for table references)."""
        if self.is_table_reference:
            return f"{self.row_count} rows (table: {self.table_name})"
        return str(self.value)

    @property
    def resolution_summary(self) -> str:
        """One-line summary of how this fact was resolved, suitable for proof tree display."""
        if self.source == FactSource.DATABASE:
            if self.query:
                # Extract just the SQL part if it's wrapped in code
                sql = self.query
                if "pd.read_sql(" in sql:
                    # Extract the SQL string from pd.read_sql call
                    import re
                    match = re.search(r'pd\.read_sql\(["\']([^"\']+)', sql)
                    if match:
                        sql = match.group(1)
                # Truncate for display
                sql = sql.replace('\n', ' ').strip()
                if len(sql) > 80:
                    sql = sql[:77] + "..."
                return f"SQL: {sql}"
            return f"queried {self.source_name or 'database'}"
        elif self.source == FactSource.DOCUMENT:
            return f"text search in '{self.source_name or 'documents'}'"
        elif self.source == FactSource.LLM_KNOWLEDGE:
            if self.reasoning:
                reason = self.reasoning[:60] + "..." if len(self.reasoning) > 60 else self.reasoning
                return f"LLM knowledge: {reason}"
            return "LLM knowledge"
        elif self.source == FactSource.RULE:
            return f"rule: {self.rule_name or 'derived'}"
        elif self.source == FactSource.CACHE:
            return "cached from prior resolution"
        elif self.source == FactSource.USER_PROVIDED:
            return "provided by user"
        elif self.source == FactSource.API:
            return f"API: {self.api_endpoint or self.source_name or 'external'}"
        elif self.source == FactSource.CONFIG:
            return "from configuration"
        else:
            # noinspection PyTypeChecker
            return self.source.value

    @property
    def derivation_trace(self) -> str:
        """Human-readable derivation chain."""
        # Build source string with specific name if available
        source_str = self.source.value
        if self.source_name:
            source_str = f"{self.source.value}:{self.source_name}"
        # Use display_value for concise table references
        lines = [f"{self.name} = {self.display_value} (confidence: {self.confidence:.2f}, source: {source_str})"]
        # noinspection DuplicatedCode
        if self.query:
            lines.append(f"  via SQL: {self.query}")
        if self.api_endpoint:
            lines.append(f"  via API: {self.api_endpoint}")
        if self.rule_name:
            lines.append(f"  via rule: {self.rule_name}")
        if self.reasoning:
            lines.append(f"  reasoning: {self.reasoning}")
        for dep in self.because:
            for line in dep.derivation_trace.split("\n"):
                lines.append(f"    {line}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for storage/API."""
        import datetime as dt

        def _convert_numpy(obj):
            """Convert numpy types to native Python for JSON serialization."""
            if hasattr(obj, 'item'):  # numpy scalar (int64, float64, etc.)
                return obj.item()
            elif isinstance(obj, dict):
                return {k: _convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert_numpy(v) for v in obj]
            return obj

        # Determine value type for proper restoration
        # Types: boolean, integer, float, string, date, datetime, time, table, array, object
        value = _convert_numpy(self.value)  # Convert numpy types
        value_type = "string"  # default

        if self.table_name:
            # Table type: value is the table name (URI to parquet)
            value_type = "table"
            value = self.table_name
        elif isinstance(self.value, bool):
            value_type = "boolean"
        elif isinstance(self.value, int):
            value_type = "integer"
        elif isinstance(self.value, float):
            value_type = "float"
        elif isinstance(self.value, dt.datetime):
            value_type = "datetime"
            value = self.value.isoformat()
        elif isinstance(self.value, dt.date):
            value_type = "date"
            value = self.value.isoformat()
        elif isinstance(self.value, dt.time):
            value_type = "time"
            value = self.value.isoformat()
        elif isinstance(self.value, list):
            value_type = "array"
        elif isinstance(self.value, dict):
            value_type = "object"
        elif isinstance(self.value, str):
            value_type = "string"
        elif hasattr(self.value, 'to_dict'):
            # DataFrame or similar - convert to serializable form
            value_type = "dataframe"
            try:
                value = f"{len(self.value)} rows"  # Store summary, not full data
            except Exception as e:
                logger.debug(f"Could not get len() of value, using str(): {e}")
                value = str(self.value)[:500]

        result = {
            "name": self.name,
            "value": value,
            "value_type": value_type,
            "confidence": self.confidence,
            "source": self.source.value,
            "source_name": self.source_name,
            "query": self.query,
            "api_endpoint": self.api_endpoint,
            "rule_name": self.rule_name,
            "reasoning": self.reasoning,
            "because": [f.name for f in self.because],
            "resolved_at": self.resolved_at.isoformat(),
            "role_id": self.role_id,
        }
        # Add row count for table types
        if self.table_name and self.row_count:
            result["row_count"] = self.row_count
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Fact":
        """Deserialize a Fact from dictionary (for redo/restore)."""
        import datetime as dt

        # Handle source conversion
        source = data.get("source", "database")
        if isinstance(source, str):
            source = FactSource(source)

        # Handle resolved_at conversion
        resolved_at = data.get("resolved_at")
        if isinstance(resolved_at, str):
            resolved_at = datetime.fromisoformat(resolved_at)
        elif resolved_at is None:
            resolved_at = datetime.now()

        # Handle value based on value_type
        value = data["value"]
        value_type = data.get("value_type", "string")
        table_name = None
        row_count = data.get("row_count")

        if value_type == "table":
            # Value is the table name (URI to parquet)
            table_name = value
            # Restore display value for table facts
            if row_count:
                value = f"{row_count} rows"
            else:
                value = f"table:{table_name}"
        elif value_type == "datetime" and isinstance(value, str):
            value = dt.datetime.fromisoformat(value)
        elif value_type == "date" and isinstance(value, str):
            value = dt.date.fromisoformat(value)
        elif value_type == "time" and isinstance(value, str):
            value = dt.time.fromisoformat(value)
        # array, object, boolean, integer, float, string - JSON handles these correctly

        return cls(
            name=data["name"],
            value=value,
            confidence=data.get("confidence", 1.0),
            source=source,
            because=[],  # Dependencies not restored (would need full cache)
            description=data.get("description"),
            source_name=data.get("source_name"),
            query=data.get("query"),
            api_endpoint=data.get("api_endpoint"),
            rule_name=data.get("rule_name"),
            reasoning=data.get("reasoning"),
            resolved_at=resolved_at,
            table_name=table_name,
            row_count=row_count,
            role_id=data.get("role_id"),
        )


@dataclass
class FactDependency:
    """A declared dependency on another fact.

    Used in ResolutionSpec to explicitly declare what facts are needed
    before the derivation logic runs.
    """
    name: str
    params: dict = field(default_factory=dict)
    source_hint: Optional[str] = None  # "database", "document", "config", etc.


@dataclass
class ResolutionSpec:
    """Declarative specification for resolving a fact.

    This separates the "what do I need" from the "how do I combine it":
    - depends_on: Facts that must be resolved first
    - logic: Python code that ONLY uses resolved facts (sandboxed)
    - sql/doc_query: For leaf facts with no dependencies

    The resolver:
    1. Resolves all dependencies first (building proof tree)
    2. Executes logic with only resolved facts as input
    3. Returns result with full provenance
    """
    fact_name: str
    depends_on: list[FactDependency] = field(default_factory=list)
    logic: Optional[str] = None  # Python code combining resolved facts

    # For leaf facts (no dependencies, direct resolution)
    sql: Optional[str] = None        # Direct SQL for database facts
    doc_query: Optional[str] = None  # Search query for document facts
    source_hint: Optional[str] = None  # Which source to use

    @property
    def is_leaf(self) -> bool:
        """True if this is a leaf fact with no dependencies."""
        return len(self.depends_on) == 0


@dataclass
class ProofNode:
    """A node in a formal derivation proof tree.

    Used in auditable mode to show exactly how a conclusion was reached:
    - What premises (other facts) it depends on
    - What source provided the data
    - What evidence (SQL, doc excerpt) supports it
    """
    conclusion: str
    source: FactSource
    source_name: Optional[str] = None  # database name, document path, etc.
    evidence: Optional[str] = None  # SQL query, document excerpt, etc.
    premises: list["ProofNode"] = field(default_factory=list)
    confidence: float = 1.0

    def to_trace(self, indent: int = 0) -> str:
        """Render as human-readable proof trace."""
        prefix = "  " * indent
        source_str = self.source.value
        if self.source_name:
            source_str = f"{self.source.value}:{self.source_name}"

        lines = [f"{prefix}∴ {self.conclusion} [{source_str}, confidence={self.confidence:.2f}]"]

        if self.evidence:
            # Truncate long evidence
            evidence_display = self.evidence[:200] + "..." if len(self.evidence) > 200 else self.evidence
            lines.append(f"{prefix}  evidence: {evidence_display}")

        for premise in self.premises:
            lines.append(premise.to_trace(indent + 1))

        return "\n".join(lines)


# Type for rule functions: (resolver, **params) -> Fact
RuleFunction = Callable[["FactResolver", dict], Fact]


@dataclass
class ResolutionStrategy:
    """Configuration for how facts should be resolved.

    Tiered Resolution Architecture:

    TIER 1: Local Sources (parallel, cheap, fast)
        - Cache, Config, Rules, Documents, Database
        - All run in parallel with timeout window
        - First successful result with sufficient confidence wins

    TIER 2: LLM Assessment (expensive, gated)
        - Single LLM call to assess best strategy
        - DERIVABLE: Can compute from 2+ inputs (triggers sub-plan)
        - KNOWN: LLM provides general knowledge directly
        - USER_REQUIRED: Needs human input

    TIER 3: User Prompt (fallback)
        - Handled by session layer, not fact resolver
    """
    # Tier 1 sources - all run in parallel
    tier1_sources: list[FactSource] = field(default_factory=lambda: [
        FactSource.CACHE,
        FactSource.CONFIG,
        FactSource.RULE,
        FactSource.DOCUMENT,
        FactSource.DATABASE,
        FactSource.API,
    ])

    # Tier 1 timeout in seconds - collect results within this window
    tier1_timeout: float = 15.0

    # Legacy: source_priority kept for backward compatibility
    # New code should use tier1_sources + tiered resolution
    source_priority: list[FactSource] = field(default_factory=lambda: [
        FactSource.CACHE,
        FactSource.RULE,
        FactSource.DATABASE,
        FactSource.SUB_PLAN,
        FactSource.DOCUMENT,
        FactSource.LLM_KNOWLEDGE,
        FactSource.USER_PROVIDED,
    ])

    # Confidence thresholds
    min_confidence: float = 0.0  # Accept any confidence
    prefer_database: bool = True  # Prefer DB over LLM when both possible

    # Sub-plan / Derivation settings
    allow_sub_plans: bool = True
    max_sub_plan_depth: int = 3  # Prevent infinite recursion
    require_multi_input_derivation: bool = True  # Derivation must have 2+ inputs (no synonyms)

    # Legacy: parallel_io_sources - now always parallel in tiered mode
    parallel_io_sources: bool = False

    # Use new tiered resolution (vs legacy sequential)
    use_tiered_resolution: bool = True


@dataclass
class Tier2AssessmentResult:
    """Result of Tier 2 LLM assessment."""
    strategy: Tier2Strategy
    confidence: float
    reasoning: str
    # For DERIVABLE
    formula: Optional[str] = None
    inputs: Optional[list[tuple[str, str]]] = None  # [(input_name, source), ...]
    # For KNOWN
    value: Optional[Any] = None
    caveat: Optional[str] = None
    # For USER_REQUIRED
    question: Optional[str] = None


from constat.prompts import load_prompt

TIER2_ASSESSMENT_PROMPT = load_prompt("tier2_assessment.md")


# Thresholds for storing arrays as tables (to avoid context bloat)
ARRAY_ROW_THRESHOLD = 5  # Store as table if array has > N items
ARRAY_SIZE_THRESHOLD = 1000  # Store as table if JSON size > N chars
