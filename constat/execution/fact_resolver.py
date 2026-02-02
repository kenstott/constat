# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Lazy fact resolution with provenance tracking.

This module provides on-demand fact resolution during plan execution.
Facts are resolved lazily when accessed, with automatic provenance tracking
for explainability.

Architecture:
1. Top-level plan is generated with assumed facts (e.g., "customer_ltv(X)")
2. During execution, when a fact is needed, the resolver:
   - Checks cache (already resolved this session)
   - Checks rules (Python functions that can derive the fact)
   - Tries database query (LLM generates SQL)
   - Tries LLM knowledge (world facts, heuristics)
   - Falls back to sub-plan generation (for complex derived facts)
3. Each resolution records provenance for explainability

This is an opt-in feature. Simple queries can run without it.

Parallel Resolution (AsyncFactResolver):
For I/O-bound fact resolution (database queries, LLM calls), use AsyncFactResolver
which provides:
- resolve_async(): Async single fact resolution
- resolve_many_async(): Parallel resolution of multiple facts (3-5x speedup)
- Parallel source resolution: Try DATABASE + LLM_KNOWLEDGE + SUB_PLAN concurrently
"""

from __future__ import annotations

import asyncio
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union
from datetime import datetime

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


# Tier 2 Assessment Prompt Template
TIER2_ASSESSMENT_PROMPT = """
Tier 1 resolution failed for: {fact_name}
Description: {fact_description}

Resolved premises in current plan:
{resolved_premises}

Pending premises in current plan:
{pending_premises}

Available data sources (already searched, fact not found directly):
{available_sources}

Assess the best resolution strategy:

STRATEGY: DERIVABLE | KNOWN | USER_REQUIRED

CRITICAL: DERIVABLE requires a plan with 2+ DISTINCT inputs being composed.
- Valid: "X = A / B" (two inputs composed with formula)
- Valid: "X = filter(A, condition from B)" (two inputs)
- INVALID: "try looking up synonym Y instead" (single lookup, REJECTED)
- INVALID: "search for alternative_name" (synonym hunting, REJECTED)

If you cannot devise a formula with 2+ distinct inputs, do NOT use DERIVABLE.

CONFIDENCE: 0.0-1.0
REASONING: <brief explanation of why this strategy>

If DERIVABLE:
  FORMULA: <computation formula, must reference 2+ inputs>
  INPUTS: <list of (input_name, source) tuples, e.g., [("salaries", "premise:P1"), ("industry_avg", "llm_knowledge")]>

If KNOWN:
  VALUE: <the answer - only use for general/industry knowledge you're confident about>
  CAVEAT: <any limitations or uncertainty>

If USER_REQUIRED:
  QUESTION: <clear question to ask the user>

Respond in valid JSON format:
{{
  "strategy": "DERIVABLE" | "KNOWN" | "USER_REQUIRED",
  "confidence": 0.0-1.0,
  "reasoning": "...",
  "formula": "..." or null,
  "inputs": [["name", "source"], ...] or null,
  "value": ... or null,
  "caveat": "..." or null,
  "question": "..." or null
}}
"""


# Thresholds for storing arrays as tables (to avoid context bloat)
ARRAY_ROW_THRESHOLD = 5  # Store as table if array has > N items
ARRAY_SIZE_THRESHOLD = 1000  # Store as table if JSON size > N chars


class FactResolver:
    """
    Lazy fact resolver with provenance tracking.

    Usage:
        resolver = FactResolver(llm=provider, schema_manager=sm)

        # Register custom rules
        @resolver.rule("customer_ltv")
        def calc_ltv(resolver, customer_id: str) -> Fact:
            transactions = resolver.resolve("customer_transactions", customer_id=customer_id)
            return Fact(
                name=f"customer_ltv:{customer_id}",
                value=sum(t["amount"] for t in transactions.value),
                confidence=transactions.confidence,
                source=FactSource.RULE,
                rule_name="calc_ltv",
                because=[transactions]
            )

        # Resolve facts (lazy, cached)
        ltv = resolver.resolve("customer_ltv", customer_id="acme")
        print(ltv.derivation_trace)
    """

    def __init__(
        self,
        llm=None,  # LLM provider for queries/knowledge
        schema_manager=None,  # For database queries
        config=None,  # For config-based facts
        strategy: Optional[ResolutionStrategy] = None,
        event_callback=None,  # Callback for resolution events (for display updates)
        datastore=None,  # For storing large array facts as tables
        doc_tools=None,  # For document search/retrieval
        learning_callback=None,  # Callback for learning from error fixes
    ):
        self.llm = llm
        self.schema_manager = schema_manager
        self.config = config
        self.strategy = strategy or ResolutionStrategy()
        self._event_callback = event_callback
        self._datastore = datastore  # Reference to session's datastore for table storage
        self._doc_tools = doc_tools  # For document-based fact resolution
        self._learning_callback = learning_callback  # For capturing learnings from error fixes

        # Caches
        self._cache: dict[str, Fact] = {}
        self._rules: dict[str, RuleFunction] = {}

        # Resolution state (for sub-plan depth tracking)
        self._resolution_depth: int = 0

        # Deadline for Tier 1 parallel resolution (None = no deadline)
        self._resolution_deadline: Optional[float] = None

        # All resolutions this session (for audit)
        self.resolution_log: list[Fact]  = []

    def _emit_event(self, event_type: str, data: dict) -> None:
        """Emit a resolution event if callback is registered."""
        if self._event_callback:
            self._event_callback(event_type, data)

    def rule(self, fact_pattern: str):
        """Decorator to register a rule function for a fact pattern.

        Example:
            @resolver.rule("customer_ltv")
            def calc_ltv(resolver, customer_id: str) -> Fact:
                ...
        """
        def decorator(func: RuleFunction) -> RuleFunction:
            self._rules[fact_pattern] = func
            return func
        return decorator

    def register_rule(self, fact_pattern: str, func: RuleFunction) -> None:
        """Register a rule function programmatically."""
        self._rules[fact_pattern] = func

    def set_resolution_context(
        self,
        resolved_premises: Optional[dict[str, "Fact"]] = None,
        pending_premises: Optional[list[dict]] = None,
        available_sources: Optional[str] = None,
    ) -> None:
        """Set context for Tier 2 LLM assessment.

        Called by session before resolving premises to provide context about
        the current plan's premises.

        Args:
            resolved_premises: Dict of fact_id -> Fact for already resolved premises
            pending_premises: List of premise dicts still to be resolved
            available_sources: Description of available data sources
        """
        self._resolution_context = {
            "resolved_premises": resolved_premises or {},
            "pending_premises": pending_premises or [],
            "available_sources": available_sources or "",
        }

    def resolve_tiered(
        self,
        fact_name: str,
        fact_description: str = "",
        **params,
    ) -> tuple[Fact, Optional[Tier2AssessmentResult]]:
        """
        Resolve a fact using the tiered resolution architecture.

        Tier 1: Parallel local sources (cache, config, rules, docs, database)
        Tier 2: LLM assessment (DERIVABLE, KNOWN, or USER_REQUIRED)

        Args:
            fact_name: The fact to resolve
            fact_description: Human-readable description of what this fact represents
            **params: Parameters for the fact

        Returns:
            Tuple of (Fact, Tier2AssessmentResult or None)
            - If Tier 1 succeeds: (resolved_fact, None)
            - If Tier 2 needed: (fact_or_unresolved, assessment_result)
        """
        import logging
        import time
        logger = logging.getLogger(__name__)

        cache_key = self._cache_key(fact_name, params)
        logger.info(f"[TIERED] Starting tiered resolution for: {cache_key}")
        logger.debug(f"resolve_tiered called for: {fact_name}, tier1_sources: {[s.value for s in self.strategy.tier1_sources]}")

        # Emit fact_start event for DAG visualization
        self._emit_event("fact_start", {
            "fact_name": cache_key,
            "fact_description": fact_description,
            "parameters": params,
            "status": "pending",
        })

        # Quick cache check BEFORE parallel race (avoids unnecessary work)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached and cached.is_resolved and cached.confidence >= self.strategy.min_confidence:
                logger.info(f"[TIERED] Cache hit for {cache_key}")
                self.resolution_log.append(cached)  # Log cache hits for audit trail
                return cached, None
            elif cached and cached.is_resolved:
                logger.debug(f"[TIERED] Cache hit but confidence {cached.confidence} < {self.strategy.min_confidence}")
                # Fall through to try other sources

        # ═══════════════════════════════════════════════════════════════════
        # TIER 1: Parallel Local Sources
        # ═══════════════════════════════════════════════════════════════════
        self._emit_event("fact_planning", {
            "fact_name": cache_key,
            "planning_type": "tier1_parallel",
            "sources": [s.value for s in self.strategy.tier1_sources],
            "status": "planning",
        })

        tier1_start = time.time()
        tier1_result = self._resolve_tier1_parallel(fact_name, params, cache_key)
        tier1_elapsed = time.time() - tier1_start

        if tier1_result and tier1_result.is_resolved:
            logger.info(f"[TIERED] Tier 1 resolved {cache_key} in {tier1_elapsed:.2f}s: {tier1_result.value}")
            self._emit_event("fact_resolved", {
                "fact_name": cache_key,
                "value": tier1_result.value,
                "source": tier1_result.source.value if tier1_result.source else "unknown",
                "confidence": tier1_result.confidence,
                "tier": 1,
                "elapsed_ms": int(tier1_elapsed * 1000),
                "status": "resolved",
            })
            return tier1_result, None

        logger.info(f"[TIERED] Tier 1 failed for {cache_key} after {tier1_elapsed:.2f}s, proceeding to Tier 2")

        # ═══════════════════════════════════════════════════════════════════
        # TIER 2: LLM Assessment
        # ═══════════════════════════════════════════════════════════════════
        self._emit_event("fact_planning", {
            "fact_name": cache_key,
            "planning_type": "tier2_assessment",
            "reason": "tier1_failed",
            "status": "planning",
        })

        assessment = self._assess_tier2_strategy(fact_name, fact_description, params)

        if assessment is None:
            # LLM assessment failed - return unresolved
            logger.warning(f"[TIERED] Tier 2 assessment failed for {cache_key}")
            self._emit_event("fact_failed", {
                "fact_name": cache_key,
                "reason": "tier2_assessment_failed",
                "status": "failed",
            })
            unresolved = Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
                reasoning="Tier 1 failed, Tier 2 assessment failed",
            )
            self.resolution_log.append(unresolved)  # Log unresolved facts too
            return unresolved, None

        logger.info(f"[TIERED] Tier 2 assessment: {assessment.strategy.value} (confidence: {assessment.confidence})")

        # Handle based on assessment strategy
        if assessment.strategy == Tier2Strategy.KNOWN:
            # LLM provided the answer directly
            fact = Fact(
                name=cache_key,
                value=assessment.value,
                confidence=assessment.confidence,
                source=FactSource.LLM_KNOWLEDGE,
                reasoning=f"LLM knowledge: {assessment.reasoning}",
                context=assessment.caveat,
            )
            self._cache[cache_key] = fact
            self.resolution_log.append(fact)
            self._emit_event("fact_resolved", {
                "fact_name": cache_key,
                "value": assessment.value,
                "source": "llm_knowledge",
                "confidence": assessment.confidence,
                "tier": 2,
                "strategy": "known",
                "status": "resolved",
            })
            return fact, assessment

        elif assessment.strategy == Tier2Strategy.DERIVABLE:
            # Attempt derivation with the formula
            self._emit_event("fact_executing", {
                "fact_name": cache_key,
                "execution_type": "derivation",
                "formula": assessment.formula,
                "status": "executing",
            })
            derived_fact = self._execute_derivation(
                fact_name, params, cache_key, assessment
            )
            if derived_fact and derived_fact.is_resolved:
                self._emit_event("fact_resolved", {
                    "fact_name": cache_key,
                    "value": derived_fact.value,
                    "source": derived_fact.source.value if derived_fact.source else "derived",
                    "confidence": derived_fact.confidence,
                    "tier": 2,
                    "strategy": "derivable",
                    "dependencies": [f.name for f in derived_fact.because] if derived_fact.because else [],
                    "status": "resolved",
                })
                return derived_fact, assessment
            # Derivation failed - return assessment for caller to handle
            self._emit_event("fact_failed", {
                "fact_name": cache_key,
                "reason": "derivation_failed",
                "formula": assessment.formula,
                "status": "failed",
            })
            unresolved = Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
                reasoning=f"Derivation failed: {assessment.formula}",
            )
            return unresolved, assessment

        elif assessment.strategy == Tier2Strategy.USER_REQUIRED:
            # Return unresolved with assessment - session layer handles user prompt
            unresolved = Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
                reasoning=f"User input required: {assessment.question}",
            )
            return unresolved, assessment

        # Fallback
        unresolved = Fact(
            name=cache_key,
            value=None,
            confidence=0.0,
            source=FactSource.UNRESOLVED,
            reasoning=f"Unknown Tier 2 strategy: {assessment.strategy}",
        )
        return unresolved, assessment

    def _resolve_tier1_parallel(
        self,
        fact_name: str,
        params: dict,
        cache_key: str,
    ) -> Optional[Fact]:
        """
        Tier 1: Race all local sources in parallel with timeout.

        Sources: cache, config, rules, documents, database
        All run concurrently, first successful result wins.
        """
        import logging
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
        import time
        logger = logging.getLogger(__name__)

        timeout = self.strategy.tier1_timeout
        sources = self.strategy.tier1_sources
        logger.debug(f"[TIER1] Racing sources: {[s.value for s in sources]} with {timeout}s timeout")

        def try_source(source: FactSource) -> tuple[FactSource, Optional[Fact], float]:
            """Try a single source, return (source, fact, elapsed_time)."""
            start = time.time()
            try:
                fact = self._try_resolve(source, fact_name, params, cache_key)
                elapsed = time.time() - start
                if fact:
                    logger.info(f"[TIER1] {source.value} returned fact: resolved={fact.is_resolved}, value_type={type(fact.value).__name__}")
                else:
                    logger.info(f"[TIER1] {source.value} returned None")
                return (source, fact, elapsed)
            except Exception as e:
                elapsed = time.time() - start
                import traceback
                logger.warning(f"[TIER1] {source.value} raised {type(e).__name__}: {e}")
                logger.debug(f"[TIER1] {source.value} traceback: {traceback.format_exc()}")
                return (source, None, elapsed)

        results: list[tuple[FactSource, Fact, float]] = []
        sources_tried: list[str] = []

        with ThreadPoolExecutor(max_workers=len(sources)) as executor:
            futures = {executor.submit(try_source, s): s for s in sources}

            try:
                for future in as_completed(futures, timeout=timeout):
                    source, fact, elapsed = future.result()

                    if fact is None:
                        sources_tried.append(f"{source.value}:no_result({elapsed:.2f}s)")
                        logger.debug(f"[TIER1] {source.value}: no result in {elapsed:.2f}s")
                    elif not fact.is_resolved:
                        sources_tried.append(f"{source.value}:unresolved({elapsed:.2f}s)")
                        logger.debug(f"[TIER1] {source.value}: unresolved in {elapsed:.2f}s")
                    elif fact.confidence < self.strategy.min_confidence:
                        sources_tried.append(f"{source.value}:low_conf({fact.confidence:.2f})")
                        logger.debug(f"[TIER1] {source.value}: low confidence {fact.confidence}")
                    else:
                        # Valid result
                        sources_tried.append(f"{source.value}:SUCCESS({elapsed:.2f}s)")
                        results.append((source, fact, elapsed))
                        logger.debug(f"[TIER1] {source.value}: success in {elapsed:.2f}s, conf={fact.confidence}")

            except TimeoutError:
                logger.warning(f"[TIER1] Timeout after {timeout}s, using available results")
                # Cancel remaining futures
                for future in futures:
                    if not future.done():
                        future.cancel()

        if not results:
            logger.debug(f"[TIER1] No valid results. Tried: {' → '.join(sources_tried)}")
            return None

        # Pick best result: highest confidence, then by source priority
        source_priority = {s: i for i, s in enumerate(sources)}
        results.sort(key=lambda x: (-x[1].confidence, source_priority.get(x[0], 999)))

        best_source, best_fact, best_elapsed = results[0]
        logger.info(f"[TIER1] Selected {best_source.value} (conf={best_fact.confidence:.2f}, {best_elapsed:.2f}s)")

        # Cache and log
        self._cache[cache_key] = best_fact
        self.resolution_log.append(best_fact)
        return best_fact

    def _assess_tier2_strategy(
        self,
        fact_name: str,
        fact_description: str,
        params: dict,
    ) -> Optional[Tier2AssessmentResult]:
        """
        Tier 2: LLM assessment of best resolution strategy.

        Returns DERIVABLE (with formula), KNOWN (with value), or USER_REQUIRED.
        """
        import logging
        import json
        logger = logging.getLogger(__name__)

        if not self.llm:
            logger.warning("[TIER2] No LLM configured, cannot assess")
            return None

        # Build context from resolution context (set by session)
        ctx = getattr(self, "_resolution_context", {})
        resolved_premises = ctx.get("resolved_premises", {})
        pending_premises = ctx.get("pending_premises", [])
        available_sources = ctx.get("available_sources", "")

        # Format resolved premises
        resolved_str = "\n".join([
            f"  - {pid}: {fact.name} = {str(fact.value)[:100]} (source: {fact.source.value})"
            for pid, fact in resolved_premises.items()
        ]) or "  (none yet)"

        # Format pending premises
        pending_str = "\n".join([
            f"  - {p.get('id', '?')}: {p.get('name', '?')} ({p.get('description', '')})"
            for p in pending_premises
        ]) or "  (none)"

        # Build prompt
        prompt = TIER2_ASSESSMENT_PROMPT.format(
            fact_name=fact_name,
            fact_description=fact_description or fact_name,
            resolved_premises=resolved_str,
            pending_premises=pending_str,
            available_sources=available_sources or "(see system context)",
        )

        logger.debug(f"[TIER2] Assessment prompt:\n{prompt}")

        try:
            response = self.llm.generate(
                system="You assess fact resolution strategies. Respond only with valid JSON.",
                user_message=prompt,
                max_tokens=500,
            )

            # Parse JSON response
            # Handle markdown code blocks if present
            response_text = response.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            data = json.loads(response_text)

            strategy_str = data.get("strategy", "").upper()
            if strategy_str == "DERIVABLE":
                strategy = Tier2Strategy.DERIVABLE
            elif strategy_str == "KNOWN":
                strategy = Tier2Strategy.KNOWN
            elif strategy_str == "USER_REQUIRED":
                strategy = Tier2Strategy.USER_REQUIRED
            else:
                logger.warning(f"[TIER2] Unknown strategy: {strategy_str}")
                return None

            # Validate DERIVABLE has 2+ inputs
            if strategy == Tier2Strategy.DERIVABLE:
                inputs = data.get("inputs", [])
                if self.strategy.require_multi_input_derivation and len(inputs) < 2:
                    logger.warning(f"[TIER2] DERIVABLE rejected: only {len(inputs)} inputs (need 2+)")
                    # Downgrade to USER_REQUIRED
                    return Tier2AssessmentResult(
                        strategy=Tier2Strategy.USER_REQUIRED,
                        confidence=0.5,
                        reasoning=f"Derivation rejected: needs 2+ inputs, got {len(inputs)}. User input required.",
                        question=f"What is the value for '{fact_name}'? ({fact_description})",
                    )

            return Tier2AssessmentResult(
                strategy=strategy,
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                formula=data.get("formula"),
                inputs=data.get("inputs"),
                value=data.get("value"),
                caveat=data.get("caveat"),
                question=data.get("question"),
            )

        except json.JSONDecodeError as e:
            logger.error(f"[TIER2] Failed to parse JSON: {e}\nResponse: {response}")
            return None
        except Exception as e:
            logger.error(f"[TIER2] Assessment failed: {e}")
            return None

    def _execute_derivation(
        self,
        fact_name: str,
        params: dict,
        cache_key: str,
        assessment: Tier2AssessmentResult,
    ) -> Optional[Fact]:
        """
        Execute a derivation based on Tier 2 assessment.

        Resolves the input facts and applies the formula.
        """
        import logging
        logger = logging.getLogger(__name__)

        if not assessment.formula or not assessment.inputs:
            logger.warning("[DERIVATION] No formula or inputs provided")
            return None

        if len(assessment.inputs) < 2:
            logger.warning("[DERIVATION] Derivation requires 2+ inputs")
            return None

        logger.info(f"[DERIVATION] Executing: {assessment.formula}")
        logger.info(f"[DERIVATION] Inputs: {assessment.inputs}")

        # Resolve each input
        resolved_inputs: dict[str, Fact] = {}
        ctx = getattr(self, "_resolution_context", {})
        resolved_premises = ctx.get("resolved_premises", {})

        for input_name, source in assessment.inputs:
            # Check if it's a reference to an existing premise
            if source.startswith("premise:"):
                premise_id = source.split(":")[1]
                if premise_id in resolved_premises:
                    resolved_inputs[input_name] = resolved_premises[premise_id]
                    continue

            # Check if it's LLM knowledge
            if source == "llm_knowledge":
                # Resolve via LLM knowledge
                knowledge_fact = self._resolve_from_llm(input_name, params)
                if knowledge_fact:
                    resolved_inputs[input_name] = knowledge_fact
                    continue

            # Try to resolve from other sources
            # Use non-tiered resolve to avoid infinite recursion
            self._resolution_depth += 1
            try:
                fact = self._resolve_legacy(input_name, params)
                if fact and fact.is_resolved:
                    resolved_inputs[input_name] = fact
            finally:
                self._resolution_depth -= 1

        # Check if all inputs resolved
        missing = [name for name, _ in assessment.inputs if name not in resolved_inputs]
        if missing:
            logger.warning(f"[DERIVATION] Failed to resolve inputs: {missing}")
            return None

        # Execute the formula
        # Build execution context with resolved values
        exec_context = {
            name: fact.value for name, fact in resolved_inputs.items()
        }

        try:
            # Simple formula evaluation
            # Security: only allow basic math operations
            allowed_names = {"__builtins__": {"min": min, "max": max, "sum": sum, "len": len, "abs": abs}}
            allowed_names.update(exec_context)

            result = eval(assessment.formula, allowed_names)

            # Calculate confidence as min of inputs
            min_confidence = min(f.confidence for f in resolved_inputs.values())

            fact = Fact(
                name=cache_key,
                value=result,
                confidence=min_confidence * 0.95,  # Slight reduction for derivation
                source=FactSource.DERIVED,
                reasoning=f"Derived: {assessment.formula}",
                because=list(resolved_inputs.values()),
            )

            self._cache[cache_key] = fact
            self.resolution_log.append(fact)
            return fact

        except Exception as e:
            logger.error(f"[DERIVATION] Formula execution failed: {e}")
            return None

    def _resolve_legacy(self, fact_name: str, params: dict) -> Fact:
        """Legacy sequential resolution (used by derivation to avoid recursion)."""
        cache_key = self._cache_key(fact_name, params)

        # Try each source in order
        for source in self.strategy.source_priority:
            if source == FactSource.SUB_PLAN:
                continue  # Skip sub-plan in legacy mode to avoid recursion
            try:
                fact = self._try_resolve(source, fact_name, params, cache_key)
                if fact and fact.is_resolved:
                    return fact
            except Exception as e:
                logger.debug(f"[_resolve_legacy] Source {source.value} failed for {fact_name}: {e}")
                continue

        return Fact(
            name=cache_key,
            value=None,
            confidence=0.0,
            source=FactSource.UNRESOLVED,
            reasoning="Legacy resolution failed",
        )

    def resolve(self, fact_name: str, **params) -> Fact:
        """
        Resolve a fact by name, trying sources in priority order.

        Args:
            fact_name: The fact to resolve (e.g., "customer_ltv", "revenue_threshold")
            **params: Parameters for the fact (e.g., customer_id="acme")

        Returns:
            Fact with value, confidence, and provenance
        """
        import logging
        from concurrent.futures import ThreadPoolExecutor, as_completed
        logger = logging.getLogger(__name__)

        # Use tiered resolution if enabled
        if self.strategy.use_tiered_resolution:
            fact, assessment = self.resolve_tiered(fact_name, **params)
            # Note: assessment contains Tier 2 result if needed by caller
            # For backward compatibility, just return the fact
            return fact

        # Legacy sequential resolution below
        # Build cache key from name + params
        cache_key = self._cache_key(fact_name, params)
        logger.debug(f"[FACT_RESOLVER] Resolving: {cache_key}")

        # Separate sources by cost/speed:
        # - Fast: CACHE, RULE, CONFIG (sync, instant)
        # - Cheap I/O: DATABASE, DOCUMENT (can parallelize)
        # - Expensive: LLM_KNOWLEDGE (API cost + latency, use as fallback)
        fast_sources = {FactSource.CACHE, FactSource.RULE, FactSource.CONFIG}
        cheap_io_sources = {FactSource.DATABASE, FactSource.DOCUMENT, FactSource.SUB_PLAN}
        expensive_sources = {FactSource.LLM_KNOWLEDGE}

        sources_tried = []

        # Phase 1: Try fast sources first (serial, quick)
        for source in self.strategy.source_priority:
            if source not in fast_sources:
                continue
            logger.debug(f"[FACT_RESOLVER] Trying fast source: {source.value}")
            try:
                fact = self._try_resolve(source, fact_name, params, cache_key)
                if fact and fact.is_resolved and fact.confidence >= self.strategy.min_confidence:
                    sources_tried.append(f"{source.value}:SUCCESS")
                    self._cache[cache_key] = fact
                    self.resolution_log.append(fact)
                    logger.info(f"[FACT_RESOLVER] Resolved {cache_key} via {source.value}: {fact.value}")
                    return fact
                elif fact is None:
                    sources_tried.append(f"{source.value}:no_result")
                elif not fact.is_resolved:
                    sources_tried.append(f"{source.value}:unresolved")
                else:
                    sources_tried.append(f"{source.value}:low_conf({fact.confidence})")
            except Exception as e:
                sources_tried.append(f"{source.value}:ERROR({type(e).__name__})")
                logger.debug(f"[FACT_RESOLVER] Error in {source.value}: {e}")

        # Phase 2: Try cheap I/O sources (DATABASE, DOCUMENT)
        cheap_io_list = [s for s in self.strategy.source_priority if s in cheap_io_sources]

        if self.strategy.parallel_io_sources and len(cheap_io_list) > 1:
            # Parallel resolution of cheap I/O sources
            logger.debug(f"[FACT_RESOLVER] Trying cheap I/O in parallel: {[s.value for s in cheap_io_list]}")
            fact = self._resolve_io_parallel(fact_name, params, cache_key, cheap_io_list, sources_tried)
            if fact:
                return fact
        else:
            # Serial cheap I/O resolution
            for source in cheap_io_list:
                logger.debug(f"[FACT_RESOLVER] Trying source: {source.value}")
                try:
                    fact = self._try_resolve(source, fact_name, params, cache_key)
                    if fact is None:
                        sources_tried.append(f"{source.value}:no_result")
                        logger.debug(f"[FACT_RESOLVER] Source {source.value} returned None - continuing to next")
                    elif not fact.is_resolved:
                        sources_tried.append(f"{source.value}:unresolved")
                        logger.debug(f"[FACT_RESOLVER] Source {source.value} returned unresolved fact - continuing")
                    elif fact.confidence < self.strategy.min_confidence:
                        sources_tried.append(f"{source.value}:low_conf({fact.confidence})")
                        logger.debug(f"[FACT_RESOLVER] Source {source.value} confidence {fact.confidence} "
                                    f"below threshold {self.strategy.min_confidence} - continuing")
                    else:
                        # Success!
                        sources_tried.append(f"{source.value}:SUCCESS")
                        self._cache[cache_key] = fact
                        self.resolution_log.append(fact)
                        logger.info(f"[FACT_RESOLVER] Resolved {cache_key} via {source.value}: {fact.value}")
                        return fact
                except Exception as e:
                    sources_tried.append(f"{source.value}:ERROR({type(e).__name__})")
                    logger.error(f"[FACT_RESOLVER] Error in source {source.value}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

        # Phase 3: Expensive fallback (LLM_KNOWLEDGE) - only if cheap sources failed
        expensive_list = [s for s in self.strategy.source_priority if s in expensive_sources]
        for source in expensive_list:
            logger.debug(f"[FACT_RESOLVER] Trying expensive fallback: {source.value}")
            try:
                fact = self._try_resolve(source, fact_name, params, cache_key)
                if fact is None:
                    sources_tried.append(f"{source.value}:no_result")
                elif not fact.is_resolved:
                    sources_tried.append(f"{source.value}:unresolved")
                elif fact.confidence < self.strategy.min_confidence:
                    sources_tried.append(f"{source.value}:low_conf({fact.confidence})")
                else:
                    sources_tried.append(f"{source.value}:SUCCESS")
                    self._cache[cache_key] = fact
                    self.resolution_log.append(fact)
                    logger.info(f"[FACT_RESOLVER] Resolved {cache_key} via {source.value}: {fact.value}")
                    return fact
            except Exception as e:
                sources_tried.append(f"{source.value}:ERROR({type(e).__name__})")
                logger.error(f"[FACT_RESOLVER] Error in expensive source {source.value}: {e}")

        # Could not resolve
        sources_summary = " → ".join(sources_tried)
        logger.debug(f"[FACT_RESOLVER] Could not resolve: {cache_key}")
        logger.debug(f"[FACT_RESOLVER] Sources tried: {sources_summary}")
        unresolved = Fact(
            name=cache_key,
            value=None,
            confidence=0.0,
            source=FactSource.UNRESOLVED,
            reasoning=f"Could not resolve fact: {fact_name}. Sources: {sources_summary}"
        )
        self.resolution_log.append(unresolved)
        return unresolved

    def _resolve_io_parallel(
        self,
        fact_name: str,
        params: dict,
        cache_key: str,
        io_sources: list[FactSource],
        sources_tried: list[str],
    ) -> Optional[Fact]:
        """
        Run I/O-bound sources in parallel and pick the best result.

        Uses ThreadPoolExecutor for true parallelism in synchronous code.
        Selection: prioritizes by source order, uses confidence as tiebreaker.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import logging
        logger = logging.getLogger(__name__)

        def try_source(source: FactSource) -> tuple[FactSource, Optional[Fact]]:
            try:
                fact = self._try_resolve(source, fact_name, params, cache_key)
                return (source, fact)
            except Exception as e:
                logger.debug(f"[_resolve_io_parallel] {source.value} raised: {e}")
                return (source, None)

        # Run all I/O sources in parallel
        valid_results: list[tuple[int, float, Fact, FactSource]] = []

        with ThreadPoolExecutor(max_workers=len(io_sources)) as executor:
            futures = {executor.submit(try_source, s): s for s in io_sources}

            for future in as_completed(futures):
                source, fact = future.result()
                if fact is None:
                    sources_tried.append(f"{source.value}:no_result")
                elif not fact.is_resolved:
                    sources_tried.append(f"{source.value}:unresolved")
                elif fact.confidence < self.strategy.min_confidence:
                    sources_tried.append(f"{source.value}:low_conf({fact.confidence})")
                else:
                    # Valid result - store with priority index
                    priority_idx = io_sources.index(source)
                    valid_results.append((priority_idx, fact.confidence, fact, source))
                    sources_tried.append(f"{source.value}:conf={fact.confidence:.2f}")
                    logger.debug(f"[_resolve_io_parallel] {source.value}: conf={fact.confidence:.2f}")

        if not valid_results:
            return None

        # Pick best: sort by (priority_index, -confidence)
        valid_results.sort(key=lambda x: (x[0], -x[1]))
        best_priority, best_conf, best_fact, best_source = valid_results[0]

        sources_tried.append(f"{best_source.value}:SELECTED")
        logger.info(f"[_resolve_io_parallel] Selected {best_source.value} with confidence {best_conf:.2f}")

        self._cache[cache_key] = best_fact
        self.resolution_log.append(best_fact)
        return best_fact

    def _cache_key(self, fact_name: str, params: dict) -> str:
        """Generate cache key from fact name and params.

        Note: Excludes metadata-only params (fact_description) from cache key
        since they don't affect the resolved value.
        """
        if not params:
            return fact_name
        # Exclude metadata params that don't affect resolution
        cache_params = {k: v for k, v in params.items() if k not in ("fact_description",)}
        if not cache_params:
            return fact_name
        param_str = ",".join(f"{k}={v}" for k, v in sorted(cache_params.items()))
        return f"{fact_name}({param_str})"

    def get_fact(self, name: str, verify_tables: bool = True) -> Optional[Fact]:
        """
        Get a specific fact by name, optionally verifying table existence.

        Args:
            name: The fact name to retrieve
            verify_tables: If True, verify table facts have their tables in datastore

        Returns:
            The Fact object if found and valid, None otherwise
        """
        cached = self._cache.get(name)
        if cached and verify_tables:
            # For table facts, verify the table still exists in datastore
            if cached.table_name and self._datastore:
                try:
                    existing_tables = [t["name"] for t in self._datastore.list_tables()]
                    if cached.table_name not in existing_tables:
                        return None  # Table doesn't exist, fact is invalid
                except Exception as e:
                    logger.debug(f"[get] Cache validation failed for {name}, assuming valid: {e}")
        return cached

    def get_all_facts(self) -> dict[str, Fact]:
        """
        Get all cached facts.

        Returns:
            Dictionary mapping fact names/keys to Fact objects
        """
        return dict(self._cache)

    def get_facts_for_role(
        self,
        role_id: Optional[str] = None,
        include_shared: bool = True,
    ) -> dict[str, Fact]:
        """
        Get facts filtered by role scope.

        Args:
            role_id: Role to filter by. None = shared facts only.
            include_shared: If True and role_id is set, also include shared facts.

        Returns:
            Dictionary mapping fact names to Fact objects
        """
        if role_id is None:
            # Return only shared facts
            return {k: v for k, v in self._cache.items() if v.role_id is None}

        result = {}
        for name, fact in self._cache.items():
            if fact.role_id == role_id:
                result[name] = fact
            elif include_shared and fact.role_id is None:
                result[name] = fact

        return result

    def get_shared_facts(self) -> dict[str, Fact]:
        """Get only shared facts (role_id=None)."""
        return {k: v for k, v in self._cache.items() if v.role_id is None}

    def get_role_facts(self, role_id: str) -> dict[str, Fact]:
        """Get only facts for a specific role (excludes shared)."""
        return {k: v for k, v in self._cache.items() if v.role_id == role_id}

    def promote_fact_to_shared(self, name: str) -> bool:
        """
        Promote a role-scoped fact to shared context.

        Used when final results from a role should be available globally.

        Args:
            name: Fact name to promote

        Returns:
            True if promoted, False if not found
        """
        if name not in self._cache:
            return False

        fact = self._cache[name]
        # Create a new fact with role_id=None (immutable dataclass workaround)
        promoted = Fact(
            name=fact.name,
            value=fact.value,
            confidence=fact.confidence,
            source=fact.source,
            because=fact.because,
            description=fact.description,
            source_name=fact.source_name,
            query=fact.query,
            api_endpoint=fact.api_endpoint,
            rule_name=fact.rule_name,
            reasoning=fact.reasoning,
            resolved_at=fact.resolved_at,
            table_name=fact.table_name,
            row_count=fact.row_count,
            context=fact.context,
            role_id=None,  # Promoted to shared
        )
        self._cache[name] = promoted
        return True

    # =========================================================================
    # Declarative Resolution (spec-based)
    # =========================================================================

    def resolve_with_spec(
        self,
        fact_name: str,
        params: dict = None,
        build_proof: bool = True,
    ) -> tuple[Fact, Optional[ProofNode]]:
        """
        Resolve a fact using declarative specification.

        This is the preferred resolution method for auditable mode:
        1. Ask LLM to generate a ResolutionSpec (dependencies + logic)
        2. Resolve all dependencies first (recursive)
        3. Execute sandboxed logic with only resolved facts
        4. Build proof tree automatically

        Args:
            fact_name: The fact to resolve
            params: Parameters for the fact
            build_proof: Whether to build a proof tree (for auditable mode)

        Returns:
            Tuple of (Fact, ProofNode) - proof is None if build_proof=False
        """
        import logging
        logger = logging.getLogger(__name__)
        params = params or {}

        cache_key = self._cache_key(fact_name, params)
        logger.debug(f"[resolve_with_spec] Resolving: {cache_key}")

        # Check cache first
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            proof = ProofNode(
                conclusion=f"{fact_name} = {cached.display_value}",
                source=FactSource.CACHE,
                confidence=cached.confidence,
            ) if build_proof else None
            return cached, proof

        # Generate resolution spec from LLM
        spec = self._generate_resolution_spec(fact_name, params)
        if spec is None:
            unresolved = Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
                reasoning=f"Could not generate resolution spec for: {fact_name}"
            )
            return unresolved, None

        # Resolve using the spec
        return self._resolve_spec(spec, params, build_proof)

    def _generate_resolution_spec(
        self,
        fact_name: str,
        params: dict,
    ) -> Optional[ResolutionSpec]:
        """
        Ask LLM to generate a ResolutionSpec for a fact.

        The spec declares:
        - What dependencies are needed (other facts)
        - How to combine them (logic code)
        - Or, for leaf facts: direct SQL/doc query
        """
        import logging
        import json
        logger = logging.getLogger(__name__)

        if not self.llm:
            logger.debug("[_generate_resolution_spec] No LLM configured")
            return None

        # Get schema info for context
        schema_info = ""
        if self.schema_manager:
            schema_info = self.schema_manager.get_overview()

        # Get database type for SQL hints
        db_type = "sqlite"
        db_names = list(self.config.databases.keys()) if self.config else []
        if db_names and self.config:
            db_config = self.config.databases.get(db_names[0])
            if db_config:
                db_type = db_config.type or "sqlite"

        prompt = f"""Generate a resolution specification for this fact:

Fact: {fact_name}
Parameters: {params}

Database type: {db_type}
Available schema:
{schema_info}

Respond with a JSON object in this exact format:

For a LEAF fact (can be resolved with a single SQL query):
{{
    "fact_name": "{fact_name}",
    "depends_on": [],
    "sql": "SELECT ... FROM ... WHERE ...",
    "source_hint": "database"
}}

For a DERIVED fact (needs other facts first):
{{
    "fact_name": "{fact_name}",
    "depends_on": [
        {{"name": "other_fact", "params": {{}}, "source_hint": "database"}},
        {{"name": "another_fact", "params": {{}}, "source_hint": "database"}}
    ],
    "logic": "def derive(facts):\\n    # facts is a dict of resolved fact values\\n    result = facts['other_fact'] + facts['another_fact']\\n    return result",
    "source_hint": "derived"
}}

IMPORTANT for SQL:
- Use {db_type} syntax
- For SQLite: use strftime('%Y-%m', col), date('now', '-6 months'), etc.
- Do NOT use schema prefixes (use 'customers' not 'sales.customers')

IMPORTANT for logic:
- The derive function receives a 'facts' dict with resolved values
- It can use pandas (available as 'pd') for DataFrames
- It must return the final value (not a Fact object)
- Keep it simple - just combine the resolved facts

Respond with ONLY the JSON object, no explanation."""

        try:
            response = self.llm.generate(
                system="You are a data resolution expert. Generate resolution specs in JSON format.",
                user_message=prompt,
                max_tokens=800,
            )

            # Clean up response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            # Parse JSON
            spec_dict = json.loads(response)

            # Build ResolutionSpec
            depends_on = []
            for dep in spec_dict.get("depends_on", []):
                depends_on.append(FactDependency(
                    name=dep["name"],
                    params=dep.get("params", {}),
                    source_hint=dep.get("source_hint"),
                ))

            spec = ResolutionSpec(
                fact_name=spec_dict.get("fact_name", fact_name),
                depends_on=depends_on,
                logic=spec_dict.get("logic"),
                sql=spec_dict.get("sql"),
                doc_query=spec_dict.get("doc_query"),
                source_hint=spec_dict.get("source_hint"),
            )

            logger.debug(f"[_generate_resolution_spec] Generated spec: {spec}")
            return spec

        except Exception as e:
            logger.error(f"[_generate_resolution_spec] Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _resolve_spec(
        self,
        spec: ResolutionSpec,
        params: dict,
        build_proof: bool = True,
    ) -> tuple[Fact, Optional[ProofNode]]:
        """
        Resolve a fact using its ResolutionSpec.

        For leaf facts: execute SQL or doc query directly
        For derived facts: resolve dependencies, then execute logic
        """
        import logging
        logger = logging.getLogger(__name__)

        cache_key = self._cache_key(spec.fact_name, params)

        # Leaf fact - resolve directly
        if spec.is_leaf:
            if spec.sql:
                return self._execute_leaf_sql(spec, params, build_proof)
            elif spec.doc_query:
                return self._execute_leaf_doc_query(spec, params, build_proof)
            else:
                logger.warning(f"[_resolve_spec] Leaf spec has no sql or doc_query")
                return Fact(
                    name=cache_key,
                    value=None,
                    confidence=0.0,
                    source=FactSource.UNRESOLVED,
                ), None

        # Derived fact - resolve dependencies first
        resolved_facts = {}
        premise_proofs = []
        all_because = []

        for dep in spec.depends_on:
            logger.debug(f"[_resolve_spec] Resolving dependency: {dep.name}")
            dep_fact, dep_proof = self.resolve_with_spec(
                dep.name,
                dep.params,
                build_proof=build_proof,
            )

            if not dep_fact.is_resolved:
                logger.warning(f"[_resolve_spec] Dependency {dep.name} unresolved")
                return Fact(
                    name=cache_key,
                    value=None,
                    confidence=0.0,
                    source=FactSource.UNRESOLVED,
                    reasoning=f"Dependency {dep.name} could not be resolved",
                ), None

            resolved_facts[dep.name] = dep_fact.value
            all_because.append(dep_fact)
            if dep_proof:
                premise_proofs.append(dep_proof)

        # Execute logic with resolved facts
        result, confidence = self._execute_sandboxed_logic(spec.logic, resolved_facts)

        if result is None:
            return Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
                reasoning="Logic execution failed",
            ), None

        # Build the fact
        fact = Fact(
            name=cache_key,
            value=result,
            confidence=confidence,
            source=FactSource.DERIVED,
            because=all_because,
        )

        # Cache it
        self._cache[cache_key] = fact
        self.resolution_log.append(fact)

        # Build proof
        proof = None
        if build_proof:
            proof = ProofNode(
                conclusion=f"{spec.fact_name} = {fact.display_value}",
                source=FactSource.DERIVED,
                evidence=spec.logic,
                premises=premise_proofs,
                confidence=confidence,
            )

        return fact, proof

    def _execute_leaf_sql(
        self,
        spec: ResolutionSpec,
        params: dict,
        build_proof: bool = True,
    ) -> tuple[Fact, Optional[ProofNode]]:
        """Execute SQL for a leaf database fact."""
        import logging
        import pandas as pd
        logger = logging.getLogger(__name__)

        cache_key = self._cache_key(spec.fact_name, params)

        if not self.schema_manager:
            logger.warning("[_execute_leaf_sql] No schema_manager")
            return Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
            ), None

        # Get database connection
        db_names = list(self.config.databases.keys()) if self.config else []
        db_name = db_names[0] if db_names else None
        if not db_name:
            return Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
            ), None

        # Get database type for SQL transformation
        db_type = "sqlite"
        if self.config:
            db_config = self.config.databases.get(db_name)
            if db_config:
                db_type = db_config.type or "sqlite"

        sql = spec.sql

        # Transform SQL for SQLite if needed
        if db_type.lower() == "sqlite":
            import re
            # Strip schema prefixes
            sql = re.sub(r'\b(\w+)\.(\w+)\b', r'\2', sql)
            # Transform date functions
            sql = self._transform_sql_for_sqlite(sql)

        logger.debug(f"[_execute_leaf_sql] Executing: {sql[:200]}...")

        try:
            conn = self.schema_manager.get_connection(db_name)
            result_df = pd.read_sql(sql, conn)

            # Convert result
            if len(result_df) == 1 and len(result_df.columns) == 1:
                value = result_df.iloc[0, 0]
            else:
                value = result_df.to_dict('records')

            fact = Fact(
                name=cache_key,
                value=value,
                confidence=1.0,
                source=FactSource.DATABASE,
                source_name=db_name,
                query=sql,
            )

            # Handle large results
            if self._datastore and isinstance(value, list) and self._should_store_as_table(value):
                table_name, row_count = self._store_value_as_table(spec.fact_name, value, db_name)
                fact.value = f"table:{table_name}"
                fact.table_name = table_name
                fact.row_count = row_count

            self._cache[cache_key] = fact
            self.resolution_log.append(fact)

            # Build proof
            proof = None
            if build_proof:
                proof = ProofNode(
                    conclusion=f"{spec.fact_name} = {fact.display_value}",
                    source=FactSource.DATABASE,
                    source_name=db_name,
                    evidence=sql,
                    confidence=1.0,
                )

            return fact, proof

        except Exception as e:
            logger.error(f"[_execute_leaf_sql] Error: {e}")
            return Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
                reasoning=f"SQL error: {e}",
            ), None

    def _execute_leaf_doc_query(
        self,
        spec: ResolutionSpec,
        params: dict,
        build_proof: bool = True,
    ) -> tuple[Fact, Optional[ProofNode]]:
        """Execute document search for a leaf document fact."""
        import logging
        logger = logging.getLogger(__name__)

        cache_key = self._cache_key(spec.fact_name, params)

        if not self._doc_tools:
            return Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
                reasoning="No document tools configured",
            ), None

        try:
            # Use doc_tools to search
            results = self._doc_tools.search(spec.doc_query)

            fact = Fact(
                name=cache_key,
                value=results,
                confidence=0.8,  # Lower confidence for doc search
                source=FactSource.DOCUMENT,
                query=spec.doc_query,
            )

            self._cache[cache_key] = fact
            self.resolution_log.append(fact)

            proof = None
            if build_proof:
                proof = ProofNode(
                    conclusion=f"{spec.fact_name} = {fact.display_value}",
                    source=FactSource.DOCUMENT,
                    evidence=spec.doc_query,
                    confidence=0.8,
                )

            return fact, proof

        except Exception as e:
            logger.error(f"[_execute_leaf_doc_query] Error: {e}")
            return Fact(
                name=cache_key,
                value=None,
                confidence=0.0,
                source=FactSource.UNRESOLVED,
                reasoning=f"Document search error: {e}",
            ), None

    def _execute_sandboxed_logic(
        self,
        logic: str,
        resolved_facts: dict[str, Any],
    ) -> tuple[Any, float]:
        """
        Execute derivation logic with only resolved facts as input.

        The logic code is sandboxed - it only gets:
        - facts: dict of resolved fact values
        - pd: pandas for DataFrame operations

        Args:
            logic: Python code with a derive(facts) function
            resolved_facts: Dict mapping fact names to their values

        Returns:
            Tuple of (result, confidence)
        """
        import logging
        import pandas as pd
        import numpy as np
        logger = logging.getLogger(__name__)

        if not logic:
            return None, 0.0

        try:
            # Validate syntax
            compile(logic, "<sandboxed_logic>", "exec")

            # Sandboxed namespace - only facts and pandas
            local_ns = {
                "pd": pd,
                "np": np,
            }

            # Execute to define the derive function
            exec(logic, local_ns)

            derive_func = local_ns.get("derive")
            if not derive_func:
                logger.error("[_execute_sandboxed_logic] No 'derive' function found")
                return None, 0.0

            # Call with resolved facts only
            result = derive_func(resolved_facts)

            # Calculate confidence as min of all input facts
            # (we don't have confidence here, assume 1.0 for now)
            confidence = 1.0

            return result, confidence

        except SyntaxError as e:
            logger.error(f"[_execute_sandboxed_logic] Syntax error: {e}")
            return None, 0.0
        except Exception as e:
            logger.error(f"[_execute_sandboxed_logic] Execution error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, 0.0

    def _should_store_as_table(self, value: Any) -> bool:
        """
        Check if a value should be stored as a table instead of inline.

        Uses threshold-based logic to avoid context bloat for large arrays.

        Args:
            value: The resolved fact value

        Returns:
            True if value should be stored as a table
        """
        if not isinstance(value, list):
            return False

        # Check row count threshold
        if len(value) > ARRAY_ROW_THRESHOLD:
            return True

        # Check JSON size threshold
        try:
            import json
            json_size = len(json.dumps(value))
            if json_size > ARRAY_SIZE_THRESHOLD:
                return True
        except (TypeError, ValueError):
            # Can't serialize - keep as-is
            pass

        return False

    def _store_value_as_table(self, fact_name: str, value: list, source_name: str = None) -> tuple[str, int]:
        """
        Store a list value as a table in the datastore.

        Args:
            fact_name: Name of the fact (used to generate table name)
            value: List of dicts to store as table
            source_name: Optional source name for table naming

        Returns:
            Tuple of (table_name, row_count)
        """
        if not self._datastore:
            raise ValueError("No datastore configured for table storage")

        import pandas as pd

        # Generate a clean table name from fact name
        # Replace special chars, ensure valid SQL identifier
        table_name = f"fact_{fact_name}".replace("(", "_").replace(")", "").replace(",", "_").replace("=", "_")
        table_name = table_name.replace("-", "_").replace(" ", "_").lower()

        # Convert to DataFrame
        if value and isinstance(value[0], dict):
            df = pd.DataFrame(value)
        else:
            # Handle list of primitives
            df = pd.DataFrame({"value": value})

        # Store in datastore
        self._datastore.save_dataframe(table_name, df)

        return table_name, len(df)

    def _try_resolve(
        self,
        source: FactSource,
        fact_name: str,
        params: dict,
        cache_key: str
    ) -> Optional[Fact]:
        """Try to resolve from a specific source."""
        import logging
        logger = logging.getLogger(__name__)

        if source == FactSource.CACHE:
            cached = self._cache.get(cache_key)
            if cached:
                # For table facts, verify the table still exists in datastore
                if cached.table_name and self._datastore:
                    try:
                        existing_tables = [t["name"] for t in self._datastore.list_tables()]
                        if cached.table_name not in existing_tables:
                            logger.debug(f"[_try_resolve] CACHE table {cached.table_name} no longer exists")
                            return None  # Table doesn't exist, need to re-resolve
                    except Exception as e:
                        logger.debug(f"[_try_resolve] Cache validation failed for {cache_key}, assuming valid: {e}")
                logger.debug(f"[_try_resolve] CACHE hit for {cache_key}")
            return cached

        elif source == FactSource.CONFIG:
            result = self._resolve_from_config(fact_name, params)
            logger.debug(f"[_try_resolve] CONFIG for {fact_name}: {result is not None}")
            return result

        elif source == FactSource.RULE:
            result = self._resolve_from_rule(fact_name, params)
            logger.debug(f"[_try_resolve] RULE for {fact_name}: {result is not None}")
            return result

        elif source == FactSource.DATABASE:
            logger.debug(f"_try_resolve DATABASE attempting for {fact_name}")
            logger.debug(f"[_try_resolve] DATABASE attempting for {fact_name}")
            result = self._resolve_from_database(fact_name, params)
            logger.debug(f"_try_resolve DATABASE for {fact_name}: result={result is not None}")
            logger.debug(f"[_try_resolve] DATABASE for {fact_name}: {result is not None}")
            return result

        elif source == FactSource.DOCUMENT:
            logger.debug(f"_try_resolve DOCUMENT attempting for {fact_name}")
            logger.debug(f"[_try_resolve] DOCUMENT attempting for {fact_name}")
            result = self._resolve_from_document(fact_name, params)
            logger.debug(f"_try_resolve DOCUMENT for {fact_name}: result={result is not None}")
            logger.debug(f"[_try_resolve] DOCUMENT for {fact_name}: {result is not None}")
            return result

        elif source == FactSource.API:
            logger.debug(f"[_try_resolve] API attempting for {fact_name}")
            result = self._resolve_from_api(fact_name, params)
            logger.debug(f"[_try_resolve] API for {fact_name}: {result is not None}")
            return result

        elif source == FactSource.LLM_KNOWLEDGE:
            logger.debug(f"[_try_resolve] LLM_KNOWLEDGE attempting for {fact_name}")
            result = self._resolve_from_llm(fact_name, params)
            logger.debug(f"[_try_resolve] LLM_KNOWLEDGE for {fact_name}: {result is not None}")
            return result

        elif source == FactSource.USER_PROVIDED:
            # User-provided facts are only in cache (added via add_user_fact)
            # This is a fallback - if we reach here, fact is unresolved
            logger.debug(f"[_try_resolve] USER_PROVIDED - no fallback for {fact_name}")
            return None

        elif source == FactSource.SUB_PLAN:
            logger.debug(f"[_try_resolve] SUB_PLAN attempting for {fact_name}")
            result = self._resolve_from_sub_plan(fact_name, params)
            logger.debug(f"[_try_resolve] SUB_PLAN for {fact_name}: {result is not None}")
            return result

        return None

    def _resolve_from_config(self, fact_name: str, params: dict) -> Optional[Fact]:
        """Check if fact is defined in config/system prompt."""
        if not self.config:
            return None

        # TODO: Parse config.system_prompt for defined facts/thresholds
        # For now, return None
        return None

    def _resolve_from_rule(self, fact_name: str, params: dict) -> Optional[Fact]:
        """Try to derive fact using a registered rule."""
        rule = self._rules.get(fact_name)
        if not rule:
            return None

        try:
            return rule(self, params)
        except Exception as e:
            # Rule failed - log but don't crash
            return None

    def _resolve_from_api(self, fact_name: str, params: dict, lazy: bool = True) -> Optional[Fact]:
        """Resolve a fact by querying external APIs.

        Uses LLM to determine which API to query and how.

        Args:
            fact_name: Name of the fact to resolve
            params: Parameters for the fact
            lazy: If True (default), return a binding without executing the API.
                  The binding shows which API/endpoint will be used.
                  If False, execute the API and return actual data.

        Returns:
            Fact with API binding (lazy) or actual data (eager)
        """
        import logging
        import json
        logger = logging.getLogger(__name__)

        if not self.config or not self.config.apis:
            logger.debug(f"[_resolve_from_api] No APIs configured")
            return None

        if not self.llm:
            logger.debug(f"[_resolve_from_api] No LLM available for API query generation")
            return None

        # Build API overview for LLM
        api_overview_lines = []
        api_descriptions = {}
        for api_name, api_config in self.config.apis.items():
            api_type = api_config.type.upper()
            desc = api_config.description or f"{api_type} endpoint"
            url = api_config.url or ""
            api_overview_lines.append(f"- {api_name} ({api_type}): {desc}")
            api_descriptions[api_name] = {"type": api_type, "desc": desc, "url": url}
            if url:
                api_overview_lines.append(f"  URL: {url}")
        api_overview = "\n".join(api_overview_lines)

        # Ask LLM which API can provide this fact and how to query it
        prompt = f"""I need to resolve this fact from an external API:

Fact: {fact_name}
Parameters: {json.dumps(params) if params else "none"}

Available APIs:
{api_overview}

If this fact can be resolved from one of these APIs, provide the query details.
Otherwise respond NOT_POSSIBLE.

For GraphQL APIs, respond in this format:
API: <api_name>
GRAPHQL: <graphql_query>

For REST/OpenAPI APIs, respond in this format:
API: <api_name>
REST: <endpoint_path>
METHOD: GET|POST|etc
PARAMS: {{"key": "value"}}

If this fact cannot be resolved from any available API, respond:
NOT_POSSIBLE: <reason>
"""

        try:
            response = self.llm.generate(
                system="You are an API expert. Determine which API can provide the requested data and how to query it.",
                user_message=prompt,
                max_tokens=500,
            )

            if "NOT_POSSIBLE" in response:
                logger.debug(f"[_resolve_from_api] LLM says not possible: {response}")
                return None

            # Parse the response
            lines = response.strip().split("\n")
            api_name = None
            graphql_query = None
            rest_endpoint = None
            rest_method = "GET"
            rest_params = {}

            for idx, line in enumerate(lines):
                line = line.strip()
                if line.startswith("API:"):
                    api_name = line.split(":", 1)[1].strip()
                elif line.startswith("GRAPHQL:"):
                    graphql_query = line.split(":", 1)[1].strip()
                    # Collect multi-line GraphQL query
                    for subsequent in lines[idx + 1:]:
                        if subsequent.strip().startswith(("API:", "REST:", "METHOD:", "PARAMS:", "NOT_POSSIBLE")):
                            break
                        graphql_query += "\n" + subsequent
                    graphql_query = graphql_query.strip()
                    # Clean up code blocks
                    graphql_query = graphql_query.replace("```graphql", "").replace("```", "").strip()
                elif line.startswith("REST:"):
                    rest_endpoint = line.split(":", 1)[1].strip()
                elif line.startswith("METHOD:"):
                    rest_method = line.split(":", 1)[1].strip().upper()
                elif line.startswith("PARAMS:"):
                    params_str = line.split(":", 1)[1].strip()
                    try:
                        rest_params = json.loads(params_str)
                    except json.JSONDecodeError:
                        pass

            if not api_name:
                logger.debug(f"[_resolve_from_api] Could not parse API name from response")
                return None

            # Build endpoint description
            if graphql_query:
                api_endpoint = graphql_query[:100] + "..." if len(graphql_query) > 100 else graphql_query
                endpoint_display = f"GraphQL query"
            elif rest_endpoint:
                api_endpoint = f"{rest_method} {rest_endpoint}"
                endpoint_display = api_endpoint
            else:
                logger.debug(f"[_resolve_from_api] No query found in LLM response")
                return None

            # Lazy binding: return API source info without executing
            if lazy:
                cache_key = self._cache_key(fact_name, params)
                api_info = api_descriptions.get(api_name, {})
                api_desc = api_info.get("desc", "API")
                api_type = api_info.get("type", "REST")

                # Build descriptive value like database does: "(api_name) endpoint_info"
                value_str = f"({api_name}) {endpoint_display}"
                reasoning = f"API '{api_name}' ({api_type}): {api_desc}. Endpoint: {api_endpoint}"

                logger.info(f"[_resolve_from_api] Lazy binding for '{fact_name}' -> {api_name}: {endpoint_display}")

                fact = Fact(
                    name=cache_key,
                    value=value_str,
                    confidence=0.95,
                    source=FactSource.API,
                    source_name=api_name,
                    api_endpoint=api_endpoint,
                    reasoning=reasoning,
                    context=f"API: {api_name} - {endpoint_display}",
                )

                self._cache[cache_key] = fact
                self.resolution_log.append(fact)
                return fact

            # Eager execution: actually call the API
            from constat.catalog.api_executor import APIExecutor, APIExecutionError
            executor = APIExecutor(self.config)

            try:
                if graphql_query:
                    logger.info(f"[_resolve_from_api] Executing GraphQL query on {api_name}")
                    result = executor.execute_graphql(api_name, graphql_query)
                elif rest_endpoint:
                    logger.info(f"[_resolve_from_api] Executing REST call on {api_name}: {rest_method} {rest_endpoint}")
                    result = executor.execute_rest(
                        api_name,
                        operation=rest_endpoint,
                        query_params=rest_params if rest_method == "GET" else None,
                        body=rest_params if rest_method in ("POST", "PUT", "PATCH") else None,
                        method=rest_method,
                    )
                else:
                    logger.debug(f"[_resolve_from_api] No query found in LLM response")
                    return None

                # Create the fact from the result
                cache_key = self._cache_key(fact_name, params)

                # Determine value and whether to store as table
                if isinstance(result, dict):
                    # Check if result has a 'data' key with list (common pattern)
                    if "data" in result and isinstance(result["data"], list):
                        value = result["data"]
                    else:
                        value = result
                elif isinstance(result, list):
                    value = result
                else:
                    value = result

                # Check if should store as table
                row_count = None
                table_name = None
                if self._datastore and isinstance(value, list) and len(value) > 10:
                    # Store large results as table
                    import pandas as pd
                    df = pd.DataFrame(value)
                    table_name = f"api_{api_name}_{fact_name}".replace(" ", "_").replace("-", "_")[:50]
                    self._datastore.save_table(table_name, df)
                    row_count = len(df)
                    logger.info(f"[_resolve_from_api] Stored {row_count} rows as table {table_name}")

                fact = Fact(
                    name=cache_key,
                    value=value,
                    confidence=0.95,
                    source=FactSource.API,
                    source_name=api_name,
                    api_endpoint=api_endpoint,
                    table_name=table_name,
                    row_count=row_count,
                    context=f"API Query: {api_endpoint}",
                )

                self._cache[cache_key] = fact
                self.resolution_log.append(fact)
                return fact

            except APIExecutionError as e:
                logger.warning(f"[_resolve_from_api] API execution failed: {e}")
                return None

        except Exception as e:
            logger.warning(f"[_resolve_from_api] Failed to resolve {fact_name} from API: {e}")
            return None

    def _transform_sql_for_sqlite(self, sql: str) -> str:
        """Transform MySQL/PostgreSQL SQL syntax to SQLite equivalents.

        Handles common incompatibilities:
        - DATE_FORMAT(col, fmt) -> strftime(fmt, col)
        - DATE_SUB(date, INTERVAL n MONTH) -> date(date, '-n months')
        - CURDATE() -> date('now')
        - YEAR(col) -> CAST(strftime('%Y', col) AS INTEGER)
        - MONTH(col) -> CAST(strftime('%m', col) AS INTEGER)
        - EXTRACT(YEAR FROM col) -> CAST(strftime('%Y', col) AS INTEGER)
        - EXTRACT(MONTH FROM col) -> CAST(strftime('%m', col) AS INTEGER)
        """
        import re

        # DATE_FORMAT(col, '%Y-%m') -> strftime('%Y-%m', col)
        def replace_date_format(match):
            col = match.group(1)
            fmt = match.group(2)
            return f"strftime({fmt}, {col})"
        sql = re.sub(r"DATE_FORMAT\s*\(\s*([^,]+),\s*('[^']+')\s*\)", replace_date_format, sql, flags=re.IGNORECASE)

        # DATE_SUB(CURDATE(), INTERVAL n MONTH) -> date('now', '-n months')
        # Also handles DATE_SUB(date_col, INTERVAL n MONTH)
        def replace_date_sub(match):
            date_expr = match.group(1)
            num = match.group(2)
            unit = match.group(3).lower()
            # Convert CURDATE() to 'now'
            if date_expr.strip().upper() == "CURDATE()":
                date_expr = "'now'"
            return f"date({date_expr}, '-{num} {unit}s')"
        sql = re.sub(r"DATE_SUB\s*\(\s*([^,]+),\s*INTERVAL\s+(\d+)\s+(MONTH|DAY|YEAR)\s*\)", replace_date_sub, sql, flags=re.IGNORECASE)

        # CURDATE() -> date('now')
        sql = re.sub(r"\bCURDATE\s*\(\s*\)", "date('now')", sql, flags=re.IGNORECASE)

        # NOW() -> datetime('now')
        sql = re.sub(r"\bNOW\s*\(\s*\)", "datetime('now')", sql, flags=re.IGNORECASE)

        # YEAR(col) -> CAST(strftime('%Y', col) AS INTEGER)
        def replace_year(match):
            col = match.group(1)
            return f"CAST(strftime('%Y', {col}) AS INTEGER)"
        sql = re.sub(r"\bYEAR\s*\(\s*([^)]+)\s*\)", replace_year, sql, flags=re.IGNORECASE)

        # MONTH(col) -> CAST(strftime('%m', col) AS INTEGER)
        def replace_month(match):
            col = match.group(1)
            return f"CAST(strftime('%m', {col}) AS INTEGER)"
        sql = re.sub(r"\bMONTH\s*\(\s*([^)]+)\s*\)", replace_month, sql, flags=re.IGNORECASE)

        # EXTRACT(YEAR FROM col) -> CAST(strftime('%Y', col) AS INTEGER)
        def replace_extract_year(match):
            col = match.group(1)
            return f"CAST(strftime('%Y', {col}) AS INTEGER)"
        sql = re.sub(r"\bEXTRACT\s*\(\s*YEAR\s+FROM\s+([^)]+)\s*\)", replace_extract_year, sql, flags=re.IGNORECASE)

        # EXTRACT(MONTH FROM col) -> CAST(strftime('%m', col) AS INTEGER)
        def replace_extract_month(match):
            col = match.group(1)
            return f"CAST(strftime('%m', {col}) AS INTEGER)"
        sql = re.sub(r"\bEXTRACT\s*\(\s*MONTH\s+FROM\s+([^)]+)\s*\)", replace_extract_month, sql, flags=re.IGNORECASE)

        return sql

    def _resolve_from_database(self, fact_name: str, params: dict) -> Optional[Fact]:
        """Generate and execute Python code to resolve a fact from data sources.

        Uses code generation (like exploratory mode) to support all data source types:
        - SQL databases: pd.read_sql()
        - NoSQL databases: connector methods
        - File sources: pd.read_csv(), pd.read_json(), etc.

        Uses PythonExecutor for consistent execution and error handling.
        Participates in learning loop when syntax errors are fixed.
        """
        import logging
        import pandas as pd
        from constat.execution.executor import PythonExecutor, format_error_for_retry
        logger = logging.getLogger(__name__)

        logger.debug(f"DB: _resolve_from_database called for: {fact_name}")
        logger.debug(f"DB: llm={self.llm is not None}, schema_manager={self.schema_manager is not None}, config={self.config is not None}")

        if not self.llm or not self.schema_manager:
            logger.debug(f"DB: MISSING: LLM={self.llm is not None}, schema_manager={self.schema_manager is not None}")
            logger.debug(f"[_resolve_from_database] Missing LLM ({self.llm is not None}) "
                        f"or schema_manager ({self.schema_manager is not None})")
            return None

        # Check if fact_name matches a table name - return "referenced" instead of loading data
        # This allows inferences to query the table directly from the original database
        fact_name_lower = fact_name.lower().strip()
        cache_tables = list(self.schema_manager.metadata_cache.keys())
        logger.debug(f"[_resolve_from_database] Checking table match for '{fact_name_lower}', metadata_cache has {len(cache_tables)} tables: {cache_tables[:5]}")
        for full_name, table_meta in self.schema_manager.metadata_cache.items():
            # Match by table name (case-insensitive)
            if table_meta.name.lower() == fact_name_lower:
                logger.info(f"[_resolve_from_database] Table match for '{fact_name}' -> {full_name}")
                # Store column metadata in reasoning for use by inferences
                columns = [c.name for c in table_meta.columns]
                reasoning = f"Table '{table_meta.name}' from database '{table_meta.database}'. Columns: {columns}"
                # Build descriptive value for UI display
                row_info = f"{table_meta.row_count:,} rows" if table_meta.row_count else "table"
                value_str = f"({table_meta.database}.{table_meta.name}) {row_info}"
                return Fact(
                    name=fact_name,
                    value=value_str,
                    source=FactSource.DATABASE,
                    source_name=table_meta.database,
                    reasoning=reasoning,
                    confidence=0.95,
                    table_name=table_meta.name,
                    row_count=table_meta.row_count,
                )

        # Build execution globals with database connections and file paths
        exec_globals = {"pd": pd, "Fact": Fact, "FactSource": FactSource}
        config_db_names = set(self.config.databases.keys()) if self.config else set()

        for db_name in config_db_names:
            db_config = self.config.databases.get(db_name)
            if db_config:
                if db_config.is_file_source():
                    # Provide file path for CSV, JSON, Parquet, etc.
                    exec_globals[f"file_{db_name}"] = db_config.path
                else:
                    # Provide database connection for SQL/NoSQL
                    conn = self.schema_manager.get_connection(db_name)
                    exec_globals[f"db_{db_name}"] = conn

        # Also include dynamically added databases (from projects) not in config
        # SQL connections
        for db_name in self.schema_manager.connections.keys():
            if db_name not in config_db_names:
                exec_globals[f"db_{db_name}"] = self.schema_manager.connections[db_name]
        # NoSQL connections
        for db_name in self.schema_manager.nosql_connections.keys():
            if db_name not in config_db_names:
                exec_globals[f"db_{db_name}"] = self.schema_manager.nosql_connections[db_name]
        # File connections
        for db_name in self.schema_manager.file_connections.keys():
            if db_name not in config_db_names:
                conn = self.schema_manager.file_connections[db_name]
                if hasattr(conn, 'path'):
                    exec_globals[f"file_{db_name}"] = conn.path

        # Build data source hints for the prompt (from config databases)
        source_hints = []
        for db_name, db_config in (self.config.databases.items() if self.config else []):
            if db_config.is_file_source():
                file_type = db_config.type
                source_hints.append(f"- {db_name} ({file_type}): use pd.read_{file_type}(file_{db_name})")
            elif db_config.is_nosql():
                source_hints.append(f"- {db_name} (NoSQL {db_config.type}): use db_{db_name} connector methods")
            else:
                # SQL database - detect dialect
                dialect = "sql"
                if db_config.type == "sql" and db_config.uri:
                    uri_lower = db_config.uri.lower()
                    if uri_lower.startswith("sqlite"):
                        dialect = "sqlite"
                    elif uri_lower.startswith("postgresql") or uri_lower.startswith("postgres"):
                        dialect = "postgresql"
                    elif uri_lower.startswith("mysql"):
                        dialect = "mysql"
                    elif uri_lower.startswith("duckdb"):
                        dialect = "duckdb"
                source_hints.append(f"- {db_name} ({dialect}): use pd.read_sql(query, db_{db_name})")

        # Add hints for dynamically added databases
        for db_name in self.schema_manager.connections.keys():
            if db_name not in config_db_names:
                source_hints.append(f"- {db_name} (sql): use pd.read_sql(query, db_{db_name})")
        for db_name in self.schema_manager.nosql_connections.keys():
            if db_name not in config_db_names:
                source_hints.append(f"- {db_name} (nosql): use db_{db_name} connector methods")
        for db_name in self.schema_manager.file_connections.keys():
            if db_name not in config_db_names:
                source_hints.append(f"- {db_name} (file): use pd.read_csv/json/parquet(file_{db_name})")

        source_hints_text = "\n".join(source_hints) if source_hints else "No data sources configured."
        logger.debug(f"DB: source_hints_text: {source_hints_text[:200]}...")
        logger.debug(f"DB: db_names: {db_names}")

        # Get schema overview
        schema_overview = self.schema_manager.get_overview()
        logger.debug(f"DB: schema_overview length: {len(schema_overview)}")

        # Build prompt for code generation
        prompt = f"""Generate Python code to resolve this fact from the available data sources.

Fact to resolve: {fact_name}
Parameters: {params}

Available data sources:
{source_hints_text}

Schema:
{schema_overview}

Generate a `get_result()` function that:
1. Queries the appropriate data source(s)
2. Returns the result value (scalar, list, or DataFrame)

IMPORTANT - PREFER SQL OVER PANDAS:
- SQL is more robust, scalable, and has clearer error messages
- Use DuckDB to query DataFrames/JSON: `duckdb.query("SELECT ... FROM df").df()`
- For SQLite: Do NOT use schema prefixes (use 'customers' not 'sales.customers')
- For SQLite: Use strftime() for date formatting, date() for date math
- Return the raw result - the caller will wrap it in a Fact

Example for SQL database:
```python
def get_result():
    df = pd.read_sql("SELECT SUM(amount) as total FROM orders", db_sales)
    return df.iloc[0, 0]  # Return scalar
```

Example for API response (use DuckDB for transformations):
```python
def get_result():
    import duckdb
    response = api_catfacts.get("/breeds", params={"limit": 10})
    data = response["data"]  # API responses often have data in nested field
    return duckdb.query("SELECT breed, country FROM data").df()
```

Example for CSV:
```python
def get_result():
    import duckdb
    df = pd.read_csv(file_web_metrics)
    return duckdb.query("SELECT page, SUM(visitors) as total FROM df GROUP BY page").df()
```

CRITICAL - When to respond NOT_POSSIBLE:
- If the fact asks for POLICY, RULES, GUIDELINES, or THRESHOLDS but no such table/config exists in the schema
- If you would need to ANALYZE PATTERNS or DERIVE rules from transactional data - that is NOT the same as having actual rules
- If the schema only has operational/transactional data (reviews, orders, etc.) but the fact asks for policy/rules ABOUT that data
- Statistical summaries of data (avg rating, count, distribution) are NOT policies - policies are prescriptive rules like "rating 5 = 10% raise"
- Do NOT return approximations, pattern analysis, or inferred rules as substitutes for explicitly stored policies

If this fact cannot be DIRECTLY resolved from the available sources, respond with "NOT_POSSIBLE: <reason>".
"""

        # Use PythonExecutor for consistent execution (DRY with exploratory mode)
        executor = PythonExecutor()
        max_retries = 3
        last_code = None
        last_error = None
        original_error_code = None  # Track original failing code for learning

        for attempt in range(1, max_retries + 1):
            # Generate code
            if attempt == 1:
                response = self.llm.generate(
                    system="You are a Python data expert. Generate code to extract facts from data sources.",
                    user_message=prompt,
                    max_tokens=2000,
                )
            else:
                # Retry with error context
                retry_prompt = f"""Your previous code failed:

{last_error}

Previous code:
```python
{last_code}
```

Please fix the error and regenerate the code.

Original request:
{prompt}"""
                response = self.llm.generate(
                    system="You are a Python data expert. Generate code to extract facts from data sources.",
                    user_message=retry_prompt,
                    max_tokens=2000,
                )

            logger.debug(f"DB: LLM response (first 300 chars): {response[:300]}...")
            if "NOT_POSSIBLE" in response:
                logger.debug(f"DB: LLM said NOT_POSSIBLE: {response}")
                logger.debug(f"[_resolve_from_database] LLM said not possible: {response}")
                return None

            # Extract code from response
            code = response
            if "```python" in code:
                code = code.split("```python", 1)[1].split("```", 1)[0]
            elif "```" in code:
                code = code.split("```", 1)[1].split("```", 1)[0]

            last_code = code
            logger.debug(f"[_resolve_from_database] Attempt {attempt} generated code:\n{code}")

            # Execute using PythonExecutor
            result = executor.execute(code, exec_globals)

            if result.compile_error:
                # Syntax error - retry with feedback
                if original_error_code is None:
                    original_error_code = code
                last_error = format_error_for_retry(result, code)
                logger.warning(f"[_resolve_from_database] Attempt {attempt} syntax error")
                continue

            if result.runtime_error:
                # Runtime error - don't retry, move to next source type
                logger.warning(f"[_resolve_from_database] Runtime error for {fact_name}: {result.runtime_error.error}")
                return None

            # Execution succeeded - check for get_result function
            get_result = result.namespace.get("get_result")
            if not get_result:
                if original_error_code is None:
                    original_error_code = code
                last_error = "No get_result() function found in generated code. Please define a get_result() function."
                logger.warning(f"[_resolve_from_database] Attempt {attempt}: no get_result() function")
                continue

            # Call get_result and process the result
            try:
                value = get_result()

                # If we fixed a syntax error, record the learning
                if original_error_code is not None and self._learning_callback:
                    self._learning_callback(
                        category="code_error",
                        context={
                            "fact_name": fact_name,
                            "error_message": last_error or "Syntax error",
                            "original_code": original_error_code,
                        },
                        fixed_code=code,
                    )

                cache_key = self._cache_key(fact_name, params)
                source_name = db_names[0] if db_names else None

                # Handle DataFrame results
                if isinstance(value, pd.DataFrame):
                    if len(value) == 1 and len(value.columns) == 1:
                        value = value.iloc[0, 0]
                    else:
                        value = value.to_dict('records')

                # Check if should store as table
                if isinstance(value, list) and self._datastore and self._should_store_as_table(value):
                    table_name, row_count = self._store_value_as_table(
                        fact_name, value, source_name=source_name
                    )
                    return Fact(
                        name=cache_key,
                        value=f"table:{table_name}",
                        confidence=1.0,
                        source=FactSource.DATABASE,
                        source_name=source_name,
                        query=code,
                        table_name=table_name,
                        row_count=row_count,
                        context=f"Generated code:\n{code}",
                    )

                return Fact(
                    name=cache_key,
                    value=value,
                    confidence=1.0,
                    source=FactSource.DATABASE,
                    source_name=source_name,
                    query=code,
                    context=f"Generated code:\n{code}",
                )

            except Exception as e:
                # Runtime error in get_result() - retry with feedback
                if original_error_code is None:
                    original_error_code = code
                import traceback
                tb = traceback.format_exc()
                last_error = f"get_result() raised an exception:\n{type(e).__name__}: {e}\n\nTraceback:\n{tb}"
                logger.warning(f"[_resolve_from_database] get_result() failed for {fact_name}: {e}")
                continue  # Retry with error feedback

        # All retry attempts exhausted
        logger.error(f"[_resolve_from_database] All {max_retries} attempts failed for {fact_name}")
        return None

    def _resolve_from_document(self, fact_name: str, params: dict) -> Optional[Fact]:
        """Search reference documents for fact information.

        Uses a two-stage approach:
        1. Semantic search to find potentially relevant document chunks
        2. If chunks have low relevance, load full document sections for better context
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.debug(f"DOC: _resolve_from_document called for: {fact_name}")

        if not self.llm:
            logger.debug(f"DOC: No LLM configured")
            logger.debug(f"[_resolve_from_document] No LLM configured")
            return None

        # Check if we have document tools configured
        doc_tools = getattr(self, '_doc_tools', None)
        logger.debug(f"DOC: doc_tools={doc_tools is not None}")
        if not doc_tools:
            logger.debug(f"DOC: No doc_tools configured - returning None")
            logger.debug(f"[_resolve_from_document] No doc_tools configured")
            return None

        cache_key = self._cache_key(fact_name, params)

        try:
            # Build search query from fact name and params
            param_str = ' '.join(str(v) for v in params.values() if v)
            fact_readable = fact_name.replace('_', ' ')
            search_query = f"{fact_readable} {param_str}".strip()

            logger.debug(f"DOC: Searching for: {search_query}")
            logger.debug(f"[_resolve_from_document] Searching for: {search_query}")

            # Get more results (top 10) and let the LLM evaluate relevance
            # Don't filter by score - semantic search scores can be misleading
            search_results = doc_tools.search_documents(search_query, limit=10)
            logger.debug(f"DOC: Search returned {len(search_results) if search_results else 0} results")

            if not search_results:
                logger.debug(f"DOC: No search results - returning None")
                logger.debug(f"[_resolve_from_document] No search results")
                return None

            best_relevance = max(r.get('relevance', 0) for r in search_results)
            logger.debug(f"DOC: Best relevance: {best_relevance}")
            logger.debug(f"[_resolve_from_document] Best relevance: {best_relevance}")

            # Include ALL results - let the LLM decide what's relevant
            # Semantic search scores are not reliable for filtering
            context = "\n\n".join([
                f"From {r.get('document', 'document')} (section: {r.get('section', 'unknown')}, relevance: {r.get('relevance', 0):.2f}):\n{r.get('excerpt', '')}"
                for r in search_results
            ])

            if not context.strip():
                logger.debug(f"DOC: No context built - returning None")
                logger.debug(f"[_resolve_from_document] No context built")
                return None

            logger.debug(f"DOC: Context built, length: {len(context)}")
            logger.debug(f"DOC: Context preview: {context[:300]}...")

            # Ask LLM to extract the fact from document context
            prompt = f"""Extract information for this fact from the document content:

Fact needed: {fact_name}
Parameters: {params}

Document content:
{context}

If relevant information is found, respond with:
VALUE: <extract the relevant content - table, paragraph, rules, or data as-is>
CONFIDENCE: <0.0-1.0>
SOURCE: <which document/section>
REASONING: <brief explanation>

The VALUE can be:
- A number or string for simple facts
- A table or list for structured policies
- A paragraph or multiple paragraphs for descriptive policies
- JSON if that best represents the data

If no relevant information is found, respond with:
NOT_FOUND
"""

            logger.debug(f"DOC: Calling LLM to extract fact...")
            response = self.llm.generate(
                system="You extract facts and policies from documents. Return content in its natural format.",
                user_message=prompt,
                max_tokens=800,
            )

            logger.debug(f"DOC: LLM response: {response[:300]}...")
            logger.debug(f"[_resolve_from_document] LLM response: {response[:200]}...")

            if "NOT_FOUND" in response:
                logger.debug(f"DOC: LLM returned NOT_FOUND - returning None")
                logger.debug(f"[_resolve_from_document] LLM returned NOT_FOUND")
                return None

            # Parse response
            value = None
            confidence = 0.8
            source_name = None
            reasoning = None

            # Extract VALUE - may be JSON spanning multiple lines
            if "VALUE:" in response:
                value_start = response.index("VALUE:") + len("VALUE:")
                # Find where VALUE ends (next field or end)
                value_end = len(response)
                for marker in ["CONFIDENCE:", "SOURCE:", "REASONING:"]:
                    if marker in response[value_start:]:
                        marker_pos = response.index(marker, value_start)
                        if marker_pos < value_end:
                            value_end = marker_pos
                value_str = response[value_start:value_end].strip()

                # Try to parse as JSON first, then number, then string
                import json
                try:
                    value = json.loads(value_str)
                except (json.JSONDecodeError, ValueError):
                    try:
                        value = float(value_str)
                    except ValueError:
                        value = value_str

            # Parse other fields
            for line in response.split("\n"):
                if line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("SOURCE:"):
                    source_name = line.split(":", 1)[1].strip()
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

            logger.debug(f"DOC: Parsed value: {value}, confidence: {confidence}, source: {source_name}")
            if value is not None:
                logger.debug(f"DOC: SUCCESS! Resolved {fact_name} = {value}")
                logger.debug(f"[_resolve_from_document] Resolved {fact_name} = {value} from {source_name}")
                return Fact(
                    name=cache_key,
                    value=value,
                    confidence=confidence,
                    source=FactSource.DOCUMENT,
                    source_name=source_name,
                    reasoning=reasoning,
                )
            else:
                logger.debug(f"DOC: No VALUE found in LLM response")
        except Exception as e:
            logger.debug(f"DOC: Exception resolving {fact_name}: {e}")
            import traceback
            logger.debug(f"DOC: Traceback: {traceback.format_exc()}")
            logger.warning(f"[_resolve_from_document] Error resolving {fact_name}: {e}")

        logger.debug(f"DOC: Returning None for {fact_name}")
        return None

    def _resolve_from_llm(self, fact_name: str, params: dict) -> Optional[Fact]:
        """Ask LLM for world knowledge or heuristics."""
        import logging
        logger = logging.getLogger(__name__)

        if not self.llm:
            logger.debug(f"[_resolve_from_llm] No LLM configured")
            return None

        logger.debug(f"[_resolve_from_llm] Asking LLM about {fact_name}")

        prompt = f"""I need to know this fact:
Fact: {fact_name}
Parameters: {params}

Do you know this from your training? This could be:
- World knowledge (e.g., "capital of France")
- Industry standards (e.g., "typical VIP threshold is $10,000")
- Common heuristics (e.g., "underperforming means <80% of target")

If you know this, respond with:
VALUE: <the value>
CONFIDENCE: <0.0-1.0, how confident are you>
TYPE: knowledge | heuristic
REASONING: <brief explanation>

If you don't know, respond with:
UNKNOWN
"""

        try:
            response = self.llm.generate(
                system="You are a knowledgeable assistant. Provide facts you're confident about.",
                user_message=prompt,
                max_tokens=300,
            )

            if "UNKNOWN" in response:
                return None

            # Parse response
            value = None
            confidence = 0.6  # Default for LLM knowledge
            reasoning = None
            source = FactSource.LLM_KNOWLEDGE

            for line in response.split("\n"):
                if line.startswith("VALUE:"):
                    value_str = line.split(":", 1)[1].strip()
                    # Try to parse as number
                    try:
                        value = float(value_str)
                    except (ValueError, TypeError):
                        value = value_str
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except (ValueError, TypeError):
                        pass
                elif line.startswith("TYPE:"):
                    if "heuristic" in line.lower():
                        source = FactSource.LLM_HEURISTIC
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

            if value is not None:
                logger.debug(f"[_resolve_from_llm] Got value for {fact_name}: {value} (conf={confidence})")
                return Fact(
                    name=self._cache_key(fact_name, params),
                    value=value,
                    confidence=confidence,
                    source=source,
                    reasoning=reasoning,
                )
            else:
                logger.debug(f"[_resolve_from_llm] No value parsed for {fact_name}")
        except Exception as e:
            logger.error(f"[_resolve_from_llm] Error for {fact_name}: {e}")
            import traceback
            logger.error(traceback.format_exc())

        return None

    def _resolve_from_sub_plan(self, fact_name: str, params: dict) -> Optional[Fact]:
        """Generate a mini-plan to derive a complex fact with parallel resolution.

        IMPORTANT: This method enforces the 2+ inputs requirement.
        Derivation must compose multiple DISTINCT facts - not just try synonyms.
        """
        import logging
        logger = logging.getLogger(__name__)

        if not self.strategy.allow_sub_plans:
            logger.debug(f"[_resolve_from_sub_plan] Sub-plans disabled")
            return None

        if self._resolution_depth >= self.strategy.max_sub_plan_depth:
            logger.debug(f"[_resolve_from_sub_plan] Max depth {self.strategy.max_sub_plan_depth} reached")
            return None  # Prevent infinite recursion

        if not self.llm:
            logger.debug(f"[_resolve_from_sub_plan] No LLM configured")
            return None

        # NEW: With tiered resolution enabled, sub-plan is handled by Tier 2 assessment
        # This legacy method should only run if tiered resolution is disabled
        if self.strategy.use_tiered_resolution:
            logger.debug(f"[_resolve_from_sub_plan] Skipping - tiered resolution handles sub-plans via Tier 2")
            return None

        logger.debug(f"[_resolve_from_sub_plan] Attempting sub-plan for {fact_name} at depth {self._resolution_depth}")

        # Emit event: starting sub-plan expansion
        self._emit_event("premise_expanding", {
            "fact_name": fact_name,
            "params": params,
            "depth": self._resolution_depth,
        })

        # Ask LLM to create a plan to derive this fact
        # CRITICAL: Enforce 2+ distinct inputs requirement to prevent synonym hunting
        prompt = f"""I need to derive this fact, but it's not directly available:
Fact: {fact_name}
Parameters: {params}

This fact needs to be COMPUTED from 2 or more OTHER facts using a formula.

CRITICAL REQUIREMENTS:
1. You MUST resolve 2+ DISTINCT facts and COMPOSE them with a formula
2. Valid: "result = fact_A / fact_B" (two inputs, mathematical composition)
3. Valid: "result = filter(fact_A, condition from fact_B)" (two inputs)
4. INVALID: Just resolving the same fact with a different name (synonym hunting)
5. INVALID: Trying to look up "alternative_name" or "similar_concept" - this is NOT derivation

If this fact CANNOT be computed from 2+ other facts, respond with:
```python
def derive(resolver, params):
    # NOT_DERIVABLE: This fact cannot be computed from other facts
    return None
```

Example with VALID derivation (2+ inputs):
```python
def derive(resolver, params):
    # Resolve 2+ DISTINCT facts
    facts = resolver.resolve_many_sync([
        ("employee_salaries", {{}}),  # Input 1: from database
        ("industry_benchmark", {{}}),  # Input 2: general knowledge
    ])
    salaries, benchmark = facts

    # COMPOSE with formula
    avg_salary = sum(s["salary"] for s in salaries.value) / len(salaries.value)
    competitive_ratio = avg_salary / benchmark.value

    return Fact(
        name="{fact_name}",
        value=competitive_ratio,
        confidence=min(f.confidence for f in facts),
        source=FactSource.DERIVED,
        because=facts
    )
```

Generate the derivation function for {fact_name}.
Remember: 2+ DISTINCT inputs with a FORMULA, or return None if not derivable.
"""

        max_retries = 3
        last_code = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                # Generate or regenerate code
                if attempt == 1:
                    response = self.llm.generate(
                        system="You are a Python expert. Generate fact derivation functions. Keep code simple and complete.",
                        user_message=prompt,
                        max_tokens=800,
                    )
                else:
                    # Retry with error context
                    retry_prompt = f"""Your previous derive() function failed with an error:

{last_error}

Previous code:
```python
{last_code}
```

Please fix the error and regenerate the derive() function.

Original request:
{prompt}"""
                    response = self.llm.generate(
                        system="You are a Python expert. Generate fact derivation functions. Keep code simple and complete.",
                        user_message=retry_prompt,
                        max_tokens=800,
                    )

                # Extract code
                code = response
                if "```python" in code:
                    code = code.split("```python", 1)[1].split("```", 1)[0]
                elif "```" in code:
                    code = code.split("```", 1)[1].split("```", 1)[0]

                last_code = code
                logger.debug(f"[_resolve_from_sub_plan] Attempt {attempt} generated code:\n{code}")

                # Validate syntax before executing
                try:
                    compile(code, "<sub_plan>", "exec")
                except SyntaxError as syn_err:
                    last_error = f"Syntax error: {syn_err}"
                    logger.warning(f"[_resolve_from_sub_plan] Attempt {attempt} syntax error: {syn_err}")
                    continue  # Retry

                # Execute the generated function
                local_ns = {"Fact": Fact, "FactSource": FactSource}
                exec(code, local_ns)

                derive_func = local_ns.get("derive")
                if not derive_func:
                    last_error = "No derive() function found in generated code. Please define a derive(resolver, params) function."
                    logger.warning(f"[_resolve_from_sub_plan] Attempt {attempt}: no derive() function")
                    continue  # Retry

                # Execute the derive function
                self._resolution_depth += 1
                try:
                    result = derive_func(self, params)
                    # Emit event: sub-plan expansion completed
                    if result and result.is_resolved:
                        self._emit_event("premise_expanded", {
                            "fact_name": fact_name,
                            "value": result.value,
                            "confidence": result.confidence,
                            "sub_facts": [f.name for f in result.because] if result.because else [],
                            "derivation_trace": result.derivation_trace,
                            "depth": self._resolution_depth,
                        })
                    return result
                except Exception as derive_err:
                    # Runtime error in derive() - retry with feedback
                    import traceback
                    tb = traceback.format_exc()
                    last_error = f"derive() raised an exception:\n{type(derive_err).__name__}: {derive_err}\n\nTraceback:\n{tb}"
                    logger.warning(f"[_resolve_from_sub_plan] Attempt {attempt} derive() failed: {derive_err}")
                    continue  # Retry
                finally:
                    self._resolution_depth -= 1

            except Exception as e:
                import traceback
                last_error = f"Unexpected error: {e}\n{traceback.format_exc()}"
                logger.error(f"[_resolve_from_sub_plan] Attempt {attempt} unexpected error: {e}")
                continue  # Retry

        # All retry attempts exhausted
        logger.error(f"[_resolve_from_sub_plan] All {max_retries} attempts failed for {fact_name}")
        return None

    def add_user_fact(
        self,
        fact_name: str,
        value: Any,
        reasoning: Optional[str] = None,
        source: FactSource = FactSource.USER_PROVIDED,
        description: Optional[str] = None,
        query: Optional[str] = None,
        source_name: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        table_name: Optional[str] = None,
        row_count: Optional[int] = None,
        context: Optional[str] = None,
        role_id: Optional[str] = None,
        **params,
    ) -> Fact:
        """
        Add a fact to the cache with specified source.

        This is used when:
        - User provides facts via natural language (USER_PROVIDED)
        - Facts are derived from query results (DATABASE)
        - Facts are computed/derived during analysis (RULE)

        Args:
            fact_name: Name of the fact (e.g., "march_attendance")
            value: The value provided
            reasoning: Optional explanation
            source: Where the fact came from (defaults to USER_PROVIDED)
            description: Human-friendly description of what this fact represents
            query: SQL query used to derive the fact (for DATABASE source)
            source_name: Name of the specific source (database name, API name, etc.)
            api_endpoint: API endpoint if from API source
            table_name: Table name in datastore if this is a table reference
            row_count: Number of rows if this is a table reference
            context: Detailed creation context (code, prompt, query that created this fact)
            role_id: Role that created this fact (None = shared context)
            **params: Parameters for the fact

        Returns:
            The created Fact
        """
        cache_key = self._cache_key(fact_name, params)

        fact = Fact(
            name=cache_key,
            value=value,
            confidence=1.0,
            source=source,
            description=description,
            reasoning=reasoning,
            query=query,
            source_name=source_name,
            api_endpoint=api_endpoint,
            table_name=table_name,
            row_count=row_count,
            context=context,
            role_id=role_id,
        )

        self._cache[cache_key] = fact
        self.resolution_log.append(fact)
        return fact

    def add_user_facts_from_text(self, user_text: str) -> list[Fact]:
        """
        Extract facts from natural language user input and add to cache.

        Uses LLM to parse statements like:
        - "There were 1 million people at the march"
        - "The revenue threshold should be $50,000"

        Args:
            user_text: Natural language text containing facts

        Returns:
            List of extracted and cached facts
        """
        if not self.llm:
            return []

        prompt = f"""Extract factual statements from this user input:

User input: {user_text}

For each fact, provide:
- FACT_NAME: A short identifier (e.g., "user_role", "revenue_threshold", "target_region")
- VALUE: The value (string, number, etc.)
- REASONING: Brief explanation

Extract these types of facts:
1. User context/persona (e.g., "my role as CFO" -> user_role: CFO)
2. Numeric values (e.g., "threshold of $50,000" -> revenue_threshold: 50000)
3. Preferences/constraints (e.g., "for the US region" -> target_region: US)
4. Time periods (e.g., "last quarter" -> time_period: last_quarter)

If the input contains multiple facts, list them all.
If the input contains no extractable facts, respond with "NO_FACTS".

Example format:
---
FACT_NAME: user_role
VALUE: CFO
REASONING: User identified their role as CFO
---
FACT_NAME: revenue_threshold
VALUE: 50000
REASONING: User specified revenue threshold should be $50,000
---
FACT_NAME: target_region
VALUE: US
REASONING: User is focused on US region analysis
---
"""

        try:
            response = self.llm.generate(
                system="You are a fact extraction assistant. Extract structured facts from natural language.",
                user_message=prompt,
                max_tokens=500,
            )

            if "NO_FACTS" in response:
                return []

            facts = []
            current_fact: dict[str, Any] = {}

            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("---"):
                    if current_fact.get("fact_name") and current_fact.get("value") is not None:
                        fact = self.add_user_fact(
                            fact_name=current_fact["fact_name"],
                            value=current_fact["value"],
                            reasoning=current_fact.get("reasoning"),
                            context=f"User text: {user_text}",
                        )
                        facts.append(fact)
                    current_fact = {}
                elif line.startswith("FACT_NAME:"):
                    current_fact["fact_name"] = line.split(":", 1)[1].strip()
                elif line.startswith("VALUE:"):
                    value_str = line.split(":", 1)[1].strip()
                    # Try to parse as number
                    try:
                        current_fact["value"] = float(value_str)
                        if current_fact["value"] == int(current_fact["value"]):
                            current_fact["value"] = int(current_fact["value"])
                    except ValueError:
                        current_fact["value"] = value_str
                elif line.startswith("REASONING:"):
                    current_fact["reasoning"] = line.split(":", 1)[1].strip()

            # Handle last fact if not terminated with ---
            if current_fact.get("fact_name") and current_fact.get("value") is not None:
                fact = self.add_user_fact(
                    fact_name=current_fact["fact_name"],
                    value=current_fact["value"],
                    reasoning=current_fact.get("reasoning"),
                    context=f"User text: {user_text}",
                )
                facts.append(fact)

            return facts

        except Exception as e:
            logger.debug(f"[extract_facts_from_text] Failed to extract facts: {e}")
            return []

    def get_unresolved_facts(self) -> list[Fact]:
        """Get all facts that could not be resolved."""
        return [f for f in self.resolution_log if f.source == FactSource.UNRESOLVED]

    def get_unresolved_summary(self) -> str:
        """Get a human-readable summary of unresolved facts."""
        unresolved = self.get_unresolved_facts()
        if not unresolved:
            return "All facts were resolved successfully."

        lines = ["The following facts could not be resolved:"]
        for fact in unresolved:
            lines.append(f"  - {fact.name}")
            if fact.reasoning:
                lines.append(f"    Reason: {fact.reasoning}")

        lines.append("")
        lines.append("You can provide these facts by describing them. For example:")
        lines.append('  "The attendance was 1 million people"')
        lines.append('  "The revenue threshold should be $50,000"')

        return "\n".join(lines)

    def clear_cache(self) -> None:
        """Clear the resolution cache."""
        self._cache.clear()

    def clear_unresolved(self) -> None:
        """Remove unresolved facts from log, allowing re-resolution."""
        self.resolution_log = [f for f in self.resolution_log if f.source != FactSource.UNRESOLVED]

    def export_cache(self) -> list[dict]:
        """Export all cached facts for persistence (for redo operations)."""
        return [fact.to_dict() for fact in self._cache.values()]

    def import_cache(self, facts: list[dict]) -> None:
        """Import facts into cache (for redo operations).

        This restores previously resolved facts so they don't need to be re-resolved.
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[IMPORT_CACHE] Importing {len(facts)} facts")
        for fact_dict in facts:
            try:
                fact = Fact.from_dict(fact_dict)
                self._cache[fact.name] = fact
                logger.debug(f"[IMPORT_CACHE] Imported: {fact.name} = {fact.value} (table_name={fact.table_name})")
            except (KeyError, ValueError) as e:
                # Skip invalid facts
                logger.debug(f"Skipping invalid fact during import: {e}")
        logger.debug(f"[IMPORT_CACHE] Cache keys after import: {list(self._cache.keys())}")

    def get_audit_log(self) -> list[dict]:
        """Get all resolutions for audit purposes."""
        return [f.to_dict() for f in self.resolution_log]

    def get_facts_as_dataframe(self) -> "pd.DataFrame":
        """
        Get all cached facts as a pandas DataFrame for export.

        Returns:
            DataFrame with columns: name, value, source, confidence, description, reasoning,
                                   context, query, api_endpoint, rule_name, table_name, row_count
        """
        import pandas as pd

        rows = []
        for cache_key, fact in self._cache.items():
            rows.append({
                "name": fact.name,
                "value": str(fact.value) if fact.value is not None else None,
                "source": fact.source.value if hasattr(fact.source, 'value') else str(fact.source),
                "confidence": fact.confidence,
                "description": fact.description or "",
                "reasoning": fact.reasoning or "",
                "context": fact.context or "",
                "query": fact.query or "",
                "api_endpoint": fact.api_endpoint or "",
                "rule_name": fact.rule_name or "",
                "table_name": fact.table_name or "",
                "row_count": fact.row_count,
            })

        columns = ["name", "value", "source", "confidence", "description", "reasoning",
                   "context", "query", "api_endpoint", "rule_name", "table_name", "row_count"]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)

    def explain(self, fact: Fact) -> str:
        """Generate a human-readable explanation of how a fact was derived."""
        return fact.derivation_trace

    def resolve_many_sync(
        self,
        fact_requests: list[tuple[str, dict]],
        on_resolve: Callable[[int, "Fact"], None] | None = None,
    ) -> list[Fact]:
        """
        Resolve multiple facts. Base implementation is sequential.

        AsyncFactResolver overrides this with parallel resolution.

        Args:
            fact_requests: List of (fact_name, params) tuples
            on_resolve: Optional callback called as each fact resolves.
                        Receives (index, fact) where index is the position in fact_requests.

        Returns:
            List of resolved Facts in same order as requests
        """
        results = []
        for idx, (name, params) in enumerate(fact_requests):
            fact = self.resolve(name, **params)
            results.append(fact)
            if on_resolve:
                on_resolve(idx, fact)
        return results

    def resolve_goal(self, question: str, schema_context: str = "") -> dict:
        """
        Resolve a question using Prolog-style goal decomposition.

        The question is decomposed into a goal and rules using Prolog syntax:
        - Goal: `answer(Q3, premium, Revenue)` - what we want to prove
        - Rules: `answer(Q, T, R) :- dates(Q, S, E), tier(T, C), revenue(C, S, E, R).`
        - Facts: Base predicates resolved through the hierarchy

        This approach naturally captures:
        - Variable binding/unification
        - Dependency graphs (implicit in rule bodies)
        - Recursive decomposition

        Args:
            question: The question to answer
            schema_context: Optional schema/database context

        Returns:
            Dict with:
            - answer: The resolved goal with bound variables
            - goal: The original goal predicate
            - rules: The decomposition rules
            - bindings: Variable -> value bindings
            - derivation: Prolog-style derivation trace
            - confidence: Overall confidence
            - unresolved: Goals that couldn't be resolved
        """
        if not self.llm:
            return {"answer": None, "error": "No LLM configured"}

        # Step 1: Decompose question into Prolog-style goal and rules
        decompose_prompt = f"""Express this question as a Prolog-style goal with decomposition rules.

Question: {question}

{f"Available data context:\\n{schema_context}" if schema_context else ""}

Format your response as:

GOAL: predicate(Arg1, Arg2, Result)

RULES:
predicate(A, B, R) :-
    subgoal1(A, X),
    subgoal2(B, Y),
    compute(X, Y, R).

SOURCES:
subgoal1: DATABASE | DOCUMENT | LLM_KNOWLEDGE | USER_PROVIDED
subgoal2: DATABASE | DOCUMENT | LLM_KNOWLEDGE | USER_PROVIDED

Example for "What was revenue by customer tier in Q3?":

GOAL: revenue_by_tier(q3, Tier, Revenue)

RULES:
revenue_by_tier(Quarter, Tier, Revenue) :-
    date_range(Quarter, StartDate, EndDate),
    tier_criteria(Tier, Criteria),
    customers_matching(Criteria, Customers),
    sum_revenue(Customers, StartDate, EndDate, Revenue).

SOURCES:
date_range: LLM_KNOWLEDGE (calendar knowledge)
tier_criteria: DOCUMENT or USER_PROVIDED (business definition)
customers_matching: DATABASE (query customers table)
sum_revenue: DATABASE (aggregate orders table)

Use uppercase for Variables that need binding, lowercase for constants.
"""

        self._emit_event("decomposing_goal", {"question": question})

        response = self.llm.generate(
            system="You decompose questions into Prolog-style goals and rules.",
            user_message=decompose_prompt,
            max_tokens=800,
        )

        # Parse the response
        goal = ""
        rules: list[str] = []
        sources: dict[str, str] = {}

        current_section = None
        rule_lines = []

        for line in response.split("\n"):
            line_stripped = line.strip()

            if line_stripped.startswith("GOAL:"):
                goal = line_stripped.split("GOAL:", 1)[1].strip()
                current_section = "goal"
            elif line_stripped == "RULES:":
                current_section = "rules"
            elif line_stripped == "SOURCES:":
                # Save accumulated rule
                if rule_lines:
                    rules.append(" ".join(rule_lines))
                    rule_lines = []
                current_section = "sources"
            elif current_section == "rules" and line_stripped:
                rule_lines.append(line_stripped)
                # Check if rule is complete (ends with period)
                if line_stripped.endswith("."):
                    rules.append(" ".join(rule_lines))
                    rule_lines = []
            elif current_section == "sources" and ":" in line_stripped:
                parts = line_stripped.split(":", 1)
                pred_name = parts[0].strip()
                source_hint = parts[1].strip()
                sources[pred_name] = source_hint

        self._emit_event("goal_decomposed", {
            "goal": goal,
            "rules": rules,
            "sources": list(sources.keys()),
        })

        # Step 2: Parse the goal to extract predicate name and arguments
        goal_pred, goal_args = self._parse_predicate(goal)
        if not goal_pred:
            return {"answer": None, "error": f"Could not parse goal: {goal}"}

        # Step 3: Resolve the goal using rules and hierarchy
        bindings: dict[str, Any] = {}
        resolved_facts: dict[str, Fact] = {}
        unresolved: list[str] = []

        # First, bind any constants in the goal
        for arg in goal_args:
            if arg and arg[0].islower():  # lowercase = constant
                bindings[arg] = arg

        # Parse rules to understand dependencies
        rule_deps = self._parse_rules(rules)

        # Resolve sub-goals in dependency order
        resolved_subgoals = self._resolve_subgoals(
            goal_pred, goal_args, rule_deps, sources, bindings, resolved_facts, unresolved
        )

        # Step 4: Build the derivation trace in Prolog style
        derivation = self._build_prolog_derivation(
            goal, rules, bindings, resolved_facts, unresolved
        )

        # Calculate confidence
        if resolved_facts:
            confidence = min(f.confidence for f in resolved_facts.values())
        else:
            confidence = 0.0

        # Build final answer by substituting bindings into goal
        answer = self._substitute_bindings(goal, bindings)

        return {
            "answer": answer,
            "goal": goal,
            "rules": rules,
            "sources": sources,
            "bindings": bindings,
            "derivation": derivation,
            "confidence": confidence,
            "unresolved": unresolved,
            "facts": {k: v.to_dict() for k, v in resolved_facts.items()},
        }

    def _parse_predicate(self, pred_str: str) -> tuple[str, list[str]]:
        """Parse a predicate string like 'foo(X, Y, Z)' into name and args."""
        pred_str = pred_str.strip().rstrip(".")
        if "(" not in pred_str:
            return pred_str, []

        name = pred_str.split("(")[0].strip()
        args_str = pred_str.split("(", 1)[1].rsplit(")", 1)[0]
        args = [a.strip() for a in args_str.split(",")]
        return name, args

    def _parse_rules(self, rules: list[str]) -> dict[str, list[str]]:
        """Parse rules to extract head -> body dependencies."""
        deps = {}
        for rule in rules:
            if ":-" not in rule:
                continue
            head, body = rule.split(":-", 1)
            head_pred, _ = self._parse_predicate(head)

            # Extract predicates from body
            body_preds = []
            # Simple parsing - split by comma but handle nested parens
            depth = 0
            current = ""
            for char in body:
                if char == "(":
                    depth += 1
                    current += char
                elif char == ")":
                    depth -= 1
                    current += char
                elif char == "," and depth == 0:
                    if current.strip():
                        pred_name, _ = self._parse_predicate(current.strip())
                        if pred_name:
                            body_preds.append(pred_name)
                    current = ""
                else:
                    current += char
            # Don't forget the last one
            if current.strip():
                pred_name, _ = self._parse_predicate(current.strip().rstrip("."))
                if pred_name:
                    body_preds.append(pred_name)

            deps[head_pred] = body_preds

        return deps

    def _resolve_subgoals(
        self,
        goal_pred: str,
        goal_args: list[str],
        rule_deps: dict[str, list[str]],
        sources: dict[str, str],
        bindings: dict[str, Any],
        resolved_facts: dict[str, Fact],
        unresolved: list[str],
    ) -> bool:
        """Recursively resolve subgoals using the hierarchy."""
        # Get subgoals for this predicate
        subgoals = rule_deps.get(goal_pred, [])

        if not subgoals:
            # This is a base predicate - resolve it directly
            return self._resolve_base_predicate(
                goal_pred, goal_args, sources, bindings, resolved_facts, unresolved
            )

        # Resolve each subgoal
        # First, identify which can be resolved in parallel (no shared unbound vars)
        independent = []
        dependent = []

        for subgoal in subgoals:
            # Check if this subgoal depends on variables from previous subgoals
            # For simplicity, assume all are potentially dependent for now
            dependent.append(subgoal)

        # Resolve subgoals
        for subgoal in dependent:
            sub_args = []  # Would need to track args from rule body
            success = self._resolve_subgoals(
                subgoal, sub_args, rule_deps, sources, bindings, resolved_facts, unresolved
            )
            if not success:
                # Continue trying others, but mark as unresolved
                pass

        return len(unresolved) == 0

    def _resolve_base_predicate(
        self,
        pred_name: str,
        pred_args: list[str],
        sources: dict[str, str],
        bindings: dict[str, Any],
        resolved_facts: dict[str, Fact],
        unresolved: list[str],
    ) -> bool:
        """Resolve a base predicate (leaf in the dependency tree)."""
        self._emit_event("resolving_predicate", {
            "predicate": pred_name,
            "args": pred_args,
        })

        # Try to resolve using our hierarchy
        fact = self.resolve(pred_name)

        if fact.is_resolved:
            resolved_facts[pred_name] = fact
            # Bind the result to any output variable
            if pred_args:
                # Last arg is typically the result/output
                result_var = pred_args[-1]
                if result_var and result_var[0].isupper():
                    bindings[result_var] = fact.value

            self._emit_event("predicate_resolved", {
                "predicate": pred_name,
                "value": fact.value,
                "source": fact.source.value,
            })
            return True
        else:
            unresolved.append(pred_name)
            self._emit_event("predicate_unresolved", {
                "predicate": pred_name,
                "source_hint": sources.get(pred_name, "unknown"),
            })
            return False

    def _substitute_bindings(self, term: str, bindings: dict[str, Any]) -> str:
        """Substitute variable bindings into a term."""
        result = term
        for var, value in bindings.items():
            # Replace variable with its bound value
            result = result.replace(var, str(value))
        return result

    def _build_prolog_derivation(
        self,
        goal: str,
        rules: list[str],
        bindings: dict[str, Any],
        resolved: dict[str, Fact],
        unresolved: list[str],
    ) -> str:
        """Build Prolog-style derivation trace."""
        lines = [
            "/* Query */",
            f"?- {goal}",
            "",
            "/* Rules */",
        ]
        for rule in rules:
            lines.append(rule)

        lines.append("")
        lines.append("/* Resolution */")

        for pred_name, fact in resolved.items():
            source = fact.source.value
            if fact.source_name:
                source = f"{source}:{fact.source_name}"
            lines.append(f"{pred_name}({fact.value}).  % from {source}, confidence={fact.confidence:.0%}")
            if fact.query:
                lines.append(f"  % SQL: {fact.query[:60]}...")

        if unresolved:
            lines.append("")
            lines.append("/* Unresolved (need user input) */")
            for pred in unresolved:
                lines.append(f"% {pred}(?).  % Could not resolve")

        lines.append("")
        lines.append("/* Answer */")
        answer = self._substitute_bindings(goal, bindings)
        lines.append(f"{answer}.")

        if bindings:
            lines.append("")
            lines.append("/* Bindings */")
            for var, val in bindings.items():
                if var[0].isupper():  # Only show variable bindings
                    lines.append(f"% {var} = {val}")

        return "\n".join(lines)

    def resolve_conclusion(self, question: str, schema_context: str = "") -> dict:
        """
        Resolve a question using template-based symbolic evaluation.

        1. Create a statement template with {variables} that would answer the question
        2. Extract variables from the template
        3. Resolve each variable through the hierarchy (recursively)
        4. Substitute resolved values back into the template
        5. Evaluate to get the final answer

        Args:
            question: The question to answer
            schema_context: Optional schema/database context

        Returns:
            Dict with:
            - answer: The final resolved answer
            - template: The statement template with variables
            - substitutions: Dict of variable -> resolved value
            - derivation: Human-readable trace
            - confidence: Overall confidence (min of all resolved facts)
            - unresolved: List of variables that couldn't be resolved
        """
        if not self.llm:
            return {"answer": None, "error": "No LLM configured"}

        # Step 1: Generate statement template with variables
        template_prompt = f"""Given this question, create a statement template that would answer it.
Use {{variable_name}} for values that need to be resolved.

Question: {question}

{f"Available data context: {schema_context}" if schema_context else ""}

Create a template where resolving all variables would answer the question.
For each variable, briefly describe what it represents.

Format:
TEMPLATE: <statement with {{variables}}>
VARIABLES:
- {{var1}}: description of what this variable represents
- {{var2}}: description of what this variable represents
...

Example for "What was total revenue in Q3 for premium customers?":
TEMPLATE: The total revenue in Q3 for premium customers was ${{total_revenue}}, calculated from {{order_count}} orders with an average of ${{avg_order_value}} per order.
VARIABLES:
- {{total_revenue}}: Sum of all order amounts for premium customers in Q3
- {{order_count}}: Number of orders from premium customers in Q3
- {{avg_order_value}}: Average order value (total_revenue / order_count)
- {{q3_start_date}}: Start date of Q3 (derived from "Q3")
- {{q3_end_date}}: End date of Q3 (derived from "Q3")
- {{premium_customer_ids}}: List of customer IDs classified as premium
"""

        self._emit_event("template_generating", {"question": question})

        template_response = self.llm.generate(
            system="You create answer templates with variables. Be thorough - include ALL variables needed.",
            user_message=template_prompt,
            max_tokens=800,
        )

        # Parse template and variables
        template = ""
        variables: dict[str, str] = {}  # var_name -> description

        for line in template_response.split("\n"):
            line = line.strip()
            if line.startswith("TEMPLATE:"):
                template = line.split("TEMPLATE:", 1)[1].strip()
            elif line.startswith("- {") and "}:" in line:
                # Parse variable definition
                var_part = line[2:]  # Remove "- "
                if "}:" in var_part:
                    var_name = var_part.split("}")[0].strip("{}")
                    var_desc = var_part.split("}:", 1)[1].strip()
                    variables[var_name] = var_desc

        self._emit_event("template_created", {
            "template": template,
            "variables": list(variables.keys()),
        })

        # Step 2: Identify dependencies between variables
        # Some variables depend on others (e.g., avg = total / count)
        dependency_prompt = f"""Given these variables, identify which ones depend on others.

Variables:
{chr(10).join(f"- {k}: {v}" for k, v in variables.items())}

For each variable, list its dependencies (other variables it needs).
Variables with no dependencies can be resolved in PARALLEL.

Format each line as:
{{variable}}: [{{dep1}}, {{dep2}}] or [] if no dependencies

Example:
{{total_revenue}}: []
{{order_count}}: []
{{avg_order_value}}: [{{total_revenue}}, {{order_count}}]
"""

        dep_response = self.llm.generate(
            system="You analyze variable dependencies.",
            user_message=dependency_prompt,
            max_tokens=400,
        )

        # Parse dependencies
        dependencies: dict[str, list[str]] = {var: [] for var in variables}
        for line in dep_response.split("\n"):
            if ":" in line and "{" in line:
                parts = line.split(":", 1)
                var_name = parts[0].strip().strip("{}")
                if var_name in dependencies:
                    deps_str = parts[1].strip()
                    if deps_str.startswith("[") and "]" in deps_str:
                        deps_list = deps_str[1:deps_str.index("]")]
                        deps = [d.strip().strip("{}") for d in deps_list.split(",") if d.strip()]
                        dependencies[var_name] = [d for d in deps if d in variables]

        # Step 3: Resolve variables in dependency order
        # Independent variables first (parallel), then dependent ones
        resolved: dict[str, Fact] = {}
        unresolved: list[str] = []

        # Find independent variables (no dependencies)
        independent = [v for v, deps in dependencies.items() if not deps]
        dependent = [v for v, deps in dependencies.items() if deps]

        self._emit_event("resolving_independent", {
            "independent": independent,
            "dependent": dependent,
        })

        # Resolve independent variables in parallel
        # Note: We don't pass description as a param since it would change the cache key
        if independent:
            fact_requests = [(var, {}) for var in independent]
            facts = self.resolve_many_sync(fact_requests)

            for var, fact in zip(independent, facts):
                if fact.is_resolved:
                    resolved[var] = fact
                    self._emit_event("variable_resolved", {
                        "variable": var,
                        "value": fact.value,
                        "source": fact.source.value,
                    })
                else:
                    unresolved.append(var)

        # Resolve dependent variables in dependency order
        max_iterations = len(dependent) + 1  # Prevent infinite loop
        iterations = 0

        while dependent and iterations < max_iterations:
            iterations += 1
            resolved_this_round = []

            for var in dependent:
                deps = dependencies[var]
                # Check if all dependencies are resolved
                if all(d in resolved for d in deps):
                    # First check if already in cache with just the variable name
                    cached = self._cache.get(var)
                    if cached and cached.is_resolved:
                        fact = cached
                    else:
                        # Try to resolve - pass dependencies for context but they won't
                        # affect the cache key since we use just the var name
                        fact = self.resolve(var)

                    if fact.is_resolved:
                        # Link to dependency facts
                        fact.because = [resolved[d] for d in deps]
                        resolved[var] = fact
                        resolved_this_round.append(var)
                        self._emit_event("variable_resolved", {
                            "variable": var,
                            "value": fact.value,
                            "source": fact.source.value,
                            "derived_from": deps,
                        })
                    else:
                        unresolved.append(var)
                        resolved_this_round.append(var)  # Remove from dependent list

            # Remove resolved variables from dependent list
            for var in resolved_this_round:
                if var in dependent:
                    dependent.remove(var)

        # Any remaining dependent variables couldn't be resolved
        unresolved.extend(dependent)

        # Step 4: Substitute resolved values into template
        final_template = template
        substitutions = {}
        for var, fact in resolved.items():
            placeholder = "{" + var + "}"
            if placeholder in final_template:
                value_str = str(fact.value)
                final_template = final_template.replace(placeholder, value_str)
                substitutions[var] = fact.value

        # Step 5: Calculate overall confidence
        if resolved:
            confidence = min(f.confidence for f in resolved.values())
        else:
            confidence = 0.0

        # Build derivation trace
        derivation = self._build_derivation_trace(
            template=template,
            resolved=resolved,
            unresolved=unresolved,
            variables=variables,
            answer=final_template,
        )

        return {
            "answer": final_template,
            "template": template,
            "variables": variables,
            "substitutions": substitutions,
            "derivation": derivation,
            "confidence": confidence,
            "unresolved": unresolved,
            "facts": {k: v.to_dict() for k, v in resolved.items()},
        }

    def _build_derivation_trace(
        self,
        template: str,
        resolved: dict[str, Fact],
        unresolved: list[str],
        variables: dict[str, str],
        answer: str,
    ) -> str:
        """Build a human-readable derivation trace with provenance."""
        lines = [
            "**Statement:**",
            f"  {template}",
            "",
            "**Variable Resolution:**",
        ]

        for var, fact in resolved.items():
            source_str = fact.source.value
            if fact.source_name:
                source_str = f"{fact.source.value}:{fact.source_name}"
            lines.append(f"  {{{var}}} = {fact.display_value}")
            lines.append(f"    source: {source_str}, confidence: {fact.confidence:.0%}")
            if fact.query:
                lines.append(f"    query: {fact.query[:80]}...")
            if fact.reasoning:
                lines.append(f"    reasoning: {fact.reasoning}")
            if fact.because:
                deps = ", ".join(f.name for f in fact.because)
                lines.append(f"    derived from: {deps}")

        if unresolved:
            lines.append("")
            lines.append("**Unresolved (need user input):**")
            for var in unresolved:
                desc = variables.get(var, "")
                lines.append(f"  {{{var}}}: {desc}")

        lines.append("")
        lines.append("**Conclusion:**")
        lines.append(f"  {answer}")

        return "\n".join(lines)

    def resolve_question(self, context: str) -> dict:
        """
        Resolve a verification question using fact-based derivation.

        Decomposes the question into required facts, resolves each fact,
        and generates a derivation trace showing how the conclusion was reached.

        Args:
            context: Context string containing the verification request and
                    any prior analysis results

        Returns:
            Dict with:
            - answer: The verification result
            - confidence: Overall confidence (0.0-1.0)
            - derivation: Human-readable derivation trace
            - sources: List of source citations
            - facts_resolved: List of fact names that were resolved
        """
        # Step 1: Decompose the question into required facts
        decompose_prompt = f"""Analyze this verification request and identify the specific facts needed to answer it.

{context}

List each required fact on its own line in the format:
FACT: <fact_name> - <description>

Example:
FACT: total_orders - Number of orders in the time period
FACT: discount_policy_max - Maximum allowed discount percentage
FACT: violations_count - Number of policy violations found

Be specific and exhaustive - list ALL facts needed to verify the claim."""

        decompose_text = self.llm.generate(
            system="You are analyzing verification requests to identify required facts.",
            user_message=decompose_prompt,
            max_tokens=1000,
        )

        # Parse required facts
        required_facts = []
        for line in decompose_text.split("\n"):
            if line.strip().startswith("FACT:"):
                fact_part = line.split("FACT:", 1)[1].strip()
                if " - " in fact_part:
                    fact_name, fact_desc = fact_part.split(" - ", 1)
                    required_facts.append((fact_name.strip(), fact_desc.strip()))
                else:
                    required_facts.append((fact_part, ""))

        # Step 2: Resolve each required fact
        resolved_facts = []
        derivation_lines = ["**Fact Resolution:**", ""]
        total_facts = len(required_facts)

        for idx, (fact_name, fact_desc) in enumerate(required_facts):
            # Emit event: starting to resolve this fact
            self._emit_event("premise_resolving", {
                "fact_name": fact_name,
                "description": fact_desc,
                "step": idx + 1,
                "total": total_facts,
            })
            # Check cache first
            if fact_name in self._cache:
                fact = self._cache[fact_name]
                derivation_lines.append(f"- {fact_name} = {fact.value} (cached)")
                resolved_facts.append(fact)
                # Emit resolved event for cached fact
                self._emit_event("premise_resolved", {
                    "fact_name": fact_name,
                    "value": fact.value,
                    "source": "cache",
                    "step": idx + 1,
                    "total": total_facts,
                })
            else:
                # Try to resolve the fact
                try:
                    fact = self.resolve(fact_name)
                    resolved_facts.append(fact)

                    # If fact has nested derivations (sub-plan resolution), show full trace
                    if fact.because:
                        # Show that this fact was derived from sub-facts
                        derivation_lines.append(f"- {fact_name} = {fact.value} (derived, confidence: {fact.confidence:.0%})")
                        derivation_lines.append(f"  ↳ Derived from:")
                        for sub_fact in fact.because:
                            sub_source = format_source_attribution(
                                sub_fact.source, sub_fact.source_name, sub_fact.api_endpoint
                            )
                            derivation_lines.append(f"    - {sub_fact.name} = {sub_fact.value} ({sub_source})")
                        source_detail = "derived"
                    else:
                        # Simple fact, use common source format
                        source_detail = format_source_attribution(
                            fact.source, fact.source_name, fact.api_endpoint
                        )
                        derivation_lines.append(f"- {fact_name} = {fact.value} ({source_detail}, confidence: {fact.confidence:.0%})")

                    # Emit resolved event
                    self._emit_event("premise_resolved", {
                        "fact_name": fact_name,
                        "value": fact.value,
                        "source": source_detail,
                        "confidence": fact.confidence,
                        "step": idx + 1,
                        "total": total_facts,
                    })
                except Exception as e:
                    # Mark as unresolved
                    unresolved = Fact(
                        name=fact_name,
                        value=None,
                        confidence=0.0,
                        source=FactSource.UNRESOLVED,
                        description=fact_desc,
                        reasoning=str(e),
                    )
                    self.resolution_log.append(unresolved)
                    derivation_lines.append(f"- {fact_name} = UNRESOLVED ({e})")
                    # Emit resolved event for unresolved fact
                    self._emit_event("premise_resolved", {
                        "fact_name": fact_name,
                        "value": None,
                        "source": "unresolved",
                        "error": str(e),
                        "step": idx + 1,
                        "total": total_facts,
                    })

        # Step 3: Synthesize the answer
        self._emit_event("synthesizing", {
            "message": "Synthesizing answer from resolved facts",
            "resolved_count": len([f for f in resolved_facts if f.is_resolved]),
            "total_count": total_facts,
        })

        facts_context = "\n".join([
            f"- {f.name}: {f.value} (confidence: {f.confidence:.0%})"
            for f in resolved_facts if f.is_resolved
        ])

        synthesis_prompt = f"""Based on the resolved facts, provide a verification answer.

{context}

Resolved Facts:
{facts_context}

Provide:
1. A direct answer to the verification question
2. The confidence level (HIGH/MEDIUM/LOW) with justification
3. Any caveats or limitations

Format your response as:
ANSWER: <your answer>
CONFIDENCE: <HIGH/MEDIUM/LOW> - <justification>
CAVEATS: <any limitations or caveats>"""

        synthesis_text = self.llm.generate(
            system="You are synthesizing verification results from resolved facts.",
            user_message=synthesis_prompt,
            max_tokens=1500,
        )

        # Parse the synthesis
        answer = ""
        confidence = 0.8  # Default
        caveats = ""

        for line in synthesis_text.split("\n"):
            if line.strip().startswith("ANSWER:"):
                answer = line.split("ANSWER:", 1)[1].strip()
            elif line.strip().startswith("CONFIDENCE:"):
                conf_part = line.split("CONFIDENCE:", 1)[1].strip()
                if conf_part.startswith("HIGH"):
                    confidence = 0.9
                elif conf_part.startswith("MEDIUM"):
                    confidence = 0.7
                elif conf_part.startswith("LOW"):
                    confidence = 0.5
            elif line.strip().startswith("CAVEATS:"):
                caveats = line.split("CAVEATS:", 1)[1].strip()

        # If we didn't parse an answer, use the full synthesis
        if not answer:
            answer = synthesis_text

        # Build derivation trace
        derivation_lines.append("")
        derivation_lines.append("**Conclusion:**")
        derivation_lines.append(answer)
        if caveats and caveats.lower() not in ("none", "n/a", "-"):
            derivation_lines.append("")
            derivation_lines.append(f"**Caveats:** {caveats}")

        derivation = "\n".join(derivation_lines)

        # Build sources list
        sources = []
        for fact in resolved_facts:
            if fact.is_resolved:
                source = {"type": fact.source.value, "description": f"{fact.name}: {fact.value}"}
                if fact.query:
                    source["query"] = fact.query
                sources.append(source)

        return {
            "answer": answer,
            "confidence": confidence,
            "derivation": derivation,
            "sources": sources,
            "facts_resolved": [f.name for f in resolved_facts if f.is_resolved],
        }


# Shared thread pool for running sync operations in async context
_DEFAULT_EXECUTOR = ThreadPoolExecutor(max_workers=10)


class RateLimitError(Exception):
    """Raised when an API rate limit is hit."""
    pass


class RateLimitExhaustedError(Exception):
    """Raised when max retries exceeded for rate limiting."""
    pass


@dataclass
class RateLimiterConfig:
    """Configuration for rate limiting."""
    max_concurrent: int = 5  # Max concurrent LLM calls
    max_retries: int = 3  # Max retry attempts on rate limit
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    jitter: float = 0.5  # Random jitter factor (0-1)


class RateLimiter:
    """
    Rate limiter with semaphore for concurrency control and exponential backoff.

    Prevents overwhelming LLM APIs with too many concurrent requests and
    handles rate limit errors gracefully with retries.

    Usage:
        limiter = RateLimiter(max_concurrent=5)

        async def call_llm():
            return await llm.generate(...)

        result = await limiter.execute(call_llm())
    """

    def __init__(self, config: Optional[RateLimiterConfig] = None):
        self.config = config or RateLimiterConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._request_count = 0
        self._rate_limit_hits = 0

    async def execute(self, coro_or_func):
        """
        Execute a coroutine or async function with rate limiting and exponential backoff.

        Args:
            coro_or_func: Either a coroutine to execute, or an async callable that
                         creates a new coroutine (for retry support)

        Returns:
            Result of the coroutine

        Raises:
            RateLimitExhaustedError: If max retries exceeded
        """
        # Determine if we got a callable (can retry) or a coroutine (single use)
        is_callable = callable(coro_or_func) and not asyncio.iscoroutine(coro_or_func)

        async with self._semaphore:
            self._request_count += 1

            for attempt in range(self.config.max_retries):
                try:
                    if is_callable:
                        return await coro_or_func()
                    else:
                        return await coro_or_func
                except Exception as e:
                    # Check if this is a rate limit error
                    error_str = str(e).lower()
                    is_rate_limit = any(
                        indicator in error_str
                        for indicator in ["rate limit", "ratelimit", "429", "too many requests"]
                    )

                    if not is_rate_limit:
                        raise  # Re-raise non-rate-limit errors

                    if not is_callable:
                        # Can't retry a single coroutine
                        raise

                    self._rate_limit_hits += 1

                    if attempt == self.config.max_retries - 1:
                        raise RateLimitExhaustedError(
                            f"Rate limit exceeded after {self.config.max_retries} retries"
                        ) from e

                    # Calculate backoff delay with exponential increase and jitter
                    delay = min(
                        self.config.base_delay * (2 ** attempt),
                        self.config.max_delay
                    )
                    jitter = random.uniform(0, self.config.jitter * delay)
                    total_delay = delay + jitter

                    await asyncio.sleep(total_delay)

    @property
    def stats(self) -> dict:
        """Get rate limiter statistics."""
        return {
            "total_requests": self._request_count,
            "rate_limit_hits": self._rate_limit_hits,
            "max_concurrent": self.config.max_concurrent,
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self._request_count = 0
        self._rate_limit_hits = 0


class AsyncFactResolver(FactResolver):
    """
    Async-enabled fact resolver with parallel resolution support.

    Provides significant speedup for I/O-bound fact resolution by:
    - Running LLM calls and database queries concurrently
    - Resolving multiple independent facts in parallel
    - Optionally trying multiple sources simultaneously

    Usage:
        resolver = AsyncFactResolver(llm=provider, schema_manager=sm)

        # Single async resolution
        fact = await resolver.resolve_async("customer_ltv", customer_id="acme")

        # Parallel resolution of multiple facts (3-5x speedup)
        facts = await resolver.resolve_many_async([
            ("customer_ltv", {"customer_id": "acme"}),
            ("customer_ltv", {"customer_id": "globex"}),
            ("revenue_threshold", {}),
        ])
    """

    def __init__(
        self,
        llm=None,
        schema_manager=None,
        config=None,
        strategy: Optional[ResolutionStrategy] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        parallel_sources: bool = False,
        rate_limiter: Optional[RateLimiter] = None,
        rate_limiter_config: Optional[RateLimiterConfig] = None,
    ):
        """
        Initialize AsyncFactResolver.

        Args:
            llm: LLM provider for queries/knowledge
            schema_manager: For database queries
            config: For config-based facts
            strategy: Resolution strategy configuration
            executor: Custom thread pool executor (uses shared default if not provided)
            parallel_sources: If True, try DATABASE + LLM_KNOWLEDGE + SUB_PLAN
                            concurrently instead of sequentially
            rate_limiter: Custom rate limiter instance (for shared limiting across resolvers)
            rate_limiter_config: Config for creating a new rate limiter
        """
        super().__init__(llm, schema_manager, config, strategy)
        self._executor = executor or _DEFAULT_EXECUTOR
        self._parallel_sources = parallel_sources

        # Rate limiting for LLM calls
        if rate_limiter:
            self._rate_limiter = rate_limiter
        elif rate_limiter_config:
            self._rate_limiter = RateLimiter(rate_limiter_config)
        else:
            # Default rate limiter
            self._rate_limiter = RateLimiter()

    @property
    def rate_limiter_stats(self) -> dict:
        """Get rate limiter statistics."""
        return self._rate_limiter.stats

    async def _call_llm_with_rate_limit(
        self,
        system: str,
        user_message: str,
        max_tokens: int = 500,
    ) -> str:
        """
        Call LLM with rate limiting and exponential backoff.

        Args:
            system: System prompt
            user_message: User message
            max_tokens: Maximum tokens in response

        Returns:
            LLM response string
        """
        async def _make_call():
            if hasattr(self.llm, 'async_generate'):
                return await self.llm.async_generate(
                    system=system,
                    user_message=user_message,
                    max_tokens=max_tokens,
                    executor=self._executor,
                )
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self._executor,
                    lambda: self.llm.generate(
                        system=system,
                        user_message=user_message,
                        max_tokens=max_tokens,
                    )
                )

        # Pass async function (not called) so it can be retried on rate limit
        return await self._rate_limiter.execute(_make_call)

    async def resolve_async(self, fact_name: str, **params) -> Fact:
        """
        Async version of resolve().

        Resolves a fact by trying sources in priority order (or parallel if configured).

        Args:
            fact_name: The fact to resolve
            **params: Parameters for the fact

        Returns:
            Fact with value, confidence, and provenance
        """
        cache_key = self._cache_key(fact_name, params)

        # Check cache first (sync, fast)
        if FactSource.CACHE in self.strategy.source_priority:
            cached = self._cache.get(cache_key)
            if cached and cached.confidence >= self.strategy.min_confidence:
                self.resolution_log.append(cached)
                return cached

        # Check config (sync, fast)
        if FactSource.CONFIG in self.strategy.source_priority:
            config_fact = self._resolve_from_config(fact_name, params)
            if config_fact and config_fact.is_resolved:
                self._cache[cache_key] = config_fact
                self.resolution_log.append(config_fact)
                return config_fact

        # Check rules - run in executor to allow true parallelism
        if FactSource.RULE in self.strategy.source_priority:
            rule_fact = await self._resolve_from_rule_async(fact_name, params)
            if rule_fact and rule_fact.is_resolved:
                self._cache[cache_key] = rule_fact
                self.resolution_log.append(rule_fact)
                return rule_fact

        # I/O-bound sources - run async
        if self._parallel_sources:
            fact = await self._resolve_parallel_sources(fact_name, params, cache_key)
        else:
            fact = await self._resolve_sequential_sources(fact_name, params, cache_key)

        if fact and fact.is_resolved:
            return fact

        # Could not resolve
        unresolved = Fact(
            name=cache_key,
            value=None,
            confidence=0.0,
            source=FactSource.UNRESOLVED,
            reasoning=f"Could not resolve fact: {fact_name} with params {params}"
        )
        self.resolution_log.append(unresolved)
        return unresolved

    async def _resolve_sequential_sources(
        self,
        fact_name: str,
        params: dict,
        cache_key: str,
    ) -> Optional[Fact]:
        """Try I/O-bound sources sequentially (default behavior).

        Tries cheap sources first (DATABASE, DOCUMENT, SUB_PLAN), then
        falls back to expensive sources (LLM_KNOWLEDGE) only if needed.
        """
        # Cheap I/O sources first
        cheap_sources = [
            s for s in self.strategy.source_priority
            if s in (FactSource.DATABASE, FactSource.DOCUMENT, FactSource.SUB_PLAN)
        ]
        # Expensive fallback
        expensive_sources = [
            s for s in self.strategy.source_priority
            if s == FactSource.LLM_KNOWLEDGE
        ]
        io_sources = cheap_sources + expensive_sources

        for source in io_sources:
            fact = await self._try_resolve_async(source, fact_name, params, cache_key)
            if fact and fact.is_resolved:
                if fact.confidence >= self.strategy.min_confidence:
                    self._cache[cache_key] = fact
                    self.resolution_log.append(fact)
                    return fact

        return None

    async def _resolve_parallel_sources(
        self,
        fact_name: str,
        params: dict,
        cache_key: str,
    ) -> Optional[Fact]:
        """
        Try I/O-bound sources in parallel, picking the best result.

        This can provide speedup when multiple sources might work,
        as we don't wait for each to fail before trying the next.

        NOTE: Only runs CHEAP I/O sources in parallel (DATABASE, DOCUMENT, SUB_PLAN).
        LLM_KNOWLEDGE is excluded - it's expensive (API cost + latency) and should
        only be used as a fallback if cheap sources fail.

        Selection strategy:
        1. Collect all successful results from parallel execution
        2. Filter by min_confidence threshold
        3. Pick best based on: (priority_index, -confidence) for stable ordering
        """
        import logging
        logger = logging.getLogger(__name__)

        # Only parallelize cheap I/O sources - exclude LLM_KNOWLEDGE (expensive)
        io_sources = [
            s for s in self.strategy.source_priority
            if s in (FactSource.DATABASE, FactSource.DOCUMENT, FactSource.SUB_PLAN)
        ]

        if not io_sources:
            return None

        # Create tasks for all sources
        tasks = [
            self._try_resolve_async(source, fact_name, params, cache_key)
            for source in io_sources
        ]

        # Use asyncio.gather with return_exceptions to get all results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect all valid results with their source priority
        valid_results: list[tuple[int, float, Fact]] = []
        for i, (source, result) in enumerate(zip(io_sources, results)):
            if isinstance(result, Exception):
                logger.debug(f"[_resolve_parallel] {source.value} raised: {result}")
                continue
            if result and result.is_resolved:
                if result.confidence >= self.strategy.min_confidence:
                    # Store (priority_index, confidence, fact)
                    valid_results.append((i, result.confidence, result))
                    logger.debug(f"[_resolve_parallel] {source.value}: conf={result.confidence:.2f}")

        if not valid_results:
            return None

        # Pick best result:
        # - Sort by priority index (lower = better), then by confidence (higher = better)
        # - This means DATABASE (priority 0) beats DOCUMENT (priority 1) at same confidence
        # - But if DOCUMENT has significantly higher confidence, it could win
        #   when confidence_weight_factor is set (future enhancement)
        valid_results.sort(key=lambda x: (x[0], -x[1]))
        best = valid_results[0]

        logger.debug(f"[_resolve_parallel] Picked {best[2].source.value} with confidence {best[1]:.2f}")

        self._cache[cache_key] = best[2]
        self.resolution_log.append(best[2])
        return best[2]

    async def _try_resolve_async(
        self,
        source: FactSource,
        fact_name: str,
        params: dict,
        cache_key: str,
    ) -> Optional[Fact]:
        """Async version of _try_resolve for I/O-bound sources."""
        if source == FactSource.DATABASE:
            return await self._resolve_from_database_async(fact_name, params)
        elif source == FactSource.DOCUMENT:
            return await self._resolve_from_document_async(fact_name, params)
        elif source == FactSource.LLM_KNOWLEDGE:
            return await self._resolve_from_llm_async(fact_name, params)
        elif source == FactSource.SUB_PLAN:
            return await self._resolve_from_sub_plan_async(fact_name, params)
        return None

    async def _resolve_from_document_async(
        self,
        fact_name: str,
        params: dict,
    ) -> Optional[Fact]:
        """Async document resolution - runs sync method in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._resolve_from_document(fact_name, params)
        )

    async def _resolve_from_rule_async(
        self,
        fact_name: str,
        params: dict,
    ) -> Optional[Fact]:
        """
        Async rule resolution that runs sync rules in executor.

        This enables true parallelism for rule-based facts by running
        blocking rule functions in a thread pool instead of on the event loop.

        After execution, checks cache again - if another concurrent request
        already cached a result for this fact, we discard ours and return
        the cached one. This ensures consistent values for concurrent requests.
        """
        rule = self._rules.get(fact_name)
        if not rule:
            return None

        cache_key = self._cache_key(fact_name, params)

        try:
            loop = asyncio.get_event_loop()
            # Run the sync rule function in the thread pool executor
            # This allows multiple rules to execute truly in parallel
            result = await loop.run_in_executor(
                self._executor,
                lambda: rule(self, params)
            )

            # After execution, check if another concurrent request already cached
            # a result for this fact. If so, discard ours and return cached.
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

            return result
        except Exception as e:
            logger.debug(f"[_resolve_from_rule_async] Rule failed for {fact_name}: {e}")
            return None

    async def _resolve_from_database_async(
        self,
        fact_name: str,
        params: dict,
    ) -> Optional[Fact]:
        """Async database resolution using LLM to generate SQL."""
        if not self.llm or not self.schema_manager:
            return None

        # Check if fact_name matches a table name - return "referenced" instead of loading data
        # This allows inferences to query the table directly from the original database
        fact_name_lower = fact_name.lower().strip()
        for full_name, table_meta in self.schema_manager.metadata_cache.items():
            # Match by table name (case-insensitive)
            if table_meta.name.lower() == fact_name_lower:
                # Store column metadata in reasoning for use by inferences
                columns = [c.name for c in table_meta.columns]
                reasoning = f"Table '{table_meta.name}' from database '{table_meta.database}'. Columns: {columns}"
                # Build descriptive value for UI display
                row_info = f"{table_meta.row_count:,} rows" if table_meta.row_count else "table"
                value_str = f"({table_meta.database}.{table_meta.name}) {row_info}"
                return Fact(
                    name=fact_name,
                    value=value_str,
                    source=FactSource.DATABASE,
                    source_name=table_meta.database,
                    reasoning=reasoning,
                    confidence=0.95,
                    table_name=table_meta.name,
                    row_count=table_meta.row_count,
                )

        schema_overview = self.schema_manager.get_overview()
        prompt = f"""I need to resolve this fact from the database:
Fact: {fact_name}
Parameters: {params}

Available schema:
{schema_overview}

If this fact can be DIRECTLY resolved with a SQL query, provide the query.

CRITICAL - When to respond NOT_POSSIBLE:
- If the fact asks for POLICY, RULES, GUIDELINES, or THRESHOLDS but no such table/config exists
- If you would need to ANALYZE PATTERNS or DERIVE rules from transactional data - that is NOT the same as having actual rules
- Statistical summaries (avg, count, distribution) are NOT policies - policies are prescriptive rules like "rating 5 = 10% raise"
- Do NOT return approximations or inferred rules as substitutes for explicitly stored policies

Respond in this format:
SQL: <your query here>
or
NOT_POSSIBLE: <reason>
"""

        try:
            # Use rate-limited LLM call
            response = await self._call_llm_with_rate_limit(
                system="You are a SQL expert. Generate precise queries to resolve facts.",
                user_message=prompt,
                max_tokens=500,
            )

            if "NOT_POSSIBLE" in response:
                return None

            if "SQL:" in response:
                sql = response.split("SQL:", 1)[1].strip()
                sql = sql.replace("```sql", "").replace("```", "").strip()

                db_names = list(self.config.databases.keys()) if self.config else []
                db_name = db_names[0] if db_names else None
                if db_name:
                    conn = self.schema_manager.get_connection(db_name)
                    import pandas as pd

                    # Run SQL in executor (blocking I/O)
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self._executor,
                        lambda: pd.read_sql(sql, conn)
                    )

                    cache_key = self._cache_key(fact_name, params)

                    if len(result) == 1 and len(result.columns) == 1:
                        # Scalar value - store directly
                        value = result.iloc[0, 0]
                        return Fact(
                            name=cache_key,
                            value=value,
                            confidence=1.0,
                            source=FactSource.DATABASE,
                            source_name=db_name,
                            query=sql,
                            context=f"SQL Query:\n{sql}",
                        )
                    else:
                        # Multi-row result - check if should store as table
                        value = result.to_dict('records')

                        if self._datastore and self._should_store_as_table(value):
                            # Store as table and return reference
                            table_name, row_count = self._store_value_as_table(
                                fact_name, value, source_name=db_name
                            )
                            return Fact(
                                name=cache_key,
                                value=f"table:{table_name}",
                                confidence=1.0,
                                source=FactSource.DATABASE,
                                source_name=db_name,
                                query=sql,
                                table_name=table_name,
                                row_count=row_count,
                                context=f"SQL Query:\n{sql}",
                            )
                        else:
                            # Small result - store inline
                            return Fact(
                                name=cache_key,
                                value=value,
                                confidence=1.0,
                                source=FactSource.DATABASE,
                                source_name=db_name,
                                query=sql,
                                context=f"SQL Query:\n{sql}",
                            )
        except Exception as e:
            logger.debug(f"[_resolve_from_database_async] Database resolution failed for {fact_name}: {e}")

        return None

    async def _resolve_from_llm_async(
        self,
        fact_name: str,
        params: dict,
    ) -> Optional[Fact]:
        """Async LLM knowledge resolution."""
        if not self.llm:
            return None

        prompt = f"""I need to know this fact:
Fact: {fact_name}
Parameters: {params}

Do you know this from your training? This could be:
- World knowledge (e.g., "capital of France")
- Industry standards (e.g., "typical VIP threshold is $10,000")
- Common heuristics (e.g., "underperforming means <80% of target")

If you know this, respond with:
VALUE: <the value>
CONFIDENCE: <0.0-1.0, how confident are you>
TYPE: knowledge | heuristic
REASONING: <brief explanation>

If you don't know, respond with:
UNKNOWN
"""

        try:
            # Use rate-limited LLM call
            response = await self._call_llm_with_rate_limit(
                system="You are a knowledgeable assistant. Provide facts you're confident about.",
                user_message=prompt,
                max_tokens=300,
            )

            if "UNKNOWN" in response:
                return None

            value = None
            confidence = 0.6
            reasoning = None
            source = FactSource.LLM_KNOWLEDGE

            for line in response.split("\n"):
                if line.startswith("VALUE:"):
                    value_str = line.split(":", 1)[1].strip()
                    try:
                        value = float(value_str)
                    except ValueError:
                        value = value_str
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("TYPE:"):
                    if "heuristic" in line.lower():
                        source = FactSource.LLM_HEURISTIC
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

            if value is not None:
                return Fact(
                    name=self._cache_key(fact_name, params),
                    value=value,
                    confidence=confidence,
                    source=source,
                    reasoning=reasoning,
                )
        except Exception as e:
            logger.debug(f"[_resolve_from_llm_async] LLM resolution failed for {fact_name}: {e}")

        return None

    async def _resolve_from_sub_plan_async(
        self,
        fact_name: str,
        params: dict,
    ) -> Optional[Fact]:
        """Async sub-plan resolution for complex derived facts."""
        if not self.strategy.allow_sub_plans:
            return None

        if self._resolution_depth >= self.strategy.max_sub_plan_depth:
            return None

        if not self.llm:
            return None

        prompt = f"""I need to derive this fact, but it's not directly available:
Fact: {fact_name}
Parameters: {params}

This fact needs to be computed from other facts.
Create a Python function that:
1. Uses resolver.resolve() to get the facts it depends on
2. Computes the final value
3. Returns a Fact with proper confidence (min of dependencies)

Example:
```python
def derive(resolver, params):
    revenue = resolver.resolve("total_revenue", customer_id=params["customer_id"])
    orders = resolver.resolve("order_count", customer_id=params["customer_id"])

    avg = revenue.value / orders.value if orders.value else 0

    return Fact(
        name=f"avg_order_value(customer_id={{params['customer_id']}})",
        value=avg,
        confidence=min(revenue.confidence, orders.confidence),
        source=FactSource.SUB_PLAN,
        because=[revenue, orders]
    )
```

Generate the derivation function for {fact_name}:
"""

        try:
            # Use rate-limited LLM call
            response = await self._call_llm_with_rate_limit(
                system="You are a Python expert. Generate fact derivation functions.",
                user_message=prompt,
                max_tokens=500,
            )

            code = response
            if "```python" in code:
                code = code.split("```python", 1)[1].split("```", 1)[0]
            elif "```" in code:
                code = code.split("```", 1)[1].split("```", 1)[0]

            local_ns = {"Fact": Fact, "FactSource": FactSource}
            exec(code, local_ns)

            derive_func = local_ns.get("derive")
            if derive_func:
                self._resolution_depth += 1
                try:
                    # Run sync derive function in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self._executor,
                        lambda: derive_func(self, params)
                    )
                    return result
                finally:
                    self._resolution_depth -= 1
        except Exception as e:
            logger.debug(f"[_resolve_from_sub_plan_async] Sub-plan resolution failed for {fact_name}: {e}")

        return None

    async def resolve_many_async(
        self,
        fact_requests: list[tuple[str, dict]],
        on_resolve: Callable[[int, "Fact"], None] | None = None,
    ) -> list[Fact]:
        """
        Resolve multiple facts in parallel.

        This is the primary method for achieving speedup with parallel resolution.
        Independent facts are resolved concurrently, providing 3-5x speedup for
        I/O-bound resolutions.

        Args:
            fact_requests: List of (fact_name, params) tuples
            on_resolve: Optional callback called as each fact resolves.
                        Receives (index, fact) where index is the position in fact_requests.

        Returns:
            List of resolved Facts in same order as requests

        Example:
            facts = await resolver.resolve_many_async([
                ("customer_ltv", {"customer_id": "acme"}),
                ("customer_ltv", {"customer_id": "globex"}),
                ("revenue_threshold", {}),
            ])
        """
        if on_resolve is None:
            # No callback - use gather for efficiency
            tasks = [
                self.resolve_async(name, **params)
                for name, params in fact_requests
            ]
            return await asyncio.gather(*tasks)
        else:
            # With callback - use as_completed to emit events as each resolves
            async def resolve_with_index(idx: int, name: str, params: dict) -> tuple[int, Fact]:
                fact = await self.resolve_async(name, **params)
                return idx, fact

            tasks = [
                resolve_with_index(i, name, params)
                for i, (name, params) in enumerate(fact_requests)
            ]

            # Results will be out of order as they complete
            results = [None] * len(fact_requests)
            for coro in asyncio.as_completed(tasks):
                idx, fact = await coro
                results[idx] = fact
                # Call callback as each fact completes
                on_resolve(idx, fact)

            return results

    def resolve_many_sync(
        self,
        fact_requests: list[tuple[str, dict]],
        on_resolve: Callable[[int, "Fact"], None] | None = None,
    ) -> list[Fact]:
        """
        Synchronous wrapper for resolve_many_async.

        Useful when calling from sync code that wants parallel resolution.
        Handles both cases: when called from sync context (no event loop)
        and when called from async context (running event loop).

        Args:
            fact_requests: List of (fact_name, params) tuples
            on_resolve: Optional callback called as each fact resolves.
                        Receives (index, fact) where index is the position in fact_requests.

        Returns:
            List of resolved Facts in same order as requests
        """
        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - we're in sync context, safe to use asyncio.run()
            return asyncio.run(self.resolve_many_async(fact_requests, on_resolve))

        # We're in an async context - need to run in a separate thread
        # to avoid "asyncio.run() cannot be called from a running event loop"
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run,
                self.resolve_many_async(fact_requests, on_resolve)
            )
            return future.result()
