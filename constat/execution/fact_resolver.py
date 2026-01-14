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

import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union
from datetime import datetime


class FactSource(Enum):
    """Where a fact was resolved from."""
    CACHE = "cache"
    DATABASE = "database"
    DOCUMENT = "document"  # From a reference document
    API = "api"  # From REST or GraphQL API
    LLM_KNOWLEDGE = "llm_knowledge"
    LLM_HEURISTIC = "llm_heuristic"
    RULE = "rule"  # Derived via a registered rule function
    SUB_PLAN = "sub_plan"  # Required a mini-plan to derive
    USER_PROVIDED = "user_provided"
    CONFIG = "config"  # From system prompt / config
    UNRESOLVED = "unresolved"


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
        result = {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source.value,
            "source_name": self.source_name,
            "query": self.query,
            "api_endpoint": self.api_endpoint,
            "rule_name": self.rule_name,
            "reasoning": self.reasoning,
            "because": [f.name for f in self.because],
            "resolved_at": self.resolved_at.isoformat(),
        }
        # Add table reference fields if present
        if self.table_name:
            result["table_name"] = self.table_name
            result["row_count"] = self.row_count
        return result


# Type for rule functions: (resolver, **params) -> Fact
RuleFunction = Callable[["FactResolver", dict], Fact]


@dataclass
class ResolutionStrategy:
    """Configuration for how facts should be resolved."""
    # Try these sources in order
    source_priority: list[FactSource] = field(default_factory=lambda: [
        FactSource.CACHE,
        FactSource.CONFIG,
        FactSource.RULE,
        FactSource.DATABASE,
        FactSource.LLM_KNOWLEDGE,
        FactSource.SUB_PLAN,
    ])

    # Confidence thresholds
    min_confidence: float = 0.0  # Accept any confidence
    prefer_database: bool = True  # Prefer DB over LLM when both possible

    # Sub-plan settings
    allow_sub_plans: bool = True
    max_sub_plan_depth: int = 3  # Prevent infinite recursion


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
    ):
        self.llm = llm
        self.schema_manager = schema_manager
        self.config = config
        self.strategy = strategy or ResolutionStrategy()
        self._event_callback = event_callback
        self._datastore = datastore  # Reference to session's datastore for table storage

        # Caches
        self._cache: dict[str, Fact] = {}
        self._rules: dict[str, RuleFunction] = {}

        # Resolution state (for sub-plan depth tracking)
        self._resolution_depth: int = 0

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

    def resolve(self, fact_name: str, **params) -> Fact:
        """
        Resolve a fact by name, trying sources in priority order.

        Args:
            fact_name: The fact to resolve (e.g., "customer_ltv", "revenue_threshold")
            **params: Parameters for the fact (e.g., customer_id="acme")

        Returns:
            Fact with value, confidence, and provenance
        """
        # Build cache key from name + params
        cache_key = self._cache_key(fact_name, params)

        # Try each source in priority order
        for source in self.strategy.source_priority:
            fact = self._try_resolve(source, fact_name, params, cache_key)
            if fact and fact.is_resolved:
                if fact.confidence >= self.strategy.min_confidence:
                    # Cache successful resolution
                    self._cache[cache_key] = fact
                    self.resolution_log.append(fact)
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

    def _cache_key(self, fact_name: str, params: dict) -> str:
        """Generate cache key from fact name and params."""
        if not params:
            return fact_name
        param_str = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{fact_name}({param_str})"

    def get_all_facts(self) -> dict[str, Fact]:
        """
        Get all cached facts.

        Returns:
            Dictionary mapping fact names/keys to Fact objects
        """
        return dict(self._cache)

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
        self._datastore.store(table_name, df)

        return table_name, len(df)

    def _try_resolve(
        self,
        source: FactSource,
        fact_name: str,
        params: dict,
        cache_key: str
    ) -> Optional[Fact]:
        """Try to resolve from a specific source."""

        if source == FactSource.CACHE:
            return self._cache.get(cache_key)

        elif source == FactSource.CONFIG:
            return self._resolve_from_config(fact_name, params)

        elif source == FactSource.RULE:
            return self._resolve_from_rule(fact_name, params)

        elif source == FactSource.DATABASE:
            return self._resolve_from_database(fact_name, params)

        elif source == FactSource.LLM_KNOWLEDGE:
            return self._resolve_from_llm(fact_name, params)

        elif source == FactSource.SUB_PLAN:
            return self._resolve_from_sub_plan(fact_name, params)

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

    def _resolve_from_database(self, fact_name: str, params: dict) -> Optional[Fact]:
        """Ask LLM to generate SQL query to resolve fact from database."""
        if not self.llm or not self.schema_manager:
            return None

        # Build prompt for LLM to generate SQL
        schema_overview = self.schema_manager.get_overview()
        prompt = f"""I need to resolve this fact from the database:
Fact: {fact_name}
Parameters: {params}

Available schema:
{schema_overview}

If this fact can be resolved with a SQL query, provide the query.
If not possible from database, respond with "NOT_POSSIBLE".

Respond in this format:
SQL: <your query here>
or
NOT_POSSIBLE: <reason>
"""

        try:
            response = self.llm.generate(
                system="You are a SQL expert. Generate precise queries to resolve facts.",
                user_message=prompt,
                max_tokens=500,
            )

            if "NOT_POSSIBLE" in response:
                return None

            # Extract SQL from response
            if "SQL:" in response:
                sql = response.split("SQL:", 1)[1].strip()
                # Clean up markdown if present
                sql = sql.replace("```sql", "").replace("```", "").strip()

                # Execute query
                # TODO: Get appropriate connection based on tables in query
                # For now, use first database
                db_names = list(self.config.databases.keys()) if self.config else []
                db_name = db_names[0] if db_names else None
                if db_name:
                    conn = self.schema_manager.get_connection(db_name)
                    import pandas as pd
                    result = pd.read_sql(sql, conn)

                    cache_key = self._cache_key(fact_name, params)

                    # Convert result to appropriate value
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
                                value=f"table:{table_name}",  # Reference, not data
                                confidence=1.0,
                                source=FactSource.DATABASE,
                                source_name=db_name,
                                query=sql,
                                table_name=table_name,
                                row_count=row_count,
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
                            )
        except Exception as e:
            # Query failed
            return None

        return None

    def _resolve_from_llm(self, fact_name: str, params: dict) -> Optional[Fact]:
        """Ask LLM for world knowledge or heuristics."""
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
                    except:
                        value = value_str
                elif line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":", 1)[1].strip())
                    except:
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
        except Exception:
            pass

        return None

    def _resolve_from_sub_plan(self, fact_name: str, params: dict) -> Optional[Fact]:
        """Generate a mini-plan to derive a complex fact."""
        if not self.strategy.allow_sub_plans:
            return None

        if self._resolution_depth >= self.strategy.max_sub_plan_depth:
            return None  # Prevent infinite recursion

        if not self.llm:
            return None

        # Emit event: starting sub-plan expansion
        self._emit_event("premise_expanding", {
            "fact_name": fact_name,
            "params": params,
            "depth": self._resolution_depth,
        })

        # Ask LLM to create a plan to derive this fact
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
            response = self.llm.generate(
                system="You are a Python expert. Generate fact derivation functions.",
                user_message=prompt,
                max_tokens=500,
            )

            # Extract code
            code = response
            if "```python" in code:
                code = code.split("```python", 1)[1].split("```", 1)[0]
            elif "```" in code:
                code = code.split("```", 1)[1].split("```", 1)[0]

            # Execute the generated function
            local_ns = {"Fact": Fact, "FactSource": FactSource}
            exec(code, local_ns)

            derive_func = local_ns.get("derive")
            if derive_func:
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
                finally:
                    self._resolution_depth -= 1
        except Exception:
            pass

        return None

    def add_user_fact(
        self,
        fact_name: str,
        value: Any,
        reasoning: Optional[str] = None,
        source: FactSource = FactSource.USER_PROVIDED,
        description: Optional[str] = None,
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
                )
                facts.append(fact)

            return facts

        except Exception:
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

    def get_audit_log(self) -> list[dict]:
        """Get all resolutions for audit purposes."""
        return [f.to_dict() for f in self.resolution_log]

    def explain(self, fact: Fact) -> str:
        """Generate a human-readable explanation of how a fact was derived."""
        return fact.derivation_trace

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
                            sub_source = sub_fact.source.value
                            if sub_fact.source_name:
                                sub_source = f"{sub_fact.source.value}:{sub_fact.source_name}"
                            derivation_lines.append(f"    - {sub_fact.name} = {sub_fact.value} ({sub_source})")
                        source_detail = "derived"
                    else:
                        # Simple fact, show source
                        source_detail = fact.source.value
                        if fact.query:
                            source_detail = f"SQL query"
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
        """Try I/O-bound sources sequentially (default behavior)."""
        io_sources = [
            s for s in self.strategy.source_priority
            if s in (FactSource.DATABASE, FactSource.LLM_KNOWLEDGE, FactSource.SUB_PLAN)
        ]

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
        Try I/O-bound sources in parallel, taking first successful result.

        This can provide speedup when multiple sources might work,
        as we don't wait for each to fail before trying the next.
        """
        io_sources = [
            s for s in self.strategy.source_priority
            if s in (FactSource.DATABASE, FactSource.LLM_KNOWLEDGE, FactSource.SUB_PLAN)
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

        # Find first successful result in priority order
        for source, result in zip(io_sources, results):
            if isinstance(result, Exception):
                continue
            if result and result.is_resolved:
                if result.confidence >= self.strategy.min_confidence:
                    self._cache[cache_key] = result
                    self.resolution_log.append(result)
                    return result

        return None

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
        elif source == FactSource.LLM_KNOWLEDGE:
            return await self._resolve_from_llm_async(fact_name, params)
        elif source == FactSource.SUB_PLAN:
            return await self._resolve_from_sub_plan_async(fact_name, params)
        return None

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
        except Exception:
            # Rule failed - log but don't crash
            return None

    async def _resolve_from_database_async(
        self,
        fact_name: str,
        params: dict,
    ) -> Optional[Fact]:
        """Async database resolution using LLM to generate SQL."""
        if not self.llm or not self.schema_manager:
            return None

        schema_overview = self.schema_manager.get_overview()
        prompt = f"""I need to resolve this fact from the database:
Fact: {fact_name}
Parameters: {params}

Available schema:
{schema_overview}

If this fact can be resolved with a SQL query, provide the query.
If not possible from database, respond with "NOT_POSSIBLE".

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
                            )
        except Exception:
            pass

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
        except Exception:
            pass

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
        except Exception:
            pass

        return None

    async def resolve_many_async(
        self,
        fact_requests: list[tuple[str, dict]],
    ) -> list[Fact]:
        """
        Resolve multiple facts in parallel.

        This is the primary method for achieving speedup with parallel resolution.
        Independent facts are resolved concurrently, providing 3-5x speedup for
        I/O-bound resolutions.

        Args:
            fact_requests: List of (fact_name, params) tuples

        Returns:
            List of resolved Facts in same order as requests

        Example:
            facts = await resolver.resolve_many_async([
                ("customer_ltv", {"customer_id": "acme"}),
                ("customer_ltv", {"customer_id": "globex"}),
                ("revenue_threshold", {}),
            ])
        """
        tasks = [
            self.resolve_async(name, **params)
            for name, params in fact_requests
        ]
        return await asyncio.gather(*tasks)

    def resolve_many_sync(
        self,
        fact_requests: list[tuple[str, dict]],
    ) -> list[Fact]:
        """
        Synchronous wrapper for resolve_many_async.

        Useful when calling from sync code that wants parallel resolution.
        Handles both cases: when called from sync context (no event loop)
        and when called from async context (running event loop).

        Args:
            fact_requests: List of (fact_name, params) tuples

        Returns:
            List of resolved Facts in same order as requests
        """
        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - we're in sync context, safe to use asyncio.run()
            return asyncio.run(self.resolve_many_async(fact_requests))

        # We're in an async context - need to run in a separate thread
        # to avoid "asyncio.run() cannot be called from a running event loop"
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run,
                self.resolve_many_async(fact_requests)
            )
            return future.result()
