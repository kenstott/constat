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
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union
from datetime import datetime


class FactSource(Enum):
    """Where a fact was resolved from."""
    CACHE = "cache"
    DATABASE = "database"
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
    query: Optional[str] = None  # SQL query if from database
    rule_name: Optional[str] = None  # Rule function name if derived
    reasoning: Optional[str] = None  # LLM explanation if from knowledge
    resolved_at: datetime = field(default_factory=datetime.now)

    @property
    def is_resolved(self) -> bool:
        return self.source != FactSource.UNRESOLVED

    @property
    def derivation_trace(self) -> str:
        """Human-readable derivation chain."""
        lines = [f"{self.name} = {self.value} (confidence: {self.confidence:.2f}, source: {self.source.value})"]
        if self.query:
            lines.append(f"  via SQL: {self.query}")
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
        return {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source.value,
            "query": self.query,
            "rule_name": self.rule_name,
            "reasoning": self.reasoning,
            "because": [f.name for f in self.because],
            "resolved_at": self.resolved_at.isoformat(),
        }


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
    ):
        self.llm = llm
        self.schema_manager = schema_manager
        self.config = config
        self.strategy = strategy or ResolutionStrategy()

        # Caches
        self._cache: dict[str, Fact] = {}
        self._rules: dict[str, RuleFunction] = {}

        # Resolution state (for sub-plan depth tracking)
        self._resolution_depth: int = 0

        # All resolutions this session (for audit)
        self.resolution_log: list[Fact] = []

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

                    # Convert result to appropriate value
                    if len(result) == 1 and len(result.columns) == 1:
                        value = result.iloc[0, 0]
                    else:
                        value = result.to_dict('records')

                    return Fact(
                        name=self._cache_key(fact_name, params),
                        value=value,
                        confidence=1.0,  # Database facts are certain
                        source=FactSource.DATABASE,
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
        **params,
    ) -> Fact:
        """
        Add a user-provided fact to the cache.

        This is used when the user provides missing facts via natural language
        follow-up. The fact is added to the cache with USER_PROVIDED source.

        Args:
            fact_name: Name of the fact (e.g., "march_attendance")
            value: The value provided by user
            reasoning: Optional explanation from user
            **params: Parameters for the fact

        Returns:
            The created Fact
        """
        cache_key = self._cache_key(fact_name, params)

        fact = Fact(
            name=cache_key,
            value=value,
            confidence=1.0,  # User-provided facts are treated as certain
            source=FactSource.USER_PROVIDED,
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
- FACT_NAME: A short identifier (e.g., "march_attendance", "revenue_threshold")
- VALUE: The value (number, string, etc.)
- REASONING: Brief explanation

If the input contains multiple facts, list them all.
If the input contains no extractable facts, respond with "NO_FACTS".

Example format:
---
FACT_NAME: march_attendance
VALUE: 1000000
REASONING: User stated there were 1 million people at the march
---
FACT_NAME: revenue_threshold
VALUE: 50000
REASONING: User specified revenue threshold should be $50,000
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
