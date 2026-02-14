# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Session mixin: fact access, specs, user facts, cache management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from ._types import (
    Fact,
    FactDependency,
    FactSource,
    ProofNode,
    ResolutionSpec,
)

if TYPE_CHECKING:
    from . import FactResolver

logger = logging.getLogger(__name__)


class SessionMixin:

    @staticmethod
    def _cache_key(fact_name: str, params: dict) -> str:
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

    def get_fact(self: "FactResolver", name: str, verify_tables: bool = True) -> Optional[Fact]:
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

    def get_all_facts(self: "FactResolver") -> dict[str, Fact]:
        """
        Get all cached facts.

        Returns:
            Dictionary mapping fact names/keys to Fact objects
        """
        return dict(self._cache)

    def get_facts_for_role(
        self: "FactResolver",
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

    def get_shared_facts(self: "FactResolver") -> dict[str, Fact]:
        """Get only shared facts (role_id=None)."""
        return {k: v for k, v in self._cache.items() if v.role_id is None}

    def get_role_facts(self: "FactResolver", role_id: str) -> dict[str, Fact]:
        """Get only facts for a specific role (excludes shared)."""
        return {k: v for k, v in self._cache.items() if v.role_id == role_id}

    def promote_fact_to_shared(self: "FactResolver", name: str) -> bool:
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
        self: "FactResolver",
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
        self: "FactResolver",
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
        import json

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
                max_tokens=self.llm.max_output_tokens,
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
        self: "FactResolver",
        spec: ResolutionSpec,
        params: dict,
        build_proof: bool = True,
    ) -> tuple[Fact, Optional[ProofNode]]:
        """
        Resolve a fact using its ResolutionSpec.

        For leaf facts: execute SQL or doc query directly
        For derived facts: resolve dependencies, then execute logic
        """
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

    def add_user_fact(
        self: "FactResolver",
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

    def add_user_facts_from_text(self: "FactResolver", user_text: str) -> list[Fact]:
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

CRITICAL: Only extract facts from DECLARATIVE STATEMENTS where the user is ASSERTING or PROVIDING information.
DO NOT extract facts from:
- Questions asking "where does X come from?" or "what is the source of X?"
- Questions asking for verification or clarification
- Hypothetical statements like "if the threshold were X"
- Numbers mentioned as part of a question being asked

The user must be STATING a fact, not ASKING ABOUT a fact.

For each fact, provide:
- FACT_NAME: A short identifier (e.g., "user_role", "revenue_threshold", "target_region")
- VALUE: The value (string, number, etc.)
- REASONING: Brief explanation

Extract these types of facts:
1. User context/persona (e.g., "my role as CFO" -> user_role: CFO)
2. Numeric values asserted by user (e.g., "the threshold should be $50,000" -> revenue_threshold: 50000)
3. Preferences/constraints (e.g., "for the US region" -> target_region: US)
4. Time periods (e.g., "use last quarter's data" -> time_period: last_quarter)

Examples of what NOT to extract (these are questions, not facts):
- "Where does the 15% raise limit come from?" -> NO_FACTS (asking about source)
- "Why is the threshold $50,000?" -> NO_FACTS (asking for explanation)
- "Is 15% the correct maximum?" -> NO_FACTS (asking for verification)

Examples of what TO extract:
- "The maximum raise should be 15%" -> raise_max: 0.15
- "I'm the CFO of the company" -> user_role: CFO

If the input is a question or contains no declarative facts, respond with "NO_FACTS".

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
                max_tokens=self.llm.max_output_tokens,
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

    def get_unresolved_facts(self: "FactResolver") -> list[Fact]:
        """Get all facts that could not be resolved."""
        return [f for f in self.resolution_log if f.source == FactSource.UNRESOLVED]

    def get_unresolved_summary(self: "FactResolver") -> str:
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

    def clear_cache(self: "FactResolver") -> None:
        """Clear the resolution cache."""
        self._cache.clear()

    def clear_unresolved(self: "FactResolver") -> None:
        """Remove unresolved facts from log, allowing re-resolution."""
        self.resolution_log = [f for f in self.resolution_log if f.source != FactSource.UNRESOLVED]

    def clear_session(self: "FactResolver") -> None:
        """Clear all session-level facts (cache and resolution log).

        Call this when starting a new query to reset session state.
        Does NOT affect persistent facts in FactStore.
        """
        self._cache.clear()
        self.resolution_log.clear()

    def export_cache(self: "FactResolver") -> list[dict]:
        """Export all cached facts for persistence (for redo operations)."""
        return [fact.to_dict() for fact in self._cache.values()]

    def import_cache(self: "FactResolver", facts: list[dict]) -> None:
        """Import facts into cache (for redo operations).

        This restores previously resolved facts so they don't need to be re-resolved.
        """
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

    def get_audit_log(self: "FactResolver") -> list[dict]:
        """Get all resolutions for audit purposes."""
        return [f.to_dict() for f in self.resolution_log]

    def get_facts_as_dataframe(self: "FactResolver") -> "pd.DataFrame":
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

    @staticmethod
    def explain(fact: Fact) -> str:
        """Generate a human-readable explanation of how a fact was derived."""
        return fact.derivation_trace

    def resolve_many_sync(
        self: "FactResolver",
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
