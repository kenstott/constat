# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""ProbLog-based fact resolution with automatic dependency tracking.

ProbLog drives the resolution - it calls out to Python to execute queries
as needed, following the dependency chain automatically.

The key insight: fact resolution IS logic programming.
- Facts are predicates (ground truths from queries)
- Dependencies are rules (symbolic derivations)
- Resolution is Prolog's depth-first search
- Proofs come for free from ProbLog

Usage:
    resolver = ProbLogResolver(schema_manager, config)

    # Register query executor
    resolver.register_sql_executor(lambda db, sql: pd.read_sql(sql, engines[db]))

    # Resolve a fact
    result = resolver.resolve_fact("monthly_revenue_by_tier", params={})
    print(result.proof.to_trace())  # Shows symbolic proof tree
"""

import atexit
import logging
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Dict, List

logger = logging.getLogger(__name__)


class FactSource(Enum):
    """Source of a resolved fact."""
    DATABASE = "database"
    DOCUMENT = "document"
    DERIVED = "derived"
    LLM = "llm"
    CONFIG = "config"
    CACHE = "cache"
    USER_PROVIDED = "user_provided"


@dataclass
class ProofNode:
    """A node in the derivation proof tree."""
    predicate: str
    value: Any
    source: FactSource
    probability: float = 1.0
    evidence: Optional[str] = None  # SQL query, doc excerpt, etc.
    children: List["ProofNode"] = field(default_factory=list)

    def to_trace(self, indent: int = 0) -> str:
        """Render as human-readable proof trace."""
        prefix = "  " * indent
        lines = [
            f"{prefix}∴ {self.predicate} = {self._format_value()}",
            f"{prefix}  [source: {self.source.value}, prob: {self.probability:.2f}]",
        ]
        if self.evidence:
            ev = self.evidence[:100] + "..." if len(self.evidence) > 100 else self.evidence
            lines.append(f"{prefix}  [evidence: {ev}]")
        for child in self.children:
            lines.append(child.to_trace(indent + 1))
        return "\n".join(lines)

    def _format_value(self) -> str:
        """Format value for display."""
        if self.value is None:
            return "None"
        if isinstance(self.value, float):
            return f"{self.value:,.2f}"
        if isinstance(self.value, int):
            return f"{self.value:,}"
        return str(self.value)[:50]


@dataclass
class ResolvedFact:
    """A fact resolved by ProbLog."""
    name: str
    value: Any
    probability: float
    source: FactSource
    proof: Optional[ProofNode] = None
    query: Optional[str] = None  # SQL or doc query used


# Global registry for resolution logging (set by ProbLogResolver before each run)
_resolution_log: List[Dict] = []
_sql_executor: Optional[Callable] = None
_nosql_executor: Optional[Callable] = None
_doc_searcher: Optional[Callable] = None
_fact_resolver: Optional[Any] = None  # Reference to FactResolver for hierarchy calls
_config: Optional[Any] = None  # Application config
_llm: Optional[Any] = None  # LLM provider


class ProbLogResolver:
    """
    Fact resolver using ProbLog for symbolic reasoning.

    ProbLog drives the resolution:
    1. LLM generates ProbLog rules with calls to Python executors
    2. ProbLog resolves dependencies, calling out to Python as needed
    3. Proof tree is built from the resolution log

    This keeps query logic in ProbLog while executing I/O in Python.
    """

    def __init__(self, schema_manager=None, config=None, llm=None):
        """
        Initialize the ProbLog resolver.

        Args:
            schema_manager: For database connections and schema info
            config: Application config with database definitions
            llm: LLM for generating resolution rules
        """
        self.schema_manager = schema_manager
        self.config = config
        self.llm = llm

        # Query executors
        self._sql_executor: Optional[Callable[[str, str], Any]] = None
        self._nosql_executor: Optional[Callable[[str, str], Any]] = None
        self._doc_searcher: Optional[Callable[[str], Any]] = None

        # Track resolved ground facts (for proof building)
        self._ground_facts: Dict[str, Dict] = {}

        # Cache resolved facts
        self._cache: Dict[str, ResolvedFact] = {}

        # Temporary module file
        self._module_path: Optional[str] = None

        # Reference to FactResolver for hierarchy calls
        self._fact_resolver = None

    def register_sql_executor(self, executor: Callable[[str, str], Any]):
        """Register the SQL execution callback."""
        self._sql_executor = executor

    def register_nosql_executor(self, executor: Callable[[str, str], Any]):
        """Register the NoSQL execution callback."""
        self._nosql_executor = executor

    def register_doc_search(self, search_fn: Callable[[str], Any]):
        """Register the document search callback."""
        self._doc_searcher = search_fn

    def register_fact_resolver(self, fact_resolver):
        """Register the FactResolver for hierarchy-based resolution."""
        self._fact_resolver = fact_resolver

    def clear_cache(self):
        """Clear the resolution cache."""
        self._cache.clear()
        self._ground_facts.clear()

    def resolve_fact(
        self,
        fact_name: str,
        params: Dict = None,
        generate_rules: bool = True,
        user_facts: Dict[str, Any] = None,
    ) -> ResolvedFact:
        """
        Resolve a fact using ProbLog-driven resolution.

        ProbLog drives the resolution:
        1. LLM generates ProbLog rules with external calls
        2. Python module with executors is generated
        3. ProbLog resolves, calling out to Python as needed
        4. Proof tree is built from resolution log

        Args:
            fact_name: The fact to resolve (e.g., "monthly_revenue_by_tier")
            params: Parameters for the fact
            generate_rules: If True, ask LLM to generate resolution rules
            user_facts: User-provided facts (from clarifications) to include

        Returns:
            ResolvedFact with value, probability, and proof
        """
        global _resolution_log, _sql_executor, _nosql_executor, _doc_searcher

        params = params or {}
        user_facts = user_facts or {}
        cache_key = f"{fact_name}({','.join(f'{k}={v}' for k,v in params.items())})"

        # Check cache
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return ResolvedFact(
                name=fact_name,
                value=cached.value,
                probability=cached.probability,
                source=FactSource.CACHE,
                proof=cached.proof,
            )

        # Clear tracking for fresh resolution
        self._ground_facts.clear()
        _resolution_log = []

        # Set global executors for the Python module to use
        _sql_executor = self._sql_executor
        _nosql_executor = self._nosql_executor
        _doc_searcher = self._doc_searcher
        _fact_resolver = self._fact_resolver
        _config = self.config
        _llm = self.llm

        # Generate resolution rules
        if generate_rules and self.llm:
            rules = self._generate_resolution_rules(fact_name, params, user_facts)
        else:
            rules = f"{fact_name}(unknown)."

        # Create the Python executor module
        module_path = self._create_executor_module()

        # Build user-provided facts section
        user_facts_prolog = ""
        if user_facts:
            for name, value in user_facts.items():
                if isinstance(value, str):
                    user_facts_prolog += f"{name}('{value}').\n"
                else:
                    user_facts_prolog += f"{name}({value}).\n"
                self._ground_facts[name] = {
                    "value": value,
                    "source": FactSource.USER_PROVIDED,
                    "query": None,
                }

        # Run ProbLog resolution
        result = self._run_problog(fact_name, rules, user_facts_prolog, module_path)

        if result:
            self._cache[cache_key] = result
            return result

        return ResolvedFact(
            name=fact_name,
            value=None,
            probability=0.0,
            source=FactSource.DERIVED,
        )

    def _generate_resolution_rules(
        self,
        fact_name: str,
        params: Dict,
        user_facts: Dict[str, Any],
    ) -> str:
        """
        Generate ProbLog rules for fact resolution.

        Returns ProbLog rules that call external Python functions for data access.
        """
        # Get schema info
        schema_info = ""
        if self.schema_manager:
            schema_info = self.schema_manager.get_overview()

        # Get database info
        db_type = "sqlite"
        db_name = "db"
        if self.config and self.config.databases:
            db_name = list(self.config.databases.keys())[0]
            db_config = self.config.databases.get(db_name)
            if db_config:
                db_type = db_config.type or "sqlite"

        # Build user facts section
        user_facts_section = ""
        if user_facts:
            user_facts_section = "\nUser-provided facts (already available as Prolog facts):\n" + "\n".join(
                f"- {name}({value})" for name, value in user_facts.items()
            )

        prompt = f"""Generate ProbLog rules to resolve this fact:

Fact to resolve: {fact_name}
Parameters: {params}

Database: {db_name} (type: {db_type})
Schema:
{schema_info}
{user_facts_section}

Available external predicates (call Python for data access):

HIERARCHY RESOLUTION (preferred):
- resolve(FactName, Result) - Resolve through full hierarchy (cache, rules, database, docs, LLM)

DIRECT ACCESS (when hierarchy not suitable):
- sql_query(Database, SQL, Result) - Execute SQL and get scalar result
- sql_query_table(Database, SQL, Rows) - Execute SQL and get row count
- nosql_query(Database, Query, Result) - Execute NoSQL query
- doc_search(Query, Result) - Search documents
- config_value(Key, Result) - Get config value
- llm_query(Question, Answer) - Query LLM for world knowledge

Generate ProbLog rules that:
1. Use external predicates to fetch data from databases
2. Derive the requested fact through symbolic reasoning
3. Handle dependencies - if a fact depends on other facts, define rules for those too

Example:
```prolog
% Fetch data via external calls
customer_count(X) :- sql_query('{db_name}', 'SELECT COUNT(*) FROM customers', X).
order_count(X) :- sql_query('{db_name}', 'SELECT COUNT(*) FROM orders', X).

% Derived fact through symbolic reasoning
avg_orders_per_customer(Avg) :-
    customer_count(C),
    order_count(O),
    Avg is O / C.
```

IMPORTANT:
- Use {db_type} SQL syntax
- For SQLite: strftime('%Y-%m', col), date('now', '-6 months')
- Do NOT use schema prefixes in SQL
- Queries should return scalar values when possible (COUNT, SUM, AVG)
- User-provided facts are already available, don't query for them
- Every intermediate value should be traceable

Generate the rules (just the Prolog code, no markdown):"""

        try:
            response = self.llm.generate(
                system="You generate ProbLog rules for fact resolution. Output ONLY valid Prolog code.",
                user_message=prompt,
                max_tokens=self.llm.max_output_tokens,
            )

            # Clean up response - remove markdown code blocks if present
            rules = response.strip()
            if rules.startswith("```"):
                lines = rules.split("\n")
                # Remove first and last lines if they're code fence markers
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                rules = "\n".join(lines)

            # Remove any remaining backticks or prolog markers
            rules = rules.replace("```prolog", "").replace("```", "").strip()

            logger.debug(f"Generated rules:\n{rules}")
            return rules

        except Exception as e:
            logger.error(f"Rule generation error: {e}")
            return f"{fact_name}(unknown)."

    def _create_executor_module(self) -> str:
        """
        Create a temporary Python module with @problog_export decorated functions.

        This module is loaded by ProbLog via use_module and provides the bridge
        between ProbLog and Python for executing queries.
        """
        # Import statement for the module
        module_code = '''"""Auto-generated ProbLog executor module.

This module provides the bridge between ProbLog and Python, allowing ProbLog
to call out to the full fact resolution hierarchy:

1. resolve(FactName, Value) - Resolves through the full FactResolver hierarchy
2. sql_query(DB, SQL, Value) - Direct SQL query
3. nosql_query(DB, Query, Value) - Direct NoSQL query
4. doc_search(Query, Value) - Document search
5. config_value(Key, Value) - Get config value
6. llm_query(Question, Answer) - Query LLM for knowledge

ProbLog drives the symbolic reasoning, calling out to Python as needed.
"""
from problog.extern import problog_export
from problog.logic import Constant, Term

# Import the global executors from problog_resolver
import constat.execution.problog_resolver as resolver


@problog_export('+str', '-term')
def resolve(fact_name):
    """Resolve a fact through the full FactResolver hierarchy."""
    name = str(fact_name).strip("'").strip('"')

    if resolver._fact_resolver is None:
        raise ValueError("FactResolver not registered")

    try:
        # Call the full hierarchy
        fact = resolver._fact_resolver.resolve(name)

        value = fact.value

        # Convert numpy types to Python native types
        if hasattr(value, 'item'):
            value = value.item()

        # Log for proof building
        resolver._resolution_log.append({
            "type": "resolve",
            "fact_name": name,
            "value": value,
            "source": str(fact.source),
            "confidence": fact.confidence,
        })

        # Return as ProbLog term
        if isinstance(value, (int, float)):
            return Constant(value)
        return Term(str(value))

    except Exception as e:
        resolver._resolution_log.append({
            "type": "resolve",
            "fact_name": name,
            "error": str(e),
        })
        raise


@problog_export('+str', '-term')
def config_value(key):
    """Get a value from the application config."""
    k = str(key).strip("'").strip('"')

    if resolver._config is None:
        raise ValueError("Config not registered")

    try:
        # Navigate config attributes
        value = resolver._config
        for part in k.split('.'):
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                raise KeyError(f"Config key not found: {k}")

        resolver._resolution_log.append({
            "type": "config",
            "key": k,
            "value": value,
        })

        if isinstance(value, (int, float)):
            return Constant(value)
        return Term(str(value))

    except Exception as e:
        resolver._resolution_log.append({
            "type": "config",
            "key": k,
            "error": str(e),
        })
        raise


@problog_export('+str', '-term')
def llm_query(question):
    """Query the LLM for world knowledge."""
    q = str(question).strip("'").strip('"')

    if resolver._llm is None:
        raise ValueError("LLM not registered")

    try:
        response = resolver._llm.generate(
            system="Answer factual questions concisely with just the value.",
            user_message=q,
            max_tokens=self.llm.max_output_tokens,
        )

        # Try to parse as number
        try:
            value = float(response.strip())
            if value == int(value):
                value = int(value)
        except ValueError:
            value = response.strip()

        resolver._resolution_log.append({
            "type": "llm",
            "question": q,
            "value": value,
        })

        if isinstance(value, (int, float)):
            return Constant(value)
        return Term(str(value))

    except Exception as e:
        resolver._resolution_log.append({
            "type": "llm",
            "question": q,
            "error": str(e),
        })
        raise


@problog_export('+str', '+str', '-term')
def sql_query(database, query):
    """Execute SQL query and return scalar result."""
    db = str(database).strip("'").strip('"')
    sql = str(query).strip("'").strip('"')

    if resolver._sql_executor is None:
        raise ValueError("SQL executor not registered")

    try:
        result = resolver._sql_executor(db, sql)

        # Extract scalar value
        if hasattr(result, 'iloc') and len(result) > 0:
            if len(result) == 1 and len(result.columns) == 1:
                value = result.iloc[0, 0]
            else:
                value = len(result)
        else:
            value = result

        # Convert numpy types to Python native types for ProbLog
        if hasattr(value, 'item'):
            value = value.item()
        elif hasattr(value, '__float__'):
            value = float(value)
        elif hasattr(value, '__int__'):
            value = int(value)

        # Log for proof building
        resolver._resolution_log.append({
            "type": "sql",
            "database": db,
            "query": sql,
            "value": value,
        })

        # Return as ProbLog term
        if isinstance(value, (int, float)):
            return Constant(value)
        return Term(str(value))

    except Exception as e:
        resolver._resolution_log.append({
            "type": "sql",
            "database": db,
            "query": sql,
            "error": str(e),
        })
        raise

@problog_export('+str', '+str', '-int')
def sql_query_table(database, query):
    """Execute SQL query and return row count."""
    db = str(database).strip("'").strip('"')
    sql = str(query).strip("'").strip('"')

    if resolver._sql_executor is None:
        raise ValueError("SQL executor not registered")

    try:
        result = resolver._sql_executor(db, sql)
        value = len(result) if hasattr(result, '__len__') else 0

        resolver._resolution_log.append({
            "type": "sql_table",
            "database": db,
            "query": sql,
            "value": value,
        })

        return value

    except Exception as e:
        resolver._resolution_log.append({
            "type": "sql_table",
            "database": db,
            "query": sql,
            "error": str(e),
        })
        raise

@problog_export('+str', '+str', '-term')
def nosql_query(database, query):
    """Execute NoSQL query and return result."""
    db = str(database).strip("'").strip('"')
    q = str(query).strip("'").strip('"')

    if resolver._nosql_executor is None:
        raise ValueError("NoSQL executor not registered")

    try:
        result = resolver._nosql_executor(db, q)
        value = len(result) if hasattr(result, '__len__') else result

        resolver._resolution_log.append({
            "type": "nosql",
            "database": db,
            "query": q,
            "value": value,
        })

        if isinstance(value, (int, float)):
            return Constant(value)
        return Term(str(value))

    except Exception as e:
        resolver._resolution_log.append({
            "type": "nosql",
            "database": db,
            "query": q,
            "error": str(e),
        })
        raise

@problog_export('+str', '-term')
def doc_search(query):
    """Search documents and return result."""
    q = str(query).strip("'").strip('"')

    if resolver._doc_searcher is None:
        raise ValueError("Document searcher not registered")

    try:
        result = resolver._doc_searcher(q)

        resolver._resolution_log.append({
            "type": "doc",
            "query": q,
            "value": result,
        })

        return Term(str(result))

    except Exception as e:
        resolver._resolution_log.append({
            "type": "doc",
            "query": q,
            "error": str(e),
        })
        raise
'''
        # Write to a temporary file
        fd, path = tempfile.mkstemp(suffix='.py', prefix='problog_exec_')
        os.write(fd, module_code.encode())
        os.close(fd)

        # Register cleanup
        atexit.register(lambda: os.path.exists(path) and os.unlink(path))

        self._module_path = path
        return path

    def _run_problog(
        self,
        fact_name: str,
        rules: str,
        user_facts_prolog: str,
        module_path: str,
    ) -> Optional[ResolvedFact]:
        """
        Run ProbLog with the generated rules.

        ProbLog will call out to Python via the executor module as needed.
        """
        from problog.program import PrologString
        from problog import get_evaluatable

        # Build complete program
        program = f"""
:- use_module('{module_path}').

% User-provided facts
{user_facts_prolog}

% Resolution rules
{rules}

% Query
query({fact_name}(X)).
"""

        logger.debug(f"ProbLog program:\n{program}")

        try:
            p = PrologString(program)
            result = get_evaluatable().create_from(p).evaluate()

            if result:
                # Get the first result
                for query_term, prob in result.items():
                    value = query_term.args[0] if query_term.args else None

                    # Convert ProbLog value to Python
                    if hasattr(value, 'functor'):
                        value = value.functor
                    elif hasattr(value, 'value'):
                        value = value.value

                    # Build proof tree from resolution log
                    proof = self._build_proof_tree(fact_name, value)

                    return ResolvedFact(
                        name=fact_name,
                        value=value,
                        probability=float(prob),
                        source=FactSource.DERIVED,
                        proof=proof,
                    )

        except Exception as e:
            logger.error(f"ProbLog reasoning error: {e}")
            import traceback
            traceback.print_exc()

        return None

    def _build_proof_tree(self, fact_name: str, value: Any) -> ProofNode:
        """Build proof tree from resolution log."""
        global _resolution_log

        children = []

        # Add children from resolution log
        for entry in _resolution_log:
            if "error" in entry:
                continue

            source = FactSource.DATABASE
            if entry["type"] == "doc":
                source = FactSource.DOCUMENT

            evidence = entry.get("query", "")
            child = ProofNode(
                predicate=f"{entry['type']}_result",
                value=entry.get("value"),
                source=source,
                probability=1.0,
                evidence=evidence,
            )
            children.append(child)

            # Also track in ground facts for reference
            self._ground_facts[f"{entry['type']}_result"] = {
                "value": entry.get("value"),
                "source": source,
                "query": evidence,
            }

        # Add user-provided facts as children
        for name, data in self._ground_facts.items():
            if data["source"] == FactSource.USER_PROVIDED:
                child = ProofNode(
                    predicate=name,
                    value=data["value"],
                    source=FactSource.USER_PROVIDED,
                    probability=1.0,
                )
                children.append(child)

        return ProofNode(
            predicate=fact_name,
            value=value,
            source=FactSource.DERIVED,
            probability=1.0 if value is not None else 0.0,
            children=children,
        )

    def get_ground_facts(self) -> Dict[str, Dict]:
        """Get all resolved ground facts."""
        return dict(self._ground_facts)
