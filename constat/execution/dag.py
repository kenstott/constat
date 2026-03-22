# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Directed Acyclic Graph for parallel fact resolution in auditable mode.

This module provides DAG data structures and parallel execution for auditable mode.
All facts are nodes in a single DAG:
- Leaf nodes: No dependencies, resolved from sources (databases, documents, cache)
- Internal nodes: Have dependencies, computed from other nodes

Execution is level-based:
- Level 0: All leaf nodes (run in parallel)
- Level 1: Nodes depending only on Level 0 (run in parallel)
- Level N: Nodes depending on Level 0..N-1 (run in parallel)
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

from constat.execution.fact_resolver import format_source_attribution

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from constat.execution.parallel_scheduler import ExecutionContext


class NodeStatus(Enum):
    """Execution status of a DAG node."""
    PENDING = "pending"
    RUNNING = "running"
    RESOLVED = "resolved"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class FactNode:
    """A node in the execution DAG.

    Represents either a source fact (leaf node) or a computed fact (internal node).
    """
    name: str                              # e.g., "employees", "joined"
    description: str                       # Human-readable description
    source: Optional[str] = None           # "database", "document", "knowledge", None (computed)
    source_db: Optional[str] = None        # Database name if source is "database"
    operation: Optional[str] = None        # e.g., "join(employees, reviews)"
    dependencies: list[str] = field(default_factory=list)  # Names of facts this depends on

    # Original plan identifiers (P1, P2, I1, I2, etc.)
    fact_id: str = ""                      # Original ID from plan (P1, I2, etc.)

    # Execution state
    status: NodeStatus = NodeStatus.PENDING
    value: Any = None
    error: Optional[str] = None
    confidence: float = 1.0

    # Execution metadata
    sql_query: Optional[str] = None        # SQL query used if resolved via database
    code: Optional[str] = None             # Python code used if computed
    row_count: Optional[int] = None        # Row count if result is a table

    @property
    def is_leaf(self) -> bool:
        """Leaf nodes have no dependencies (source facts)."""
        return len(self.dependencies) == 0

    @property
    def is_table(self) -> bool:
        """True if this node represents tabular data."""
        return self.row_count is not None and self.row_count > 1

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if isinstance(other, FactNode):
            return self.name == other.name
        return False


class ExecutionDAG:
    """DAG for parallel fact resolution.

    Manages a directed acyclic graph of fact nodes with:
    - Dependency tracking
    - Cycle detection
    - Level-based execution ordering
    - Ready node detection for parallel execution
    """

    def __init__(self, nodes: Optional[list[FactNode]] = None):
        """Initialize the DAG with optional list of nodes.

        Args:
            nodes: List of FactNode objects
        """
        self.nodes: dict[str, FactNode] = {}
        if nodes:
            for node in nodes:
                self.add_node(node)
            self._validate_dag()

    def add_node(self, node: FactNode) -> None:
        """Add a node to the DAG."""
        self.nodes[node.name] = node

    def get_node(self, name: str) -> Optional[FactNode]:
        """Get a node by name."""
        return self.nodes.get(name)

    def _validate_dag(self) -> None:
        """Ensure no cycles and all dependencies exist.

        Raises:
            ValueError: If a cycle is detected or dependencies are missing
        """
        # Check all dependencies exist
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep not in self.nodes:
                    raise ValueError(
                        f"Node '{node.name}' depends on '{dep}' which doesn't exist"
                    )

        # Topological sort will fail if cycles exist
        try:
            self.get_execution_order()
        except ValueError as e:
            raise ValueError(f"Invalid DAG: {e}")

    def get_ready_nodes(self) -> list[FactNode]:
        """Get all nodes whose dependencies are resolved.

        Returns:
            List of nodes that are PENDING and have all dependencies RESOLVED
        """
        ready = []
        for node in self.nodes.values():
            if node.status != NodeStatus.PENDING:
                continue
            deps_resolved = all(
                self.nodes[dep].status == NodeStatus.RESOLVED
                for dep in node.dependencies
            )
            if deps_resolved:
                ready.append(node)
        return ready

    def get_execution_order(self) -> list[list[str]]:
        """Get execution levels (nodes at same level can run in parallel).

        Returns:
            List of lists, where each inner list contains node names
            that can execute in parallel

        Raises:
            ValueError: If a cycle is detected
        """
        levels: list[list[str]] = []
        remaining = set(self.nodes.keys())
        resolved = set()

        while remaining:
            # Find all nodes whose deps are resolved
            level = [
                name for name in remaining
                if all(dep in resolved for dep in self.nodes[name].dependencies)
            ]
            if not level:
                # Find the problematic nodes for better error message
                stuck_nodes = list(remaining)[:3]
                raise ValueError(
                    f"Circular dependency detected in plan. Affected facts: {stuck_nodes}"
                )
            levels.append(sorted(level))  # Sort for deterministic ordering
            resolved.update(level)
            remaining -= set(level)

        return levels

    def get_leaf_nodes(self) -> list[FactNode]:
        """Get all leaf nodes (no dependencies)."""
        return [n for n in self.nodes.values() if n.is_leaf]

    def get_internal_nodes(self) -> list[FactNode]:
        """Get all internal nodes (have dependencies)."""
        return [n for n in self.nodes.values() if not n.is_leaf]

    def mark_resolved(self, name: str, value: Any, confidence: float = 1.0) -> None:
        """Mark a node as resolved with its value."""
        if name in self.nodes:
            node = self.nodes[name]
            node.status = NodeStatus.RESOLVED
            node.value = value
            node.confidence = confidence

    def mark_failed(self, name: str, error: str) -> None:
        """Mark a node as failed with error message."""
        if name in self.nodes:
            node = self.nodes[name]
            node.status = NodeStatus.FAILED
            node.error = error

    def all_resolved(self) -> bool:
        """Check if all nodes are resolved."""
        return all(n.status == NodeStatus.RESOLVED for n in self.nodes.values())

    def is_terminal(self) -> bool:
        """Check if all nodes are in a terminal state (resolved, failed, or blocked)."""
        terminal = {NodeStatus.RESOLVED, NodeStatus.FAILED, NodeStatus.BLOCKED}
        return all(n.status in terminal for n in self.nodes.values())

    def get_failed_nodes(self) -> list[FactNode]:
        """Get all failed nodes."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.FAILED]

    def get_blocked_nodes(self) -> list[FactNode]:
        """Get all blocked nodes."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.BLOCKED]

    def get_transitive_dependents(self, name: str) -> list[str]:
        """Get all nodes that transitively depend on the given node (BFS)."""
        # Build reverse adjacency: node -> list of nodes that depend on it
        dependents: dict[str, list[str]] = {n: [] for n in self.nodes}
        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep in dependents:
                    dependents[dep].append(node.name)

        visited: set[str] = set()
        queue = list(dependents.get(name, []))
        result: list[str] = []
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            result.append(current)
            queue.extend(dependents.get(current, []))
        return result

    def mark_blocked(self, name: str, blocked_by: str = "") -> None:
        """Mark a node as blocked."""
        if name in self.nodes:
            node = self.nodes[name]
            if node.status == NodeStatus.PENDING:
                node.status = NodeStatus.BLOCKED
                node.error = f"blocked by {blocked_by}" if blocked_by else "dependency failed"


def parse_plan_to_dag(
    premises: list[dict],
    inferences: list[dict],
    validate: bool = True,
    known_databases: set[str] | None = None,
    known_documents: set[str] | None = None,
    known_apis: set[str] | None = None,
) -> ExecutionDAG:
    """Parse auditable mode plan into an executable DAG.

    Args:
        premises: List of premise dicts with keys: id, name, description, source
        inferences: List of inference dicts with keys: id, name, operation, explanation
        validate: If True, validate plan before parsing (recommended)

    Returns:
        ExecutionDAG ready for parallel execution

    Raises:
        ValueError: If validation fails or duplicate names are detected
    """
    # Validate plan first if requested
    if validate:
        validation = validate_proof_plan(premises, inferences)
        if not validation.valid:
            raise ValueError(validation.format_for_retry())
    nodes = []
    name_to_id: dict[str, str] = {}  # Map names to P/I IDs for dependency resolution

    # Track names for duplicate detection
    premise_names: dict[str, str] = {}  # name -> fact_id for error messages
    inference_names: dict[str, str] = {}  # name -> fact_id for error messages

    # Parse premises as leaf nodes
    for p in premises:
        fact_id = p.get("id", "")  # P1, P2, etc.
        name = p.get("name", fact_id)
        source = p.get("source")

        if not source:
            raise ValueError(
                f"Premise {fact_id} ('{name}') is missing required 'source' field. "
                f"Must specify source: database:<db_name>, document, or knowledge"
            )

        # Parse source type and database name
        source_type = source
        source_db = None
        if ":" in source:
            parts = source.split(":", 1)
            source_type = parts[0]
            source_db = parts[1].strip()

        # Validate source references against known configured sources
        if source_type == "database" and known_databases is not None and source_db:
            if source_db not in known_databases:
                raise ValueError(
                    f"Premise {fact_id} references unknown database '{source_db}'. "
                    f"Available databases: {', '.join(sorted(known_databases))}"
                )
        if source_type == "document" and known_documents is not None and source_db:
            if source_db not in known_documents:
                raise ValueError(
                    f"Premise {fact_id} references unknown document '{source_db}'. "
                    f"Available documents: {', '.join(sorted(known_documents))}"
                )
        if source_type == "api" and known_apis is not None and source_db:
            if source_db not in known_apis:
                raise ValueError(
                    f"Premise {fact_id} references unknown API '{source_db}'. "
                    f"Available APIs: {', '.join(sorted(known_apis))}"
                )

        # Check for embedded value in name (e.g., "pi_value = 3.14159")
        embedded_value = None
        clean_name = name
        if " = " in name and not name.endswith(" = ?"):
            parts = name.rsplit(" = ", 1)
            clean_name = parts[0].strip()
            value_str = parts[1].strip()
            # Skip if value equals name (malformed plan, not actual embedded value)
            if value_str != clean_name:
                try:
                    if "." in value_str:
                        embedded_value = float(value_str)
                    else:
                        embedded_value = int(value_str)
                except ValueError:
                    embedded_value = value_str.strip("'\"")

        # Check for duplicate premise names
        if clean_name in premise_names:
            existing_id = premise_names[clean_name]
            raise ValueError(
                f"Duplicate premise name '{clean_name}': defined in both {existing_id} and {fact_id}. "
                f"Each premise must have a unique name."
            )
        premise_names[clean_name] = fact_id

        node = FactNode(
            name=clean_name,
            fact_id=fact_id,
            description=p.get("description", ""),
            source=source_type,
            source_db=source_db,
            dependencies=[],  # Premises have no dependencies
        )

        # If embedded value, pre-resolve the node
        if embedded_value is not None:
            node.status = NodeStatus.RESOLVED
            node.value = embedded_value
            node.confidence = 0.95

        nodes.append(node)
        name_to_id[fact_id] = clean_name  # Map P1 -> name for dependency resolution
        name_to_id[clean_name] = clean_name  # Also map name -> name

    # Parse inferences as internal nodes
    for inf in inferences:
        fact_id = inf.get("id", "")  # I1, I2, etc.
        name = inf.get("name", "") or fact_id
        operation = inf.get("operation", "")

        # Check for duplicate inference names
        if name in inference_names:
            existing_id = inference_names[name]
            raise ValueError(
                f"Duplicate inference name '{name}': defined in both {existing_id} and {fact_id}. "
                f"Each inference must have a unique result_name. "
                f"Consider using distinct names like '{name}_intermediate' and '{name}_final'."
            )
        # Check if inference name conflicts with a premise name
        if name in premise_names:
            conflicting_id = premise_names[name]
            raise ValueError(
                f"Inference name '{name}' in {fact_id} conflicts with premise {conflicting_id}. "
                f"Inference result names must be distinct from premise names."
            )
        inference_names[name] = fact_id

        # Extract dependencies from operation
        # Look for P1, P2, I1, I2, etc. and also named references
        dependencies = extract_dependencies(operation, name_to_id)

        node = FactNode(
            name=name,
            fact_id=fact_id,
            description=inf.get("explanation", ""),
            operation=operation,
            dependencies=dependencies,
        )
        nodes.append(node)
        name_to_id[fact_id] = name  # Map I1 -> name
        name_to_id[name] = name

    return ExecutionDAG(nodes)


@dataclass
class PlanValidationError:
    """A single validation error in a proof plan."""
    fact_id: str  # The fact ID where error occurred (e.g., "I2")
    error_type: str  # "unknown_reference", "unused_premise", "circular_dependency", etc.
    message: str  # Human-readable error message
    invalid_refs: list[str] = field(default_factory=list)  # The invalid references found


@dataclass
class PlanValidationResult:
    """Result of validating a proof plan."""
    valid: bool
    errors: list[PlanValidationError] = field(default_factory=list)

    def format_for_retry(self) -> str:
        """Format errors as feedback for LLM retry."""
        if self.valid:
            return ""
        lines = ["PLAN VALIDATION FAILED - Please fix the following errors:\n"]
        for err in self.errors:
            lines.append(f"  [{err.fact_id}] {err.error_type}: {err.message}")
            if err.invalid_refs:
                lines.append(f"      Invalid references: {', '.join(err.invalid_refs)}")
        lines.append("\nEnsure all fact references (P1, P2, I1, etc.) exist in the plan.")
        return "\n".join(lines)


def validate_proof_plan(
    premises: list[dict],
    inferences: list[dict],
    known_databases: set[str] | None = None,
    known_documents: set[str] | None = None,
    known_apis: set[str] | None = None,
) -> PlanValidationResult:
    """Validate a proof plan for internal consistency before execution.

    Checks:
    1. All P/I references in operations exist in the plan
    2. All premises are used by at least one inference
    3. Inference dependencies form a valid DAG (no forward references)
    4. No duplicate names
    5. Source references match configured databases/documents/APIs

    Args:
        premises: List of premise dicts with keys: id, name, description, source
        inferences: List of inference dicts with keys: id, name, operation, explanation
        known_databases: Set of configured database names (None = skip check)
        known_documents: Set of configured document names (None = skip check)
        known_apis: Set of configured API names (None = skip check)

    Returns:
        PlanValidationResult with valid=True if plan is consistent, or errors if not
    """
    errors: list[PlanValidationError] = []

    # Build set of valid fact IDs and names
    valid_ids: set[str] = set()
    valid_names: set[str] = set()
    id_to_name: dict[str, str] = {}

    # Track premise usage
    premise_ids: set[str] = set()
    used_premises: set[str] = set()

    # Parse premises
    premise_name_to_id: dict[str, str] = {}
    for p in premises:
        fact_id = p.get("id", "")
        name = p.get("name", fact_id)

        # Clean embedded values from name
        clean_name = name
        if " = " in name and not name.endswith(" = ?"):
            clean_name = name.rsplit(" = ", 1)[0].strip()

        # Check for duplicate premise names
        if clean_name in premise_name_to_id:
            errors.append(PlanValidationError(
                fact_id=fact_id,
                error_type="duplicate_premise",
                message=f"Premise name '{clean_name}' already defined in {premise_name_to_id[clean_name]}",
            ))
        else:
            premise_name_to_id[clean_name] = fact_id

        # Validate source references against known configured sources
        source = p.get("source", "")
        if source == "cache":
            # Cache is not a valid planning source — it's an internal resolution optimization.
            # Premises must reference their actual data source.
            errors.append(PlanValidationError(
                fact_id=fact_id,
                error_type="invalid_source",
                message=(
                    f"'cache' is not a valid source. Use the actual data source "
                    f"(database:<name>, document:<name>, or api:<name>). "
                    f"Caching is handled automatically."
                ),
            ))
        elif ":" in source:
            src_type, src_name = source.split(":", 1)
            src_name = src_name.strip()
            if src_type == "database" and known_databases is not None and src_name not in known_databases:
                errors.append(PlanValidationError(
                    fact_id=fact_id,
                    error_type="unknown_source",
                    message=f"Unknown database '{src_name}'. Available: {', '.join(sorted(known_databases))}",
                ))
            elif src_type == "document" and known_documents is not None and src_name not in known_documents:
                errors.append(PlanValidationError(
                    fact_id=fact_id,
                    error_type="unknown_source",
                    message=f"Unknown document '{src_name}'. Available: {', '.join(sorted(known_documents))}",
                ))
            elif src_type == "api" and known_apis is not None and src_name not in known_apis:
                errors.append(PlanValidationError(
                    fact_id=fact_id,
                    error_type="unknown_source",
                    message=f"Unknown API '{src_name}'. Available: {', '.join(sorted(known_apis))}",
                ))

        valid_ids.add(fact_id)
        valid_names.add(clean_name)
        id_to_name[fact_id] = clean_name
        premise_ids.add(fact_id)

    # Parse inferences and validate references
    inference_name_to_id: dict[str, str] = {}
    defined_inferences: set[str] = set()  # Track which inferences are defined (for forward ref check)

    for inf in inferences:
        fact_id = inf.get("id", "")
        name = inf.get("name", "") or fact_id
        operation = inf.get("operation", "")

        # Check for duplicate inference names
        if name in inference_name_to_id:
            errors.append(PlanValidationError(
                fact_id=fact_id,
                error_type="duplicate_inference",
                message=f"Inference name '{name}' already defined in {inference_name_to_id[name]}",
            ))
        elif name in premise_name_to_id:
            errors.append(PlanValidationError(
                fact_id=fact_id,
                error_type="name_conflict",
                message=f"Inference name '{name}' conflicts with premise {premise_name_to_id[name]}",
            ))
        else:
            inference_name_to_id[name] = fact_id

        # Find all P/I references in operation
        pi_refs = re.findall(r'\b([PI]\d+)\b', operation)
        invalid_refs = []
        forward_refs = []

        for ref in pi_refs:
            if ref not in valid_ids and ref not in defined_inferences:
                invalid_refs.append(ref)
            elif ref.startswith("I") and ref not in defined_inferences:
                # Forward reference to inference not yet defined
                forward_refs.append(ref)
            elif ref.startswith("P"):
                used_premises.add(ref)

        if invalid_refs:
            errors.append(PlanValidationError(
                fact_id=fact_id,
                error_type="unknown_reference",
                message=f"Operation '{operation}' references unknown facts",
                invalid_refs=invalid_refs,
            ))

        if forward_refs:
            errors.append(PlanValidationError(
                fact_id=fact_id,
                error_type="forward_reference",
                message=f"Operation '{operation}' references inferences not yet defined",
                invalid_refs=forward_refs,
            ))

        # Mark inference and its name as defined for subsequent inferences
        valid_ids.add(fact_id)
        valid_names.add(name)
        id_to_name[fact_id] = name
        defined_inferences.add(fact_id)

    # Check for unused premises — warn but don't block (constraints like
    # "last 12 months" or "no budget constraints" may not feed into inferences
    # directly but still constrain the generated code)
    unused_premises = premise_ids - used_premises
    if unused_premises:
        unused_list = sorted(unused_premises)
        unused_names = [f"{pid} ({id_to_name.get(pid, '?')})" for pid in unused_list]
        import logging as _logging
        _logging.getLogger(__name__).info(
            f"[PLAN_VALIDATION] Unused premises (kept as constraints): {', '.join(unused_names)}"
        )

    return PlanValidationResult(
        valid=len(errors) == 0,
        errors=errors,
    )


def deduplicate_inferences(
    premises: list[dict],
    inferences: list[dict],
    router,
) -> list[dict]:
    """LLM review pass: fix miswired inferences in the DAG.

    A later inference that re-does work from scratch (e.g. re-classifying raw
    data) when a prior inference already produced that result is a wiring
    error — it should consume the prior output, not redo the computation.

    This function asks a fast model to identify two symptoms of the same
    underlying problem:
    - **rewire**: an inference references the wrong input (e.g. raw premise
      instead of a prior inference's output). Fix: rewrite the reference.
    - **remove**: an inference is fully redundant because a prior inference
      already produced equivalent output. Fix: drop it and point downstream
      consumers at the earlier inference.

    Args:
        premises: List of premise dicts
        inferences: List of inference dicts
        router: LLM router instance

    Returns:
        Cleaned inference list (unchanged if no issues found)
    """
    from constat.core.models import TaskType

    # Format compact DAG representation
    dag_lines = ["PREMISES:"]
    for p in premises:
        source = p.get("source", "database")
        dag_lines.append(f"  {p['id']}: {p.get('name', '')} ({p.get('description', '')}) [source: {source}]")
    dag_lines.append("INFERENCES:")
    for inf in inferences:
        dag_lines.append(f"  {inf['id']}: {inf.get('name', '')} = {inf.get('operation', '')} -- {inf.get('explanation', '')}")
    dag_text = "\n".join(dag_lines)

    system = (
        "You review data derivation plans for wiring correctness. "
        "Return ONLY valid JSON, no markdown fences."
    )
    prompt = f"""{dag_text}

Each inference should consume the output of prior steps, not redo their work.
Review the wiring and identify:

1. REWIRE: An inference that references a raw input (premise or earlier inference) when a later inference already transformed that data and this inference should consume the transformed result instead.
2. REMOVE: An inference that is fully redundant — a prior inference already produces equivalent output. Downstream consumers should point at the prior inference instead.

Return JSON:
{{"rewire": [{{"inference": "I4", "old_ref": "P1", "new_ref": "I2", "reason": "..."}}], "remove": [{{"drop": "I3", "use_instead": "I2", "reason": "..."}}]}}

Return {{"rewire": [], "remove": []}} if the wiring is correct.
IMPORTANT: Only flag clear-cut issues. Do not flag inferences that intentionally build on prior results in a new way.
A reference IS a dependency. If I4 references I3, then I4 NEEDS I3's output. NEVER rewire I4 away from I3 unless I3 is truly redundant with the replacement.
NEVER rewire away from an inference that ENRICHES data with new columns (e.g., llm_score, llm_classify, llm_map, llm_extract, join, enrich). An enrichment step adds data that downstream steps need — skipping it loses those columns."""

    logger.info("DAG review: checking %d inferences for wiring issues", len(inferences))

    try:
        result = router.execute(
            task_type=TaskType.INTENT_CLASSIFICATION,
            system=system,
            user_message=prompt,
            max_tokens=1024,
        )
    except Exception:
        logger.warning("DAG review LLM call failed, skipping")
        return inferences

    # Parse response
    content = result.content.strip()
    logger.info("DAG review raw response: %s", content[:500])
    # Extract JSON from markdown fences or surrounding text
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    elif content.startswith("```"):
        content = re.sub(r'^```(?:json)?\s*', '', content)
        content = re.sub(r'\s*```.*', '', content, flags=re.DOTALL)

    try:
        review = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("DAG review: could not parse LLM response, skipping")
        return inferences

    rewires = review.get("rewire", [])
    removals = review.get("remove", [])

    if not rewires and not removals:
        return inferences

    # Build ID sets for validation
    valid_ids = {p["id"] for p in premises} | {inf["id"] for inf in inferences}
    inferences_by_id = {inf["id"]: inf for inf in inferences}

    # Build reverse dependency map: ref_id -> set of inference IDs that reference it
    def _build_ref_counts(inf_list):
        counts: dict[str, set[str]] = {inf["id"]: set() for inf in inf_list}
        for inf in inf_list:
            for ref in re.findall(r'\b([PI]\d+)\b', inf["operation"]):
                if ref in counts:
                    counts[ref].add(inf["id"])
        return counts

    ref_counts = _build_ref_counts(inferences)

    # Apply rewires first (before removing inferences)
    for fix in rewires:
        inf_id = fix.get("inference", "")
        old_ref = fix.get("old_ref", "")
        new_ref = fix.get("new_ref", "")
        if inf_id not in inferences_by_id or old_ref not in valid_ids or new_ref not in valid_ids:
            continue
        # Reject if this rewire would orphan old_ref (make it referenced by nothing)
        remaining_refs = ref_counts.get(old_ref, set()) - {inf_id}
        if not remaining_refs and old_ref in inferences_by_id:
            logger.info(
                "DAG review: skipping rewire %s: %s -> %s (would orphan %s)",
                inf_id, old_ref, new_ref, old_ref,
            )
            continue
        inf = inferences_by_id[inf_id]
        old_op = inf["operation"]
        new_op = re.sub(rf'\b{re.escape(old_ref)}\b', new_ref, old_op)
        if new_op != old_op:
            inf["operation"] = new_op
            ref_counts.get(old_ref, set()).discard(inf_id)
            ref_counts.setdefault(new_ref, set()).add(inf_id)
            logger.info("DAG review: rewired %s: %s -> %s (%s)", inf_id, old_ref, new_ref, fix.get("reason", ""))

    # Apply removals: drop redundant inferences and rewire downstream refs
    removed_ids: dict[str, str] = {}  # removed_id -> use_instead_id
    for rem in removals:
        drop_id = rem.get("drop", "")
        use_instead = rem.get("use_instead", "")
        if drop_id not in inferences_by_id or use_instead not in valid_ids:
            continue
        if drop_id == use_instead:
            continue
        removed_ids[drop_id] = use_instead
        logger.info("DAG review: dropping %s (use %s instead) — %s", drop_id, use_instead, rem.get("reason", ""))

    if not removed_ids:
        return inferences

    # Remove redundant inferences
    cleaned = [inf for inf in inferences if inf["id"] not in removed_ids]

    # Rewire downstream references
    for inf in cleaned:
        op = inf["operation"]
        for removed_id, replacement_id in removed_ids.items():
            op = re.sub(rf'\b{re.escape(removed_id)}\b', replacement_id, op)
        inf["operation"] = op

    # Renumber sequentially
    for idx, inf in enumerate(cleaned, start=1):
        old_id = inf["id"]
        new_id = f"I{idx}"
        if old_id != new_id:
            # Update references in all subsequent inferences
            for other in cleaned[idx:]:
                other["operation"] = re.sub(rf'\b{re.escape(old_id)}\b', new_id, other["operation"])
            inf["id"] = new_id

    return cleaned


def extract_dependencies(operation: str, known_names: dict[str, str]) -> list[str]:
    """Extract fact names referenced in an operation.

    Args:
        operation: Operation string like "join(employees, reviews)"
        known_names: Dict mapping IDs/names to canonical names

    Returns:
        List of dependency names (canonical names, not IDs)

    Note: This function assumes the plan has already been validated.
    Unknown references are silently skipped (validation catches these).

    Examples:
        "join(P1, P2)" with known_names={"P1": "employees", "P2": "reviews"}
            -> ["employees", "reviews"]
        "filter(I1, has_review)" with known_names={"I1": "joined"}
            -> ["joined"]
    """
    dependencies = []

    # Find all P/I references (P1, P2, I1, I2, etc.)
    pi_refs = re.findall(r'\b([PI]\d+)\b', operation)
    for ref in pi_refs:
        if ref in known_names:
            dep_name = known_names[ref]
            if dep_name not in dependencies:
                dependencies.append(dep_name)

    # Also look for known named references (e.g., "filter(employees, ...)")
    # Split by common delimiters and check each token
    tokens = re.split(r'[(),\s]+', operation)
    for token in tokens:
        token = token.strip()
        if token and token in known_names:
            dep_name = known_names[token]
            if dep_name not in dependencies:
                dependencies.append(dep_name)

    return dependencies


@dataclass
class ExecutionResult:
    """Result of DAG execution."""
    success: bool
    nodes: dict[str, FactNode]
    execution_levels: list[list[str]]
    failed_nodes: list[str] = field(default_factory=list)
    total_duration_ms: int = 0
    cancelled: bool = False


class DAGExecutor:
    """Execute a DAG with maximum parallelism.

    Executes nodes level by level:
    - All nodes in a level run in parallel (up to max_workers)
    - Each level waits for completion before starting the next
    """

    def __init__(
        self,
        dag: ExecutionDAG,
        node_executor: Callable[[FactNode], tuple[Any, ...]],
        max_workers: int = 10,
        event_callback: Optional[Callable] = None,
        fail_fast: bool = True,
        execution_context: Optional["ExecutionContext"] = None,
    ):
        """Initialize the DAG executor.

        Args:
            dag: The DAG to execute
            node_executor: Function that executes a single node, returns (value, confidence)
            max_workers: Maximum parallel workers
            event_callback: Optional callback for progress events
            fail_fast: If True, stop on first failure
            execution_context: Optional context for cancellation support
        """
        self.dag = dag
        self.node_executor = node_executor
        self.max_workers = max_workers
        self.event_callback = event_callback
        self.fail_fast = fail_fast
        self._execution_context = execution_context

    def execute(self) -> ExecutionResult:
        """Execute the DAG with parallel node resolution.

        Returns:
            ExecutionResult with all resolved nodes
        """
        import time
        start_time = time.time()

        execution_levels = self.dag.get_execution_order()
        failed_nodes = []
        cancelled = False

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for level_idx, level in enumerate(execution_levels):
                # Check for cancellation before each level
                if self._execution_context and self._execution_context.is_cancelled():
                    cancelled = True
                    if self.event_callback:
                        self.event_callback("execution_cancelled", {
                            "level": level_idx,
                            "message": "Execution cancelled by user",
                        })
                    break

                # Get nodes in this level that need execution
                nodes_to_run = []
                for name in level:
                    node = self.dag.get_node(name)
                    if not node:
                        continue
                    if node.status == NodeStatus.RESOLVED:
                        # Pre-resolved (embedded value) — emit event so UI shows it
                        if self.event_callback:
                            self.event_callback("node_resolved", {
                                "name": node.name,
                                "fact_id": node.fact_id,
                                "value": node.value,
                                "confidence": node.confidence,
                                "source": "embedded",
                            })
                    elif node.status == NodeStatus.PENDING:
                        nodes_to_run.append(node)

                if not nodes_to_run:
                    continue

                # Mark all as running
                for node in nodes_to_run:
                    node.status = NodeStatus.RUNNING
                    if self.event_callback:
                        self.event_callback("node_running", {
                            "name": node.name,
                            "fact_id": node.fact_id,
                            "level": level_idx,
                        })

                # Submit all nodes in this level
                futures = {
                    executor.submit(self._execute_node, node): node
                    for node in nodes_to_run
                }

                # Wait for all to complete
                for future in as_completed(futures):
                    # Check for cancellation while waiting
                    if self._execution_context and self._execution_context.is_cancelled():
                        cancelled = True
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        if self.event_callback:
                            self.event_callback("execution_cancelled", {
                                "level": level_idx,
                                "message": "Execution cancelled by user",
                            })
                        break

                    node = futures[future]
                    try:
                        result = future.result()
                        # Support tuples up to 7 elements: (v, conf, src, validations, profile, elapsed_ms, attempt)
                        validations = None
                        profile = None
                        elapsed_ms = None
                        attempt = None
                        if isinstance(result, tuple) and len(result) >= 7:
                            value, confidence, source, validations, profile, elapsed_ms, attempt = result[0], result[1], result[2], result[3], result[4], result[5], result[6]
                        elif isinstance(result, tuple) and len(result) >= 5:
                            value, confidence, source, validations, profile = result[0], result[1], result[2], result[3], result[4]
                        elif isinstance(result, tuple) and len(result) >= 4:
                            value, confidence, source, validations = result[0], result[1], result[2], result[3]
                        elif isinstance(result, tuple) and len(result) >= 3:
                            value, confidence, source = result[0], result[1], result[2]
                        elif isinstance(result, tuple) and len(result) == 2:
                            value, confidence = result
                            source = ""
                        else:
                            value, confidence, source = result, 0.9, ""

                        # Propagate confidence: cap by weakest dependency
                        if node.dependencies:
                            dep_confidences = []
                            for dep_name in node.dependencies:
                                dep_node = self.dag.get_node(dep_name)
                                if dep_node and dep_node.confidence is not None:
                                    dep_confidences.append(dep_node.confidence)
                            if dep_confidences:
                                confidence = min(confidence, min(dep_confidences))

                        node.status = NodeStatus.RESOLVED
                        node.value = value
                        node.confidence = confidence

                        if self.event_callback:
                            event_data = {
                                "name": node.name,
                                "fact_id": node.fact_id,
                                "value": value,
                                "confidence": confidence,
                                "source": source,
                            }
                            if validations:
                                event_data["validations"] = validations
                            if profile:
                                event_data["profile"] = profile
                            if elapsed_ms is not None:
                                event_data["elapsed_ms"] = elapsed_ms
                            if attempt is not None:
                                event_data["attempt"] = attempt
                            self.event_callback("node_resolved", event_data)
                    except Exception as e:
                        import traceback
                        logger.error(f"DAG node {node.fact_id} ({node.name}) failed with {type(e).__name__}: {e}\n{traceback.format_exc()}")
                        node.status = NodeStatus.FAILED
                        node.error = str(e)
                        failed_nodes.append(node.name)

                        if self.event_callback:
                            self.event_callback("node_failed", {
                                "name": node.name,
                                "fact_id": node.fact_id,
                                "error": str(e),
                            })

                        # Propagate blocked to all transitive dependents
                        for dep_name in self.dag.get_transitive_dependents(node.name):
                            self.dag.mark_blocked(dep_name, blocked_by=node.name)
                            dep_node = self.dag.get_node(dep_name)
                            if dep_node and dep_node.status == NodeStatus.BLOCKED and self.event_callback:
                                self.event_callback("node_blocked", {
                                    "name": dep_node.name,
                                    "fact_id": dep_node.fact_id,
                                    "blocked_by": node.name,
                                })

                if cancelled:
                    break

        total_duration = int((time.time() - start_time) * 1000)

        return ExecutionResult(
            success=len(failed_nodes) == 0 and not cancelled,
            nodes=self.dag.nodes,
            execution_levels=execution_levels,
            failed_nodes=failed_nodes,
            total_duration_ms=total_duration,
            cancelled=cancelled,
        )

    def _execute_node(self, node: FactNode) -> tuple[Any, ...]:
        """Execute a single node using the configured executor.

        Args:
            node: The node to execute

        Returns:
            Tuple of (value, confidence)
        """
        import time
        start_ms = int(time.time() * 1000)

        # Emit actual start event (this runs in the worker thread)
        if self.event_callback:
            self.event_callback("node_started", {
                "name": node.name,
                "fact_id": node.fact_id,
                "start_time_ms": start_ms,
            })

        try:
            result = self.node_executor(node)
            end_ms = int(time.time() * 1000)

            # Emit timing info with result
            if self.event_callback:
                self.event_callback("node_timing", {
                    "name": node.name,
                    "fact_id": node.fact_id,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": end_ms - start_ms,
                })

            return result
        except Exception:
            end_ms = int(time.time() * 1000)
            if self.event_callback:
                self.event_callback("node_timing", {
                    "name": node.name,
                    "fact_id": node.fact_id,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": end_ms - start_ms,
                    "failed": True,
                })
            raise


def dag_to_display_format(dag: ExecutionDAG) -> str:
    """Convert DAG to human-readable P/I format for display.

    Generates the traditional PREMISES/INFERENCES format
    for plan approval display.

    Args:
        dag: The execution DAG

    Returns:
        Formatted string with PREMISES and INFERENCES sections
    """
    premise_lines = []
    inference_lines = []

    # Map node names back to P/I IDs for display
    name_to_display_id: dict[str, str] = {}

    for level in dag.get_execution_order():
        for name in level:
            node = dag.nodes[name]
            if node.is_leaf:
                # Premise
                display_id = node.fact_id or f"P{len(premise_lines) + 1}"
                name_to_display_id[name] = display_id

                source_str = format_source_attribution(
                    node.source or "database", node.source_db
                )

                premise_lines.append(
                    f"{display_id}: {node.name} = ? ({node.description}) [source: {source_str}]"
                )
            else:
                # Inference
                display_id = node.fact_id or f"I{len(inference_lines) + 1}"
                name_to_display_id[name] = display_id

                # Replace dependency names with their display IDs in operation
                display_op = node.operation or ""
                for dep_name in node.dependencies:
                    dep_id = name_to_display_id.get(dep_name, dep_name)
                    # Replace the name with the ID in the operation
                    display_op = re.sub(
                        rf'\b{re.escape(dep_name)}\b',
                        dep_id,
                        display_op
                    )

                inference_lines.append(
                    f"{display_id}: {node.name} = {display_op} -- {node.description}"
                )

    sections = []
    if premise_lines:
        sections.append("PREMISES:")
        sections.extend(f"  {line}" for line in premise_lines)

    if inference_lines:
        sections.append("")
        sections.append("INFERENCES:")
        sections.extend(f"  {line}" for line in inference_lines)

    return "\n".join(sections)


def dag_to_proof_format(dag: ExecutionDAG, conclusion: str = "") -> str:
    """Convert completed DAG to proof format showing resolved values.

    Args:
        dag: The executed DAG with resolved nodes
        conclusion: The conclusion statement

    Returns:
        Formatted proof string showing premises, inferences, and conclusion
    """
    premise_lines = []
    inference_lines = []

    for level in dag.get_execution_order():
        for name in level:
            node = dag.nodes[name]
            status_icon = "✓" if node.status == NodeStatus.RESOLVED else "✗"

            if node.is_leaf:
                # Premise - show resolved value
                value_str = str(node.value)[:60] + "..." if node.value and len(str(node.value)) > 60 else str(node.value) if node.value else "?"
                source_str = f"[{format_source_attribution(node.source or 'database', node.source_db)}]"
                premise_lines.append(
                    f"{status_icon} {node.fact_id}: {node.name} = {value_str} {source_str}"
                )
            else:
                # Inference - show result
                value_str = str(node.value)[:60] + "..." if node.value and len(str(node.value)) > 60 else str(node.value) if node.value else "?"
                inference_lines.append(
                    f"{status_icon} {node.fact_id}: {node.name} = {value_str}"
                )

    sections = ["", "═══ PROOF ═══", ""]
    if premise_lines:
        sections.append("Premises:")
        sections.extend(f"  {line}" for line in premise_lines)

    if inference_lines:
        sections.append("")
        sections.append("Inferences:")
        sections.extend(f"  {line}" for line in inference_lines)

    if conclusion:
        sections.append("")
        sections.append(f"Conclusion: {conclusion}")

    sections.append("═════════════")
    return "\n".join(sections)
