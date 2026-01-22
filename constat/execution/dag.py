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

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from constat.execution.parallel_scheduler import ExecutionContext


class NodeStatus(Enum):
    """Execution status of a DAG node."""
    PENDING = "pending"
    RUNNING = "running"
    RESOLVED = "resolved"
    FAILED = "failed"


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

    def get_failed_nodes(self) -> list[FactNode]:
        """Get all failed nodes."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.FAILED]


def parse_plan_to_dag(
    premises: list[dict],
    inferences: list[dict],
    validate: bool = True,
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
) -> PlanValidationResult:
    """Validate a proof plan for internal consistency before execution.

    Checks:
    1. All P/I references in operations exist in the plan
    2. All premises are used by at least one inference
    3. Inference dependencies form a valid DAG (no forward references)
    4. No duplicate names

    Args:
        premises: List of premise dicts with keys: id, name, description, source
        inferences: List of inference dicts with keys: id, name, operation, explanation

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

    # Check for unused premises
    unused_premises = premise_ids - used_premises
    if unused_premises:
        unused_list = sorted(unused_premises)
        unused_names = [f"{pid} ({id_to_name.get(pid, '?')})" for pid in unused_list]
        errors.append(PlanValidationError(
            fact_id="PLAN",
            error_type="unused_premises",
            message=f"Premises not used in any inference: {', '.join(unused_names)}",
            invalid_refs=list(unused_premises),
        ))

    return PlanValidationResult(
        valid=len(errors) == 0,
        errors=errors,
    )


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
        node_executor: Callable[[FactNode], tuple[Any, float]],
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

                if self.fail_fast and failed_nodes:
                    break

                # Get nodes in this level that need execution
                nodes_to_run = []
                for name in level:
                    node = self.dag.get_node(name)
                    if node and node.status == NodeStatus.PENDING:
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
                        # Support both (value, confidence) and (value, confidence, source) returns
                        if isinstance(result, tuple) and len(result) >= 3:
                            value, confidence, source = result[0], result[1], result[2]
                        elif isinstance(result, tuple) and len(result) == 2:
                            value, confidence = result
                            source = ""
                        else:
                            value, confidence, source = result, 0.9, ""

                        node.status = NodeStatus.RESOLVED
                        node.value = value
                        node.confidence = confidence

                        if self.event_callback:
                            self.event_callback("node_resolved", {
                                "name": node.name,
                                "fact_id": node.fact_id,
                                "value": value,
                                "confidence": confidence,
                                "source": source,
                            })
                    except Exception as e:
                        node.status = NodeStatus.FAILED
                        node.error = str(e)
                        failed_nodes.append(node.name)

                        if self.event_callback:
                            self.event_callback("node_failed", {
                                "name": node.name,
                                "fact_id": node.fact_id,
                                "error": str(e),
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

    def _execute_node(self, node: FactNode) -> tuple[Any, float]:
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

                source_str = node.source or "database"
                if node.source_db:
                    source_str = f"{node.source}:{node.source_db}"

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
                source_str = f"[{node.source or 'database'}]" if node.source else ""
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
