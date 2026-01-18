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
from typing import Any, Callable, Optional


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
                    f"Cycle detected in DAG. Stuck nodes: {stuck_nodes}"
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
) -> ExecutionDAG:
    """Parse auditable mode plan into an executable DAG.

    Args:
        premises: List of premise dicts with keys: id, name, description, source
        inferences: List of inference dicts with keys: id, name, operation, explanation

    Returns:
        ExecutionDAG ready for parallel execution
    """
    nodes = []
    name_to_id: dict[str, str] = {}  # Map names to P/I IDs for dependency resolution

    # Parse premises as leaf nodes
    for p in premises:
        fact_id = p.get("id", "")  # P1, P2, etc.
        name = p.get("name", fact_id)
        source = p.get("source", "database")

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
            try:
                value_str = parts[1].strip()
                if "." in value_str:
                    embedded_value = float(value_str)
                else:
                    embedded_value = int(value_str)
            except ValueError:
                embedded_value = parts[1].strip().strip("'\"")

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


def extract_dependencies(operation: str, known_names: dict[str, str]) -> list[str]:
    """Extract fact names referenced in an operation.

    Args:
        operation: Operation string like "join(employees, reviews)"
        known_names: Dict mapping IDs/names to canonical names

    Returns:
        List of dependency names (canonical names, not IDs)

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
    ):
        """Initialize the DAG executor.

        Args:
            dag: The DAG to execute
            node_executor: Function that executes a single node, returns (value, confidence)
            max_workers: Maximum parallel workers
            event_callback: Optional callback for progress events
            fail_fast: If True, stop on first failure
        """
        self.dag = dag
        self.node_executor = node_executor
        self.max_workers = max_workers
        self.event_callback = event_callback
        self.fail_fast = fail_fast

    def execute(self) -> ExecutionResult:
        """Execute the DAG with parallel node resolution.

        Returns:
            ExecutionResult with all resolved nodes
        """
        import time
        start_time = time.time()

        execution_levels = self.dag.get_execution_order()
        failed_nodes = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for level_idx, level in enumerate(execution_levels):
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
                    node = futures[future]
                    try:
                        value, confidence = future.result()
                        node.status = NodeStatus.RESOLVED
                        node.value = value
                        node.confidence = confidence

                        if self.event_callback:
                            self.event_callback("node_resolved", {
                                "name": node.name,
                                "fact_id": node.fact_id,
                                "value": value,
                                "confidence": confidence,
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

        total_duration = int((time.time() - start_time) * 1000)

        return ExecutionResult(
            success=len(failed_nodes) == 0,
            nodes=self.dag.nodes,
            execution_levels=execution_levels,
            failed_nodes=failed_nodes,
            total_duration_ms=total_duration,
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
