"""Tests for DAG data structures and parallel execution."""

import pytest
import time
from concurrent.futures import ThreadPoolExecutor

from constat.execution.dag import (
    NodeStatus,
    FactNode,
    ExecutionDAG,
    parse_plan_to_dag,
    extract_dependencies,
    DAGExecutor,
    ExecutionResult,
    dag_to_display_format,
)


class TestFactNode:
    """Tests for FactNode dataclass."""

    def test_leaf_node_detection(self):
        """Leaf nodes have no dependencies."""
        leaf = FactNode(name="employees", description="All employees")
        assert leaf.is_leaf

        internal = FactNode(
            name="joined",
            description="Joined data",
            dependencies=["employees", "reviews"],
        )
        assert not internal.is_leaf

    def test_node_equality(self):
        """Nodes are equal if they have the same name."""
        node1 = FactNode(name="employees", description="Desc 1")
        node2 = FactNode(name="employees", description="Desc 2")
        node3 = FactNode(name="reviews", description="Desc 1")

        assert node1 == node2
        assert node1 != node3

    def test_node_hash(self):
        """Nodes can be used in sets/dicts."""
        node1 = FactNode(name="employees", description="Desc 1")
        node2 = FactNode(name="employees", description="Desc 2")

        node_set = {node1, node2}
        assert len(node_set) == 1  # Same name, same hash


class TestExecutionDAG:
    """Tests for ExecutionDAG class."""

    def test_empty_dag(self):
        """Empty DAG works correctly."""
        dag = ExecutionDAG()
        assert len(dag.nodes) == 0
        assert dag.get_execution_order() == []
        assert dag.all_resolved()

    def test_single_node(self):
        """DAG with single leaf node."""
        node = FactNode(name="employees", description="All employees", fact_id="P1")
        dag = ExecutionDAG([node])

        assert len(dag.nodes) == 1
        assert dag.get_execution_order() == [["employees"]]
        assert dag.get_leaf_nodes() == [node]
        assert dag.get_internal_nodes() == []

    def test_linear_chain(self):
        """DAG with linear dependencies: A -> B -> C."""
        a = FactNode(name="A", description="First")
        b = FactNode(name="B", description="Second", dependencies=["A"])
        c = FactNode(name="C", description="Third", dependencies=["B"])

        dag = ExecutionDAG([a, b, c])

        levels = dag.get_execution_order()
        assert levels == [["A"], ["B"], ["C"]]

    def test_parallel_nodes(self):
        """DAG with parallel independent nodes."""
        a = FactNode(name="A", description="First")
        b = FactNode(name="B", description="Second")
        c = FactNode(name="C", description="Third")

        dag = ExecutionDAG([a, b, c])

        levels = dag.get_execution_order()
        assert len(levels) == 1
        assert set(levels[0]) == {"A", "B", "C"}

    def test_diamond_dag(self):
        """DAG with diamond shape: A -> B,C -> D."""
        a = FactNode(name="A", description="Root")
        b = FactNode(name="B", description="Left", dependencies=["A"])
        c = FactNode(name="C", description="Right", dependencies=["A"])
        d = FactNode(name="D", description="Merge", dependencies=["B", "C"])

        dag = ExecutionDAG([a, b, c, d])

        levels = dag.get_execution_order()
        assert levels[0] == ["A"]
        assert set(levels[1]) == {"B", "C"}
        assert levels[2] == ["D"]

    def test_cycle_detection(self):
        """DAG rejects cycles."""
        a = FactNode(name="A", description="First", dependencies=["C"])
        b = FactNode(name="B", description="Second", dependencies=["A"])
        c = FactNode(name="C", description="Third", dependencies=["B"])

        with pytest.raises(ValueError, match="Circular dependency"):
            ExecutionDAG([a, b, c])

    def test_missing_dependency(self):
        """DAG rejects missing dependencies."""
        a = FactNode(name="A", description="First", dependencies=["X"])

        with pytest.raises(ValueError, match="doesn't exist"):
            ExecutionDAG([a])

    def test_ready_nodes(self):
        """get_ready_nodes returns nodes with resolved dependencies."""
        a = FactNode(name="A", description="First")
        b = FactNode(name="B", description="Second", dependencies=["A"])
        c = FactNode(name="C", description="Third")

        dag = ExecutionDAG([a, b, c])

        # Initially A and C are ready (no deps)
        ready = dag.get_ready_nodes()
        assert len(ready) == 2
        assert set(n.name for n in ready) == {"A", "C"}

        # Mark A as resolved
        dag.mark_resolved("A", "value_a")

        # Now B is also ready
        ready = dag.get_ready_nodes()
        assert len(ready) == 2
        assert set(n.name for n in ready) == {"B", "C"}

    def test_mark_resolved(self):
        """mark_resolved updates node status."""
        node = FactNode(name="A", description="First")
        dag = ExecutionDAG([node])

        dag.mark_resolved("A", "test_value", 0.9)

        assert dag.nodes["A"].status == NodeStatus.RESOLVED
        assert dag.nodes["A"].value == "test_value"
        assert dag.nodes["A"].confidence == 0.9

    def test_mark_failed(self):
        """mark_failed updates node status."""
        node = FactNode(name="A", description="First")
        dag = ExecutionDAG([node])

        dag.mark_failed("A", "Some error")

        assert dag.nodes["A"].status == NodeStatus.FAILED
        assert dag.nodes["A"].error == "Some error"

    def test_all_resolved(self):
        """all_resolved returns True when all nodes are resolved."""
        a = FactNode(name="A", description="First")
        b = FactNode(name="B", description="Second")

        dag = ExecutionDAG([a, b])

        assert not dag.all_resolved()

        dag.mark_resolved("A", "val_a")
        assert not dag.all_resolved()

        dag.mark_resolved("B", "val_b")
        assert dag.all_resolved()

    def test_get_failed_nodes(self):
        """get_failed_nodes returns failed nodes."""
        a = FactNode(name="A", description="First")
        b = FactNode(name="B", description="Second")

        dag = ExecutionDAG([a, b])

        dag.mark_failed("A", "Error")
        dag.mark_resolved("B", "value")

        failed = dag.get_failed_nodes()
        assert len(failed) == 1
        assert failed[0].name == "A"


class TestParsePlanToDAG:
    """Tests for parse_plan_to_dag function."""

    def test_premises_only(self):
        """Parse plan with only premises."""
        premises = [
            {"id": "P1", "name": "employees", "description": "All employees", "source": "database:sales_db"},
            {"id": "P2", "name": "reviews", "description": "Performance reviews", "source": "database"},
        ]

        dag = parse_plan_to_dag(premises, [])

        assert len(dag.nodes) == 2
        assert dag.nodes["employees"].is_leaf
        assert dag.nodes["employees"].source == "database"
        assert dag.nodes["employees"].source_db == "sales_db"
        assert dag.nodes["reviews"].source_db is None

    def test_premises_and_inferences(self):
        """Parse plan with premises and inferences."""
        premises = [
            {"id": "P1", "name": "employees", "description": "All employees", "source": "database"},
            {"id": "P2", "name": "reviews", "description": "Reviews", "source": "database"},
        ]
        inferences = [
            {"id": "I1", "name": "joined", "operation": "join(P1, P2)", "explanation": "Join data"},
            {"id": "I2", "name": "filtered", "operation": "filter(I1, active)", "explanation": "Filter"},
        ]

        dag = parse_plan_to_dag(premises, inferences)

        assert len(dag.nodes) == 4
        assert dag.nodes["joined"].dependencies == ["employees", "reviews"]
        assert dag.nodes["filtered"].dependencies == ["joined"]

    def test_embedded_value(self):
        """Parse premise with embedded value."""
        premises = [
            {"id": "P1", "name": "pi_value = 3.14159", "description": "Pi constant", "source": "knowledge"},
        ]

        dag = parse_plan_to_dag(premises, [])

        node = dag.nodes["pi_value"]
        assert node.status == NodeStatus.RESOLVED
        assert node.value == 3.14159
        assert node.confidence == 0.95

    def test_named_references_in_operation(self):
        """Operations can reference facts by name, not just ID."""
        premises = [
            {"id": "P1", "name": "employees", "description": "Employees", "source": "database"},
        ]
        inferences = [
            {"id": "I1", "name": "filtered", "operation": "filter(employees, active)", "explanation": "Filter"},
        ]

        dag = parse_plan_to_dag(premises, inferences)

        assert dag.nodes["filtered"].dependencies == ["employees"]

    def test_duplicate_premise_names_raises_error(self):
        """Duplicate premise names should raise ValueError with clear message."""
        premises = [
            {"id": "P1", "name": "employees", "description": "First employees", "source": "database"},
            {"id": "P2", "name": "employees", "description": "Duplicate employees", "source": "database"},
        ]

        with pytest.raises(ValueError) as exc_info:
            parse_plan_to_dag(premises, [])

        assert "Duplicate premise name 'employees'" in str(exc_info.value)
        assert "P1" in str(exc_info.value)
        assert "P2" in str(exc_info.value)

    def test_duplicate_inference_names_raises_error(self):
        """Duplicate inference names should raise ValueError with clear message."""
        premises = [
            {"id": "P1", "name": "employees", "description": "Employees", "source": "database"},
        ]
        inferences = [
            {"id": "I1", "name": "data_verified", "operation": "validate(P1)", "explanation": "First validation"},
            {"id": "I2", "name": "data_verified", "operation": "verify_exists(I1)", "explanation": "Second validation"},
        ]

        with pytest.raises(ValueError) as exc_info:
            parse_plan_to_dag(premises, inferences)

        assert "Duplicate inference name 'data_verified'" in str(exc_info.value)
        assert "I1" in str(exc_info.value)
        assert "I2" in str(exc_info.value)

    def test_inference_name_conflicts_with_premise_raises_error(self):
        """Inference name that matches a premise name should raise ValueError."""
        premises = [
            {"id": "P1", "name": "employees", "description": "Employees", "source": "database"},
        ]
        inferences = [
            {"id": "I1", "name": "employees", "operation": "filter(P1, active)", "explanation": "Filtered"},
        ]

        with pytest.raises(ValueError) as exc_info:
            parse_plan_to_dag(premises, inferences)

        assert "conflicts with premise" in str(exc_info.value)
        assert "employees" in str(exc_info.value)


class TestExtractDependencies:
    """Tests for extract_dependencies function."""

    def test_p_references(self):
        """Extract P1, P2, etc. references."""
        known = {"P1": "employees", "P2": "reviews"}
        deps = extract_dependencies("join(P1, P2)", known)
        assert deps == ["employees", "reviews"]

    def test_i_references(self):
        """Extract I1, I2, etc. references."""
        known = {"I1": "joined", "I2": "filtered"}
        deps = extract_dependencies("format(I2)", known)
        assert deps == ["filtered"]

    def test_mixed_references(self):
        """Extract mixed P and I references."""
        known = {"P1": "employees", "P2": "rules", "I1": "joined"}
        deps = extract_dependencies("apply(I1, P2)", known)
        assert set(deps) == {"joined", "rules"}

    def test_named_references(self):
        """Extract named references."""
        known = {"employees": "employees", "reviews": "reviews"}
        deps = extract_dependencies("join(employees, reviews)", known)
        assert deps == ["employees", "reviews"]

    def test_no_duplicates(self):
        """No duplicate dependencies."""
        known = {"P1": "data", "data": "data"}
        deps = extract_dependencies("process(P1, data, P1)", known)
        assert deps == ["data"]


class TestDAGExecutor:
    """Tests for DAGExecutor class."""

    def test_execute_single_node(self):
        """Execute single-node DAG."""
        node = FactNode(name="A", description="Test")
        dag = ExecutionDAG([node])

        def executor(n):
            return f"value_{n.name}", 1.0

        result = DAGExecutor(dag, executor).execute()

        assert result.success
        assert dag.nodes["A"].value == "value_A"

    def test_execute_parallel_nodes(self):
        """Parallel nodes execute concurrently."""
        nodes = [
            FactNode(name=f"N{i}", description=f"Node {i}")
            for i in range(3)
        ]
        dag = ExecutionDAG(nodes)

        execution_order = []

        def executor(n):
            execution_order.append(("start", n.name))
            time.sleep(0.1)  # Simulate work
            execution_order.append(("end", n.name))
            return f"value_{n.name}", 1.0

        start = time.time()
        result = DAGExecutor(dag, executor, max_workers=3).execute()
        duration = time.time() - start

        assert result.success
        # If truly parallel, should take ~0.1s not ~0.3s
        assert duration < 0.25

    def test_execute_with_dependencies(self):
        """Dependencies are respected during execution."""
        a = FactNode(name="A", description="First")
        b = FactNode(name="B", description="Second", dependencies=["A"])
        dag = ExecutionDAG([a, b])

        execution_order = []

        def executor(n):
            execution_order.append(n.name)
            return f"value_{n.name}", 1.0

        result = DAGExecutor(dag, executor).execute()

        assert result.success
        assert execution_order.index("A") < execution_order.index("B")

    def test_execute_with_failure(self):
        """Handle node execution failures."""
        a = FactNode(name="A", description="First")
        b = FactNode(name="B", description="Second", dependencies=["A"])
        dag = ExecutionDAG([a, b])

        def executor(n):
            if n.name == "A":
                raise Exception("Failed!")
            return f"value_{n.name}", 1.0

        result = DAGExecutor(dag, executor, fail_fast=True).execute()

        assert not result.success
        assert "A" in result.failed_nodes
        assert dag.nodes["A"].status == NodeStatus.FAILED
        # B never ran because A failed (fail_fast=True)
        assert dag.nodes["B"].status != NodeStatus.RESOLVED

    def test_event_callback(self):
        """Event callback is invoked for node events."""
        node = FactNode(name="A", description="Test")
        dag = ExecutionDAG([node])

        events = []

        def callback(event_type, data):
            events.append((event_type, data["name"]))

        def executor(n):
            return f"value_{n.name}", 1.0

        DAGExecutor(dag, executor, event_callback=callback).execute()

        event_types = [e[0] for e in events]
        assert "node_running" in event_types
        assert "node_resolved" in event_types

    def test_execution_levels_in_result(self):
        """Result includes execution levels."""
        a = FactNode(name="A", description="First")
        b = FactNode(name="B", description="Second", dependencies=["A"])
        dag = ExecutionDAG([a, b])

        def executor(n):
            return f"value_{n.name}", 1.0

        result = DAGExecutor(dag, executor).execute()

        assert len(result.execution_levels) == 2


class TestDagToDisplayFormat:
    """Tests for dag_to_display_format function."""

    def test_premises_format(self):
        """Premises are formatted correctly."""
        premises = [
            {"id": "P1", "name": "employees", "description": "All employees", "source": "database:sales_db"},
        ]
        dag = parse_plan_to_dag(premises, [])

        display = dag_to_display_format(dag)

        assert "PREMISES:" in display
        assert "P1: employees = ? (All employees) [source: database:sales_db]" in display

    def test_inferences_format(self):
        """Inferences are formatted correctly."""
        premises = [
            {"id": "P1", "name": "employees", "description": "Employees", "source": "database"},
        ]
        inferences = [
            {"id": "I1", "name": "filtered", "operation": "filter(P1, active)", "explanation": "Filter active"},
        ]
        dag = parse_plan_to_dag(premises, inferences)

        display = dag_to_display_format(dag)

        assert "INFERENCES:" in display
        assert "I1:" in display
        assert "filtered" in display
        assert "filter" in display

    def test_complete_plan_format(self):
        """Complete plan with premises and inferences."""
        premises = [
            {"id": "P1", "name": "employees", "description": "Employees", "source": "database"},
            {"id": "P2", "name": "reviews", "description": "Reviews", "source": "database"},
        ]
        inferences = [
            {"id": "I1", "name": "joined", "operation": "join(P1, P2)", "explanation": "Join data"},
        ]
        dag = parse_plan_to_dag(premises, inferences)

        display = dag_to_display_format(dag)

        assert "PREMISES:" in display
        assert "INFERENCES:" in display
        assert "P1" in display
        assert "P2" in display
        assert "I1" in display


class TestIntegration:
    """Integration tests for full DAG workflow."""

    def test_full_workflow(self):
        """Test complete parse -> execute -> display workflow."""
        # Parse plan
        premises = [
            {"id": "P1", "name": "employees", "description": "All employees", "source": "database"},
            {"id": "P2", "name": "departments", "description": "Departments", "source": "database"},
        ]
        inferences = [
            {"id": "I1", "name": "joined", "operation": "join(P1, P2)", "explanation": "Join"},
            {"id": "I2", "name": "summary", "operation": "summarize(I1)", "explanation": "Summary"},
        ]

        dag = parse_plan_to_dag(premises, inferences)

        # Verify structure
        levels = dag.get_execution_order()
        assert len(levels) == 3  # [P1, P2], [I1], [I2]

        # Execute
        def executor(n):
            if n.is_leaf:
                return f"data_for_{n.name}", 0.9
            else:
                # Verify dependencies are resolved
                for dep in n.dependencies:
                    assert dag.nodes[dep].status == NodeStatus.RESOLVED
                return f"computed_{n.name}", 0.85

        result = DAGExecutor(dag, executor).execute()

        assert result.success
        assert all(n.status == NodeStatus.RESOLVED for n in dag.nodes.values())

        # Display
        display = dag_to_display_format(dag)
        assert "PREMISES:" in display
        assert "INFERENCES:" in display
