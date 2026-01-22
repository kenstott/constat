# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Proof tree display for auditable mode.

Displays a live tree view of fact resolution with status updates and intermediate results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from rich.tree import Tree
from rich.text import Text
from rich.panel import Panel
from rich.console import Group


class NodeStatus(Enum):
    """Status of a proof tree node."""
    PENDING = "pending"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class ProofNode:
    """A node in the proof tree representing a fact."""
    name: str
    description: str = ""
    status: NodeStatus = NodeStatus.PENDING
    value: Any = None
    source: str = ""
    confidence: float = 0.0
    error: Optional[str] = None
    children: list["ProofNode"] = field(default_factory=list)
    # For display: SQL query, code snippet, or other context
    query: str = ""
    # Brief summary of the result (for intermediate display)
    result_summary: str = ""
    # Depth in tree for indentation
    depth: int = 0

    def add_child(self, child: "ProofNode") -> "ProofNode":
        """Add a child dependency."""
        child.depth = self.depth + 1
        self.children.append(child)
        return child

    def find_node(self, name: str) -> Optional["ProofNode"]:
        """Find a node by name in this subtree."""
        if self.name == name:
            return self
        for child in self.children:
            found = child.find_node(name)
            if found:
                return found
        return None

    def get_status_icon(self, spinner_char: str = None) -> str:
        """Get the status icon for this node.

        Args:
            spinner_char: Optional animated spinner character for RESOLVING status.
                         If not provided, uses static icon.
        """
        if self.status == NodeStatus.RESOLVING and spinner_char:
            return spinner_char
        icons = {
            NodeStatus.PENDING: "○",
            NodeStatus.RESOLVING: "⋯",
            NodeStatus.RESOLVED: "✓",
            NodeStatus.FAILED: "✗",
            NodeStatus.CACHED: "⚡",
        }
        return icons.get(self.status, "?")

    def get_status_style(self) -> str:
        """Get the Rich style for this node's status."""
        styles = {
            NodeStatus.PENDING: "dim",
            NodeStatus.RESOLVING: "yellow",
            NodeStatus.RESOLVED: "green",
            NodeStatus.FAILED: "red",
            NodeStatus.CACHED: "cyan",
        }
        return styles.get(self.status, "")


class ProofTree:
    """
    Manages the proof tree structure and renders it as a Rich Tree.

    Tracks the resolution of facts and their dependencies, updating
    the display as facts are resolved.
    """

    def __init__(self, conclusion_name: str, conclusion_description: str = ""):
        """Initialize the proof tree with the conclusion to prove."""
        self.root = ProofNode(
            name=conclusion_name,
            description=conclusion_description,
            status=NodeStatus.RESOLVING,
        )
        # Map of fact names to nodes for quick lookup
        self._nodes: dict[str, ProofNode] = {conclusion_name: self.root}
        # Current node being resolved (for adding children)
        self._current_node: Optional[ProofNode] = self.root

    def add_fact(
        self,
        name: str,
        description: str = "",
        parent_name: Optional[str] = None,
    ) -> ProofNode:
        """Add a fact to the tree as a child of the specified parent.

        Args:
            name: Fact name
            description: Fact description
            parent_name: Parent fact name (defaults to root if not specified)
        """
        if name in self._nodes:
            return self._nodes[name]

        node = ProofNode(name=name, description=description)

        # Find parent node - default to root if not specified
        import logging
        logger = logging.getLogger(__name__)
        parent = None
        if parent_name:
            parent = self._nodes.get(parent_name)
            logger.debug(f"[PROOF_TREE] add_fact '{name}' looking for parent '{parent_name}': exact match={parent is not None}")
            # If not found by exact match, try matching by bare name
            # (nodes are stored as "P1: employees" but deps may be "employees")
            if not parent:
                for key, candidate in self._nodes.items():
                    # Check if key ends with ": parent_name"
                    if key.endswith(f": {parent_name}"):
                        logger.debug(f"[PROOF_TREE] Found parent by suffix: key='{key}' matches ':{parent_name}'")
                        parent = candidate
                        break
        if not parent:
            logger.debug(f"[PROOF_TREE] No parent found for '{name}', using root. Available nodes: {list(self._nodes.keys())}")
            parent = self.root
        else:
            logger.debug(f"[PROOF_TREE] '{name}' -> parent '{parent.name}'")

        parent.add_child(node)
        self._nodes[name] = node
        return node

    def start_resolving(self, name: str, description: str = "", parent_name: str = None) -> ProofNode:
        """Mark a fact as being resolved.

        Args:
            name: Fact name
            description: Fact description
            parent_name: Name of parent fact (if known), otherwise uses root
        """
        node = self._nodes.get(name)
        if not node:
            # Add as child of specified parent, or root if not specified
            node = self.add_fact(name, description, parent_name=parent_name or self.root.name)

        node.status = NodeStatus.RESOLVING
        node.description = description or node.description
        return node

    def resolve_fact(
        self,
        name: str,
        value: Any,
        source: str = "",
        confidence: float = 1.0,
        query: str = "",
        result_summary: str = "",
        from_cache: bool = False,
    ) -> ProofNode:
        """Mark a fact as resolved with its value."""
        node = self._nodes.get(name)
        if not node:
            node = self.add_fact(name)

        node.status = NodeStatus.CACHED if from_cache else NodeStatus.RESOLVED
        node.value = value
        node.source = source
        node.confidence = confidence
        node.query = query
        node.result_summary = result_summary or self._format_value_summary(value)
        return node

    def fail_fact(self, name: str, error: str) -> ProofNode:
        """Mark a fact as failed."""
        node = self._nodes.get(name)
        if not node:
            node = self.add_fact(name)

        node.status = NodeStatus.FAILED
        node.error = error
        return node

    def add_dependency(self, fact_name: str, depends_on: str) -> None:
        """Record that fact_name depends on depends_on."""
        parent = self._nodes.get(fact_name)
        if not parent:
            parent = self.add_fact(fact_name)

        # Check if dependency already exists
        if depends_on not in self._nodes:
            child = ProofNode(name=depends_on)
            parent.add_child(child)
            self._nodes[depends_on] = child

    def _format_value_summary(self, value: Any, max_len: int = 60) -> str:
        """Format a value for brief display."""
        if value is None:
            return ""

        if isinstance(value, (int, float)):
            if isinstance(value, float):
                return f"{value:,.2f}"
            return f"{value:,}"

        if isinstance(value, str):
            if value.startswith("table:"):
                # Don't show internal table reference - show it as a table result
                # The format is "N rows (table: name)" from Fact.display_value
                if " rows" in value:
                    return value  # Already formatted nicely like "20 rows (table: fact_xxx)"
                # Raw table reference - show as simple indicator
                return "[table result]"
            # Collapse newlines for compact single-line display
            value = value.replace("\n", " ").replace("  ", " ")
            if len(value) > max_len:
                return value[:max_len] + "..."
            return value

        if isinstance(value, list):
            if len(value) == 0:
                return "[]"
            if len(value) <= 3:
                return str(value)
            return f"[{len(value)} items]"

        if isinstance(value, dict):
            if len(value) == 0:
                return "{}"
            if len(value) <= 3:
                items = ", ".join(f"{k}: {v}" for k, v in list(value.items())[:3])
                return f"{{{items}}}"
            return f"{{{len(value)} keys}}"

        return str(value)[:max_len]

    def render(self, spinner_char: str = None) -> Tree:
        """Render the proof tree as a Rich Tree.

        Args:
            spinner_char: Optional animated spinner character for RESOLVING nodes.
        """
        # Build the root label
        root_label = self._build_node_label(self.root, is_root=True, spinner_char=spinner_char)
        tree = Tree(root_label)

        # Recursively add children
        self._add_children_to_tree(tree, self.root, spinner_char=spinner_char)

        return tree

    def _build_node_label(self, node: ProofNode, is_root: bool = False, spinner_char: str = None) -> Text:
        """Build the label for a tree node."""
        text = Text()

        # Status icon (use animated spinner if provided and node is resolving)
        icon = node.get_status_icon(spinner_char)
        style = node.get_status_style()
        text.append(f"{icon} ", style=style)

        # Fact name
        if is_root:
            text.append("Prove: ", style="bold")
        text.append(node.name, style="bold" if is_root else "")

        # Status-specific suffix
        if node.status == NodeStatus.RESOLVING:
            text.append(" resolving...", style="yellow dim")
        elif node.status == NodeStatus.PENDING:
            text.append(" pending", style="dim")
        elif node.status == NodeStatus.FAILED:
            text.append(f" FAILED: {node.error or 'unknown'}", style="red")
        elif node.status in (NodeStatus.RESOLVED, NodeStatus.CACHED):
            # Show source and summary
            if node.source:
                source_label = "cached" if node.status == NodeStatus.CACHED else node.source
                text.append(f" ({source_label})", style="dim")
            if node.result_summary:
                text.append(f" → {node.result_summary}", style="cyan")
            if node.confidence < 1.0:
                text.append(f" [{node.confidence:.0%}]", style="dim")

        return text

    def _add_children_to_tree(self, tree: Tree, node: ProofNode, spinner_char: str = None) -> None:
        """Recursively add children to a Rich Tree."""
        for child in node.children:
            child_label = self._build_node_label(child, spinner_char=spinner_char)
            subtree = tree.add(child_label)
            # Recurse to add any nested children
            self._add_children_to_tree(subtree, child, spinner_char=spinner_char)

    def render_with_panel(self) -> Panel:
        """Render the proof tree in a panel."""
        tree = self.render()
        return Panel(
            tree,
            title="[bold]Proof Structure[/bold]",
            border_style="blue",
            padding=(0, 1),
        )

    def get_summary(self) -> dict:
        """Get a summary of the proof state."""
        total = len(self._nodes)
        resolved = sum(1 for n in self._nodes.values() if n.status == NodeStatus.RESOLVED)
        cached = sum(1 for n in self._nodes.values() if n.status == NodeStatus.CACHED)
        failed = sum(1 for n in self._nodes.values() if n.status == NodeStatus.FAILED)
        pending = sum(1 for n in self._nodes.values() if n.status in (NodeStatus.PENDING, NodeStatus.RESOLVING))

        return {
            "total": total,
            "resolved": resolved,
            "cached": cached,
            "failed": failed,
            "pending": pending,
            "complete": pending == 0 and failed == 0,
            "confidence": self.root.confidence if self.root.status == NodeStatus.RESOLVED else 0.0,
        }
