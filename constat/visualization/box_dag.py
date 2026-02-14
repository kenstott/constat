# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""
Compact DAG renderer - nodes as plain text without boxes.
Long linear chains rendered horizontally with count-based snaking.
"""

from typing import Dict, List, Set, Tuple

import networkx as nx


class CompactDAG:
    """Renders DAGs compactly with optional horizontal snaking for linear chains."""

    def __init__(self, graph: nx.DiGraph, min_spacing: int = 3,
                 snake_chains: bool = True, max_chain_width: int = 80,
                 min_chain_length: int = 3):
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Graph must be a DAG")

        self.original_graph = graph.copy()
        self.graph = graph.copy()
        self.min_spacing = min_spacing
        self.snake_chains = snake_chains
        self.max_chain_width = max_chain_width
        self.min_chain_length = min_chain_length
        self.layers: List[List[str]] = []
        self.node_layer: Dict[str, int] = {}
        self.node_pos: Dict[str, Tuple[int, int]] = {}
        self.node_center: Dict[str, int] = {}

    def _find_linear_tail(self) -> List[str]:
        """Find longest linear tail ending at a sink."""
        sinks = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]

        best_chain = []
        for sink in sinks:
            chain = [sink]
            current = sink

            while self.graph.in_degree(current) == 1:
                pred = list(self.graph.predecessors(current))[0]
                if self.graph.out_degree(pred) != 1:
                    break
                chain.append(pred)
                current = pred

            chain.reverse()
            if len(chain) > len(best_chain):
                best_chain = chain

        return best_chain if len(best_chain) >= self.min_chain_length else []

    def _render_chain_snaked(self, chain: List[str], indent: int = 0) -> List[str]:
        """Render a linear chain horizontally with width-based snaking."""
        if len(chain) <= 1:
            return [" " * indent + chain[0]] if chain else []

        arrow_right = " ─▶ "
        arrow_left = " ◀─ "
        arrow_len = len(arrow_right)

        # Split into rows based on max_chain_width
        rows = []
        current_row = []
        current_width = indent

        for node in chain:
            node_add = len(node) + (arrow_len if current_row else 0)

            if current_width + node_add > self.max_chain_width and current_row:
                # Start new row
                rows.append(current_row)
                current_row = [node]
                current_width = indent + len(node)
            else:
                current_row.append(node)
                current_width += node_add

        if current_row:
            rows.append(current_row)

        if len(rows) == 1:
            return [" " * indent + arrow_right.join(rows[0])]

        lines = []

        # Calculate row width
        def row_width(r):
            return sum(len(n) for n in r) + arrow_len * (len(r) - 1)

        # Pre-calculate positions for each row
        row_starts = [indent]  # Row 0 starts at indent

        for i in range(len(rows) - 1):
            row_w = row_width(rows[i])
            next_row_w = row_width(rows[i + 1])

            if i % 2 == 0:
                # L→R row: next row (R→L) right-aligns with this row's right edge
                right_edge = row_starts[i] + row_w
                row_starts.append(right_edge - next_row_w)
            else:
                # R→L row: next row (L→R) left-aligns with this row's left edge
                row_starts.append(row_starts[i])

        # Render each row with connectors
        for i, row in enumerate(rows):
            row_start = row_starts[i]
            row_w = row_width(row)

            if i % 2 == 0:
                # Left to right
                text = arrow_right.join(row)
                lines.append(" " * row_start + text)

                # Down connector from last node: │ then ▼
                if i < len(rows) - 1:
                    last_node_pos = row_start + row_w - len(row[-1])
                    arrow_pos = last_node_pos + len(row[-1]) // 2
                    lines.append(" " * arrow_pos + "│")
                    lines.append(" " * arrow_pos + "▼")
            else:
                # Right to left (reversed in display)
                text = arrow_left.join(reversed(row))
                lines.append(" " * row_start + text)

                # Down connector from first node (leftmost in display): │ then ▼
                if i < len(rows) - 1:
                    arrow_pos = row_start + len(row[0]) // 2
                    lines.append(" " * arrow_pos + "│")
                    lines.append(" " * arrow_pos + "▼")

        return lines

    def _assign_layers_and_dummies(self):
        """Assign layers and insert dummy nodes."""
        for layer_idx, generation in enumerate(nx.topological_generations(self.graph)):
            layer = sorted(generation)
            self.layers.append(layer)
            for node in layer:
                self.node_layer[node] = layer_idx

        dummy_count = 0
        edges_to_process = [(s, t) for s, t in self.graph.edges()
                           if self.node_layer[t] - self.node_layer[s] > 1]

        for src, tgt in edges_to_process:
            src_layer = self.node_layer[src]
            tgt_layer = self.node_layer[tgt]
            self.graph.remove_edge(src, tgt)
            prev = src
            for j in range(1, tgt_layer - src_layer):
                dummy = f"__d{dummy_count}__"
                dummy_count += 1
                self.layers[src_layer + j].append(dummy)
                self.node_layer[dummy] = src_layer + j
                self.graph.add_edge(prev, dummy)
                prev = dummy
            self.graph.add_edge(prev, tgt)

    def _calculate_positions(self):
        """Calculate node positions."""
        for layer in self.layers:
            col = 0
            for node in sorted(layer):
                width = 1 if node.startswith("__d") else len(node)
                self.node_pos[node] = (col, col + width - 1)
                self.node_center[node] = col + width // 2
                col += width + self.min_spacing

        for layer_idx in range(1, len(self.layers)):
            layer = self.layers[layer_idx]
            desired = {}
            for node in layer:
                parents = list(self.graph.predecessors(node))
                if parents:
                    desired[node] = sum(self.node_center[p] for p in parents) // len(parents)
                else:
                    desired[node] = self.node_center[node]

            sorted_nodes = sorted(layer, key=lambda n: desired.get(n, 0))
            col = 0
            for node in sorted_nodes:
                width = 1 if node.startswith("__d") else len(node)
                target = max(col, desired.get(node, col) - width // 2)
                self.node_pos[node] = (target, target + width - 1)
                self.node_center[node] = target + width // 2
                col = target + width + self.min_spacing

    def _get_width(self) -> int:
        return max((end + 1 for start, end in self.node_pos.values()), default=1)

    def _render_layer(self, layer_idx: int, exclude: Set[str] = None) -> str:
        exclude = exclude or set()
        layer = [n for n in self.layers[layer_idx] if n not in exclude]
        if not layer:
            return ""

        width = self._get_width()
        line = [' '] * width

        for node in layer:
            start, end = self.node_pos[node]
            if node.startswith("__d"):
                line[start] = '│'
            else:
                for j, ch in enumerate(node):
                    if start + j < width:
                        line[start + j] = ch

        return ''.join(line).rstrip()

    def _render_connectors(self, layer_idx: int, exclude: Set[str] = None) -> List[str]:
        exclude = exclude or set()

        if layer_idx >= len(self.layers) - 1:
            return []

        curr = [n for n in self.layers[layer_idx] if n not in exclude]
        next_l = [n for n in self.layers[layer_idx + 1] if n not in exclude]

        if not curr or not next_l:
            return []

        edges = []
        for src in curr:
            for tgt in self.graph.successors(src):
                if tgt in next_l:
                    edges.append((self.node_center[src], self.node_center[tgt],
                                 tgt.startswith("__d")))

        if not edges:
            return []

        src_cols = {e[0] for e in edges}
        tgt_cols = {e[1] for e in edges}
        tgt_dummy = {e[1]: e[2] for e in edges}
        all_cols = src_cols | tgt_cols
        min_c, max_c = min(all_cols), max(all_cols)
        width = self._get_width()

        lines = []

        v = [' '] * width
        for c in src_cols:
            v[c] = '│'
        lines.append(''.join(v).rstrip())

        h = [' '] * width
        for c in range(min_c, max_c + 1):
            h[c] = '─'
        for c in range(min_c, max_c + 1):
            above = c in src_cols
            below = c in tgt_cols
            left = c == min_c
            right = c == max_c
            if above and below:
                h[c] = '┼'
            elif above:
                h[c] = '└' if left else ('┘' if right else '┴')
            elif below:
                h[c] = '┌' if left else ('┐' if right else '┬')
        lines.append(''.join(h).rstrip())

        a = [' '] * width
        for c in tgt_cols:
            a[c] = '│' if tgt_dummy.get(c) else '▼'
        lines.append(''.join(a).rstrip())

        return lines

    def render(self) -> str:
        if not self.graph.nodes():
            return ""

        self._assign_layers_and_dummies()
        self._calculate_positions()

        if self.snake_chains:
            tail = self._find_linear_tail()
            real_nodes = [n for n in self.graph.nodes() if not n.startswith("__d")]

            # Entire graph is linear
            if len(tail) == len(real_nodes):
                return '\n'.join(self._render_chain_snaked(tail))

            # Has a tail worth snaking
            if tail:
                exclude_from_vertical = set(tail[1:])  # Keep tail[0] in vertical

                output = []
                for i in range(len(self.layers)):
                    layer_line = self._render_layer(i, exclude_from_vertical)
                    if layer_line.strip():
                        output.append(layer_line)

                    if i < len(self.layers) - 1:
                        next_non_tail = [n for n in self.layers[i + 1]
                                        if n not in exclude_from_vertical]
                        if next_non_tail:
                            conn = self._render_connectors(i, exclude_from_vertical)
                            output.extend(conn)

                # Add connector and snake, aligned under tail[0]
                if len(tail) > 1:
                    tail_start = tail[0]
                    indent = self.node_center[tail_start]

                    # Connector under tail[0]
                    conn_line = ' ' * indent + '│'
                    output.append(conn_line)
                    arrow_line = ' ' * indent + '▼'
                    output.append(arrow_line)

                    # Snake the rest of the tail
                    output.extend(self._render_chain_snaked(tail[1:], indent=indent))

                return '\n'.join(output)

        # Standard rendering
        output = []
        for i in range(len(self.layers)):
            output.append(self._render_layer(i))
            if i < len(self.layers) - 1:
                output.extend(self._render_connectors(i))

        return '\n'.join(output)


def render_dag(graph: nx.DiGraph, _style: str = 'single',
               snake: bool = True, max_width: int = 80) -> str:
    """Render a DAG compactly. Linear chains snake horizontally at max_width.

    Args:
        graph: NetworkX DiGraph (must be a DAG)
        style: Ignored (kept for backward compatibility)
        snake: Whether to snake linear chains horizontally
        max_width: Maximum width before snaking to next line

    Returns:
        String representation of the DAG
    """
    return CompactDAG(graph, snake_chains=snake, max_chain_width=max_width).render()


def render_compact(graph: nx.DiGraph, spacing: int = 3,
                   snake: bool = True, max_width: int = 80) -> str:
    """Render a DAG compactly. Linear chains snake horizontally at max_width."""
    return CompactDAG(graph, min_spacing=spacing,
                      snake_chains=snake, max_chain_width=max_width).render()


def generate_proof_dfd(
    steps: List[Dict],
    max_width: int = 80,
    max_name_len: int = 10,
) -> str:
    """
    Generate a Data Flow Diagram from proof steps.

    Args:
        steps: List of step dicts with 'fact_id', 'goal', 'type' keys
               - type: 'premise', 'inference', or 'conclusion'
               - fact_id: P1, P2, I1, I2, etc.
               - goal: "var_name = operation" or "var_name = ? (desc) [source: x]"
        max_width: Maximum width for rendering
        max_name_len: Max characters for node labels (truncated)

    Returns:
        Rendered ASCII diagram string
    """
    import re

    # Build mapping from fact_id (P1, I1) to English variable name
    fact_to_name: Dict[str, str] = {}
    used_names: Dict[str, int] = {}
    for s in steps:
        fact_id = s.get("fact_id", "")
        goal = s.get("goal", "")
        # Extract English name from BEFORE the first '='
        # Premise format: "employees = ? (description) [source: xxx]"
        # Inference format: "remaining_days = P1 - P2 -- explanation"
        if "=" in goal:
            english_name = goal.split("=", 1)[0].strip()
            # Truncate to fit
            if len(english_name) > max_name_len:
                english_name = english_name[:max_name_len]
            # Deduplicate: append suffix if name already used
            if english_name in used_names:
                used_names[english_name] += 1
                english_name = f"{english_name}_{used_names[english_name]}"
            else:
                used_names[english_name] = 1
            fact_to_name[fact_id] = english_name
        elif fact_id:
            fact_to_name[fact_id] = fact_id  # Fallback to fact_id

    # Build NetworkX graph using English names
    G = nx.DiGraph()

    for s in steps:
        step_type = s.get("type")
        fact_id = s.get("fact_id", "")
        goal = s.get("goal", "")
        node_name = fact_to_name.get(fact_id, fact_id)

        if step_type == "premise":
            G.add_node(node_name)
        elif step_type == "inference":
            G.add_node(node_name)
            # Extract dependencies from the operation
            inf_match = re.match(r'^(\w+)\s*=\s*(.+)', goal)
            if inf_match:
                operation = inf_match.group(2)
            else:
                operation = goal
            deps = re.findall(r'[PI]\d+', operation)
            for dep in deps:
                dep_name = fact_to_name.get(dep, dep)
                if G.has_node(dep_name):
                    G.add_edge(dep_name, node_name)

    if G.number_of_nodes() == 0:
        return "(No derivation graph available)"

    # Find terminal inference and add conclusion
    inference_names = [fact_to_name.get(f, f) for f in fact_to_name if f.startswith('I')]
    if inference_names:
        terminal = None
        for inf_name in inference_names:
            successors = list(G.successors(inf_name))
            if not any(s in inference_names for s in successors):
                terminal = inf_name
                break
        if terminal is None:
            terminal = inference_names[-1]
        G.add_node("conclusion")
        G.add_edge(terminal, "conclusion")

    return render_dag(G, max_width=max_width)
