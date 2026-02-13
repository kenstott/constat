# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""LivePlanExecutionDisplay for DAG-based plan execution."""

from __future__ import annotations

import re
import threading
import time
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table

from constat.repl.feedback._models import SPINNER_FRAMES, PlanItem


class LivePlanExecutionDisplay:
    """
    Live-updating display for plan execution.

    Shows all premises and inferences upfront with status indicators:
    - ○ pending
    - ⠋ running (animated spinner)
    - ✓ resolved
    - ✗ failed
    """

    def __init__(self, console: Console, premises: list[dict], inferences: list[dict]):
        self.console = console
        self.premises = premises
        self.inferences = inferences
        self._lock = threading.Lock()
        self._live: Optional[Live] = None
        self._spinner_frame: int = 0
        self._animation_thread: Optional[threading.Thread] = None
        self._animation_running: bool = False

        # Initialize plan items
        self.items: dict[str, PlanItem] = {}
        for p in premises:
            fact_id = p.get("id", "")
            self.items[fact_id] = PlanItem(
                fact_id=fact_id,
                name=p.get("name", fact_id),
                item_type="premise",
            )
        for inf in inferences:
            fact_id = inf.get("id", "")
            # Extract dependencies from operation (P1, P2, I1, etc.)
            operation = inf.get("operation", "")
            deps = re.findall(r'\b([PI]\d+)\b', operation)
            self.items[fact_id] = PlanItem(
                fact_id=fact_id,
                name=inf.get("name", "") or fact_id,
                item_type="inference",
                dependencies=deps,
            )

    def start(self) -> None:
        """Start the live display."""
        self._live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()
        self._start_animation()

    def stop(self) -> None:
        """Stop the live display, ensuring cleanup even on error."""
        self._stop_animation()
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass  # Ensure we don't fail during cleanup
            finally:
                self._live = None

    def _start_animation(self) -> None:
        """Start background animation thread."""
        self._animation_running = True
        self._animation_thread = threading.Thread(
            target=self._animation_loop,
            daemon=True,
        )
        self._animation_thread.start()

    def _stop_animation(self) -> None:
        """Stop animation thread."""
        self._animation_running = False
        if self._animation_thread:
            self._animation_thread.join(timeout=0.5)
            self._animation_thread = None

    def _animation_loop(self) -> None:
        """Background loop for spinner animation."""
        while self._animation_running:
            with self._lock:
                self._spinner_frame = (self._spinner_frame + 1) % len(SPINNER_FRAMES)
                if self._live:
                    try:
                        self._live.update(self._build_display())
                    except Exception:
                        pass
            time.sleep(0.1)

    def update_status(self, fact_id: str, status: str, value: str = None,
                      error: str = None, confidence: float = 0.0) -> None:
        """Update the status of a plan item."""
        with self._lock:
            matched_fact_id = None
            if fact_id in self.items:
                item = self.items[fact_id]
                item.status = status
                if value is not None:
                    item.value = value
                if error is not None:
                    item.error = error
                item.confidence = confidence
                matched_fact_id = fact_id
            # Also check by name (for events that use names)
            else:
                for item in self.items.values():
                    if item.name == fact_id or fact_id in item.name:
                        item.status = status
                        if value is not None:
                            item.value = value
                        if error is not None:
                            item.error = error
                        item.confidence = confidence
                        matched_fact_id = item.fact_id
                        break

            # If a node failed or blocked, cascade to all dependent nodes
            if status in ("failed", "blocked") and matched_fact_id:
                failed_name = self.items[matched_fact_id].name if matched_fact_id in self.items else matched_fact_id
                self._cascade_blocked(matched_fact_id, failed_name)

    def _cascade_blocked(self, blocked_id: str, root_cause: str) -> None:
        """Recursively mark all nodes depending on blocked_id as blocked."""
        newly_blocked = []
        for item in self.items.values():
            if item.status == "pending" and blocked_id in item.dependencies:
                item.status = "blocked"
                item.error = f"blocked: {root_cause} failed"
                newly_blocked.append(item.fact_id)
        # Recursively block dependents of newly blocked nodes
        for fact_id in newly_blocked:
            self._cascade_blocked(fact_id, root_cause)

    def _get_status_indicator(self, item: PlanItem) -> str:
        """Get the status indicator for an item."""
        if item.status == "pending":
            return "[dim]○[/dim]"
        elif item.status == "running":
            return f"[cyan]{SPINNER_FRAMES[self._spinner_frame]}[/cyan]"
        elif item.status == "resolved":
            return "[green]✓[/green]"
        elif item.status == "failed":
            return "[red]✗[/red]"
        elif item.status == "blocked":
            return "[yellow]⊘[/yellow]"
        return "[dim]?[/dim]"

    def _format_value(self, item: PlanItem) -> str:
        """Format the value/result for display."""
        if item.status == "pending":
            return "[dim]pending[/dim]"
        elif item.status == "running":
            return "[cyan]resolving...[/cyan]"
        elif item.status == "resolved":
            if item.value:
                val_str = str(item.value)
                # Collapse and truncate
                val_str = val_str.replace("\n", " ")
                if len(val_str) > 50:
                    val_str = val_str[:47] + "..."
                conf_str = f" [dim]({item.confidence:.0%})[/dim]" if item.confidence else ""
                return f"[green]{val_str}[/green]{conf_str}"
            return "[green]done[/green]"
        elif item.status == "failed":
            err = item.error or "error"
            if len(err) > 40:
                err = err[:37] + "..."
            return f"[red]{err}[/red]"
        elif item.status == "blocked":
            err = item.error or "blocked by failed dependency"
            if len(err) > 50:
                err = err[:47] + "..."
            return f"[yellow]{err}[/yellow]"
        return ""

    def _build_display(self) -> Table:
        """Build the display table."""
        table = Table(
            show_header=False,
            box=None,
            padding=(0, 1),
            collapse_padding=True,
        )
        table.add_column("status", width=2)
        table.add_column("id", width=4)
        table.add_column("name", min_width=20)
        table.add_column("result", min_width=30)

        # Add premises
        for p in self.premises:
            fact_id = p.get("id", "")
            if fact_id in self.items:
                item = self.items[fact_id]
                table.add_row(
                    self._get_status_indicator(item),
                    f"[bold]{fact_id}[/bold]",
                    item.name,
                    self._format_value(item),
                )

        # Separator
        if self.premises and self.inferences:
            table.add_row("", "", "[dim]───────────────────[/dim]", "")

        # Add inferences
        for inf in self.inferences:
            fact_id = inf.get("id", "")
            if fact_id in self.items:
                item = self.items[fact_id]
                table.add_row(
                    self._get_status_indicator(item),
                    f"[bold]{fact_id}[/bold]",
                    item.name,
                    self._format_value(item),
                )

        return table
