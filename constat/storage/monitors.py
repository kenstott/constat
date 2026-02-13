# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Monitor storage for scheduled plan execution.

Provides persistent storage for monitors that schedule periodic re-execution
of saved plans with trigger conditions and actions.

Storage locations:
- Monitors: .constat/<user_id>/monitors.json
- Run history: .constat/<user_id>/monitor_runs/<monitor_id>.jsonl
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class MonitorTrigger:
    """Trigger condition for a monitor.

    Trigger types:
    - threshold: Fire when a fact crosses a threshold (e.g., revenue < 10000)
    - change: Fire when a fact changes by a percentage (e.g., users changed by 10%)
    - schedule_only: Fire on every scheduled run (no condition)
    """

    type: Literal["threshold", "change", "schedule_only"]
    fact_name: Optional[str] = None
    operator: Optional[str] = None  # <, >, <=, >=, ==, !=
    value: Optional[float] = None
    change_pct: Optional[float] = None

    def evaluate(self, current_value: Any, previous_value: Any = None) -> bool:
        """Evaluate whether the trigger condition is met.

        Args:
            current_value: Current value of the fact
            previous_value: Previous value (for change detection)

        Returns:
            True if the trigger should fire
        """
        if self.type == "schedule_only":
            return True

        if self.type == "threshold" and self.operator and self.value is not None:
            try:
                current = float(current_value)
                ops = {
                    "<": lambda a, b: a < b,
                    ">": lambda a, b: a > b,
                    "<=": lambda a, b: a <= b,
                    ">=": lambda a, b: a >= b,
                    "==": lambda a, b: a == b,
                    "!=": lambda a, b: a != b,
                }
                if self.operator in ops:
                    return ops[self.operator](current, self.value)
            except (TypeError, ValueError):
                return False

        if self.type == "change" and self.change_pct is not None and previous_value is not None:
            try:
                current = float(current_value)
                previous = float(previous_value)
                if previous == 0:
                    return current != 0
                pct_change = abs((current - previous) / previous) * 100
                return pct_change >= self.change_pct
            except (TypeError, ValueError):
                return False

        return False

    @classmethod
    def from_dict(cls, data: dict) -> MonitorTrigger:
        """Create a MonitorTrigger from a dictionary."""
        return cls(
            # noinspection PyTypeChecker
            type=data.get("type", "schedule_only"),
            fact_name=data.get("fact_name"),
            operator=data.get("operator"),
            value=data.get("value"),
            change_pct=data.get("change_pct"),
        )


@dataclass
class MonitorAction:
    """Action to execute when a monitor triggers.

    Action types:
    - log: Write to monitor history (always happens)
    - email: Send email notification
    - slack: Send Slack message
    - webhook: Call a webhook URL
    """

    type: str  # "log", "email", "slack", "webhook"
    config: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> MonitorAction:
        """Create a MonitorAction from a dictionary."""
        return cls(
            type=data.get("type", "log"),
            config=data.get("config", {}),
        )


@dataclass
class Monitor:
    """A scheduled monitor definition.

    Monitors periodically re-execute a saved plan and evaluate trigger
    conditions to determine whether to fire actions.
    """

    id: str
    name: str
    saved_plan_name: str
    schedule: str  # cron expression (e.g., "0 9 * * *" for 9am daily)
    trigger: Optional[MonitorTrigger] = None
    actions: list[MonitorAction] = field(default_factory=list)
    enabled: bool = True
    created_at: str = ""
    last_run: Optional[str] = None
    next_run: Optional[str] = None
    description: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "saved_plan_name": self.saved_plan_name,
            "schedule": self.schedule,
            "trigger": asdict(self.trigger) if self.trigger else None,
            "actions": [asdict(a) for a in self.actions],
            "enabled": self.enabled,
            "created_at": self.created_at,
            "last_run": self.last_run,
            "next_run": self.next_run,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Monitor:
        """Create a Monitor from a dictionary."""
        trigger = None
        if data.get("trigger"):
            trigger = MonitorTrigger.from_dict(data["trigger"])

        actions = [MonitorAction.from_dict(a) for a in data.get("actions", [])]

        return cls(
            id=data["id"],
            name=data["name"],
            saved_plan_name=data["saved_plan_name"],
            schedule=data["schedule"],
            trigger=trigger,
            actions=actions,
            enabled=data.get("enabled", True),
            created_at=data.get("created_at", ""),
            last_run=data.get("last_run"),
            next_run=data.get("next_run"),
            description=data.get("description", ""),
        )


@dataclass
class MonitorRun:
    """Record of a monitor execution."""

    id: str
    monitor_id: str
    started_at: str
    completed_at: Optional[str] = None
    status: Literal["running", "success", "failed", "triggered"] = "running"
    result_snapshot: dict = field(default_factory=dict)
    triggered: bool = False
    action_results: list[dict] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "monitor_id": self.monitor_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "result_snapshot": self.result_snapshot,
            "triggered": self.triggered,
            "action_results": self.action_results,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MonitorRun:
        """Create a MonitorRun from a dictionary."""
        return cls(
            id=data["id"],
            monitor_id=data["monitor_id"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            # noinspection PyTypeChecker
            status=data.get("status", "running"),
            result_snapshot=data.get("result_snapshot", {}),
            triggered=data.get("triggered", False),
            action_results=data.get("action_results", []),
            error=data.get("error"),
        )


class MonitorStore:
    """Persistence for monitors and run history.

    Monitors are stored per-user in .constat/<user_id>/monitors.json.
    Run history is stored in .constat/<user_id>/monitor_runs/<monitor_id>.jsonl.
    """

    CONSTAT_BASE_DIR = Path(".constat")

    def __init__(self, user_id: str, base_path: Optional[Path] = None):
        """Initialize monitor store.

        Args:
            user_id: User ID for scoping monitors
            base_path: Override base .constat directory
        """
        self.user_id = user_id
        self.base_path = base_path or self.CONSTAT_BASE_DIR
        self._monitors: Optional[dict[str, Monitor]] = None

    def _get_monitors_file(self) -> Path:
        """Get path to monitors file for this user."""
        return self.base_path / self.user_id / "monitors.json"

    def _get_runs_dir(self) -> Path:
        """Get path to monitor runs directory for this user."""
        return self.base_path / self.user_id / "monitor_runs"

    def _get_run_file(self, monitor_id: str) -> Path:
        """Get path to run history file for a specific monitor."""
        return self._get_runs_dir() / f"{monitor_id}.jsonl"

    def _load(self) -> dict[str, Monitor]:
        """Load monitors from JSON file."""
        if self._monitors is not None:
            return self._monitors

        monitors_file = self._get_monitors_file()
        if not monitors_file.exists():
            self._monitors = {}
            return self._monitors

        try:
            data = json.loads(monitors_file.read_text())
            self._monitors = {
                mid: Monitor.from_dict(mdata) for mid, mdata in data.items()
            }
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning(f"Could not load monitors from {monitors_file}: {e}")
            # Backup corrupted file before discarding
            if monitors_file.exists():
                try:
                    backup_path = monitors_file.with_suffix('.corrupted')
                    monitors_file.rename(backup_path)
                    logger.warning(f"Corrupted monitors file backed up to {backup_path}")
                except OSError as backup_err:
                    logger.error(f"Failed to backup corrupted monitors file: {backup_err}")
            self._monitors = {}

        return self._monitors

    def _save(self) -> None:
        """Save monitors to JSON file."""
        if self._monitors is None:
            return

        monitors_file = self._get_monitors_file()
        monitors_file.parent.mkdir(parents=True, exist_ok=True)

        data = {mid: m.to_dict() for mid, m in self._monitors.items()}
        monitors_file.write_text(json.dumps(data, indent=2))

    def save(self, monitor: Monitor) -> None:
        """Save a monitor (create or update).

        Args:
            monitor: Monitor to save
        """
        monitors = self._load()
        monitors[monitor.id] = monitor
        self._save()

    def get(self, monitor_id: str) -> Optional[Monitor]:
        """Get a monitor by ID.

        Args:
            monitor_id: Monitor ID

        Returns:
            Monitor or None if not found
        """
        monitors = self._load()
        return monitors.get(monitor_id)

    def get_by_name(self, name: str) -> Optional[Monitor]:
        """Get a monitor by name.

        Args:
            name: Monitor name

        Returns:
            Monitor or None if not found
        """
        monitors = self._load()
        for monitor in monitors.values():
            if monitor.name == name:
                return monitor
        return None

    def list_all(self) -> list[Monitor]:
        """List all monitors for this user.

        Returns:
            List of all monitors
        """
        monitors = self._load()
        return list(monitors.values())

    def delete(self, monitor_id: str) -> bool:
        """Delete a monitor by ID.

        Args:
            monitor_id: Monitor ID

        Returns:
            True if deleted, False if not found
        """
        monitors = self._load()
        if monitor_id not in monitors:
            return False

        del monitors[monitor_id]
        self._save()

        # Also delete run history
        run_file = self._get_run_file(monitor_id)
        if run_file.exists():
            run_file.unlink()

        return True

    def delete_by_name(self, name: str) -> bool:
        """Delete a monitor by name.

        Args:
            name: Monitor name

        Returns:
            True if deleted, False if not found
        """
        monitor = self.get_by_name(name)
        if monitor:
            return self.delete(monitor.id)
        return False

    def update(self, monitor: Monitor) -> None:
        """Update an existing monitor.

        Args:
            monitor: Monitor with updated fields
        """
        self.save(monitor)

    def record_run(self, run: MonitorRun) -> None:
        """Record a monitor run to history.

        Args:
            run: MonitorRun record to save
        """
        runs_dir = self._get_runs_dir()
        runs_dir.mkdir(parents=True, exist_ok=True)

        run_file = self._get_run_file(run.monitor_id)
        with open(run_file, "a") as f:
            f.write(json.dumps(run.to_dict()) + "\n")

        # Update monitor's last_run timestamp
        monitor = self.get(run.monitor_id)
        if monitor and run.completed_at:
            monitor.last_run = run.completed_at
            self.save(monitor)

    def get_runs(self, monitor_id: str, limit: int = 10) -> list[MonitorRun]:
        """Get recent runs for a monitor.

        Args:
            monitor_id: Monitor ID
            limit: Maximum number of runs to return

        Returns:
            List of MonitorRun records, most recent first
        """
        run_file = self._get_run_file(monitor_id)
        if not run_file.exists():
            return []

        runs = []
        try:
            with open(run_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        runs.append(MonitorRun.from_dict(json.loads(line)))
        except (json.JSONDecodeError, OSError):
            return []

        # Return most recent first, limited
        runs.reverse()
        return runs[:limit]

    def get_last_run(self, monitor_id: str) -> Optional[MonitorRun]:
        """Get the most recent run for a monitor.

        Args:
            monitor_id: Monitor ID

        Returns:
            Most recent MonitorRun or None
        """
        runs = self.get_runs(monitor_id, limit=1)
        return runs[0] if runs else None


def generate_monitor_id() -> str:
    """Generate a unique monitor ID."""
    return f"mon_{uuid.uuid4().hex[:12]}"


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"run_{uuid.uuid4().hex[:12]}"
