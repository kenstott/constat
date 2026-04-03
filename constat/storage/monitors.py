# Copyright (c) 2025 Kenneth Stott
# Canary: c2eb6848-601b-41cc-861c-ee3b59e1254e
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Monitor storage for scheduled plan execution, backed by DuckDB.

Provides persistent storage for monitors that schedule periodic re-execution
of saved plans with trigger conditions and actions.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Optional

from constat.storage.duckdb_pool import ThreadLocalDuckDB

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
        # noinspection PyTypeChecker
        return cls(
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
        # noinspection PyTypeChecker
        return cls(
            id=data["id"],
            monitor_id=data["monitor_id"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            status=data.get("status", "running"),
            result_snapshot=data.get("result_snapshot", {}),
            triggered=data.get("triggered", False),
            action_results=data.get("action_results", []),
            error=data.get("error"),
        )


class MonitorStore:
    """Persistence for monitors and run history, backed by DuckDB."""

    CONSTAT_BASE_DIR = Path(".constat")

    _MONITORS_DDL = """
    CREATE TABLE IF NOT EXISTS monitors (
        id VARCHAR PRIMARY KEY,
        data TEXT NOT NULL
    )
    """

    _RUNS_DDL = """
    CREATE TABLE IF NOT EXISTS monitor_runs (
        id VARCHAR PRIMARY KEY,
        monitor_id VARCHAR NOT NULL,
        data TEXT NOT NULL,
        started_at TIMESTAMP
    )
    """

    def __init__(
        self,
        user_id: str,
        base_path: Optional[Path] = None,
        db: Optional[ThreadLocalDuckDB] = None,
    ):
        """Initialize monitor store.

        Args:
            user_id: User ID for scoping monitors
            base_path: Override base .constat directory
            db: Existing ThreadLocalDuckDB connection to reuse. If None,
                opens a standalone connection to the user vault.
        """
        self.user_id = user_id
        self.base_path = base_path or self.CONSTAT_BASE_DIR
        self._owns_db = db is None
        if db is not None:
            self._db = db
        else:
            from constat.core.paths import user_vault_dir, migrate_db_name
            vault = user_vault_dir(self.base_path, user_id)
            vault.mkdir(parents=True, exist_ok=True)
            db_path = migrate_db_name(vault, "vectors.duckdb", "user.duckdb")
            self._db = ThreadLocalDuckDB(str(db_path))
        self._tables_ensured = False

    def _ensure_tables(self) -> None:
        if self._tables_ensured:
            return
        self._db.execute(self._MONITORS_DDL)
        self._db.execute(self._RUNS_DDL)
        self._tables_ensured = True
        self._import_json()

    def _import_json(self) -> None:
        """One-time import from legacy monitors.json + monitor_runs/*.jsonl."""
        from constat.core.paths import user_vault_dir
        vault = user_vault_dir(self.base_path, self.user_id)
        monitors_file = vault / "monitors.json"
        if not monitors_file.exists():
            return
        count = self._db.execute("SELECT COUNT(*) FROM monitors").fetchone()[0]
        if count > 0:
            return

        data = json.loads(monitors_file.read_text())
        for mid, mdata in data.items():
            self._db.execute(
                "INSERT INTO monitors (id, data) VALUES (?, ?)",
                [mid, json.dumps(mdata)],
            )

        # Import run history
        runs_dir = vault / "monitor_runs"
        if runs_dir.exists():
            for run_file in runs_dir.glob("*.jsonl"):
                with open(run_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        run_data = json.loads(line)
                        started = run_data.get("started_at")
                        self._db.execute(
                            "INSERT INTO monitor_runs (id, monitor_id, data, started_at) VALUES (?, ?, ?, ?)",
                            [run_data["id"], run_data["monitor_id"], json.dumps(run_data), started],
                        )

        monitors_file.rename(monitors_file.with_suffix(".json.imported"))
        logger.info("Imported monitors from %s", monitors_file)

    def save(self, monitor: Monitor) -> None:
        """Save a monitor (create or update).

        Args:
            monitor: Monitor to save
        """
        self._ensure_tables()
        self._db.execute(
            """INSERT INTO monitors (id, data) VALUES (?, ?)
               ON CONFLICT (id) DO UPDATE SET data = excluded.data""",
            [monitor.id, json.dumps(monitor.to_dict())],
        )

    def get(self, monitor_id: str) -> Optional[Monitor]:
        """Get a monitor by ID.

        Args:
            monitor_id: Monitor ID

        Returns:
            Monitor or None if not found
        """
        self._ensure_tables()
        row = self._db.execute(
            "SELECT data FROM monitors WHERE id = ?", [monitor_id]
        ).fetchone()
        if row is None:
            return None
        return Monitor.from_dict(json.loads(row[0]))

    def get_by_name(self, name: str) -> Optional[Monitor]:
        """Get a monitor by name.

        Args:
            name: Monitor name

        Returns:
            Monitor or None if not found
        """
        self._ensure_tables()
        rows = self._db.execute("SELECT data FROM monitors").fetchall()
        for (data_str,) in rows:
            data = json.loads(data_str)
            if data.get("name") == name:
                return Monitor.from_dict(data)
        return None

    def list_all(self) -> list[Monitor]:
        """List all monitors for this user.

        Returns:
            List of all monitors
        """
        self._ensure_tables()
        rows = self._db.execute("SELECT data FROM monitors").fetchall()
        return [Monitor.from_dict(json.loads(row[0])) for row in rows]

    def delete(self, monitor_id: str) -> bool:
        """Delete a monitor by ID.

        Args:
            monitor_id: Monitor ID

        Returns:
            True if deleted, False if not found
        """
        self._ensure_tables()
        row = self._db.execute(
            "SELECT id FROM monitors WHERE id = ?", [monitor_id]
        ).fetchone()
        if row is None:
            return False
        self._db.execute("DELETE FROM monitors WHERE id = ?", [monitor_id])
        self._db.execute("DELETE FROM monitor_runs WHERE monitor_id = ?", [monitor_id])
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
        self._ensure_tables()
        self._db.execute(
            "INSERT INTO monitor_runs (id, monitor_id, data, started_at) VALUES (?, ?, ?, ?)",
            [run.id, run.monitor_id, json.dumps(run.to_dict()), run.started_at],
        )
        # Update monitor's last_run timestamp
        if run.completed_at:
            monitor = self.get(run.monitor_id)
            if monitor:
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
        self._ensure_tables()
        rows = self._db.execute(
            "SELECT data FROM monitor_runs WHERE monitor_id = ? ORDER BY started_at DESC LIMIT ?",
            [monitor_id, limit],
        ).fetchall()
        return [MonitorRun.from_dict(json.loads(row[0])) for row in rows]

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
