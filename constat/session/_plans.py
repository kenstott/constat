# Copyright (c) 2025 Kenneth Stott
# Canary: 95817e83-dfe3-47f1-a0a4-076b27fce997
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.
"""Plans mixin: saved plans CRUD, sharing, approval, replay."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from constat.core.models import StepResult
from constat.session._types import StepEvent
from constat.storage.duckdb_pool import ThreadLocalDuckDB

logger = logging.getLogger(__name__)

_PLANS_DDL = """
CREATE TABLE IF NOT EXISTS saved_plans (
    name VARCHAR NOT NULL,
    user_id VARCHAR NOT NULL,
    problem TEXT,
    created_by VARCHAR,
    shared_by VARCHAR,
    steps TEXT,
    shared BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (name, user_id)
)
"""


def _open_standalone_db(base_dir: Path, user_id: str) -> ThreadLocalDuckDB:
    """Open a standalone DuckDB connection to the user vault."""
    from constat.core.paths import user_vault_dir, migrate_db_name
    vault = user_vault_dir(base_dir, user_id)
    vault.mkdir(parents=True, exist_ok=True)
    db_path = migrate_db_name(vault, "vectors.duckdb", "user.duckdb")
    return ThreadLocalDuckDB(str(db_path))


def _ensure_plans_table(db: ThreadLocalDuckDB) -> None:
    """Create plans table idempotently."""
    db.execute(_PLANS_DDL)


def _import_json_plans(db: ThreadLocalDuckDB, base_dir: Path, user_id: str) -> None:
    """One-time import from legacy saved_plans.json (user) and shared/saved_plans.json."""
    count = db.execute("SELECT COUNT(*) FROM saved_plans").fetchone()[0]
    if count > 0:
        return

    from constat.core.paths import user_vault_dir

    # Import user plans
    user_file = user_vault_dir(base_dir, user_id) / "saved_plans.json"
    if user_file.exists():
        data = json.loads(user_file.read_text())
        for name, plan in data.items():
            db.execute(
                "INSERT INTO saved_plans (name, user_id, problem, created_by, shared_by, steps, shared) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    name, user_id, plan.get("problem"),
                    plan.get("created_by"), plan.get("shared_by"),
                    json.dumps(plan.get("steps", [])), False,
                ],
            )
        user_file.rename(user_file.with_suffix(".json.imported"))
        logger.info("Imported user plans from %s", user_file)

    # Import shared plans
    shared_file = base_dir / "shared" / "saved_plans.json"
    if shared_file.exists():
        data = json.loads(shared_file.read_text())
        shared_user = "__shared__"
        for name, plan in data.items():
            db.execute(
                "INSERT INTO saved_plans (name, user_id, problem, created_by, shared_by, steps, shared) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (name, user_id) DO NOTHING",
                [
                    name, shared_user, plan.get("problem"),
                    plan.get("created_by"), None,
                    json.dumps(plan.get("steps", [])), True,
                ],
            )
        shared_file.rename(shared_file.with_suffix(".json.imported"))
        logger.info("Imported shared plans from %s", shared_file)


# noinspection PyUnresolvedReferences
class PlansMixin:

    CONSTAT_BASE_DIR = Path(".constat")
    DEFAULT_USER_ID = "default"

    _plans_table_ensured: bool = False

    def _get_plans_db(self) -> ThreadLocalDuckDB:
        """Get DuckDB connection for plans storage."""
        if hasattr(self, "_split_store") and self._split_store is not None:
            return self._split_store.db
        if not hasattr(self, "_plans_standalone_db"):
            user_id = getattr(self, "user_id", self.DEFAULT_USER_ID)
            self._plans_standalone_db = _open_standalone_db(self.CONSTAT_BASE_DIR, user_id)
        return self._plans_standalone_db

    @classmethod
    def _get_plans_db_standalone(cls, user_id: str) -> ThreadLocalDuckDB:
        """Get a standalone DuckDB connection (for classmethod access)."""
        return _open_standalone_db(cls.CONSTAT_BASE_DIR, user_id)

    @classmethod
    def _ensure_plans(cls, db: ThreadLocalDuckDB, user_id: str) -> None:
        """Ensure plans table exists and import legacy data."""
        _ensure_plans_table(db)
        _import_json_plans(db, cls.CONSTAT_BASE_DIR, user_id)

    def save_plan(self, name: str, problem: str, user_id: Optional[str] = None, shared: bool = False) -> None:
        """
        Save the current session's plan and code for future replay.

        Args:
            name: Name for the saved plan
            problem: The original problem (for replay context)
            user_id: User ID (defaults to DEFAULT_USER_ID)
            shared: If True, save as shared plan accessible to all users
        """
        if not self.datastore:
            raise ValueError("No datastore available")

        entries = self.datastore.get_scratchpad()
        if not entries:
            raise ValueError("No steps to save")

        user_id = user_id or self.DEFAULT_USER_ID
        db = self._get_plans_db()
        self._ensure_plans(db, user_id)

        steps = [
            {
                "step_number": e["step_number"],
                "goal": e["goal"],
                "code": e["code"],
            }
            for e in entries
        ]

        target_user = "__shared__" if shared else user_id
        db.execute(
            """INSERT INTO saved_plans (name, user_id, problem, created_by, shared_by, steps, shared)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (name, user_id) DO UPDATE SET
                   problem = excluded.problem,
                   created_by = excluded.created_by,
                   steps = excluded.steps,
                   shared = excluded.shared""",
            [name, target_user, problem, user_id, None, json.dumps(steps), shared],
        )

    @classmethod
    def load_saved_plan(cls, name: str, user_id: Optional[str] = None) -> dict:
        """
        Load a saved plan by name.

        Searches user's plans first, then shared plans.

        Args:
            name: Name of the saved plan
            user_id: User ID (defaults to DEFAULT_USER_ID)

        Returns:
            Dict with problem and steps
        """
        user_id = user_id or cls.DEFAULT_USER_ID
        db = cls._get_plans_db_standalone(user_id)
        cls._ensure_plans(db, user_id)

        # Check user's plans first
        row = db.execute(
            "SELECT problem, created_by, shared_by, steps FROM saved_plans WHERE name = ? AND user_id = ?",
            [name, user_id],
        ).fetchone()
        if row is not None:
            return {
                "problem": row[0],
                "created_by": row[1],
                "shared_by": row[2],
                "steps": json.loads(row[3]) if row[3] else [],
            }

        # Check shared plans
        row = db.execute(
            "SELECT problem, created_by, shared_by, steps FROM saved_plans WHERE name = ? AND user_id = ?",
            [name, "__shared__"],
        ).fetchone()
        if row is not None:
            return {
                "problem": row[0],
                "created_by": row[1],
                "shared_by": row[2],
                "steps": json.loads(row[3]) if row[3] else [],
            }

        raise ValueError(f"No saved plan named '{name}'")

    @classmethod
    def list_saved_plans(cls, user_id: Optional[str] = None, include_shared: bool = True) -> list[dict]:
        """
        List saved plans accessible to the user.

        Args:
            user_id: User ID (defaults to DEFAULT_USER_ID)
            include_shared: Include shared plans in the list

        Returns:
            List of dicts with name, problem, shared flag
        """
        user_id = user_id or cls.DEFAULT_USER_ID
        db = cls._get_plans_db_standalone(user_id)
        cls._ensure_plans(db, user_id)

        result = []

        # User's plans
        rows = db.execute(
            "SELECT name, problem, steps FROM saved_plans WHERE user_id = ? AND shared = FALSE",
            [user_id],
        ).fetchall()
        for name, problem, steps_json in rows:
            steps = json.loads(steps_json) if steps_json else []
            result.append({
                "name": name,
                "problem": problem or "",
                "shared": False,
                "steps": len(steps),
            })

        # Shared plans
        if include_shared:
            rows = db.execute(
                "SELECT name, problem, created_by, steps FROM saved_plans WHERE user_id = ?",
                ["__shared__"],
            ).fetchall()
            for name, problem, created_by, steps_json in rows:
                steps = json.loads(steps_json) if steps_json else []
                result.append({
                    "name": name,
                    "problem": problem or "",
                    "shared": True,
                    "created_by": created_by or "unknown",
                    "steps": len(steps),
                })

        return result

    @classmethod
    def delete_saved_plan(cls, name: str, user_id: Optional[str] = None) -> bool:
        """Delete a saved plan by name (only user's own plans)."""
        user_id = user_id or cls.DEFAULT_USER_ID
        db = cls._get_plans_db_standalone(user_id)
        cls._ensure_plans(db, user_id)

        row = db.execute(
            "SELECT name FROM saved_plans WHERE name = ? AND user_id = ?",
            [name, user_id],
        ).fetchone()
        if row is None:
            return False

        db.execute(
            "DELETE FROM saved_plans WHERE name = ? AND user_id = ?",
            [name, user_id],
        )
        return True

    @classmethod
    def share_plan_with(cls, name: str, target_user: str, from_user: Optional[str] = None) -> bool:
        """
        Share a plan with a specific user (copy to their plans).

        Args:
            name: Name of the plan to share
            target_user: User ID to share with
            from_user: Source user ID (defaults to DEFAULT_USER_ID)

        Returns:
            True if shared successfully
        """
        from_user = from_user or cls.DEFAULT_USER_ID
        db = cls._get_plans_db_standalone(from_user)
        cls._ensure_plans(db, from_user)

        # Find the plan (check user's plans first, then shared)
        row = db.execute(
            "SELECT problem, created_by, steps FROM saved_plans WHERE name = ? AND user_id = ?",
            [name, from_user],
        ).fetchone()
        if row is None:
            row = db.execute(
                "SELECT problem, created_by, steps FROM saved_plans WHERE name = ? AND user_id = ?",
                [name, "__shared__"],
            ).fetchone()
        if row is None:
            return False

        problem, created_by, steps_json = row

        # Copy to target user's plans
        db.execute(
            """INSERT INTO saved_plans (name, user_id, problem, created_by, shared_by, steps, shared)
               VALUES (?, ?, ?, ?, ?, ?, FALSE)
               ON CONFLICT (name, user_id) DO UPDATE SET
                   problem = excluded.problem,
                   created_by = excluded.created_by,
                   shared_by = excluded.shared_by,
                   steps = excluded.steps""",
            [name, target_user, problem, created_by, from_user, steps_json],
        )
        return True

    def replay_saved(self, name: str, user_id: Optional[str] = None) -> dict:
        """
        Replay a saved plan by name.

        Args:
            name: Name of the saved plan
            user_id: User ID for plan lookup (defaults to DEFAULT_USER_ID)

        Returns:
            Dict with results (same format as solve())
        """
        plan_data = self.load_saved_plan(name, user_id=user_id)

        if not self.datastore:
            raise ValueError("No datastore available for replay")

        # Clear existing scratchpad and load saved steps
        # (We'll execute fresh but use stored code)
        problem = plan_data["problem"]
        steps = plan_data["steps"]

        # Emit plan ready
        self._emit_event(StepEvent(
            event_type="plan_ready",
            step_number=0,
            data={
                "steps": [
                    {"number": s["step_number"], "goal": s["goal"], "depends_on": []}
                    for s in steps
                ],
                "reasoning": f"Replaying saved plan: {name}",
                "is_followup": False,
            }
        ))

        # noinspection DuplicatedCode
        all_results = []
        for step_data in steps:
            step_number = step_data["step_number"]
            goal = step_data["goal"]
            code = step_data["code"]

            if not code:
                raise ValueError(f"Step {step_number} has no stored code")

            self._emit_event(StepEvent(
                event_type="step_start",
                step_number=step_number,
                data={"goal": goal}
            ))

            self._emit_event(StepEvent(
                event_type="executing",
                step_number=step_number,
                data={"attempt": 1, "code": code}
            ))

            start_time = time.time()
            tables_before_list = self.datastore.list_tables()
            tables_before = set(t['name'] for t in tables_before_list)
            versions_before = {t['name']: t.get('version', 1) for t in tables_before_list}

            exec_globals = self._get_execution_globals()
            result = self.executor.execute(code, exec_globals)

            if result.success:
                self._auto_save_results(result.namespace, step_number)

            duration_ms = int((time.time() - start_time) * 1000)
            tables_after_list = self.datastore.list_tables()
            tables_after = set(t['name'] for t in tables_after_list)
            versions_after = {t['name']: t.get('version', 1) for t in tables_after_list}
            new_tables = tables_after - tables_before
            updated_tables = {
                name for name in tables_before & tables_after
                if versions_after.get(name, 1) > versions_before.get(name, 1)
                and not name.startswith('_')
            }
            tables_created = list(new_tables | updated_tables)

            if result.success:
                self._emit_event(StepEvent(
                    event_type="step_complete",
                    step_number=step_number,
                    data={
                        "goal": goal,
                        "code": code,
                        "stdout": result.stdout,
                        "attempts": 1,
                        "duration_ms": duration_ms,
                        "tables_created": tables_created,
                    }
                ))

                all_results.append(StepResult(
                    success=True,
                    stdout=result.stdout,
                    attempts=1,
                    duration_ms=duration_ms,
                    tables_created=tables_created,
                    code=code,
                ))
            else:
                self._emit_event(StepEvent(
                    event_type="step_error",
                    step_number=step_number,
                    data={"error": result.stderr or "Execution failed", "attempt": 1}
                ))
                return {
                    "success": False,
                    "error": result.stderr or "Replay execution failed",
                    "step_number": step_number,
                }

        # Synthesize answer (respects insights config)
        combined_output = "\n\n".join([
            f"Step {s['step_number']}: {s['goal']}\n{r.stdout}"
            for s, r in zip(steps, all_results)
        ])

        # Emit raw results first
        self._emit_event(StepEvent(
            event_type="raw_results_ready",
            step_number=0,
            data={"output": combined_output}
        ))

        # Check for created artifacts (to mention in synthesis)
        from constat.visualization.output import peek_pending_outputs
        pending_artifacts = peek_pending_outputs()

        # Check if insights are enabled
        skip_insights = not self.session_config.enable_insights

        if skip_insights:
            final_answer = combined_output
            self._emit_event(StepEvent(
                event_type="answer_ready",
                step_number=0,
                data={"answer": final_answer, "brief": True}
            ))
        else:
            self._emit_event(StepEvent(
                event_type="synthesizing",
                step_number=0,
                data={"message": "Synthesizing final answer..."}
            ))

            final_answer = self._synthesize_answer(problem, combined_output, pending_artifacts)

            self._emit_event(StepEvent(
                event_type="answer_ready",
                step_number=0,
                data={"answer": final_answer}
            ))

        total_duration = sum(r.duration_ms for r in all_results)

        return {
            "success": True,
            "results": all_results,
            "output": combined_output,
            "final_answer": final_answer,
            "brief": skip_insights,
            "datastore_tables": self.datastore.list_tables(),
            "duration_ms": total_duration,
            "replay": True,
            "plan_name": name,
        }
