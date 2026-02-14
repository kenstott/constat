# Copyright (c) 2025 Kenneth Stott
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

logger = logging.getLogger(__name__)


# noinspection PyUnresolvedReferences
class PlansMixin:

    CONSTAT_BASE_DIR = Path(".constat")
    DEFAULT_USER_ID = "default"

    @classmethod
    def _get_user_plans_file(cls, user_id: str) -> Path:
        """Get path to user-scoped saved plans file."""
        return cls.CONSTAT_BASE_DIR / user_id / "saved_plans.json"

    @classmethod
    def _get_shared_plans_file(cls) -> Path:
        """Get path to shared plans file."""
        return cls.CONSTAT_BASE_DIR / "shared" / "saved_plans.json"

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

        plan_data = {
            "problem": problem,
            "created_by": user_id,
            "steps": [
                {
                    "step_number": e["step_number"],
                    "goal": e["goal"],
                    "code": e["code"],
                }
                for e in entries
            ],
        }

        if shared:
            plans = self._load_shared_plans()
            plans[name] = plan_data
            self._save_shared_plans(plans)
        else:
            plans = self._load_user_plans(user_id)
            plans[name] = plan_data
            self._save_user_plans(user_id, plans)

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

        # Check user's plans first
        user_plans = cls._load_user_plans(user_id)
        if name in user_plans:
            return user_plans[name]

        # Check shared plans
        shared_plans = cls._load_shared_plans()
        if name in shared_plans:
            return shared_plans[name]

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
        result = []

        # User's plans
        user_plans = cls._load_user_plans(user_id)
        for name, data in user_plans.items():
            result.append({
                "name": name,
                "problem": data.get("problem", ""),
                "shared": False,
                "steps": len(data.get("steps", [])),
            })

        # Shared plans
        if include_shared:
            shared_plans = cls._load_shared_plans()
            for name, data in shared_plans.items():
                result.append({
                    "name": name,
                    "problem": data.get("problem", ""),
                    "shared": True,
                    "created_by": data.get("created_by", "unknown"),
                    "steps": len(data.get("steps", [])),
                })

        return result

    @classmethod
    def delete_saved_plan(cls, name: str, user_id: Optional[str] = None) -> bool:
        """Delete a saved plan by name (only user's own plans)."""
        user_id = user_id or cls.DEFAULT_USER_ID
        user_plans = cls._load_user_plans(user_id)

        if name not in user_plans:
            return False

        del user_plans[name]
        cls._save_user_plans(user_id, user_plans)
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

        # Find the plan (check user's plans first, then shared)
        source_plans = cls._load_user_plans(from_user)
        if name in source_plans:
            plan_data = source_plans[name].copy()
        else:
            shared_plans = cls._load_shared_plans()
            if name in shared_plans:
                plan_data = shared_plans[name].copy()
            else:
                return False

        # Copy to target user's plans
        target_plans = cls._load_user_plans(target_user)
        plan_data["shared_by"] = from_user
        target_plans[name] = plan_data
        cls._save_user_plans(target_user, target_plans)
        return True

    @classmethod
    def _load_user_plans(cls, user_id: str) -> dict:
        """Load saved plans for a specific user."""
        plans_file = cls._get_user_plans_file(user_id)
        if not plans_file.exists():
            return {}
        try:
            return json.loads(plans_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Could not load user plans from {plans_file}: {e}")
            return {}

    @classmethod
    def _save_user_plans(cls, user_id: str, plans: dict) -> None:
        """Save plans to user-scoped file."""
        plans_file = cls._get_user_plans_file(user_id)
        plans_file.parent.mkdir(parents=True, exist_ok=True)
        plans_file.write_text(json.dumps(plans, indent=2))

    @classmethod
    def _load_shared_plans(cls) -> dict:
        """Load shared plans accessible to all users."""
        plans_file = cls._get_shared_plans_file()
        if not plans_file.exists():
            return {}
        try:
            return json.loads(plans_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"Could not load shared plans from {plans_file}: {e}")
            return {}

    @classmethod
    def _save_shared_plans(cls, plans: dict) -> None:
        """Save shared plans."""
        plans_file = cls._get_shared_plans_file()
        plans_file.parent.mkdir(parents=True, exist_ok=True)
        plans_file.write_text(json.dumps(plans, indent=2))

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
            "datastore_tables": self.datastore.list_tables(),
            "duration_ms": total_duration,
            "replay": True,
            "plan_name": name,
        }
