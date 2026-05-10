# Copyright (c) 2025 Kenneth Stott
# Canary: 48cf7a26-5d5e-4a0f-a43b-ff05308181e0
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

from __future__ import annotations


class PrintProgress:
    """Print-based progress renderer (Phase 1 — no ipywidgets dependency)."""

    def __init__(self, step_offset: int = 0) -> None:
        self._step_offset = step_offset
        self._total_steps = 0
        self._current_step = 0
        self._base_step = 1

    def handle_event(self, event_type: str, data: dict) -> None:
        match event_type:
            case "planning_start":
                print("[planning] Generating plan...")

            case "plan_ready":
                plan = data.get("plan", {})
                if plan and plan.get("steps"):
                    steps = plan["steps"]
                    problem = plan.get("problem", "")
                else:
                    steps = data.get("steps", [])
                    problem = data.get("problem", "")
                self._total_steps = len(steps)
                self._base_step = steps[0].get("number", 1) if steps else 1
                last = self._base_step + self._total_steps - 1
                print(f"[plan] {problem} (steps {self._base_step}\u2013{last})")
                for i, s in enumerate(steps):
                    num = self._base_step + i
                    print(f"  {num}. {s.get('goal', '')}")

            case "step_start":
                n = data.get("step_number", 0)
                goal = data.get("goal", "")
                self._current_step = n
                total = self._base_step + self._total_steps - 1
                print(f"[step {n}/{total}] {goal}...")

            case "step_generating":
                n = data.get("step_number", self._current_step)
                attempt = data.get("attempt", 1)
                if data.get("is_retry"):
                    print(f"  step {n} generating (retry #{attempt - 1})...")
                else:
                    print(f"  step {n} generating...")

            case "step_executing":
                n = data.get("step_number", self._current_step)
                attempt = data.get("attempt", 1)
                if data.get("is_retry"):
                    print(f"  step {n} executing (retry #{attempt - 1})...")
                else:
                    print(f"  step {n} executing...")

            case "model_escalation":
                n = data.get("step_number", self._current_step)
                to_model = data.get("to_model", "")
                reason = data.get("reason", "")
                print(f"  step {n} escalated to {to_model}: {reason}")

            case "step_complete":
                d = data.get("duration_ms")
                n = data.get("step_number", self._current_step)
                print(f"  step {n} done ({d / 1000:.1f}s)" if d else f"  step {n} done")

            case "step_error":
                n = data.get("step_number", self._current_step)
                attempt = data.get("attempt", 1)
                max_attempts = data.get("max_attempts", "?")
                error = (data.get("error", "") or "").split("\n")[0][:80]
                print(f"  step {n} error (attempt {attempt}/{max_attempts}): {error}")

            case "step_failed":
                n = data.get("step_number", self._current_step)
                attempts = data.get("attempts", "?")
                print(f"  step {n} failed after {attempts} attempts")

            case "synthesizing":
                print("[synthesizing] Generating answer...")

            # Reasoning chain events
            case "proof_start":
                print("[chain] Reasoning chain starting...")

            case "fact_start":
                print(f"  [premise] {data.get('name', '')} resolving...")

            case "fact_resolved":
                print(f"  [premise] {data.get('name', '')} resolved")

            case "fact_failed":
                print(f"  [premise] {data.get('name', '')} FAILED")

            case "dag_execution_start":
                print("[chain] Executing inferences...")

            case "inference_executing":
                print(f"  [inference] {data.get('name', '')} executing...")

            case "inference_complete":
                print(f"  [inference] {data.get('name', '')} complete")

            case "proof_complete":
                print("[chain] Complete")

            case "query_complete":
                print("[done]")

            case "query_error":
                error = data.get("error", "")
                # Show just the first line of the error
                first_line = error.split("\n")[0][:120] if error else "Unknown error"
                print(f"[error] {first_line}")

            case "clarification_needed":
                pass  # Handled by input() prompt in client
