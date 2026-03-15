from __future__ import annotations


class PrintProgress:
    """Print-based progress renderer (Phase 1 — no ipywidgets dependency)."""

    def __init__(self) -> None:
        self._total_steps = 0
        self._current_step = 0

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
                print(f"[plan] {problem} ({self._total_steps} steps)")
                for s in steps:
                    print(f"  {s.get('number', '?')}. {s.get('goal', '')}")

            case "step_start":
                n = data.get("step_number", 0)
                goal = data.get("goal", "")
                self._current_step = n
                print(f"[step {n}/{self._total_steps}] {goal}...")

            case "step_generating" | "step_executing":
                pass

            case "step_complete":
                d = data.get("duration_ms")
                n = data.get("step_number", self._current_step)
                print(f"  step {n} done ({d / 1000:.1f}s)" if d else f"  step {n} done")

            case "step_error" | "step_failed":
                print(f"  step {self._current_step} retrying...")

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
