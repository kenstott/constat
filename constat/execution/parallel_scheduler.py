"""Parallel step scheduler for exploratory mode execution.

Executes plan steps in parallel based on dependency DAG.
Steps without dependencies run concurrently; dependent steps
wait for their prerequisites to complete.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from constat.core.models import Plan, Step, StepResult, StepStatus


@dataclass
class SchedulerConfig:
    """Configuration for parallel step execution."""
    max_concurrent_steps: int = 5  # Max steps to run in parallel
    step_timeout_seconds: float = 60.0  # Per-step timeout
    fail_fast: bool = True  # Stop on first failure


@dataclass
class SchedulerResult:
    """Result of executing a plan with the parallel scheduler."""
    success: bool
    completed_steps: list[int]
    failed_steps: list[int]
    step_results: dict[int, StepResult]
    execution_waves: list[list[int]]  # Which steps ran in each wave
    total_duration_ms: int = 0


class ParallelStepScheduler:
    """
    Execute plan steps in parallel based on dependency DAG.

    Steps are executed in waves:
    - Wave 1: All steps with no dependencies (run in parallel)
    - Wave 2: Steps whose dependencies completed in wave 1 (run in parallel)
    - Wave 3: etc.

    Within each wave, steps execute concurrently up to max_concurrent_steps.

    Usage:
        scheduler = ParallelStepScheduler(step_executor=my_executor)
        result = await scheduler.execute_plan(plan)

        # Or synchronously:
        result = scheduler.execute_plan_sync(plan)
    """

    def __init__(
        self,
        step_executor: Callable[[Step, dict[str, Any]], StepResult],
        config: Optional[SchedulerConfig] = None,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        """
        Initialize the parallel scheduler.

        Args:
            step_executor: Function that executes a single step.
                          Takes (Step, namespace_dict) -> StepResult
            config: Scheduler configuration
            executor: Thread pool for running sync step executors
        """
        self.step_executor = step_executor
        self.config = config or SchedulerConfig()
        self._executor = executor or ThreadPoolExecutor(max_workers=10)
        self._semaphore: Optional[asyncio.Semaphore] = None

    async def execute_plan(
        self,
        plan: Plan,
        initial_namespace: Optional[dict[str, Any]] = None,
    ) -> SchedulerResult:
        """
        Execute a plan with parallel step scheduling.

        Steps are executed in waves based on dependencies.
        Each wave runs steps concurrently.

        Args:
            plan: The plan to execute
            initial_namespace: Initial variables available to all steps

        Returns:
            SchedulerResult with execution details
        """
        import time

        start_time = time.time()

        # Initialize semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_steps)

        # Shared namespace for step results
        namespace = dict(initial_namespace or {})

        # Track execution
        step_results: dict[int, StepResult] = {}
        execution_waves: list[list[int]] = []
        failed = False

        # Ensure dependencies are set (infer if needed)
        if all(not step.depends_on for step in plan.steps):
            plan.infer_dependencies()

        # Get execution order as waves
        waves = plan.get_execution_order()

        for wave_num, wave_steps in enumerate(waves):
            if failed and self.config.fail_fast:
                break

            execution_waves.append(wave_steps)

            # Execute all steps in this wave in parallel
            tasks = []
            for step_num in wave_steps:
                step = plan.get_step(step_num)
                if step and step.status == StepStatus.PENDING:
                    step.status = StepStatus.RUNNING
                    tasks.append(self._execute_step_async(step, namespace))

            # Wait for all steps in this wave to complete
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for i, result in enumerate(results):
                    step_num = wave_steps[i]
                    step = plan.get_step(step_num)

                    if isinstance(result, Exception):
                        # Step raised an exception
                        step_result = StepResult(
                            success=False,
                            stdout="",
                            error=str(result),
                        )
                        plan.mark_step_failed(step_num, step_result)
                        step_results[step_num] = step_result
                        failed = True
                    elif isinstance(result, StepResult):
                        step_results[step_num] = result
                        if result.success:
                            plan.mark_step_completed(step_num, result)
                            # Merge step outputs into namespace
                            namespace.update(result.variables)
                        else:
                            plan.mark_step_failed(step_num, result)
                            failed = True
                    else:
                        # Unexpected result type
                        step_result = StepResult(
                            success=False,
                            stdout="",
                            error=f"Unexpected result type: {type(result)}",
                        )
                        plan.mark_step_failed(step_num, step_result)
                        step_results[step_num] = step_result
                        failed = True

        total_duration = int((time.time() - start_time) * 1000)

        return SchedulerResult(
            success=not failed and plan.is_complete,
            completed_steps=list(plan.completed_steps),
            failed_steps=list(plan.failed_steps),
            step_results=step_results,
            execution_waves=execution_waves,
            total_duration_ms=total_duration,
        )

    async def _execute_step_async(
        self,
        step: Step,
        namespace: dict[str, Any],
    ) -> StepResult:
        """
        Execute a single step with concurrency control.

        Args:
            step: The step to execute
            namespace: Current namespace with available variables

        Returns:
            StepResult from step execution
        """
        async with self._semaphore:
            try:
                # Apply timeout
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        lambda: self.step_executor(step, namespace)
                    ),
                    timeout=self.config.step_timeout_seconds
                )
                return result
            except asyncio.TimeoutError:
                return StepResult(
                    success=False,
                    stdout="",
                    error=f"Step {step.number} timed out after {self.config.step_timeout_seconds}s",
                )
            except Exception as e:
                return StepResult(
                    success=False,
                    stdout="",
                    error=f"Step {step.number} failed: {e}",
                )

    def execute_plan_sync(
        self,
        plan: Plan,
        initial_namespace: Optional[dict[str, Any]] = None,
    ) -> SchedulerResult:
        """
        Synchronous wrapper for execute_plan.

        Args:
            plan: The plan to execute
            initial_namespace: Initial variables available to all steps

        Returns:
            SchedulerResult with execution details
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - safe to use asyncio.run()
            return asyncio.run(self.execute_plan(plan, initial_namespace))

        # Already in async context - run in separate thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run,
                self.execute_plan(plan, initial_namespace)
            )
            return future.result()


def create_simple_step_executor(
    code_generator: Callable[[Step, dict], str],
    code_executor: Callable[[str, dict], Any],
) -> Callable[[Step, dict], StepResult]:
    """
    Create a simple step executor from code generator and executor functions.

    Args:
        code_generator: Function that generates code for a step
        code_executor: Function that executes the generated code

    Returns:
        Step executor function suitable for ParallelStepScheduler
    """
    import time

    def execute_step(step: Step, namespace: dict) -> StepResult:
        start = time.time()
        try:
            # Generate code for this step
            code = code_generator(step, namespace)
            step.code = code

            # Execute the code
            result = code_executor(code, namespace)

            duration = int((time.time() - start) * 1000)

            if hasattr(result, 'success'):
                # ExecutionResult from PythonExecutor
                return StepResult(
                    success=result.success,
                    stdout=getattr(result, 'stdout', ''),
                    error=getattr(result, 'error_message', lambda: None)() if not result.success else None,
                    duration_ms=duration,
                    variables=getattr(result, 'namespace', {}) or {},
                )
            else:
                # Plain result - assume success
                return StepResult(
                    success=True,
                    stdout=str(result) if result else "",
                    duration_ms=duration,
                    variables=namespace,
                )
        except Exception as e:
            duration = int((time.time() - start) * 1000)
            return StepResult(
                success=False,
                stdout="",
                error=str(e),
                duration_ms=duration,
            )

    return execute_step
