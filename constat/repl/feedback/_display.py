# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""FeedbackDisplay — step execution UI and output display."""

from __future__ import annotations

import re
import sys
import time
from typing import Optional

from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from constat.execution.mode import (
    PlanApprovalRequest,
    PlanApprovalResponse,
)
from constat.repl.feedback._display_core import FeedbackDisplayCore
from constat.repl.feedback._models import _left_align_markdown, StepDisplay
from constat.session import ClarificationRequest, ClarificationResponse, ClarificationQuestion


class FeedbackDisplay(FeedbackDisplayCore):
    """FeedbackDisplay with step execution UI and output display methods."""

    def _get_step(self, step_number: int) -> Optional[StepDisplay]:
        """Get a step by number."""
        for step in self.plan_steps:
            if step.number == step_number:
                return step
        return None

    def reset(self) -> None:
        """Reset all display state for a fresh start."""
        self.plan_steps = []
        self.current_step = None
        self.problem = ""
        self._step_number_map = {}
        self._execution_started = False
        self._start_time = None
        self._output_lines = []
        self._completed_outputs = []
        self._active_step_goal = ""
        # Reset status bar counts
        self._status_bar._tables_count = 0
        self._status_bar._facts_count = 0
        self._status_bar.reset()
        self.stop()

    def show_user_input(self, user_input: str) -> None:
        """Display user input with YOU header (right-aligned)."""
        self.console.print()
        self.console.print(Rule("[bold green]YOU[/bold green]", align="right"))
        self.console.print(f"[white]{user_input}[/white]")

    def set_problem(self, problem: str) -> None:
        """Set the problem being solved."""
        self.problem = problem
        self.console.print()  # Blank line before plan

    def show_plan(self, steps: list[dict], is_followup: bool = False) -> None:
        """Display the execution plan.

        Args:
            steps: List of step dicts with number, goal, depends_on, type (optional)
            is_followup: If True, continue step numbering from previous plan
        """
        # Determine starting step number
        if is_followup and self.plan_steps:
            # Continue from last step number
            start_num = max(self._step_number_map.values()) + 1 if self._step_number_map else 1
        else:
            # New problem - start from 1
            start_num = 1
            self._step_number_map = {}
            self.plan_steps = []

        # Build mapping from session step numbers to display numbers
        for i, s in enumerate(steps):
            session_num = s.get("number", i + 1)
            self._step_number_map[session_num] = start_num + i

        # Append new steps (for follow-up) or replace (for new problem)
        new_steps = [
            StepDisplay(number=s.get("number", i+1), goal=s.get("goal", ""))
            for i, s in enumerate(steps)
        ]
        if is_followup:
            self.plan_steps.extend(new_steps)
        else:
            self.plan_steps = new_steps

        # Always show the plan so user knows what's coming
        self.console.print(Rule("[bold cyan]CONSTAT[/bold cyan]", align="left"))

        # Check if this is an auditable proof structure (has type field)
        is_proof_structure = any(s.get("type") in ("premise", "inference", "conclusion") for s in steps)
        current_section = None

        for i, s in enumerate(steps):
            display_num = self._step_number_map.get(s.get("number", i + 1), start_num + i)
            goal = s.get("goal", "")
            depends_on = s.get("depends_on", [])
            step_type = s.get("type")

            # Show section headers for proof structure
            if is_proof_structure and step_type != current_section:
                current_section = step_type
                if step_type == "premise":
                    self.console.print("\n  [bold yellow]PREMISES[/bold yellow] [dim](facts to retrieve from sources)[/dim]")
                elif step_type == "inference":
                    self.console.print("\n  [bold yellow]INFERENCES[/bold yellow] [dim](facts derived from premises)[/dim]")
                elif step_type == "conclusion":
                    # Show data flow DAG before conclusion
                    self._show_data_flow_dag(steps)
                    self.console.print("\n  [bold yellow]CONCLUSION[/bold yellow]")

            # Format dependency info with remapped step numbers
            dep_str = ""
            if depends_on and not is_proof_structure:
                # Only show depends_on for non-proof plans (proof structure is implicit)
                remapped_deps = [str(self._step_number_map.get(d, d)) for d in depends_on]
                dep_str = f" [dim](depends on {', '.join(remapped_deps)})[/dim]"

            # Use fact_id (P1, I1, C) for proof structures, numeric for regular plans
            if is_proof_structure:
                fact_id = s.get("fact_id", "")
                if step_type == "conclusion":
                    fact_id = "C"
                self.console.print(f"  [dim]{fact_id}:[/dim] {goal}")
            else:
                self.console.print(f"  [dim]{display_num}.[/dim] {goal}{dep_str}")

        self.console.print()

    def _show_data_flow_dag(self, steps: list[dict]) -> None:
        """Display an ASCII data flow DAG with proper box-drawing characters.

        Args:
            steps: List of proof steps with type, fact_id, goal, etc.
        """
        try:
            from constat.visualization.box_dag import generate_proof_dfd
            diagram = generate_proof_dfd(steps, max_width=60, max_name_len=10)
            if diagram and diagram != "(No derivation graph available)":
                self.console.print("\n  [bold yellow]DATA FLOW[/bold yellow]")
                for line in diagram.split('\n'):
                    if line.strip():
                        self.console.print(f"      [dim]{line}[/dim]")
                self.console.print()
        except Exception:
            pass  # Skip diagram on error

    def start_execution(self) -> None:
        """Start the live execution display."""
        self.console.print(Rule("[bold cyan]CONSTAT[/bold cyan]", align="left"))
        self._execution_started = True
        self._start_time = time.time()  # Start timing
        self._completed_outputs = []  # Clear completed outputs buffer
        self._active_step_goal = ""
        if self._use_live_display:
            self.start()
            self._start_animation_thread()  # Start background animation
            self._update_live()

    def request_plan_approval(self, request: PlanApprovalRequest) -> PlanApprovalResponse:
        """
        Request user approval for a generated plan.

        Displays the plan and prompts for approval.
        Returns user's decision with optional feedback.

        Approval flow (simplified):
        - approve: Execute the plan as-is
        - reject: Cancel execution
        - suggest: User provides feedback for replanning

        Args:
            request: PlanApprovalRequest with full context

        Returns:
            PlanApprovalResponse with user's decision
        """
        # Stop any running animation/spinner before prompting for input
        self.stop_spinner()

        # Show the plan
        self.show_plan(request.steps)

        # Show reasoning if available
        if request.reasoning:
            self.console.print("[bold]Reasoning:[/bold]")
            self.console.print(Panel(request.reasoning, border_style="dim"))

        # Prompt for approval - simplified to approve/reject/suggest
        # Mode switching should be done via /proof or /explore commands
        self.console.print()
        self.console.print("[dim]Enter to execute, 'n' to cancel, or type changes[/dim]")

        while True:
            try:
                response = self.prompt_with_status("> ")
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[red]Cancelled.[/red]")
                return PlanApprovalResponse.reject("User cancelled")

            # Empty or affirmative = approve
            if not response or response.lower() in ("y", "yes", "ok", "go", "execute"):
                # Don't print "Executing..." - proof tree display shows "Resolving proof..."
                return PlanApprovalResponse.approve()

            # Reject
            elif response.lower() in ("n", "no", "cancel", "stop"):
                self.console.print("[red]Plan rejected.[/red]\n")
                return PlanApprovalResponse.reject()

            # Slash commands - pass through to REPL for global handling
            # This includes /proof and /explore for mode switching
            elif response.startswith("/"):
                return PlanApprovalResponse.pass_command(response)

            # Anything else is steering feedback (suggest)
            else:
                self.console.print("[yellow]Incorporating feedback and replanning...[/yellow]\n")
                return PlanApprovalResponse.suggest(response)

    def request_clarification(self, request: ClarificationRequest) -> ClarificationResponse:
        """
        Request clarification from user for ambiguous questions.

        Displays questions with numbered suggestions. Users can:
        - Enter a number to select a suggestion
        - Type a custom answer
        - Press Enter to skip

        Args:
            request: ClarificationRequest with questions

        Returns:
            ClarificationResponse with user's answers
        """
        # Stop any running animation before prompting for input
        self.stop_spinner()

        self.console.print()
        self.console.print(Rule("[bold cyan]Clarification Needed[/bold cyan]", align="left"))

        if request.ambiguity_reason:
            self.console.print(f"[dim]{request.ambiguity_reason}[/dim]")
            self.console.print()

        self.console.print("[bold]Please clarify[/bold] [dim](enter number, custom text, or press Enter to skip):[/dim]")
        self.console.print()

        answers = {}
        for i, question in enumerate(request.questions, 1):
            # Handle both old format (str) and new format (ClarificationQuestion)
            if isinstance(question, ClarificationQuestion):
                question_text = question.text
                suggestions = question.suggestions
            else:
                question_text = str(question)
                suggestions = []

            self.console.print(f"  [cyan]{i}.[/cyan] {question_text}")

            # Show numbered suggestions if available
            if suggestions:
                for j, suggestion in enumerate(suggestions, 1):
                    self.console.print(f"      [dim]{j})[/dim] [yellow]{suggestion}[/yellow]")

            # Flush all output to ensure suggestions are visible before prompt
            sys.stdout.flush()
            sys.stderr.flush()
            # Force Rich console to flush as well
            self.console.file.flush() if hasattr(self.console, 'file') else None

            # Get answer with status bar - use prompt_toolkit
            default_hint = suggestions[0] if suggestions else ""
            prompt_str = f"     > ({default_hint}): " if default_hint else "     > "
            answer = self.prompt_with_status(prompt_str, default=default_hint)

            # Process the answer
            if answer.lower() == "skip" or not answer:
                # When skipping with no suggestions, use empty
                answers[question_text] = ""
                self.console.print(f"     [dim]Skipped (will use defaults)[/dim]")
            elif suggestions and answer == suggestions[0]:
                # User accepted the default suggestion
                answers[question_text] = answer
                self.console.print(f"     [green]Using: {answer}[/green]")
            elif answer.isdigit() and suggestions:
                # User selected a suggestion by number
                idx = int(answer) - 1
                if 0 <= idx < len(suggestions):
                    answers[question_text] = suggestions[idx]
                    self.console.print(f"     [green]Selected: {suggestions[idx]}[/green]")
                else:
                    # Invalid number, treat as custom input
                    answers[question_text] = answer
            else:
                answers[question_text] = answer

            self.console.print()

        # Filter out empty answers
        non_empty_answers = {q: a for q, a in answers.items() if a}
        all_answered = len(non_empty_answers) == len(request.questions)

        # Show summary of user's clarifications with YOU header
        if non_empty_answers:
            self.console.print()
            self.console.print(Rule("[bold green]YOU[/bold green]", align="right"))
            for q, a in non_empty_answers.items():
                # Shorten question for display
                short_q = q[:50] + "..." if len(q) > 50 else q
                self.console.print(f"[dim]{short_q}:[/dim] [white]{a}[/white]")
            self.console.print()

        if all_answered:
            # All questions answered - proceed automatically
            return ClarificationResponse(answers=non_empty_answers, skip=False)
        elif non_empty_answers:
            # Some questions skipped - ask if they want to continue
            self.console.print("[yellow]Some clarifications skipped.[/yellow] [dim]Press Enter to continue anyway, or 's' to cancel[/dim]")
            skip = self.prompt_with_status("> ").lower()
            if skip == "s":
                self.console.print("[dim]Cancelled.[/dim]")
                return ClarificationResponse(answers={}, skip=True)
            return ClarificationResponse(answers=non_empty_answers, skip=False)
        else:
            # No answers provided at all
            self.console.print("[yellow]No clarifications provided.[/yellow] [dim]Press Enter to try anyway, or 's' to cancel[/dim]")
            skip = self.prompt_with_status("> ").lower()
            if skip == "s":
                self.console.print("[dim]Cancelled.[/dim]")
                return ClarificationResponse(answers={}, skip=True)
            self.console.print("[dim]Proceeding with original question...[/dim]\n")
            return ClarificationResponse(answers={}, skip=False)

    def show_replan_notice(self, attempt: int, max_attempts: int) -> None:
        """Show notice that we're replanning based on feedback."""
        self.console.print(
            f"[yellow]Replanning (attempt {attempt}/{max_attempts})...[/yellow]"
        )

    def step_start(self, step_number: int, goal: str) -> None:
        """Mark a step as starting."""
        self.current_step = step_number
        self._active_step_goal = goal  # For animated header

        # Update step status
        step = self._get_step(step_number)
        if step:
            step.status = "running"
            step.status_message = "starting..."

        # Ensure live display is initialized
        self._ensure_live()
        self._update_live()

    def step_generating(self, step_number: int, attempt: int) -> None:
        """Show code generation in progress."""
        step = self._get_step(step_number)
        if step:
            step.status = "generating"
            step.status_message = f"retry #{attempt}..." if attempt > 1 else "working..."
            step.attempts = attempt

        # Ensure live display is initialized
        self._ensure_live()
        self._update_live()

    def step_executing(self, step_number: int, attempt: int, code: Optional[str] = None) -> None:
        """Show code execution in progress."""
        step = self._get_step(step_number)
        if step:
            step.status = "executing"
            step.status_message = "executing..."
            step.code = code

        # Ensure live display is initialized
        self._ensure_live()
        self._update_live()

        # In verbose mode, also show the code being executed
        if self.verbose and code:
            self.console.print()
            self.console.print(Syntax(code, "python", theme="monokai", line_numbers=True))

    def step_complete(
        self,
        step_number: int,
        output: str,
        attempts: int,
        duration_ms: int,
        tables_created: Optional[list[str]] = None,
    ) -> None:
        """Mark a step as completed successfully."""
        # Accumulate completed step output for animated display
        display_num = self._step_number_map.get(step_number, step_number)
        if output:
            self._completed_outputs.append((display_num, output.strip()))

        # Build output summary
        output_summary = ""
        if output:
            lines = [l.strip() for l in output.strip().split("\n") if l.strip()]
            if lines:
                # Take first 2 lines max as summary for Live display
                summary_lines = lines[:2]
                output_summary = " | ".join(summary_lines)
                if len(output_summary) > 80:
                    output_summary = output_summary[:77] + "..."

        step = self._get_step(step_number)
        if step:
            step.status = "completed"
            step.output = output
            step.output_summary = output_summary
            step.attempts = attempts
            step.duration_ms = duration_ms
            step.tables_created = tables_created or []

            # Update status bar tables count
            if tables_created:
                self._status_bar._tables_count += len(tables_created)

        # Clear active step goal after completion
        self._active_step_goal = ""

        # Ensure live display is initialized and update
        self._ensure_live()
        self._update_live()

        # In verbose mode, show tables that were created
        if self.verbose and tables_created:
            self.console.print(f"  [dim]tables:[/dim] {', '.join(tables_created)}")

    def step_error(self, step_number: int, error: str, attempt: int) -> None:
        """Show a step error (before retry)."""
        error_lines = error.strip().split("\n")
        brief = error_lines[-1] if error_lines else error

        step = self._get_step(step_number)
        if step:
            step.status_message = f"retry #{attempt}... ({brief[:40]})"

        # Ensure live display is initialized and update
        self._ensure_live()
        self._update_live()

        # In verbose mode, show the full error details
        if self.verbose:
            self.console.print(f"  [red]Error:[/red] {brief[:80]}")
            self.console.print(Panel(error, title="Full Error", border_style="red"))

    def step_failed(
        self,
        step_number: int,
        error: str,
        attempts: int,
        suggestions: Optional[list] = None
    ) -> None:
        """Mark a step as permanently failed and show suggestions.

        Args:
            step_number: The step that failed
            error: Error message
            attempts: Number of attempts made
            suggestions: List of FailureSuggestion objects for alternative approaches
        """
        step = self._get_step(step_number)
        if step:
            step.status = "failed"
            step.error = error
            step.attempts = attempts

        # Ensure live display is initialized and update
        self._ensure_live()
        self._update_live()

        # Show suggestions if available
        if suggestions:
            self._show_failure_suggestions(suggestions)

    def _show_failure_suggestions(self, suggestions: list) -> None:
        """Display failure recovery suggestions to the user.

        Args:
            suggestions: List of FailureSuggestion objects
        """
        from rich.table import Table

        self.console.print()
        self.console.print("[bold yellow]Alternative approaches:[/bold yellow]")

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Option", style="cyan", width=3)
        table.add_column("Label", style="bold")
        table.add_column("Description", style="dim")

        for i, suggestion in enumerate(suggestions, 1):
            label = getattr(suggestion, 'label', str(suggestion))
            description = getattr(suggestion, 'description', '')
            table.add_row(f"{i}.", label, description)

        self.console.print(table)
        self.console.print()
        self.console.print("[dim]Enter a number to try that approach, or type your own suggestion[/dim]")

    def show_summary(self, success: bool, total_steps: int, duration_ms: int) -> None:
        """Show final execution summary."""
        # Stop the Live display before printing summary
        self.stop()

        if not success:
            completed = sum(1 for s in self.plan_steps if s.status == "completed")
            self.console.print(
                f"\n[bold red]FAILED[/bold red] "
                f"({completed}/{total_steps} steps completed)"
            )
        # Success case: timing shown after tables hint

    def show_tables(self, tables: list[dict], duration_ms: int = 0, force_show: bool = False) -> None:
        """Show available tables in the datastore."""
        # Update status bar with current table count
        self._status_bar._tables_count = len(tables)
        if not tables:
            if duration_ms:
                self.console.print(f"\n[dim]({duration_ms/1000:.1f}s total)[/dim]")
            return

        if self.verbose or force_show:
            self.console.print("\n[bold]Available Tables:[/bold]")
            table = Table(show_header=True, box=None)
            table.add_column("Name", style="cyan")
            table.add_column("Rows", justify="right")
            table.add_column("From Step", justify="right")

            for t in tables:
                table.add_row(t["name"], str(t["row_count"]), str(t["step_number"]))

            self.console.print(table)
            if duration_ms:
                self.console.print(f"[dim]({duration_ms/1000:.1f}s total)[/dim]")
        else:
            # Compact: tables hint with timing
            time_str = f", {duration_ms/1000:.1f}s total" if duration_ms else ""
            self.console.print(f"\n[dim]({len(tables)} tables available - use /tables to view{time_str})[/dim]")

    def show_output(self, output: str) -> None:
        """Show final output."""
        # Stop any running spinner first
        self.stop_spinner()
        # Clean up output: strip trailing whitespace and collapse multiple blank lines
        cleaned = re.sub(r'\n{3,}', '\n\n', output.rstrip())
        self.console.print()
        self.console.print(Markdown(_left_align_markdown(cleaned)))
        self.console.print()  # One blank line after output

    def show_final_answer(self, answer: str) -> None:
        """Show the final synthesized answer from Vera."""
        self.console.print()
        self.console.print(Rule("[bold blue]VERA[/bold blue]", align="left"))
        self.console.print(Markdown(_left_align_markdown(answer)))
        self.console.print()

    def show_suggestions(self, suggestions: list[str]) -> None:
        """Show follow-up suggestions."""
        if not suggestions:
            return

        # No extra blank line - caller (show_output) handles spacing
        if len(suggestions) == 1:
            self.console.print(f"[dim]Suggestion:[/dim] [cyan]{suggestions[0]}[/cyan]")
        else:
            self.console.print("[dim]Suggestions:[/dim]")
            for i, s in enumerate(suggestions, 1):
                self.console.print(f"  [dim]{i}.[/dim] [cyan]{s}[/cyan]")

    def show_facts_extracted(self, facts: list[dict], source: str) -> None:
        """Show facts that were extracted and cached.

        Args:
            facts: List of fact dicts with 'name' and 'value' keys
            source: Where facts came from ('question' or 'response')
        """
        if not facts:
            return

        # Only show response-derived facts (question facts are implicit)
        if source == "response":
            fact_strs = [f"[cyan]{f['name']}[/cyan]={f['value']}" for f in facts[:5]]
            self.console.print(f"[dim]Remembered: {', '.join(fact_strs)}[/dim]")

    def request_failure_recovery(self, error_message: str, step_info: str = "") -> str:
        """
        Request user decision on how to handle execution failure.

        Displays the failure context and prompts for recovery action:
        - retry: Re-run the failed step (for transient/probabilistic failures)
        - replan: Return to planning with failure context
        - abandon: Return to idle state

        Args:
            error_message: The error that caused the failure
            step_info: Optional context about which step failed

        Returns:
            One of: "retry", "replan", "abandon"
        """
        # Stop any running animation/spinner before prompting
        self.stop_spinner()
        self.stop_live_plan_display()

        self.console.print()
        self.console.print(Rule("[bold red]Execution Failed[/bold red]", align="left"))

        # Show step context if available
        if step_info:
            self.console.print(f"[dim]Step: {step_info}[/dim]")

        # Show error message (truncated if very long)
        error_display = error_message
        if len(error_display) > 200:
            error_display = error_display[:197] + "..."
        self.console.print(f"[red]Error:[/red] {error_display}")

        self.console.print()
        self.console.print("[bold]What would you like to do?[/bold]")
        self.console.print("  [cyan]1.[/cyan] [bold]retry[/bold]   - Try again (transient errors)")
        self.console.print("  [cyan]2.[/cyan] [bold]replan[/bold]  - Modify the plan")
        self.console.print("  [cyan]3.[/cyan] [bold]abandon[/bold] - Start over")
        self.console.print()
        self.console.print("[dim]Enter choice (1/2/3 or retry/replan/abandon)[/dim]")

        while True:
            try:
                response = self.prompt_with_status("> ").lower()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]Abandoning...[/dim]")
                return "abandon"

            # Parse response
            if response in ("1", "retry", "r", "try", "again"):
                self.console.print("[yellow]Retrying...[/yellow]\n")
                return "retry"
            elif response in ("2", "replan", "modify", "change", "m"):
                self.console.print("[yellow]Returning to planning...[/yellow]\n")
                return "replan"
            elif response in ("3", "abandon", "cancel", "quit", "a", "q", ""):
                self.console.print("[dim]Abandoned.[/dim]\n")
                return "abandon"
            else:
                self.console.print("[dim]Please enter 1/2/3 or retry/replan/abandon[/dim]")
