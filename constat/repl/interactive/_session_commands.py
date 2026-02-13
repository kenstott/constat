# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Session commands mixin — display toggles, context, reset, history, plans, audit, prove."""

from rich.panel import Panel
from rich.table import Table

from constat.session import Session


class _SessionCommandsMixin:
    """Session-related REPL commands: toggles, context, reset, redo, history, plans, audit, prove, summarize."""

    def _apply_display_overrides(self, overrides) -> dict:
        """Apply display overrides and return original values for restoration."""
        original = {}

        for setting, value in overrides.persistent.items():
            if setting == "verbose":
                self.verbose = value
                self.display.verbose = value
                self.console.print(f"[dim]Verbose: {'on' if value else 'off'} (persistent)[/dim]")
            elif setting == "raw":
                self.session_config.show_raw_output = value
                self.console.print(f"[dim]Raw output: {'on' if value else 'off'} (persistent)[/dim]")
            elif setting == "insights":
                self.session_config.enable_insights = value
                self.console.print(f"[dim]Insights: {'on' if value else 'off'} (persistent)[/dim]")

        for setting, value in overrides.single_turn.items():
            if setting in overrides.persistent:
                continue

            if setting == "verbose":
                original["verbose"] = self.verbose
                self.verbose = value
                self.display.verbose = value
            elif setting == "raw":
                original["raw"] = self.session_config.show_raw_output
                self.session_config.show_raw_output = value
            elif setting == "insights":
                original["insights"] = self.session_config.enable_insights
                self.session_config.enable_insights = value

        return original

    def _restore_display_settings(self, original: dict) -> None:
        """Restore display settings after single-turn override."""
        if "verbose" in original:
            self.verbose = original["verbose"]
            self.display.verbose = original["verbose"]
        if "raw" in original:
            self.session_config.show_raw_output = original["raw"]
        if "insights" in original:
            self.session_config.enable_insights = original["insights"]

    def _toggle_verbose(self, arg: str = "") -> None:
        """Toggle or set verbose mode on/off."""
        arg_lower = arg.lower().strip()
        if arg_lower == "on":
            self.verbose = True
        elif arg_lower == "off":
            self.verbose = False
        else:
            self.verbose = not self.verbose

        self.display.verbose = self.verbose
        status = "on" if self.verbose else "off"
        self.console.print(f"Verbose: [bold]{status}[/bold]")
        if self.verbose:
            self.console.print("[dim]Will show detailed step execution info[/dim]")

    def _toggle_raw(self, arg: str = "") -> None:
        """Toggle or set raw output display on/off."""
        arg_lower = arg.lower().strip()
        if arg_lower == "on":
            self.session_config.show_raw_output = True
        elif arg_lower == "off":
            self.session_config.show_raw_output = False
        else:
            self.session_config.show_raw_output = not self.session_config.show_raw_output

        status = "on" if self.session_config.show_raw_output else "off"
        self.console.print(f"Raw output: [bold]{status}[/bold]")
        if self.session_config.show_raw_output:
            self.console.print("[dim]Raw step results will be shown before synthesis[/dim]")
        else:
            self.console.print("[dim]Only synthesized answer will be shown[/dim]")

    def _toggle_insights(self, arg: str = "") -> None:
        """Toggle or set insight synthesis on/off."""
        arg_lower = arg.lower().strip()
        if arg_lower == "on":
            self.session_config.enable_insights = True
        elif arg_lower == "off":
            self.session_config.enable_insights = False
        else:
            self.session_config.enable_insights = not self.session_config.enable_insights

        status = "on" if self.session_config.enable_insights else "off"
        self.console.print(f"Insights: [bold]{status}[/bold]")
        if not self.session_config.enable_insights:
            self.console.print("[dim]Raw results will be shown without synthesis[/dim]")

    def _show_preferences(self) -> None:
        """Show current preferences/settings."""
        table = Table(title="Preferences", show_header=True, box=None)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("verbose", "on" if self.verbose else "off")
        table.add_row("raw", "on" if self.session_config.show_raw_output else "off")
        table.add_row("insights", "on" if self.session_config.enable_insights else "off")
        table.add_row("user", self.user_id)

        default_mode = self.session_config.default_mode
        mode_display = default_mode.value if default_mode else "auto (LLM decides)"
        table.add_row("default_mode", mode_display)

        self.console.print(table)

    def _show_user(self, name: str = "") -> None:
        """Show or set current user."""
        if name:
            self.user_id = name
            self.console.print(f"User set to: [bold]{self.user_id}[/bold]")
        else:
            self.console.print(f"Current user: [bold]{self.user_id}[/bold]")

    def _show_context(self) -> None:
        """Show context size and token usage statistics."""
        if not self.api.session:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        stats = self.api.session.get_context_stats()
        if not stats:
            self.console.print("[yellow]No datastore available.[/yellow]")
            return

        from rich.panel import Panel as RichPanel

        if stats.is_critical:
            title_style = "bold red"
            status = "CRITICAL - Consider using /compact"
        elif stats.is_warning:
            title_style = "bold yellow"
            status = "WARNING - Context growing large"
        else:
            title_style = "bold green"
            status = "OK"

        content = stats.summary()
        content += f"\n\nStatus: {status}"

        self.console.print(RichPanel(
            content,
            title="[bold]Context Size[/bold]",
            title_align="left",
            border_style=title_style,
        ))

    def _compact_context(self) -> None:
        """Compact context to reduce token usage."""
        if not self.api.session:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        stats_before = self.api.session.get_context_stats()
        if not stats_before:
            self.console.print("[yellow]No datastore available.[/yellow]")
            return

        self.console.print(f"[dim]Before compaction: ~{stats_before.total_tokens:,} tokens[/dim]")

        self.display.start_spinner("Compacting context...")
        try:
            result = self.api.session.compact_context(
                summarize_scratchpad=True,
                sample_tables=True,
                clear_old_state=False,
                keep_recent_steps=3,
            )
            self.display.stop_spinner()

            if result:
                self.console.print(f"[green]{result.message}[/green]")
                self.console.print(result.summary())
            else:
                self.console.print("[yellow]Compaction returned no result.[/yellow]")

        except Exception as e:
            self.display.stop_spinner()
            self.console.print(f"[red]Error during compaction:[/red] {e}")

    def _reset_session(self) -> None:
        """Reset session state and start a new session."""
        self.display.reset()

        self.api = self._create_api(new_session=True)

        self.last_problem = ""
        self.suggestions = []

        session_id = self.api.session.session_id
        self.console.print(f"[green]New session: {session_id[:8]}...[/green]")

    def _handle_redo(self, arg: str) -> None:
        """Handle /redo command - retry last query with optional modifications."""
        if not self.last_problem:
            self.console.print("[yellow]No previous query to redo.[/yellow]")
            return

        if not self.api.session or not self.api.session.session_id:
            self.console.print("[yellow]No active session. Use the original query.[/yellow]")
            return

        if arg:
            redo_query = f"redo. {arg}"
            self.console.print(f"[dim]Retrying with: {arg}[/dim]")
        else:
            redo_query = "redo"
            self.console.print("[dim]Retrying last query...[/dim]")

        self._solve(redo_query)

    def _resume_session(self, session_id: str) -> None:
        """Resume a previous session."""
        if not self.api.session:
            self.api = self._create_api()

        sessions = self.api.session.history.list_sessions(limit=50)
        match = None
        for s in sessions:
            if s.session_id.startswith(session_id) or session_id in s.session_id:
                match = s.session_id
                break

        if not match:
            self.console.print(f"[red]Session not found: {session_id}[/red]")
            return

        if self.api.session.resume(match):
            self.console.print(f"[green]Resumed session:[/green] {match[:30]}...")
            tables = self.api.session.datastore.list_tables() if self.api.session.datastore else []
            if tables:
                self.console.print(f"[dim]{len(tables)} tables available - use /tables to view[/dim]")
        else:
            self.console.print(f"[red]Failed to resume session: {match}[/red]")

    def _handle_auto_resume(self) -> None:
        """Handle auto-resume from --continue flag."""
        if not self.api.session:
            self.api = self._create_api()

        sessions = self.api.session.history.list_sessions(limit=1)
        if not sessions:
            self.console.print("[dim]No previous session to resume.[/dim]")
            return

        latest = sessions[0]
        if self.api.session.resume(latest.session_id):
            self.console.print(f"[green]Resumed last session:[/green] {latest.session_id[:30]}...")
            if latest.summary:
                self.console.print(f"[dim]{latest.summary}[/dim]")
            tables = self.api.session.datastore.list_tables() if self.api.session.datastore else []
            if tables:
                self.console.print(f"[dim]{len(tables)} tables available - use /tables to view[/dim]")
            self.console.print()
        else:
            self.console.print(f"[yellow]Could not resume session {latest.session_id[:20]}...[/yellow]")

    def _show_history(self) -> None:
        """Show recent session history for current user."""
        if not self.api.session:
            self.api = self._create_api()

        sessions = self.api.session.history.list_sessions(limit=10)
        if not sessions:
            self.console.print("[dim]No session history.[/dim]")
            return

        table = Table(title=f"Sessions for {self.user_id}", show_header=True, box=None)
        table.add_column("ID", style="cyan")
        table.add_column("Started")
        table.add_column("Summary")
        table.add_column("Queries", justify="right")
        table.add_column("Status")

        for s in sessions:
            short_id = s.session_id[:20] + "..." if len(s.session_id) > 20 else s.session_id
            started = s.created_at[:16] if s.created_at else "?"
            summary = s.summary[:40] + "..." if s.summary and len(s.summary) > 40 else (s.summary or "-")
            table.add_row(short_id, started, summary, str(s.total_queries), s.status or "?")

        self.console.print(table)
        self.console.print("[dim]Use /resume <id> or /restore <id> to continue a session[/dim]")

    def _save_plan(self, name: str, shared: bool = False) -> None:
        """Save current plan for replay."""
        if not self.api.session or not self.api.session.datastore:
            self.console.print("[yellow]No active session.[/yellow]")
            return
        if not self.last_problem:
            self.console.print("[yellow]No problem executed yet.[/yellow]")
            return
        try:
            self.api.session.save_plan(name, self.last_problem, user_id=self.user_id, shared=shared)
            if shared:
                self.console.print(f"[green]Plan saved as shared:[/green] {name}")
            else:
                self.console.print(f"[green]Plan saved:[/green] {name}")
        except Exception as e:
            self.console.print(f"[red]Error saving plan:[/red] {e}")

    def _share_with(self, args: str) -> None:
        """Share a plan with a specific user."""
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            self.console.print("[yellow]Usage: /sharewith <user> <plan_name>[/yellow]")
            return

        target_user, plan_name = parts
        if Session.share_plan_with(plan_name, target_user, from_user=self.user_id):
            self.console.print(f"[green]Plan '{plan_name}' shared with {target_user}[/green]")
        else:
            self.console.print(f"[red]Plan '{plan_name}' not found[/red]")

    def _list_plans(self) -> None:
        """List saved plans."""
        plans = Session.list_saved_plans(user_id=self.user_id)
        if not plans:
            self.console.print("[dim]No saved plans.[/dim]")
            return

        table = Table(title="Saved Plans", show_header=True, box=None)
        table.add_column("Name", style="cyan")
        table.add_column("Problem")
        table.add_column("Steps", justify="right")
        table.add_column("Type")

        for p in plans:
            plan_type = "[dim]shared[/dim]" if p.get("shared") else "private"
            problem = p.get("problem", "")[:50]
            if len(p.get("problem", "")) > 50:
                problem += "..."
            table.add_row(p["name"], problem, str(p.get("steps", 0)), plan_type)

        self.console.print(table)

    def _replay_plan(self, name: str) -> None:
        """Replay a saved plan."""
        if not self.api.session:
            self.api = self._create_api()

        try:
            plan_data = Session.load_saved_plan(name, user_id=self.user_id)
            self.last_problem = plan_data["problem"]
            self.display.set_problem(f"[Replay: {name}] {self.last_problem}")

            result = self.api.session.replay_saved(name, user_id=self.user_id)

            if result.get("success"):
                tables = result.get("datastore_tables", [])
                self.display.show_tables(tables, duration_ms=result.get("duration_ms", 0))
                self.display.show_summary(
                    success=True,
                    total_steps=len(result.get("results", [])),
                    duration_ms=result.get("duration_ms", 0),
                )
            else:
                self.console.print(f"[red]Error:[/red] {result.get('error', 'Unknown error')}")
        except ValueError as e:
            self.console.print(f"[red]Error:[/red] {e}")
        except Exception as e:
            self.console.print(f"[red]Error during replay:[/red] {e}")

    def _handle_audit(self) -> None:
        """Handle /audit command - re-derive last result with full audit trail."""
        if not self.api.session:
            self.console.print("[yellow]No active session. Ask a question first.[/yellow]")
            return

        self.display.start_spinner("Re-deriving with full audit trail...")
        try:
            result = self.api.session.audit()
            self.display.stop_spinner()

            if result.get("success"):
                output = result.get("output", "")
                if output:
                    self.console.print()
                    self.console.print(Panel(
                        output,
                        title="[bold]Audit Result[/bold]",
                        border_style="green",
                    ))

                verification = result.get("verification")
                if verification:
                    status = verification.get("verified", False)
                    msg = verification.get("message", "")
                    if status:
                        self.console.print(f"\n[bold green]Verified:[/bold green] {msg}")
                    else:
                        self.console.print(f"\n[bold yellow]Discrepancy:[/bold yellow] {msg}")

                suggestions = result.get("suggestions", [])
                if suggestions:
                    self.display.show_suggestions(suggestions)
            else:
                error = result.get("error", "Unknown error")
                self.console.print(f"[red]Audit failed:[/red] {error}")

        except Exception as e:
            self.display.stop_spinner()
            self.console.print(f"[red]Error during audit:[/red] {e}")

    def _handle_prove(self) -> None:
        """Handle /prove command - verify conversation claims with auditable proof."""
        if not self.api.session:
            self.console.print("[yellow]No active session. Ask questions first, then use /prove to verify.[/yellow]")
            return

        if not self.api.session.session_id:
            self.console.print("[yellow]No conversation to prove. Ask questions first.[/yellow]")
            return

        self.console.print("[cyan]Generating auditable proof for conversation claims...[/cyan]")
        self.display.start_spinner("Analyzing conversation...")

        try:
            result = self.api.session.prove_conversation()

            self.display.stop_spinner()

            if result.get("error"):
                self.console.print(f"[red]Error:[/red] {result['error']}")
                return

            if result.get("no_claims"):
                self.console.print("[yellow]No verifiable claims found in conversation.[/yellow]")
                self.console.print("[dim]Try asking data-related questions first, then use /prove.[/dim]")
                return

            claims = result.get("claims", [])
            self.console.print(f"\n[bold]Verified {len(claims)} claim(s):[/bold]\n")

            for i, claim in enumerate(claims, 1):
                status = "[green]VERIFIED[/green]" if claim.get("verified") else "[red]UNVERIFIED[/red]"
                self.console.print(f"  {i}. {status} {claim.get('claim', '')}")
                if claim.get("proof"):
                    self.console.print(f"     [dim]Proof: {claim['proof'][:100]}...[/dim]")
                if claim.get("discrepancy"):
                    self.console.print(f"     [yellow]Discrepancy: {claim['discrepancy']}[/yellow]")

        except Exception as e:
            self.display.stop_spinner()
            self.console.print(f"[red]Error during proof generation:[/red] {e}")

    def _handle_summarize(self, arg: str) -> None:
        """Handle /summarize command - generate LLM summary of plan, session, facts, or table."""
        if not arg.strip():
            self.console.print("[yellow]Usage: /summarize plan|session|facts|<table_name>[/yellow]")
            self.console.print("[dim]Examples:[/dim]")
            self.console.print("  /summarize plan     - Summarize execution plan")
            self.console.print("  /summarize session  - Summarize session state")
            self.console.print("  /summarize facts    - Summarize cached facts")
            self.console.print("  /summarize orders   - Summarize 'orders' table")
            return

        target = arg.strip().lower()
        self.display.start_spinner(f"Generating summary of {target}...")

        try:
            if target == "plan":
                result = self.api.summarize_plan()
            elif target == "session":
                result = self.api.summarize_session()
            elif target == "facts":
                result = self.api.summarize_facts()
            else:
                result = self.api.summarize_table(arg.strip())

            self.display.stop_spinner()

            if result.success and result.summary:
                self.console.print()
                self.console.print(Panel(
                    result.summary,
                    title=f"[bold]Summary: {target}[/bold]",
                    border_style="cyan",
                ))
            elif result.error:
                self.console.print(f"[yellow]{result.error}[/yellow]")
            else:
                self.console.print(f"[yellow]No data to summarize for '{target}'[/yellow]")

        except Exception as e:
            self.display.stop_spinner()
            self.console.print(f"[red]Error generating summary:[/red] {e}")
