# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Core REPL mixin — __init__, prompt toolkit, dispatch, solve, run loop."""

import html as html_module
import logging
import os
import shutil
import sys
from collections.abc import Callable

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style as PTStyle
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from constat.api.impl import ConstatAPIImpl
from constat.core.config import Config
from constat.execution.mode import Mode
from constat.messages import get_starter_suggestions, get_vera_adjectives, get_vera_tagline
from constat.repl.feedback import FeedbackDisplay, SessionFeedbackHandler
from constat.session import Session, SessionConfig
from constat.session._types import StepEvent
from constat.storage.facts import FactStore
from constat.storage.learnings import LearningStore
from constat.visualization.output import clear_pending_outputs

logger = logging.getLogger(__name__)


class _CoreMixin:
    """Core REPL mixin: __init__, prompt toolkit, dispatch, solve, run loop."""

    def __init__(
        self,
        config: Config,
        verbose: bool = False,
        console: Console | None = None,
        progress_callback: Callable[..., None] | None = None,
        user_id: str = "default",
        auto_resume: bool = False,
    ):
        self.config = config
        self.verbose = verbose
        self.console = console or Console()
        self.display = FeedbackDisplay(console=self.console, verbose=verbose)
        self.progress_callback = progress_callback
        self.session_config = SessionConfig(verbose=verbose)
        self.user_id = user_id
        self.auto_resume = auto_resume
        self.last_problem = ""
        self.suggestions: list[str] = []

        self.api: ConstatAPIImpl = self._create_api()

        self._prompt_style = PTStyle.from_dict({
            'bottom-toolbar': 'bg:#1a1a1a #888888',
            'bottom-toolbar.text': '#888888',
        })

    def _create_api(self, new_session: bool = False) -> ConstatAPIImpl:
        """Create a new API instance with callbacks configured."""
        from constat.storage.session_store import SessionStore
        session_store = SessionStore(user_id=self.user_id)
        if new_session:
            session_id = session_store.create_new()
        else:
            session_id = session_store.get_or_create()

        session = Session(
            self.config,
            session_id=session_id,
            session_config=self.session_config,
            progress_callback=self.progress_callback,
            user_id=self.user_id,
        )

        fact_store = FactStore(user_id=self.user_id)
        learning_store = LearningStore(user_id=self.user_id)

        fact_store.load_into_session(session)

        api = ConstatAPIImpl(
            session=session,
            fact_store=fact_store,
            learning_store=learning_store,
        )

        handler = SessionFeedbackHandler(self.display, self.session_config)
        api.on_event(lambda event_type, data: handler.handle_event(
            StepEvent(event_type=event_type, step_number=0, data=data)
        ))

        api.set_approval_callback(self.display.request_plan_approval)
        api.set_clarification_callback(self.display.request_clarification)

        if self.session_config.default_mode:
            self.display.update_status_line(mode=self.session_config.default_mode)

        return api

    def _get_suggestion_context(self) -> dict:
        """Provide context for typeahead suggestions."""
        context = {"tables": [], "columns": [], "plans": []}

        if self.api.session and self.api.session.datastore:
            tables = self.api.session.datastore.list_tables()
            context["tables"] = [t["name"] for t in tables]

        try:
            plans = Session.list_saved_plans(user_id=self.user_id)
            context["plans"] = [p["name"] for p in plans]
        except Exception as e:
            logger.debug("Failed to list plans for suggestions: %s", e)

        return context

    def _get_bottom_toolbar(self):
        """Get the status bar text for the bottom toolbar as HTML."""
        data = self.display._status_bar.get_toolbar_data()

        mode = data["mode"]
        phase = data["phase"]

        if mode == Mode.PROOF:
            mode_html = '<style bg="ansiyellow" fg="ansiblack"><b> PROOF </b></style>'
        else:
            mode_html = '<style bg="ansicyan" fg="ansiblack"><b> EXPLORE </b></style>'

        if data["status_message"]:
            phase_text = html_module.escape(data["status_message"])
        elif phase.value == "idle":
            phase_text = 'ready'
        elif phase.value == "planning":
            plan = data["plan_name"] or ""
            if plan:
                plan = plan[:40] + "..." if len(plan) > 40 else plan
                phase_text = f'planning: {html_module.escape(plan)}'
            else:
                phase_text = 'planning...'
        elif phase.value == "executing":
            step = data["step_current"]
            total = data["step_total"]
            desc = data["step_description"] or ""
            if desc:
                desc = desc[:30] + "..." if len(desc) > 30 else desc
                phase_text = f'executing step {step}/{total}: {html_module.escape(desc)}'
            else:
                phase_text = f'executing step {step}/{total}'
        elif phase.value == "failed":
            err = data["error_message"] or "error"
            err = err[:40] + "..." if len(err) > 40 else err
            phase_text = f'failed: {html_module.escape(err)}'
        else:
            phase_text = phase.value

        stats_html = f'<style fg="ansigray">tables:{data["tables_count"]} facts:{data["facts_count"]}</style>'

        terminal_width = shutil.get_terminal_size().columns
        rule_line = '─' * terminal_width

        return HTML(f'<style fg="ansigray" bg="#333333">{rule_line}</style>\n{mode_html} {phase_text}  {stats_html}')

    def _get_completer(self) -> WordCompleter:
        """Build a completer with commands and dynamic context."""
        from constat.repl.interactive import REPL_COMMANDS
        words = list(REPL_COMMANDS)

        if self.api.session and self.api.session.datastore:
            try:
                tables = self.api.session.datastore.list_tables()
                for t in tables:
                    words.append(t["name"])
            except Exception as e:
                logger.debug("Failed to list tables for completer: %s", e)

        try:
            plans = Session.list_saved_plans(user_id=self.user_id)
            for p in plans:
                words.append(p["name"])
        except Exception as e:
            logger.debug("Failed to list plans for completer: %s", e)

        return WordCompleter(words, ignore_case=True)

    def _get_input(self) -> str:
        """Get user input with status bar at bottom."""
        self.console.print()
        self.console.print(Rule("[bold green]YOU[/bold green]", align="right"))

        result = pt_prompt(
            "> ",
            completer=self._get_completer(),
            complete_while_typing=True,
            bottom_toolbar=self._get_bottom_toolbar,
            style=self._prompt_style,
        )
        return result.strip()

    def _show_help(self) -> None:
        """Show available commands."""
        table = Table(title="Commands", show_header=True, box=None)
        table.add_column("Command", style="cyan")
        table.add_column("Description")
        commands = [
            ("/help, /h", "Show this help message"),
            ("/tables", "List available tables"),
            ("/show <table>", "Show table contents"),
            ("/export <table> [file]", "Export table to CSV or XLSX"),
            ("/query <sql>", "Run SQL query on datastore"),
            ("/code [step]", "Show generated code (all or specific step)"),
            ("/state", "Show session state"),
            ("/update, /refresh", "Refresh metadata and rebuild preload cache"),
            ("/reset", "Clear session state and start fresh"),
            ("/redo [instruction]", "Retry last query (optionally with modifications)"),
            ("/user [name]", "Show or set current user"),
            ("/save <name>", "Save current plan for replay"),
            ("/share <name>", "Save plan as shared (all users)"),
            ("/sharewith <user> <name>", "Share plan with specific user"),
            ("/plans", "List saved plans"),
            ("/replay <name>", "Replay a saved plan"),
            ("/history, /sessions", "List your recent sessions"),
            ("/resume, /restore <id>", "Resume a previous session"),
            ("/context", "Show context size and token usage"),
            ("/compact", "Compact context to reduce token usage"),
            ("/facts", "Show cached facts from this session"),
            ("/remember <fact> [as name]", "Persist a session fact or extract from text"),
            ("/forget <name>", "Forget a remembered fact by name"),
            ("/verbose [on|off]", "Toggle or set verbose mode (step details)"),
            ("/raw [on|off]", "Toggle or set raw output display"),
            ("/insights [on|off]", "Toggle or set insight synthesis"),
            ("/preferences", "Show current preferences"),
            ("/artifacts", "Show saved artifacts with file:// URIs"),
            ("/database, /databases, /db", "List or manage databases"),
            ("/database save <n> <t> <uri>", "Save database bookmark"),
            ("/database delete <name>", "Delete database bookmark"),
            ("/database <name>", "Use bookmarked database"),
            ("/file, /files", "List all data files"),
            ("/file save <n> <uri>", "Save file bookmark"),
            ("/file delete <name>", "Delete file bookmark"),
            ("/file <name>", "Use bookmarked file"),
            ("/correct <text>", "Record a correction for future reference"),
            ("/learnings [category]", "Show learnings and rules"),
            ("/compact-learnings", "Compact learnings into rules"),
            ("/forget-learning <id>", "Delete a learning by ID"),
            ("/audit", "Re-derive last result with full audit trail"),
            ("/summarize <target>", "Summarize plan|session|facts|<table>"),
            ("/prove", "Verify conversation claims with auditable proof"),
            ("/quit, /q", "Exit"),
        ]
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        self.console.print(table)

    def _handle_command(self, cmd_input: str) -> bool:
        """Handle a slash command.

        Routes core commands through session.solve() to use centralized command registry,
        keeping only REPL-specific presentation here.

        Returns:
            True if the REPL should exit, False otherwise.
        """
        parts = cmd_input.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit", "/q"):
            self.console.print("[dim]Goodbye![/dim]")
            return True

        registry_commands = {
            "/help", "/h", "/tables", "/show", "/query", "/code",
            "/artifacts", "/export", "/state", "/status", "/reset",
            "/facts", "/context", "/preferences",
            "/databases", "/apis", "/documents", "/docs", "/files",
        }

        if cmd in registry_commands or (cmd == "/show" and arg) or (cmd == "/query" and arg):
            return self._run_registry_command(cmd_input)

        if cmd in ("/update", "/refresh"):
            self._refresh_metadata()
        elif cmd == "/redo":
            self._handle_redo(arg)
        elif cmd == "/user":
            self._show_user(arg)
        elif cmd == "/save" and arg:
            self._save_plan(arg, shared=False)
        elif cmd == "/share" and arg:
            self._save_plan(arg, shared=True)
        elif cmd == "/sharewith" and arg:
            self._share_with(arg)
        elif cmd == "/plans":
            self._list_plans()
        elif cmd == "/replay" and arg:
            self._replay_plan(arg)
        elif cmd in ("/history", "/sessions"):
            self._show_history()
        elif cmd in ("/resume", "/restore") and arg:
            self._resume_session(arg)
        elif cmd == "/compact":
            self._compact_context()
        elif cmd == "/remember" and arg:
            self._remember_fact(arg)
        elif cmd == "/forget" and arg:
            self._forget_fact(arg)
        elif cmd == "/verbose":
            self._toggle_verbose(arg)
        elif cmd == "/raw":
            self._toggle_raw(arg)
        elif cmd == "/insights":
            self._toggle_insights(arg)
        elif cmd in ("/database", "/db"):
            self._handle_database(arg)
        elif cmd == "/file":
            self._handle_file(arg)
        elif cmd == "/correct":
            self._handle_correct(arg)
        elif cmd == "/learnings":
            self._show_learnings(arg)
        elif cmd == "/compact-learnings":
            self._compact_learnings()
        elif cmd == "/forget-learning" and arg:
            self._forget_learning(arg)
        elif cmd == "/audit":
            self._handle_audit()
        elif cmd == "/summarize":
            self._handle_summarize(arg)
        elif cmd == "/prove":
            self._handle_prove()
        else:
            self.console.print(f"[yellow]Unknown: {cmd}[/yellow]")

        return False

    def _run_registry_command(self, cmd_input: str) -> bool:
        """Run a command through the session's centralized command registry."""
        if not self.api.session:
            self.api = self._create_api()

        try:
            result = self.api.session.solve(cmd_input)

            if result.get("success") is False:
                self.console.print(f"[red]{result.get('output', 'Command failed')}[/red]")
            else:
                output = result.get("output", "")
                if output:
                    from rich.markdown import Markdown
                    self.console.print(Markdown(output))

        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")

        return False

    def _solve(self, problem: str) -> str | None:
        """Solve a problem.

        Returns:
            Command string if user entered a slash command during approval,
            None otherwise.
        """
        overrides = self.api.detect_display_overrides(problem)
        original_settings = self._apply_display_overrides(overrides)

        nl_correction = self.api.detect_nl_correction(problem)
        if nl_correction.detected:
            self._save_nl_correction(nl_correction, problem)
            self.console.print(f"[dim]Noted: {nl_correction.correction_type.replace('_', ' ')}[/dim]")

        clear_pending_outputs()

        self.last_problem = problem
        self.suggestions = []
        self.display.set_problem(problem)

        try:
            if self.api.session.session_id:
                result = self.api.session.follow_up(problem)
            else:
                result = self.api.session.solve(problem)

            if result.get("command"):
                return result["command"]

            if result.get("meta_response"):
                self.display.show_output(result.get("output", ""))
                self.suggestions = result.get("suggestions", [])
                if self.suggestions:
                    self.display.show_suggestions(self.suggestions)
                self.display.show_summary(success=True, total_steps=0, duration_ms=0)
            elif result.get("mode") == Mode.PROOF.value and result.get("success", True):
                self.display.show_output(result.get("output", ""))
                tables = result.get("datastore_tables", [])
                self.display.show_tables(tables, force_show=False)
                self.suggestions = result.get("suggestions", [])
                if self.suggestions:
                    self.display.show_suggestions(self.suggestions)
                self.display.show_summary(success=True, total_steps=0, duration_ms=0)
            elif result.get("success"):
                tables = result.get("datastore_tables", [])
                total_duration = sum(r.duration_ms for r in result.get("results", []))
                self.display.show_tables(tables, duration_ms=total_duration)
                self.display.show_summary(
                    success=True,
                    total_steps=len(result.get("results", [])),
                    duration_ms=total_duration,
                )
                self.suggestions = result.get("suggestions", [])

                self._display_outputs()

                self._check_context_warning()
            elif result.get("mode") != Mode.PROOF.value:
                self.console.print(f"[red]Error:[/red] {result.get('error', 'Unknown error')}")
        except KeyboardInterrupt:
            if self.api.session:
                self.api.session.cancel_execution()
            self.display.stop()
            self.display.stop_spinner()
            self.console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            self.display.stop()
            self.display.stop_spinner()
            self.console.print(f"[red]Error:[/red] {e}")
        finally:
            self._restore_display_settings(original_settings)

        return None

    def _run_query(self, sql: str) -> None:
        """Run SQL query on datastore."""
        if not self.api.session or not self.api.session.datastore:
            self.console.print("[yellow]No active session.[/yellow]")
            return
        try:
            result = self.api.session.datastore.query(sql)
            self.console.print(result.to_string())
        except Exception as e:
            self.console.print(f"[red]Query error:[/red] {e}")

    def run(self, initial_problem: str | None = None) -> None:
        """Run the interactive REPL."""
        os.environ["CONSTAT_REPL_MODE"] = "1"
        try:
            self.display.enable_status_bar()
            self._run_repl_body(initial_problem)
        finally:
            self.display.disable_status_bar()
            if self.api.session and self.api.session.datastore:
                self.api.session.datastore.close()

    def _run_repl_body(self, initial_problem: str | None = None) -> None:
        """Run the REPL body (banner + loop)."""
        sys.stdout.flush()
        sys.stderr.flush()

        self._maybe_auto_compact()

        if self.auto_resume:
            self._handle_auto_resume()

        reliable_adj, honest_adj = get_vera_adjectives()
        hints = "Tab completes commands | Ctrl+C interrupts"

        self.console.print()
        self.console.print(
            f"Hi, I'm [bold]Vera[/bold], your {reliable_adj} and {honest_adj} data analyst."
        )
        self.console.print(
            f"[dim]{get_vera_tagline()}[/dim]"
        )
        self.console.print()
        self.console.print(
            "[dim]Powered by[/dim] [blue bold]Constat[/blue bold] "
            "[dim](Latin: \"it is established\") — Multi-Step AI Reasoning Agent[/dim]"
        )
        self.console.print(
            f"[dim]Type /help for commands, or ask a question. | {hints}[/dim]"
        )

        if not initial_problem:
            self.console.print()
            self.console.print("[dim]Try asking:[/dim]")
            starter_suggestions = get_starter_suggestions()
            for i, s in enumerate(starter_suggestions, 1):
                self.console.print(f"  [dim]{i}.[/dim] [cyan]{s}[/cyan]")
            self.suggestions = starter_suggestions
            self.console.print()

        if initial_problem:
            self._solve(initial_problem)

        self._loop_body()

    def _loop_body(self) -> None:
        """The actual REPL loop body."""
        while True:
            try:
                user_input = self._get_input()

                if not user_input:
                    continue

                suggestion_to_run = None
                lower_input = user_input.lower().strip()

                if self.suggestions:
                    if lower_input.isdigit():
                        idx = int(lower_input) - 1
                        if 0 <= idx < len(self.suggestions):
                            suggestion_to_run = self.suggestions[idx]
                        else:
                            self.console.print(f"[yellow]No suggestion #{lower_input}[/yellow]")
                            continue
                    elif lower_input in ("ok", "yes", "sure", "y"):
                        suggestion_to_run = self.suggestions[0]

                if suggestion_to_run:
                    passthrough_cmd = self._solve(suggestion_to_run)
                    if passthrough_cmd and self._handle_command(passthrough_cmd):
                        break
                elif user_input.startswith("/"):
                    if self._handle_command(user_input):
                        break
                else:
                    passthrough_cmd = self._solve(user_input)
                    if passthrough_cmd and self._handle_command(passthrough_cmd):
                        break

            except KeyboardInterrupt:
                self.console.print("\n[dim]Type /quit to exit.[/dim]")
            except EOFError:
                break

    def _check_context_warning(self) -> None:
        """Check context size and warn if getting large."""
        if not self.api.session:
            return

        stats = self.api.session.get_context_stats()
        if not stats:
            return

        if stats.is_critical:
            self.console.print(
                "\n[bold red]Warning:[/bold red] Context size is critical "
                f"(~{stats.total_tokens:,} tokens). Use [cyan]/compact[/cyan] to reduce."
            )
        elif stats.is_warning:
            self.console.print(
                f"\n[yellow]Note:[/yellow] Context is growing (~{stats.total_tokens:,} tokens). "
                "Use [cyan]/context[/cyan] to view details."
            )
