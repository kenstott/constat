# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""CommandsMixin — all slash command handlers for ConstatREPLApp."""

from __future__ import annotations

import logging
import threading
import json
from pathlib import Path
from typing import TYPE_CHECKING

from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from constat.execution.mode import Phase
from constat.textual_repl._messages import ConsolidateComplete, DocumentAddComplete, GlossaryRefineComplete
from constat.textual_repl._widgets import (
    OutputLog, StatusBar, SidePanel, ProofTreePanel,
    make_file_link_markup,
)

if TYPE_CHECKING:
    from constat.textual_repl._app import ConstatREPLApp

logger = logging.getLogger(__name__)


class CommandsMixin:
    """Mixin providing all slash command handlers for ConstatREPLApp."""

    async def _handle_command(self: "ConstatREPLApp", command: str) -> None:
        """Handle a slash command."""
        log = self.query_one("#output-log", OutputLog)

        cmd_parts = command.split(maxsplit=1)
        cmd = cmd_parts[0].lower()
        args = cmd_parts[1] if len(cmd_parts) > 1 else ""

        if cmd in ("/quit", "/q"):
            self.exit()
        elif cmd in ("/help", "/h"):
            await self._show_help()
        elif cmd == "/tables":
            await self._show_tables()
        elif cmd == "/show" and args:
            await self._show_table(args)
        elif cmd == "/query" and args:
            await self._run_query(args)
        elif cmd == "/facts":
            await self._show_facts()
        elif cmd == "/state":
            await self._show_state()
        elif cmd == "/reset":
            await self._reset_session()
        elif cmd == "/redo":
            await self._redo(args)
        elif cmd == "/artifacts":
            show_all = args.strip().lower() == "all"
            await self._show_artifacts(show_all=show_all)
        elif cmd == "/code":
            await self._show_code(args)
        elif cmd in ("/prove", "/audit"):
            await self._handle_prove()
        elif cmd == "/preferences":
            await self._show_preferences()
        elif cmd == "/databases":
            await self._show_databases()
        elif cmd in ("/database", "/db"):
            await self._add_database(args)
        elif cmd == "/apis":
            await self._show_apis()
        elif cmd == "/api":
            await self._add_api(args)
        elif cmd in ("/documents", "/docs"):
            await self._show_documents()
        elif cmd in ("/files", "/file"):
            await self._show_files()
        elif cmd == "/doc":
            await self._add_document(args)
        elif cmd == "/context":
            await self._show_context()
        elif cmd in ("/history", "/sessions"):
            await self._show_history()
        elif cmd == "/verbose":
            await self._toggle_setting("verbose", args)
        elif cmd == "/raw":
            await self._toggle_setting("raw", args)
        elif cmd == "/insights":
            await self._toggle_setting("insights", args)
        elif cmd in ("/update", "/refresh"):
            await self._refresh_metadata()
        elif cmd == "/learnings":
            await self._show_learnings()
        elif cmd in ("/consolidate", "/compact-learnings"):
            await self._consolidate_learnings()
        elif cmd == "/compact":
            await self._compact_context()
        elif cmd == "/remember" and args:
            await self._remember_fact(args)
        elif cmd == "/forget" and args:
            await self._forget_fact(args)
        elif cmd == "/correct" and args:
            await self._handle_correct(args)
        elif cmd == "/save" and args:
            await self._save_plan(args)
        elif cmd == "/share" and args:
            await self._save_plan(args, shared=True)
        elif cmd == "/plans":
            await self._list_plans()
        elif cmd == "/replay" and args:
            await self._replay_plan(args)
        elif cmd == "/resume" and args:
            await self._resume_session(args)
        elif cmd == "/export" and args:
            await self._export_table(args)
        elif cmd == "/summarize" and args:
            await self._handle_summarize(args)
        elif cmd == "/prove":
            await self._handle_audit()
        elif cmd == "/user":
            log.write(Text(f"Current user: {self.user_id}", style="dim"))
        elif cmd == "/discover":
            await self._discover(args)
        elif cmd == "/glossary":
            await self._show_glossary(args)
        elif cmd == "/define" and args:
            await self._define_term(args)
        elif cmd == "/undefine" and args:
            await self._undefine_term(args)
        elif cmd == "/refine" and args:
            await self._refine_term(args)
        else:
            log.write(Text(f"Unknown command: {cmd}", style="yellow"))
            log.write(Text("Type /help for available commands.", style="dim"))

    async def _show_help(self: "ConstatREPLApp") -> None:
        """Show help information using centralized HELP_COMMANDS."""
        from constat.commands import HELP_COMMANDS
        log = self.query_one("#output-log", OutputLog)

        table = Table(title="Commands", show_header=True, box=None)
        table.add_column("Command", style="cyan")
        table.add_column("Description")

        for cmd, desc, _category in HELP_COMMANDS:
            table.add_row(cmd, desc)

        log.write(table)

        log.write("")
        shortcuts_table = Table(title="Keyboard Shortcuts", show_header=True, box=None)
        shortcuts_table.add_column("Key", style="cyan")
        shortcuts_table.add_column("Action")

        shortcuts = [
            ("Ctrl+Left", "Shrink side panel"),
            ("Ctrl+Right", "Expand side panel"),
            ("Up/Down", "Navigate command history"),
            ("Ctrl+C / Esc", "Cancel current operation"),
            ("Ctrl+D", "Exit"),
        ]

        for key, action in shortcuts:
            shortcuts_table.add_row(key, action)

        log.write(shortcuts_table)

    async def _show_tables(self: "ConstatREPLApp") -> None:
        """Show available tables with file:// URIs."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session or not self.session.session_id:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            from constat.storage.registry import ConstatRegistry
            registry = ConstatRegistry()
            tables = registry.list_tables(user_id=self.user_id, session_id=self.session.session_id)
            registry.close()

            if not tables:
                log.write(Text("No tables yet.", style="dim"))
                return

            log.write(Text(f"Tables ({len(tables)})", style="bold"))
            for t in tables:
                role_suffix = f" @{t.role_id}" if getattr(t, "role_id", None) else ""
                log.write(Text.assemble(
                    ("  ", ""),
                    (t.name, "cyan"),
                    (f" ({t.row_count} rows)", "dim"),
                    (role_suffix, "blue"),
                ))
                file_path = Path(t.file_path)
                if file_path.exists():
                    file_uri = file_path.resolve().as_uri()
                    link_markup = make_file_link_markup(file_uri, style="dim cyan underline", indent="    ")
                    log.write(link_markup)
        except Exception as e:
            log.write(Text(f"Error listing tables: {e}", style="red"))

    async def _show_table(self: "ConstatREPLApp", table_name: str) -> None:
        """Show contents of a specific table."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session or not self.session.datastore:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            df = self.session.datastore.query(f"SELECT * FROM {table_name} LIMIT 20")
            if df.empty:
                log.write(Text(f"Table '{table_name}' is empty.", style="dim"))
                return

            table = Table(title=f"{table_name} ({len(df)} rows shown)", show_header=True)
            for col in df.columns:
                table.add_column(str(col), style="cyan")

            for _, row in df.iterrows():
                table.add_row(*[str(v)[:50] for v in row.values])

            log.write(table)
        except Exception as e:
            log.write(Text(f"Error showing table: {e}", style="red"))

    async def _run_query(self: "ConstatREPLApp", sql: str) -> None:
        """Run a SQL query on the datastore."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session or not self.session.datastore:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            df = self.session.datastore.query(sql)
            if df.empty:
                log.write(Text("Query returned no results.", style="dim"))
                return

            table = Table(show_header=True)
            for col in df.columns:
                table.add_column(str(col), style="cyan")

            for _, row in df.head(20).iterrows():
                table.add_row(*[str(v)[:50] for v in row.values])

            log.write(table)
            if len(df) > 20:
                log.write(Text(f"... and {len(df) - 20} more rows", style="dim"))
        except Exception as e:
            log.write(Text(f"Query error: {e}", style="red"))

    async def _show_facts(self: "ConstatREPLApp") -> None:
        """Show cached facts from this session."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session or not hasattr(self.session, 'fact_resolver'):
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            facts = self.session.fact_resolver.get_all_facts()
            if not facts:
                log.write(Text("No facts cached.", style="dim"))
                return

            log.write(Text(f"Cached Facts ({len(facts)})", style="bold"))
            from constat.execution.fact_resolver import format_source_attribution

            for fact_id, fact in facts.items():
                value = getattr(fact, 'value', None)
                confidence = getattr(fact, 'confidence', 1.0)
                source = getattr(fact, 'source', None)
                source_name = getattr(fact, 'source_name', None)
                api_endpoint = getattr(fact, 'api_endpoint', None)
                role_id = getattr(fact, 'role_id', None)

                value_str = str(value)[:60] + "..." if len(str(value)) > 60 else str(value)

                if confidence >= 0.9:
                    status = "✓"
                elif confidence >= 0.5:
                    status = "◐"
                else:
                    status = "○"

                if source:
                    source_str = f"[{format_source_attribution(source, source_name, api_endpoint)}]"
                else:
                    source_str = ""

                role_str = f" @{role_id}" if role_id else ""

                log.write(Text(f"  {status} {fact_id}: {value_str} {source_str}{role_str}", style="dim"))
        except Exception as e:
            log.write(Text(f"Error showing facts: {e}", style="red"))
            logger.debug(f"_show_facts error: {e}", exc_info=True)

    async def _show_state(self: "ConstatREPLApp") -> None:
        """Show session state."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        log.write(Text("Session State", style="bold"))
        log.write(Text(f"  Phase: {status_bar.phase.value}", style="dim"))
        log.write(Text(f"  Verbose: {self.verbose}", style="dim"))
        log.write(Text(f"  User: {self.user_id}", style="dim"))
        if self.session:
            log.write(Text(f"  Session ID: {self.session.session_id}", style="dim"))
            log.write(Text(f"  Tables: {status_bar.tables_count}", style="dim"))
            log.write(Text(f"  Facts: {status_bar.facts_count}", style="dim"))

    async def _reset_session(self: "ConstatREPLApp") -> None:
        """Reset session and create a new session ID."""
        from constat.session import Session
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)
        side_panel = self.query_one("#side-panel", SidePanel)
        panel_content = self.query_one("#proof-tree-panel", ProofTreePanel)

        from constat.storage.session_store import SessionStore
        session_store = SessionStore(user_id=self.user_id)
        new_session_id = session_store.create_new()

        if self.session:
            # noinspection PyUnresolvedReferences
            old_config = self.session._config
            old_session_config = self.session.session_config
            self.session = Session(
                old_config,
                session_id=new_session_id,
                session_config=old_session_config,
                user_id=self.user_id,
            )
            if self._feedback_handler:
                self.session.on_event(self._feedback_handler.handle_event)

        self._plan_steps = []
        self._completed_plan_steps = []

        if self._feedback_handler:
            self._feedback_handler._steps_initialized = False

        panel_content.reset()
        side_panel.remove_class("visible")

        status_bar.update_status(
            phase=Phase.IDLE,
            status_message=None,
            tables_count=0,
            facts_count=0,
        )
        log.write(Text(f"New session: {new_session_id[:8]}...", style="green"))

    async def _redo(self: "ConstatREPLApp", instruction: str = "") -> None:
        """Retry last query, optionally with modifications."""
        log = self.query_one("#output-log", OutputLog)

        if not self.last_problem:
            log.write(Text("No previous query to redo.", style="yellow"))
            return

        problem = self.last_problem
        if instruction:
            problem = f"{problem}\n\nModification: {instruction}"

        log.write(Text(f"Redoing: {self.last_problem[:50]}...", style="dim"))
        await self._solve(problem)

    async def _show_artifacts(self: "ConstatREPLApp", show_all: bool = False) -> None:
        """Show saved artifacts with file:// URIs."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            from constat.storage.registry import ConstatRegistry
            registry = ConstatRegistry()

            all_tables = registry.list_tables(user_id=self.user_id, session_id=self.session.session_id)
            all_artifacts = registry.list_artifacts(user_id=self.user_id, session_id=self.session.session_id)

            if show_all:
                tables = all_tables
                artifacts = all_artifacts
            else:
                tables = registry.list_published_tables(user_id=self.user_id, session_id=self.session.session_id)
                artifacts = registry.list_published_artifacts(user_id=self.user_id, session_id=self.session.session_id)

            registry.close()

            intermediate_tables = len(all_tables) - len(tables) if not show_all else 0
            intermediate_artifacts = len(all_artifacts) - len(artifacts) if not show_all else 0
            intermediate_count = intermediate_tables + intermediate_artifacts

            if not tables and not artifacts:
                if intermediate_count > 0:
                    log.write(Text(f"No published artifacts. ({intermediate_count} intermediate - use /artifacts all to see)", style="dim"))
                else:
                    log.write(Text("No artifacts.", style="dim"))
                return

            log.write(Text("Artifacts", style="bold"))

            for t in tables:
                file_path = Path(t.file_path)
                if file_path.exists():
                    file_uri = file_path.resolve().as_uri()
                    role_suffix = f" @{t.role_id}" if getattr(t, "role_id", None) else ""
                    log.write(Text.assemble(
                        ("  📊 ", ""),
                        (t.name, "cyan"),
                        (f" ({t.row_count} rows)", "dim"),
                        (role_suffix, "blue"),
                    ))
                    link_markup = make_file_link_markup(file_uri, style="dim cyan underline", indent="     ")
                    log.write(link_markup)

            for artifact in artifacts:
                file_path = Path(artifact.file_path)
                if file_path.exists():
                    file_uri = file_path.resolve().as_uri()
                    if artifact.artifact_type in ("chart", "html", "map"):
                        icon = "📈"
                    elif artifact.artifact_type in ("image", "png", "svg", "jpeg"):
                        icon = "🖼️"
                    else:
                        icon = "📄"
                    role_suffix = f" @{artifact.role_id}" if getattr(artifact, "role_id", None) else ""
                    log.write(Text.assemble(
                        (f"  {icon} ", ""),
                        (artifact.name, "cyan"),
                        (role_suffix, "blue"),
                    ))
                    link_markup = make_file_link_markup(file_uri, style="dim cyan underline", indent="     ")
                    log.write(link_markup)

            if intermediate_count > 0 and not show_all:
                log.write(Text(f"\n  ({intermediate_count} intermediate artifacts - use /artifacts all to see)", style="dim"))
        except Exception as e:
            log.write(Text(f"Error showing artifacts: {e}", style="red"))

    async def _show_code(self: "ConstatREPLApp", step_arg: str = "") -> None:
        """Show generated code from execution history."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            history_df = None
            if self.session.datastore:
                history_df = self.session.datastore.get_execution_history_table()

            if history_df is not None and len(history_df) > 0:
                code_rows = history_df[history_df['code'].notna() & (history_df['code'] != '')]
                if len(code_rows) == 0:
                    log.write(Text("No code generated yet.", style="dim"))
                    return

                log.write(Text(f"Generated Code ({len(code_rows)} blocks)", style="bold"))
                for _, row in code_rows.iterrows():
                    step_num = row.get('step_number', 0)
                    if step_arg and str(step_num) != step_arg:
                        continue
                    step_goal = row.get('step_goal', '')
                    code = row.get('code', '')
                    success = row.get('success', True)

                    status = "✓" if success else "✗"
                    log.write(Text(f"\n--- Step {step_num}: {step_goal[:50]} {status} ---", style="dim"))
                    if code:
                        syntax = Syntax(code, "python", theme="monokai", line_numbers=False)
                        log.write(syntax)
                return

            code_blocks = self.session.datastore.get_session_meta("code_blocks") if self.session.datastore else []
            if code_blocks:
                log.write(Text(f"Generated Code ({len(code_blocks)} blocks)", style="bold"))
                for i, block in enumerate(code_blocks):
                    if step_arg and str(i + 1) != step_arg:
                        continue
                    log.write(Text(f"\n--- Step {i + 1} ---", style="dim"))
                    code = block.get("code", "") if isinstance(block, dict) else str(block)
                    if code:
                        syntax = Syntax(code, "python", theme="monokai", line_numbers=False)
                        log.write(syntax)
                return

            log.write(Text("No code generated yet.", style="dim"))
        except Exception as e:
            log.write(Text(f"Error showing code: {e}", style="red"))
            logger.debug(f"_show_code error: {e}", exc_info=True)

    async def _show_preferences(self: "ConstatREPLApp") -> None:
        """Show current preferences."""
        log = self.query_one("#output-log", OutputLog)
        log.write(Text("Preferences", style="bold"))
        log.write(Text(f"  verbose: {self.verbose}", style="dim"))
        log.write(Text(f"  user: {self.user_id}", style="dim"))

    async def _show_databases(self: "ConstatREPLApp") -> None:
        """Show configured databases."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session or not hasattr(self.session, 'schema_manager'):
            log.write(Text("No session available.", style="yellow"))
            return

        try:
            databases = self.session.config.databases or {}
            if not databases:
                log.write(Text("No databases configured.", style="dim"))
                return
            log.write(Text(f"Databases ({len(databases)})", style="bold"))
            for name, db in databases.items():
                uri_display = db.uri[:50] + "..." if db.uri and len(db.uri) > 50 else (db.uri or "(no uri)")
                log.write(Text(f"  {name}: {uri_display}", style="dim"))
                if db.description:
                    first_line = db.description.strip().split('\n')[0][:60]
                    log.write(Text(f"    {first_line}", style="dim italic"))
        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))

    async def _show_apis(self: "ConstatREPLApp") -> None:
        """Show configured APIs."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No session available.", style="yellow"))
            return

        try:
            apis = self.session.config.apis or {}
            if not apis:
                log.write(Text("No APIs configured.", style="dim"))
                return

            log.write(Text(f"APIs ({len(apis)})", style="bold"))
            for name, api in apis.items():
                api_type = api.type.value if hasattr(api.type, "value") else api.type
                log.write(Text(f"  {name}: {api_type} - {api.url}", style="dim"))
                if api.description:
                    first_line = api.description.strip().split('\n')[0][:60]
                    log.write(Text(f"    {first_line}", style="dim italic"))
        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))

    async def _show_documents(self: "ConstatREPLApp") -> None:
        """Show configured and session documents."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No session available.", style="yellow"))
            return

        try:
            docs = self.session.config.documents or {}
            if docs:
                log.write(Text(f"Configured Documents ({len(docs)})", style="bold"))
                for name, doc in docs.items():
                    doc_type = doc.type.value if hasattr(doc.type, "value") else doc.type
                    doc_path = doc.path or doc.url or "(inline)"
                    log.write(Text(f"  {name}: {doc_type} - {doc_path}", style="dim"))
                    if doc.description:
                        first_line = doc.description.strip().split('\n')[0][:60]
                        log.write(Text(f"    {first_line}", style="dim italic"))

            if self.session.doc_tools:
                session_docs = self.session.doc_tools.get_session_documents()
                if session_docs:
                    log.write(Text(f"Session Documents ({len(session_docs)})", style="bold cyan"))
                    for name, info in session_docs.items():
                        fmt = info.get("format", "text")
                        chars = info.get("char_count", 0)
                        log.write(Text(f"  {name}: {fmt} ({chars:,} chars)", style="dim"))

            if not docs and not (self.session.doc_tools and self.session.doc_tools.get_session_documents()):
                log.write(Text("No documents configured or added.", style="dim"))
        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))

    async def _discover(self: "ConstatREPLApp", args: str) -> None:
        """Unified semantic search across ALL data sources."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one(StatusBar)

        if not args.strip():
            log.write(Text("Usage: /discover [scope] <query>", style="yellow"))
            log.write(Text("  scope: database|api|document (optional)", style="dim"))
            log.write(Text("  query: semantic search terms", style="dim"))
            return

        parts = args.strip().split()
        source_filter = None
        query_parts = parts

        scope_map = {
            "database": "schema", "db": "schema", "databases": "schema",
            "table": "schema", "tables": "schema",
            "api": "api", "apis": "api",
            "document": "document", "documents": "document",
            "doc": "document", "docs": "document",
        }

        if parts and parts[0].lower() in scope_map:
            source_filter = scope_map[parts[0].lower()]
            query_parts = parts[1:]

        if not query_parts:
            log.write(Text("Please provide a search query.", style="yellow"))
            return

        query = " ".join(query_parts)
        status_bar.update_status(status_message=f"Searching: {query[:40]}...")

        try:
            vector_store = None
            if self.session:
                if hasattr(self.session, 'schema_manager') and self.session.schema_manager:
                    vector_store = getattr(self.session.schema_manager, '_vector_store', None)
                if not vector_store and hasattr(self.session, 'doc_tools') and self.session.doc_tools:
                    vector_store = getattr(self.session.doc_tools, '_vector_store', None)

            if not vector_store:
                log.write(Text("No vector store available.", style="yellow"))
                return

            from constat.embedding_loader import EmbeddingModelLoader
            import numpy as np
            model = EmbeddingModelLoader.get_instance().get_model()
            query_embedding = model.encode(query, convert_to_numpy=True)
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding)

            enriched_results = vector_store.search_enriched(
                query_embedding=query_embedding,
                limit=20,
            )

            def get_source_type(document_name: str) -> str:
                if document_name.startswith("schema:"):
                    return "schema"
                elif document_name.startswith("api:"):
                    return "api"
                return "document"

            if source_filter:
                enriched_results = [
                    r for r in enriched_results
                    if get_source_type(r.chunk.document_name) == source_filter
                ]

            enriched_results = [r for r in enriched_results if r.score >= 0.3]

            if not enriched_results:
                scope_str = f" in {source_filter}" if source_filter else ""
                log.write(Text(f"No results found for '{query}'{scope_str}.", style="dim"))
                return

            scope_str = f" ({source_filter})" if source_filter else ""
            log.write(Text(f"Found {len(enriched_results)} matches{scope_str}:", style="bold"))

            table = Table(show_header=True, box=None, padding=(0, 1))
            table.add_column("Score", style="dim", width=5)
            table.add_column("Source", style="cyan", width=10)
            table.add_column("Name", style="bold", width=30)
            table.add_column("Content", style="dim")

            for r in enriched_results:
                score = f"{r.score:.2f}"
                doc_name = r.chunk.document_name
                source_type = get_source_type(doc_name)

                if source_type == "schema":
                    source_display = "DATABASE"
                    name = doc_name.replace("schema:", "")
                elif source_type == "api":
                    source_display = "API"
                    name = doc_name.replace("api:", "")
                else:
                    source_display = "DOCUMENT"
                    name = doc_name

                content_preview = r.chunk.content[:60].replace("\n", " ")
                if len(r.chunk.content) > 60:
                    content_preview += "..."

                table.add_row(score, source_display, name, content_preview)

            log.write(table)
            status_bar.update_status(status_message="Search complete")

        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))
            logger.debug(f"_discover error: {e}", exc_info=True)
            status_bar.update_status(status_message="Search failed")

    async def _show_context(self: "ConstatREPLApp") -> None:
        """Show context size and token usage."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            stats = self.session.get_context_stats()
            if not stats:
                log.write(Text("No context stats available.", style="dim"))
                return

            log.write(Text("Context Usage", style="bold"))
            log.write(Text(f"  Total tokens: ~{stats.total_tokens:,}", style="dim"))

            if hasattr(stats, 'scratchpad_tokens') and stats.scratchpad_tokens:
                log.write(Text(f"  Scratchpad: ~{stats.scratchpad_tokens:,} tokens", style="dim"))
            if hasattr(stats, 'tables_tokens') and stats.tables_tokens:
                log.write(Text(f"  Tables: ~{stats.tables_tokens:,} tokens", style="dim"))
            if hasattr(stats, 'state_tokens') and stats.state_tokens:
                log.write(Text(f"  State: ~{stats.state_tokens:,} tokens", style="dim"))

            if hasattr(stats, 'is_critical') and stats.is_critical:
                log.write(Text("  ⚠️  Context is critical - consider /compact", style="yellow"))
            elif hasattr(stats, 'is_warning') and stats.is_warning:
                log.write(Text("  ⚡ Context is getting large", style="yellow"))
            else:
                log.write(Text("  ✓ Context size is healthy", style="green"))

        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))
            logger.debug(f"_show_context error: {e}", exc_info=True)

    async def _show_history(self: "ConstatREPLApp") -> None:
        """Show recent sessions with IDs and summaries."""
        log = self.query_one("#output-log", OutputLog)

        try:
            from constat.storage.history import SessionHistory
            hist = SessionHistory(user_id=self.user_id)
            sessions = hist.list_sessions(limit=10)

            if not sessions:
                log.write(Text("No previous sessions.", style="dim"))
                return

            log.write(Text(f"Recent Sessions ({len(sessions)})", style="bold"))
            log.write("")

            for s in sessions:
                short_id = s.session_id[:20]
                _status_style = "green" if s.status == "completed" else "yellow" if s.status == "active" else "dim"

                log.write(Text(f"  {short_id}", style="cyan bold"))

                date_str = s.created_at[:16] if s.created_at else "unknown"
                stats = f"    {date_str}  |  {s.status}  |  {s.total_queries} queries"
                log.write(Text(stats, style="dim"))

                summary = s.summary
                if not summary:
                    try:
                        detail = hist.get_session(s.session_id)
                        if detail and detail.queries:
                            first_q = detail.queries[0].question
                            summary = first_q[:80] + "..." if len(first_q) > 80 else first_q
                    except (OSError, ValueError, KeyError):
                        pass

                if summary:
                    log.write(Text(f"    {summary}", style="white"))

                log.write("")

            log.write(Text("Use /resume <id> to continue a session (can use partial ID)", style="dim"))
        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))
            logger.debug(f"_show_history error: {e}", exc_info=True)

    @staticmethod
    def _parse_bool_toggle(value: str, current: bool) -> bool:
        """Parse a boolean value string, or toggle current if empty/unrecognized."""
        lower = value.lower()
        if lower in ("on", "true", "1"):
            return True
        elif lower in ("off", "false", "0"):
            return False
        return not current

    async def _toggle_setting(self: "ConstatREPLApp", setting: str, value: str = "") -> None:
        """Toggle or set a boolean setting."""
        log = self.query_one("#output-log", OutputLog)

        if setting == "verbose":
            self.verbose = self._parse_bool_toggle(value, self.verbose)
            self.session_config.verbose = self.verbose
            log.write(Text(f"Verbose: {'on' if self.verbose else 'off'}", style="dim"))
        elif setting == "raw":
            self.session_config.show_raw_output = self._parse_bool_toggle(value, self.session_config.show_raw_output)
            status = "on" if self.session_config.show_raw_output else "off"
            log.write(Text(f"Raw output: {status}", style="dim"))
            self._update_settings_display()
        elif setting == "insights":
            self.session_config.enable_insights = self._parse_bool_toggle(value, self.session_config.enable_insights)
            status = "on" if self.session_config.enable_insights else "off"
            log.write(Text(f"Insights: {status}", style="dim"))
            self._update_settings_display()

    async def _refresh_metadata(self: "ConstatREPLApp") -> None:
        """Refresh metadata and rebuild cache."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        status_bar.update_status(status_message="Refreshing metadata...")
        try:
            self.session.schema_manager.refresh()
            log.write(Text("Metadata refreshed.", style="green"))
        except Exception as e:
            log.write(Text(f"Refresh failed: {e}", style="red"))
        finally:
            status_bar.update_status(status_message=None)

    async def _show_learnings(self: "ConstatREPLApp") -> None:
        """Show learnings and rules."""
        log = self.query_one("#output-log", OutputLog)

        try:
            rules = self.learning_store.list_rules()
            if rules:
                log.write(Text(f"Rules ({len(rules)})", style="bold"))
                for r in rules[:10]:
                    conf = r.get("confidence", 0) * 100
                    applied = r.get("applied_count", 0)
                    log.write(Text(f"  [{conf:.0f}%] {r.get('summary', '')[:60]} (applied {applied}x)", style="dim"))

            raw = self.learning_store.list_raw_learnings(limit=None)
            pending = [l for l in raw if not l.get("promoted_to")]
            if pending:
                log.write(Text(f"Pending Learnings ({len(pending)} total)", style="bold"))
                for l in pending[:10]:
                    cat = l.get("category", "")[:10]
                    lid = l.get("id", "")[:12]
                    log.write(Text(f"  {lid} [{cat}] {l.get('correction', '')[:50]}...", style="dim"))
                if len(pending) > 10:
                    log.write(Text(f"  ... and {len(pending) - 10} more", style="dim"))

            if not rules and not pending:
                log.write(Text("No learnings yet.", style="dim"))
            elif len(pending) >= 5:
                log.write(Text(f"  Tip: Use /compact-learnings to promote similar learnings to rules", style="dim cyan"))
        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))

    async def _consolidate_learnings(self: "ConstatREPLApp") -> None:
        """Consolidate similar learnings into rules using LLM."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        stats = self.learning_store.get_stats()
        unpromoted = stats.get("unpromoted", 0)

        if unpromoted < 2:
            log.write(Text(f"Not enough learnings to consolidate ({unpromoted} pending, need at least 2).", style="yellow"))
            return

        status_bar.update_status(status_message="Consolidating learnings...", phase=Phase.EXECUTING)
        status_bar.start_timer()
        await self._start_spinner()

        log.write(Text(f"Analyzing {unpromoted} pending learnings...", style="dim"))

        consolidate_thread = threading.Thread(
            target=self._consolidate_in_thread,
            daemon=True
        )
        consolidate_thread.start()
        logger.debug("Consolidate thread started")

    def _consolidate_in_thread(self: "ConstatREPLApp") -> None:
        """Run consolidation in a thread and post result message when done."""
        logger.debug("_consolidate_in_thread starting")
        try:
            from constat.learning.compactor import LearningCompactor

            compactor = LearningCompactor(self.learning_store, self.session.llm)
            result = compactor.compact(dry_run=False)

            result_dict = {
                "success": True,
                "rules_created": result.rules_created,
                "rules_strengthened": result.rules_strengthened,
                "rules_merged": result.rules_merged,
                "learnings_archived": result.learnings_archived,
                "groups_found": result.groups_found,
                "errors": result.errors,
            }
        except Exception as e:
            result_dict = {"success": False, "error": str(e)}
            logger.debug(f"_consolidate_in_thread error: {e}", exc_info=True)

        logger.debug("_consolidate_in_thread complete, posting ConsolidateComplete message")
        self.post_message(ConsolidateComplete(result_dict))

    async def _compact_context(self: "ConstatREPLApp") -> None:
        """Compact context to reduce token usage."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        status_bar.update_status(status_message="Compacting context...")
        try:
            stats_before = self.session.get_context_stats()
            if stats_before:
                log.write(Text(f"Before: ~{stats_before.total_tokens:,} tokens", style="dim"))

            result = self.session.compact_context(
                summarize_scratchpad=True,
                sample_tables=True,
                clear_old_state=False,
                keep_recent_steps=3,
            )

            if result:
                log.write(Text(f"{result.message}", style="green"))
                log.write(Text(result.summary(), style="dim"))
            else:
                log.write(Text("Compaction returned no result.", style="yellow"))
        except Exception as e:
            log.write(Text(f"Error during compaction: {e}", style="red"))
        finally:
            status_bar.update_status(status_message=None)

    async def _remember_fact(self: "ConstatREPLApp", fact_text: str) -> None:
        """Remember a fact persistently."""
        log = self.query_one("#output-log", OutputLog)
        import re

        if not fact_text.strip():
            log.write(Text("Usage: /remember <fact>", style="yellow"))
            log.write(Text("  /remember churn_rate        - persist a session fact", style="dim"))
            log.write(Text("  /remember churn as baseline - persist with new name", style="dim"))
            return

        session_fact_match = re.match(r'^(\S+)(?:\s+as\s+(\S+))?$', fact_text.strip())

        if session_fact_match and self.session:
            fact_name = session_fact_match.group(1)
            new_name = session_fact_match.group(2)

            session_facts = self.session.fact_resolver.get_all_facts()

            matching_fact = None
            for key, fact in session_facts.items():
                if key == fact_name or key == f"{fact_name}()":
                    matching_fact = fact
                    break
                if hasattr(fact, 'name') and fact.name == fact_name:
                    matching_fact = fact
                    break

            if matching_fact:
                persist_name = new_name if new_name else (matching_fact.name if hasattr(matching_fact, 'name') else fact_name)

                context_parts = []
                if hasattr(matching_fact, 'source'):
                    context_parts.append(f"Source: {matching_fact.source.value}")
                if hasattr(matching_fact, 'source_name') and matching_fact.source_name:
                    context_parts.append(f"From: {matching_fact.source_name}")
                context = "\n".join(context_parts)

                description = matching_fact.description if hasattr(matching_fact, 'description') else f"Persisted from session"

                self.fact_store.save_fact(
                    name=persist_name,
                    value=matching_fact.value if hasattr(matching_fact, 'value') else str(matching_fact),
                    description=description,
                    context=context,
                )

                display_value = matching_fact.display_value if hasattr(matching_fact, 'display_value') else str(matching_fact)[:50]
                log.write(Text(f"Remembered: {persist_name} = {display_value}", style="green"))
                log.write(Text("This fact will persist across sessions.", style="dim"))
                return

        log.write(Text(f"Fact '{fact_text[:30]}...' not found in session.", style="yellow"))
        log.write(Text("Use /facts to see available session facts.", style="dim"))

    async def _forget_fact(self: "ConstatREPLApp", fact_name: str) -> None:
        """Forget a fact by name."""
        log = self.query_one("#output-log", OutputLog)

        if not fact_name.strip():
            log.write(Text("Usage: /forget <fact_name>", style="yellow"))
            return

        fact_name = fact_name.strip()
        found = False

        if self.fact_store.delete_fact(fact_name):
            log.write(Text(f"Forgot persistent fact: {fact_name}", style="green"))
            found = True

        if self.session:
            facts = self.session.fact_resolver.get_all_facts()
            if fact_name in facts:
                self.session.fact_resolver._cache.pop(fact_name, None)
                if not found:
                    log.write(Text(f"Forgot session fact: {fact_name}", style="green"))
                found = True

        if not found:
            log.write(Text(f"Fact '{fact_name}' not found.", style="yellow"))
            log.write(Text("Use /facts to see available facts.", style="dim"))

    async def _handle_correct(self: "ConstatREPLApp", correction: str) -> None:
        """Handle /correct command - record user correction."""
        log = self.query_one("#output-log", OutputLog)
        from constat.storage.learnings import LearningCategory, LearningSource

        if not correction.strip():
            log.write(Text("Usage: /correct <correction>", style="yellow"))
            log.write(Text("  /correct 'active users' means logged in within 30 days", style="dim"))
            return

        self.learning_store.save_learning(
            category=LearningCategory.USER_CORRECTION,
            context={
                "previous_question": self.last_problem,
                "correction_text": correction,
            },
            correction=correction,
            source=LearningSource.EXPLICIT_COMMAND,
        )
        log.write(Text(f"Learned: {correction[:60]}{'...' if len(correction) > 60 else ''}", style="green"))

    async def _save_plan(self: "ConstatREPLApp", name: str, shared: bool = False) -> None:
        """Save current plan for replay."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return
        if not self.last_problem:
            log.write(Text("No problem executed yet.", style="yellow"))
            return

        try:
            self.session.save_plan(name, self.last_problem, user_id=self.user_id, shared=shared)
            if shared:
                log.write(Text(f"Plan saved as shared: {name}", style="green"))
            else:
                log.write(Text(f"Plan saved: {name}", style="green"))
        except Exception as e:
            log.write(Text(f"Error saving plan: {e}", style="red"))

    async def _list_plans(self: "ConstatREPLApp") -> None:
        """List saved plans."""
        from constat.session import Session
        log = self.query_one("#output-log", OutputLog)

        try:
            plans = Session.list_saved_plans(user_id=self.user_id)
            if not plans:
                log.write(Text("No saved plans.", style="dim"))
                return

            table = Table(title="Saved Plans", show_header=True, box=None)
            table.add_column("Name", style="cyan")
            table.add_column("Problem")
            table.add_column("Steps", justify="right")
            table.add_column("Type")

            for p in plans:
                plan_type = "shared" if p.get("shared") else "private"
                problem = p.get("problem", "")[:50]
                if len(p.get("problem", "")) > 50:
                    problem += "..."
                table.add_row(p["name"], problem, str(p.get("steps", 0)), plan_type)

            log.write(table)
        except Exception as e:
            log.write(Text(f"Error listing plans: {e}", style="red"))

    async def _replay_plan(self: "ConstatREPLApp", name: str) -> None:
        """Replay a saved plan."""
        from constat.session import Session
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            plan_data = Session.load_saved_plan(name, user_id=self.user_id)
            self.last_problem = plan_data["problem"]
            log.write(Text(f"Replaying: {self.last_problem[:50]}...", style="dim"))
            await self._solve(self.last_problem)
        except Exception as e:
            log.write(Text(f"Error replaying plan: {e}", style="red"))

    async def _resume_session(self: "ConstatREPLApp", session_id: str) -> None:
        """Resume a previous session."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            sessions = self.session.history.list_sessions(limit=50)
            match = None
            for s in sessions:
                if s.session_id.startswith(session_id) or session_id in s.session_id:
                    match = s.session_id
                    break

            if not match:
                log.write(Text(f"Session not found: {session_id}", style="red"))
                return

            if self.session.resume(match):
                log.write(Text(f"Resumed session: {match[:30]}...", style="green"))
                tables = self.session.datastore.list_tables() if self.session.datastore else []
                if tables:
                    log.write(Text(f"{len(tables)} tables available - use /tables to view", style="dim"))
            else:
                log.write(Text(f"Failed to resume session: {match}", style="red"))
        except Exception as e:
            log.write(Text(f"Error resuming session: {e}", style="red"))

    async def _export_table(self: "ConstatREPLApp", arg: str) -> None:
        """Export a table to CSV/XLSX or export inspection state to JSON file.

        Usage:
          /export <table> [filename]
          /export inspection [filename]
        """
        log = self.query_one("#output-log", OutputLog)

        if not arg.strip():
            log.write(Text("Usage:", style="yellow"))
            log.write(Text("  /export <table> [filename]      - Export a table to CSV/XLSX", style="dim"))
            log.write(Text("  /export inspection [filename]   - Export session inspection to JSON", style="dim"))
            return

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        parts = arg.strip().split(maxsplit=1)
        target = parts[0].lower()

        # Branch: export inspection/state to JSON
        if target in ("inspection", "state"):
            filename = parts[1] if len(parts) > 1 else "inspection.json"
            try:
                state = self.session.get_state() if hasattr(self.session, "get_state") else {}
                output_path = Path(filename).resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("w", encoding="utf-8") as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
                log.write(Text("Exported inspection to:", style="green"))
                link_markup = make_file_link_markup(output_path.as_uri(), style="cyan underline", indent="  ")
                log.write(link_markup)
            except Exception as e:
                log.write(Text(f"Export failed: {e}", style="red"))
            return

        # Default: export a table
        if not self.session.datastore:
            log.write(Text("No datastore available in this session.", style="yellow"))
            return

        table_name = parts[0]
        filename = parts[1] if len(parts) > 1 else f"{table_name}.csv"

        ext = Path(filename).suffix.lower()
        if ext not in (".csv", ".xlsx"):
            log.write(Text(f"Unsupported format: {ext}. Use .csv or .xlsx", style="yellow"))
            return

        try:
            df = self.session.datastore.query(f"SELECT * FROM {table_name}")

            output_path = Path(filename).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if ext == ".csv":
                df.to_csv(output_path, index=False)
            else:
                df.to_excel(output_path, index=False)

            log.write(Text(f"Exported {len(df)} rows to:", style="green"))
            link_markup = make_file_link_markup(output_path.as_uri(), style="cyan underline", indent="  ")
            log.write(link_markup)
        except Exception as e:
            log.write(Text(f"Export failed: {e}", style="red"))

    async def _handle_summarize(self: "ConstatREPLApp", arg: str) -> None:
        """Generate LLM summary of plan, session, facts, or table."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        if not arg.strip():
            log.write(Text("Usage: /summarize plan|session|facts|<table>", style="yellow"))
            return

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        target = arg.strip().lower()
        status_bar.update_status(status_message=f"Summarizing {target}...")

        try:
            llm = self.session.router._get_provider(self.session.router.models["planning"])

            if target == "plan":
                if not self.session.plan:
                    log.write(Text("No plan to summarize.", style="yellow"))
                    return
                plan_text = str(self.session.plan)
                prompt = f"Summarize this execution plan concisely:\n\n{plan_text}"
            elif target == "session":
                tables = self.session.datastore.list_tables() if self.session.datastore else []
                facts = self.session.fact_resolver.get_all_facts() if hasattr(self.session, 'fact_resolver') else {}
                session_text = f"Tables: {len(tables)}, Facts: {len(facts)}"
                prompt = f"Summarize this session state:\n\n{session_text}"
            elif target == "facts":
                facts = self.session.fact_resolver.get_all_facts() if hasattr(self.session, 'fact_resolver') else {}
                if not facts:
                    log.write(Text("No facts to summarize.", style="yellow"))
                    return
                facts_text = "\n".join([f"{k}: {v}" for k, v in list(facts.items())[:20]])
                prompt = f"Summarize these facts concisely:\n\n{facts_text}"
            else:
                df = self.session.datastore.query(f"SELECT * FROM {target} LIMIT 100")
                if df.empty:
                    log.write(Text(f"Table '{target}' is empty.", style="yellow"))
                    return
                table_text = df.to_string()[:2000]
                prompt = f"Summarize this table data concisely:\n\n{table_text}"

            response = llm.complete(prompt, max_tokens=llm.max_output_tokens)
            summary = response.content if hasattr(response, 'content') else str(response)

            log.write(Text(f"Summary: {target}", style="bold"))
            log.write(Text(summary, style="dim"))
        except Exception as e:
            log.write(Text(f"Error generating summary: {e}", style="red"))
        finally:
            status_bar.update_status(status_message=None)

    async def _handle_audit(self: "ConstatREPLApp") -> None:
        """Re-derive last result with full audit trail."""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        if not self.session:
            log.write(Text("No active session. Ask a question first.", style="yellow"))
            return

        status_bar.update_status(status_message="Re-deriving with audit trail...")
        try:
            # noinspection PyUnresolvedReferences
            result = self.session.audit()

            if result.get("success"):
                output = result.get("output", "")
                if output:
                    log.write(Text("Audit Result", style="bold green"))
                    log.write(Text(output, style="dim"))

                verification = result.get("verification")
                if verification:
                    status = verification.get("verified", False)
                    msg = verification.get("message", "")
                    if status:
                        log.write(Text(f"Verified: {msg}", style="green"))
                    else:
                        log.write(Text(f"Discrepancy: {msg}", style="yellow"))
            else:
                error = result.get("error", "Unknown error")
                log.write(Text(f"Audit failed: {error}", style="red"))
        except Exception as e:
            log.write(Text(f"Error during audit: {e}", style="red"))
        finally:
            status_bar.update_status(status_message=None)

    async def _show_files(self: "ConstatREPLApp") -> None:
        """Show data files."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            files = getattr(self.session.config, 'files', [])
            if not files:
                log.write(Text("No data files configured.", style="dim"))
                return

            log.write(Text(f"Data Files ({len(files)})", style="bold"))
            for f in files:
                name = f.get('name', 'unknown') if isinstance(f, dict) else str(f)
                uri = f.get('uri', '') if isinstance(f, dict) else ''
                log.write(Text(f"  {name}", style="cyan"))
                if uri:
                    log.write(Text(f"    {uri}", style="dim underline"))
        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))

    async def _add_document(self: "ConstatREPLApp", args: str) -> None:
        """Add a document to the current session (runs in background)."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        if not args:
            log.write(Text("Usage: /doc <path_or_uri> [name] [description]", style="yellow"))
            log.write(Text("  Add a document to the current session for reference.", style="dim"))
            log.write(Text("  Supports: local paths, file:// URIs, http:// URLs, https:// URLs", style="dim"))
            log.write(Text("  File types: .pdf, .docx, .xlsx, .pptx, .md, .txt, .json, .yaml", style="dim"))
            log.write(Text("  Example: /doc ./README.md readme Project documentation", style="dim"))
            return

        parts = args.split(maxsplit=2)
        path_or_uri = parts[0]
        doc_name = parts[1] if len(parts) > 1 else None
        description = parts[2] if len(parts) > 2 else ""

        if not self.session.doc_tools:
            log.write(Text("Document tools not available.", style="red"))
            return

        log.write(Text(f"  Adding document: {path_or_uri}...", style="dim"))

        thread = threading.Thread(
            target=self._add_document_thread,
            args=(path_or_uri, doc_name, description),
            daemon=True
        )
        thread.start()

    def _add_document_thread(self: "ConstatREPLApp", path_or_uri: str, doc_name: str | None, description: str) -> None:
        """Add document in background thread, post result via message."""
        import tempfile
        import urllib.request
        from urllib.parse import urlparse

        try:
            file_path = path_or_uri
            parsed = urlparse(path_or_uri)

            if parsed.scheme in ("http", "https"):
                suffix = Path(parsed.path).suffix or ".txt"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    with urllib.request.urlopen(path_or_uri, timeout=30) as response:
                        tmp.write(response.read())
                    file_path = tmp.name

            elif parsed.scheme == "file":
                file_path = parsed.path
                if file_path.startswith("/") and len(file_path) > 2 and file_path[2] == ":":
                    file_path = file_path[1:]

            else:
                if path_or_uri.startswith("~"):
                    file_path = str(Path(path_or_uri).expanduser())

            success, message = self.session.doc_tools.add_document_from_file(
                file_path=file_path,
                name=doc_name,
                description=description,
            )
            self.post_message(DocumentAddComplete(success, message))

        except Exception as e:
            self.post_message(DocumentAddComplete(False, f"Error: {e}"))

    async def _show_glossary(self: "ConstatREPLApp", args: str = "") -> None:
        """Show glossary terms. /glossary [all|defined|deprecated]"""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        try:
            vector_store = None
            if hasattr(self.session, 'schema_manager') and self.session.schema_manager:
                vector_store = getattr(self.session.schema_manager, '_vector_store', None)
            if not vector_store and hasattr(self.session, 'doc_tools') and self.session.doc_tools:
                vector_store = getattr(self.session.doc_tools, '_vector_store', None)

            if not vector_store:
                log.write(Text("No vector store available.", style="yellow"))
                return

            scope = args.strip().lower() if args.strip() else "all"

            if scope == "deprecated":
                terms = vector_store.get_deprecated_glossary(self.session.session_id)
                if not terms:
                    log.write(Text("No deprecated glossary terms.", style="dim"))
                    return
                log.write(Text(f"Deprecated Glossary Terms ({len(terms)})", style="bold"))
                for t in terms:
                    name = t.display_name if hasattr(t, 'display_name') else t.name
                    defn = t.definition[:60] + "..." if t.definition and len(t.definition) > 60 else (t.definition or "")
                    log.write(Text(f"  {name}: {defn}", style="dim"))
                return

            unified = vector_store.get_unified_glossary(self.session.session_id)

            if scope == "defined":
                unified = [t for t in unified if t.get("glossary_status") == "defined"]
            elif scope != "all":
                log.write(Text(f"Unknown scope: {scope}. Use all|defined|deprecated", style="yellow"))
                return

            if not unified:
                log.write(Text("No glossary terms.", style="dim"))
                return

            defined = [t for t in unified if t.get("glossary_status") == "defined"]
            self_desc = [t for t in unified if t.get("glossary_status") == "self_describing"]

            log.write(Text(f"Glossary ({len(defined)} defined, {len(self_desc)} self-describing)", style="bold"))

            table = Table(show_header=True, box=None, padding=(0, 1))
            table.add_column("Name", style="cyan", width=25)
            table.add_column("Type", style="dim", width=10)
            table.add_column("Status", width=8)
            table.add_column("Definition", style="dim")

            for t in unified[:30]:
                name = t.get("display_name", t.get("name", ""))
                sem_type = t.get("semantic_type", "") or ""
                glossary_status = t.get("glossary_status", "")
                defn = t.get("definition", "") or ""
                if len(defn) > 50:
                    defn = defn[:50] + "..."

                if glossary_status == "defined":
                    status_style = "green"
                else:
                    status_style = "dim"

                table.add_row(
                    name,
                    sem_type,
                    Text(glossary_status, style=status_style),
                    defn,
                )

            log.write(table)

            if len(unified) > 30:
                log.write(Text(f"  ... and {len(unified) - 30} more", style="dim"))

        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))
            logger.debug(f"_show_glossary error: {e}", exc_info=True)

    async def _define_term(self: "ConstatREPLApp", args: str) -> None:
        """Add a glossary definition. /define <name> <definition>"""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            log.write(Text("Usage: /define <name> <definition>", style="yellow"))
            return

        name = parts[0]
        definition = parts[1]

        try:
            vector_store = None
            if hasattr(self.session, 'schema_manager') and self.session.schema_manager:
                vector_store = getattr(self.session.schema_manager, '_vector_store', None)
            if not vector_store and hasattr(self.session, 'doc_tools') and self.session.doc_tools:
                vector_store = getattr(self.session.doc_tools, '_vector_store', None)

            if not vector_store:
                log.write(Text("No vector store available.", style="yellow"))
                return

            existing = vector_store.get_glossary_term(name, self.session.session_id)
            if existing:
                vector_store.update_glossary_term(name, self.session.session_id, {
                    "definition": definition,
                    "provenance": "human",
                })
                log.write(Text(f"Updated definition for: {name}", style="green"))
            else:
                from constat.discovery.models import GlossaryTerm
                import uuid
                term = GlossaryTerm(
                    id=str(uuid.uuid4()),
                    name=name,
                    display_name=name.replace("_", " ").title(),
                    definition=definition,
                    status="draft",
                    provenance="human",
                    session_id=self.session.session_id,
                )
                vector_store.add_glossary_term(term)
                log.write(Text(f"Defined: {name}", style="green"))
        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))

    async def _undefine_term(self: "ConstatREPLApp", args: str) -> None:
        """Remove a glossary definition. /undefine <name>"""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        name = args.strip()
        if not name:
            log.write(Text("Usage: /undefine <name>", style="yellow"))
            return

        try:
            vector_store = None
            if hasattr(self.session, 'schema_manager') and self.session.schema_manager:
                vector_store = getattr(self.session.schema_manager, '_vector_store', None)
            if not vector_store and hasattr(self.session, 'doc_tools') and self.session.doc_tools:
                vector_store = getattr(self.session.doc_tools, '_vector_store', None)

            if not vector_store:
                log.write(Text("No vector store available.", style="yellow"))
                return

            vector_store.delete_glossary_term(name, self.session.session_id)
            log.write(Text(f"Removed definition: {name}", style="green"))
        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))

    async def _refine_term(self: "ConstatREPLApp", args: str) -> None:
        """AI-refine a glossary definition. /refine <name>"""
        log = self.query_one("#output-log", OutputLog)
        status_bar = self.query_one("#status-bar", StatusBar)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        name = args.strip()
        if not name:
            log.write(Text("Usage: /refine <name>", style="yellow"))
            return

        try:
            vector_store = None
            if hasattr(self.session, 'schema_manager') and self.session.schema_manager:
                vector_store = getattr(self.session.schema_manager, '_vector_store', None)
            if not vector_store and hasattr(self.session, 'doc_tools') and self.session.doc_tools:
                vector_store = getattr(self.session.doc_tools, '_vector_store', None)

            if not vector_store:
                log.write(Text("No vector store available.", style="yellow"))
                return

            term = vector_store.get_glossary_term(name, self.session.session_id)
            if not term or not term.definition:
                log.write(Text(f"No defined term found: {name}", style="yellow"))
                return

            status_bar.update_status(status_message=f"Refining: {name}...")
            log.write(Text(f"Refining definition for: {name}...", style="dim"))

            refine_thread = threading.Thread(
                target=self._refine_in_thread,
                args=(name, term.definition, vector_store),
                daemon=True,
            )
            refine_thread.start()
        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))

    def _refine_in_thread(self: "ConstatREPLApp", name: str, current_def: str, vector_store) -> None:
        """Run glossary refinement in a background thread."""
        try:
            llm = self.session.router._get_provider(self.session.router.models["planning"])

            prompt = (
                f"Refine this business glossary definition. Keep it concise and precise.\n\n"
                f"Term: {name}\n"
                f"Current definition: {current_def}\n\n"
                f"Return ONLY the improved definition text, nothing else."
            )
            response = llm.complete(prompt, max_tokens=200)
            refined = response.content if hasattr(response, 'content') else str(response)
            refined = refined.strip().strip('"').strip("'")

            vector_store.update_glossary_term(name, self.session.session_id, {
                "definition": refined,
                "provenance": "hybrid",
            })

            self.post_message(GlossaryRefineComplete({
                "success": True,
                "name": name,
                "before": current_def,
                "after": refined,
            }))
        except Exception as e:
            self.post_message(GlossaryRefineComplete({
                "success": False,
                "name": name,
                "error": str(e),
            }))

    async def _add_database(self: "ConstatREPLApp", args: str) -> None:
        """Add a temporary database connection to the current session."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        if not args:
            log.write(Text("Usage: /database <uri> [name]", style="yellow"))
            log.write(Text("  Add a database connection to this session temporarily.", style="dim"))
            log.write(Text("  Example: /database sqlite:///path/to/data.db mydata", style="dim"))
            log.write(Text("  Example: /database postgresql://user:pass@host/db", style="dim"))
            return

        parts = args.split(maxsplit=1)
        uri = parts[0]
        name = parts[1] if len(parts) > 1 else None

        if not name:
            if ":///" in uri:
                name = Path(uri.split(":///")[-1]).stem
            elif "://" in uri:
                name = uri.split("/")[-1].split("?")[0]
            else:
                name = "temp_db"

        log.write(Text(f"  Adding database: {name}...", style="dim"))

        try:
            if hasattr(self.session, 'add_session_database'):
                success, message = self.session.add_session_database(uri=uri, name=name)
                if success:
                    log.write(Text(f"  {message}", style="green"))
                else:
                    log.write(Text(f"  {message}", style="red"))
            else:
                log.write(Text("  Temporary database addition not yet implemented.", style="yellow"))
                log.write(Text("  Add database to config.yaml to use it.", style="dim"))
        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))

    async def _add_api(self: "ConstatREPLApp", args: str) -> None:
        """Add a temporary API connection to the current session."""
        log = self.query_one("#output-log", OutputLog)

        if not self.session:
            log.write(Text("No active session.", style="yellow"))
            return

        if not args:
            log.write(Text("Usage: /api <spec_url> [name]", style="yellow"))
            log.write(Text("  Add an API to this session temporarily.", style="dim"))
            log.write(Text("  Example: /api https://api.example.com/openapi.json myapi", style="dim"))
            log.write(Text("  Example: /api https://example.com/graphql (auto-detects GraphQL)", style="dim"))
            return

        parts = args.split(maxsplit=1)
        spec_url = parts[0]
        name = parts[1] if len(parts) > 1 else None

        if not name:
            from urllib.parse import urlparse
            parsed = urlparse(spec_url)
            name = parsed.netloc.replace(".", "_").replace(":", "_") or "temp_api"

        log.write(Text(f"  Adding API: {name}...", style="dim"))

        try:
            if hasattr(self.session, 'add_session_api'):
                success, message = self.session.add_session_api(spec_url=spec_url, name=name)
                if success:
                    log.write(Text(f"  {message}", style="green"))
                else:
                    log.write(Text(f"  {message}", style="red"))
            else:
                log.write(Text("  Temporary API addition not yet implemented.", style="yellow"))
                log.write(Text("  Add API to config.yaml to use it.", style="dim"))
        except Exception as e:
            log.write(Text(f"Error: {e}", style="red"))
