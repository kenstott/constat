# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Data commands mixin — tables, export, artifacts, databases, files, code, state."""

import logging
import re
from pathlib import Path

from rich.syntax import Syntax

logger = logging.getLogger(__name__)


class _DataCommandsMixin:
    """Data-related REPL commands: tables, export, artifacts, databases, files, code, state."""

    @staticmethod
    def _role_suffix(item) -> str:
        """Format role suffix for display."""
        role_id = item.get("role_id") if isinstance(item, dict) else getattr(item, "role_id", None)
        return f" [blue]@{role_id}[/blue]" if role_id else ""

    def _show_tables(self) -> None:
        """Show tables in current session with file:// URIs for Parquet files."""
        if not self.api.session or not self.api.session.session_id:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        try:
            from constat.storage.registry import ConstatRegistry
            registry = ConstatRegistry()
            try:
                tables = registry.list_tables(user_id=self.user_id, session_id=self.api.session.session_id)
            finally:
                registry.close()

            if not tables:
                self.console.print("[dim]No tables yet.[/dim]")
                return

            self.console.print(f"\n[bold]Tables[/bold] ({len(tables)})")
            for t in tables:
                self.console.print(f"  [cyan]{t.name}[/cyan] [dim]({t.row_count} rows)[/dim]{self._role_suffix(t)}")
                file_path = Path(t.file_path)
                if file_path.exists():
                    file_uri = file_path.resolve().as_uri()
                    self.console.print(f"    {file_uri}")

        except Exception as e:
            logger.debug("Registry unavailable for tables: %s", e)
            self.console.print("[yellow]Registry unavailable, showing datastore tables.[/yellow]")
            if not self.api.session.datastore:
                self.console.print("[yellow]No active session.[/yellow]")
                return
            tables = self.api.session.datastore.list_tables()
            if not tables:
                self.console.print("[dim]No tables yet.[/dim]")
                return
            self.display.show_tables(tables, force_show=True)

    def _export_table(self, arg: str) -> None:
        """Export a table to CSV or XLSX file."""
        if not arg.strip():
            self.console.print("[yellow]Usage: /export <table> [filename][/yellow]")
            self.console.print("[dim]Example: /export orders orders.csv[/dim]")
            self.console.print("[dim]Example: /export orders report.xlsx[/dim]")
            self.console.print("[dim]Example: /export _facts[/dim]")
            return

        parts = arg.strip().split(maxsplit=1)
        table_name = parts[0]
        filename = parts[1] if len(parts) > 1 else f"{table_name}.csv"

        ext = Path(filename).suffix.lower()
        if ext not in (".csv", ".xlsx"):
            self.console.print(f"[yellow]Unsupported format: {ext}. Use .csv or .xlsx[/yellow]")
            return

        try:
            if table_name == "_facts":
                if not self.api.session:
                    self.console.print("[yellow]No active session.[/yellow]")
                    return
                df = self.api.session.fact_resolver.get_facts_as_dataframe()
                if df.empty:
                    self.console.print("[dim]No facts to export.[/dim]")
                    return
            else:
                if not self.api.session or not self.api.session.datastore:
                    self.console.print("[yellow]No active session.[/yellow]")
                    return

                tables = self.api.session.datastore.list_tables()
                table_names = [t['name'] for t in tables]
                if table_name not in table_names:
                    self.console.print(f"[yellow]Table '{table_name}' not found.[/yellow]")
                    self.console.print(f"[dim]Available: {', '.join(table_names) or '(none)'}[/dim]")
                    return

                df = self.api.session.datastore.query(f'SELECT * FROM "{table_name}"')

            output_path = Path(filename).resolve()
            if ext == ".csv":
                df.to_csv(output_path, index=False)
            else:
                df.to_excel(output_path, index=False)

            self.console.print(f"[green]Exported {len(df)} rows to:[/green]")
            self.console.print(f"  {output_path.as_uri()}")

        except Exception as e:
            self.console.print(f"[red]Export failed:[/red] {e}")

    def _show_artifacts(self) -> None:
        """Show session artifacts: tables (Parquet) and saved files from registry."""
        if not self.api.session or not self.api.session.session_id:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        has_artifacts = False
        session_id = self.api.session.session_id

        try:
            from constat.storage.registry import ConstatRegistry
            registry = ConstatRegistry()
            try:
                tables = registry.list_tables(user_id=self.user_id, session_id=session_id)
                if tables:
                    has_artifacts = True
                    self.console.print(f"\n[bold]Tables[/bold] ({len(tables)})")
                    for t in tables:
                        self.console.print(f"  [cyan]{t.name}[/cyan] [dim]({t.row_count} rows)[/dim]{self._role_suffix(t)}")
                        if t.description:
                            self.console.print(f"    {t.description}")
                        file_path = Path(t.file_path)
                        if file_path.exists():
                            file_uri = file_path.resolve().as_uri()
                            self.console.print(f"    {file_uri}")

                artifacts = registry.list_artifacts(user_id=self.user_id, session_id=session_id)
                if artifacts:
                    has_artifacts = True
                    self.console.print(f"\n[bold]Files[/bold] ({len(artifacts)})")
                    for a in artifacts[:20]:
                        file_path = Path(a.file_path)
                        if file_path.exists():
                            file_uri = file_path.resolve().as_uri()
                            size_str = f"{a.size_bytes / 1024:.0f}KB" if a.size_bytes else ""
                            self.console.print(f"  [cyan]{a.name}[/cyan] [dim]({a.artifact_type}) {size_str}[/dim]{self._role_suffix(a)}")
                            if a.description:
                                self.console.print(f"    {a.description}")
                            self.console.print(f"    {file_uri}")

                    if len(artifacts) > 20:
                        self.console.print(f"\n[dim]... and {len(artifacts) - 20} more[/dim]")
            finally:
                registry.close()

        except Exception as e:
            logger.debug("Registry unavailable for artifacts: %s", e)
            self.console.print("[yellow]Registry unavailable, showing datastore tables.[/yellow]")
            if self.api.session.datastore:
                tables = self.api.session.datastore.list_tables()
                if tables:
                    has_artifacts = True
                    self.console.print(f"\n[bold]Tables[/bold] ({len(tables)})")
                    for t in tables:
                        self.console.print(f"  [cyan]{t['name']}[/cyan] [dim]({t['row_count']} rows)[/dim]{self._role_suffix(t)}")

        if not has_artifacts:
            self.console.print("[dim]No artifacts in this session.[/dim]")

    def _handle_database(self, arg: str) -> None:
        """Handle /database command variants."""
        from constat.storage.bookmarks import BookmarkStore

        if not arg:
            self._show_databases()
            return

        parts = arg.split()
        subcommand = parts[0].lower()

        if subcommand == "save" and len(parts) >= 4:
            name, db_type, uri = parts[1], parts[2], parts[3]
            description = self._extract_flag(arg, "--desc") or ""
            bookmarks = BookmarkStore()
            bookmarks.save_database(name, db_type, uri, description)
            self.console.print(f"[green]Saved database bookmark:[/green] {name}")

        elif subcommand == "delete" and len(parts) >= 2:
            name = parts[1]
            bookmarks = BookmarkStore()
            if bookmarks.delete_database(name):
                self.console.print(f"[green]Deleted database bookmark:[/green] {name}")
            else:
                self.console.print(f"[yellow]Bookmark not found:[/yellow] {name}")

        elif len(parts) == 1:
            name = parts[0]
            bookmarks = BookmarkStore()
            bm = bookmarks.get_database(name)
            if bm:
                if self.api.session:
                    self.api.session.add_database(name, bm["type"], bm["uri"], bm["description"])
                    self.console.print(f"[green]Added database to session:[/green] {name} ({bm['type']})")
                else:
                    self.console.print("[yellow]Start a session first by asking a question.[/yellow]")
            else:
                self.console.print(f"[yellow]Bookmark not found:[/yellow] {name}")

        elif len(parts) >= 3:
            name, db_type, uri = parts[0], parts[1], parts[2]
            description = self._extract_flag(arg, "--desc") or ""
            if self.api.session:
                self.api.session.add_database(name, db_type, uri, description)
                self.console.print(f"[green]Added database to session:[/green] {name} ({db_type})")
            else:
                self.console.print("[yellow]Start a session first by asking a question.[/yellow]")

        else:
            self.console.print("[yellow]Usage: /database [save|delete] <name> [<type> <uri>] [--desc \"...\"][/yellow]")

    def _handle_file(self, arg: str) -> None:
        """Handle /file command variants."""
        from constat.storage.bookmarks import BookmarkStore

        if not arg:
            self._show_files()
            return

        parts = arg.split()
        subcommand = parts[0].lower()

        if subcommand == "save" and len(parts) >= 3:
            name, uri = parts[1], parts[2]
            auth = self._extract_flag(arg, "--auth") or ""
            description = self._extract_flag(arg, "--desc") or ""
            bookmarks = BookmarkStore()
            bookmarks.save_file(name, uri, description, auth)
            self.console.print(f"[green]Saved file bookmark:[/green] {name}")

        elif subcommand == "delete" and len(parts) >= 2:
            name = parts[1]
            bookmarks = BookmarkStore()
            if bookmarks.delete_file(name):
                self.console.print(f"[green]Deleted file bookmark:[/green] {name}")
            else:
                self.console.print(f"[yellow]Bookmark not found:[/yellow] {name}")

        elif len(parts) == 1:
            name = parts[0]
            bookmarks = BookmarkStore()
            bm = bookmarks.get_file(name)
            if bm:
                if self.api.session:
                    self.api.session.add_file(name, bm["uri"], bm.get("auth", ""), bm["description"])
                    self.console.print(f"[green]Added file to session:[/green] {name}")
                else:
                    self.console.print("[yellow]Start a session first by asking a question.[/yellow]")
            else:
                self.console.print(f"[yellow]Bookmark not found:[/yellow] {name}")

        elif len(parts) >= 2:
            name, uri = parts[0], parts[1]
            auth = self._extract_flag(arg, "--auth") or ""
            description = self._extract_flag(arg, "--desc") or ""
            if self.api.session:
                self.api.session.add_file(name, uri, auth, description)
                self.console.print(f"[green]Added file to session:[/green] {name}")
            else:
                self.console.print("[yellow]Start a session first by asking a question.[/yellow]")

        else:
            self.console.print("[yellow]Usage: /file [save|delete] <name> [<uri>] [--auth \"...\"] [--desc \"...\"][/yellow]")

    def _extract_flag(self, text: str, flag: str) -> str | None:
        """Extract a flag value from command text."""
        pattern = rf'{flag}\s+"([^"]*)"'
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        pattern = rf'{flag}\s+(\S+)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)

        return None

    def _show_databases(self) -> None:
        """Show all databases (config + bookmarks + session)."""
        from constat.storage.bookmarks import BookmarkStore

        config_dbs = {}
        bookmark_dbs = {}
        session_dbs = {}

        if self.api.session:
            all_dbs = self.api.session.get_all_databases()
            for name, db in all_dbs.items():
                if db["source"] == "config":
                    config_dbs[name] = db
                elif db["source"] == "bookmark":
                    bookmark_dbs[name] = db
                elif db["source"] == "session":
                    session_dbs[name] = db
        else:
            if self.config and self.config.databases:
                for name, db_config in self.config.databases.items():
                    config_dbs[name] = {
                        "type": db_config.type or "sql",
                        "uri": db_config.uri or db_config.path or "",
                        "description": db_config.description or "",
                    }
            bookmarks = BookmarkStore()
            bookmark_dbs = bookmarks.list_databases()

        has_any = config_dbs or bookmark_dbs or session_dbs
        if not has_any:
            self.console.print("[dim]No databases configured.[/dim]")
            return

        self._print_database_section("Config Databases", config_dbs)
        self._print_database_section("Bookmarked Databases", bookmark_dbs)
        self._print_database_section("Session Databases", session_dbs)

    def _print_database_section(self, title: str, dbs: dict) -> None:
        """Print a section of databases with masked URIs."""
        if not dbs:
            return
        self.console.print(f"\n[bold]{title}[/bold] ({len(dbs)})")
        for name, db in dbs.items():
            uri_display = self._mask_credentials(db["uri"])
            self.console.print(f"  [cyan]{name}[/cyan] [dim]({db['type']})[/dim]")
            if db["description"]:
                self.console.print(f"    {db['description']}")
            self.console.print(f"    [dim]{uri_display}[/dim]")

    def _show_files(self) -> None:
        """Show all files (config docs + file sources + bookmarks + session)."""
        from constat.storage.bookmarks import BookmarkStore

        config_files = {}
        bookmark_files = {}
        session_files = {}

        if self.api.session:
            all_files = self.api.session.get_all_files()
            for name, f in all_files.items():
                if f["source"] == "config":
                    config_files[name] = f
                elif f["source"] == "bookmark":
                    bookmark_files[name] = f
                elif f["source"] == "session":
                    session_files[name] = f
        else:
            if self.config:
                if self.config.documents:
                    for name, doc in self.config.documents.items():
                        uri = ""
                        if doc.path:
                            uri = f"file://{doc.path}"
                        elif doc.url:
                            uri = doc.url
                        config_files[name] = {
                            "uri": uri,
                            "description": doc.description or "",
                            "file_type": "document",
                        }
                for name, db in self.config.databases.items():
                    if db.type in ("csv", "json", "jsonl", "parquet", "arrow", "feather"):
                        path = db.path or db.uri or ""
                        config_files[name] = {
                            "uri": f"file://{path}" if not path.startswith(("file://", "http")) else path,
                            "description": db.description or "",
                            "file_type": db.type,
                        }
            bookmarks = BookmarkStore()
            bookmark_files = bookmarks.list_files()

        has_any = config_files or bookmark_files or session_files
        if not has_any:
            self.console.print("[dim]No files configured.[/dim]")
            return

        if config_files:
            self.console.print(f"\n[bold]Config Files[/bold] ({len(config_files)})")
            for name, f in config_files.items():
                file_type = f.get("file_type", "file")
                self.console.print(f"  [cyan]{name}[/cyan] [dim]({file_type})[/dim]")
                if f.get("description"):
                    self.console.print(f"    {f['description']}")
                self.console.print(f"    [dim]{f['uri']}[/dim]")

        self._print_file_section("Bookmarked Files", bookmark_files)
        self._print_file_section("Session Files", session_files)

    def _print_file_section(self, title: str, files: dict) -> None:
        """Print a section of files with auth status and URIs."""
        if not files:
            return
        self.console.print(f"\n[bold]{title}[/bold] ({len(files)})")
        for name, f in files.items():
            auth_status = " [auth]" if f.get("auth") else ""
            self.console.print(f"  [cyan]{name}[/cyan]{auth_status}")
            if f.get("description"):
                self.console.print(f"    {f['description']}")
            self.console.print(f"    [dim]{f['uri']}[/dim]")

    def _mask_credentials(self, uri: str) -> str:
        """Mask credentials in a URI for display."""
        return re.sub(r'://([^:]+):([^@]+)@', r'://\1:***@', uri)

    def _display_outputs(self) -> None:
        """Display any pending outputs from artifact saves."""
        from constat.visualization.output import get_pending_outputs
        outputs = get_pending_outputs()
        if not outputs:
            return

        self.console.print()
        self.console.print("[bold]Outputs:[/bold]")
        for output in outputs:
            file_uri = output["file_uri"]
            desc = output.get("description", "")
            file_type = output.get("type", "")
            type_hint = f" [dim]({file_type})[/dim]" if file_type else ""
            self.console.print(f"  [cyan]{desc}[/cyan]{type_hint}")
            self.console.print(f"    {file_uri}")

    def _show_code(self, step_arg: str = "") -> None:
        """Show generated code for steps."""
        if not self.api.session or not self.api.session.datastore:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        entries = self.api.session.datastore.get_scratchpad()
        if not entries:
            self.console.print("[dim]No steps executed yet.[/dim]")
            return

        if step_arg:
            try:
                step_num = int(step_arg)
                entry = next((e for e in entries if e["step_number"] == step_num), None)
                if not entry:
                    self.console.print(f"[yellow]Step {step_num} not found.[/yellow]")
                    return
                self.console.print(f"\n[bold]Step {step_num}:[/bold] {entry['goal']}")
                if entry["code"]:
                    self.console.print(Syntax(entry["code"], "python", theme="monokai", line_numbers=True))
                else:
                    self.console.print("[dim]No code stored for this step.[/dim]")
            except ValueError:
                self.console.print("[red]Invalid step number.[/red]")
        else:
            for entry in entries:
                self.console.print(f"\n[bold]Step {entry['step_number']}:[/bold] {entry['goal']}")
                if entry["code"]:
                    self.console.print(Syntax(entry["code"], "python", theme="monokai", line_numbers=True))
                else:
                    self.console.print("[dim]No code stored.[/dim]")

    def _show_state(self) -> None:
        """Show current session state."""
        if not self.api.session:
            self.console.print("[yellow]No active session.[/yellow]")
            return
        state = self.api.session.get_state()
        self.console.print(f"\n[bold]Session:[/bold] {state['session_id']}")
        if state['datastore_tables']:
            self.console.print("[bold]Tables:[/bold]")
            for t in state['datastore_tables']:
                self.console.print(f"  - {t['name']} ({t['row_count']} rows)")

    def _refresh_metadata(self) -> None:
        """Refresh database metadata, documents, and preload cache."""
        if not self.api.session:
            self.console.print("[yellow]No active session.[/yellow]")
            return
        self.display.start_spinner("Refreshing metadata and documents...")
        try:
            stats = self.api.session.refresh_metadata()
            self.display.stop_spinner()

            parts = ["[green]Refreshed:[/green]"]

            if stats.get("preloaded_tables", 0) > 0:
                parts.append(f"{stats['preloaded_tables']} tables preloaded")

            doc_stats = stats.get("documents", {})
            if doc_stats:
                doc_parts = []
                if doc_stats.get("added", 0) > 0:
                    doc_parts.append(f"{doc_stats['added']} added")
                if doc_stats.get("updated", 0) > 0:
                    doc_parts.append(f"{doc_stats['updated']} updated")
                if doc_stats.get("removed", 0) > 0:
                    doc_parts.append(f"{doc_stats['removed']} removed")
                if doc_stats.get("unchanged", 0) > 0:
                    doc_parts.append(f"{doc_stats['unchanged']} unchanged")
                if doc_parts:
                    parts.append(f"docs: {', '.join(doc_parts)}")

            self.console.print(" ".join(parts) if len(parts) > 1 else "[green]Metadata refreshed.[/green]")
        except Exception as e:
            self.display.stop_spinner()
            self.console.print(f"[red]Error refreshing metadata:[/red] {e}")
