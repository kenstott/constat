"""Command-line interface for Constat."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.status import Status

from constat.core.config import Config, DatabaseCredentials
from constat.session import Session, SessionConfig
from constat.feedback import FeedbackDisplay, SessionFeedbackHandler
from constat.repl import InteractiveREPL


console = Console()


def create_progress_callback(status: Status):
    """Create a progress callback that updates a Rich Status."""
    stage_messages = {
        "connecting": "Connecting to data sources",
        "introspecting": "Introspecting schemas",
        "indexing": "Building search index",
    }

    def callback(stage: str, current: int, total: int, detail: str):
        base_msg = stage_messages.get(stage, stage)
        if total > 1:
            status.update(f"{base_msg} ({current}/{total}): [dim]{detail}[/dim]")
        else:
            status.update(f"{base_msg}: [dim]{detail}[/dim]")

    return callback


@click.group()
@click.version_option(version="0.1.0", prog_name="constat")
def cli():
    """Constat - Multi-Step AI Reasoning Engine.

    A system for LLM-powered data analysis with verifiable, auditable logic.

    \b
    Quick start:
        constat solve "What are the top 5 customers?" --config config.yaml
        constat repl --config config.yaml
    """
    pass


@cli.command()
@click.argument("problem")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to config YAML file.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed output including generated code.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Write output to file instead of stdout.",
)
def solve(problem: str, config: str, verbose: bool, output: Optional[str]):
    """Solve a problem with multi-step planning.

    \b
    Examples:
        constat solve "What are the top selling products?" -c config.yaml
        constat solve "Compare Q1 vs Q2 revenue" -c config.yaml -v
    """
    try:
        cfg = Config.from_yaml(config)
    except Exception as e:
        console.print(f"[red]Config error:[/red] {e}")
        sys.exit(1)

    display = FeedbackDisplay(console=console, verbose=verbose)
    session_config = SessionConfig(verbose=verbose)

    # Initialize session with progress feedback
    with console.status("[bold]Initializing...", spinner="dots") as status:
        progress_cb = create_progress_callback(status)
        session = Session(cfg, session_config=session_config, progress_callback=progress_cb)
    console.print("[green]Ready[/green]")

    # Wire up feedback
    handler = SessionFeedbackHandler(display)
    session.on_event(handler.handle_event)

    display.set_problem(problem)

    try:
        result = session.solve(problem)

        # Show plan
        if result.get("plan"):
            plan = result["plan"]
            steps = [{"number": s.number, "goal": s.goal} for s in plan.steps]
            display.show_plan(steps)

        if result.get("success"):
            total_duration = sum(r.duration_ms for r in result.get("results", []))
            display.show_summary(True, len(result.get("results", [])), total_duration)

            if output:
                Path(output).write_text(result.get("output", ""))
                console.print(f"\n[dim]Output written to {output}[/dim]")
            else:
                display.show_output(result.get("output", ""))

            display.show_tables(result.get("datastore_tables", []))

        else:
            display.show_summary(False, 0, 0)
            console.print(f"\n[red]Error:[/red] {result.get('error')}")
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to config YAML file.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed output including generated code.",
)
@click.option(
    "--problem", "-p",
    help="Initial problem to solve.",
)
def repl(config: str, verbose: bool, problem: Optional[str]):
    """Start interactive REPL session.

    The REPL allows you to:
    - Ask questions about your data
    - Follow up with additional questions (context preserved)
    - Inspect tables and state
    - Resume previous sessions

    \b
    Examples:
        constat repl -c config.yaml
        constat repl -c config.yaml -p "Show me the sales data"
    """
    try:
        cfg = Config.from_yaml(config)
    except Exception as e:
        console.print(f"[red]Config error:[/red] {e}")
        sys.exit(1)

    # Initialize REPL with progress feedback
    with console.status("[bold]Initializing...", spinner="dots") as status:
        progress_cb = create_progress_callback(status)
        interactive = InteractiveREPL(cfg, verbose=verbose, progress_callback=progress_cb)
    console.print("[green]Ready[/green]\n")

    try:
        interactive.run(initial_problem=problem)
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option(
    "--limit", "-n",
    default=10,
    help="Number of sessions to show.",
)
@click.option("--user", "-u", default="default", help="User ID to list sessions for")
def history(limit: int, user: str):
    """List recent sessions.

    Shows session IDs, timestamps, and status for recent sessions.
    Use 'constat resume <id>' to continue a previous session.
    """
    from constat.storage.history import SessionHistory

    hist = SessionHistory(user_id=user)
    sessions = hist.list_sessions(limit=limit)

    if not sessions:
        console.print("[dim]No previous sessions.[/dim]")
        return

    from rich.table import Table
    table = Table(title="Recent Sessions", show_header=True)
    table.add_column("Session ID", style="cyan")
    table.add_column("Created")
    table.add_column("Status")
    table.add_column("Queries")
    table.add_column("Databases")

    for s in sessions:
        table.add_row(
            s.session_id[:16] + "...",
            s.created_at[:16] if s.created_at else "",
            s.status,
            str(s.total_queries),
            ", ".join(s.databases[:2]) + ("..." if len(s.databases) > 2 else ""),
        )

    console.print(table)
    console.print("\n[dim]Use 'constat resume <id>' to continue a session.[/dim]")


@cli.command()
@click.argument("session_id")
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to config YAML file.",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed output.",
)
def resume(session_id: str, config: str, verbose: bool):
    """Resume a previous session.

    Continues an interrupted or completed session, preserving all context,
    tables, and state from previous work.

    \b
    Examples:
        constat resume abc123 -c config.yaml
    """
    try:
        cfg = Config.from_yaml(config)
    except Exception as e:
        console.print(f"[red]Config error:[/red] {e}")
        sys.exit(1)

    interactive = InteractiveREPL(cfg, verbose=verbose)

    # Resume the session
    if not interactive.session:
        interactive.session = interactive._create_session()

    if interactive.session.resume(session_id):
        console.print(f"[green]Resumed session:[/green] {session_id}")
        interactive._show_state()
        interactive.run()
    else:
        console.print(f"[red]Session not found:[/red] {session_id}")
        sys.exit(1)


@cli.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to config YAML file.",
)
def validate(config: str):
    """Validate a config file.

    Checks that:
    - Config file is valid YAML
    - Required fields are present
    - Environment variables are set
    - Database connections work

    \b
    Examples:
        constat validate -c config.yaml
    """
    from rich.table import Table

    console.print(f"Validating: {config}\n")

    # Load config
    try:
        cfg = Config.from_yaml(config)
        console.print("[green]OK[/green] Config file parsed")
    except Exception as e:
        console.print(f"[red]FAIL[/red] Config parsing: {e}")
        sys.exit(1)

    # Check LLM config
    if cfg.llm.api_key:
        console.print("[green]OK[/green] LLM API key configured")
    else:
        console.print("[yellow]WARN[/yellow] LLM API key not set")

    # Check databases
    console.print(f"\nDatabases ({len(cfg.databases)}):")

    from constat.catalog.schema_manager import SchemaManager
    schema_manager = SchemaManager(cfg)

    for db in cfg.databases:
        try:
            schema_manager._connect_database(db)
            table_count = len(schema_manager.get_tables_for_db(db.name))
            console.print(f"  [green]OK[/green] {db.name}: {table_count} tables")
        except Exception as e:
            console.print(f"  [red]FAIL[/red] {db.name}: {e}")

    console.print()


@cli.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to config YAML file.",
)
def schema(config: str):
    """Show database schema overview.

    Displays all tables, their columns, and relationships.
    """
    try:
        cfg = Config.from_yaml(config)
    except Exception as e:
        console.print(f"[red]Config error:[/red] {e}")
        sys.exit(1)

    from constat.catalog.schema_manager import SchemaManager

    with console.status("[bold]Loading schema...", spinner="dots") as status:
        progress_cb = create_progress_callback(status)
        schema_manager = SchemaManager(cfg)
        schema_manager.initialize(progress_callback=progress_cb)

    overview = schema_manager.get_overview()
    console.print("\n" + overview)


@cli.command()
def init():
    """Create a sample config file.

    Generates config.yaml in the current directory with example settings.
    """
    sample_config = '''# Constat Configuration
# See documentation for all options: https://github.com/constat/constat

# LLM Configuration
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${ANTHROPIC_API_KEY}

  # Optional: Use different models for different tasks
  # tiers:
  #   planning: claude-sonnet-4-20250514
  #   codegen: claude-sonnet-4-20250514
  #   simple: claude-3-5-haiku-20241022

# Database Connections
databases:
  - name: main
    uri: sqlite:///data/sample.db
    description: Main application database

# Optional: Global context for all databases
# databases_description: |
#   These databases represent a retail company's data systems.

# Optional: Domain context for the LLM
# system_prompt: |
#   You are analyzing data for a retail company.
#   Key concepts:
#   - customer_tier: gold/silver/bronze
#   - revenue: aggregate by SUM(amount)

# Execution Settings
execution:
  timeout_seconds: 60
  max_retries: 10
  # allowed_imports:
  #   - pandas
  #   - numpy
  #   - scipy

# Optional: Artifact storage
# storage:
#   artifact_store_uri: sqlite:///~/.constat/artifacts.db
'''

    config_path = Path("config.yaml")

    if config_path.exists():
        if not click.confirm("config.yaml already exists. Overwrite?"):
            console.print("[dim]Aborted.[/dim]")
            return

    config_path.write_text(sample_config)
    console.print(f"[green]Created:[/green] {config_path}")
    console.print("\n[dim]Edit the file to configure your databases and API key.[/dim]")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
