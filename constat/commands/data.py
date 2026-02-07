# Copyright (c) 2025 Kenneth Stott
#
# Data exploration commands - tables, show, query, code, artifacts, export.

"""Data exploration commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from constat.commands.base import (
    CommandContext,
    CommandResult,
    TableResult,
    ListResult,
    TextResult,
    ErrorResult,
)


def tables_command(ctx: CommandContext) -> TableResult:
    """List available tables in the datastore."""
    if not ctx.has_datastore:
        return TableResult(
            success=True,
            title="Tables",
            columns=["Name", "Rows", "Step"],
            rows=[],
            footer="No tables yet. Run a query to create tables.",
        )

    tables = ctx.session.datastore.list_tables()

    rows = []
    for t in tables:
        name = t.get("name", "unknown")
        row_count = t.get("row_count", 0)
        step = t.get("step_number", "-")
        rows.append([name, row_count, step])

    return TableResult(
        success=True,
        title="Tables",
        columns=["Name", "Rows", "Step"],
        rows=rows,
        footer=f"{len(rows)} table(s)" if rows else "No tables yet.",
    )


def show_command(ctx: CommandContext) -> CommandResult:
    """Show contents of a table."""
    table_name = ctx.args.strip()

    if not table_name:
        return ErrorResult(error="Usage: /show <table_name>")

    if not ctx.has_datastore:
        return ErrorResult(error="No datastore available.")

    try:
        df = ctx.session.datastore.load_dataframe(table_name)

        # Convert to table result
        columns = list(df.columns)
        rows = df.head(100).to_numpy().tolist()  # Limit to 100 rows

        footer = None
        if len(df) > 100:
            footer = f"Showing 100 of {len(df)} rows"

        return TableResult(
            success=True,
            title=table_name,
            columns=columns,
            rows=rows,
            footer=footer,
        )
    except Exception as e:
        return ErrorResult(error=f"Error loading table '{table_name}': {e}")


def query_command(ctx: CommandContext) -> CommandResult:
    """Run SQL query on the datastore."""
    sql = ctx.args.strip()

    if not sql:
        return ErrorResult(error="Usage: /query <sql>")

    if not ctx.has_datastore:
        return ErrorResult(error="No datastore available.")

    try:
        df = ctx.session.datastore.query(sql)

        columns = list(df.columns)
        rows = df.head(100).to_numpy().tolist()

        footer = None
        if len(df) > 100:
            footer = f"Showing 100 of {len(df)} rows"

        return TableResult(
            success=True,
            title="Query Result",
            columns=columns,
            rows=rows,
            footer=footer,
        )
    except Exception as e:
        return ErrorResult(error=f"Query error: {e}")


def code_command(ctx: CommandContext) -> CommandResult:
    """Show generated code for steps."""
    step_arg = ctx.args.strip()

    if not ctx.has_plan:
        return ErrorResult(error="No plan available. Run a query first.")

    plan = ctx.session.plan
    steps = plan.steps if plan else []

    if not steps:
        return TextResult(content="No steps in current plan.")

    # Parse step number if provided
    step_num: Optional[int] = None
    if step_arg:
        try:
            step_num = int(step_arg)
        except ValueError:
            return ErrorResult(error=f"Invalid step number: {step_arg}")

    # Collect code blocks
    code_blocks: list[dict[str, Any]] = []

    for step in steps:
        sn = step.number if hasattr(step, "number") else steps.index(step) + 1

        if step_num is not None and sn != step_num:
            continue

        code = getattr(step, "code", None) or getattr(step, "generated_code", None)
        if code:
            code_blocks.append({
                "step": sn,
                "goal": getattr(step, "goal", f"Step {sn}"),
                "code": code,
                "language": "python",
            })

    if not code_blocks:
        if step_num:
            return TextResult(content=f"No code found for step {step_num}.")
        return TextResult(content="No generated code available.")

    # Return as list of code items
    return ListResult(
        success=True,
        title="Generated Code",
        items=code_blocks,
    )


def artifacts_command(ctx: CommandContext) -> CommandResult:
    """List artifacts from the session."""
    show_all = ctx.args.strip().lower() == "all"

    if not ctx.has_datastore:
        return ListResult(
            success=True,
            title="Artifacts",
            items=[],
            empty_message="No artifacts yet.",
        )

    try:
        artifacts = ctx.session.datastore.list_artifacts()

        items = []
        for a in artifacts:
            name = a.get("name", "")
            atype = a.get("type", "")

            # Determine if this is a key result (vs intermediate artifact)
            # Intermediate artifacts have auto-generated names like "artifact_N_type"
            is_intermediate = (
                name.startswith("artifact_") and
                "_" in name[9:] and
                atype in ("code", "output", "error")
            )

            # Key results have meaningful names or titles
            is_key = not is_intermediate or a.get("title")

            if not show_all and not is_key:
                continue

            items.append({
                "id": a.get("id"),
                "name": name,
                "type": atype,
                "step": a.get("step_number", "-"),
                "title": a.get("title"),
                "description": a.get("description"),
            })

        return ListResult(
            success=True,
            title="Artifacts" + (" (all)" if show_all else ""),
            items=items,
            empty_message="No artifacts yet.",
        )
    except Exception as e:
        return ErrorResult(error=f"Error listing artifacts: {e}")


def export_command(ctx: CommandContext) -> CommandResult:
    """Export a table to CSV or XLSX file."""
    args = ctx.args.strip().split(maxsplit=1)

    if not args:
        return ErrorResult(error="Usage: /export <table_name> [filename]")

    table_name = args[0]
    filename = args[1] if len(args) > 1 else None

    if not ctx.has_datastore:
        return ErrorResult(error="No datastore available.")

    try:
        df = ctx.session.datastore.load_dataframe(table_name)

        # Determine output path
        if filename:
            output_path = Path(filename)
        else:
            output_path = Path(f"{table_name}.csv")

        # Export based on extension
        ext = output_path.suffix.lower()
        if ext == ".xlsx":
            df.to_excel(output_path, index=False)
        elif ext == ".parquet":
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)

        return TextResult(
            success=True,
            content=f"Exported {len(df)} rows to {output_path}",
        )
    except Exception as e:
        return ErrorResult(error=f"Export error: {e}")


def download_code_command(ctx: CommandContext) -> CommandResult:
    """Download all step code as a standalone Python script.

    Creates a single Python file with:
    - Standard imports (pandas, numpy, duckdb)
    - A function for each executed step
    - A main() function that calls each step in order
    """
    filename = ctx.args.strip() or "analysis_script.py"

    # Ensure .py extension
    if not filename.endswith(".py"):
        filename += ".py"

    # Get step codes from history
    if not ctx.session.session_id:
        return ErrorResult(error="No active session. Run a query first.")

    step_codes = ctx.session.history.list_step_codes(ctx.session.session_id)

    if not step_codes:
        # Fall back to in-memory plan steps
        if ctx.has_plan and ctx.session.plan:
            for step in ctx.session.plan.steps:
                code = getattr(step, "code", None)
                if code:
                    step_codes.append({
                        "step_number": step.number,
                        "goal": step.goal,
                        "code": code,
                    })

    if not step_codes:
        return ErrorResult(error="No code available. Run a query first.")

    # Build the script
    script_lines = [
        '#!/usr/bin/env python3',
        '"""',
        'Auto-generated analysis script from Constat session.',
        '',
        'This script contains the code executed during the analysis session.',
        'Each step is wrapped in a function that can be run independently.',
        '"""',
        '',
        'import pandas as pd',
        'import numpy as np',
        'import duckdb',
        '',
        '',
        '# ============================================================================',
        '# Database and Storage Setup',
        '# ============================================================================',
        '',
        '# Create a DuckDB connection for storing intermediate results',
        '_conn = duckdb.connect(":memory:")',
        '',
        '',
        'def save_dataframe(name: str, df: pd.DataFrame) -> None:',
        '    """Save a DataFrame to the in-memory store."""',
        '    _conn.register(name, df)',
        '    _conn.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM {name}")',
        '',
        '',
        'def load_dataframe(name: str) -> pd.DataFrame:',
        '    """Load a DataFrame from the in-memory store."""',
        '    return _conn.execute(f"SELECT * FROM {name}").df()',
        '',
        '',
        'def query(sql: str) -> pd.DataFrame:',
        '    """Run a SQL query on stored data."""',
        '    return _conn.execute(sql).df()',
        '',
        '',
        '# ============================================================================',
        '# Step Functions',
        '# ============================================================================',
        '',
    ]

    # Add each step as a function
    for step in sorted(step_codes, key=lambda s: s["step_number"]):
        step_num = step["step_number"]
        goal = step["goal"].replace('"', '\\"').replace('\n', ' ')
        code = step["code"]

        # Clean up the code - remove the header comment if it exists
        code_lines = code.split('\n')
        if code_lines and code_lines[0].startswith('# Step'):
            code_lines = code_lines[1:]
        if code_lines and code_lines[0].strip() == '':
            code_lines = code_lines[1:]

        # Indent the code for the function body
        indented_code = '\n'.join('    ' + line if line.strip() else '' for line in code_lines)

        script_lines.extend([
            f'def step_{step_num}():',
            f'    """Step {step_num}: {goal}"""',
            indented_code,
            '',
            '',
        ])

    # Add main function
    script_lines.extend([
        '# ============================================================================',
        '# Main Execution',
        '# ============================================================================',
        '',
        'def main():',
        '    """Run all analysis steps in order."""',
    ])

    for step in sorted(step_codes, key=lambda s: s["step_number"]):
        step_num = step["step_number"]
        goal = step["goal"][:50] + "..." if len(step["goal"]) > 50 else step["goal"]
        script_lines.append(f'    print("Step {step_num}: {goal}")')
        script_lines.append(f'    step_{step_num}()')
        script_lines.append('')

    script_lines.extend([
        '    print("\\nAnalysis complete.")',
        '',
        '',
        'if __name__ == "__main__":',
        '    main()',
        '',
    ])

    # Write the file
    script_content = '\n'.join(script_lines)
    output_path = Path(filename)

    try:
        output_path.write_text(script_content)
        return TextResult(
            success=True,
            content=f"Saved {len(step_codes)} step(s) to {output_path.absolute()}",
        )
    except Exception as e:
        return ErrorResult(error=f"Error writing file: {e}")
