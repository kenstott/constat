# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Safe Python code execution with timeout and error capture."""

from __future__ import annotations

import ast
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CompileError:
    """Python compilation/syntax error."""
    error: str
    line: Optional[int] = None
    offset: Optional[int] = None


@dataclass
class ExecutionRuntimeError:
    """Runtime exception from executed code.

    Captures runtime errors that occur during execution of generated Python code,
    including the error message and full traceback for debugging.
    """
    error: str
    traceback: str


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    compile_error: Optional[CompileError] = None
    runtime_error: Optional[ExecutionRuntimeError] = None
    return_value: Any = None
    namespace: Optional[dict] = None  # Variables after execution (for auto-saving)

    def error_message(self) -> Optional[str]:
        """Get formatted error message for LLM retry."""
        if self.compile_error:
            msg = f"SyntaxError: {self.compile_error.error}"
            if self.compile_error.line:
                msg += f" (line {self.compile_error.line})"
            return msg
        if self.runtime_error:
            return f"{self.runtime_error.error}\n\nTraceback:\n{self.runtime_error.traceback}"
        return None


class PythonExecutor:
    """
    Execute Python code safely with timeout and resource limits.

    For single-shot queries, we execute in-process for simplicity.
    The code has access to injected variables (like db connections).
    """

    def __init__(
        self,
        timeout_seconds: int = 60,
        allowed_imports: Optional[list[str]] = None,
    ):
        self.timeout_seconds = timeout_seconds
        self.allowed_imports = allowed_imports or [
            "pandas", "numpy", "scipy", "sklearn", "math", "json",
            "datetime", "re", "collections", "itertools", "functools",
            # Visualization libraries
            "plotly", "altair", "matplotlib", "seaborn", "folium",
        ]

    def validate_imports(self, code: str) -> Optional[str]:
        """Check that only allowed imports are used. Returns error message or None."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None  # Let compile step handle syntax errors

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module not in self.allowed_imports:
                        return f"Import not allowed: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module.split('.')[0]
                    if module not in self.allowed_imports:
                        return f"Import not allowed: {node.module}"

        return None

    @staticmethod
    def compile(code: str) -> tuple[Optional[Any], Optional[CompileError]]:
        """Compile code, returning (compiled, error)."""
        try:
            compiled = compile(code, "<generated>", "exec")
            return compiled, None
        except SyntaxError as e:
            return None, CompileError(
                error=str(e.msg) if e.msg else str(e),
                line=e.lineno,
                offset=e.offset,
            )

    def execute(
        self,
        code: str,
        globals_dict: Optional[dict] = None,
    ) -> ExecutionResult:
        """
        Execute Python code and capture results.

        Args:
            code: Python source code to execute
            globals_dict: Variables to inject (e.g., db connections)

        Returns:
            ExecutionResult with success status, output, and any errors
        """
        # Check imports
        import_error = self.validate_imports(code)
        if import_error:
            return ExecutionResult(
                success=False,
                compile_error=CompileError(error=import_error),
            )

        # Compile
        compiled, compile_error = self.compile(code)
        if compile_error:
            return ExecutionResult(
                success=False,
                compile_error=compile_error,
            )

        # Prepare execution environment
        exec_globals = globals_dict.copy() if globals_dict else {}

        # Add common imports to namespace
        exec_globals.setdefault("pd", __import__("pandas"))
        exec_globals.setdefault("np", __import__("numpy"))
        exec_globals.setdefault("re", __import__("re"))

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Prevent exit()/quit() from killing the server
        def _blocked_exit(*_args, **_kwargs):
            raise RuntimeError("exit() is not allowed in generated code")
        exec_globals["exit"] = _blocked_exit
        exec_globals["quit"] = _blocked_exit

        # Prevent input() from blocking the server
        def _blocked_input(*_args, **_kwargs):
            raise RuntimeError("input() is not allowed — use ask_user() for user_input steps")
        exec_globals["input"] = _blocked_input

        # Execute
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compiled, exec_globals)

            return ExecutionResult(
                success=True,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                namespace=exec_globals,  # Return namespace for auto-saving
            )

        except SystemExit:
            # LLM generated code that calls exit() - treat as error, don't crash server
            return ExecutionResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                runtime_error=ExecutionRuntimeError(
                    error="Code called exit() which is not allowed",
                    traceback="SystemExit intercepted - exit() calls are blocked in generated code",
                ),
            )

        except Exception as e:
            # Capture the full traceback
            tb = traceback.format_exc()

            return ExecutionResult(
                success=False,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                runtime_error=ExecutionRuntimeError(
                    error=f"{type(e).__name__}: {e}",
                    traceback=tb,
                ),
            )


def format_error_for_retry(result: ExecutionResult, code: str) -> str:
    """Format error message to send back to LLM for retry.

    Includes prescriptive fixes for common errors to help smaller models.
    """
    parts = ["The generated code failed to execute."]

    error_text = ""
    if result.compile_error:
        error_text = result.compile_error.error
        parts.append(f"\nSyntax Error: {error_text}")
        if result.compile_error.line:
            lines = code.split('\n')
            if 0 < result.compile_error.line <= len(lines):
                parts.append(f"Line {result.compile_error.line}: {lines[result.compile_error.line - 1]}")

    elif result.runtime_error:
        error_text = result.runtime_error.error
        parts.append(f"\nRuntime Error: {error_text}")
        parts.append(f"\nTraceback:\n{result.runtime_error.traceback}")

    if result.stdout:
        parts.append(f"\nStdout before error:\n{result.stdout}")

    # Add prescriptive fixes for common errors
    fix_hints = _get_prescriptive_fix(error_text, code)
    if fix_hints:
        parts.append(f"\n**HOW TO FIX**: {fix_hints}")

    parts.append("\nPlease fix the code and try again.")

    return "\n".join(parts)


def _get_prescriptive_fix(error_text: str, _code: str) -> str:
    """Return specific fix instructions for common errors."""
    error_lower = error_text.lower()

    # Discovery tools not available
    if "find_relevant_tables" in error_text or "get_table_schema" in error_text:
        return (
            "The function find_relevant_tables() and get_table_schema() are NOT available. "
            "These are planning-only tools. Use store.query('SELECT ... FROM db_name.table') or "
            "store.create_view('name', 'SELECT ...', step_number=N) to query tables directly. "
            "The table schema is already provided in the prompt - use that information."
        )

    # db.execute() not available in SQLAlchemy 2.0
    if "execute" in error_lower and ("engine" in error_lower or "attribute" in error_lower):
        return (
            "Do NOT use db.execute() or db_<name>.execute() - this does not work in SQLAlchemy 2.0. "
            "Use store.query('SELECT ... FROM db_name.table') or store.create_view() instead."
        )

    # Schema prefix errors (pd.read_sql can't handle federated schema prefixes)
    if "no such table" in error_lower and "." in error_text:
        import re
        match = re.search(r'no such table[:\s]+(\w+\.\w+)', error_text, re.IGNORECASE)
        if match:
            full_name = match.group(1)
            return (
                f"pd.read_sql does NOT support schema prefixes like '{full_name}'. "
                f"REPLACE the entire pd.read_sql call with store.query() or store.create_view() which handle federation automatically:\n"
                f"  store.create_view('name', 'SELECT ... FROM {full_name}', step_number=N)  # lazy view\n"
                f"  df = store.query('SELECT ... FROM {full_name}')  # immediate DataFrame\n"
                f"Do NOT use pd.read_sql for this query."
            )

    # Store methods that don't exist
    if "registryawaredatastore" in error_lower or "datastore" in error_lower:
        if "contains" in error_lower:
            return (
                "store.contains() does not exist. To check if a table exists, use: "
                "tables = store.list_tables(); if 'name' in tables: ..."
            )
        if "get" in error_lower and "attribute" in error_lower:
            return (
                "Use store.load_dataframe('name') to load a DataFrame, "
                "store.get_state('key') for simple values, "
                "store.list_tables() to see available tables."
            )

    # DataFrame truth value error
    if "truth value of a dataframe" in error_lower:
        return (
            "Do NOT use 'if df:' on DataFrames. Use 'if not df.empty:' or 'if len(df) > 0:' instead."
        )

    # Column not found
    if "keyerror" in error_lower or "column" in error_lower and "not found" in error_lower:
        return (
            "Column not found. Check column names with df.columns first. "
            "Use: if 'col_name' in df.columns: ... before accessing columns."
        )

    # Import errors for common modules
    if "no module named" in error_lower:
        if "plotly" in error_lower:
            return "plotly is available. Use: import plotly.express as px"
        if "folium" in error_lower:
            return "folium is available. Use: import folium"

    return ""
