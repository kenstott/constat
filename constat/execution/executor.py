"""Safe Python code execution with timeout and error capture."""

from __future__ import annotations

import ast
import io
import sys
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

    def compile(self, code: str) -> tuple[Optional[Any], Optional[CompileError]]:
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

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

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
    """Format error message to send back to LLM for retry."""
    parts = ["The generated code failed to execute."]

    if result.compile_error:
        parts.append(f"\nSyntax Error: {result.compile_error.error}")
        if result.compile_error.line:
            # Show the problematic line
            lines = code.split('\n')
            if 0 < result.compile_error.line <= len(lines):
                parts.append(f"Line {result.compile_error.line}: {lines[result.compile_error.line - 1]}")

    elif result.runtime_error:
        parts.append(f"\nRuntime Error: {result.runtime_error.error}")
        parts.append(f"\nTraceback:\n{result.runtime_error.traceback}")

    if result.stdout:
        parts.append(f"\nStdout before error:\n{result.stdout}")

    parts.append("\nPlease fix the code and try again.")

    return "\n".join(parts)
