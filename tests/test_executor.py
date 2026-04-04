# Copyright (c) 2025 Kenneth Stott
# Canary: 739d8c85-a64d-4a2b-aaac-a04dd8353b37
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for Python executor - compile and runtime error handling."""

import pytest

from constat.execution.executor import PythonExecutor, format_error_for_retry


@pytest.fixture
def executor() -> PythonExecutor:
    return PythonExecutor()


class TestCompilation:
    """Test compile-time error detection."""

    def test_valid_code_compiles(self, executor: PythonExecutor):
        """Valid Python code compiles successfully."""
        code = "x = 1 + 2\nprint(x)"
        compiled, error = executor.compile(code)
        assert compiled is not None
        assert error is None

    def test_syntax_error_detected(self, executor: PythonExecutor):
        """Syntax errors are caught at compile time."""
        code = "if True\n    print('missing colon')"
        compiled, error = executor.compile(code)
        assert compiled is None
        assert error is not None
        assert error.line == 1

    def test_indentation_error_detected(self, executor: PythonExecutor):
        """Indentation errors are caught."""
        code = "def foo():\nprint('not indented')"
        compiled, error = executor.compile(code)
        assert compiled is None
        assert error is not None


class TestImportValidation:
    """Test import allowlist enforcement."""

    def test_allowed_import_passes(self, executor: PythonExecutor):
        """Allowed imports pass validation."""
        code = "import pandas as pd\nimport numpy as np"
        error = executor.validate_imports(code)
        assert error is None

    def test_disallowed_import_blocked(self, executor: PythonExecutor):
        """Disallowed imports are blocked."""
        code = "import os\nos.system('ls')"
        error = executor.validate_imports(code)
        assert error is not None
        assert "os" in error

    def test_disallowed_from_import_blocked(self, executor: PythonExecutor):
        """From imports of disallowed modules are blocked."""
        code = "from subprocess import run"
        error = executor.validate_imports(code)
        assert error is not None
        assert "subprocess" in error


class TestExecution:
    """Test code execution and output capture."""

    def test_stdout_captured(self, executor: PythonExecutor):
        """Stdout is captured."""
        code = "print('hello world')"
        result = executor.execute(code)
        assert result.success
        assert "hello world" in result.stdout

    def test_multiple_prints_captured(self, executor: PythonExecutor):
        """Multiple print statements are captured."""
        code = "print('line 1')\nprint('line 2')"
        result = executor.execute(code)
        assert result.success
        assert "line 1" in result.stdout
        assert "line 2" in result.stdout

    def test_injected_variables_accessible(self, executor: PythonExecutor):
        """Injected variables are accessible in code."""
        code = "print(f'Value is {my_var}')"
        result = executor.execute(code, {"my_var": 42})
        assert result.success
        assert "42" in result.stdout

    def test_pandas_available(self, executor: PythonExecutor):
        """Pandas is pre-imported as pd."""
        code = "df = pd.DataFrame({'a': [1,2,3]})\nprint(len(df))"
        result = executor.execute(code)
        assert result.success
        assert "3" in result.stdout

    def test_numpy_available(self, executor: PythonExecutor):
        """Numpy is pre-imported as np."""
        code = "arr = np.array([1,2,3])\nprint(arr.sum())"
        result = executor.execute(code)
        assert result.success
        assert "6" in result.stdout


class TestRuntimeErrors:
    """Test runtime error handling."""

    def test_name_error_caught(self, executor: PythonExecutor):
        """NameError is caught with traceback."""
        code = "print(undefined_variable)"
        result = executor.execute(code)
        assert not result.success
        assert result.runtime_error is not None
        assert "NameError" in result.runtime_error.error
        assert "undefined_variable" in result.runtime_error.error

    def test_type_error_caught(self, executor: PythonExecutor):
        """TypeError is caught with traceback."""
        code = "x = 'string' + 123"
        result = executor.execute(code)
        assert not result.success
        assert result.runtime_error is not None
        assert "TypeError" in result.runtime_error.error

    def test_key_error_caught(self, executor: PythonExecutor):
        """KeyError is caught with traceback."""
        code = "d = {'a': 1}\nprint(d['b'])"
        result = executor.execute(code)
        assert not result.success
        assert result.runtime_error is not None
        assert "KeyError" in result.runtime_error.error

    def test_division_by_zero_caught(self, executor: PythonExecutor):
        """ZeroDivisionError is caught."""
        code = "x = 1 / 0"
        result = executor.execute(code)
        assert not result.success
        assert result.runtime_error is not None
        assert "ZeroDivisionError" in result.runtime_error.error

    def test_traceback_included(self, executor: PythonExecutor):
        """Full traceback is included in error."""
        code = """
def inner():
    raise ValueError("test error")

def outer():
    inner()

outer()
"""
        result = executor.execute(code)
        assert not result.success
        assert "inner" in result.runtime_error.traceback
        assert "outer" in result.runtime_error.traceback
        assert "test error" in result.runtime_error.error

    def test_partial_stdout_captured_before_error(self, executor: PythonExecutor):
        """Stdout before error is still captured."""
        code = "print('before')\nraise Exception('boom')\nprint('after')"
        result = executor.execute(code)
        assert not result.success
        assert "before" in result.stdout
        assert "after" not in result.stdout


class TestEnginePromptConsistency:
    """Test that the engine system prompt matches the actual execution environment."""

    def test_engine_system_prompt_does_not_instruct_store(self):
        """engine_system.md must not instruct LLM to use store.* — store is not injected in engine."""
        from constat.prompts import load_prompt
        prompt = load_prompt("engine_system.md")
        # store.query() and store.create_view() are session-only — not available in QueryEngine
        assert "store.query" not in prompt, (
            "engine_system.md instructs 'store.query()' but store is not in engine execution globals. "
            "LLM will generate code that fails at runtime with NameError: store."
        )
        assert "store.create_view" not in prompt, (
            "engine_system.md instructs 'store.create_view()' but store is not in engine execution globals."
        )

    def test_engine_system_prompt_provides_correct_sql_access(self):
        """engine_system.md must tell LLM to use pd.read_sql(sql, db_<name>.engine)."""
        from constat.prompts import load_prompt
        prompt = load_prompt("engine_system.md")
        # The correct pattern for the engine is pd.read_sql with .engine attribute
        assert "db_<name>.engine" in prompt or "db_chinook.engine" in prompt or ".engine" in prompt, (
            "engine_system.md does not show how to use db_<name>.engine for SQL access. "
            "pd.read_sql(sql, db_<name>.engine) is the correct pattern."
        )

    def test_engine_execution_globals_has_no_store(self):
        """QueryEngine execution globals must not include store — confirming store is absent."""
        from pathlib import Path
        from constat.core.config import Config
        from constat.catalog.schema_manager import SchemaManager
        from constat.execution.engine import QueryEngine

        CHINOOK_DB = Path(__file__).parent.parent / "data" / "chinook.db"
        config = Config(
            databases={"chinook": {"uri": f"sqlite:///{CHINOOK_DB}"}},
        )
        schema_manager = SchemaManager(config)
        schema_manager.initialize()
        engine = QueryEngine(config, schema_manager, max_retries=1)
        globals_dict = engine._get_execution_globals()

        assert "store" not in globals_dict, (
            "store is unexpectedly present in engine execution globals"
        )
        assert "db_chinook" in globals_dict, (
            "db_chinook must be present in engine execution globals"
        )

    def test_engine_db_connection_supports_pd_read_sql_via_engine_attr(self):
        """db_<name>.engine must support pd.read_sql — the correct engine access pattern."""
        import pandas as pd
        from pathlib import Path
        from constat.core.config import Config
        from constat.catalog.schema_manager import SchemaManager
        from constat.execution.engine import QueryEngine

        CHINOOK_DB = Path(__file__).parent.parent / "data" / "chinook.db"
        config = Config(
            databases={"chinook": {"uri": f"sqlite:///{CHINOOK_DB}"}},
        )
        schema_manager = SchemaManager(config)
        schema_manager.initialize()
        engine = QueryEngine(config, schema_manager, max_retries=1)
        globals_dict = engine._get_execution_globals()

        db_chinook = globals_dict["db_chinook"]
        df = pd.read_sql("SELECT TrackId, Name FROM Track LIMIT 3", db_chinook.engine)
        assert len(df) == 3
        assert "TrackId" in df.columns


class TestErrorFormatting:
    """Test error message formatting for LLM retry."""

    def test_format_compile_error(self, executor: PythonExecutor):
        """Compile errors are formatted for retry."""
        code = "if True\n    print('x')"
        result = executor.execute(code)
        formatted = format_error_for_retry(result, code)

        assert "Syntax Error" in formatted
        assert "fix the code" in formatted.lower()

    def test_format_runtime_error(self, executor: PythonExecutor):
        """Runtime errors include traceback."""
        code = "x = undefined"
        result = executor.execute(code)
        formatted = format_error_for_retry(result, code)

        assert "Runtime Error" in formatted
        assert "NameError" in formatted
        assert "Traceback" in formatted

    def test_format_includes_stdout_before_error(self, executor: PythonExecutor):
        """Partial stdout is included in error message."""
        code = "print('got here')\nraise Exception('failed')"
        result = executor.execute(code)
        formatted = format_error_for_retry(result, code)

        assert "got here" in formatted
