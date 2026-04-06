# Copyright (c) 2025 Kenneth Stott
# Canary: 1dcb97c6-8791-4f5a-a805-4b8312a18447
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for environment variable substitution in config."""

from __future__ import annotations

import pytest

from constat.core.config import Config


class TestEnvVarSubstitution:
    """Comprehensive tests for environment variable substitution in config."""

    # ============================================================
    # P0 - Critical: Values that could break parsing or security
    # ============================================================

    def test_env_var_value_with_yaml_special_chars_colon(self, tmp_path, monkeypatch):
        """Env var value containing YAML special char ':' is handled."""
        monkeypatch.setenv("TEST_VAR", "host:port:extra")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${TEST_VAR}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "host:port:extra"

    def test_env_var_value_with_yaml_special_chars_quotes(self, tmp_path, monkeypatch):
        """Env var value containing quotes doesn't break YAML parsing."""
        monkeypatch.setenv("TEST_VAR", 'value with "double" and \'single\' quotes')

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${TEST_VAR}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == 'value with "double" and \'single\' quotes'

    def test_env_var_value_with_newlines_breaks_yaml(self, tmp_path, monkeypatch):
        """KNOWN LIMITATION: Env var value containing newlines breaks YAML parsing.

        When an env var value contains literal newlines and is substituted into
        unquoted YAML, it corrupts the YAML structure. This is a security/reliability
        concern - env var values should be validated or the field should be quoted.
        """
        monkeypatch.setenv("TEST_MULTILINE", "line1\nline2\nline3")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${TEST_MULTILINE}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        # Newlines in env var break YAML parsing - this is expected to fail
        with pytest.raises(Exception):  # yaml.scanner.ScannerError
            Config.from_yaml(str(config_file))

    def test_env_var_value_with_newlines_works_when_quoted(self, tmp_path, monkeypatch):
        """Env var value with newlines works if YAML field is quoted."""
        monkeypatch.setenv("TEST_MULTILINE", "line1\nline2\nline3")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: "${TEST_MULTILINE}"

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        # When the YAML value is quoted, multiline substitution works
        config = Config.from_yaml(str(config_file))
        assert "line1" in config.llm.api_key
        assert "line2" in config.llm.api_key

    def test_env_var_value_with_yaml_block_indicators(self, tmp_path, monkeypatch):
        """Env var value with YAML block indicators (| or >) doesn't break parsing."""
        monkeypatch.setenv("TEST_VAR", "value | with | pipes")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${TEST_VAR}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "value | with | pipes"

    # ============================================================
    # P1 - High: Common patterns and edge cases
    # ============================================================

    def test_empty_env_var_value_becomes_none(self, tmp_path, monkeypatch):
        """Empty string env var value is substituted but YAML parses it as None.

        This is YAML behavior: an unquoted empty value becomes null/None.
        Use quotes if you need to preserve empty string: api_key: "${EMPTY_VAR}"
        """
        monkeypatch.setenv("EMPTY_VAR", "")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${EMPTY_VAR}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        # Empty string does not raise, but YAML parses empty as None
        config = Config.from_yaml(str(config_file))
        # YAML behavior: unquoted empty value becomes None
        assert config.llm.api_key is None

    def test_empty_env_var_value_preserved_when_quoted(self, tmp_path, monkeypatch):
        """Empty string env var value is preserved when YAML field is quoted."""
        monkeypatch.setenv("EMPTY_VAR", "")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: "${EMPTY_VAR}"

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        # When quoted, empty string is preserved
        assert config.llm.api_key == ""

    def test_multiple_env_vars_in_single_field(self, tmp_path, monkeypatch):
        """Multiple env vars in one field are all substituted."""
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("DB_PORT", "5432")
        monkeypatch.setenv("DB_NAME", "mydb")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key

databases:
  main:
    uri: postgresql://${DB_HOST}:${DB_PORT}/${DB_NAME}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.databases["main"].uri == "postgresql://localhost:5432/mydb"

    def test_adjacent_env_vars(self, tmp_path, monkeypatch):
        """Adjacent env vars without separator are substituted correctly."""
        monkeypatch.setenv("PREFIX", "pre")
        monkeypatch.setenv("SUFFIX", "suf")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${PREFIX}${SUFFIX}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "presuf"

    def test_env_var_value_containing_dollar_sign(self, tmp_path, monkeypatch):
        """Env var value containing $ doesn't cause issues."""
        monkeypatch.setenv("DOLLAR_VAR", "price is $100")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${DOLLAR_VAR}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "price is $100"

    def test_env_var_value_containing_env_var_syntax(self, tmp_path, monkeypatch):
        """Env var value containing ${...} pattern is NOT double-substituted."""
        monkeypatch.setenv("META_VAR", "value is ${OTHER_VAR}")
        # Note: OTHER_VAR is intentionally NOT set

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${META_VAR}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        # The substitution happens once. After substituting META_VAR,
        # the result contains "${OTHER_VAR}" as literal text.
        # It should NOT try to substitute ${OTHER_VAR} again.
        # Current implementation does a single regex.sub pass, so this should work.
        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "value is ${OTHER_VAR}"

    def test_env_var_name_with_underscores(self, tmp_path, monkeypatch):
        """Env var names with underscores are handled correctly."""
        monkeypatch.setenv("MY_LONG_VARIABLE_NAME_HERE", "works")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${MY_LONG_VARIABLE_NAME_HERE}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "works"

    def test_env_var_name_with_numbers(self, tmp_path, monkeypatch):
        """Env var names with numbers are handled correctly."""
        monkeypatch.setenv("VAR123", "numbered")
        monkeypatch.setenv("123VAR", "starts_with_number")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${VAR123}

databases:
  main:
    uri: sqlite:///${123VAR}.db
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "numbered"
        assert config.databases["main"].uri == "sqlite:///starts_with_number.db"

    def test_one_missing_among_multiple_env_vars(self, tmp_path, monkeypatch):
        """When one of multiple env vars is missing, error specifies which one."""
        monkeypatch.setenv("PRESENT_VAR", "exists")
        # MISSING_VAR is intentionally not set

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${PRESENT_VAR}-${MISSING_VAR}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        with pytest.raises(ValueError, match="MISSING_VAR"):
            Config.from_yaml(str(config_file))

    # ============================================================
    # P2 - Medium: Edge cases and potential confusion
    # ============================================================

    def test_partial_syntax_dollar_without_braces(self, tmp_path):
        """$VAR (without braces) is NOT substituted - only ${VAR} works."""
        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: $NOT_SUBSTITUTED

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        # Should not raise and should keep literal $NOT_SUBSTITUTED
        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "$NOT_SUBSTITUTED"

    def test_incomplete_syntax_unclosed_brace_not_matched(self, tmp_path, monkeypatch):
        """Unclosed brace with non-word characters is not matched by the regex.

        The regex pattern [A-Za-z0-9_]+ only matches word characters, so
        ${MYVAR" is not a valid env var reference (the quote breaks the match).
        The literal string is preserved as-is.
        """
        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: "${MYVAR"

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        # The regex does not match ${MYVAR" because " is not [A-Za-z0-9_]
        # So the literal string is preserved (YAML strips outer quotes)
        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "${MYVAR"

    def test_double_dollar_not_special(self, tmp_path, monkeypatch):
        """$${VAR} - double dollar is not escape syntax."""
        monkeypatch.setenv("VAR", "value")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: $${VAR}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        # ${VAR} is substituted, leaving "$value"
        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "$value"

    def test_default_value_syntax_not_supported(self, tmp_path, monkeypatch):
        """${VAR:-default} syntax is NOT supported - non-word chars break the match.

        The regex [A-Za-z0-9_]+ only matches word characters, so the ':'
        in ${VAR_WITH_DEFAULT:-fallback} prevents the pattern from matching.
        The literal string is preserved as-is.
        """
        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${VAR_WITH_DEFAULT:-fallback}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        # The regex does not match because ':' breaks the word-char pattern
        # The literal string is preserved
        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "${VAR_WITH_DEFAULT:-fallback}"

    def test_env_var_in_quoted_yaml_string(self, tmp_path, monkeypatch):
        """Env var in quoted YAML string is still substituted."""
        monkeypatch.setenv("QUOTED_VAR", "inside_quotes")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: "${QUOTED_VAR}"

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "inside_quotes"

    def test_env_var_with_text_before_and_after(self, tmp_path, monkeypatch):
        """Env var with surrounding text is handled."""
        monkeypatch.setenv("MIDDLE", "center")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: prefix-${MIDDLE}-suffix

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "prefix-center-suffix"

    def test_unicode_in_env_var_value(self, tmp_path, monkeypatch):
        """Unicode characters in env var value are preserved."""
        monkeypatch.setenv("UNICODE_VAR", "hello-\u4e16\u754c-\U0001F600")  # "world" in Chinese + emoji

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${UNICODE_VAR}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert "\u4e16\u754c" in config.llm.api_key  # Chinese chars preserved

    def test_whitespace_only_env_var_value_becomes_none(self, tmp_path, monkeypatch):
        """Whitespace-only env var value becomes None in YAML.

        Similar to empty string, YAML treats unquoted whitespace as null/None.
        Use quotes to preserve: api_key: "${WHITESPACE_VAR}"
        """
        monkeypatch.setenv("WHITESPACE_VAR", "   ")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${WHITESPACE_VAR}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        # YAML behavior: unquoted whitespace-only becomes None
        assert config.llm.api_key is None

    def test_whitespace_only_env_var_value_preserved_when_quoted(self, tmp_path, monkeypatch):
        """Whitespace-only env var value is preserved when YAML field is quoted."""
        monkeypatch.setenv("WHITESPACE_VAR", "   ")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: "${WHITESPACE_VAR}"

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        # When quoted, whitespace is preserved
        assert config.llm.api_key == "   "

    # ============================================================
    # P3 - Low: Defensive and unusual scenarios
    # ============================================================

    def test_very_long_env_var_value(self, tmp_path, monkeypatch):
        """Very long env var values are handled."""
        long_value = "a" * 10000
        monkeypatch.setenv("LONG_VAR", long_value)

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${LONG_VAR}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert len(config.llm.api_key) == 10000

    def test_env_var_empty_name(self, tmp_path, monkeypatch):
        """${} with empty name - edge case behavior."""
        # The regex pattern r'\$\{([^}]+)\}' requires at least one char
        # So ${} won't match and will be left as literal

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: "${}literal"

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        # ${} should not be matched (requires at least one char)
        assert config.llm.api_key == "${}literal"

    def test_nested_braces_not_supported(self, tmp_path, monkeypatch):
        """Nested braces ${VAR_${OTHER}} are not supported.

        The inner ${SUFFIX} is matched and substituted, but the outer
        ${VAR_...} is not matched because '$' breaks the word-char pattern.
        Result is the literal '${VAR_SUFFIX}' (not double-substituted).
        """
        monkeypatch.setenv("VAR_SUFFIX", "works")
        monkeypatch.setenv("SUFFIX", "SUFFIX")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${VAR_${SUFFIX}}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        # The regex matches ${SUFFIX} (inner) and substitutes it with "SUFFIX"
        # Result after substitution: ${VAR_SUFFIX} (a literal, not re-substituted)
        # The trailing } from the outer brace remains as literal text
        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "${VAR_SUFFIX}"

    def test_env_var_in_list_item(self, tmp_path, monkeypatch):
        """Env vars in YAML list items are substituted."""
        monkeypatch.setenv("IMPORT1", "pandas")
        monkeypatch.setenv("IMPORT2", "numpy")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key

databases: {}

execution:
  allowed_imports:
    - ${IMPORT1}
    - ${IMPORT2}
    - polars
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert "pandas" in config.execution.allowed_imports
        assert "numpy" in config.execution.allowed_imports
        assert "polars" in config.execution.allowed_imports

    def test_env_var_in_nested_config(self, tmp_path, monkeypatch):
        """Env vars work in deeply nested config structures."""
        monkeypatch.setenv("NESTED_TOKEN", "deep_secret")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key

databases: {}

documents:
  wiki:
    type: http
    url: https://wiki.example.com
    headers:
      Authorization: "Bearer ${NESTED_TOKEN}"
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.documents["wiki"].headers["Authorization"] == "Bearer deep_secret"

    def test_env_var_at_start_of_value(self, tmp_path, monkeypatch):
        """Env var at start of value works."""
        monkeypatch.setenv("START_VAR", "beginning")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${START_VAR}_rest

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "beginning_rest"

    def test_env_var_at_end_of_value(self, tmp_path, monkeypatch):
        """Env var at end of value works."""
        monkeypatch.setenv("END_VAR", "ending")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: rest_${END_VAR}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "rest_ending"

    def test_same_env_var_used_multiple_times(self, tmp_path, monkeypatch):
        """Same env var used multiple times in config."""
        monkeypatch.setenv("REPEATED_VAR", "repeated")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: ${REPEATED_VAR}

databases:
  db1:
    uri: sqlite:///${REPEATED_VAR}.db
  db2:
    uri: sqlite:///${REPEATED_VAR}_2.db
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "repeated"
        assert config.databases["db1"].uri == "sqlite:///repeated.db"
        assert config.databases["db2"].uri == "sqlite:///repeated_2.db"

    def test_env_var_case_sensitivity(self, tmp_path, monkeypatch):
        """Env var names are case-sensitive."""
        monkeypatch.setenv("MyVar", "lowercase_m")
        monkeypatch.setenv("MYVAR", "uppercase")

        yaml_content = """
llm:
  provider: anthropic
  model: ${MyVar}
  api_key: ${MYVAR}

databases: {}
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.llm.model == "lowercase_m"
        assert config.llm.api_key == "uppercase"

    def test_env_var_value_with_backslash(self, tmp_path, monkeypatch):
        """Env var value with backslashes (Windows paths)."""
        monkeypatch.setenv("WIN_PATH", "C:\\Users\\name\\data")

        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: test-key

databases:
  local:
    uri: sqlite:///${WIN_PATH}\\db.sqlite
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert "C:\\Users\\name\\data" in config.databases["local"].uri

    def test_no_env_vars_in_content(self, tmp_path):
        """Config without any env vars loads normally."""
        yaml_content = """
llm:
  provider: anthropic
  model: claude-sonnet-4-20250514
  api_key: hardcoded-key

databases:
  main:
    uri: sqlite:///test.db
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        config = Config.from_yaml(str(config_file))
        assert config.llm.api_key == "hardcoded-key"
        assert config.databases["main"].uri == "sqlite:///test.db"
