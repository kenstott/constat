# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for table-aware chunking and new LLM extract primitives."""

import json
import pytest
from unittest.mock import patch

from constat.discovery.doc_tools._core import (
    _is_table_line,
    _is_list_line,
    _merge_blocks,
)


# =============================================================================
# _is_table_line tests
# =============================================================================

class TestIsTableLine:
    def test_markdown_table_row(self):
        assert _is_table_line("| col1 | col2 | col3 |")

    def test_markdown_separator(self):
        assert _is_table_line("|---|---|---|")

    def test_docx_pipe_row(self):
        assert _is_table_line("cell1 | cell2 | cell3")

    def test_single_pipe_not_table(self):
        assert not _is_table_line("this | has one pipe")

    def test_no_pipe(self):
        assert not _is_table_line("just a regular line")

    def test_empty_line(self):
        assert not _is_table_line("")

    def test_two_pipes_minimum(self):
        assert _is_table_line("a | b | c")


# =============================================================================
# _is_list_line tests
# =============================================================================

class TestIsListLine:
    def test_dash_list(self):
        assert _is_list_line("- item one")

    def test_asterisk_list(self):
        assert _is_list_line("* item two")

    def test_plus_list(self):
        assert _is_list_line("+ item three")

    def test_ordered_list(self):
        assert _is_list_line("1. first item")

    def test_ordered_list_double_digit(self):
        assert _is_list_line("12. twelfth item")

    def test_indented_list(self):
        assert _is_list_line("  - indented item")

    def test_not_a_list(self):
        assert not _is_list_line("regular paragraph text")

    def test_empty_line(self):
        assert not _is_list_line("")

    def test_dash_without_space(self):
        assert not _is_list_line("-not a list")

    def test_number_without_dot(self):
        assert not _is_list_line("123 not a list")


# =============================================================================
# _merge_blocks tests
# =============================================================================

class TestMergeBlocks:
    def test_merge_consecutive_table_paragraphs(self):
        paragraphs = [
            "Header text",
            "| col1 | col2 |",
            "|------|------|",
            "| val1 | val2 |",
            "Footer text",
        ]
        result = _merge_blocks(paragraphs, "\n\n")
        assert len(result) == 3
        assert result[0] == "Header text"
        assert "| col1 | col2 |" in result[1]
        assert "| val1 | val2 |" in result[1]
        assert result[2] == "Footer text"

    def test_merge_consecutive_list_paragraphs(self):
        paragraphs = [
            "Intro",
            "- item one",
            "- item two",
            "- item three",
            "Outro",
        ]
        result = _merge_blocks(paragraphs, "\n\n")
        assert len(result) == 3
        assert result[0] == "Intro"
        assert "- item one" in result[1]
        assert "- item three" in result[1]
        assert result[2] == "Outro"

    def test_no_merge_non_table_non_list(self):
        paragraphs = ["para one", "para two", "para three"]
        result = _merge_blocks(paragraphs, "\n\n")
        assert result == paragraphs

    def test_mixed_table_and_list_not_merged(self):
        paragraphs = [
            "| a | b |",
            "- list item",
            "| c | d |",
        ]
        result = _merge_blocks(paragraphs, "\n\n")
        # Table, then list (breaks table sequence), then table again
        assert len(result) == 3

    def test_pipe_separated_docx_rows(self):
        paragraphs = [
            "Intro paragraph",
            "Name | Age | City",
            "Alice | 30 | NYC",
            "Bob | 25 | LA",
            "Summary paragraph",
        ]
        result = _merge_blocks(paragraphs, "\n\n")
        assert len(result) == 3
        assert "Name | Age | City" in result[1]
        assert "Bob | 25 | LA" in result[1]

    def test_ordered_list_merge(self):
        paragraphs = [
            "Steps:",
            "1. Do this",
            "2. Do that",
            "3. Done",
            "End.",
        ]
        result = _merge_blocks(paragraphs, "\n\n")
        assert len(result) == 3
        assert "1. Do this" in result[1]
        assert "3. Done" in result[1]

    def test_empty_paragraphs_preserved(self):
        paragraphs = ["| a | b |", "", "| c | d |"]
        result = _merge_blocks(paragraphs, "\n\n")
        # Empty string paragraph breaks the table sequence
        assert len(result) == 3

    def test_separator_used_correctly(self):
        paragraphs = ["| a | b |", "| c | d |"]
        result = _merge_blocks(paragraphs, "\n")
        assert result == ["| a | b |\n| c | d |"]

        result2 = _merge_blocks(paragraphs, "\n\n")
        assert result2 == ["| a | b |\n\n| c | d |"]

    def test_single_table_paragraph_no_merge(self):
        paragraphs = ["text", "| a | b |", "text"]
        result = _merge_blocks(paragraphs, "\n\n")
        assert len(result) == 3


# =============================================================================
# llm_extract_table (internal text-based function) tests
# =============================================================================

class TestLlmExtractTableInternal:
    @patch("constat.llm._execute")
    def test_basic_extraction(self, mock_execute):
        mock_execute.return_value = (
            json.dumps([
                {"rating": "Exceeds", "raise_pct": "5-7%"},
                {"rating": "Meets", "raise_pct": "3-4%"},
            ]),
            "claude-sonnet",
            "anthropic",
        )
        from constat.llm import llm_extract_table
        df = llm_extract_table("some doc text", "raise guidelines")
        assert len(df) == 2
        # Range values "5-7%" auto-expand into raise_rate_min / raise_rate_max
        # (pct → rate because values converted to decimal)
        assert list(df.columns) == ["rating", "raise_rate_min", "raise_rate_max"]
        assert df.iloc[0]["rating"] == "Exceeds"
        assert df.iloc[0]["raise_rate_min"] == 0.05
        assert df.iloc[0]["raise_rate_max"] == 0.07

    @patch("constat.llm._execute")
    def test_with_columns(self, mock_execute):
        mock_execute.return_value = (
            json.dumps([{"col_a": "x", "col_b": "y"}]),
            "claude-sonnet",
            "anthropic",
        )
        from constat.llm import llm_extract_table
        df = llm_extract_table("text", "table", columns=["col_a", "col_b"])
        assert list(df.columns) == ["col_a", "col_b"]

    @patch("constat.llm._execute")
    def test_empty_result_raises(self, mock_execute):
        mock_execute.return_value = ("[]", "claude-sonnet", "anthropic")
        from constat.llm import llm_extract_table
        with pytest.raises(ValueError, match="found no rows"):
            llm_extract_table("text", "nonexistent table")

    @patch("constat.llm._execute")
    def test_invalid_json_raises(self, mock_execute):
        mock_execute.return_value = ("not json", "claude-sonnet", "anthropic")
        from constat.llm import llm_extract_table
        with pytest.raises(ValueError, match="failed to extract"):
            llm_extract_table("text", "table")


# =============================================================================
# llm_extract_facts (internal text-based function) tests
# =============================================================================

class TestLlmExtractFactsInternal:
    @patch("constat.llm._execute")
    def test_basic_extraction(self, mock_execute):
        mock_execute.return_value = (
            json.dumps([
                {"name": "VIP threshold", "value": 100000, "dtype": "scalar",
                 "metadata": {"unit": "USD", "data_type": "currency"}},
                {"name": "ratings", "value": ["Exceeds", "Meets", "Below"],
                 "dtype": "list", "metadata": {"items": ["Exceeds", "Meets", "Below"], "count": 3}},
            ]),
            "claude-sonnet",
            "anthropic",
        )
        from constat.llm import llm_extract_facts
        facts = llm_extract_facts("some policy text")
        assert len(facts) == 2
        assert facts[0]["name"] == "VIP threshold"
        assert facts[0]["dtype"] == "scalar"
        assert facts[1]["dtype"] == "list"

    @patch("constat.llm._execute")
    def test_with_context(self, mock_execute):
        mock_execute.return_value = (
            json.dumps([{"name": "fact1", "value": "x", "dtype": "text", "metadata": {}}]),
            "claude-sonnet",
            "anthropic",
        )
        from constat.llm import llm_extract_facts
        facts = llm_extract_facts("text", context="HR policies")
        assert len(facts) == 1
        # Verify context was included in the prompt
        call_args = mock_execute.call_args
        assert "HR policies" in call_args[1]["user_message"] or "HR policies" in call_args[0][1]

    @patch("constat.llm._execute")
    def test_invalid_json_returns_empty(self, mock_execute):
        mock_execute.return_value = ("invalid", "claude-sonnet", "anthropic")
        from constat.llm import llm_extract_facts
        facts = llm_extract_facts("text")
        assert facts == []

    @patch("constat.llm._execute")
    def test_empty_result(self, mock_execute):
        mock_execute.return_value = ("[]", "claude-sonnet", "anthropic")
        from constat.llm import llm_extract_facts
        facts = llm_extract_facts("text")
        assert facts == []


# =============================================================================
# Table continuation marker tests
# =============================================================================

class TestTableContinuationMarkers:
    """Test that _assemble_chunk_text correctly strips TABLE markers."""

    def test_strip_table_markers(self):
        from constat.discovery.models import DocumentChunk

        # Simulate what _execution.py._assemble_chunk_text does
        def assemble(chunks):
            parts = []
            for chunk in chunks:
                text = chunk.content
                text = text.replace("[TABLE:start]\n", "")
                text = text.replace("\n[TABLE:cont]", "")
                text = text.replace("[TABLE:cont]\n", "")
                text = text.replace("\n[TABLE:end]", "")
                parts.append(text)
            return "\n\n".join(parts)

        chunks = [
            DocumentChunk(
                document_name="doc",
                content="[TABLE:start]\n| a | b |\n| 1 | 2 |\n[TABLE:cont]",
                chunk_index=0,
            ),
            DocumentChunk(
                document_name="doc",
                content="[TABLE:cont]\n| 3 | 4 |\n| 5 | 6 |\n[TABLE:end]",
                chunk_index=1,
            ),
        ]
        result = assemble(chunks)
        assert "[TABLE:" not in result
        assert "| a | b |" in result
        assert "| 5 | 6 |" in result

    def test_no_markers_passthrough(self):
        from constat.discovery.models import DocumentChunk

        def assemble(chunks):
            parts = []
            for chunk in chunks:
                text = chunk.content
                text = text.replace("[TABLE:start]\n", "")
                text = text.replace("\n[TABLE:cont]", "")
                text = text.replace("[TABLE:cont]\n", "")
                text = text.replace("\n[TABLE:end]", "")
                parts.append(text)
            return "\n\n".join(parts)

        chunks = [
            DocumentChunk(document_name="doc", content="regular text", chunk_index=0),
        ]
        result = assemble(chunks)
        assert result == "regular text"
