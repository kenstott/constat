# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for ExemplarGenerator with mocked LLM."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from constat.learning.exemplar_generator import ExemplarGenerator, ExemplarResult
from constat.storage.learnings import LearningStore, LearningCategory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeGlossaryTerm:
    name: str
    definition: str
    aliases: list[str] = field(default_factory=list)
    status: str = "draft"
    provenance: str = "llm"


@pytest.fixture
def tmp_store(tmp_path):
    return LearningStore(base_dir=tmp_path, user_id="test-user")


@pytest.fixture
def populated_store(tmp_store):
    """Store with rules at varying confidence/applied_count."""
    store = tmp_store
    # High confidence, high applied — should appear in minimal
    store.save_rule("Always use UTC for timestamps", LearningCategory.CODEGEN_ERROR, 0.9, [], ["datetime"])
    store.save_rule("Never use SELECT * in production queries", LearningCategory.CODEGEN_ERROR, 0.85, [], ["sql"])
    # Bump applied_count for the first two rules
    rules = store.list_rules()
    for r in rules:
        for _ in range(5):
            store.increment_rule_applied(r["id"])

    # Lower confidence — should NOT appear in minimal
    store.save_rule("Consider using async for I/O", LearningCategory.CODEGEN_ERROR, 0.65, [], ["async"])
    store.save_rule("Prefer f-strings over .format()", LearningCategory.CODEGEN_ERROR, 0.5, [], ["python"])
    return store


@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    vs.list_glossary_terms.return_value = [
        FakeGlossaryTerm("Revenue", "Total income from sales", ["Sales Revenue"], status="approved", provenance="human"),
        FakeGlossaryTerm("Churn Rate", "Rate of customer attrition", ["Attrition Rate"], status="approved", provenance="llm"),
        FakeGlossaryTerm("ARR", "Annual recurring revenue", [], status="draft", provenance="llm"),
        FakeGlossaryTerm("MRR", "Monthly recurring revenue", ["Monthly Rev"], status="draft", provenance="human"),
    ]
    # Mock relationship query
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [
        ("r1", "Revenue", "DRIVES", "Profit", "Revenue drives profit", 0.9),
        ("r2", "Customer", "HAS", "Subscription", "Customer has subscription", 0.8),
        ("r3", "Churn Rate", "AFFECTS", "Revenue", "Churn rate affects revenue", 0.7),
    ]
    vs._conn = mock_conn
    return vs


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.max_output_tokens = 4096

    def fake_generate(system, user_message, max_tokens=4096, **kwargs):
        # Return valid JSON pairs based on what's being requested
        if "coding rules" in system.lower() or "coding rules" in user_message.lower():
            return json.dumps([
                {"rule_index": 0, "user": "How should I handle timestamps?", "assistant": "Always use UTC."},
                {"rule_index": 0, "user": "What timezone for dates?", "assistant": "Use UTC for all timestamps."},
            ])
        elif "glossary" in system.lower():
            return json.dumps([
                {"term_index": 0, "user": "What is Sales Revenue?", "assistant": "The canonical term is Revenue."},
                {"term_index": 0, "user": "Tell me about total income", "assistant": "Revenue is the total income from sales."},
            ])
        elif "relationship" in system.lower():
            return json.dumps([
                {"rel_index": 0, "user": "How does revenue relate to profit?", "assistant": "Revenue drives profit."},
            ])
        return "[]"

    llm.generate.side_effect = fake_generate
    return llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCoverageLevelSelection:
    """Test that each coverage level selects the correct sources."""

    def test_minimal_selects_high_confidence_high_applied_rules(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path

        rules = gen._select_rules("minimal")
        # Only the 2 rules with confidence >= 0.8 and applied_count >= 3
        assert len(rules) == 2
        for r in rules:
            assert r["confidence"] >= 0.8
            assert r["applied_count"] >= 3

    def test_minimal_selects_no_glossary(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path
        assert gen._select_glossary_terms("minimal") == []

    def test_minimal_selects_no_relationships(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path
        assert gen._select_relationships("minimal") == []

    def test_standard_selects_all_rules(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path

        rules = gen._select_rules("standard")
        assert len(rules) == 4  # All rules

    def test_standard_selects_approved_or_human_terms(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path

        terms = gen._select_glossary_terms("standard")
        # Revenue (approved+human), Churn Rate (approved), MRR (human)
        assert len(terms) == 3
        names = {t.name for t in terms}
        assert "Revenue" in names
        assert "Churn Rate" in names
        assert "MRR" in names
        assert "ARR" not in names  # draft + llm

    def test_standard_selects_no_relationships(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path
        assert gen._select_relationships("standard") == []

    def test_comprehensive_selects_all_rules(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path
        assert len(gen._select_rules("comprehensive")) == 4

    def test_comprehensive_selects_all_defined_terms(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path

        terms = gen._select_glossary_terms("comprehensive")
        assert len(terms) == 4  # All terms have definitions

    def test_comprehensive_selects_all_relationships(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path

        rels = gen._select_relationships("comprehensive")
        assert len(rels) == 3


class TestGeneration:
    """Test end-to-end generation and output."""

    def test_generate_minimal(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path

        result = gen.generate("minimal")
        assert isinstance(result, ExemplarResult)
        assert result.rule_pairs > 0
        assert result.glossary_pairs == 0
        assert result.relationship_pairs == 0
        assert result.total == result.rule_pairs

    def test_generate_standard(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path

        result = gen.generate("standard")
        assert result.rule_pairs > 0
        assert result.glossary_pairs > 0
        assert result.relationship_pairs == 0
        assert result.total == result.rule_pairs + result.glossary_pairs

    def test_generate_comprehensive(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path

        result = gen.generate("comprehensive")
        assert result.rule_pairs > 0
        assert result.glossary_pairs > 0
        assert result.relationship_pairs > 0
        assert result.total == result.rule_pairs + result.glossary_pairs + result.relationship_pairs

    def test_minimal_fewer_than_comprehensive(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path

        minimal = gen.generate("minimal")
        gen2 = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen2.output_dir = tmp_path
        comprehensive = gen2.generate("comprehensive")

        assert minimal.total <= comprehensive.total

    def test_invalid_coverage_raises(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path
        with pytest.raises(ValueError, match="coverage must be"):
            gen.generate("extreme")


class TestOutputFormats:
    """Test JSONL output files."""

    def test_messages_format(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path

        result = gen.generate("minimal")
        messages_path = Path(result.output_paths["messages"])
        assert messages_path.exists()

        lines = messages_path.read_text().strip().split("\n")
        assert len(lines) == result.total

        for line in lines:
            obj = json.loads(line)
            assert "messages" in obj
            msgs = obj["messages"]
            assert len(msgs) == 3
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"
            assert msgs[2]["role"] == "assistant"

    def test_alpaca_format(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path

        result = gen.generate("minimal")
        alpaca_path = Path(result.output_paths["alpaca"])
        assert alpaca_path.exists()

        lines = alpaca_path.read_text().strip().split("\n")
        assert len(lines) == result.total

        for line in lines:
            obj = json.loads(line)
            assert "instruction" in obj
            assert "input" in obj
            assert "output" in obj
            assert obj["input"] == ""


class TestExemplarRunTracking:
    """Test that exemplar runs are persisted in LearningStore."""

    def test_run_saved(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path

        gen.generate("standard")

        runs = populated_store.get_exemplar_runs()
        assert len(runs) == 1
        assert runs[0]["coverage"] == "standard"
        assert runs[0]["total"] > 0
        assert "timestamp" in runs[0]

    def test_multiple_runs_tracked(self, populated_store, mock_vector_store, mock_llm, tmp_path):
        gen = ExemplarGenerator(populated_store, mock_vector_store, mock_llm, "sess1", "test-user")
        gen.output_dir = tmp_path

        gen.generate("minimal")
        gen.generate("comprehensive")

        runs = populated_store.get_exemplar_runs()
        assert len(runs) == 2
        # Newest first
        assert runs[0]["coverage"] == "comprehensive"
        assert runs[1]["coverage"] == "minimal"


class TestLLMFailureHandling:
    """Test graceful handling of LLM errors."""

    def test_llm_error_produces_empty_exemplars(self, populated_store, mock_vector_store, tmp_path):
        llm = MagicMock()
        llm.max_output_tokens = 4096
        llm.generate.side_effect = RuntimeError("LLM unavailable")

        gen = ExemplarGenerator(populated_store, mock_vector_store, llm, "sess1", "test-user")
        gen.output_dir = tmp_path

        result = gen.generate("minimal")
        assert result.total == 0
        # Files should still be created (empty)
        assert Path(result.output_paths["messages"]).exists()
        assert Path(result.output_paths["alpaca"]).exists()

    def test_partial_llm_failure(self, populated_store, mock_vector_store, tmp_path):
        """If one batch fails, others still succeed."""
        call_count = 0
        llm = MagicMock()
        llm.max_output_tokens = 4096

        def flaky_generate(system, user_message, max_tokens=4096, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Transient error")
            return json.dumps([
                {"rule_index": 0, "user": "Q", "assistant": "A"},
            ])

        llm.generate.side_effect = flaky_generate

        # Need enough rules to get 2+ batches
        for i in range(15):
            populated_store.save_rule(f"Rule {i}", LearningCategory.CODEGEN_ERROR, 0.9, [], [])
            for r in populated_store.list_rules():
                if r["summary"] == f"Rule {i}":
                    for _ in range(5):
                        populated_store.increment_rule_applied(r["id"])

        gen = ExemplarGenerator(populated_store, mock_vector_store, llm, "sess1", "test-user")
        gen.output_dir = tmp_path

        result = gen.generate("minimal")
        # First batch failed, second should succeed
        assert result.total > 0
