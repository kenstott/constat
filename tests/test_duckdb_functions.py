# Copyright (c) 2025 Kenneth Stott
# Canary: be3beedd-c619-4f83-9771-c841c2c2358f
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for DuckDBSessionStore fuzzy_map and extract_table methods."""

from __future__ import annotations
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from constat.storage.duckdb_session_store import DuckDBSessionStore
from constat.storage.registry import ConstatRegistry


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def registry(tmp_dir):
    return ConstatRegistry(base_dir=tmp_dir / "registry")


@pytest.fixture
def store(tmp_dir, registry):
    s = DuckDBSessionStore(
        db_path=tmp_dir / "session.duckdb",
        registry=registry,
        user_id="test_user",
        session_id="test_session",
    )
    yield s
    s.close()


# --- Mock return value helper ---

def _make_llm_map_result(source_values, target_values):
    """Build a mock _raw_llm_map return: each source maps to first target."""
    result = {}
    for i, sv in enumerate(source_values):
        tv = target_values[i % len(target_values)] if target_values else None
        result[sv] = {"value": tv, "reason": f"matched {sv} to {tv}", "score": 0.8}
    return result


def _make_llm_map_result_varied_scores(source_values, target_values, scores):
    """Build mock result with specific scores per source value."""
    result = {}
    for i, sv in enumerate(source_values):
        tv = target_values[i % len(target_values)] if target_values else None
        sc = scores[i] if i < len(scores) else 0.5
        result[sv] = {"value": tv, "reason": f"matched {sv}", "score": sc}
    return result


class TestFuzzyMap:
    def test_fuzzy_map_from_tables(self, store):
        """Create two tables, call fuzzy_map, verify mapping table schema."""
        store.save_dataframe("products", pd.DataFrame({"product_name": ["Widget Pro", "Gadget X"]}))
        store.save_dataframe("breeds", pd.DataFrame({"breed_name": ["Bengal", "Siamese"]}))

        mock_result = _make_llm_map_result(["Widget Pro", "Gadget X"], ["Bengal", "Siamese"])

        with patch("constat.storage.duckdb_session_store.DuckDBSessionStore.fuzzy_map.__wrapped__", side_effect=None) if False else \
             patch("constat.llm.llm_map", return_value=mock_result):
            name = store.fuzzy_map(
                "products", "product_name", "breeds", "breed_name",
                source_desc="products", target_desc="cat breeds",
            )

        assert name == "_fuzzy_map_product_name_breed_name"
        df = store.load_dataframe(name)
        assert df is not None
        assert set(df.columns) == {"source_value", "target_value", "confidence", "reason"}
        assert len(df) == 2

    def test_fuzzy_map_from_lists(self, store):
        """Pass ad-hoc lists, verify table created."""
        source = ["Widget Pro", "Gadget X"]
        target = ["Bengal", "Siamese"]
        mock_result = _make_llm_map_result(source, target)

        with patch("constat.llm.llm_map", return_value=mock_result):
            name = store.fuzzy_map(
                source, None, target, None,
                source_desc="products", target_desc="breeds",
            )

        assert name == "_fuzzy_map_products_breeds"
        df = store.load_dataframe(name)
        assert len(df) == 2
        assert list(df["source_value"]) == ["Widget Pro", "Gadget X"]

    def test_fuzzy_map_min_score(self, store):
        """Entries below min_score are filtered out."""
        source = ["Widget Pro", "Gadget X", "Thingamajig"]
        target = ["Bengal", "Siamese"]
        mock_result = _make_llm_map_result_varied_scores(
            source, target, scores=[0.9, 0.2, 0.5],
        )

        with patch("constat.llm.llm_map", return_value=mock_result):
            name = store.fuzzy_map(
                source, None, target, None,
                source_desc="products", target_desc="breeds",
                min_score=0.4,
            )

        df = store.load_dataframe(name)
        assert len(df) == 2  # Gadget X (0.2) filtered out
        assert "Gadget X" not in df["source_value"].tolist()

    def test_fuzzy_map_join(self, store):
        """Create mapping, JOIN in SQL, verify results."""
        store.save_dataframe("products", pd.DataFrame({
            "product_name": ["Widget Pro", "Gadget X"],
            "price": [10.0, 20.0],
        }))
        store.save_dataframe("breeds", pd.DataFrame({"breed_name": ["Bengal", "Siamese"]}))

        mock_result = {
            "Widget Pro": {"value": "Bengal", "reason": "widget-bengal", "score": 0.85},
            "Gadget X": {"value": "Siamese", "reason": "gadget-siamese", "score": 0.75},
        }

        with patch("constat.llm.llm_map", return_value=mock_result):
            mapping = store.fuzzy_map(
                "products", "product_name", "breeds", "breed_name",
                source_desc="products", target_desc="breeds",
            )

        result = store.query(f"""
            SELECT p.product_name, p.price, m.target_value as matched, m.confidence
            FROM products p
            JOIN {mapping} m ON p.product_name = m.source_value
            ORDER BY p.product_name
        """)
        assert len(result) == 2
        assert "matched" in result.columns

    def test_fuzzy_map_custom_name(self, store):
        """table_name= override works."""
        source = ["A", "B"]
        target = ["X", "Y"]
        mock_result = _make_llm_map_result(source, target)

        with patch("constat.llm.llm_map", return_value=mock_result):
            name = store.fuzzy_map(
                source, None, target, None,
                source_desc="src", target_desc="tgt",
                table_name="my_custom_map",
            )

        assert name == "my_custom_map"
        assert store.load_dataframe("my_custom_map") is not None


    def test_fuzzy_map_expression_source_col(self, store):
        """source_col can be a SQL expression, not just a column name."""
        store.save_dataframe("items", pd.DataFrame({
            "sku": ["SKU-1", "SKU-2"],
            "name": ["Widget", "Gadget"],
            "detail": ["Pro tool", "Mini device"],
        }))

        # The expression concatenates name and detail
        mock_result = {
            "Widget - Pro tool": {"value": "X", "reason": "r1", "score": 0.9},
            "Gadget - Mini device": {"value": "Y", "reason": "r2", "score": 0.8},
        }

        with patch("constat.llm.llm_map", return_value=mock_result):
            mapping = store.fuzzy_map(
                "items", "name || ' - ' || detail", ["X", "Y"], None,
                source_desc="items", target_desc="targets",
            )

        df = store.load_dataframe(mapping)
        assert len(df) == 2
        assert "Widget - Pro tool" in df["source_value"].tolist()


class TestExtractTable:
    def test_extract_table_basic(self, store):
        """Mock llm_extract_table, verify table created with correct data."""
        mock_df = pd.DataFrame({"rating": ["A", "B"], "min_raise": [0.03, 0.01]})

        with patch("constat.llm.llm_extract_table", return_value=mock_df):
            name = store.extract_table(
                "Some document text...",
                "raise percentage guidelines",
            )

        assert name == "_extracted_raise_percentage_guidelines"
        df = store.load_dataframe(name)
        assert len(df) == 2
        assert "rating" in df.columns
        assert "min_raise" in df.columns

    def test_extract_table_with_columns(self, store):
        """Column enforcement passed through to llm_extract_table."""
        mock_df = pd.DataFrame({"rating": ["A"], "min_raise": [0.03], "max_raise": [0.05]})

        with patch("constat.llm.llm_extract_table", return_value=mock_df) as mock_fn:
            store.extract_table(
                "text", "guidelines",
                columns=["rating", "min_raise", "max_raise"],
                dtypes={"rating": "str"},
            )

        mock_fn.assert_called_once_with(
            "text", "guidelines",
            columns=["rating", "min_raise", "max_raise"],
            dtypes={"rating": "str"},
        )

    def test_extract_table_custom_name(self, store):
        """table_name override."""
        mock_df = pd.DataFrame({"a": [1]})

        with patch("constat.llm.llm_extract_table", return_value=mock_df):
            name = store.extract_table("text", "desc", table_name="my_table")

        assert name == "my_table"
        assert store.load_dataframe("my_table") is not None

    def test_extract_table_auto_name(self, store):
        """Auto-generated name from description."""
        mock_df = pd.DataFrame({"x": [1]})

        with patch("constat.llm.llm_extract_table", return_value=mock_df):
            name = store.extract_table("text", "Employee Rating Scale")

        assert name == "_extracted_employee_rating_scale"
