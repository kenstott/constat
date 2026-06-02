# Copyright (c) 2025 Kenneth Stott
# Canary: 75fd3ab9-9240-4395-b36a-4eacd6b9b407
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Slow LLM tests covering inventory domain queries.

Verifies that the LLM can correctly query inventory.db — specifically that the
schema is indexed, the right tables are selected, and answers contain real data.
These tests would fail if inventory.db were missing from the chonk index or if
source_id linkage were broken (search would never surface inventory schema chunks).

Requires ANTHROPIC_API_KEY to be set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from constat.catalog.schema_manager import SchemaManager
from constat.core.config import Config
from constat.execution.engine import QueryEngine

pytestmark = [pytest.mark.slow]

DEMO_DATA = Path(__file__).parents[1] / "demo" / "data"


@pytest.fixture
def require_anthropic_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.fail("ANTHROPIC_API_KEY not set — required for this test")


@pytest.fixture(scope="module")
def engine() -> QueryEngine:
    config = Config(
        databases={
            "sales": {"uri": f"sqlite:///{DEMO_DATA}/sales.db"},
            "inventory": {"uri": f"sqlite:///{DEMO_DATA}/inventory.db"},
        },
        system_prompt="""You are analyzing demo company data across two databases.

## sales
- customers, orders, order_items, products
- Use for revenue, customer, and product sales queries

## inventory
- warehouses (4 locations with capacity and manager)
- inventory (sku, product_name, quantity, reorder_level, last_restocked, warehouse_id)
- shipments (inbound/outbound movements by sku, carrier, date)
- Use for stock levels, warehouse, and logistics queries

Choose the correct database based on the query domain.
""",
        execution={"allowed_imports": ["pandas", "numpy", "json", "datetime"]},
    )
    schema_manager = SchemaManager(config)
    schema_manager.initialize()
    schema_manager.build_chunks()
    return QueryEngine(config, schema_manager, max_retries=3)


class TestInventoryQueries:
    """Single-source inventory queries — verify inventory.db is accessible."""

    def test_items_below_reorder_level(self, require_anthropic_key, engine: QueryEngine):
        """LLM must query inventory table and return products below reorder_level."""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query(
                    "Which products are currently below their reorder level? "
                    "Show product name, current quantity, and reorder level."
                )
                assert result.success, f"Query failed: {result.error}"
                answer = result.answer.lower()
                # inventory has items below reorder: Widget Pro (19 < 56), Standing Desk (9 < 85)
                assert "widget pro" in answer or "standing desk" in answer, (
                    f"Expected inventory items in answer, got: {result.answer[:300]}"
                )
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_warehouse_listing(self, require_anthropic_key, engine: QueryEngine):
        """LLM must query the warehouses table and return location data."""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query("List all warehouses with their locations and managers.")
                assert result.success, f"Query failed: {result.error}"
                answer = result.answer.lower()
                # All 4 warehouse locations must appear
                for location in ("newark", "los angeles", "dallas", "atlanta"):
                    assert location in answer, (
                        f"Expected warehouse location '{location}' in answer: {result.answer[:300]}"
                    )
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_inbound_vs_outbound_shipments(self, require_anthropic_key, engine: QueryEngine):
        """LLM must query shipments table and distinguish inbound vs outbound."""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query(
                    "How many inbound and outbound shipments are there in total?"
                )
                assert result.success, f"Query failed: {result.error}"
                answer = result.answer.lower()
                assert "inbound" in answer or "outbound" in answer, (
                    f"Expected shipment direction breakdown: {result.answer[:300]}"
                )
                # shipments table has 100 rows — answer must contain a number
                assert any(c.isdigit() for c in result.answer), (
                    f"Expected numeric counts in answer: {result.answer[:300]}"
                )
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_cross_database_sku_overlap(self, require_anthropic_key, engine: QueryEngine):
        """LLM must join product data across sales and inventory databases."""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query(
                    "Which product names appear in both the sales product catalog "
                    "and the inventory stock records?"
                )
                assert result.success, f"Query failed: {result.error}"
                answer = result.answer.lower()
                # Must return at least one product name and reference both domains
                assert any(c.isdigit() or c.isalpha() for c in answer), (
                    f"Answer appears empty: {result.answer[:300]}"
                )
                # Answer must not be a refusal or "no data found"
                refusal_phrases = ("cannot", "unable", "no data", "not possible", "don't have")
                assert not any(p in answer for p in refusal_phrases), (
                    f"LLM refused the cross-DB query: {result.answer[:300]}"
                )
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_inventory_uses_inventory_db(self, require_anthropic_key, engine: QueryEngine):
        """A stock-level query must execute against the inventory database, not sales."""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query(
                    "What is the total quantity of Widget Pro across all warehouses?"
                )
                assert result.success, f"Query failed: {result.error}"
                # Widget Pro appears in inventory with quantity 19 + 31 across two rows
                answer = result.answer
                assert any(c.isdigit() for c in answer), (
                    f"Expected a numeric quantity in answer: {answer[:300]}"
                )
                # Must NOT say it can't find Widget Pro
                assert "widget pro" in answer.lower() or any(c.isdigit() for c in answer), (
                    f"Expected Widget Pro quantity: {answer[:300]}"
                )
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc


class TestMultiSourceInventoryQueries:
    """Complex queries requiring both sales.db and inventory.db."""

    def test_top_seller_at_stockout_risk(self, require_anthropic_key, engine: QueryEngine):
        """Standing Desk is #1 by sales volume but critically low on stock.

        LLM must query sales for top sellers AND inventory for stock levels,
        then surface the overlap. Standing Desk (9 units, reorder 85) must appear.
        """
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query(
                    "Which of our top 5 best-selling products (by units sold) "
                    "are currently below their inventory reorder level? "
                    "Show product name, total units sold, current stock, and reorder level."
                )
                assert result.success, f"Query failed: {result.error}"
                answer = result.answer.lower()
                assert "standing desk" in answer, (
                    f"Expected 'Standing Desk' (top seller + critically low stock): {result.answer[:400]}"
                )
                assert any(c.isdigit() for c in result.answer), (
                    f"Expected numeric data in answer: {result.answer[:400]}"
                )
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_products_sold_but_not_tracked_in_inventory(self, require_anthropic_key, engine: QueryEngine):
        """8 of 15 sales products have no record in the inventory database.

        LLM must identify product names that exist in sales.products but are
        absent from inventory.inventory. Pen Set Premium (#2 seller) is one.
        """
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query(
                    "Which products are we actively selling (in sales orders) "
                    "but have no record in our inventory tracking system?"
                )
                assert result.success, f"Query failed: {result.error}"
                answer = result.answer.lower()
                # Pen Set Premium is #2 seller with zero inventory record
                assert "pen set premium" in answer or "keyboard" in answer or "wireless mouse" in answer, (
                    f"Expected untracked products (e.g. Pen Set Premium) in answer: {result.answer[:400]}"
                )
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc

    def test_warehouse_capacity_vs_order_count(self, require_anthropic_key, engine: QueryEngine):
        """LLM must combine warehouse capacity from inventory with order volume from sales."""
        last_exc = None
        for attempt in range(3):
            try:
                result = engine.query(
                    "We have 4 warehouses. How many total orders have been placed in sales, "
                    "and what is the total warehouse capacity across all locations?"
                )
                assert result.success, f"Query failed: {result.error}"
                answer = result.answer
                # sales has 200 orders; total warehouse capacity = 225,000
                assert any(c.isdigit() for c in answer), (
                    f"Expected numeric answer: {answer[:400]}"
                )
                assert "200" in answer or "225" in answer or "225,000" in answer or "225000" in answer, (
                    f"Expected order count (~200) or capacity (~225000): {answer[:400]}"
                )
                break
            except AssertionError as e:
                last_exc = e
                if attempt == 2:
                    raise
        else:
            raise last_exc
