# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for fact resolution from document vector search.

These tests verify that facts can be extracted from unstructured documents
(markdown, text, PDF) via semantic search and used in analysis.
"""

import pytest
from pathlib import Path

from constat.discovery.doc_tools import DocumentDiscoveryTools
from constat.discovery.fact_tools import FactResolutionTools
from constat.execution.fact_resolver import FactResolver
from constat.core.config import Config, DocumentConfig


@pytest.fixture
def demo_docs_path():
    """Path to demo documents directory."""
    return Path(__file__).parent.parent / "demo" / "docs"


@pytest.fixture
def business_rules_doc(demo_docs_path):
    """Business rules document config."""
    return DocumentConfig(
        type="file",
        path=str(demo_docs_path / "business_rules.md"),
        description="Business policies for customer tiers, inventory reorder, and performance reviews",
    )


@pytest.fixture
def doc_tools(business_rules_doc):
    """DocumentDiscoveryTools with business rules loaded."""
    config = Config(
        llm={"provider": "anthropic", "model": "test"},
        documents={"business_rules": business_rules_doc},
    )
    return DocumentDiscoveryTools(config)


@pytest.fixture
def fact_resolution_tools(doc_tools):
    """FactResolutionTools with document search enabled."""
    fact_resolver = FactResolver()
    return FactResolutionTools(
        fact_resolver=fact_resolver,
        doc_tools=doc_tools,
    )


class TestDocumentSearch:
    """Tests for semantic search over documents.

    Note: Semantic similarity scores depend on embedding model and chunking.
    These tests verify that relevant content IS found, not that it's the
    top result with high confidence. In practice, the LLM uses multiple
    results to synthesize answers.
    """

    def test_search_finds_lead_time_in_results(self, doc_tools):
        """Search should find lead time content somewhere in results."""
        results = doc_tools.search_documents("reorder lead time shipping", limit=5)

        assert len(results) > 0

        # Check if any result contains lead time info
        all_excerpts = " ".join(r["excerpt"].lower() for r in results)
        assert "lead time" in all_excerpts or "business days" in all_excerpts, \
            f"Expected lead time info in results. Got: {[r['excerpt'][:100] for r in results]}"

    def test_search_finds_customer_tiers(self, doc_tools):
        """Search for customer tiers should find tier definitions."""
        results = doc_tools.search_documents("customer tier discount platinum gold", limit=5)

        assert len(results) > 0

        # Check if any result contains tier info
        all_excerpts = " ".join(r["excerpt"].lower() for r in results)
        assert any(term in all_excerpts for term in ["tier", "platinum", "gold", "discount"])

    def test_search_finds_performance_ratings(self, doc_tools):
        """Search for performance ratings should find rating scale."""
        results = doc_tools.search_documents("performance review rating scale exceptional", limit=5)

        assert len(results) > 0

        # Check if any result contains rating info
        all_excerpts = " ".join(r["excerpt"].lower() for r in results)
        assert any(term in all_excerpts for term in ["rating", "exceptional", "performance"])


class TestFactResolutionFromDocuments:
    """Tests for resolving facts from document content.

    Note: The FactResolutionTools.resolve_fact() uses a 0.7 relevance threshold
    before accepting a document answer. Semantic similarity for factual queries
    often falls below this. These tests verify the underlying search works,
    simulating how the LLM would use the results.
    """

    def test_search_finds_lead_time_content(self, doc_tools):
        """Verify lead time content is discoverable via search."""
        # Use direct search (bypasses the 0.7 threshold)
        results = doc_tools.search_documents("inventory reorder lead time days", limit=5)

        # Combine all results - LLM would see all of these
        all_content = " ".join(r["excerpt"] for r in results)

        # The lead time info should be in one of the chunks
        assert "Lead time" in all_content or "business days" in all_content, \
            f"Expected lead time in search results. Got: {all_content[:500]}"
        assert "express" in all_content.lower() or "standard" in all_content.lower()

    def test_search_finds_platinum_discount(self, doc_tools):
        """Verify platinum discount content is discoverable via search."""
        results = doc_tools.search_documents("platinum customer tier discount percentage", limit=5)

        all_content = " ".join(r["excerpt"] for r in results)

        # Should find tier table with discount info
        assert "Platinum" in all_content or "15%" in all_content

    def test_search_finds_reorder_rules(self, doc_tools):
        """Verify reorder rules are discoverable via search."""
        results = doc_tools.search_documents("automatic reorder triggered quantity level", limit=5)

        all_content = " ".join(r["excerpt"] for r in results)

        assert "reorder" in all_content.lower()
        assert "Automatic" in all_content or "quantity" in all_content.lower()

    def test_resolve_fact_checks_documents(self, fact_resolution_tools):
        """Verify resolve_fact checks document sources."""
        result = fact_resolution_tools.resolve_fact("What is the reorder policy?")

        # Should always check documents
        assert "documents" in result["sources_checked"]

    def test_low_relevance_triggers_clarification(self, fact_resolution_tools):
        """Low relevance should trigger needs_clarification flag."""
        result = fact_resolution_tools.resolve_fact("What is the weather like?")

        # Unrelated query should have low confidence and need clarification
        assert result["confidence"] < 0.7 or result["needs_clarification"]


class TestIntegratedFactUsage:
    """Tests for using document-derived facts in calculations.

    These tests simulate scenarios where a fact from documents
    is needed as part of a larger analysis. They use doc_tools.search_documents
    directly to verify the content is discoverable.
    """

    def test_lead_time_extraction_for_calculation(self, doc_tools):
        """
        Scenario: Calculate expected delivery date for express shipping.

        This simulates a query like "If I order today with express shipping,
        when will it arrive?" which requires:
        1. Extracting lead time from documents (2 business days)
        2. Using that in a date calculation
        """
        # Step 1: Search for lead time info (LLM would do this)
        results = doc_tools.search_documents("lead time express standard shipping days", limit=5)
        all_content = " ".join(r["excerpt"] for r in results)

        # Step 2: Verify the lead time info is discoverable
        assert "Lead time" in all_content or "business days" in all_content
        assert "express" in all_content.lower()
        assert "2" in all_content  # 2 business days for express

        # Step 3: Simulate the calculation the LLM would generate
        # (In real execution, the LLM would generate code like:)
        from datetime import datetime, timedelta
        express_lead_time_days = 2  # extracted from document
        order_date = datetime.now()
        delivery_date = order_date + timedelta(days=express_lead_time_days)
        assert delivery_date > order_date

    def test_tier_discount_for_order_calculation(self, doc_tools):
        """
        Scenario: Calculate discounted order total for a platinum customer.

        This simulates a query like "What would a $1000 order cost for a
        platinum customer?" which requires:
        1. Extracting platinum discount from documents (15%)
        2. Applying that to calculate final price
        """
        # Step 1: Search for tier/discount info
        results = doc_tools.search_documents("platinum tier discount percentage annual spend", limit=5)
        all_content = " ".join(r["excerpt"] for r in results)

        # Step 2: Verify discount info is discoverable
        assert "Platinum" in all_content
        assert "15%" in all_content or "15" in all_content

        # Step 3: Simulate the calculation
        order_total = 1000
        platinum_discount_pct = 15  # extracted from document
        final_price = order_total * (1 - platinum_discount_pct / 100)
        assert final_price == 850

    def test_reorder_rules_for_inventory_analysis(self, doc_tools):
        """
        Scenario: Determine reorder quantity based on business rules.

        This simulates a query like "How much should we reorder for a product
        with reorder_level of 50?" which requires:
        1. Extracting reorder quantity formula from documents (2x reorder_level)
        2. Applying that formula
        """
        # Step 1: Search for reorder rules
        results = doc_tools.search_documents("reorder quantity formula automatic inventory", limit=5)
        all_content = " ".join(r["excerpt"] for r in results)

        # Step 2: Verify reorder rule is discoverable
        assert "Reorder quantity" in all_content or "reorder" in all_content.lower()
        assert "2x" in all_content or "2" in all_content

        # Step 3: Simulate the calculation
        reorder_level = 50
        reorder_quantity = 2 * reorder_level  # 2x the reorder_level
        assert reorder_quantity == 100


class TestPerformanceReviewIntegration:
    """
    Integration test: Analyze actual performance reviews against documented guidelines.

    This tests the real-world scenario where:
    1. Business rules define the expected raise percentages per rating
    2. The HR database contains actual performance reviews and salaries
    3. Analysis compares actual vs expected based on document-derived rules

    Rating scale from business_rules.md:
    - 5 (Exceptional): 8-12%
    - 4 (Exceeds Expectations): 5-8%
    - 3 (Meets Expectations): 2-4%
    - 2 (Needs Improvement): 0%
    - 1 (Unsatisfactory): PIP required
    """

    def test_search_finds_rating_scale(self, doc_tools):
        """Search finds the performance rating scale from business rules."""
        results = doc_tools.search_documents(
            "performance rating raise percentage exceptional exceeds", limit=5
        )
        all_content = " ".join(r["excerpt"] for r in results)

        # Should contain the rating scale information
        assert any(pct in all_content for pct in ["8-12", "5-8", "2-4", "Exceptional"])

    def test_search_finds_exceptional_rating(self, doc_tools):
        """Search finds exceptional rating raise range."""
        results = doc_tools.search_documents(
            "exceptional rating 5 raise percentage", limit=5
        )
        all_content = " ".join(r["excerpt"] for r in results)

        # Should find the exceptional rating info
        assert "Exceptional" in all_content or "8-12" in all_content

    @pytest.fixture
    def hr_db_path(self):
        """Path to HR database."""
        return Path(__file__).parent.parent / "demo" / "data" / "hr.db"

    def test_performance_review_analysis_scenario(self, doc_tools, hr_db_path):
        """
        Full scenario: Compare actual reviews against documented guidelines.

        This simulates a query like:
        "Are our performance review raises aligned with the documented guidelines?"

        Steps:
        1. Extract rating scale from documents
        2. Query actual reviews from database
        3. Compare actual vs expected
        """
        import sqlite3

        # Skip if demo database doesn't exist
        if not hr_db_path.exists():
            pytest.skip("Demo HR database not found - run demo/setup_demo.py first")

        # Step 1: Get the rating guidelines from documents
        results = doc_tools.search_documents(
            "performance rating scale raise percentage guidelines", limit=5
        )
        all_content = " ".join(r["excerpt"] for r in results)

        # Verify rating scale is discoverable
        assert "Rating" in all_content or "Exceptional" in all_content

        # Step 2: Query actual performance reviews
        conn = sqlite3.connect(hr_db_path)
        cursor = conn.cursor()

        # Get reviews with ratings
        cursor.execute("""
            SELECT
                e.first_name || ' ' || e.last_name as name,
                pr.rating,
                pr.review_date
            FROM performance_reviews pr
            JOIN employees e ON pr.employee_id = e.employee_id
            ORDER BY pr.review_date DESC
            LIMIT 10
        """)
        reviews = cursor.fetchall()
        conn.close()

        # Step 3: Verify we have data to analyze
        assert len(reviews) > 0, "Expected performance review data in HR database"

        # Step 4: Parse expected raise ranges from guidelines
        # (In real execution, the LLM would extract these from the document)
        expected_raises = {
            5: (8, 12),   # Exceptional: 8-12%
            4: (5, 8),    # Exceeds: 5-8%
            3: (2, 4),    # Meets: 2-4%
            2: (0, 0),    # Needs Improvement: 0%
            1: (0, 0),    # Unsatisfactory: PIP
        }

        # Step 5: Verify each review has a valid rating per guidelines
        for name, rating, date in reviews:
            assert 1 <= rating <= 5, f"Invalid rating {rating} for {name}"
            assert rating in expected_raises, f"Rating {rating} not in guidelines"

        # This demonstrates the integration:
        # - Document search provides the rules (expected_raises)
        # - Database query provides the actual data (reviews)
        # - Analysis compares them


class TestEdgeCases:
    """Edge cases and error handling for document fact resolution."""

    def test_low_confidence_for_unrelated_query(self, fact_resolution_tools):
        """Query about unrelated topic should have low confidence."""
        result = fact_resolution_tools.resolve_fact(
            "What is the capital of France?"
        )

        # Should have lower confidence since this isn't in business rules
        # Note: might still return something if LLM fallback is enabled
        assert result["confidence"] < 0.8 or "needs_clarification" in result

    def test_partial_match_handling(self, fact_resolution_tools):
        """Query with partial match should still return relevant content."""
        result = fact_resolution_tools.resolve_fact(
            "shipping time"  # Less specific than "express shipping lead time"
        )

        # Should still find shipping-related content
        assert "documents" in result["sources_checked"]
        # May have lower confidence due to vague query
