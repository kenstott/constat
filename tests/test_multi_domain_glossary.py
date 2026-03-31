# Copyright (c) 2025 Kenneth Stott
# Canary: 1be40db1-2e36-4f3a-a05c-5c73f80c4744

"""Tests for multi-domain glossary support (Gap 1 + Gap 5)."""

import uuid

import duckdb
import pytest

from constat.discovery.models import GlossaryTerm
from constat.discovery.vector_store import DuckDBVectorStore

SESSION_ID = "test-session"
USER_ID = "test-user"


def _make_term(name: str = "revenue", domain: str = "test", **kwargs) -> GlossaryTerm:
    return GlossaryTerm(
        id=kwargs.pop("id", str(uuid.uuid4())),
        name=name,
        display_name=kwargs.pop("display_name", name.replace("_", " ").title()),
        definition=kwargs.pop("definition", f"Definition of {name}"),
        domain=domain,
        session_id=kwargs.pop("session_id", SESSION_ID),
        user_id=kwargs.pop("user_id", USER_ID),
        **kwargs,
    )


@pytest.fixture()
def vs():
    store = DuckDBVectorStore(db_path=":memory:")
    return store


@pytest.fixture()
def rel(vs):
    return vs._store.relational


class TestMultiDomainGlossary:
    def test_add_same_name_different_domains(self, rel):
        """Same term name in different domains should coexist."""
        rel.add_glossary_term(_make_term("revenue", domain="sales"))
        rel.add_glossary_term(_make_term("revenue", domain="finance"))
        terms = rel.list_glossary_terms(SESSION_ID, user_id=USER_ID)
        revenue_terms = [t for t in terms if t.name == "revenue"]
        assert len(revenue_terms) == 2
        domains = {t.domain for t in revenue_terms}
        assert domains == {"sales", "finance"}

    def test_get_glossary_term_with_domain_filter(self, rel):
        """get_glossary_term with domain= returns only the matching domain."""
        rel.add_glossary_term(_make_term("revenue", domain="sales", definition="Sales revenue"))
        rel.add_glossary_term(_make_term("revenue", domain="finance", definition="Finance revenue"))
        term = rel.get_glossary_term("revenue", SESSION_ID, user_id=USER_ID, domain="finance")
        assert term is not None
        assert term.domain == "finance"
        assert term.definition == "Finance revenue"

    def test_get_glossary_term_without_domain_returns_any(self, rel):
        """get_glossary_term without domain= returns one of the matching terms."""
        rel.add_glossary_term(_make_term("revenue", domain="sales"))
        rel.add_glossary_term(_make_term("revenue", domain="finance"))
        term = rel.get_glossary_term("revenue", SESSION_ID, user_id=USER_ID)
        assert term is not None
        assert term.domain in ("sales", "finance")

    def test_update_with_domain_filter(self, rel):
        """update_glossary_term with domain= only affects the targeted domain."""
        rel.add_glossary_term(_make_term("revenue", domain="sales", definition="Old sales"))
        rel.add_glossary_term(_make_term("revenue", domain="finance", definition="Old finance"))
        updated = rel.update_glossary_term(
            "revenue", SESSION_ID, {"definition": "New sales"}, user_id=USER_ID, domain="sales",
        )
        assert updated is True
        sales = rel.get_glossary_term("revenue", SESSION_ID, user_id=USER_ID, domain="sales")
        finance = rel.get_glossary_term("revenue", SESSION_ID, user_id=USER_ID, domain="finance")
        assert sales.definition == "New sales"
        assert finance.definition == "Old finance"

    def test_delete_with_domain_filter(self, rel):
        """delete_glossary_term with domain= only removes the targeted domain."""
        rel.add_glossary_term(_make_term("revenue", domain="sales"))
        rel.add_glossary_term(_make_term("revenue", domain="finance"))
        deleted = rel.delete_glossary_term("revenue", SESSION_ID, user_id=USER_ID, domain="sales")
        assert deleted is True
        remaining = rel.list_glossary_terms(SESSION_ID, user_id=USER_ID)
        revenue_terms = [t for t in remaining if t.name == "revenue"]
        assert len(revenue_terms) == 1
        assert revenue_terms[0].domain == "finance"

    def test_unique_constraint_prevents_duplicate(self, rel):
        """Inserting same (name, domain, user_id) with different id should raise ConstraintException."""
        term1 = _make_term("revenue", domain="sales")
        rel.add_glossary_term(term1)
        term2 = _make_term("revenue", domain="sales", definition="Duplicate")
        with pytest.raises(duckdb.ConstraintException):
            rel.add_glossary_term(term2)
        # Original term is preserved
        terms = rel.list_glossary_terms(SESSION_ID, user_id=USER_ID)
        revenue_sales = [t for t in terms if t.name == "revenue" and t.domain == "sales"]
        assert len(revenue_sales) == 1
        assert revenue_sales[0].definition == "Definition of revenue"

    def test_term_source_schema_with_domain(self, rel):
        """term_source_schema with domain= filters correctly (non-split returns 'main')."""
        rel.add_glossary_term(_make_term("revenue", domain="sales"))
        rel.add_glossary_term(_make_term("revenue", domain="finance"))
        schema = rel.term_source_schema("revenue", SESSION_ID, user_id=USER_ID, domain="sales")
        assert schema == "main"

    def test_get_glossary_term_by_name_or_alias_with_domain(self, rel):
        """get_glossary_term_by_name_or_alias with domain= filters correctly."""
        rel.add_glossary_term(_make_term("revenue", domain="sales", aliases=["rev"]))
        rel.add_glossary_term(_make_term("revenue", domain="finance", aliases=["rev"]))
        term = rel.get_glossary_term_by_name_or_alias(
            "revenue", SESSION_ID, user_id=USER_ID, domain="finance",
        )
        assert term is not None
        assert term.domain == "finance"

    def test_get_glossary_terms_by_name_or_alias_with_domain(self, rel):
        """get_glossary_terms_by_name_or_alias with domain= returns only matching domain."""
        rel.add_glossary_term(_make_term("revenue", domain="sales"))
        rel.add_glossary_term(_make_term("revenue", domain="finance"))
        terms = rel.get_glossary_terms_by_name_or_alias(
            "revenue", SESSION_ID, user_id=USER_ID, domain="sales",
        )
        assert len(terms) == 1
        assert terms[0].domain == "sales"
