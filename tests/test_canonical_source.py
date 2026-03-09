# Copyright (c) 2025 Kenneth Stott

"""Tests for canonical_source on glossary terms."""

import uuid

import pytest

from constat.discovery.models import GlossaryTerm
from constat.discovery.vector_store import DuckDBVectorStore
from constat.storage.relational import RelationalStore

SESSION_ID = "test-session"
USER_ID = "test-user"


def _make_term(name: str = "revenue", **kwargs) -> GlossaryTerm:
    return GlossaryTerm(
        id=kwargs.pop("id", str(uuid.uuid4())),
        name=name,
        display_name=kwargs.pop("display_name", name.replace("_", " ").title()),
        definition=kwargs.pop("definition", f"Definition of {name}"),
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


class TestCanonicalSourceDataclass:
    def test_default_none(self):
        term = _make_term()
        assert term.canonical_source is None

    def test_set_value(self):
        term = _make_term(canonical_source="sales.customers")
        assert term.canonical_source == "sales.customers"


class TestCanonicalSourceRelational:
    def test_add_and_retrieve(self, vs, rel):
        term = _make_term(canonical_source="sales.customers")
        rel.add_glossary_term(term)
        got = rel.get_glossary_term("revenue", SESSION_ID, user_id=USER_ID)
        assert got is not None
        assert got.canonical_source == "sales.customers"

    def test_default_none_in_db(self, vs, rel):
        term = _make_term()
        rel.add_glossary_term(term)
        got = rel.get_glossary_term("revenue", SESSION_ID, user_id=USER_ID)
        assert got is not None
        assert got.canonical_source is None

    def test_update_set(self, vs, rel):
        term = _make_term()
        rel.add_glossary_term(term)
        rel.update_glossary_term("revenue", SESSION_ID, {"canonical_source": "sales.customers"}, user_id=USER_ID)
        got = rel.get_glossary_term("revenue", SESSION_ID, user_id=USER_ID)
        assert got.canonical_source == "sales.customers"

    def test_update_clear(self, vs, rel):
        term = _make_term(canonical_source="sales.customers")
        rel.add_glossary_term(term)
        rel.update_glossary_term("revenue", SESSION_ID, {"canonical_source": None}, user_id=USER_ID)
        got = rel.get_glossary_term("revenue", SESSION_ID, user_id=USER_ID)
        assert got.canonical_source is None


class TestCanonicalSourceColumns:
    def test_in_glossary_columns(self):
        assert "canonical_source" in RelationalStore._ALL_GLOSSARY_COLUMNS
