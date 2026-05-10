from __future__ import annotations

# Copyright (c) 2025 Kenneth Stott
# Canary: ddee5f13-3545-4ff4-8baa-f19accc751a8
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Glossary term CRUD tests: create, update, delete, relationships, multi-domain."""

import pytest

from tests.e2e.test_glossary_ui import (
    _gql,
    BULK_UPDATE_STATUS,
    CREATE_RELATIONSHIP,
    CREATE_TERM,
    CREATE_TERM_WITH_DOMAIN,
    DELETE_RELATIONSHIP,
    DELETE_TERM,
    DELETE_TERM_WITH_DOMAIN,
    GLOSSARY_QUERY,
    TERM_QUERY,
    UPDATE_RELATIONSHIP,
    UPDATE_TERM,
    UPDATE_TERM_WITH_DOMAIN,
)

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# GraphQL mutation tests
# ---------------------------------------------------------------------------

class TestGlossaryMutations:
    """Test glossary CRUD mutations via GraphQL API."""

    def test_create_term(self, server_url, session_id):
        """Create a glossary term via GraphQL mutation."""
        result = _gql(server_url, CREATE_TERM, {
            "sid": session_id,
            "input": {
                "name": "customer",
                "definition": "An entity that purchases goods or services",
                "parentId": "__root__",
            },
        })
        term = result["createGlossaryTerm"]
        assert term["name"] == "customer"
        assert term["definition"] == "An entity that purchases goods or services"
        assert term["glossaryStatus"] == "defined"

    def test_update_term_definition(self, server_url, session_id):
        """Update a term's definition via GraphQL mutation."""
        result = _gql(server_url, UPDATE_TERM, {
            "sid": session_id,
            "name": "customer",
            "input": {"definition": "Updated: a buyer of products or services"},
        })
        assert result["updateGlossaryTerm"]["definition"] == "Updated: a buyer of products or services"

    def test_update_term_tags(self, server_url, session_id):
        """Update a term's tags via GraphQL mutation."""
        result = _gql(server_url, UPDATE_TERM, {
            "sid": session_id,
            "name": "customer",
            "input": {"tags": {"PII": {}, "ENTITY": {"note": "core business entity"}}},
        })
        tags = result["updateGlossaryTerm"].get("tags") or {}
        assert "PII" in tags
        assert "ENTITY" in tags

    def test_update_term_aliases(self, server_url, session_id):
        """Update a term's aliases via GraphQL mutation."""
        result = _gql(server_url, UPDATE_TERM, {
            "sid": session_id,
            "name": "customer",
            "input": {"aliases": ["client", "buyer"]},
        })
        aliases = result["updateGlossaryTerm"].get("aliases") or []
        assert "client" in aliases
        assert "buyer" in aliases

    def test_create_second_term(self, server_url, session_id):
        """Create a second term for relationship testing."""
        result = _gql(server_url, CREATE_TERM, {
            "sid": session_id,
            "input": {
                "name": "order",
                "definition": "A purchase transaction placed by a customer",
                "parentId": "__root__",
            },
        })
        assert result["createGlossaryTerm"]["name"] == "order"

    def test_create_relationship(self, server_url, session_id):
        """Create a relationship between two terms via GraphQL."""
        result = _gql(server_url, CREATE_RELATIONSHIP, {
            "sid": session_id,
            "subject": "customer",
            "verb": "PLACES",
            "object": "order",
        })
        rel = result["createRelationship"]
        assert rel["subject"] == "customer"
        assert rel["verb"] == "PLACES"
        assert rel["object"] == "order"
        assert rel["id"]

    def test_update_relationship(self, server_url, session_id):
        """Update a relationship's verb via GraphQL."""
        created = _gql(server_url, CREATE_RELATIONSHIP, {
            "sid": session_id,
            "subject": "customer",
            "verb": "HAS_ONE",
            "object": "order",
        })
        rel_id = created["createRelationship"]["id"]

        result = _gql(server_url, UPDATE_RELATIONSHIP, {
            "sid": session_id,
            "relId": rel_id,
            "verb": "HAS_MANY",
        })
        assert result["updateRelationship"]["verb"] == "HAS_MANY"

    def test_delete_relationship(self, server_url, session_id):
        """Delete a relationship via GraphQL."""
        created = _gql(server_url, CREATE_RELATIONSHIP, {
            "sid": session_id,
            "subject": "order",
            "verb": "HAS_KIND",
            "object": "customer",
        })
        rel_id = created["createRelationship"]["id"]

        result = _gql(server_url, DELETE_RELATIONSHIP, {
            "sid": session_id,
            "relId": rel_id,
        })
        assert result["deleteRelationship"] is True

    def test_verify_tags_via_query(self, server_url, session_id):
        """Verify tags set earlier are readable via single-term query."""
        data = _gql(server_url, TERM_QUERY, {"sid": session_id, "name": "customer"})
        term = data["glossaryTerm"]
        assert term is not None, "customer term not found via glossaryTerm query"
        tags = term.get("tags") or {}
        assert "PII" in tags
        assert "ENTITY" in tags

    def test_remove_tags(self, server_url, session_id):
        """Remove tags from a term by setting empty tags."""
        result = _gql(server_url, UPDATE_TERM, {
            "sid": session_id,
            "name": "customer",
            "input": {"tags": {}},
        })
        tags = result["updateGlossaryTerm"].get("tags") or {}
        assert "PII" not in tags

        data = _gql(server_url, TERM_QUERY, {"sid": session_id, "name": "customer"})
        assert not (data["glossaryTerm"].get("tags") or {})

    def test_delete_terms(self, server_url, session_id):
        """Delete created terms via GraphQL mutation."""
        result = _gql(server_url, DELETE_TERM, {
            "sid": session_id,
            "name": "order",
        })
        assert result["deleteGlossaryTerm"] is True

        result = _gql(server_url, DELETE_TERM, {
            "sid": session_id,
            "name": "customer",
        })
        assert result["deleteGlossaryTerm"] is True

        for name in ["customer", "order"]:
            data = _gql(server_url, TERM_QUERY, {"sid": session_id, "name": name})
            assert data["glossaryTerm"] is None, f"Term '{name}' should be deleted"


class TestMultiDomainGlossary:
    """Test creating/updating/deleting the same term across domains."""

    def test_create_same_term_two_domains(self, server_url, session_id):
        """Create 'metric' in both 'sales' and 'hr' domains."""
        r1 = _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "input": {
                "name": "metric",
                "definition": "A sales performance indicator",
                "domain": "sales",
                "parentId": "__root__",
            },
        })
        assert r1["createGlossaryTerm"]["name"] == "metric"
        assert r1["createGlossaryTerm"]["domain"] == "sales"

        r2 = _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "input": {
                "name": "metric",
                "definition": "An HR analytics measure",
                "domain": "hr",
                "parentId": "__root__",
            },
        })
        assert r2["createGlossaryTerm"]["name"] == "metric"
        assert r2["createGlossaryTerm"]["domain"] == "hr"

    def test_update_term_by_domain(self, server_url, session_id):
        """Update 'metric' in 'sales' domain without affecting 'hr'."""
        result = _gql(server_url, UPDATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "name": "metric",
            "domain": "sales",
            "input": {"definition": "Updated sales metric definition"},
        })
        assert result["updateGlossaryTerm"]["definition"] == "Updated sales metric definition"

    def test_delete_term_by_domain(self, server_url, session_id):
        """Delete 'metric' from 'hr' domain, leaving 'sales' intact."""
        result = _gql(server_url, DELETE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "name": "metric",
            "domain": "hr",
        })
        assert result["deleteGlossaryTerm"] is True

        _gql(server_url, DELETE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "name": "metric",
            "domain": "sales",
        })


class TestAbstractTerms:
    """Test abstract term creation and parent/child management."""

    def test_create_abstract_term(self, server_url, session_id):
        """Create an abstract term (is_abstract=true, no entity grounding needed)."""
        result = _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "input": {
                "name": "kpi",
                "definition": "Key performance indicator",
                "isAbstract": True,
            },
        })
        term = result["createGlossaryTerm"]
        assert term["name"] == "kpi"
        assert term["glossaryStatus"] == "defined"

    def test_set_and_remove_parent(self, server_url, session_id):
        """Create a child term, set parent, then remove parent."""
        child = _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "input": {
                "name": "conversion rate",
                "definition": "Percentage of leads that become customers",
                "parentId": "__root__",
            },
        })
        assert child["createGlossaryTerm"]["name"] == "conversion rate"

        result = _gql(server_url, UPDATE_TERM, {
            "sid": session_id,
            "name": "conversion rate",
            "input": {"parentId": "kpi"},
        })
        assert result["updateGlossaryTerm"]["name"] == "conversion rate"

        new_term = _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
            "sid": session_id,
            "input": {
                "name": "churn rate",
                "definition": "Rate of customer attrition",
                "parentId": "kpi",
            },
        })
        assert new_term["createGlossaryTerm"]["name"] == "churn rate"

        result = _gql(server_url, UPDATE_TERM, {
            "sid": session_id,
            "name": "churn rate",
            "input": {"parentId": ""},
        })
        assert result["updateGlossaryTerm"]["name"] == "churn rate"

        _gql(server_url, DELETE_TERM, {"sid": session_id, "name": "churn rate"})
        _gql(server_url, DELETE_TERM, {"sid": session_id, "name": "conversion rate"})
        _gql(server_url, DELETE_TERM, {"sid": session_id, "name": "kpi"})


class TestBatchDraftDeletion:
    """Test bulk status update for draft terms."""

    def test_batch_delete_drafts_progressive(self, server_url, session_id):
        """Create draft terms, bulk-approve some, verify counts change."""
        names = [f"batch_term_{i}" for i in range(4)]

        for name in names:
            _gql(server_url, CREATE_TERM_WITH_DOMAIN, {
                "sid": session_id,
                "input": {
                    "name": name,
                    "definition": f"Definition for {name}",
                    "parentId": "__root__",
                },
            })

        result = _gql(server_url, BULK_UPDATE_STATUS, {
            "sid": session_id,
            "names": names[:2],
            "newStatus": "approved",
        })
        assert result["bulkUpdateStatus"] == 2

        data = _gql(server_url, GLOSSARY_QUERY, {"sid": session_id})
        all_terms = data["glossary"]["terms"]
        batch_terms = [t for t in all_terms if t["name"].startswith("batch_term_")]
        assert len(batch_terms) >= 4

        result = _gql(server_url, BULK_UPDATE_STATUS, {
            "sid": session_id,
            "names": names[2:],
            "newStatus": "approved",
        })
        assert result["bulkUpdateStatus"] == 2

        for name in names:
            _gql(server_url, DELETE_TERM, {"sid": session_id, "name": name})


class TestGlossaryMutationLifecycle:
    """End-to-end lifecycle: create, tag, relate, then delete all."""

    def test_full_lifecycle(self, server_url, session_id):
        """Create terms, tag them, relate them, then clean up everything."""
        sid = session_id

        _gql(server_url, CREATE_TERM, {
            "sid": sid,
            "input": {"name": "product", "definition": "A good or service offered for sale", "parentId": "__root__"},
        })
        _gql(server_url, CREATE_TERM, {
            "sid": sid,
            "input": {"name": "category", "definition": "A classification group for products", "parentId": "__root__"},
        })

        result = _gql(server_url, UPDATE_TERM, {
            "sid": sid,
            "name": "product",
            "input": {"tags": {"CORE": {}, "CATALOG": {}}},
        })
        tags = result["updateGlossaryTerm"].get("tags") or {}
        assert "CORE" in tags

        data = _gql(server_url, TERM_QUERY, {"sid": sid, "name": "product"})
        assert "CORE" in (data["glossaryTerm"].get("tags") or {})

        rel = _gql(server_url, CREATE_RELATIONSHIP, {
            "sid": sid,
            "subject": "product",
            "verb": "HAS_ONE",
            "object": "category",
        })
        rel_id = rel["createRelationship"]["id"]

        _gql(server_url, DELETE_RELATIONSHIP, {"sid": sid, "relId": rel_id})
        _gql(server_url, DELETE_TERM, {"sid": sid, "name": "product"})
        _gql(server_url, DELETE_TERM, {"sid": sid, "name": "category"})
