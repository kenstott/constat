# Copyright (c) 2025 Kenneth Stott
# Canary: 5357bfe8-453f-4b17-b8e7-b59822201a43
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tests for constat_jupyter.entity_cache — in-memory cache + inflate."""

import copy
import sys
from pathlib import Path

import pytest

# Ensure constat-jupyter package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "constat-jupyter"))

from constat_jupyter.entity_cache import (
    EntityCache,
    inflate_glossary,
    K_ENTITIES,
    K_GLOSSARY,
    K_RELATIONSHIPS,
    K_CLUSTERS,
    GK_NAME,
    GK_DISPLAY,
    GK_DEF,
    GK_STATUS,
    GK_PARENT,
    GK_ALIASES,
    GK_DOMAIN,
    GK_DOMAIN_PATH,
    GK_PARENT_VERB,
    GK_GLOSSARY_STATUS,
    GK_ENTITY_ID,
    GK_STYPE,
    GK_NER_TYPE,
    GK_TAGS,
    GK_IGNORED,
    GK_CANONICAL_SOURCE,
    RK_SUBJECT,
    RK_VERB,
    RK_OBJECT,
    RK_CONFIDENCE,
    RK_USER_EDITED,
    EMPTY_STATE,
)


# -- Fixtures ---------------------------------------------------------------

def _sample_state() -> dict:
    return {
        K_ENTITIES: {
            "ent-1": {"a": "revenue", "b": "Revenue", "c": "metric"},
        },
        K_GLOSSARY: {
            "revenue": {
                GK_NAME: "revenue",
                GK_DISPLAY: "Revenue",
                GK_DEF: "Total income from sales",
                GK_STATUS: "approved",
                GK_DOMAIN: "sales-analytics",
                GK_DOMAIN_PATH: "/domains/sales",
                GK_GLOSSARY_STATUS: "defined",
                GK_ENTITY_ID: "ent-1",
                GK_STYPE: "metric",
                GK_NER_TYPE: "QUANTITY",
                GK_TAGS: {"source": "schema"},
                GK_CANONICAL_SOURCE: "sales_db",
            },
            "customer": {
                GK_NAME: "customer",
                GK_DISPLAY: "Customer",
                GK_ENTITY_ID: "ent-2",
                GK_STYPE: "entity",
            },
        },
        K_RELATIONSHIPS: {
            "rel-1": {
                RK_SUBJECT: "customer",
                RK_VERB: "GENERATES",
                RK_OBJECT: "revenue",
                RK_CONFIDENCE: 0.9,
                RK_USER_EDITED: True,
            },
        },
        K_CLUSTERS: {
            "revenue": ["income", "sales"],
        },
    }


# -- EntityCache tests ------------------------------------------------------

class TestEntityCache:
    def test_get_empty(self):
        cache = EntityCache()
        assert cache.get("nonexistent") is None

    def test_set_and_get(self):
        cache = EntityCache()
        state = _sample_state()
        cache.set("s1", state, 3)
        entry = cache.get("s1")
        assert entry is not None
        assert entry.version == 3
        assert entry.state is state

    def test_apply_patch_add(self):
        cache = EntityCache()
        state = _sample_state()
        cache.set("s1", copy.deepcopy(state), 1)

        patch_ops = [
            {"op": "add", "path": "/g/profit", "value": {
                GK_NAME: "profit",
                GK_DISPLAY: "Profit",
                GK_DEF: "Revenue minus costs",
                GK_GLOSSARY_STATUS: "defined",
            }},
        ]
        new_state = cache.apply_patch("s1", patch_ops, 2)
        assert "profit" in new_state[K_GLOSSARY]
        assert cache.get("s1").version == 2

    def test_apply_patch_replace(self):
        cache = EntityCache()
        state = _sample_state()
        cache.set("s1", copy.deepcopy(state), 1)

        patch_ops = [
            {"op": "replace", "path": "/g/revenue/c", "value": "Net revenue after returns"},
        ]
        new_state = cache.apply_patch("s1", patch_ops, 2)
        assert new_state[K_GLOSSARY]["revenue"][GK_DEF] == "Net revenue after returns"

    def test_apply_patch_remove(self):
        cache = EntityCache()
        state = _sample_state()
        cache.set("s1", copy.deepcopy(state), 1)

        patch_ops = [
            {"op": "remove", "path": "/g/customer"},
        ]
        new_state = cache.apply_patch("s1", patch_ops, 2)
        assert "customer" not in new_state[K_GLOSSARY]

    def test_apply_patch_no_prior_state(self):
        cache = EntityCache()
        patch_ops = [
            {"op": "add", "path": "/g", "value": {}},
            {"op": "add", "path": "/g/x", "value": {GK_NAME: "x", GK_DISPLAY: "X"}},
        ]
        new_state = cache.apply_patch("s1", patch_ops, 1)
        assert "x" in new_state[K_GLOSSARY]

    def test_clear(self):
        cache = EntityCache()
        cache.set("s1", {}, 1)
        cache.clear("s1")
        assert cache.get("s1") is None

    def test_clear_nonexistent(self):
        cache = EntityCache()
        cache.clear("nope")  # Should not raise


# -- inflate_glossary tests --------------------------------------------------

class TestInflateGlossary:
    def test_empty_state(self):
        terms = inflate_glossary(EMPTY_STATE)
        assert terms == []

    def test_basic_inflation(self):
        state = _sample_state()
        terms = inflate_glossary(state)
        assert len(terms) == 2

        by_name = {t["name"]: t for t in terms}
        rev = by_name["revenue"]
        assert rev["display_name"] == "Revenue"
        assert rev["definition"] == "Total income from sales"
        assert rev["domain"] == "sales-analytics"
        assert rev["domain_path"] == "/domains/sales"
        assert rev["glossary_status"] == "defined"
        assert rev["entity_id"] == "ent-1"
        assert rev["semantic_type"] == "metric"
        assert rev["ner_type"] == "QUANTITY"
        assert rev["tags"] == {"source": "schema"}
        assert rev["canonical_source"] == "sales_db"
        assert rev["cluster_siblings"] == ["income", "sales"]
        assert rev["ignored"] is False

    def test_relationships_inflated(self):
        state = _sample_state()
        terms = inflate_glossary(state)
        by_name = {t["name"]: t for t in terms}

        # Both customer and revenue should see the relationship
        cust_rels = by_name["customer"]["relationships"]
        rev_rels = by_name["revenue"]["relationships"]
        assert len(cust_rels) == 1
        assert len(rev_rels) == 1
        assert cust_rels[0]["verb"] == "GENERATES"
        assert cust_rels[0]["user_edited"] is True

    def test_self_describing_default(self):
        state = _sample_state()
        terms = inflate_glossary(state)
        by_name = {t["name"]: t for t in terms}

        # customer has no definition and no explicit glossary_status
        cust = by_name["customer"]
        assert cust["glossary_status"] == "self_describing"

    def test_parent_child_resolution(self):
        state = {
            K_ENTITIES: {},
            K_GLOSSARY: {
                "animal": {
                    GK_NAME: "animal",
                    GK_DISPLAY: "Animal",
                    GK_ENTITY_ID: "ent-animal",
                },
                "dog": {
                    GK_NAME: "dog",
                    GK_DISPLAY: "Dog",
                    GK_PARENT: "ent-animal",
                    GK_PARENT_VERB: "IS_A",
                    GK_ENTITY_ID: "ent-dog",
                },
            },
            K_RELATIONSHIPS: {},
            K_CLUSTERS: {},
        }
        terms = inflate_glossary(state)
        by_name = {t["name"]: t for t in terms}

        dog = by_name["dog"]
        assert dog["parent_id"] == "ent-animal"
        assert dog["parent_verb"] == "IS_A"
        assert dog["parent"] == {"name": "animal", "display_name": "Animal"}

        animal = by_name["animal"]
        assert len(animal["children"]) == 1
        assert animal["children"][0]["name"] == "dog"

    def test_no_clusters(self):
        state = _sample_state()
        terms = inflate_glossary(state)
        by_name = {t["name"]: t for t in terms}
        # customer has no cluster entry
        assert by_name["customer"]["cluster_siblings"] is None

    def test_aliases(self):
        state = {
            K_ENTITIES: {},
            K_GLOSSARY: {
                "rev": {
                    GK_NAME: "rev",
                    GK_DISPLAY: "Rev",
                    GK_ALIASES: ["revenue", "income"],
                },
            },
            K_RELATIONSHIPS: {},
            K_CLUSTERS: {},
        }
        terms = inflate_glossary(state)
        assert terms[0]["aliases"] == ["revenue", "income"]

    def test_ignored_term(self):
        state = {
            K_ENTITIES: {},
            K_GLOSSARY: {
                "noise": {
                    GK_NAME: "noise",
                    GK_DISPLAY: "Noise",
                    GK_IGNORED: True,
                },
            },
            K_RELATIONSHIPS: {},
            K_CLUSTERS: {},
        }
        terms = inflate_glossary(state)
        assert terms[0]["ignored"] is True
