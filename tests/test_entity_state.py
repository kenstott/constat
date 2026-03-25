"""Tests for constat.server.entity_state — compact state builder and JSON Patch differ."""

from unittest.mock import MagicMock

import pytest

from constat.server.entity_state import (
    K_ENTITIES,
    K_GLOSSARY,
    K_RELATIONSHIPS,
    K_CLUSTERS,
    EK_NAME,
    EK_DISPLAY,
    EK_STYPE,
    EK_NER,
    EK_DOMAIN,
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
    RK_SUBJECT,
    RK_VERB,
    RK_OBJECT,
    RK_CONFIDENCE,
    build_compact_state,
    compute_entity_patch,
    _build_domain_maps,
    _resolve_entity_domain,
    _resolve_entity_domains,
)


def _mock_managed(entities=None, glossary_rows=None, relationships=None, clusters=None):
    """Create a mock ManagedSession with test data."""
    entity_rows = entities or []
    rel_rows = relationships or []
    cluster_rows = clusters or []
    unified_glossary = glossary_rows or []

    # Mock relational store conn
    conn = MagicMock()

    def execute_side_effect(sql, params=None):
        result = MagicMock()
        if "FROM entities" in sql:
            result.fetchall.return_value = entity_rows
        elif "FROM glossary_clusters" in sql:
            result.fetchall.return_value = cluster_rows
        else:
            result.fetchall.return_value = []
        return result

    conn.execute.side_effect = execute_side_effect

    # Mock relational store
    store = MagicMock()
    store._conn = conn
    store.list_session_relationships.return_value = rel_rows

    # Mock vector store
    vs = MagicMock()
    vs._relational = store
    vs.get_unified_glossary.return_value = unified_glossary
    vs.get_entity_document_names.return_value = []

    # Mock config with no domains
    config = MagicMock()
    config.domains = {}
    config.databases = {}
    config.apis = None
    config.documents = None

    # Mock session
    session = MagicMock()
    session.doc_tools._vector_store = vs
    session.config = config
    session.resources = None

    # Mock ManagedSession
    managed = MagicMock()
    managed.session = session
    managed.active_domains = []
    managed.user_id = "test-user"

    return managed


def _make_glossary_row(
    name, display_name, definition=None, status=None, parent_id=None,
    aliases=None, domain=None, entity_id=None, glossary_id=None,
    parent_verb=None, glossary_status="self_describing", semantic_type=None,
    ner_type=None, tags=None, ignored=False,
):
    """Helper to build a unified glossary dict row."""
    return {
        "entity_id": entity_id,
        "name": name,
        "display_name": display_name,
        "semantic_type": semantic_type,
        "ner_type": ner_type,
        "session_id": "sess1",
        "glossary_id": glossary_id,
        "domain": domain,
        "entity_domain_id": None,
        "definition": definition,
        "parent_id": parent_id,
        "parent_verb": parent_verb,
        "aliases": aliases or [],
        "cardinality": None,
        "plural": None,
        "status": status,
        "provenance": None,
        "glossary_status": glossary_status,
        "ignored": ignored,
        "tags": tags,
    }


class TestBuildCompactState:
    def test_empty_session(self):
        managed = _mock_managed()
        state = build_compact_state("sess1", managed)
        assert state == {K_ENTITIES: {}, K_GLOSSARY: {}, K_RELATIONSHIPS: {}, K_CLUSTERS: {}}

    def test_entities(self):
        entities = [
            ("id1", "order", "Order", "concept", "SCHEMA", "sales"),
            ("id2", "norway", "Norway", "concept", "GPE", None),
        ]
        managed = _mock_managed(entities=entities)
        state = build_compact_state("sess1", managed)

        assert "id1" in state[K_ENTITIES]
        e1 = state[K_ENTITIES]["id1"]
        assert e1[EK_NAME] == "order"
        assert e1[EK_DISPLAY] == "Order"
        assert e1[EK_STYPE] == "concept"
        assert e1[EK_NER] == "SCHEMA"
        assert e1[EK_DOMAIN] == "sales"

        e2 = state[K_ENTITIES]["id2"]
        assert e2[EK_NER] == "GPE"
        assert EK_DOMAIN not in e2  # None domain omitted

    def test_entity_optional_fields_omitted(self):
        entities = [("id1", "item", "Item", "concept", None, None)]
        managed = _mock_managed(entities=entities)
        state = build_compact_state("sess1", managed)
        e = state[K_ENTITIES]["id1"]
        assert EK_NER not in e
        assert EK_DOMAIN not in e

    def test_glossary_terms_keyed_by_name(self):
        rows = [
            _make_glossary_row("order", "Order", definition="A purchase", status="reviewed"),
            _make_glossary_row("customer", "Customer", definition="A buyer", status="draft",
                               parent_id="g1", aliases=["client", "buyer"]),
        ]
        managed = _mock_managed(glossary_rows=rows)
        state = build_compact_state("sess1", managed)

        # Keyed by lowercased name, not by ID
        assert "order" in state[K_GLOSSARY]
        assert "customer" in state[K_GLOSSARY]

        g1 = state[K_GLOSSARY]["order"]
        assert g1[GK_NAME] == "order"
        assert g1[GK_STATUS] == "reviewed"
        assert GK_PARENT not in g1
        assert GK_ALIASES not in g1  # empty list omitted

        g2 = state[K_GLOSSARY]["customer"]
        assert g2[GK_PARENT] == "g1"
        assert g2[GK_ALIASES] == ["client", "buyer"]

    def test_glossary_new_fields(self):
        rows = [
            _make_glossary_row(
                "revenue", "Revenue", definition="Total income", status="reviewed",
                domain="sales.yaml", entity_id="ent1", glossary_id="g1",
                parent_verb="MEASURES", glossary_status="defined",
                semantic_type="metric", ner_type="SCHEMA",
                tags={"source": "db", "_internal": "skip"},
                ignored=True,
            ),
        ]
        managed = _mock_managed(glossary_rows=rows)
        # Set up domain_path_map via config
        dcfg = MagicMock()
        dcfg.path = "Sales Analytics"
        dcfg.databases = {}
        dcfg.apis = {}
        dcfg.documents = {}
        managed.session.config.domains = {"sales.yaml": dcfg}

        state = build_compact_state("sess1", managed)
        g = state[K_GLOSSARY]["revenue"]

        assert g[GK_DOMAIN] == "sales.yaml"
        assert g[GK_DOMAIN_PATH] == "Sales Analytics"
        assert g[GK_PARENT_VERB] == "MEASURES"
        assert g[GK_GLOSSARY_STATUS] == "defined"
        assert g[GK_ENTITY_ID] == "ent1"
        assert g[GK_STYPE] == "metric"
        assert g[GK_NER_TYPE] == "SCHEMA"
        assert g[GK_TAGS] == {"source": "db"}  # _internal filtered
        assert g[GK_IGNORED] is True

    def test_glossary_defaults_omitted(self):
        """Default values (HAS_KIND, self_describing) should not appear in compact output."""
        rows = [
            _make_glossary_row("item", "Item", parent_verb="HAS_KIND", glossary_status="self_describing"),
        ]
        managed = _mock_managed(glossary_rows=rows)
        state = build_compact_state("sess1", managed)
        g = state[K_GLOSSARY]["item"]
        assert GK_PARENT_VERB not in g
        assert GK_GLOSSARY_STATUS not in g

    def test_glossary_base_domain_becomes_system(self):
        """__base__ domain should be rendered as 'system'."""
        rows = [
            _make_glossary_row("item", "Item", domain="__base__"),
        ]
        managed = _mock_managed(glossary_rows=rows)
        state = build_compact_state("sess1", managed)
        g = state[K_GLOSSARY]["item"]
        assert g[GK_DOMAIN] == "system"

    def test_relationships(self):
        rels = [
            {"id": "r1", "subject_name": "customer", "verb": "PLACES",
             "object_name": "order", "confidence": 0.95},
        ]
        managed = _mock_managed(relationships=rels)
        state = build_compact_state("sess1", managed)

        r1 = state[K_RELATIONSHIPS]["r1"]
        assert r1[RK_SUBJECT] == "customer"
        assert r1[RK_VERB] == "PLACES"
        assert r1[RK_OBJECT] == "order"
        assert r1[RK_CONFIDENCE] == 0.95

    def test_clusters(self):
        cluster_rows = [
            ("norway", 1), ("sweden", 1), ("denmark", 1),
            ("order", 2), ("invoice", 2),
            ("lonely", 3),  # single-member cluster — excluded
        ]
        managed = _mock_managed(clusters=cluster_rows)
        state = build_compact_state("sess1", managed)

        clusters = state[K_CLUSTERS]
        assert "norway" in clusters
        assert "sweden" in clusters["norway"]
        assert "denmark" in clusters["norway"]
        assert "norway" not in clusters["norway"]  # self excluded

        assert "order" in clusters
        assert clusters["order"] == ["invoice"]

        assert "lonely" not in clusters  # single-member excluded


class TestDomainHelpers:
    def test_build_domain_maps_empty_config(self):
        config = MagicMock()
        config.domains = {}
        config.databases = {}
        config.apis = None
        config.documents = None
        path_map, src_map = _build_domain_maps(config)
        assert path_map == {}
        assert src_map == {}

    def test_build_domain_maps_with_domains(self):
        dcfg = MagicMock()
        dcfg.path = "Sales Analytics"
        dcfg.databases = {"sales_db": True}
        dcfg.apis = {}
        dcfg.documents = {}

        config = MagicMock()
        config.domains = {"sales.yaml": dcfg}
        config.databases = {}
        config.apis = None
        config.documents = None

        path_map, src_map = _build_domain_maps(config)
        assert path_map["sales.yaml"] == "Sales Analytics"
        assert path_map["sales"] == "Sales Analytics"
        assert src_map["sales_db"] == "sales.yaml"

    def test_resolve_entity_domain_single(self):
        vs = MagicMock()
        vs.get_entity_document_names.return_value = ["schema:sales_db.orders"]
        result = _resolve_entity_domain("ent1", vs, {"sales_db": "sales.yaml"})
        assert result == "sales.yaml"

    def test_resolve_entity_domain_cross(self):
        vs = MagicMock()
        vs.get_entity_document_names.return_value = [
            "schema:sales_db.orders", "schema:hr_db.employees"
        ]
        result = _resolve_entity_domain("ent1", vs, {
            "sales_db": "sales.yaml", "hr_db": "hr.yaml"
        })
        assert result == "cross-domain"

    def test_resolve_entity_domain_none(self):
        assert _resolve_entity_domain(None, MagicMock(), {}) is None

    def test_resolve_entity_domains_crawled(self):
        vs = MagicMock()
        vs.get_entity_document_names.return_value = ["hr_management:crawled_8"]
        result = _resolve_entity_domains("ent1", vs, {"hr_management": "hr.yaml"})
        assert result == ["hr.yaml"]


class TestComputeEntityPatch:
    def test_identical_states(self):
        state = {K_ENTITIES: {"id1": {EK_NAME: "x"}}}
        assert compute_entity_patch(state, state) == []

    def test_added_entity(self):
        old = {K_ENTITIES: {}}
        new = {K_ENTITIES: {"id1": {EK_NAME: "order"}}}
        ops = compute_entity_patch(old, new)
        assert len(ops) == 1
        assert ops[0]["op"] == "add"
        assert ops[0]["path"] == "/e/id1"

    def test_removed_entity(self):
        old = {K_ENTITIES: {"id1": {EK_NAME: "order"}}}
        new = {K_ENTITIES: {}}
        ops = compute_entity_patch(old, new)
        assert len(ops) == 1
        assert ops[0]["op"] == "remove"
        assert ops[0]["path"] == "/e/id1"

    def test_modified_entity(self):
        old = {K_ENTITIES: {"id1": {EK_NAME: "order", EK_DISPLAY: "Order"}}}
        new = {K_ENTITIES: {"id1": {EK_NAME: "order", EK_DISPLAY: "Purchase Order"}}}
        ops = compute_entity_patch(old, new)
        assert len(ops) >= 1
        # Should have a replace op for the display field
        replace_ops = [op for op in ops if op["op"] == "replace"]
        assert len(replace_ops) == 1
        assert replace_ops[0]["path"] == "/e/id1/b"

    def test_empty_to_empty(self):
        assert compute_entity_patch({}, {}) == []

    def test_rfc_6902_format(self):
        old = {K_ENTITIES: {}}
        new = {K_ENTITIES: {"x": {"a": "test"}}, K_GLOSSARY: {"g1": {"a": "term"}}}
        ops = compute_entity_patch(old, new)
        for op in ops:
            assert "op" in op
            assert "path" in op
            assert op["op"] in ("add", "remove", "replace", "move", "copy", "test")
