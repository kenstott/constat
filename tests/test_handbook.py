# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for the Domain Handbook feature."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from constat.session._handbook import (
    DomainHandbook,
    HandbookEntry,
    HandbookMixin,
    HandbookSection,
)


# ---------------------------------------------------------------------------
# Helpers: lightweight stubs that satisfy the mixin's attribute expectations
# ---------------------------------------------------------------------------


@dataclass
class FakeDbConfig:
    name: str
    type: str


@dataclass
class FakeApiConfig:
    name: str


@dataclass
class FakeConfig:
    databases: list = field(default_factory=list)
    apis: list = field(default_factory=list)
    domains: list = field(default_factory=list)


@dataclass
class FakeGlossaryTerm:
    id: str
    name: str
    display_name: str
    definition: str
    domain: str | None = None
    aliases: list = field(default_factory=list)
    semantic_type: str | None = None
    status: str = "draft"
    provenance: str = "llm"
    session_id: str = "sess-1"
    user_id: str = "default"
    grounded_in: str | None = None


@dataclass
class FakeAgent:
    name: str
    description: str
    domain: str = ""


@dataclass
class FakeSkill:
    name: str
    description: str
    domain: str = ""


class FakeRelationalStore:
    """Minimal stub for RelationalStore."""

    def __init__(self, entities=None, glossary_terms=None):
        self._entities = entities or []
        self._glossary_terms = glossary_terms or []

    @staticmethod
    def entity_visibility_filter(session_id, active_domains=None, alias="e", cross_session=False):
        return "1=1", []

    def list_entities_with_refcount(self, vis_filter, vis_params):
        return self._entities

    def list_glossary_terms(self, session_id, domain=None, scope="all", user_id=None):
        if domain:
            return [t for t in self._glossary_terms if t.domain == domain]
        return self._glossary_terms

    def get_glossary_term_by_id(self, term_id):
        for t in self._glossary_terms:
            if t.id == term_id:
                return t
        return None

    def update_glossary_term(self, name, session_id, updates, user_id=None, domain=None):
        return True

    def update_entity_name(self, entity_id, name, display_name):
        pass


class FakeVectorStore:
    def __init__(self, relational=None):
        self._relational = relational


class FakeDocTools:
    def __init__(self, vector_store=None):
        self._vector_store = vector_store


class FakeSchemaManager:
    def get_tables(self, db_name):
        return ["table1", "table2"]


class FakeAgentManager:
    def __init__(self, agents=None):
        self._agents = {a.name: a for a in (agents or [])}

    def list_agents(self, domain=None):
        if domain:
            return [n for n, a in self._agents.items() if a.domain == domain]
        return list(self._agents.keys())

    def get_agent(self, name):
        return self._agents.get(name)


class FakeSkillManager:
    def __init__(self, skills=None):
        self._skills = {s.name: s for s in (skills or [])}

    def list_skills(self, domain=None):
        if domain:
            return [n for n, s in self._skills.items() if s.domain == domain]
        return list(self._skills.keys())

    def get_skill(self, name):
        return self._skills.get(name)


class FakeHistory:
    def __init__(self, sessions=None):
        self._sessions = sessions or []

    def list_sessions(self, limit=20):
        return self._sessions[:limit]


class FakeLearningStore:
    def __init__(self, rules=None):
        self._rules = rules or []

    def list_rules(self, domain=None, **kwargs):
        if domain:
            return [r for r in self._rules if r.get("domain") == domain]
        return self._rules

    def update_rule(self, rule_id, summary=None, tags=None, confidence=None):
        return True


# ---------------------------------------------------------------------------
# The testable session stub: combines HandbookMixin with fake attributes
# ---------------------------------------------------------------------------


class StubSession(HandbookMixin):
    """A minimal object satisfying HandbookMixin's attribute expectations."""

    def __init__(
        self,
        config=None,
        relational=None,
        learning_store=None,
        agent_manager=None,
        skill_manager=None,
        history=None,
        session_databases=None,
        session_files=None,
        session_id="test-session",
        user_id="default",
        active_domains=None,
    ):
        self.config = config or FakeConfig()
        self.session_id = session_id
        self.user_id = user_id
        self.active_domains = active_domains or []
        self.session_databases = session_databases or {}
        self.session_files = session_files or {}
        self.learning_store = learning_store or FakeLearningStore()
        self.agent_manager = agent_manager or FakeAgentManager()
        self.skill_manager = skill_manager or FakeSkillManager()
        self.history = history or FakeHistory()
        self.schema_manager = FakeSchemaManager()

        # Wire up doc_tools -> vector_store -> relational
        relational = relational or FakeRelationalStore()
        vs = FakeVectorStore(relational=relational)
        self.doc_tools = FakeDocTools(vector_store=vs)


# ===========================================================================
# Tests
# ===========================================================================


class TestHandbookGeneration:
    """Test the top-level generate_handbook method."""

    def test_generate_returns_handbook(self):
        session = StubSession()
        handbook = session.generate_handbook(domain="test-domain")
        assert isinstance(handbook, DomainHandbook)
        assert handbook.domain == "test-domain"
        assert handbook.generated_at  # non-empty ISO string

    def test_generate_includes_all_sections(self):
        session = StubSession()
        handbook = session.generate_handbook(domain="test-domain")
        expected_keys = {
            "overview", "sources", "entities", "glossary",
            "rules", "patterns", "agents_skills", "limitations",
        }
        assert set(handbook.sections.keys()) == expected_keys

    def test_generate_uses_first_active_domain_when_none(self):
        session = StubSession(active_domains=["my-domain.yaml"])
        handbook = session.generate_handbook()
        assert handbook.domain == "my-domain.yaml"

    def test_generate_defaults_to_default_when_no_active(self):
        session = StubSession(active_domains=[])
        handbook = session.generate_handbook()
        assert handbook.domain == "default"

    def test_summary_reflects_non_empty_sections(self):
        rules = [{"id": "r1", "summary": "Test rule", "confidence": 0.9, "source_learnings": [], "tags": []}]
        session = StubSession(learning_store=FakeLearningStore(rules=rules))
        handbook = session.generate_handbook(domain="default")
        # rules section has content, so summary should mention it
        assert "Learned Rules" in handbook.summary


class TestOverviewSection:
    """Test _build_overview_section."""

    def test_overview_with_domain_config(self):
        @dataclass
        class DomainYaml:
            name: str = "Sales"
            description: str = "Sales analytics domain"
            filename: str = "sales.yaml"
            domains: list = field(default_factory=list)

        config = FakeConfig(domains=[DomainYaml()])
        session = StubSession(config=config)
        section = session._build_overview_section("sales.yaml")
        assert section.title == "Overview"
        assert len(section.content) >= 2  # name + description
        assert any("Sales" in e.display for e in section.content)

    def test_overview_empty_when_no_config(self):
        session = StubSession(config=FakeConfig())
        section = session._build_overview_section("unknown-domain")
        assert section.title == "Overview"
        assert len(section.content) == 0


class TestSourcesSection:
    def test_sources_with_databases(self):
        config = FakeConfig(databases=[FakeDbConfig(name="main_db", type="sql")])
        session = StubSession(config=config)
        section = session._build_sources_section("test")
        assert any("main_db" in e.display for e in section.content)

    def test_sources_with_session_databases(self):
        session = StubSession(session_databases={
            "uploads": {"type": "csv", "description": "Uploaded CSV"},
        })
        section = session._build_sources_section("test")
        assert any("uploads" in e.display for e in section.content)

    def test_sources_with_apis(self):
        config = FakeConfig(apis=[FakeApiConfig(name="rest_api")])
        session = StubSession(config=config)
        section = session._build_sources_section("test")
        assert any("rest_api" in e.display for e in section.content)

    def test_sources_with_files(self):
        session = StubSession(session_files={"doc.pdf": {"uri": "/tmp/doc.pdf"}})
        section = session._build_sources_section("test")
        assert any("doc.pdf" in e.display for e in section.content)


class TestEntitiesSection:
    def test_entities_listed(self):
        entities = [
            ("ent-1", "customer", "Customer", "CONCEPT", "ORG", 5),
            ("ent-2", "order", "Order", "CONCEPT", None, 3),
        ]
        relational = FakeRelationalStore(entities=entities)
        session = StubSession(relational=relational)
        section = session._build_entities_section("test")
        assert len(section.content) == 2
        assert section.content[0].metadata["name"] == "customer"
        assert section.content[0].editable is True

    def test_entities_empty_when_no_relational(self):
        session = StubSession()
        session.doc_tools = None  # no relational store
        section = session._build_entities_section("test")
        assert len(section.content) == 0


class TestGlossarySection:
    def test_glossary_terms_listed(self):
        terms = [
            FakeGlossaryTerm(id="g1", name="revenue", display_name="Revenue", definition="Total income", domain="test"),
            FakeGlossaryTerm(id="g2", name="churn", display_name="Churn", definition="Customer loss", domain="test"),
        ]
        relational = FakeRelationalStore(glossary_terms=terms)
        session = StubSession(relational=relational)
        section = session._build_glossary_section("test")
        assert len(section.content) == 2
        assert section.content[0].editable is True
        assert "Revenue" in section.content[0].display

    def test_glossary_empty(self):
        session = StubSession()
        section = session._build_glossary_section("test")
        assert len(section.content) == 0


class TestRulesSection:
    def test_rules_listed(self):
        rules = [
            {"id": "r1", "summary": "Always join on region_id", "confidence": 0.85,
             "category": "codegen_error", "applied_count": 3, "tags": ["sql"],
             "source_learnings": ["l1", "l2"]},
        ]
        session = StubSession(learning_store=FakeLearningStore(rules=rules))
        section = session._build_rules_section("default")
        assert len(section.content) == 1
        assert "85%" in section.content[0].display
        assert section.content[0].metadata["source_count"] == 2

    def test_rules_empty(self):
        session = StubSession(learning_store=FakeLearningStore(rules=[]))
        section = session._build_rules_section("default")
        assert len(section.content) == 0


class TestPatternsSection:
    def test_patterns_from_history(self):
        sessions = [
            {"queries": ["show revenue", "show revenue", "top customers"]},
            {"queries": ["show revenue"]},
        ]
        session = StubSession(history=FakeHistory(sessions=sessions))
        section = session._build_patterns_section("test")
        assert len(section.content) >= 1
        # "show revenue" appears 3 times, should be first
        assert section.content[0].metadata["count"] == 3

    def test_patterns_empty(self):
        session = StubSession(history=FakeHistory(sessions=[]))
        section = session._build_patterns_section("test")
        assert len(section.content) == 0


class TestAgentsSkillsSection:
    def test_agents_and_skills_listed(self):
        agents = [FakeAgent(name="analyst", description="Data analyst")]
        skills = [FakeSkill(name="sql_gen", description="SQL generation")]
        session = StubSession(
            agent_manager=FakeAgentManager(agents=agents),
            skill_manager=FakeSkillManager(skills=skills),
        )
        section = session._build_agents_skills_section("default")
        assert len(section.content) == 2
        types = [e.metadata["type"] for e in section.content]
        assert "agent" in types
        assert "skill" in types


class TestLimitationsSection:
    def test_draft_terms_reported(self):
        terms = [
            FakeGlossaryTerm(id="g1", name="foo", display_name="Foo", definition="", status="draft"),
            FakeGlossaryTerm(id="g2", name="bar", display_name="Bar", definition="", status="reviewed"),
        ]
        relational = FakeRelationalStore(glossary_terms=terms)
        session = StubSession(relational=relational)
        section = session._build_limitations_section("default")
        # Should report 1 draft term (the "default" domain filter is None, so all returned)
        assert len(section.content) == 1
        assert section.content[0].metadata["count"] == 1


class TestEditRouting:
    def test_glossary_edit_routes_to_relational(self):
        terms = [FakeGlossaryTerm(id="g1", name="revenue", display_name="Revenue", definition="Old def", domain="test")]
        relational = FakeRelationalStore(glossary_terms=terms)
        session = StubSession(relational=relational)
        result = session.update_handbook_entry("glossary", "glossary:g1", "definition", "New def")
        assert result is True

    def test_rule_edit_routes_to_learning_store(self):
        rules = [{"id": "r1", "summary": "Old summary", "confidence": 0.8, "domain": ""}]
        store = FakeLearningStore(rules=rules)
        session = StubSession(learning_store=store)
        result = session.update_handbook_entry("rules", "rule:r1", "summary", "New summary")
        assert result is True

    def test_entity_edit_routes_to_relational(self):
        entities = [("ent-1", "customer", "Customer", "CONCEPT", "ORG", 5)]
        relational = FakeRelationalStore(entities=entities)
        session = StubSession(relational=relational)
        result = session.update_handbook_entry("entities", "entity:ent-1", "name", "Client")
        assert result is True

    def test_unsupported_section_raises(self):
        session = StubSession()
        with pytest.raises(ValueError, match="does not support edits"):
            session.update_handbook_entry("patterns", "p:1", "query", "new")

    def test_unsupported_field_raises(self):
        terms = [FakeGlossaryTerm(id="g1", name="revenue", display_name="Revenue", definition="", domain="test")]
        relational = FakeRelationalStore(glossary_terms=terms)
        session = StubSession(relational=relational)
        with pytest.raises(ValueError, match="Cannot edit field"):
            session.update_handbook_entry("glossary", "glossary:g1", "nonexistent_field", "val")


class TestSectionFiltering:
    """Verify that domain-filtered handbook returns appropriate entries."""

    def test_glossary_filtered_by_domain(self):
        terms = [
            FakeGlossaryTerm(id="g1", name="rev", display_name="Rev", definition="", domain="sales"),
            FakeGlossaryTerm(id="g2", name="churn", display_name="Churn", definition="", domain="hr"),
        ]
        relational = FakeRelationalStore(glossary_terms=terms)
        session = StubSession(relational=relational)
        section = session._build_glossary_section("sales")
        assert len(section.content) == 1
        assert section.content[0].metadata["name"] == "rev"
