# Copyright (c) 2025 Kenneth Stott
# Canary: 2ef8ed4d-8a6b-4892-9701-1bc5b854f304
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for GraphQL domain resolvers."""

from __future__ import annotations

import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


def _make_context(user_id="test-user", config=None):
    from constat.server.graphql.session_context import GraphQLContext

    mock_sm = MagicMock()
    mock_server_config = MagicMock()
    mock_server_config.auth_disabled = True
    mock_server_config.base_url = "http://localhost:3000"
    mock_server_config.data_dir = Path("/tmp/test-graphql-domains")

    ctx = GraphQLContext(
        session_manager=mock_sm,
        server_config=mock_server_config,
        user_id=user_id,
        config=config,
    )
    mock_request = MagicMock()
    ctx.request = mock_request
    return ctx


def _make_mock_config(tmp_path: Path):
    """Create a mock Config with domain support."""
    config = MagicMock()
    config.config_dir = str(tmp_path / "system")
    config.databases = {"demo_db": MagicMock()}
    config.apis = {"demo_api": MagicMock()}
    config.documents = {}
    config.domains = {}
    config.domain_refs = []
    config.facts = {}

    def list_domains():
        return [
            {
                "filename": k,
                "name": d.name,
                "description": d.description,
                "tier": d.tier,
                "active": d.active,
                "owner": getattr(d, "owner", ""),
                "steward": getattr(d, "steward", ""),
            }
            for k, d in config.domains.items()
        ]

    config.list_domains = list_domains

    def load_domain(filename):
        return config.domains.get(filename)

    config.load_domain = load_domain

    def get_domain_tree():
        return [{"filename": k} for k in config.domains]

    config.get_domain_tree = get_domain_tree

    return config


def _make_domain_config(
    filename="test-domain",
    name="Test Domain",
    description="A test domain",
    tier="user",
    owner="test-user",
    source_path="",
    databases=None,
    apis=None,
    documents=None,
    domains=None,
):
    from constat.core.config import DomainConfig
    return DomainConfig(
        filename=filename,
        name=name,
        description=description,
        tier=tier,
        owner=owner,
        source_path=source_path,
        databases=databases or {},
        apis=apis or {},
        documents=documents or {},
        domains=domains or [],
    )


# ============================================================================
# Schema stitching tests
# ============================================================================


class TestDomainSchemaStitching:
    """Verify all domain operations appear in the SDL."""

    def _get_sdl(self):
        from constat.server.graphql import schema
        return schema.as_str()

    # Queries
    def test_domains_query(self):
        assert "domains:" in self._get_sdl() or "domains(" in self._get_sdl()

    def test_domain_tree_query(self):
        assert "domainTree:" in self._get_sdl() or "domainTree(" in self._get_sdl()

    def test_domain_query(self):
        assert "domain(" in self._get_sdl()

    def test_domain_content_query(self):
        assert "domainContent(" in self._get_sdl()

    def test_domain_skills_query(self):
        assert "domainSkills(" in self._get_sdl()

    def test_domain_agents_query(self):
        assert "domainAgents(" in self._get_sdl()

    def test_domain_rules_query(self):
        assert "domainRules(" in self._get_sdl()

    def test_domain_facts_query(self):
        assert "domainFacts(" in self._get_sdl()

    # Mutations
    def test_create_domain_mutation(self):
        assert "createDomain(" in self._get_sdl()

    def test_update_domain_mutation(self):
        assert "updateDomain(" in self._get_sdl()

    def test_delete_domain_mutation(self):
        assert "deleteDomain(" in self._get_sdl()

    def test_update_domain_content_mutation(self):
        assert "updateDomainContent(" in self._get_sdl()

    def test_promote_domain_mutation(self):
        assert "promoteDomain(" in self._get_sdl()

    def test_move_domain_skill_mutation(self):
        assert "moveDomainSkill(" in self._get_sdl()

    def test_move_domain_agent_mutation(self):
        assert "moveDomainAgent(" in self._get_sdl()

    def test_move_domain_rule_mutation(self):
        assert "moveDomainRule(" in self._get_sdl()

    def test_move_domain_source_mutation(self):
        assert "moveDomainSource(" in self._get_sdl()

    # Types
    def test_domain_info_type(self):
        sdl = self._get_sdl()
        assert "DomainInfoType" in sdl

    def test_domain_tree_node_type(self):
        sdl = self._get_sdl()
        assert "DomainTreeNodeType" in sdl

    def test_domain_detail_type(self):
        sdl = self._get_sdl()
        assert "DomainDetailType" in sdl


# ============================================================================
# Query resolver tests
# ============================================================================


class TestDomainsQuery:
    @pytest.mark.asyncio
    async def test_domains_returns_user_domain(self):
        from constat.server.graphql import schema

        config = _make_mock_config(Path("/tmp/test-gql-domains"))
        ctx = _make_context(config=config)

        result = await schema.execute(
            "{ domains { domains { filename name tier } } }",
            context_value=ctx,
        )
        assert result.errors is None
        # _ensure_user_domain_config always creates user domain entry
        domains = result.data["domains"]["domains"]
        user_domains = [d for d in domains if d["tier"] == "user"]
        assert len(user_domains) >= 1

    @pytest.mark.asyncio
    async def test_domains_with_data(self):
        from constat.server.graphql import schema

        config = _make_mock_config(Path("/tmp/test-gql-domains"))
        dc = _make_domain_config(filename="sales", name="Sales", tier="system")
        config.domains["sales"] = dc

        ctx = _make_context(config=config)

        result = await schema.execute(
            "{ domains { domains { filename name tier active } } }",
            context_value=ctx,
        )
        assert result.errors is None
        domains = result.data["domains"]["domains"]
        sales = [d for d in domains if d["filename"] == "sales"]
        assert len(sales) == 1
        assert sales[0]["name"] == "Sales"
        assert sales[0]["tier"] == "system"
        assert sales[0]["active"] is True


class TestDomainDetailQuery:
    @pytest.mark.asyncio
    async def test_domain_detail(self):
        from constat.server.graphql import schema

        config = _make_mock_config(Path("/tmp/test-gql-domains"))
        dc = _make_domain_config(
            filename="sales", name="Sales", databases={"demo_db": {}},
        )
        config.domains["sales"] = dc

        ctx = _make_context(config=config)

        result = await schema.execute(
            '{ domain(filename: "sales") { filename name databases apis documents } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["domain"]["filename"] == "sales"
        assert result.data["domain"]["databases"] == ["demo_db"]

    @pytest.mark.asyncio
    async def test_domain_not_found(self):
        from constat.server.graphql import schema

        config = _make_mock_config(Path("/tmp/test-gql-domains"))
        ctx = _make_context(config=config)

        result = await schema.execute(
            '{ domain(filename: "nonexistent") { filename } }',
            context_value=ctx,
        )
        assert result.errors is not None
        assert "not found" in result.errors[0].message.lower()


class TestDomainContentQuery:
    @pytest.mark.asyncio
    async def test_domain_content(self, tmp_path):
        from constat.server.graphql import schema

        domain_dir = tmp_path / "domain"
        domain_dir.mkdir()
        config_file = domain_dir / "config.yaml"
        content = yaml.dump({"name": "Sales", "description": "Sales domain"})
        config_file.write_text(content)

        config = _make_mock_config(tmp_path)
        dc = _make_domain_config(
            filename="sales", name="Sales", source_path=str(config_file),
        )
        config.domains["sales"] = dc

        ctx = _make_context(config=config)

        result = await schema.execute(
            '{ domainContent(filename: "sales") { content filename path } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["domainContent"]["filename"] == "sales"
        assert "Sales" in result.data["domainContent"]["content"]

    @pytest.mark.asyncio
    async def test_root_content(self, tmp_path):
        from constat.server.graphql import schema

        system_dir = tmp_path / "system"
        system_dir.mkdir()
        root_config = system_dir / "config.yaml"
        root_config.write_text(yaml.dump({"databases": {}}))

        config = _make_mock_config(tmp_path)

        ctx = _make_context(config=config)

        result = await schema.execute(
            '{ domainContent(filename: "root") { content filename } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["domainContent"]["filename"] == "root"


class TestDomainSkillsQuery:
    @pytest.mark.asyncio
    async def test_domain_skills_empty(self):
        from constat.server.graphql import schema

        config = _make_mock_config(Path("/tmp/test-gql-domains"))
        dc = _make_domain_config(filename="sales", name="Sales")
        config.domains["sales"] = dc

        ctx = _make_context(config=config)

        mock_manager = MagicMock()
        mock_manager.get_domain_skills.return_value = []

        with patch(
            "constat.server.routes.skills.get_skill_manager",
            return_value=mock_manager,
        ):
            result = await schema.execute(
                '{ domainSkills(filename: "sales") { name description domain } }',
                context_value=ctx,
            )
        assert result.errors is None
        assert result.data["domainSkills"] == []


class TestDomainRulesQuery:
    @pytest.mark.asyncio
    async def test_domain_rules_empty(self):
        from constat.server.graphql import schema

        config = _make_mock_config(Path("/tmp/test-gql-domains"))
        dc = _make_domain_config(filename="sales", name="Sales")
        config.domains["sales"] = dc

        ctx = _make_context(config=config)

        mock_store = MagicMock()
        mock_store.list_rules.return_value = []

        with patch(
            "constat.storage.learnings.LearningStore",
            return_value=mock_store,
        ):
            result = await schema.execute(
                '{ domainRules(filename: "sales") { id summary category } }',
                context_value=ctx,
            )
        assert result.errors is None
        assert result.data["domainRules"] == []


class TestDomainFactsQuery:
    @pytest.mark.asyncio
    async def test_domain_facts_empty(self):
        from constat.server.graphql import schema

        config = _make_mock_config(Path("/tmp/test-gql-domains"))
        dc = _make_domain_config(filename="sales", name="Sales")
        config.domains["sales"] = dc

        ctx = _make_context(config=config)

        mock_store = MagicMock()
        mock_store.list_all_facts.return_value = {}

        with patch(
            "constat.storage.facts.FactStore",
            return_value=mock_store,
        ):
            result = await schema.execute(
                '{ domainFacts(filename: "sales") { name value domain } }',
                context_value=ctx,
            )
        assert result.errors is None
        assert result.data["domainFacts"] == []


# ============================================================================
# Mutation resolver tests
# ============================================================================


class TestCreateDomainMutation:
    @pytest.mark.asyncio
    async def test_create_domain(self, tmp_path):
        from constat.server.graphql import schema

        config = _make_mock_config(tmp_path)
        ctx = _make_context(config=config)

        user_dir = tmp_path / ".constat" / "test-user.vault" / "domains"
        with patch(
            "constat.server.graphql.domain_resolvers.user_vault_dir",
            return_value=tmp_path / ".constat" / "test-user.vault",
        ):
            result = await schema.execute(
                """
                mutation {
                    createDomain(input: { name: "My Domain", description: "A test" }) {
                        status filename name description
                    }
                }
                """,
                context_value=ctx,
            )

        assert result.errors is None
        data = result.data["createDomain"]
        assert data["status"] == "created"
        assert data["filename"] == "my-domain"
        assert data["name"] == "My Domain"

    @pytest.mark.asyncio
    async def test_create_domain_empty_name(self, tmp_path):
        from constat.server.graphql import schema

        config = _make_mock_config(tmp_path)
        ctx = _make_context(config=config)

        result = await schema.execute(
            """
            mutation {
                createDomain(input: { name: "" }) {
                    status filename
                }
            }
            """,
            context_value=ctx,
        )
        assert result.errors is not None
        assert "required" in result.errors[0].message.lower()

    @pytest.mark.asyncio
    async def test_create_domain_conflict(self, tmp_path):
        from constat.server.graphql import schema

        config = _make_mock_config(tmp_path)
        dc = _make_domain_config(filename="existing", name="Existing")
        config.domains["existing"] = dc
        ctx = _make_context(config=config)

        result = await schema.execute(
            """
            mutation {
                createDomain(input: { name: "Existing" }) {
                    status filename
                }
            }
            """,
            context_value=ctx,
        )
        assert result.errors is not None
        assert "already exists" in result.errors[0].message.lower()


class TestUpdateDomainMutation:
    @pytest.mark.asyncio
    async def test_update_domain(self, tmp_path):
        from constat.server.graphql import schema

        domain_dir = tmp_path / "domain"
        domain_dir.mkdir()
        config_file = domain_dir / "config.yaml"
        config_file.write_text(yaml.dump({"name": "Sales", "description": "Old"}))

        config = _make_mock_config(tmp_path)
        dc = _make_domain_config(
            filename="sales", name="Sales", source_path=str(config_file),
        )
        config.domains["sales"] = dc

        ctx = _make_context(config=config)

        result = await schema.execute(
            """
            mutation {
                updateDomain(filename: "sales", input: { description: "Updated" }) {
                    status filename
                }
            }
            """,
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["updateDomain"]["status"] == "updated"

        # Verify file was updated
        written = yaml.safe_load(config_file.read_text())
        assert written["description"] == "Updated"

    @pytest.mark.asyncio
    async def test_update_domain_not_found(self, tmp_path):
        from constat.server.graphql import schema

        config = _make_mock_config(tmp_path)
        ctx = _make_context(config=config)

        result = await schema.execute(
            """
            mutation {
                updateDomain(filename: "nonexistent", input: { description: "x" }) {
                    status filename
                }
            }
            """,
            context_value=ctx,
        )
        assert result.errors is not None
        assert "not found" in result.errors[0].message.lower()


class TestDeleteDomainMutation:
    @pytest.mark.asyncio
    async def test_delete_domain(self, tmp_path):
        from constat.server.graphql import schema

        domain_dir = tmp_path / "domain"
        domain_dir.mkdir()
        config_file = domain_dir / "config.yaml"
        config_file.write_text(yaml.dump({"name": "Sales"}))

        config = _make_mock_config(tmp_path)
        dc = _make_domain_config(
            filename="sales", name="Sales", source_path=str(config_file),
        )
        config.domains["sales"] = dc

        ctx = _make_context(config=config)

        result = await schema.execute(
            """
            mutation {
                deleteDomain(filename: "sales") {
                    status filename
                }
            }
            """,
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["deleteDomain"]["status"] == "deleted"
        assert not domain_dir.exists()

    @pytest.mark.asyncio
    async def test_delete_domain_referenced(self, tmp_path):
        from constat.server.graphql import schema

        domain_dir = tmp_path / "domain"
        domain_dir.mkdir()
        config_file = domain_dir / "config.yaml"
        config_file.write_text(yaml.dump({"name": "Sales"}))

        config = _make_mock_config(tmp_path)
        dc_child = _make_domain_config(
            filename="sales", name="Sales", source_path=str(config_file),
        )
        dc_parent = _make_domain_config(
            filename="parent", name="Parent", domains=["sales"],
        )
        config.domains["sales"] = dc_child
        config.domains["parent"] = dc_parent

        ctx = _make_context(config=config)

        result = await schema.execute(
            """
            mutation {
                deleteDomain(filename: "sales") {
                    status filename
                }
            }
            """,
            context_value=ctx,
        )
        assert result.errors is not None
        assert "composed by" in result.errors[0].message.lower()


class TestUpdateDomainContentMutation:
    @pytest.mark.asyncio
    async def test_update_domain_content(self, tmp_path):
        from constat.server.graphql import schema

        domain_dir = tmp_path / "domain"
        domain_dir.mkdir()
        config_file = domain_dir / "config.yaml"
        config_file.write_text(yaml.dump({"name": "Sales"}))

        config = _make_mock_config(tmp_path)
        dc = _make_domain_config(
            filename="sales", name="Sales", source_path=str(config_file),
        )
        config.domains["sales"] = dc

        ctx = _make_context(config=config)

        new_content = yaml.dump({"name": "Sales Updated", "description": "New"})
        result = await schema.execute(
            """
            mutation UpdateContent($content: String!) {
                updateDomainContent(filename: "sales", content: $content) {
                    status filename path
                }
            }
            """,
            context_value=ctx,
            variable_values={"content": new_content},
        )
        assert result.errors is None
        assert result.data["updateDomainContent"]["status"] == "saved"

        written = yaml.safe_load(config_file.read_text())
        assert written["name"] == "Sales Updated"

    @pytest.mark.asyncio
    async def test_update_domain_content_invalid_yaml(self, tmp_path):
        from constat.server.graphql import schema

        domain_dir = tmp_path / "domain"
        domain_dir.mkdir()
        config_file = domain_dir / "config.yaml"
        config_file.write_text(yaml.dump({"name": "Sales"}))

        config = _make_mock_config(tmp_path)
        dc = _make_domain_config(
            filename="sales", name="Sales", source_path=str(config_file),
        )
        config.domains["sales"] = dc

        ctx = _make_context(config=config)

        result = await schema.execute(
            """
            mutation {
                updateDomainContent(filename: "sales", content: ": invalid: yaml: [") {
                    status filename
                }
            }
            """,
            context_value=ctx,
        )
        assert result.errors is not None
        assert "invalid yaml" in result.errors[0].message.lower()


class TestMoveDomainSourceMutation:
    @pytest.mark.asyncio
    async def test_move_source(self, tmp_path):
        from constat.server.graphql import schema

        # Create two domain configs
        from_dir = tmp_path / "from_domain"
        from_dir.mkdir()
        from_config = from_dir / "config.yaml"
        from_config.write_text(yaml.safe_dump({
            "name": "From",
            "databases": {"demo_db": {"type": "sqlite"}},
            "apis": {},
            "documents": {},
        }))

        to_dir = tmp_path / "to_domain"
        to_dir.mkdir()
        to_config = to_dir / "config.yaml"
        to_config.write_text(yaml.safe_dump({
            "name": "To",
            "databases": {},
            "apis": {},
            "documents": {},
        }))

        config = _make_mock_config(tmp_path)
        config.domains["from-domain"] = _make_domain_config(
            filename="from-domain", name="From", source_path=str(from_config),
            databases={"demo_db": {"type": "sqlite"}},
        )
        config.domains["to-domain"] = _make_domain_config(
            filename="to-domain", name="To", source_path=str(to_config),
        )

        ctx = _make_context(config=config)

        result = await schema.execute(
            """
            mutation {
                moveDomainSource(input: {
                    sourceType: "databases"
                    sourceName: "demo_db"
                    fromDomain: "from-domain"
                    toDomain: "to-domain"
                }) {
                    status
                }
            }
            """,
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["moveDomainSource"]["status"] == "moved"

        # Verify files updated
        from_data = yaml.safe_load(from_config.read_text())
        to_data = yaml.safe_load(to_config.read_text())
        assert "demo_db" not in from_data.get("databases", {})
        assert "demo_db" in to_data.get("databases", {})

    @pytest.mark.asyncio
    async def test_move_source_same_domain(self, tmp_path):
        from constat.server.graphql import schema

        config = _make_mock_config(tmp_path)
        ctx = _make_context(config=config)

        result = await schema.execute(
            """
            mutation {
                moveDomainSource(input: {
                    sourceType: "databases"
                    sourceName: "db"
                    fromDomain: "same"
                    toDomain: "same"
                }) {
                    status
                }
            }
            """,
            context_value=ctx,
        )
        assert result.errors is not None
        assert "must differ" in result.errors[0].message.lower()
