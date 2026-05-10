# Copyright (c) 2025 Kenneth Stott
# Canary: 118fd41d-2022-4566-8650-139439c4ab1f
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for GraphQL data resolvers (tables, artifacts, facts, entities)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_context(user_id="test-user"):
    from constat.server.graphql.session_context import GraphQLContext

    mock_sm = MagicMock()
    mock_server_config = MagicMock()
    mock_server_config.auth_disabled = True
    mock_server_config.base_url = "http://localhost:3000"
    mock_server_config.data_dir = Path("/tmp/test-graphql-data")

    ctx = GraphQLContext(
        session_manager=mock_sm,
        server_config=mock_server_config,
        user_id=user_id,
    )
    mock_request = MagicMock()
    ctx.request = mock_request
    return ctx


def _make_managed(session_id="s1", user_id="test-user"):
    from constat.server.models import SessionStatus

    managed = MagicMock()
    managed.session_id = session_id
    managed.user_id = user_id
    managed.status = SessionStatus("idle")
    managed.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    managed.last_activity = datetime(2025, 1, 1, tzinfo=timezone.utc)
    managed.current_query = "SELECT 1"
    managed.active_domains = ["sales-analytics.yaml"]
    managed.session.session_id = "hist-s1"
    managed.session.datastore = MagicMock()
    managed.session.datastore.list_tables.return_value = []
    managed.session.datastore.list_artifacts.return_value = []
    managed.session.datastore.get_starred_tables.return_value = []
    managed.session.datastore.get_state.return_value = None
    managed.session.history = MagicMock()
    managed.session.config = MagicMock()
    managed.session.config.facts = {}
    managed.session.config.apis = {}
    managed.session.config.documents = {}
    managed.session.fact_resolver = MagicMock()
    managed.session.fact_resolver.get_all_facts.return_value = {}
    managed.session.schema_manager = None
    managed.session.doc_tools = None
    return managed


# ============================================================================
# Schema stitching tests
# ============================================================================


class TestDataSchemaStitching:
    """Verify all 20 data operations appear in the SDL."""

    def _get_sdl(self):
        from constat.server.graphql import schema
        return schema.as_str()

    # Queries
    def test_tables_query(self):
        assert "tables(" in self._get_sdl()

    def test_table_data_query(self):
        assert "tableData(" in self._get_sdl()

    def test_table_versions_query(self):
        assert "tableVersions(" in self._get_sdl()

    def test_table_version_data_query(self):
        assert "tableVersionData(" in self._get_sdl()

    def test_artifacts_query(self):
        assert "artifacts(" in self._get_sdl()

    def test_artifact_query(self):
        assert "artifact(" in self._get_sdl()

    def test_artifact_versions_query(self):
        assert "artifactVersions(" in self._get_sdl()

    def test_facts_query(self):
        assert "facts(" in self._get_sdl()

    def test_entities_query(self):
        assert "entities(" in self._get_sdl()

    # Mutations
    def test_delete_table_mutation(self):
        assert "deleteTable(" in self._get_sdl()

    def test_toggle_table_star_mutation(self):
        assert "toggleTableStar(" in self._get_sdl()

    def test_delete_artifact_mutation(self):
        assert "deleteArtifact(" in self._get_sdl()

    def test_toggle_artifact_star_mutation(self):
        assert "toggleArtifactStar(" in self._get_sdl()

    def test_add_fact_mutation(self):
        assert "addFact(" in self._get_sdl()

    def test_edit_fact_mutation(self):
        assert "editFact(" in self._get_sdl()

    def test_persist_fact_mutation(self):
        assert "persistFact(" in self._get_sdl()

    def test_forget_fact_mutation(self):
        assert "forgetFact(" in self._get_sdl()

    def test_move_fact_mutation(self):
        assert "moveFact(" in self._get_sdl()

    def test_add_entity_to_glossary_mutation(self):
        assert "addEntityToGlossary(" in self._get_sdl()


# ============================================================================
# Resolver unit tests
# ============================================================================


class TestTablesResolvers:
    @pytest.mark.asyncio
    async def test_tables_empty(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ tables(sessionId: "s1") { tables { name rowCount } total } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["tables"]["total"] == 0
        assert result.data["tables"]["tables"] == []

    @pytest.mark.asyncio
    async def test_tables_with_data(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.datastore.list_tables.return_value = [
            {
                "name": "sales",
                "row_count": 100,
                "step_number": 1,
                "columns": ["id", "amount"],
                "is_view": False,
                "is_published": True,
                "is_final_step": False,
                "version": 1,
                "version_count": 1,
            },
        ]
        managed.session.datastore.get_starred_tables.return_value = []
        managed.session.datastore.get_state.return_value = None
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ tables(sessionId: "s1") { tables { name rowCount stepNumber columns isStarred isView } total } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["tables"]
        assert data["total"] == 1
        assert data["tables"][0]["name"] == "sales"
        assert data["tables"][0]["rowCount"] == 100
        assert data["tables"][0]["columns"] == ["id", "amount"]
        assert data["tables"][0]["isStarred"] is True  # auto-starred (published)

    @pytest.mark.asyncio
    async def test_tables_skips_internal(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.datastore.list_tables.return_value = [
            {"name": "_facts", "row_count": 5, "step_number": 0, "columns": []},
            {"name": "execution_history", "row_count": 3, "step_number": 0, "columns": []},
            {"name": "sales", "row_count": 10, "step_number": 1, "columns": ["id"]},
        ]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ tables(sessionId: "s1") { tables { name } total } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["tables"]["total"] == 1
        assert result.data["tables"]["tables"][0]["name"] == "sales"

    @pytest.mark.asyncio
    async def test_tables_no_datastore(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.datastore = None
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ tables(sessionId: "s1") { total } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["tables"]["total"] == 0

    @pytest.mark.asyncio
    async def test_table_data(self):
        import pandas as pd
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.datastore.load_dataframe.return_value = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
        })
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ tableData(sessionId: "s1", tableName: "sales") { name columns data totalRows page pageSize hasMore } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["tableData"]
        assert data["name"] == "sales"
        assert data["columns"] == ["id", "name"]
        assert data["totalRows"] == 3
        assert data["page"] == 1
        assert data["hasMore"] is False
        assert len(data["data"]) == 3

    @pytest.mark.asyncio
    async def test_table_data_not_found(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.datastore.load_dataframe.return_value = None
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ tableData(sessionId: "s1", tableName: "nope") { name } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_table_versions(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.datastore.get_table_versions.return_value = [
            {"version": 2, "step_number": 3, "row_count": 50, "created_at": "2025-01-02"},
            {"version": 1, "step_number": 1, "row_count": 10, "created_at": "2025-01-01"},
        ]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ tableVersions(sessionId: "s1", tableName: "sales") { name currentVersion versions { version stepNumber rowCount } } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["tableVersions"]
        assert data["name"] == "sales"
        assert data["currentVersion"] == 2
        assert len(data["versions"]) == 2


class TestArtifactResolvers:
    @pytest.mark.asyncio
    async def test_artifacts_empty(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ artifacts(sessionId: "s1") { artifacts { id name } total } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["artifacts"]["total"] == 0

    @pytest.mark.asyncio
    async def test_artifacts_with_data(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        artifact_obj = MagicMock()
        artifact_obj.metadata = {"is_starred": True}
        managed.session.datastore.list_artifacts.return_value = [
            {"id": 1, "name": "chart1", "type": "plotly", "step_number": 2, "title": "Revenue Chart", "content_type": "text/html", "version": 1, "version_count": 1},
        ]
        managed.session.datastore.get_artifact_by_id.return_value = artifact_obj
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ artifacts(sessionId: "s1") { artifacts { id name artifactType isStarred } total } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["artifacts"]
        assert data["total"] == 1
        assert data["artifacts"][0]["name"] == "chart1"
        assert data["artifacts"][0]["isStarred"] is True

    @pytest.mark.asyncio
    async def test_artifacts_skips_internal(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        internal_artifact = MagicMock()
        internal_artifact.metadata = {"internal": True}
        managed.session.datastore.list_artifacts.return_value = [
            {"id": 1, "name": "internal_profile", "type": "text", "step_number": 0},
        ]
        managed.session.datastore.get_artifact_by_id.return_value = internal_artifact
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ artifacts(sessionId: "s1") { artifacts { name } total } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["artifacts"]["total"] == 0

    @pytest.mark.asyncio
    async def test_artifact_content(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        artifact = MagicMock()
        artifact.id = 1
        artifact.name = "chart1"
        artifact.artifact_type.value = "plotly"
        artifact.content = "<div>chart</div>"
        artifact.mime_type = "text/html"
        artifact.is_binary = False
        managed.session.datastore.get_artifact_by_id.return_value = artifact
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ artifact(sessionId: "s1", artifactId: 1) { id name artifactType content mimeType isBinary } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["artifact"]
        assert data["name"] == "chart1"
        assert data["content"] == "<div>chart</div>"
        assert data["isBinary"] is False

    @pytest.mark.asyncio
    async def test_artifact_not_found(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.datastore.get_artifact_by_id.return_value = None
        managed.session.datastore.list_tables.return_value = []
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ artifact(sessionId: "s1", artifactId: 999) { name } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_artifact_versions(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        artifact = MagicMock()
        artifact.name = "chart1"
        managed.session.datastore.get_artifact_by_id.return_value = artifact
        managed.session.datastore.get_artifact_versions.return_value = [
            {"id": 2, "version": 2, "step_number": 3, "attempt": 1, "created_at": "2025-01-02"},
            {"id": 1, "version": 1, "step_number": 1, "attempt": 1, "created_at": "2025-01-01"},
        ]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ artifactVersions(sessionId: "s1", artifactId: 1) { name currentVersion versions { id version stepNumber attempt } } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["artifactVersions"]
        assert data["name"] == "chart1"
        assert data["currentVersion"] == 2
        assert len(data["versions"]) == 2


class TestFactResolvers:
    @pytest.mark.asyncio
    async def test_facts_empty(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.storage.facts.FactStore") as mock_fs_cls:
            mock_fs = mock_fs_cls.return_value
            mock_fs.list_facts.return_value = {}
            mock_fs.list_all_facts.return_value = {}
            result = await schema.execute(
                '{ facts(sessionId: "s1") { facts { name value source } total } }',
                context_value=ctx,
            )

        assert result.errors is None
        assert result.data["facts"]["total"] == 0

    @pytest.mark.asyncio
    async def test_facts_with_config_and_session(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.config.facts = {"region": "US"}

        fact = MagicMock()
        fact.value = 42
        fact.source.value = "computed"
        fact.reasoning = "Sum of sales"
        fact.confidence = 0.95
        fact.role_id = None
        managed.session.fact_resolver.get_all_facts.return_value = {"revenue": fact}

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.storage.facts.FactStore") as mock_fs_cls:
            mock_fs = mock_fs_cls.return_value
            mock_fs.list_facts.return_value = {"revenue": 42}
            mock_fs.list_all_facts.return_value = {"revenue": {"domain": "sales"}}
            result = await schema.execute(
                '{ facts(sessionId: "s1") { facts { name value source isPersisted domain } total } }',
                context_value=ctx,
            )

        assert result.errors is None
        data = result.data["facts"]
        assert data["total"] == 2
        # Config fact
        config_fact = next(f for f in data["facts"] if f["name"] == "region")
        assert config_fact["source"] == "config"
        assert config_fact["isPersisted"] is False
        # Session fact
        session_fact = next(f for f in data["facts"] if f["name"] == "revenue")
        assert session_fact["value"] == 42
        assert session_fact["isPersisted"] is True
        assert session_fact["domain"] == "sales"

    @pytest.mark.asyncio
    async def test_add_fact(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation($value: JSON!) {
                addFact(sessionId: "s1", name: "region", value: $value) { status fact { name value source } }
            }""",
            variable_values={"value": "US"},
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["addFact"]
        assert data["status"] == "created"
        assert data["fact"]["name"] == "region"
        assert data["fact"]["value"] == "US"

    @pytest.mark.asyncio
    async def test_edit_fact(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        fact = MagicMock()
        fact.value = "old"
        managed.session.fact_resolver.get_all_facts.return_value = {"region": fact}
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation($value: JSON!) {
                editFact(sessionId: "s1", factName: "region", value: $value) { status fact { name value } }
            }""",
            variable_values={"value": "EU"},
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["editFact"]["status"] == "updated"
        assert result.data["editFact"]["fact"]["value"] == "EU"

    @pytest.mark.asyncio
    async def test_edit_fact_not_found(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.fact_resolver.get_all_facts.return_value = {}
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            """mutation($value: JSON!) {
                editFact(sessionId: "s1", factName: "nope", value: $value) { status }
            }""",
            variable_values={"value": "x"},
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_persist_fact(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        fact = MagicMock()
        managed.session.fact_resolver.get_all_facts.return_value = {"revenue": fact}
        managed.session.fact_resolver.persist_fact = MagicMock()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { persistFact(sessionId: "s1", factName: "revenue") { status } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["persistFact"]["status"] == "persisted"

    @pytest.mark.asyncio
    async def test_forget_fact(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        fact = MagicMock()
        managed.session.fact_resolver.get_all_facts.return_value = {"revenue": fact}
        managed.session.fact_resolver._cache = {"revenue": fact}
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.storage.facts.FactStore") as mock_fs_cls:
            mock_fs = mock_fs_cls.return_value
            mock_fs.delete_fact.return_value = True
            result = await schema.execute(
                'mutation { forgetFact(sessionId: "s1", factName: "revenue") { status } }',
                context_value=ctx,
            )

        assert result.errors is None
        assert result.data["forgetFact"]["status"] == "forgotten"


class TestDeleteAndStarMutations:
    @pytest.mark.asyncio
    async def test_delete_table(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.datastore.list_tables.return_value = [{"name": "sales"}]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { deleteTable(sessionId: "s1", tableName: "sales") { status name } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["deleteTable"]["status"] == "deleted"
        assert result.data["deleteTable"]["name"] == "sales"
        managed.session.datastore.drop_table.assert_called_once_with("sales")

    @pytest.mark.asyncio
    async def test_delete_table_not_found(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.datastore.list_tables.return_value = []
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { deleteTable(sessionId: "s1", tableName: "nope") { status } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_toggle_table_star(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.datastore.list_tables.return_value = [
            {"name": "sales", "row_count": 10, "is_published": False, "is_final_step": False},
        ]
        managed.session.datastore.get_starred_tables.return_value = []
        managed.session.datastore.get_state.return_value = None
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { toggleTableStar(sessionId: "s1", tableName: "sales") { name isStarred } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["toggleTableStar"]
        assert data["name"] == "sales"
        # Was not starred (not published, not final step), so toggled to True
        assert data["isStarred"] is True

    @pytest.mark.asyncio
    async def test_delete_artifact(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        artifact = MagicMock()
        artifact.name = "chart1"
        managed.session.datastore.get_artifact_by_id.return_value = artifact
        managed.session.datastore.delete_artifact.return_value = True
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { deleteArtifact(sessionId: "s1", artifactId: 1) { status name } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["deleteArtifact"]["status"] == "deleted"
        assert result.data["deleteArtifact"]["name"] == "chart1"

    @pytest.mark.asyncio
    async def test_toggle_artifact_star(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        artifact = MagicMock()
        artifact.name = "chart1"
        artifact.metadata = {"is_starred": False}
        managed.session.datastore.get_artifact_by_id.return_value = artifact
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { toggleArtifactStar(sessionId: "s1", artifactId: 1) { name isStarred } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["toggleArtifactStar"]
        assert data["name"] == "chart1"
        assert data["isStarred"] is True


class TestEntityResolvers:
    @pytest.mark.asyncio
    async def test_entities_empty(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.config = MagicMock()
        managed.session.config.apis = {}
        managed.session.config.documents = {}
        managed.session.config.load_domain.return_value = None
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ entities(sessionId: "s1") { entities { id name type } total } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["entities"]["total"] == 0

    @pytest.mark.asyncio
    async def test_entities_session_not_found(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = None

        result = await schema.execute(
            '{ entities(sessionId: "nope") { total } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_add_entity_to_glossary(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.add_to_glossary = MagicMock()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { addEntityToGlossary(sessionId: "s1", entityId: "e1") { status entityId note } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["addEntityToGlossary"]
        assert data["status"] == "added"
        assert data["entityId"] == "e1"
        assert data["note"] is None
