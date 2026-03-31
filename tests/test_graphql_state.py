# Copyright (c) 2025 Kenneth Stott
# Canary: 351dd23d-e2db-4e2e-abbc-67c4a4b45a71
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for GraphQL session state resolvers."""

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
    mock_server_config.data_dir = Path("/tmp/test-graphql-state")

    ctx = GraphQLContext(
        session_manager=mock_sm,
        server_config=mock_server_config,
        user_id=user_id,
    )
    mock_request = MagicMock()
    ctx.request = mock_request
    return ctx


def _make_managed(session_id="s1", user_id="test-user", status="idle"):
    from constat.server.models import SessionStatus

    managed = MagicMock()
    managed.session_id = session_id
    managed.user_id = user_id
    managed.status = SessionStatus(status)
    managed.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    managed.last_activity = datetime(2025, 1, 1, tzinfo=timezone.utc)
    managed.current_query = "SELECT 1"
    managed.active_domains = ["sales-analytics.yaml"]
    managed.session.session_id = "hist-s1"
    managed.session.datastore = MagicMock()
    managed.session.datastore.get_scratchpad.return_value = []
    managed.session.datastore.get_ddl.return_value = "CREATE TABLE t1 (id INT)"
    managed.session.datastore.get_session_meta.return_value = None
    managed.session.scratchpad = None
    managed.session.history = MagicMock()
    managed.session.config = MagicMock()
    managed.session.config.system_prompt = "You are a data analyst."
    managed.session.config.llm.provider = "anthropic"
    managed.session.config.apis = {}
    managed.session.fact_resolver = MagicMock()
    managed.session.fact_resolver.get_all_facts.return_value = {}
    managed.session.schema_manager = MagicMock()
    managed.session.api_schema_manager = MagicMock()
    managed.has_database = MagicMock(return_value=True)
    return managed


class TestStateSchemaStitching:
    """Verify all 16 state operations appear in the SDL."""

    def _get_sdl(self):
        from constat.server.graphql import schema
        return schema.as_str()

    def test_steps_query(self):
        assert "steps(" in self._get_sdl()

    def test_inference_codes_query(self):
        assert "inferenceCodes(" in self._get_sdl()

    def test_scratchpad_query(self):
        assert "scratchpad(" in self._get_sdl()

    def test_session_ddl_query(self):
        assert "sessionDdl(" in self._get_sdl()

    def test_execution_output_query(self):
        assert "executionOutput(" in self._get_sdl()

    def test_session_routing_query(self):
        assert "sessionRouting(" in self._get_sdl()

    def test_proof_tree_query(self):
        assert "proofTree(" in self._get_sdl()

    def test_proof_facts_query(self):
        assert "proofFacts(" in self._get_sdl()

    def test_messages_query(self):
        assert "messages(" in self._get_sdl()

    def test_objectives_query(self):
        assert "objectives(" in self._get_sdl()

    def test_prompt_context_query(self):
        assert "promptContext(" in self._get_sdl()

    def test_database_schema_query(self):
        assert "databaseSchema(" in self._get_sdl()

    def test_api_schema_query(self):
        assert "apiSchema(" in self._get_sdl()

    def test_save_proof_facts_mutation(self):
        assert "saveProofFacts(" in self._get_sdl()

    def test_save_messages_mutation(self):
        assert "saveMessages(" in self._get_sdl()

    def test_update_system_prompt_mutation(self):
        assert "updateSystemPrompt(" in self._get_sdl()


class TestStateResolvers:
    @pytest.mark.asyncio
    async def test_steps_empty(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.history.list_step_codes.return_value = []
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ steps(sessionId: "s1") { steps { stepNumber goal code } total } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["steps"]["total"] == 0
        assert result.data["steps"]["steps"] == []

    @pytest.mark.asyncio
    async def test_steps_with_data(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.history.list_step_codes.return_value = [
            {"step_number": 1, "goal": "Load data", "code": "SELECT 1", "prompt": "p1", "model": "gpt-4"},
        ]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ steps(sessionId: "s1") { steps { stepNumber goal code prompt model } total } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["steps"]
        assert data["total"] == 1
        assert data["steps"][0]["stepNumber"] == 1
        assert data["steps"][0]["goal"] == "Load data"
        assert data["steps"][0]["code"] == "SELECT 1"
        assert data["steps"][0]["prompt"] == "p1"
        assert data["steps"][0]["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_steps_disk_fallback(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = None

        with patch("constat.storage.history.SessionHistory") as mock_hist_cls:
            mock_hist = mock_hist_cls.return_value
            mock_hist.find_session_by_server_id.return_value = "disk-session-1"
            mock_hist.list_step_codes.return_value = [
                {"step_number": 1, "goal": "Disk step", "code": "SELECT 2"},
            ]
            result = await schema.execute(
                '{ steps(sessionId: "s1") { steps { stepNumber goal } total } }',
                context_value=ctx,
            )

        assert result.errors is None
        assert result.data["steps"]["total"] == 1
        assert result.data["steps"]["steps"][0]["goal"] == "Disk step"

    @pytest.mark.asyncio
    async def test_inference_codes_empty(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.history.list_inference_codes.return_value = []
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ inferenceCodes(sessionId: "s1") { inferences { inferenceId name } total } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["inferenceCodes"]["total"] == 0

    @pytest.mark.asyncio
    async def test_inference_codes_with_data(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.history.list_inference_codes.return_value = [
            {"inference_id": "i1", "name": "Revenue", "operation": "SUM", "code": "sum(x)", "attempt": 1},
        ]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ inferenceCodes(sessionId: "s1") { inferences { inferenceId name operation code attempt } total } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["inferenceCodes"]
        assert data["total"] == 1
        assert data["inferences"][0]["inferenceId"] == "i1"
        assert data["inferences"][0]["name"] == "Revenue"

    @pytest.mark.asyncio
    async def test_scratchpad_empty(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ scratchpad(sessionId: "s1") { entries { stepNumber goal } total } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["scratchpad"]["total"] == 0

    @pytest.mark.asyncio
    async def test_scratchpad_with_entries(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.datastore.get_scratchpad.return_value = [
            {
                "step_number": 1,
                "goal": "Analyze sales",
                "narrative": "Loaded sales data",
                "tables_created": ["sales_2024"],
                "code": "SELECT * FROM sales",
                "user_query": "Show sales",
                "objective_index": 0,
            },
        ]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ scratchpad(sessionId: "s1") { entries { stepNumber goal narrative tablesCreated code userQuery objectiveIndex } total } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["scratchpad"]
        assert data["total"] == 1
        assert data["entries"][0]["goal"] == "Analyze sales"
        assert data["entries"][0]["tablesCreated"] == ["sales_2024"]

    @pytest.mark.asyncio
    async def test_scratchpad_session_not_found(self):
        from constat.server.graphql import schema

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = None

        result = await schema.execute(
            '{ scratchpad(sessionId: "no-such") { total } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_session_ddl(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ sessionDdl(sessionId: "s1") }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["sessionDdl"] == "CREATE TABLE t1 (id INT)"

    @pytest.mark.asyncio
    async def test_execution_output(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ executionOutput(sessionId: "s1") { output suggestions currentQuery } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["executionOutput"]
        assert data["output"] == ""
        assert data["suggestions"] == []
        assert data["currentQuery"] == "SELECT 1"

    @pytest.mark.asyncio
    async def test_execution_output_with_scratchpad(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.scratchpad = MagicMock()
        managed.session.scratchpad.get_recent_context.return_value = "Step 1 output"
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ executionOutput(sessionId: "s1") { output } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["executionOutput"]["output"] == "Step 1 output"

    @pytest.mark.asyncio
    async def test_session_routing_no_router(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.router = None
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ sessionRouting(sessionId: "s1") }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["sessionRouting"] == {}

    @pytest.mark.asyncio
    async def test_proof_tree_empty(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ proofTree(sessionId: "s1") { facts { name source } } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["proofTree"]["facts"] == []

    @pytest.mark.asyncio
    async def test_proof_tree_with_facts(self):
        from constat.server.graphql import schema

        fact = MagicMock()
        fact.value = 42
        fact.source.value = "computed"
        fact.reasoning = "Sum of values"
        fact.dependencies = ["dep1"]
        managed = _make_managed()
        managed.session.fact_resolver.get_all_facts.return_value = {"revenue": fact}
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ proofTree(sessionId: "s1") { facts { name value source reasoning dependencies } } }',
            context_value=ctx,
        )
        assert result.errors is None
        facts = result.data["proofTree"]["facts"]
        assert len(facts) == 1
        assert facts[0]["name"] == "revenue"
        assert facts[0]["value"] == 42
        assert facts[0]["source"] == "computed"
        assert facts[0]["dependencies"] == ["dep1"]

    @pytest.mark.asyncio
    async def test_proof_facts(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.storage.history.SessionHistory") as mock_hist_cls:
            mock_hist = mock_hist_cls.return_value
            mock_hist.load_proof_facts_by_server_id.return_value = (
                [{"id": "f1", "name": "Revenue", "status": "verified", "value": 100}],
                "All facts verified",
            )
            result = await schema.execute(
                '{ proofFacts(sessionId: "s1") { facts { id name status value } summary } }',
                context_value=ctx,
            )

        assert result.errors is None
        data = result.data["proofFacts"]
        assert data["summary"] == "All facts verified"
        assert len(data["facts"]) == 1
        assert data["facts"][0]["name"] == "Revenue"
        assert data["facts"][0]["value"] == 100

    @pytest.mark.asyncio
    async def test_messages(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.storage.history.SessionHistory") as mock_hist_cls:
            mock_hist = mock_hist_cls.return_value
            mock_hist.load_messages_by_server_id.return_value = [
                {"id": "m1", "type": "assistant", "content": "Hello", "timestamp": "2025-01-01T00:00:00Z"},
            ]
            result = await schema.execute(
                '{ messages(sessionId: "s1") { messages { id type content timestamp } } }',
                context_value=ctx,
            )

        assert result.errors is None
        msgs = result.data["messages"]["messages"]
        assert len(msgs) == 1
        assert msgs[0]["id"] == "m1"
        assert msgs[0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_objectives_empty(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ objectives(sessionId: "s1") { type text } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert result.data["objectives"] == []

    @pytest.mark.asyncio
    async def test_objectives_with_data(self):
        import json

        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.datastore.get_session_meta.return_value = json.dumps([
            {"type": "question", "text": "What is revenue?", "ts": "2025-01-01"},
        ])
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ objectives(sessionId: "s1") { type text ts } }',
            context_value=ctx,
        )
        assert result.errors is None
        assert len(result.data["objectives"]) == 1
        assert result.data["objectives"][0]["type"] == "question"
        assert result.data["objectives"][0]["text"] == "What is revenue?"

    @pytest.mark.asyncio
    async def test_prompt_context(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session._get_system_prompt = MagicMock(return_value="You are a data analyst.")
        # No agent_manager or skill_manager
        del managed.session.agent_manager
        del managed.session.skill_manager
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ promptContext(sessionId: "s1") { systemPrompt activeAgent { name } activeSkills { name } } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["promptContext"]
        assert data["systemPrompt"] == "You are a data analyst."
        assert data["activeAgent"] is None
        assert data["activeSkills"] == []

    @pytest.mark.asyncio
    async def test_prompt_context_forbidden(self):
        from constat.server.graphql import schema

        managed = _make_managed(user_id="other-user")
        ctx = _make_context(user_id="test-user")
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ promptContext(sessionId: "s1") { systemPrompt } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_database_schema(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        table_meta = MagicMock()
        table_meta.name = "sales"
        table_meta.row_count = 1000
        table_meta.columns = [MagicMock(), MagicMock(), MagicMock()]
        managed.session.schema_manager.get_tables_for_db.return_value = [table_meta]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ databaseSchema(sessionId: "s1", dbName: "warehouse") { database tables { name rowCount columnCount } } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["databaseSchema"]
        assert data["database"] == "warehouse"
        assert len(data["tables"]) == 1
        assert data["tables"][0]["name"] == "sales"
        assert data["tables"][0]["rowCount"] == 1000
        assert data["tables"][0]["columnCount"] == 3

    @pytest.mark.asyncio
    async def test_database_schema_not_found(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.has_database.return_value = False
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ databaseSchema(sessionId: "s1", dbName: "nope") { database } }',
            context_value=ctx,
        )
        assert result.errors is not None

    @pytest.mark.asyncio
    async def test_api_schema(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        api_config = MagicMock()
        api_config.type = "rest"
        api_config.description = "Pet API"
        managed.session.config.apis = {"petstore": api_config}

        ep = MagicMock()
        ep.endpoint_name = "listPets"
        ep.api_type = "rest"
        ep.return_type = "[Pet]"
        ep.description = "List all pets"
        ep.http_method = "GET"
        ep.http_path = "/pets"
        field = MagicMock()
        field.name = "limit"
        field.type = "integer"
        field.description = "Max results"
        field.is_required = False
        ep.fields = [field]
        managed.session.api_schema_manager.get_api_schema.return_value = [ep]
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ apiSchema(sessionId: "s1", apiName: "petstore") { name type description endpoints { name kind fields { name type isRequired } } } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["apiSchema"]
        assert data["name"] == "petstore"
        assert data["type"] == "rest"
        assert len(data["endpoints"]) == 1
        assert data["endpoints"][0]["name"] == "listPets"
        assert data["endpoints"][0]["fields"][0]["name"] == "limit"

    @pytest.mark.asyncio
    async def test_api_schema_not_found(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        managed.session.config.apis = {}
        managed.session.config.load_domain.return_value = None
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            '{ apiSchema(sessionId: "s1", apiName: "nope") { name } }',
            context_value=ctx,
        )
        assert result.errors is not None


class TestStateMutations:
    @pytest.mark.asyncio
    async def test_save_proof_facts(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.storage.history.SessionHistory") as mock_hist_cls:
            result = await schema.execute(
                """mutation($facts: JSON!, $summary: String) {
                    saveProofFacts(sessionId: "s1", facts: $facts, summary: $summary) { status count }
                }""",
                variable_values={
                    "facts": [{"id": "f1", "name": "Revenue", "status": "verified"}],
                    "summary": "Done",
                },
                context_value=ctx,
            )

        assert result.errors is None
        data = result.data["saveProofFacts"]
        assert data["status"] == "saved"
        assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_save_messages(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        with patch("constat.storage.history.SessionHistory") as mock_hist_cls:
            result = await schema.execute(
                """mutation($messages: JSON!) {
                    saveMessages(sessionId: "s1", messages: $messages) { status count }
                }""",
                variable_values={
                    "messages": [{"id": "m1", "type": "assistant", "content": "Hi"}],
                },
                context_value=ctx,
            )

        assert result.errors is None
        data = result.data["saveMessages"]
        assert data["status"] == "saved"
        assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_update_system_prompt(self):
        from constat.server.graphql import schema

        managed = _make_managed()
        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { updateSystemPrompt(sessionId: "s1", systemPrompt: "New prompt") { status systemPrompt } }',
            context_value=ctx,
        )
        assert result.errors is None
        data = result.data["updateSystemPrompt"]
        assert data["status"] == "updated"
        assert data["systemPrompt"] == "New prompt"
        assert managed.session.config.system_prompt == "New prompt"
        assert managed.session_prompt == "New prompt"

    @pytest.mark.asyncio
    async def test_update_system_prompt_forbidden(self):
        from constat.server.graphql import schema

        managed = _make_managed(user_id="other-user")
        ctx = _make_context(user_id="test-user")
        ctx.session_manager.get_session_or_none.return_value = managed

        result = await schema.execute(
            'mutation { updateSystemPrompt(sessionId: "s1", systemPrompt: "Hacked") { status } }',
            context_value=ctx,
        )
        assert result.errors is not None
