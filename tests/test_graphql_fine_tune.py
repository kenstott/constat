# Copyright (c) 2025 Kenneth Stott
# Canary: 1ca6e72b-30d4-4652-b03c-1fa5b5f391c7
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for GraphQL fine-tune resolvers (Phase 9)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_context(user_id="test-user"):
    from constat.server.graphql.session_context import GraphQLContext

    mock_sm = MagicMock()
    mock_server_config = MagicMock()
    mock_server_config.auth_disabled = True
    mock_server_config.data_dir = Path("/tmp/test-graphql-fine-tune")

    ctx = GraphQLContext(
        session_manager=mock_sm,
        server_config=mock_server_config,
        user_id=user_id,
        config=None,
    )
    mock_request = MagicMock()
    ctx.request = mock_request
    return ctx


def _make_mock_job(
    job_id="job-1",
    name="test-job",
    status="pending",
    provider="openai",
    base_model="gpt-4o",
):
    job = MagicMock()
    job.id = job_id
    job.name = name
    job.provider = provider
    job.base_model = base_model
    job.fine_tuned_model_id = None
    job.task_types = ["corrections"]
    job.domain = None
    job.status = status
    job.created = "2026-01-01T00:00:00"
    job.exemplar_count = 10
    job.metrics = None
    job.training_data_path = None
    return job


# ============================================================================
# Schema stitching tests
# ============================================================================


class TestFineTuneSchemaStitching:
    def _get_sdl(self):
        from constat.server.graphql import schema
        return schema.as_str()

    def test_fine_tune_jobs_query(self):
        assert "fineTuneJobs" in self._get_sdl()

    def test_fine_tune_job_query(self):
        assert "fineTuneJob(" in self._get_sdl()

    def test_fine_tune_providers_query(self):
        assert "fineTuneProviders" in self._get_sdl()

    def test_start_fine_tune_job_mutation(self):
        assert "startFineTuneJob" in self._get_sdl()

    def test_cancel_fine_tune_job_mutation(self):
        assert "cancelFineTuneJob" in self._get_sdl()

    def test_delete_fine_tune_job_mutation(self):
        assert "deleteFineTuneJob" in self._get_sdl()

    def test_recreate_fine_tune_job_mutation(self):
        assert "recreateFineTuneJob" in self._get_sdl()

    def test_fine_tune_job_type(self):
        assert "FineTuneJobType" in self._get_sdl()

    def test_fine_tune_provider_type(self):
        assert "FineTuneProviderType" in self._get_sdl()

    def test_start_fine_tune_input(self):
        assert "StartFineTuneInput" in self._get_sdl()


# ============================================================================
# Query resolver tests
# ============================================================================


class TestFineTuneJobsQuery:
    @pytest.mark.asyncio
    async def test_fine_tune_jobs_returns_list(self):
        from constat.server.graphql.fine_tune_resolvers import Query

        ctx = _make_context()
        mock_manager = MagicMock()
        mock_manager.registry.list.return_value = [_make_mock_job()]
        ctx.request.app.state.fine_tune_manager = mock_manager

        info = MagicMock()
        info.context = ctx
        result = await Query().fine_tune_jobs(info)

        assert len(result) == 1
        assert result[0].id == "job-1"
        assert result[0].name == "test-job"
        assert result[0].status == "pending"

    @pytest.mark.asyncio
    async def test_fine_tune_jobs_with_status_filter(self):
        from constat.server.graphql.fine_tune_resolvers import Query

        ctx = _make_context()
        mock_manager = MagicMock()
        mock_manager.registry.list.return_value = []
        ctx.request.app.state.fine_tune_manager = mock_manager

        info = MagicMock()
        info.context = ctx
        result = await Query().fine_tune_jobs(info, status="completed")

        mock_manager.registry.list.assert_called_once_with(status="completed", domain=None)
        assert result == []

    @pytest.mark.asyncio
    async def test_fine_tune_jobs_requires_auth(self):
        from constat.server.graphql.fine_tune_resolvers import Query

        ctx = _make_context(user_id=None)
        info = MagicMock()
        info.context = ctx

        with pytest.raises((ValueError, Exception)):
            await Query().fine_tune_jobs(info)

    @pytest.mark.asyncio
    async def test_fine_tune_jobs_no_manager_raises(self):
        from constat.server.graphql.fine_tune_resolvers import Query

        ctx = _make_context()
        ctx.request.app.state.fine_tune_manager = None

        info = MagicMock()
        info.context = ctx

        with pytest.raises(ValueError, match="not initialized"):
            await Query().fine_tune_jobs(info)


class TestFineTuneJobQuery:
    @pytest.mark.asyncio
    async def test_fine_tune_job_returns_single(self):
        from constat.server.graphql.fine_tune_resolvers import Query

        ctx = _make_context()
        mock_manager = MagicMock()
        mock_manager.check_status.return_value = _make_mock_job(job_id="abc123")
        ctx.request.app.state.fine_tune_manager = mock_manager

        info = MagicMock()
        info.context = ctx
        result = await Query().fine_tune_job(info, model_id="abc123")

        assert result.id == "abc123"

    @pytest.mark.asyncio
    async def test_fine_tune_job_not_found_raises(self):
        from constat.server.graphql.fine_tune_resolvers import Query

        ctx = _make_context()
        mock_manager = MagicMock()
        mock_manager.check_status.side_effect = KeyError("not found")
        ctx.request.app.state.fine_tune_manager = mock_manager

        info = MagicMock()
        info.context = ctx

        with pytest.raises(ValueError, match="not found"):
            await Query().fine_tune_job(info, model_id="missing")


class TestFineTuneProvidersQuery:
    @pytest.mark.asyncio
    async def test_fine_tune_providers_returns_list(self):
        from constat.server.graphql.fine_tune_resolvers import Query

        ctx = _make_context()
        info = MagicMock()
        info.context = ctx

        mock_providers = [
            {"name": "openai", "models": ["gpt-4o", "gpt-3.5-turbo"]},
            {"name": "anthropic", "models": ["claude-sonnet-4-6"]},
        ]

        with patch(
            "constat.learning.fine_tune_providers.get_available_providers",
            return_value=mock_providers,
        ):
            result = await Query().fine_tune_providers(info)

        assert len(result) == 2
        assert result[0].name == "openai"
        assert "gpt-4o" in result[0].models

    @pytest.mark.asyncio
    async def test_fine_tune_providers_requires_auth(self):
        from constat.server.graphql.fine_tune_resolvers import Query

        ctx = _make_context(user_id=None)
        info = MagicMock()
        info.context = ctx

        with pytest.raises((ValueError, Exception)):
            await Query().fine_tune_providers(info)


# ============================================================================
# Mutation resolver tests
# ============================================================================


class TestStartFineTuneJobMutation:
    @pytest.mark.asyncio
    async def test_start_fine_tune_job_success(self):
        from constat.server.graphql.fine_tune_resolvers import Mutation
        from constat.server.graphql.types import StartFineTuneInput

        ctx = _make_context()
        mock_manager = MagicMock()
        mock_manager.start_fine_tune.return_value = _make_mock_job(
            status="preparing", name="my-job"
        )
        ctx.request.app.state.fine_tune_manager = mock_manager

        info = MagicMock()
        info.context = ctx
        inp = StartFineTuneInput(
            name="my-job",
            provider="openai",
            base_model="gpt-4o",
            task_types=["corrections"],
        )
        result = await Mutation().start_fine_tune_job(info, input=inp)

        assert result.name == "my-job"
        mock_manager.start_fine_tune.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_fine_tune_job_requires_auth(self):
        from constat.server.graphql.fine_tune_resolvers import Mutation
        from constat.server.graphql.types import StartFineTuneInput

        ctx = _make_context(user_id=None)
        info = MagicMock()
        info.context = ctx

        inp = StartFineTuneInput(
            name="x", provider="openai", base_model="gpt-4o", task_types=[]
        )
        with pytest.raises((ValueError, Exception)):
            await Mutation().start_fine_tune_job(info, input=inp)


class TestCancelFineTuneJobMutation:
    @pytest.mark.asyncio
    async def test_cancel_fine_tune_job_success(self):
        from constat.server.graphql.fine_tune_resolvers import Mutation

        ctx = _make_context()
        mock_manager = MagicMock()
        mock_manager.check_status.return_value = _make_mock_job(status="cancelled")
        ctx.request.app.state.fine_tune_manager = mock_manager

        info = MagicMock()
        info.context = ctx
        result = await Mutation().cancel_fine_tune_job(info, model_id="job-1")

        mock_manager.cancel.assert_called_once_with("job-1")
        assert result.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_fine_tune_job_not_found(self):
        from constat.server.graphql.fine_tune_resolvers import Mutation

        ctx = _make_context()
        mock_manager = MagicMock()
        mock_manager.cancel.side_effect = KeyError("not found")
        ctx.request.app.state.fine_tune_manager = mock_manager

        info = MagicMock()
        info.context = ctx

        with pytest.raises(ValueError, match="not found"):
            await Mutation().cancel_fine_tune_job(info, model_id="bad-id")


class TestDeleteFineTuneJobMutation:
    @pytest.mark.asyncio
    async def test_delete_fine_tune_job_success(self):
        from constat.server.graphql.fine_tune_resolvers import Mutation

        ctx = _make_context()
        mock_manager = MagicMock()
        ctx.request.app.state.fine_tune_manager = mock_manager

        info = MagicMock()
        info.context = ctx
        result = await Mutation().delete_fine_tune_job(info, model_id="job-1")

        mock_manager.delete.assert_called_once_with("job-1")
        assert result.status == "deleted"
        assert result.name == "job-1"

    @pytest.mark.asyncio
    async def test_delete_fine_tune_job_not_found(self):
        from constat.server.graphql.fine_tune_resolvers import Mutation

        ctx = _make_context()
        mock_manager = MagicMock()
        mock_manager.delete.side_effect = KeyError("not found")
        ctx.request.app.state.fine_tune_manager = mock_manager

        info = MagicMock()
        info.context = ctx

        with pytest.raises(ValueError, match="not found"):
            await Mutation().delete_fine_tune_job(info, model_id="bad-id")

    @pytest.mark.asyncio
    async def test_delete_fine_tune_job_requires_auth(self):
        from constat.server.graphql.fine_tune_resolvers import Mutation

        ctx = _make_context(user_id=None)
        info = MagicMock()
        info.context = ctx

        with pytest.raises((ValueError, Exception)):
            await Mutation().delete_fine_tune_job(info, model_id="job-1")
