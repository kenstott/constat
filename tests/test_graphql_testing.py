# Copyright (c) 2025 Kenneth Stott
# Canary: 499f3b31-98b7-42d8-80d8-dca512d53f38
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for GraphQL testing (golden question) resolvers (Phase 9)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_context(user_id="test-user"):
    from constat.server.graphql.session_context import GraphQLContext

    mock_sm = MagicMock()
    mock_server_config = MagicMock()
    mock_server_config.auth_disabled = True
    mock_server_config.data_dir = Path("/tmp/test-graphql-testing")

    ctx = GraphQLContext(
        session_manager=mock_sm,
        server_config=mock_server_config,
        user_id=user_id,
        config=None,
    )
    mock_request = MagicMock()
    ctx.request = mock_request
    return ctx


def _make_mock_domain_config(name="sales", golden_questions=None):
    dc = MagicMock()
    dc.name = name
    dc.golden_questions = golden_questions or []
    dc.owner = None
    dc.steward = None
    dc.source_path = "/tmp/fake-domain.yaml"
    return dc


def _make_mock_managed(session_id="sess1", domain_configs=None):
    mock_config = MagicMock()
    mock_config.domains = domain_configs or {}

    def _load_domain(filename):
        return domain_configs.get(filename) if domain_configs else None

    mock_config.load_domain.side_effect = _load_domain

    mock_session = MagicMock()
    mock_session.config = mock_config
    mock_session.session_id = session_id

    mock_managed = MagicMock()
    mock_managed.user_id = "test-user"
    mock_managed.session = mock_session
    return mock_managed


# ============================================================================
# Schema stitching tests
# ============================================================================


class TestTestingSchemaStitching:
    def _get_sdl(self):
        from constat.server.graphql import schema
        return schema.as_str()

    def test_testable_domains_query(self):
        assert "testableDomains" in self._get_sdl()

    def test_golden_questions_query(self):
        assert "goldenQuestions" in self._get_sdl()

    def test_extract_expectations_mutation(self):
        assert "extractExpectations" in self._get_sdl()

    def test_create_golden_question_mutation(self):
        assert "createGoldenQuestion" in self._get_sdl()

    def test_update_golden_question_mutation(self):
        assert "updateGoldenQuestion" in self._get_sdl()

    def test_delete_golden_question_mutation(self):
        assert "deleteGoldenQuestion" in self._get_sdl()

    def test_move_golden_question_mutation(self):
        assert "moveGoldenQuestion" in self._get_sdl()

    def test_testable_domain_type(self):
        assert "TestableDomainType" in self._get_sdl()

    def test_golden_question_type(self):
        assert "GoldenQuestionType" in self._get_sdl()

    def test_golden_question_expectation_type(self):
        assert "GoldenQuestionExpectationType" in self._get_sdl()

    def test_create_golden_question_input(self):
        assert "CreateGoldenQuestionInput" in self._get_sdl()

    def test_move_golden_question_input(self):
        assert "MoveGoldenQuestionInput" in self._get_sdl()


# ============================================================================
# Query resolver tests
# ============================================================================


class TestTestableDomainQuery:
    @pytest.mark.asyncio
    async def test_testable_domains_returns_domains_with_questions(self):
        from constat.server.graphql.testing_resolvers import Query

        ctx = _make_context()

        mock_q = MagicMock()
        mock_q.tags = ["smoke", "unit"]

        dc = _make_mock_domain_config(name="sales", golden_questions=[{"question": "Q1"}])
        mock_managed = _make_mock_managed(
            domain_configs={"sales.yaml": dc}
        )
        ctx.session_manager.get_session.return_value = mock_managed

        with patch(
            "constat.testing.models.parse_golden_questions", return_value=[mock_q]
        ):
            info = MagicMock()
            info.context = ctx
            result = await Query().testable_domains(info, session_id="sess1")

        assert len(result) == 1
        assert result[0].filename == "sales.yaml"
        assert result[0].name == "sales"
        assert result[0].question_count == 1

    @pytest.mark.asyncio
    async def test_testable_domains_skips_domains_without_questions(self):
        from constat.server.graphql.testing_resolvers import Query

        ctx = _make_context()

        dc_with = _make_mock_domain_config(name="sales", golden_questions=[{"question": "Q1"}])
        dc_without = _make_mock_domain_config(name="hr", golden_questions=[])
        mock_managed = _make_mock_managed(
            domain_configs={"sales.yaml": dc_with, "hr.yaml": dc_without}
        )
        ctx.session_manager.get_session.return_value = mock_managed

        mock_q = MagicMock()
        mock_q.tags = []
        with patch("constat.testing.models.parse_golden_questions", return_value=[mock_q]):
            info = MagicMock()
            info.context = ctx
            result = await Query().testable_domains(info, session_id="sess1")

        # Only sales domain has questions, hr has none
        assert len(result) == 1
        assert result[0].filename == "sales.yaml"

    @pytest.mark.asyncio
    async def test_testable_domains_requires_auth(self):
        from constat.server.graphql.testing_resolvers import Query

        ctx = _make_context(user_id=None)
        info = MagicMock()
        info.context = ctx

        with pytest.raises((ValueError, Exception)):
            await Query().testable_domains(info, session_id="sess1")


class TestGoldenQuestionsQuery:
    @pytest.mark.asyncio
    async def test_golden_questions_returns_list(self):
        from constat.server.graphql.testing_resolvers import Query

        ctx = _make_context()
        dc = _make_mock_domain_config(
            name="sales",
            golden_questions=[
                {"question": "Q1", "tags": ["smoke"], "expect": {}},
                {"question": "Q2", "tags": [], "expect": {}},
            ],
        )
        mock_managed = _make_mock_managed(domain_configs={"sales.yaml": dc})
        ctx.session_manager.get_session.return_value = mock_managed

        mock_response = MagicMock()
        mock_response.question = "Q1"
        mock_response.tags = ["smoke"]
        mock_response.expect = MagicMock()
        mock_response.expect.model_dump.return_value = {}
        mock_response.objectives = []
        mock_response.system_prompt = None
        mock_response.index = 0

        with patch(
            "constat.server.graphql.testing_resolvers._gq_response_to_type"
        ) as mock_convert:
            from constat.server.graphql.types import GoldenQuestionType

            mock_convert.return_value = GoldenQuestionType(
                question="Q1",
                tags=["smoke"],
                expect={},
                objectives=[],
                system_prompt=None,
                index=0,
            )
            with patch(
                "constat.server.routes.testing._gq_to_response",
                return_value=mock_response,
            ):
                info = MagicMock()
                info.context = ctx
                result = await Query().golden_questions(
                    info, session_id="sess1", domain="sales.yaml"
                )

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_golden_questions_domain_not_found(self):
        from constat.server.graphql.testing_resolvers import Query

        ctx = _make_context()
        mock_managed = _make_mock_managed(domain_configs={})
        ctx.session_manager.get_session.return_value = mock_managed

        info = MagicMock()
        info.context = ctx

        with pytest.raises(ValueError, match="Domain not found"):
            await Query().golden_questions(
                info, session_id="sess1", domain="missing.yaml"
            )


# ============================================================================
# Mutation resolver tests
# ============================================================================


class TestCreateGoldenQuestionMutation:
    @pytest.mark.asyncio
    async def test_create_golden_question_success(self, tmp_path):
        from constat.server.graphql.testing_resolvers import Mutation
        from constat.server.graphql.types import CreateGoldenQuestionInput

        domain_file = tmp_path / "sales.yaml"
        domain_file.write_text("golden_questions: []\n")

        ctx = _make_context()
        dc = _make_mock_domain_config(name="sales", golden_questions=[])
        dc.source_path = str(domain_file)
        mock_managed = _make_mock_managed(domain_configs={"sales.yaml": dc})
        ctx.session_manager.get_session.return_value = mock_managed

        mock_response = MagicMock()
        mock_response.question = "What is revenue?"
        mock_response.tags = ["smoke"]
        mock_response.expect = MagicMock()
        mock_response.expect.model_dump.return_value = {"grounding": [], "terms": []}
        mock_response.objectives = ["Compute revenue"]
        mock_response.system_prompt = None
        mock_response.index = 0

        with patch(
            "constat.server.routes.testing._can_modify_domain", return_value=True
        ), patch(
            "constat.server.routes.testing._gq_to_response", return_value=mock_response
        ):
            info = MagicMock()
            info.context = ctx
            inp = CreateGoldenQuestionInput(
                question="What is revenue?",
                tags=["smoke"],
                expect={"grounding": [], "terms": []},
                objectives=["Compute revenue"],
            )
            result = await Mutation().create_golden_question(
                info, session_id="sess1", domain="sales.yaml", input=inp
            )

        assert result.question == "What is revenue?"

    @pytest.mark.asyncio
    async def test_create_golden_question_requires_auth(self):
        from constat.server.graphql.testing_resolvers import Mutation
        from constat.server.graphql.types import CreateGoldenQuestionInput

        ctx = _make_context(user_id=None)
        info = MagicMock()
        info.context = ctx

        with pytest.raises((ValueError, Exception)):
            await Mutation().create_golden_question(
                info,
                session_id="sess1",
                domain="sales.yaml",
                input=CreateGoldenQuestionInput(
                    question="q", tags=[], expect={}
                ),
            )


class TestUpdateGoldenQuestionMutation:
    @pytest.mark.asyncio
    async def test_update_golden_question_success(self, tmp_path):
        from constat.server.graphql.testing_resolvers import Mutation
        from constat.server.graphql.types import UpdateGoldenQuestionInput

        domain_file = tmp_path / "sales.yaml"
        import yaml
        domain_file.write_text(
            yaml.dump({"golden_questions": [{"question": "Old Q", "tags": [], "expect": {}}]})
        )

        ctx = _make_context()
        dc = _make_mock_domain_config(name="sales", golden_questions=[{"question": "Old Q"}])
        dc.source_path = str(domain_file)
        mock_managed = _make_mock_managed(domain_configs={"sales.yaml": dc})
        ctx.session_manager.get_session.return_value = mock_managed

        mock_response = MagicMock()
        mock_response.question = "New Q"
        mock_response.tags = []
        mock_response.expect = MagicMock()
        mock_response.expect.model_dump.return_value = {}
        mock_response.objectives = []
        mock_response.system_prompt = None
        mock_response.index = 0

        with patch(
            "constat.server.routes.testing._can_modify_domain", return_value=True
        ), patch(
            "constat.server.routes.testing._gq_to_response", return_value=mock_response
        ):
            info = MagicMock()
            info.context = ctx
            inp = UpdateGoldenQuestionInput(
                question="New Q",
                tags=[],
                expect={},
            )
            result = await Mutation().update_golden_question(
                info, session_id="sess1", domain="sales.yaml", index=0, input=inp
            )

        assert result.question == "New Q"

    @pytest.mark.asyncio
    async def test_update_golden_question_out_of_range(self, tmp_path):
        from constat.server.graphql.testing_resolvers import Mutation
        from constat.server.graphql.types import UpdateGoldenQuestionInput

        domain_file = tmp_path / "sales.yaml"
        import yaml
        domain_file.write_text(yaml.dump({"golden_questions": []}))

        ctx = _make_context()
        dc = _make_mock_domain_config(name="sales", golden_questions=[])
        dc.source_path = str(domain_file)
        mock_managed = _make_mock_managed(domain_configs={"sales.yaml": dc})
        ctx.session_manager.get_session.return_value = mock_managed

        with patch(
            "constat.server.routes.testing._can_modify_domain", return_value=True
        ):
            info = MagicMock()
            info.context = ctx

            with pytest.raises(ValueError, match="out of range"):
                await Mutation().update_golden_question(
                    info,
                    session_id="sess1",
                    domain="sales.yaml",
                    index=99,
                    input=UpdateGoldenQuestionInput(question="q", tags=[], expect={}),
                )


class TestDeleteGoldenQuestionMutation:
    @pytest.mark.asyncio
    async def test_delete_golden_question_success(self, tmp_path):
        from constat.server.graphql.testing_resolvers import Mutation

        domain_file = tmp_path / "sales.yaml"
        import yaml
        domain_file.write_text(
            yaml.dump({"golden_questions": [{"question": "Q1"}, {"question": "Q2"}]})
        )

        ctx = _make_context()
        dc = _make_mock_domain_config(
            name="sales",
            golden_questions=[{"question": "Q1"}, {"question": "Q2"}],
        )
        dc.source_path = str(domain_file)
        mock_managed = _make_mock_managed(domain_configs={"sales.yaml": dc})
        ctx.session_manager.get_session.return_value = mock_managed

        with patch(
            "constat.server.routes.testing._can_modify_domain", return_value=True
        ):
            info = MagicMock()
            info.context = ctx
            result = await Mutation().delete_golden_question(
                info, session_id="sess1", domain="sales.yaml", index=0
            )

        assert result.status == "deleted"

    @pytest.mark.asyncio
    async def test_delete_golden_question_out_of_range(self, tmp_path):
        from constat.server.graphql.testing_resolvers import Mutation

        domain_file = tmp_path / "sales.yaml"
        import yaml
        domain_file.write_text(yaml.dump({"golden_questions": []}))

        ctx = _make_context()
        dc = _make_mock_domain_config(name="sales", golden_questions=[])
        dc.source_path = str(domain_file)
        mock_managed = _make_mock_managed(domain_configs={"sales.yaml": dc})
        ctx.session_manager.get_session.return_value = mock_managed

        with patch(
            "constat.server.routes.testing._can_modify_domain", return_value=True
        ):
            info = MagicMock()
            info.context = ctx

            with pytest.raises(ValueError, match="out of range"):
                await Mutation().delete_golden_question(
                    info, session_id="sess1", domain="sales.yaml", index=5
                )

    @pytest.mark.asyncio
    async def test_delete_golden_question_requires_auth(self):
        from constat.server.graphql.testing_resolvers import Mutation

        ctx = _make_context(user_id=None)
        info = MagicMock()
        info.context = ctx

        with pytest.raises((ValueError, Exception)):
            await Mutation().delete_golden_question(
                info, session_id="sess1", domain="sales.yaml", index=0
            )


class TestMoveGoldenQuestionMutation:
    @pytest.mark.asyncio
    async def test_move_golden_question_validate_only(self, tmp_path):
        from constat.server.graphql.testing_resolvers import Mutation
        from constat.server.graphql.types import MoveGoldenQuestionInput

        src_file = tmp_path / "sales.yaml"
        tgt_file = tmp_path / "hr.yaml"
        import yaml
        src_file.write_text(
            yaml.dump({"golden_questions": [{"question": "Q1", "grounding": []}]})
        )
        tgt_file.write_text(yaml.dump({"golden_questions": []}))

        ctx = _make_context()
        src_dc = _make_mock_domain_config(
            name="sales", golden_questions=[{"question": "Q1", "grounding": []}]
        )
        src_dc.source_path = str(src_file)
        tgt_dc = _make_mock_domain_config(name="hr", golden_questions=[])
        tgt_dc.source_path = str(tgt_file)
        mock_managed = _make_mock_managed(
            domain_configs={"sales.yaml": src_dc, "hr.yaml": tgt_dc}
        )
        ctx.session_manager.get_session.return_value = mock_managed

        mock_response = MagicMock()
        mock_response.question = "Q1"
        mock_response.tags = []
        mock_response.expect = MagicMock()
        mock_response.expect.model_dump.return_value = {}
        mock_response.objectives = []
        mock_response.system_prompt = None
        mock_response.index = 0
        mock_response.warnings = []

        with patch(
            "constat.server.routes.testing._can_modify_domain", return_value=True
        ), patch(
            "constat.server.routes.testing._gq_to_response", return_value=mock_response
        ), patch(
            "constat.core.resource_validation.extract_resources_from_grounding",
            return_value=[],
        ), patch(
            "constat.core.resource_validation.validate_resource_compatibility",
            return_value=[],
        ):
            info = MagicMock()
            info.context = ctx
            inp = MoveGoldenQuestionInput(
                target_domain="hr.yaml", validate_only=True
            )
            result = await Mutation().move_golden_question(
                info,
                session_id="sess1",
                domain="sales.yaml",
                index=0,
                input=inp,
            )

        assert result.question == "Q1"

    @pytest.mark.asyncio
    async def test_move_golden_question_requires_auth(self):
        from constat.server.graphql.testing_resolvers import Mutation
        from constat.server.graphql.types import MoveGoldenQuestionInput

        ctx = _make_context(user_id=None)
        info = MagicMock()
        info.context = ctx

        with pytest.raises((ValueError, Exception)):
            await Mutation().move_golden_question(
                info,
                session_id="sess1",
                domain="sales.yaml",
                index=0,
                input=MoveGoldenQuestionInput(target_domain="hr.yaml"),
            )
