# Copyright (c) 2025 Kenneth Stott
# Canary: e2159858-84ee-460c-897b-f85caae56dea
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Unit tests for GraphQL learning, skill, and agent resolvers (Phase 8)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_context(user_id="test-user", config=None):
    from constat.server.graphql.session_context import GraphQLContext

    mock_sm = MagicMock()
    mock_server_config = MagicMock()
    mock_server_config.auth_disabled = True
    mock_server_config.data_dir = Path("/tmp/test-graphql-learnings")

    ctx = GraphQLContext(
        session_manager=mock_sm,
        server_config=mock_server_config,
        user_id=user_id,
        config=config,
    )
    mock_request = MagicMock()
    ctx.request = mock_request
    return ctx


def _make_mock_config():
    config = MagicMock()
    config.domains = {}
    config.llm = MagicMock()
    return config


# ============================================================================
# Schema stitching tests
# ============================================================================


class TestLearningSchemaStitching:
    """Verify all learning/skill/agent operations appear in the SDL."""

    def _get_sdl(self):
        from constat.server.graphql import schema
        return schema.as_str()

    # Queries
    def test_learnings_query(self):
        assert "learnings" in self._get_sdl()

    def test_skills_query(self):
        assert "skills" in self._get_sdl()

    def test_skill_query(self):
        assert "skill(" in self._get_sdl()

    def test_agents_query(self):
        assert "agents(" in self._get_sdl()

    # Mutations
    def test_compact_learnings_mutation(self):
        assert "compactLearnings" in self._get_sdl()

    def test_delete_learning_mutation(self):
        assert "deleteLearning" in self._get_sdl()

    def test_create_rule_mutation(self):
        assert "createRule" in self._get_sdl()

    def test_update_rule_mutation(self):
        assert "updateRule" in self._get_sdl()

    def test_delete_rule_mutation(self):
        assert "deleteRule" in self._get_sdl()

    def test_activate_agent_mutation(self):
        assert "activateAgent" in self._get_sdl()

    def test_set_active_skills_mutation(self):
        assert "setActiveSkills" in self._get_sdl()

    # Types
    def test_learning_info_type(self):
        sdl = self._get_sdl()
        assert "LearningInfoType" in sdl

    def test_rule_info_type(self):
        assert "RuleInfoType" in self._get_sdl()

    def test_skill_info_type(self):
        assert "SkillInfoType" in self._get_sdl()

    def test_compaction_result_type(self):
        assert "CompactionResultType" in self._get_sdl()

    def test_agent_info_type(self):
        assert "AgentInfoType" in self._get_sdl()

    def test_set_agent_result_type(self):
        assert "SetAgentResultType" in self._get_sdl()

    def test_skills_list_type(self):
        assert "SkillsListType" in self._get_sdl()


# ============================================================================
# Type field tests
# ============================================================================


class TestLearningTypes:
    def _get_sdl(self):
        from constat.server.graphql import schema
        return schema.as_str()

    def test_learning_info_type_fields(self):
        sdl = self._get_sdl()
        assert "id:" in sdl
        assert "content:" in sdl
        assert "category:" in sdl
        assert "appliedCount:" in sdl
        assert "createdAt:" in sdl

    def test_rule_info_type_fields(self):
        sdl = self._get_sdl()
        assert "summary:" in sdl
        assert "confidence:" in sdl
        assert "sourceCount:" in sdl
        assert "tags:" in sdl

    def test_skill_info_type_fields(self):
        sdl = self._get_sdl()
        assert "isActive:" in sdl
        assert "domain:" in sdl
        assert "source:" in sdl

    def test_compaction_result_type_fields(self):
        sdl = self._get_sdl()
        assert "status:" in sdl
        assert "rulesCreated:" in sdl
        assert "learningsArchived:" in sdl


# ============================================================================
# Query resolver tests
# ============================================================================


class TestLearningsQuery:
    @pytest.mark.asyncio
    async def test_learnings_returns_list(self):
        from constat.server.graphql.learning_resolvers import Query

        mock_config = _make_mock_config()
        ctx = _make_context(config=mock_config)

        mock_store = MagicMock()
        mock_store.list_raw_learnings.return_value = [
            {
                "id": "l1",
                "correction": "Use metric tons not short tons",
                "category": "user_correction",
                "source": "explicit_command",
                "applied_count": 0,
            }
        ]
        mock_store.list_rules.return_value = []

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx
            result = await Query().learnings(info)

        assert len(result.learnings) == 1
        assert result.learnings[0].id == "l1"
        assert result.learnings[0].content == "Use metric tons not short tons"
        assert result.rules == []

    @pytest.mark.asyncio
    async def test_learnings_includes_rules(self):
        from constat.server.graphql.learning_resolvers import Query

        mock_config = _make_mock_config()
        ctx = _make_context(config=mock_config)

        mock_store = MagicMock()
        mock_store.list_raw_learnings.return_value = []
        mock_store.list_rules.return_value = [
            {
                "id": "r1",
                "summary": "Always use metric units",
                "category": "user_correction",
                "confidence": 0.9,
                "source_learnings": ["l1", "l2"],
                "tags": ["units"],
            }
        ]

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx
            result = await Query().learnings(info)

        assert len(result.rules) == 1
        assert result.rules[0].id == "r1"
        assert result.rules[0].source_count == 2

    @pytest.mark.asyncio
    async def test_learnings_requires_auth(self):
        from constat.server.graphql.learning_resolvers import Query

        ctx = _make_context(user_id=None)
        info = MagicMock()
        info.context = ctx

        with pytest.raises((ValueError, Exception)):
            await Query().learnings(info)


class TestSkillsQuery:
    @pytest.mark.asyncio
    async def test_skills_returns_list(self):
        from constat.server.graphql.learning_resolvers import Query

        ctx = _make_context()

        mock_skill = MagicMock()
        mock_skill.name = "my-skill"
        mock_skill.description = "A test skill"
        mock_skill.prompt = "Do things"
        mock_skill.filename = "my-skill/SKILL.md"
        mock_skill.domain = ""
        mock_skill.source = "user"

        mock_manager = MagicMock()
        mock_manager.get_all_skills.return_value = [mock_skill]
        mock_manager.active_skills = {"my-skill"}
        mock_manager.skills_dir = Path("/tmp/skills")

        with patch("constat.server.graphql.learning_resolvers.get_skill_manager", return_value=mock_manager):
            info = MagicMock()
            info.context = ctx
            result = await Query().skills(info)

        assert len(result.skills) == 1
        assert result.skills[0].name == "my-skill"
        assert result.skills[0].is_active is True
        assert "my-skill" in result.active_skills

    @pytest.mark.asyncio
    async def test_skill_content_returned(self):
        from constat.server.graphql.learning_resolvers import Query

        ctx = _make_context()

        mock_manager = MagicMock()
        mock_manager.get_skill_content.return_value = ("# SKILL content", "/path/to/SKILL.md")

        with patch("constat.server.graphql.learning_resolvers.get_skill_manager", return_value=mock_manager):
            info = MagicMock()
            info.context = ctx
            result = await Query().skill(info, name="my-skill")

        assert result.name == "my-skill"
        assert result.content == "# SKILL content"
        assert result.path == "/path/to/SKILL.md"

    @pytest.mark.asyncio
    async def test_skill_not_found_raises(self):
        from constat.server.graphql.learning_resolvers import Query

        ctx = _make_context()

        mock_manager = MagicMock()
        mock_manager.get_skill_content.return_value = None

        with patch("constat.server.graphql.learning_resolvers.get_skill_manager", return_value=mock_manager):
            info = MagicMock()
            info.context = ctx

            with pytest.raises(ValueError, match="Skill not found"):
                await Query().skill(info, name="missing-skill")


class TestAgentsQuery:
    @pytest.mark.asyncio
    async def test_agents_returns_list(self):
        from constat.server.graphql.learning_resolvers import Query

        ctx = _make_context()

        mock_agent = MagicMock()
        mock_agent.description = "An analyst agent"
        mock_agent.domain = "sales"
        mock_agent.source = "domain"

        mock_agent_manager = MagicMock()
        mock_agent_manager.active_agent_name = "analyst"
        mock_agent_manager.list_agents.return_value = ["analyst"]
        mock_agent_manager.get_agent.return_value = mock_agent

        mock_session = MagicMock()
        mock_session.agent_manager = mock_agent_manager

        mock_managed = MagicMock()
        mock_managed.user_id = "test-user"
        mock_managed.session = mock_session

        ctx.session_manager.get_session_or_none.return_value = mock_managed

        info = MagicMock()
        info.context = ctx
        result = await Query().agents(info, session_id="sess1")

        assert len(result) == 1
        assert result[0].name == "analyst"
        assert result[0].is_active is True

    @pytest.mark.asyncio
    async def test_agents_session_not_found(self):
        from constat.server.graphql.learning_resolvers import Query

        ctx = _make_context()
        ctx.session_manager.get_session_or_none.return_value = None

        info = MagicMock()
        info.context = ctx

        with pytest.raises(ValueError, match="Session not found"):
            await Query().agents(info, session_id="bad-session")


# ============================================================================
# Mutation resolver tests
# ============================================================================


class TestCompactLearningsMutation:
    @pytest.mark.asyncio
    async def test_compact_skipped_when_too_few(self):
        from constat.server.graphql.learning_resolvers import Mutation

        mock_config = _make_mock_config()
        ctx = _make_context(config=mock_config)

        mock_store = MagicMock()
        mock_store.get_stats.return_value = {"unpromoted": 1}

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx
            result = await Mutation().compact_learnings(info)

        assert result.status == "skipped"

    @pytest.mark.asyncio
    async def test_compact_success(self):
        from constat.server.graphql.learning_resolvers import Mutation

        mock_config = _make_mock_config()
        ctx = _make_context(config=mock_config)

        mock_store = MagicMock()
        mock_store.get_stats.return_value = {"unpromoted": 5}

        mock_result = MagicMock()
        mock_result.rules_created = 2
        mock_result.rules_strengthened = 1
        mock_result.rules_merged = 0
        mock_result.learnings_archived = 4
        mock_result.groups_found = 2
        mock_result.skipped_low_confidence = 0
        mock_result.errors = []

        mock_compactor = MagicMock()
        mock_compactor.compact.return_value = mock_result

        with (
            patch("constat.storage.learnings.LearningStore", return_value=mock_store),
            patch("constat.learning.compactor.LearningCompactor", return_value=mock_compactor),
            patch("constat.providers.TaskRouter"),
        ):
            info = MagicMock()
            info.context = ctx
            result = await Mutation().compact_learnings(info)

        assert result.status == "success"
        assert result.rules_created == 2
        assert result.learnings_archived == 4


class TestDeleteLearningMutation:
    @pytest.mark.asyncio
    async def test_delete_learning_success(self):
        from constat.server.graphql.learning_resolvers import Mutation

        ctx = _make_context()

        mock_store = MagicMock()
        mock_store.delete_learning.return_value = True

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx
            result = await Mutation().delete_learning(info, learning_id="l1")

        assert result.status == "deleted"
        assert result.name == "l1"

    @pytest.mark.asyncio
    async def test_delete_learning_not_found(self):
        from constat.server.graphql.learning_resolvers import Mutation

        ctx = _make_context()

        mock_store = MagicMock()
        mock_store.delete_learning.return_value = False

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx

            with pytest.raises(ValueError, match="Learning not found"):
                await Mutation().delete_learning(info, learning_id="bad-id")


class TestCreateRuleMutation:
    @pytest.mark.asyncio
    async def test_create_rule_success(self):
        from constat.server.graphql.learning_resolvers import Mutation
        from constat.server.graphql.types import CreateRuleInput

        ctx = _make_context()

        mock_store = MagicMock()
        mock_store.save_rule.return_value = "new-rule-id"

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx
            rule_input = CreateRuleInput(
                summary="Always format as JSON",
                category="user_correction",
                confidence=0.95,
                tags=["format"],
            )
            result = await Mutation().create_rule(info, input=rule_input)

        assert result.id == "new-rule-id"
        assert result.summary == "Always format as JSON"
        assert result.confidence == 0.95
        assert result.source_count == 0


class TestUpdateRuleMutation:
    @pytest.mark.asyncio
    async def test_update_rule_success(self):
        from constat.server.graphql.learning_resolvers import Mutation
        from constat.server.graphql.types import UpdateRuleInput

        ctx = _make_context()

        existing = {
            "id": "r1",
            "summary": "Old summary",
            "category": "user_correction",
            "confidence": 0.8,
            "source_learnings": [],
            "tags": [],
        }
        updated = {**existing, "summary": "New summary", "confidence": 0.9}

        mock_store = MagicMock()
        mock_store.list_rules.side_effect = [[existing], [updated]]
        mock_store.update_rule.return_value = True

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx
            update_input = UpdateRuleInput(summary="New summary", confidence=0.9)
            result = await Mutation().update_rule(info, rule_id="r1", input=update_input)

        assert result.id == "r1"
        assert result.summary == "New summary"

    @pytest.mark.asyncio
    async def test_update_rule_not_found(self):
        from constat.server.graphql.learning_resolvers import Mutation
        from constat.server.graphql.types import UpdateRuleInput

        ctx = _make_context()

        mock_store = MagicMock()
        mock_store.list_rules.return_value = []

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx

            with pytest.raises(ValueError, match="Rule not found"):
                await Mutation().update_rule(
                    info, rule_id="missing", input=UpdateRuleInput(summary="x")
                )


class TestDeleteRuleMutation:
    @pytest.mark.asyncio
    async def test_delete_rule_success(self):
        from constat.server.graphql.learning_resolvers import Mutation

        mock_config = _make_mock_config()
        ctx = _make_context(config=mock_config)

        mock_store = MagicMock()
        mock_store.list_rules.return_value = [
            {"id": "r1", "summary": "test rule", "domain": "user"}
        ]
        mock_store.delete_rule.return_value = True

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx
            result = await Mutation().delete_rule(info, rule_id="r1")

        assert result.status == "deleted"
        assert result.name == "r1"

    @pytest.mark.asyncio
    async def test_delete_rule_not_found(self):
        from constat.server.graphql.learning_resolvers import Mutation

        mock_config = _make_mock_config()
        ctx = _make_context(config=mock_config)

        mock_store = MagicMock()
        mock_store.list_rules.return_value = []
        mock_store.delete_rule.return_value = False

        with patch("constat.storage.learnings.LearningStore", return_value=mock_store):
            info = MagicMock()
            info.context = ctx

            with pytest.raises(ValueError, match="Rule not found"):
                await Mutation().delete_rule(info, rule_id="missing")


class TestActivateAgentMutation:
    @pytest.mark.asyncio
    async def test_activate_agent_success(self):
        from constat.server.graphql.learning_resolvers import Mutation

        ctx = _make_context()

        mock_agent_manager = MagicMock()
        mock_agent_manager.set_active_agent.return_value = True

        mock_session = MagicMock()
        mock_session.agent_manager = mock_agent_manager

        mock_managed = MagicMock()
        mock_managed.user_id = "test-user"
        mock_managed.session = mock_session

        ctx.session_manager.get_session_or_none.return_value = mock_managed

        info = MagicMock()
        info.context = ctx
        result = await Mutation().activate_agent(info, session_id="sess1", agent_name="analyst")

        assert result.success is True
        assert result.current_agent == "analyst"

    @pytest.mark.asyncio
    async def test_activate_agent_clear(self):
        from constat.server.graphql.learning_resolvers import Mutation

        ctx = _make_context()

        mock_agent_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.agent_manager = mock_agent_manager
        mock_managed = MagicMock()
        mock_managed.user_id = "test-user"
        mock_managed.session = mock_session

        ctx.session_manager.get_session_or_none.return_value = mock_managed

        info = MagicMock()
        info.context = ctx
        result = await Mutation().activate_agent(info, session_id="sess1", agent_name=None)

        assert result.success is True
        assert result.current_agent is None
        mock_agent_manager.set_active_agent.assert_called_with(None)

    @pytest.mark.asyncio
    async def test_activate_agent_not_found(self):
        from constat.server.graphql.learning_resolvers import Mutation

        ctx = _make_context()

        mock_agent_manager = MagicMock()
        mock_agent_manager.set_active_agent.return_value = False
        mock_agent_manager.list_agents.return_value = ["analyst"]

        mock_session = MagicMock()
        mock_session.agent_manager = mock_agent_manager
        mock_managed = MagicMock()
        mock_managed.user_id = "test-user"
        mock_managed.session = mock_session

        ctx.session_manager.get_session_or_none.return_value = mock_managed

        info = MagicMock()
        info.context = ctx

        with pytest.raises(ValueError, match="Agent not found"):
            await Mutation().activate_agent(info, session_id="sess1", agent_name="missing-agent")


class TestSetActiveSkillsMutation:
    @pytest.mark.asyncio
    async def test_set_active_skills_success(self):
        from constat.server.graphql.learning_resolvers import Mutation

        ctx = _make_context()

        mock_manager = MagicMock()
        mock_manager.set_active_skills.return_value = ["skill-a", "skill-b"]

        with patch("constat.server.graphql.learning_resolvers.get_skill_manager", return_value=mock_manager):
            info = MagicMock()
            info.context = ctx
            result = await Mutation().set_active_skills(info, skill_names=["skill-a", "skill-b"])

        assert result.status == "updated"
        assert "skill-a" in result.active_skills
        assert "skill-b" in result.active_skills
