"""Tests for deploy GraphQL resolvers."""

from __future__ import annotations
import os
import tempfile

import pytest
import yaml

from constat.deploy.differ import ConfigDiffer
from constat.server.graphql.deploy_resolvers import (
    DeployApplyResultType,
    DeployDiffType,
    Mutation,
    Query,
    _diff_to_json,
    _require_admin,
)


def _make_config_dirs(source_data: dict, target_data: dict):
    """Create temp source and target config directories with config.yaml."""
    source_dir = tempfile.mkdtemp()
    target_dir = tempfile.mkdtemp()
    with open(os.path.join(source_dir, "config.yaml"), "w") as f:
        yaml.dump(source_data, f)
    with open(os.path.join(target_dir, "config.yaml"), "w") as f:
        yaml.dump(target_data, f)
    return source_dir, target_dir


class _FakePerms:
    is_admin = True


class _FakeNonAdminPerms:
    is_admin = False


class _FakeServerConfig:
    auth_disabled = False
    permissions = None

    def __init__(self, admin: bool = True):
        self._admin = admin


class _FakeContext:
    def __init__(self, admin: bool = True):
        self.user_id = "test-user"
        self.server_config = _FakeServerConfig(admin=admin)


class _FakeInfo:
    def __init__(self, admin: bool = True):
        self.context = _FakeContext(admin=admin)


# --- Tests ---


def test_deploy_diff_with_temp_dirs():
    """Test deploy_diff produces valid diff with temp config dirs."""
    source_data = {"llm": {"model": "claude-sonnet"}, "databases": {"db1": {"uri": "pg://"}}}
    target_data = {"llm": {"model": "claude-haiku"}}
    source_dir, target_dir = _make_config_dirs(source_data, target_data)

    differ = ConfigDiffer(source_dir, target_dir)
    config_diff = differ.diff()
    serialized = _diff_to_json(config_diff)

    assert isinstance(serialized["sections"], list)
    assert len(serialized["sections"]) > 0
    assert serialized["summary"]["total_changes"] > 0
    # llm.model is modified, databases.db1 is added
    assert serialized["summary"]["modified"] >= 1
    assert serialized["summary"]["added"] >= 1


def test_deploy_generate_returns_valid_yaml():
    """Test deploy_generate returns valid YAML string."""
    source_data = {"llm": {"model": "claude-sonnet"}, "glossary": {"ARR": "Annual Recurring Revenue"}}
    target_data = {"llm": {"model": "claude-haiku"}}
    source_dir, target_dir = _make_config_dirs(source_data, target_data)

    from constat.deploy.cli import _script_to_dict

    differ = ConfigDiffer(source_dir, target_dir)
    script = differ.generate_script()
    data = _script_to_dict(script)
    yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)

    # Should be valid YAML
    parsed = yaml.safe_load(yaml_str)
    assert "operations" in parsed
    assert "source_path" in parsed
    assert len(parsed["operations"]) > 0


def test_deploy_apply_dry_run_makes_no_changes():
    """Test deploy_apply with dry_run=True makes no file changes."""
    source_data = {"llm": {"model": "claude-sonnet"}}
    target_data = {"llm": {"model": "claude-haiku"}}
    source_dir, target_dir = _make_config_dirs(source_data, target_data)

    differ = ConfigDiffer(source_dir, target_dir)
    script = differ.generate_script()

    from constat.deploy.applier import DeployApplier

    applier = DeployApplier(target_dir, dry_run=True)
    result = applier.apply(script)

    assert len(result["applied"]) > 0
    assert len(result["errors"]) == 0

    # Target should be unchanged
    with open(os.path.join(target_dir, "config.yaml")) as f:
        after = yaml.safe_load(f.read())
    assert after["llm"]["model"] == "claude-haiku"


def test_deploy_apply_with_actual_changes():
    """Test deploy_apply with dry_run=False writes changes."""
    source_data = {"llm": {"model": "claude-sonnet"}}
    target_data = {"llm": {"model": "claude-haiku"}}
    source_dir, target_dir = _make_config_dirs(source_data, target_data)

    differ = ConfigDiffer(source_dir, target_dir)
    script = differ.generate_script()

    from constat.deploy.applier import DeployApplier

    applier = DeployApplier(target_dir, dry_run=False)
    result = applier.apply(script)

    assert len(result["applied"]) > 0
    assert len(result["errors"]) == 0

    # Target should now match source
    with open(os.path.join(target_dir, "config.yaml")) as f:
        after = yaml.safe_load(f.read())
    assert after["llm"]["model"] == "claude-sonnet"


def test_deploy_types_importable():
    """Verify Query and Mutation types can be imported."""
    assert hasattr(Query, "deploy_diff")
    assert hasattr(Mutation, "deploy_generate")
    assert hasattr(Mutation, "deploy_apply")
