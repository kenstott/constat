"""Tests for the deployment script generator."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from constat.deploy.sensitive import is_sensitive_path, mask_value, SENSITIVE_KEYS
from constat.deploy.script import Change, ConfigDiff, DeployScript, Operation, SectionDiff
from constat.deploy.differ import ConfigDiffer
from constat.deploy.applier import DeployApplier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f.read()) or {}


def _make_config_dirs(tmp: Path):
    """Create source and target config directories with known configs."""
    source = tmp / "source"
    target = tmp / "target"

    # Source root config
    _write_yaml(source / "config.yaml", {
        "llm": {"model": "claude-sonnet-4-20250514", "provider": "anthropic"},
        "databases": {
            "main": {"uri": "postgres://localhost/main", "description": "Main DB"},
            "analytics": {"uri": "postgres://localhost/analytics"},
        },
        "execution": {"timeout_seconds": 120},
    })

    # Target root config (different)
    _write_yaml(target / "config.yaml", {
        "llm": {"model": "claude-haiku-4-5-20251001", "provider": "anthropic"},
        "databases": {
            "main": {"uri": "postgres://localhost/main_old", "description": "Main database"},
        },
        "execution": {"timeout_seconds": 60},
        "legacy_setting": True,
    })

    # Source domain
    _write_yaml(source / "domains" / "sales" / "config.yaml", {
        "name": "Sales Analytics",
        "glossary": {"ARR": {"definition": "Annual Recurring Revenue"}},
        "golden_questions": [{"question": "What is total revenue?"}],
    })

    # Target domain (partially different)
    _write_yaml(target / "domains" / "sales" / "config.yaml", {
        "name": "Sales Analytics",
        "glossary": {"ARR": {"definition": "Annual Revenue"}},
    })

    # Source-only domain
    _write_yaml(source / "domains" / "logistics" / "config.yaml", {
        "name": "Logistics",
        "databases": {"warehouse": {"uri": "postgres://localhost/wh"}},
    })

    # Target-only domain
    _write_yaml(target / "domains" / "hr" / "config.yaml", {
        "name": "HR Reporting",
    })

    # Source permissions
    _write_yaml(source / "permissions.yaml", {
        "users": {
            "alice@co.com": {"persona": "admin"},
            "bob@co.com": {"persona": "viewer"},
        },
    })

    # Target permissions
    _write_yaml(target / "permissions.yaml", {
        "users": {
            "alice@co.com": {"persona": "viewer"},
        },
    })

    # Source skills
    skill_dir = source / "skills" / "quarterly-report"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Quarterly Report Skill")

    # Source learnings
    _write_yaml(source / "domains" / "sales" / "learnings.yaml", {
        "rules": {"join_on_id": {"text": "Always join on region_id"}},
    })

    return source, target


# ---------------------------------------------------------------------------
# Sensitive path detection
# ---------------------------------------------------------------------------

class TestSensitive:
    def test_sensitive_keys_detected(self):
        assert is_sensitive_path("databases.main.uri")
        assert is_sensitive_path("databases.main.password")
        assert is_sensitive_path("llm.api_key")
        assert is_sensitive_path("smtp_password")
        assert is_sensitive_path("some.nested.secret.value")

    def test_non_sensitive_paths(self):
        assert not is_sensitive_path("databases.main.description")
        assert not is_sensitive_path("llm.model")
        assert not is_sensitive_path("glossary.ARR.definition")
        assert not is_sensitive_path("execution.timeout_seconds")

    def test_mask_value(self):
        assert mask_value("my-secret-value") == "***"
        assert mask_value(12345) == "***"
        assert mask_value(None) == "***"

    def test_all_sensitive_keys_are_strings(self):
        for key in SENSITIVE_KEYS:
            assert isinstance(key, str)


# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------

class TestCategorization:
    def test_source_category(self):
        assert ConfigDiffer._categorize_path("databases.main.uri") == "source"
        assert ConfigDiffer._categorize_path("apis.rest.url") == "source"
        assert ConfigDiffer._categorize_path("documents.readme.path") == "source"

    def test_glossary_category(self):
        assert ConfigDiffer._categorize_path("glossary.ARR.definition") == "glossary"

    def test_relationship_category(self):
        assert ConfigDiffer._categorize_path("relationships.orders_to_customers") == "relationship"

    def test_rule_category(self):
        assert ConfigDiffer._categorize_path("learnings.join_on_id") == "rule"
        assert ConfigDiffer._categorize_path("rights.admin") == "rule"
        assert ConfigDiffer._categorize_path("facts.company_name") == "rule"

    def test_permission_category(self):
        assert ConfigDiffer._categorize_path("permissions.users.alice") == "permission"

    def test_skill_category(self):
        assert ConfigDiffer._categorize_path("skills.quarterly-report") == "skill"

    def test_agent_category(self):
        assert ConfigDiffer._categorize_path("agents.planner") == "agent"

    def test_test_category(self):
        assert ConfigDiffer._categorize_path("golden_questions.q1") == "test"

    def test_config_category(self):
        assert ConfigDiffer._categorize_path("llm.model") == "config"
        assert ConfigDiffer._categorize_path("execution.timeout_seconds") == "config"


# ---------------------------------------------------------------------------
# Deep diff accuracy
# ---------------------------------------------------------------------------

class TestDeepDiff:
    def test_added_keys(self):
        differ = ConfigDiffer("", "")
        changes = differ._deep_diff({"a": 1, "b": 2}, {"a": 1})
        assert len(changes) == 1
        assert changes[0].kind == "added"
        assert changes[0].path == "b"
        assert changes[0].source_value == 2

    def test_removed_keys(self):
        differ = ConfigDiffer("", "")
        changes = differ._deep_diff({"a": 1}, {"a": 1, "b": 2})
        assert len(changes) == 1
        assert changes[0].kind == "removed"
        assert changes[0].path == "b"
        assert changes[0].target_value == 2

    def test_modified_keys(self):
        differ = ConfigDiffer("", "")
        changes = differ._deep_diff({"a": 2}, {"a": 1})
        assert len(changes) == 1
        assert changes[0].kind == "modified"
        assert changes[0].source_value == 2
        assert changes[0].target_value == 1

    def test_nested_dicts(self):
        differ = ConfigDiffer("", "")
        source = {"llm": {"model": "sonnet", "provider": "anthropic"}}
        target = {"llm": {"model": "haiku", "provider": "anthropic"}}
        changes = differ._deep_diff(source, target)
        assert len(changes) == 1
        assert changes[0].path == "llm.model"
        assert changes[0].kind == "modified"

    def test_list_changes(self):
        differ = ConfigDiffer("", "")
        changes = differ._deep_diff({"items": [1, 2, 3]}, {"items": [1, 2]})
        assert len(changes) == 1
        assert changes[0].kind == "modified"
        assert changes[0].path == "items"

    def test_no_changes(self):
        differ = ConfigDiffer("", "")
        changes = differ._deep_diff({"a": 1, "b": "x"}, {"a": 1, "b": "x"})
        assert len(changes) == 0

    def test_null_deletion(self):
        differ = ConfigDiffer("", "")
        changes = differ._deep_diff({"a": None}, {"a": "value"})
        assert len(changes) == 1
        assert changes[0].kind == "removed"

    def test_sensitive_flag_set(self):
        differ = ConfigDiffer("", "")
        changes = differ._deep_diff(
            {"databases": {"main": {"uri": "new"}}},
            {"databases": {"main": {"uri": "old"}}},
        )
        assert len(changes) == 1
        assert changes[0].sensitive is True


# ---------------------------------------------------------------------------
# Full diff with config dirs
# ---------------------------------------------------------------------------

class TestConfigDiff:
    def test_full_diff(self, tmp_path):
        source, target = _make_config_dirs(tmp_path)
        differ = ConfigDiffer(str(source), str(target))
        result = differ.diff()

        assert isinstance(result, ConfigDiff)
        assert result.summary.total_changes > 0
        assert result.summary.added > 0
        assert result.summary.removed > 0
        assert "logistics" in result.summary.domains_added
        assert "hr" in result.summary.domains_removed

    def test_domain_added(self, tmp_path):
        source, target = _make_config_dirs(tmp_path)
        differ = ConfigDiffer(str(source), str(target))
        result = differ.diff()

        domain_sections = [s for s in result.sections if s.section == "domain:logistics"]
        assert len(domain_sections) == 1
        # All changes in the new domain should be "added"
        for change in domain_sections[0].changes:
            assert change.kind == "added"

    def test_domain_removed(self, tmp_path):
        source, target = _make_config_dirs(tmp_path)
        differ = ConfigDiffer(str(source), str(target))
        result = differ.diff()

        domain_sections = [s for s in result.sections if s.section == "domain:hr"]
        assert len(domain_sections) == 1
        for change in domain_sections[0].changes:
            assert change.kind == "removed"

    def test_sensitive_changes_counted(self, tmp_path):
        source, target = _make_config_dirs(tmp_path)
        differ = ConfigDiffer(str(source), str(target))
        result = differ.diff()
        assert result.summary.sensitive_changes > 0


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------

class TestScriptGeneration:
    def test_generate_script(self, tmp_path):
        source, target = _make_config_dirs(tmp_path)
        differ = ConfigDiffer(str(source), str(target))
        script = differ.generate_script()

        assert isinstance(script, DeployScript)
        assert len(script.operations) > 0

        # Should have create_domain for logistics
        create_ops = [o for o in script.operations if o.op == "create_domain"]
        assert any(o.domain == "logistics" for o in create_ops)

        # Should have delete_domain for hr
        delete_ops = [o for o in script.operations if o.op == "delete_domain"]
        assert any(o.domain == "hr" for o in delete_ops)

        # Should have copy_skill
        skill_ops = [o for o in script.operations if o.op == "copy_skill"]
        assert any(o.skill == "quarterly-report" for o in skill_ops)

    def test_script_has_set_and_delete(self, tmp_path):
        source, target = _make_config_dirs(tmp_path)
        differ = ConfigDiffer(str(source), str(target))
        script = differ.generate_script()

        set_ops = [o for o in script.operations if o.op == "set"]
        delete_ops = [o for o in script.operations if o.op == "delete"]

        assert len(set_ops) > 0
        assert len(delete_ops) > 0


# ---------------------------------------------------------------------------
# Apply engine
# ---------------------------------------------------------------------------

class TestApplyEngine:
    def test_apply_set(self, tmp_path):
        target = tmp_path / "target"
        _write_yaml(target / "config.yaml", {"llm": {"model": "haiku"}})

        applier = DeployApplier(str(target))
        script = DeployScript(
            source_path="", target_path=str(target),
            generated_at="2026-01-01T00:00:00Z",
            operations=[
                Operation(op="set", file="config.yaml", path="llm.model", value="sonnet"),
            ],
        )
        result = applier.apply(script)
        assert len(result["applied"]) == 1
        assert len(result["errors"]) == 0

        data = _read_yaml(target / "config.yaml")
        assert data["llm"]["model"] == "sonnet"

    def test_apply_delete(self, tmp_path):
        target = tmp_path / "target"
        _write_yaml(target / "config.yaml", {"llm": {"model": "haiku"}, "legacy": True})

        applier = DeployApplier(str(target))
        script = DeployScript(
            source_path="", target_path=str(target),
            generated_at="2026-01-01T00:00:00Z",
            operations=[
                Operation(op="delete", file="config.yaml", path="legacy"),
            ],
        )
        result = applier.apply(script)
        assert len(result["applied"]) == 1

        data = _read_yaml(target / "config.yaml")
        assert "legacy" not in data

    def test_dry_run_no_changes(self, tmp_path):
        target = tmp_path / "target"
        _write_yaml(target / "config.yaml", {"llm": {"model": "haiku"}})

        applier = DeployApplier(str(target), dry_run=True)
        script = DeployScript(
            source_path="", target_path=str(target),
            generated_at="2026-01-01T00:00:00Z",
            operations=[
                Operation(op="set", file="config.yaml", path="llm.model", value="sonnet"),
            ],
        )
        result = applier.apply(script)
        assert len(result["applied"]) == 1

        # File should NOT be changed
        data = _read_yaml(target / "config.yaml")
        assert data["llm"]["model"] == "haiku"

    def test_backup_created(self, tmp_path):
        target = tmp_path / "target"
        _write_yaml(target / "config.yaml", {"llm": {"model": "haiku"}})

        applier = DeployApplier(str(target))
        script = DeployScript(
            source_path="", target_path=str(target),
            generated_at="2026-01-01T00:00:00Z",
            operations=[
                Operation(op="set", file="config.yaml", path="llm.model", value="sonnet"),
            ],
        )
        result = applier.apply(script)
        backup_path = result["backup_path"]
        assert Path(backup_path).is_dir()

        # Backup should contain original config
        backup_data = _read_yaml(Path(backup_path) / "config.yaml")
        assert backup_data["llm"]["model"] == "haiku"

    def test_rollback(self, tmp_path):
        target = tmp_path / "target"
        _write_yaml(target / "config.yaml", {"llm": {"model": "haiku"}})

        applier = DeployApplier(str(target))

        # Create a backup manually
        backup_path = applier._backup()

        # Corrupt the config
        _write_yaml(target / "config.yaml", {"corrupted": True})

        # Rollback
        applier._rollback(backup_path)

        data = _read_yaml(target / "config.yaml")
        assert data["llm"]["model"] == "haiku"
        assert "corrupted" not in data

    def test_category_filter_only(self, tmp_path):
        target = tmp_path / "target"
        _write_yaml(target / "config.yaml", {"llm": {"model": "haiku"}})
        _write_yaml(target / "permissions.yaml", {"users": {}})

        applier = DeployApplier(str(target))
        script = DeployScript(
            source_path="", target_path=str(target),
            generated_at="2026-01-01T00:00:00Z",
            operations=[
                Operation(op="set", file="config.yaml", path="llm.model", value="sonnet", category="config"),
                Operation(op="set", file="permissions.yaml", path="users.alice", value="admin", category="permission"),
            ],
        )
        result = applier.apply(script, only={"config"})
        assert len(result["applied"]) == 1
        assert len(result["skipped"]) == 1

        # Only config should have changed
        data = _read_yaml(target / "config.yaml")
        assert data["llm"]["model"] == "sonnet"

        perm = _read_yaml(target / "permissions.yaml")
        assert "alice" not in perm.get("users", {})

    def test_category_filter_exclude(self, tmp_path):
        target = tmp_path / "target"
        _write_yaml(target / "config.yaml", {"llm": {"model": "haiku"}})
        _write_yaml(target / "permissions.yaml", {"users": {}})

        applier = DeployApplier(str(target))
        script = DeployScript(
            source_path="", target_path=str(target),
            generated_at="2026-01-01T00:00:00Z",
            operations=[
                Operation(op="set", file="config.yaml", path="llm.model", value="sonnet", category="config"),
                Operation(op="set", file="permissions.yaml", path="users.alice", value="admin", category="permission"),
            ],
        )
        result = applier.apply(script, exclude={"permission"})
        assert len(result["applied"]) == 1
        assert len(result["skipped"]) == 1

        data = _read_yaml(target / "config.yaml")
        assert data["llm"]["model"] == "sonnet"

    def test_create_domain(self, tmp_path):
        source = tmp_path / "source"
        target = tmp_path / "target"
        target.mkdir(parents=True)

        _write_yaml(source / "domains" / "new-domain" / "config.yaml", {
            "name": "New Domain",
        })
        _write_yaml(target / "config.yaml", {})

        applier = DeployApplier(str(target))
        script = DeployScript(
            source_path=str(source), target_path=str(target),
            generated_at="2026-01-01T00:00:00Z",
            operations=[
                Operation(op="create_domain", domain="new-domain",
                          source_dir="domains/new-domain/"),
            ],
        )
        result = applier.apply(script)
        assert len(result["applied"]) == 1

        domain_config = _read_yaml(target / "domains" / "new-domain" / "config.yaml")
        assert domain_config["name"] == "New Domain"

    def test_delete_domain(self, tmp_path):
        target = tmp_path / "target"
        _write_yaml(target / "config.yaml", {})
        _write_yaml(target / "domains" / "old-domain" / "config.yaml", {
            "name": "Old Domain",
        })

        applier = DeployApplier(str(target))
        script = DeployScript(
            source_path="", target_path=str(target),
            generated_at="2026-01-01T00:00:00Z",
            operations=[
                Operation(op="delete_domain", domain="old-domain"),
            ],
        )
        result = applier.apply(script)
        assert len(result["applied"]) == 1
        assert not (target / "domains" / "old-domain").exists()

    def test_idempotency(self, tmp_path):
        target = tmp_path / "target"
        _write_yaml(target / "config.yaml", {"llm": {"model": "haiku"}})

        script = DeployScript(
            source_path="", target_path=str(target),
            generated_at="2026-01-01T00:00:00Z",
            operations=[
                Operation(op="set", file="config.yaml", path="llm.model", value="sonnet"),
            ],
        )

        # Apply twice
        applier1 = DeployApplier(str(target))
        applier1.apply(script)

        applier2 = DeployApplier(str(target))
        result2 = applier2.apply(script)

        # Second apply should still succeed
        assert len(result2["errors"]) == 0

        data = _read_yaml(target / "config.yaml")
        assert data["llm"]["model"] == "sonnet"

    def test_set_creates_nested_path(self, tmp_path):
        target = tmp_path / "target"
        _write_yaml(target / "config.yaml", {})

        applier = DeployApplier(str(target))
        script = DeployScript(
            source_path="", target_path=str(target),
            generated_at="2026-01-01T00:00:00Z",
            operations=[
                Operation(op="set", file="config.yaml",
                          path="databases.new_db.description", value="A new database"),
            ],
        )
        result = applier.apply(script)
        assert len(result["applied"]) == 1

        data = _read_yaml(target / "config.yaml")
        assert data["databases"]["new_db"]["description"] == "A new database"

    def test_set_creates_file(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir(parents=True)

        applier = DeployApplier(str(target))
        script = DeployScript(
            source_path="", target_path=str(target),
            generated_at="2026-01-01T00:00:00Z",
            operations=[
                Operation(op="set", file="config.yaml", path="llm.model", value="sonnet"),
            ],
        )
        result = applier.apply(script)
        assert len(result["applied"]) == 1

        data = _read_yaml(target / "config.yaml")
        assert data["llm"]["model"] == "sonnet"


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_generate_apply_rediff_zero_changes(self, tmp_path):
        source, target = _make_config_dirs(tmp_path)

        # Generate script
        differ = ConfigDiffer(str(source), str(target))
        script = differ.generate_script()

        # Apply
        applier = DeployApplier(str(target))
        result = applier.apply(script)
        assert len(result["errors"]) == 0

        # Re-diff: should have zero changes for root and domains present in both
        # (domain create/delete ops physically copy/remove dirs, so those should match)
        differ2 = ConfigDiffer(str(source), str(target))
        diff2 = differ2.diff()

        # After apply, source and target should match
        assert diff2.summary.total_changes == 0, (
            f"Expected 0 changes after round-trip, got {diff2.summary.total_changes}: "
            f"{[(c.path, c.kind) for s in diff2.sections for c in s.changes]}"
        )


# ---------------------------------------------------------------------------
# Null deletion
# ---------------------------------------------------------------------------

class TestNullDeletion:
    def test_null_in_source_removes_from_target(self):
        differ = ConfigDiffer("", "")
        changes = differ._deep_diff(
            {"keep": "yes", "remove_me": None},
            {"keep": "yes", "remove_me": "old_value"},
        )
        assert len(changes) == 1
        assert changes[0].kind == "removed"
        assert changes[0].path == "remove_me"
