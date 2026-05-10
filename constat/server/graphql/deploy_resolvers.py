"""GraphQL resolvers for deployment diff, generate, and apply operations."""

from __future__ import annotations

import dataclasses
import json
import logging

import strawberry
import yaml
from strawberry.scalars import JSON

from constat.server.graphql.session_context import GqlInfo as Info
from constat.server.permissions import get_user_permissions

logger = logging.getLogger(__name__)


@strawberry.type
class DeployDiffType:
    source_path: str
    target_path: str
    generated_at: str
    sections: JSON  # serialized list[SectionDiff]
    summary: JSON  # serialized DiffSummary


@strawberry.type
class DeployApplyResultType:
    applied: int
    skipped: int
    errors: list[str]
    backup_path: str | None = None


def _require_admin(info: Info) -> None:
    """Raise if user is not a platform admin."""
    server_config = info.context.server_config
    user_id = info.context.user_id
    if not user_id:
        raise PermissionError("Authentication required")
    if server_config.auth_disabled:
        return
    perms = get_user_permissions(server_config, user_id)
    if not perms.is_admin:
        raise PermissionError("platform_admin role required")


def _diff_to_json(config_diff) -> dict:
    """Serialize ConfigDiff sections and summary to JSON-safe dicts."""
    sections = []
    for s in config_diff.sections:
        changes = []
        for c in s.changes:
            changes.append(dataclasses.asdict(c))
        sections.append({"section": s.section, "changes": changes})
    summary = dataclasses.asdict(config_diff.summary)
    return {"sections": sections, "summary": summary}


@strawberry.type
class Query:
    @strawberry.field
    async def deploy_diff(
        self, info: Info, source_path: str, target_path: str
    ) -> DeployDiffType:
        """Generate diff between two config directories. Requires platform_admin."""
        _require_admin(info)

        from constat.deploy.differ import ConfigDiffer

        differ = ConfigDiffer(source_path, target_path)
        config_diff = differ.diff()
        serialized = _diff_to_json(config_diff)

        return DeployDiffType(
            source_path=config_diff.source_path,
            target_path=config_diff.target_path,
            generated_at=config_diff.generated_at,
            sections=serialized["sections"],
            summary=serialized["summary"],
        )


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def deploy_generate(
        self, info: Info, source_path: str, target_path: str
    ) -> JSON:
        """Generate deployment script. Returns YAML string."""
        _require_admin(info)

        from constat.deploy.cli import _script_to_dict
        from constat.deploy.differ import ConfigDiffer

        differ = ConfigDiffer(source_path, target_path)
        script = differ.generate_script()
        data = _script_to_dict(script)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    @strawberry.mutation
    async def deploy_apply(
        self,
        info: Info,
        script: JSON,
        target_path: str,
        dry_run: bool = True,
        only: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> DeployApplyResultType:
        """Apply deployment script to target."""
        _require_admin(info)

        from constat.deploy.applier import DeployApplier
        from constat.deploy.cli import _dict_to_script

        # Parse the script from JSON (could be YAML string or dict)
        if isinstance(script, str):
            script_data = yaml.safe_load(script)
        else:
            script_data = script

        deploy_script = _dict_to_script(script_data)

        only_set = set(only) if only else None
        exclude_set = set(exclude) if exclude else None

        applier = DeployApplier(target_path, dry_run=dry_run)
        result = applier.apply(deploy_script, only=only_set, exclude=exclude_set)

        return DeployApplyResultType(
            applied=len(result["applied"]),
            skipped=len(result["skipped"]),
            errors=result["errors"],
            backup_path=result.get("backup_path"),
        )
