"""Config diff engine: compares two config directories and produces structured diffs."""

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from constat.deploy.script import (
    Change,
    ConfigDiff,
    DeployScript,
    DiffSummary,
    Operation,
    SectionDiff,
)
from constat.deploy.sensitive import is_sensitive_path, mask_value


class ConfigDiffer:
    """Produces a structured diff between two config directories."""

    def __init__(self, source_path: str, target_path: str):
        self.source_path = source_path
        self.target_path = target_path

    def diff(self) -> ConfigDiff:
        """Compare source and target, return structured diff."""
        source = self._load_config_dir(self.source_path)
        target = self._load_config_dir(self.target_path)

        sections: list[SectionDiff] = []
        domains_added: list[str] = []
        domains_removed: list[str] = []

        # 1. Diff root config.yaml
        root_changes = self._deep_diff(
            source.get("root", {}),
            target.get("root", {}),
        )
        if root_changes:
            sections.append(SectionDiff(section="root", changes=root_changes))

        # 2. Diff each domain
        source_domains = set(source.get("domains", {}).keys())
        target_domains = set(target.get("domains", {}).keys())

        for domain in sorted(source_domains | target_domains):
            if domain in source_domains and domain not in target_domains:
                # New domain in source (to be added)
                domain_changes = self._flatten_as_added(
                    source["domains"][domain], ""
                )
                if domain_changes:
                    sections.append(
                        SectionDiff(
                            section=f"domain:{domain}", changes=domain_changes
                        )
                    )
                domains_added.append(domain)
            elif domain not in source_domains and domain in target_domains:
                # Domain only in target (to be removed)
                domain_changes = self._flatten_as_removed(
                    target["domains"][domain], ""
                )
                if domain_changes:
                    sections.append(
                        SectionDiff(
                            section=f"domain:{domain}", changes=domain_changes
                        )
                    )
                domains_removed.append(domain)
            else:
                # Domain in both
                domain_changes = self._deep_diff(
                    source["domains"][domain],
                    target["domains"][domain],
                )
                if domain_changes:
                    sections.append(
                        SectionDiff(
                            section=f"domain:{domain}", changes=domain_changes
                        )
                    )

        # 3. Diff permissions.yaml
        perm_changes = self._deep_diff(
            source.get("permissions", {}),
            target.get("permissions", {}),
        )
        if perm_changes:
            sections.append(SectionDiff(section="permissions", changes=perm_changes))

        # 4. Diff skills
        source_skills = set(source.get("skills", {}).keys())
        target_skills = set(target.get("skills", {}).keys())

        for skill in sorted(source_skills | target_skills):
            if skill in source_skills and skill not in target_skills:
                sections.append(
                    SectionDiff(
                        section=f"skills:{skill}",
                        changes=[
                            Change(
                                path=f"skills.{skill}",
                                kind="added",
                                source_value="(skill directory)",
                                category="skill",
                            )
                        ],
                    )
                )
            elif skill not in source_skills and skill in target_skills:
                sections.append(
                    SectionDiff(
                        section=f"skills:{skill}",
                        changes=[
                            Change(
                                path=f"skills.{skill}",
                                kind="removed",
                                target_value="(skill directory)",
                                category="skill",
                            )
                        ],
                    )
                )

        # 5. Diff agents
        agent_changes = self._deep_diff(
            source.get("agents", {}),
            target.get("agents", {}),
            prefix="agents",
        )
        if agent_changes:
            sections.append(SectionDiff(section="agents", changes=agent_changes))

        # 6. Diff learnings per domain
        for domain in sorted(source_domains & target_domains):
            src_learnings = source.get("learnings", {}).get(domain, {})
            tgt_learnings = target.get("learnings", {}).get(domain, {})
            learning_changes = self._deep_diff(
                src_learnings, tgt_learnings
            )
            if learning_changes:
                sections.append(
                    SectionDiff(
                        section=f"learnings:{domain}", changes=learning_changes
                    )
                )

        # New domain learnings
        for domain in sorted(source_domains - target_domains):
            src_learnings = source.get("learnings", {}).get(domain, {})
            if src_learnings:
                learning_changes = self._flatten_as_added(
                    src_learnings, ""
                )
                if learning_changes:
                    sections.append(
                        SectionDiff(
                            section=f"learnings:{domain}",
                            changes=learning_changes,
                        )
                    )

        # 7. Compute summary
        summary = self._compute_summary(sections, domains_added, domains_removed)

        return ConfigDiff(
            source_path=self.source_path,
            target_path=self.target_path,
            generated_at=datetime.now(timezone.utc).isoformat(),
            sections=sections,
            summary=summary,
        )

    def _load_config_dir(self, path: str) -> dict:
        """Load all config files from a directory into a structured dict."""
        base = Path(path)
        result: dict[str, Any] = {
            "root": {},
            "domains": {},
            "permissions": {},
            "skills": {},
            "agents": {},
            "learnings": {},
        }

        # Load root config.yaml
        root_config = base / "config.yaml"
        if root_config.exists():
            result["root"] = self._load_yaml(root_config)

        # Load domains
        domains_dir = base / "domains"
        if domains_dir.is_dir():
            for entry in sorted(domains_dir.iterdir()):
                if entry.is_dir() and (entry / "config.yaml").exists():
                    domain_name = entry.name
                    result["domains"][domain_name] = self._load_yaml(
                        entry / "config.yaml"
                    )
                    # Load learnings
                    learnings_file = entry / "learnings.yaml"
                    if learnings_file.exists():
                        result["learnings"][domain_name] = self._load_yaml(
                            learnings_file
                        )

        # Load permissions.yaml
        permissions_file = base / "permissions.yaml"
        if permissions_file.exists():
            result["permissions"] = self._load_yaml(permissions_file)

        # Load skills
        skills_dir = base / "skills"
        if skills_dir.is_dir():
            for entry in sorted(skills_dir.iterdir()):
                if entry.is_dir():
                    result["skills"][entry.name] = True  # presence marker

        # Load agents
        agents_dir = base / "agents"
        if agents_dir.is_dir():
            for entry in sorted(agents_dir.iterdir()):
                if entry.suffix in (".yaml", ".yml"):
                    agent_name = entry.stem
                    result["agents"][agent_name] = self._load_yaml(entry)

        return result

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        """Load a YAML file, returning its data as a dict."""
        with open(path) as f:
            data = yaml.safe_load(f.read())
        return data if isinstance(data, dict) else {}

    def _deep_diff(
        self, source: Any, target: Any, path: str = "", prefix: str = ""
    ) -> list[Change]:
        """Recursive structural diff between source and target."""
        changes: list[Change] = []
        current_path = prefix

        if isinstance(source, dict) and isinstance(target, dict):
            all_keys = set(source.keys()) | set(target.keys())
            for key in sorted(all_keys):
                key_path = f"{current_path}.{key}" if current_path else key
                if path:
                    key_path = f"{path}.{key}" if path else key_path

                if key in source and key not in target:
                    # Added in source (needs to be set on target)
                    changes.append(
                        Change(
                            path=key_path,
                            kind="added",
                            source_value=source[key],
                            sensitive=is_sensitive_path(key_path),
                            category=self._categorize_path(key_path),
                        )
                    )
                elif key not in source and key in target:
                    # Removed from source (needs to be deleted from target)
                    changes.append(
                        Change(
                            path=key_path,
                            kind="removed",
                            target_value=target[key],
                            sensitive=is_sensitive_path(key_path),
                            category=self._categorize_path(key_path),
                        )
                    )
                elif source[key] is None and target.get(key) is not None:
                    # Null in source means delete from target
                    changes.append(
                        Change(
                            path=key_path,
                            kind="removed",
                            target_value=target[key],
                            sensitive=is_sensitive_path(key_path),
                            category=self._categorize_path(key_path),
                        )
                    )
                else:
                    sub = self._deep_diff(source[key], target[key], key_path)
                    changes.extend(sub)

        elif isinstance(source, list) and isinstance(target, list):
            key_path = path if path else current_path
            if source != target:
                changes.append(
                    Change(
                        path=key_path,
                        kind="modified",
                        source_value=source,
                        target_value=target,
                        sensitive=is_sensitive_path(key_path),
                        category=self._categorize_path(key_path),
                    )
                )
        else:
            key_path = path if path else current_path
            if source != target:
                if target is None and source is not None:
                    changes.append(
                        Change(
                            path=key_path,
                            kind="added",
                            source_value=source,
                            sensitive=is_sensitive_path(key_path),
                            category=self._categorize_path(key_path),
                        )
                    )
                elif source is None and target is not None:
                    changes.append(
                        Change(
                            path=key_path,
                            kind="removed",
                            target_value=target,
                            sensitive=is_sensitive_path(key_path),
                            category=self._categorize_path(key_path),
                        )
                    )
                else:
                    changes.append(
                        Change(
                            path=key_path,
                            kind="modified",
                            source_value=source,
                            target_value=target,
                            sensitive=is_sensitive_path(key_path),
                            category=self._categorize_path(key_path),
                        )
                    )

        return changes

    def _flatten_as_added(self, data: Any, prefix: str) -> list[Change]:
        """Flatten a nested structure as all-added changes."""
        changes: list[Change] = []
        if isinstance(data, dict):
            for key in sorted(data.keys()):
                key_path = f"{prefix}.{key}" if prefix else key
                changes.append(
                    Change(
                        path=key_path,
                        kind="added",
                        source_value=data[key],
                        sensitive=is_sensitive_path(key_path),
                        category=self._categorize_path(key_path),
                    )
                )
        else:
            changes.append(
                Change(
                    path=prefix,
                    kind="added",
                    source_value=data,
                    sensitive=is_sensitive_path(prefix),
                    category=self._categorize_path(prefix),
                )
            )
        return changes

    def _flatten_as_removed(self, data: Any, prefix: str) -> list[Change]:
        """Flatten a nested structure as all-removed changes."""
        changes: list[Change] = []
        if isinstance(data, dict):
            for key in sorted(data.keys()):
                key_path = f"{prefix}.{key}" if prefix else key
                changes.append(
                    Change(
                        path=key_path,
                        kind="removed",
                        target_value=data[key],
                        sensitive=is_sensitive_path(key_path),
                        category=self._categorize_path(key_path),
                    )
                )
        else:
            changes.append(
                Change(
                    path=prefix,
                    kind="removed",
                    target_value=data,
                    sensitive=is_sensitive_path(prefix),
                    category=self._categorize_path(prefix),
                )
            )
        return changes

    @staticmethod
    def _categorize_path(path: str) -> str:
        """Categorize a change by its config path."""
        if any(p in path for p in ("databases.", "apis.", "documents.")):
            return "source"
        if "glossary." in path:
            return "glossary"
        if "relationships." in path:
            return "relationship"
        if any(p in path for p in ("learnings.", "rights.", "facts.")):
            return "rule"
        if "permissions." in path:
            return "permission"
        if "skills." in path:
            return "skill"
        if "agents." in path:
            return "agent"
        if "golden_questions." in path or "golden_questions" == path:
            return "test"
        return "config"

    def _compute_summary(
        self,
        sections: list[SectionDiff],
        domains_added: list[str],
        domains_removed: list[str],
    ) -> DiffSummary:
        """Compute summary statistics from sections."""
        added = 0
        removed = 0
        modified = 0
        by_category: dict[str, int] = {}
        sensitive_count = 0

        for section in sections:
            for change in section.changes:
                if change.kind == "added":
                    added += 1
                elif change.kind == "removed":
                    removed += 1
                elif change.kind == "modified":
                    modified += 1
                by_category[change.category] = (
                    by_category.get(change.category, 0) + 1
                )
                if change.sensitive:
                    sensitive_count += 1

        return DiffSummary(
            total_changes=added + removed + modified,
            added=added,
            removed=removed,
            modified=modified,
            by_category=by_category,
            domains_added=domains_added,
            domains_removed=domains_removed,
            sensitive_changes=sensitive_count,
        )

    def generate_script(self) -> DeployScript:
        """Generate a deployment script from the diff."""
        config_diff = self.diff()
        operations: list[Operation] = []

        for section in config_diff.sections:
            # Domain-level operations
            if section.section.startswith("domain:"):
                domain_name = section.section.split(":", 1)[1]

                if domain_name in config_diff.summary.domains_added:
                    operations.append(
                        Operation(
                            op="create_domain",
                            domain=domain_name,
                            source_dir=f"domains/{domain_name}/",
                            category="config",
                        )
                    )
                    continue

                if domain_name in config_diff.summary.domains_removed:
                    operations.append(
                        Operation(
                            op="delete_domain",
                            domain=domain_name,
                            category="config",
                        )
                    )
                    continue

                # Individual changes within a domain
                for change in section.changes:
                    self._change_to_operation(
                        change,
                        f"domains/{domain_name}/config.yaml",
                        operations,
                    )

            elif section.section.startswith("skills:"):
                skill_name = section.section.split(":", 1)[1]
                for change in section.changes:
                    if change.kind == "added":
                        operations.append(
                            Operation(
                                op="copy_skill",
                                skill=skill_name,
                                source_dir=f"skills/{skill_name}/",
                                category="skill",
                            )
                        )
                    elif change.kind == "removed":
                        operations.append(
                            Operation(
                                op="delete_skill",
                                skill=skill_name,
                                category="skill",
                            )
                        )

            elif section.section.startswith("learnings:"):
                domain_name = section.section.split(":", 1)[1]
                for change in section.changes:
                    self._change_to_operation(
                        change,
                        f"domains/{domain_name}/learnings.yaml",
                        operations,
                    )

            elif section.section == "root":
                for change in section.changes:
                    self._change_to_operation(
                        change, "config.yaml", operations
                    )

            elif section.section == "permissions":
                for change in section.changes:
                    self._change_to_operation(
                        change, "permissions.yaml", operations
                    )

            elif section.section == "agents":
                for change in section.changes:
                    # Agent changes go to agent-specific files
                    parts = change.path.split(".", 1)
                    if len(parts) >= 1:
                        agent_file = f"agents/{parts[0]}.yaml"
                    else:
                        agent_file = "agents/"
                    self._change_to_operation(
                        change, agent_file, operations
                    )

        return DeployScript(
            source_path=config_diff.source_path,
            target_path=config_diff.target_path,
            generated_at=config_diff.generated_at,
            operations=operations,
        )

    @staticmethod
    def _change_to_operation(
        change: Change, file: str, operations: list[Operation]
    ) -> None:
        """Convert a Change to an Operation and append to the list."""
        if change.kind == "added":
            operations.append(
                Operation(
                    op="set",
                    file=file,
                    path=change.path,
                    value=change.source_value,
                    sensitive=change.sensitive,
                    category=change.category,
                )
            )
        elif change.kind == "removed":
            operations.append(
                Operation(
                    op="delete",
                    file=file,
                    path=change.path,
                    sensitive=change.sensitive,
                    category=change.category,
                )
            )
        elif change.kind == "modified":
            operations.append(
                Operation(
                    op="set",
                    file=file,
                    path=change.path,
                    value=change.source_value,
                    sensitive=change.sensitive,
                    category=change.category,
                )
            )
