"""Apply engine: applies deployment scripts to target config directories."""

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from constat.deploy.script import DeployScript, Operation


class DeployApplier:
    """Applies a deployment script to a target config directory."""

    def __init__(self, target_path: str, dry_run: bool = False):
        self.target_path = target_path
        self.dry_run = dry_run
        self.applied: list[str] = []
        self.skipped: list[str] = []
        self.errors: list[str] = []

    def apply(
        self,
        script: DeployScript,
        only: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> dict:
        """Apply operations from the deployment script.

        Args:
            script: The deployment script to apply.
            only: If set, only apply operations with these categories.
            exclude: If set, skip operations with these categories.

        Returns:
            Dict with applied, skipped, errors lists.
        """
        self.applied = []
        self.skipped = []
        self.errors = []

        # 1. Backup target
        backup_path = self._backup()

        # 2. Filter operations by category
        ops = self._filter_operations(script.operations, only, exclude)

        # 3. Separate operations by type for ordering
        domain_ops = [o for o in ops if o.op in ("create_domain", "delete_domain")]
        skill_ops = [o for o in ops if o.op in ("copy_skill", "delete_skill")]
        yaml_ops = [o for o in ops if o.op in ("set", "delete")]

        # 4. Apply domain ops first
        for op in domain_ops:
            self._apply_domain_op(op, script.source_path)

        # 5. Apply skill ops
        for op in skill_ops:
            self._apply_skill_op(op, script.source_path)

        # 6. Apply YAML mutations
        for op in yaml_ops:
            if op.op == "set":
                self._apply_set(op)
            elif op.op == "delete":
                self._apply_delete(op)

        # 7. Validate target config loads
        if not self.dry_run and self.errors:
            self._rollback(backup_path)

        return {
            "applied": list(self.applied),
            "skipped": list(self.skipped),
            "errors": list(self.errors),
            "backup_path": backup_path,
        }

    def _filter_operations(
        self,
        operations: list[Operation],
        only: set[str] | None,
        exclude: set[str] | None,
    ) -> list[Operation]:
        """Filter operations by category."""
        filtered = []
        for op in operations:
            if only and op.category not in only:
                self.skipped.append(f"{op.op} {op.path or op.domain or op.skill}")
                continue
            if exclude and op.category in exclude:
                self.skipped.append(f"{op.op} {op.path or op.domain or op.skill}")
                continue
            filtered.append(op)
        return filtered

    def _backup(self) -> str:
        """Create timestamped backup of target config directory."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        backup_dir = Path(self.target_path) / ".backups" / timestamp

        if self.dry_run:
            return str(backup_dir)

        backup_dir.parent.mkdir(parents=True, exist_ok=True)

        # Copy everything except .backups
        target = Path(self.target_path)
        for item in target.iterdir():
            if item.name == ".backups":
                continue
            dest = backup_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dest)

        return str(backup_dir)

    def _apply_domain_op(self, op: Operation, source_path: str) -> None:
        """Apply create_domain or delete_domain operations."""
        target = Path(self.target_path)

        if op.op == "create_domain":
            src_domain = Path(source_path) / "domains" / op.domain
            dst_domain = target / "domains" / op.domain
            desc = f"create_domain {op.domain}"

            if self.dry_run:
                self.applied.append(desc)
                return

            if src_domain.is_dir():
                dst_domain.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_domain, dst_domain, dirs_exist_ok=True)
                self.applied.append(desc)
            else:
                self.errors.append(f"{desc}: source directory not found")

        elif op.op == "delete_domain":
            dst_domain = target / "domains" / op.domain
            desc = f"delete_domain {op.domain}"

            if self.dry_run:
                self.applied.append(desc)
                return

            if dst_domain.is_dir():
                shutil.rmtree(dst_domain)
                self.applied.append(desc)
            else:
                self.errors.append(f"{desc}: target directory not found")

    def _apply_skill_op(self, op: Operation, source_path: str) -> None:
        """Apply copy_skill or delete_skill operations."""
        target = Path(self.target_path)

        if op.op == "copy_skill":
            src_skill = Path(source_path) / "skills" / op.skill
            dst_skill = target / "skills" / op.skill
            desc = f"copy_skill {op.skill}"

            if self.dry_run:
                self.applied.append(desc)
                return

            if src_skill.is_dir():
                dst_skill.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src_skill, dst_skill, dirs_exist_ok=True)
                self.applied.append(desc)
            else:
                self.errors.append(f"{desc}: source directory not found")

        elif op.op == "delete_skill":
            dst_skill = target / "skills" / op.skill
            desc = f"delete_skill {op.skill}"

            if self.dry_run:
                self.applied.append(desc)
                return

            if dst_skill.is_dir():
                shutil.rmtree(dst_skill)
                self.applied.append(desc)
            else:
                self.errors.append(f"{desc}: target directory not found")

    def _apply_set(self, op: Operation) -> None:
        """Set a value at a YAML path in the target file."""
        file_path = Path(self.target_path) / op.file
        desc = f"set {op.file}:{op.path}"

        if self.dry_run:
            self.applied.append(desc)
            return

        try:
            # Load existing file or start empty
            if file_path.exists():
                with open(file_path) as f:
                    data = yaml.safe_load(f.read())
                if data is None:
                    data = {}
            else:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                data = {}

            # Navigate to parent and set value
            self._set_nested(data, op.path, op.value)

            # Write back
            with open(file_path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

            self.applied.append(desc)
        except Exception as e:
            self.errors.append(f"{desc}: {e}")

    def _apply_delete(self, op: Operation) -> None:
        """Delete a key at a YAML path in the target file."""
        file_path = Path(self.target_path) / op.file
        desc = f"delete {op.file}:{op.path}"

        if self.dry_run:
            self.applied.append(desc)
            return

        try:
            if not file_path.exists():
                self.skipped.append(f"{desc} (file not found)")
                return

            with open(file_path) as f:
                data = yaml.safe_load(f.read())
            if data is None:
                self.skipped.append(f"{desc} (empty file)")
                return

            # Navigate to parent and delete key
            if self._delete_nested(data, op.path):
                with open(file_path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
                self.applied.append(desc)
            else:
                self.skipped.append(f"{desc} (key not found)")
        except Exception as e:
            self.errors.append(f"{desc}: {e}")

    @staticmethod
    def _resolve_path_parts(data: dict, path: str) -> list[str]:
        """Resolve a dot-delimited path into actual key segments.

        Handles keys containing dots (e.g. email addresses) by greedily
        matching against existing dict keys at each level. For new keys
        (not yet in the dict), checks sibling key patterns to infer whether
        dots are part of the key name.
        """
        parts = path.split(".")
        resolved: list[str] = []
        current = data
        i = 0
        while i < len(parts):
            matched = False
            if isinstance(current, dict):
                # Try progressively longer segments to match existing keys
                for end in range(len(parts), i, -1):
                    candidate = ".".join(parts[i:end])
                    if candidate in current:
                        resolved.append(candidate)
                        current = current[candidate]
                        i = end
                        matched = True
                        break

                if not matched:
                    # Key doesn't exist yet. Check if sibling keys contain dots
                    # to infer whether the remaining segments form a single key.
                    siblings_have_dots = any("." in k for k in current)
                    if siblings_have_dots and i < len(parts) - 1:
                        # Treat all remaining parts (except any that could be
                        # a final sub-key) as one dotted key. Try longest first.
                        for end in range(len(parts), i, -1):
                            candidate = ".".join(parts[i:end])
                            # Accept if it looks like it could be a sibling
                            resolved.append(candidate)
                            current = {}
                            i = end
                            matched = True
                            break

            if not matched:
                resolved.append(parts[i])
                if isinstance(current, dict):
                    current = current.get(parts[i], {})
                i += 1
        return resolved

    @staticmethod
    def _set_nested(data: dict, path: str, value: Any) -> None:
        """Set a value at a dot-delimited path in a nested dict."""
        parts = DeployApplier._resolve_path_parts(data, path)
        current = data
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    @staticmethod
    def _delete_nested(data: dict, path: str) -> bool:
        """Delete a key at a dot-delimited path. Returns True if deleted."""
        parts = DeployApplier._resolve_path_parts(data, path)
        current = data
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                return False
            current = current[part]
        if parts[-1] in current:
            del current[parts[-1]]
            return True
        return False

    def _validate(self) -> bool:
        """Validate target config loads without errors."""
        config_file = Path(self.target_path) / "config.yaml"
        if not config_file.exists():
            return True
        try:
            with open(config_file) as f:
                yaml.safe_load(f.read())
            return True
        except Exception:
            return False

    def _rollback(self, backup_path: str) -> None:
        """Restore from backup."""
        backup = Path(backup_path)
        if not backup.is_dir():
            return

        target = Path(self.target_path)

        # Remove current contents (except .backups)
        for item in target.iterdir():
            if item.name == ".backups":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        # Restore from backup
        for item in backup.iterdir():
            dest = target / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
