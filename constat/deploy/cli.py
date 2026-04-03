"""CLI commands for constat deploy."""

import argparse
import sys
from datetime import datetime, timezone

import yaml

from constat.deploy.differ import ConfigDiffer
from constat.deploy.applier import DeployApplier
from constat.deploy.script import DeployScript, Operation
from constat.deploy.sensitive import is_sensitive_path, mask_value


def main(args: list[str] | None = None):
    """Entry point for deploy subcommands."""
    parser = argparse.ArgumentParser(prog="constat deploy")
    subparsers = parser.add_subparsers(dest="command")

    # diff
    diff_parser = subparsers.add_parser("diff", help="Show diff between configs")
    diff_parser.add_argument("--source", required=True, help="Source config directory")
    diff_parser.add_argument("--target", required=True, help="Target config directory")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate deployment script")
    gen_parser.add_argument("--source", required=True, help="Source config directory")
    gen_parser.add_argument("--target", required=True, help="Target config directory")
    gen_parser.add_argument("-o", "--output", default="deploy.yaml", help="Output file")

    # apply
    apply_parser = subparsers.add_parser("apply", help="Apply deployment script")
    apply_parser.add_argument("script", help="Path to deployment script YAML")
    apply_parser.add_argument("--target", required=True, help="Target config directory")
    apply_parser.add_argument(
        "--dry-run", action="store_true", default=True, help="Dry run (default)"
    )
    apply_parser.add_argument(
        "--no-dry-run", action="store_true", help="Actually apply changes"
    )
    apply_parser.add_argument("--only", help="Comma-separated categories to include")
    apply_parser.add_argument("--exclude", help="Comma-separated categories to exclude")

    parsed = parser.parse_args(args)

    if parsed.command is None:
        parser.print_help()
        sys.exit(1)

    if parsed.command == "diff":
        _cmd_diff(parsed)
    elif parsed.command == "generate":
        _cmd_generate(parsed)
    elif parsed.command == "apply":
        _cmd_apply(parsed)


def _cmd_diff(parsed: argparse.Namespace) -> None:
    """Show human-readable diff between two config directories."""
    differ = ConfigDiffer(parsed.source, parsed.target)
    config_diff = differ.diff()

    for section in config_diff.sections:
        print(f"\n{section.section}:")
        for change in section.changes:
            symbol = {"added": "+", "removed": "-", "modified": "~"}[change.kind]
            if change.sensitive:
                src = mask_value(change.source_value) if change.source_value else ""
                tgt = mask_value(change.target_value) if change.target_value else ""
            else:
                src = repr(change.source_value) if change.source_value is not None else ""
                tgt = repr(change.target_value) if change.target_value is not None else ""

            if change.kind == "modified":
                print(f"  {symbol} {change.path}: {tgt} -> {src}")
            elif change.kind == "added":
                print(f"  {symbol} {change.path}: {src}")
            else:
                print(f"  {symbol} {change.path}")

    s = config_diff.summary
    print(
        f"\nSummary: {s.total_changes} changes "
        f"({s.added} added, {s.removed} removed, {s.modified} modified) | "
        f"{s.sensitive_changes} sensitive"
    )
    if s.domains_added:
        print(f"  New domains: {', '.join(s.domains_added)}")
    if s.domains_removed:
        print(f"  Removed domains: {', '.join(s.domains_removed)}")


def _cmd_generate(parsed: argparse.Namespace) -> None:
    """Generate a deployment script YAML file."""
    differ = ConfigDiffer(parsed.source, parsed.target)
    script = differ.generate_script()

    # Serialize to YAML
    data = _script_to_dict(script)
    with open(parsed.output, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"Deployment script written to: {parsed.output}")


def _cmd_apply(parsed: argparse.Namespace) -> None:
    """Apply a deployment script to a target directory."""
    dry_run = not parsed.no_dry_run

    # Load script
    with open(parsed.script) as f:
        data = yaml.safe_load(f.read())
    script = _dict_to_script(data)

    only = set(parsed.only.split(",")) if parsed.only else None
    exclude = set(parsed.exclude.split(",")) if parsed.exclude else None

    applier = DeployApplier(parsed.target, dry_run=dry_run)
    result = applier.apply(script, only=only, exclude=exclude)

    mode = "DRY RUN" if dry_run else "APPLIED"
    print(f"\n=== {mode} ===")
    print(f"Applied: {len(result['applied'])}")
    print(f"Skipped: {len(result['skipped'])}")
    print(f"Errors: {len(result['errors'])}")

    if result["errors"]:
        for err in result["errors"]:
            print(f"  ERROR: {err}")
        sys.exit(1)


def _script_to_dict(script: DeployScript) -> dict:
    """Serialize a DeployScript to a plain dict for YAML output."""
    ops = []
    for op in script.operations:
        d: dict = {"op": op.op}
        if op.file:
            d["file"] = op.file
        if op.path:
            d["path"] = op.path
        if op.value is not None:
            if op.sensitive:
                d["value"] = mask_value(op.value)
            else:
                d["value"] = op.value
        if op.domain:
            d["domain"] = op.domain
        if op.skill:
            d["skill"] = op.skill
        if op.source_dir:
            d["source_dir"] = op.source_dir
        if op.sensitive:
            d["sensitive"] = True
        if op.category != "config":
            d["category"] = op.category
        ops.append(d)

    return {
        "source_path": script.source_path,
        "target_path": script.target_path,
        "generated_at": script.generated_at,
        "operations": ops,
    }


def _dict_to_script(data: dict) -> DeployScript:
    """Deserialize a dict (from YAML) into a DeployScript."""
    operations = []
    for op_data in data.get("operations", []):
        operations.append(
            Operation(
                op=op_data["op"],
                file=op_data.get("file", ""),
                path=op_data.get("path", ""),
                value=op_data.get("value"),
                domain=op_data.get("domain", ""),
                skill=op_data.get("skill", ""),
                source_dir=op_data.get("source_dir", ""),
                sensitive=op_data.get("sensitive", False),
                category=op_data.get("category", "config"),
            )
        )

    return DeployScript(
        source_path=data["source_path"],
        target_path=data["target_path"],
        generated_at=data["generated_at"],
        operations=operations,
    )
