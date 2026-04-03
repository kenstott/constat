"""Data models for deployment diffs and scripts."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class Change:
    """A single change between source and target configs."""
    path: str  # dot-delimited key path
    kind: Literal["added", "removed", "modified"]
    source_value: Any = None
    target_value: Any = None
    sensitive: bool = False
    category: str = "config"


@dataclass
class SectionDiff:
    """Changes within a single config section."""
    section: str  # "root", "domain:sales-analytics", "permissions", etc.
    changes: list[Change] = field(default_factory=list)


@dataclass
class DiffSummary:
    """Summary statistics for a config diff."""
    total_changes: int = 0
    added: int = 0
    removed: int = 0
    modified: int = 0
    by_category: dict[str, int] = field(default_factory=dict)
    domains_added: list[str] = field(default_factory=list)
    domains_removed: list[str] = field(default_factory=list)
    sensitive_changes: int = 0


@dataclass
class ConfigDiff:
    """Complete diff between two config directories."""
    source_path: str
    target_path: str
    generated_at: str
    sections: list[SectionDiff] = field(default_factory=list)
    summary: DiffSummary = field(default_factory=DiffSummary)


@dataclass
class Operation:
    """A single deployment operation."""
    op: str  # set, delete, create_domain, delete_domain, copy_skill, delete_skill
    file: str = ""
    path: str = ""
    value: Any = None
    domain: str = ""
    skill: str = ""
    source_dir: str = ""
    sensitive: bool = False
    category: str = "config"


@dataclass
class DeployScript:
    """A deployment script: ordered list of operations to apply."""
    source_path: str
    target_path: str
    generated_at: str
    operations: list[Operation] = field(default_factory=list)
