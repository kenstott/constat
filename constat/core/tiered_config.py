# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""5-tier hierarchical configuration loader.

Merges configuration from 5 precedence tiers into a flat ResolvedConfig
that represents a session's composite domain:

    Tier 1: System        — base config.yaml (databases, apis, docs, skills, glossary, etc.)
    Tier 2: System Domain — config_dir/domains/<name>/config.yaml (one or more)
    Tier 3: User          — .constat/<user_id>/config.yaml (user overrides)
    Tier 4: User Domain   — .constat/<user_id>/domains/<name>/config.yaml
    Tier 5: Session       — runtime additions (dynamic dbs, apis, file refs)

Higher tiers override lower. Within a tier, multiple domains merge additively.
Null values in higher tiers delete keys from lower tiers.
"""

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml

from constat.core.config import (
    Config,
    DomainConfig,
    LLMConfig,
    _resolve_refs,
    _substitute_env_vars,
)

logger = logging.getLogger(__name__)


class ConfigSource(str, Enum):
    """Identifies which tier a config value came from."""
    SYSTEM = "system"
    SYSTEM_DOMAIN = "system_domain"
    USER = "user"
    USER_DOMAIN = "user_domain"
    SESSION = "session"


@dataclass
class SourcesConfig:
    """Flattened data sources for a session."""
    databases: dict[str, Any] = field(default_factory=dict)
    apis: dict[str, Any] = field(default_factory=dict)
    documents: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedConfig:
    """Fully resolved configuration for a session.

    This is the flat composite domain — everything the session needs.
    Built by merging all 5 tiers in precedence order.
    """
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    rights: dict[str, Any] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)
    learnings: dict[str, Any] = field(default_factory=dict)
    glossary: dict[str, Any] = field(default_factory=dict)
    relationships: dict[str, Any] = field(default_factory=dict)
    skills: dict[str, Any] = field(default_factory=dict)
    llm: Optional[LLMConfig] = None
    preferences: Optional[dict] = None

    # System prompt fragments (merged from all tiers)
    system_prompt: str = ""
    databases_description: str = ""

    # Attribution: dotted path → ConfigSource that set it
    _attribution: dict[str, ConfigSource] = field(default_factory=dict)

    # Active domain names (for reference)
    active_domains: list[str] = field(default_factory=list)

    def source_of(self, path: str) -> Optional[ConfigSource]:
        """Get the tier that set a specific config path."""
        return self._attribution.get(path)


def _deep_merge(base: dict, overlay: dict, path_prefix: str = "",
                attribution: dict | None = None,
                source: ConfigSource | None = None) -> dict:
    """Deep merge overlay into base. Null values in overlay delete keys.

    Args:
        base: Base dictionary
        overlay: Dictionary to merge on top
        path_prefix: Dotted path prefix for attribution tracking
        attribution: Attribution dict to update
        source: ConfigSource for attribution

    Returns:
        Merged dictionary
    """
    result = dict(base)
    for key, value in overlay.items():
        full_path = f"{path_prefix}.{key}" if path_prefix else key

        if value is None:
            # Null deletes the key
            result.pop(key, None)
            if attribution is not None and source is not None:
                attribution[full_path] = source
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            # Recursive merge for nested dicts
            result[key] = _deep_merge(
                result[key], value, full_path, attribution, source
            )
        else:
            result[key] = value
            if attribution is not None and source is not None:
                attribution[full_path] = source

    return result


def _load_yaml_file(path: Path) -> dict:
    """Load and parse a YAML file with env var substitution and $ref resolution."""
    if not path.exists():
        return {}
    with open(path) as f:
        raw = f.read()
    substituted = _substitute_env_vars(raw)
    data = yaml.safe_load(substituted) or {}
    return _resolve_refs(data, path.parent)


def _extract_mergeable_sections(data: dict) -> dict:
    """Extract sections that participate in tiered merging."""
    sections = {}
    for key in ("databases", "apis", "documents", "glossary", "relationships",
                "rights", "facts", "learnings", "skills", "system_prompt",
                "databases_description"):
        if key in data:
            sections[key] = data[key]
    return sections


class TieredConfigLoader:
    """Loads and merges configuration from 5 precedence tiers.

    Usage:
        loader = TieredConfigLoader(
            config=system_config,
            user_id="alice",
            base_dir=Path(".constat"),
            domain_names=["sales", "hr"],
        )
        resolved = loader.resolve()
        # resolved.sources.databases has the merged flat set
        # resolved.glossary has merged glossary terms
        # resolved.source_of("databases.main") → ConfigSource.SYSTEM
    """

    def __init__(
        self,
        config: Config,
        user_id: str = "default",
        base_dir: Optional[Path] = None,
        domain_names: Optional[list[str]] = None,
        session_overrides: Optional[dict] = None,
    ):
        """Initialize the tiered config loader.

        Args:
            config: Parsed system Config (tier 1 + system domains at tier 2)
            user_id: User ID for locating user-tier config
            base_dir: Base directory for .constat user data
            domain_names: Active domain names to include
            session_overrides: Session-scoped config additions (tier 5)
        """
        self._config = config
        self._user_id = user_id
        self._base_dir = base_dir or Path(".constat")
        self._domain_names = domain_names or []
        self._session_overrides = session_overrides or {}

    def resolve(self) -> ResolvedConfig:
        """Build the fully resolved configuration.

        Merges tiers 1-5 in order, tracking attribution.

        Returns:
            ResolvedConfig with all sections merged
        """
        attribution: dict[str, ConfigSource] = {}
        merged: dict[str, Any] = {}

        # --- Tier 1: System config ---
        tier1 = self._load_system_tier()
        merged = _deep_merge(merged, tier1, "", attribution, ConfigSource.SYSTEM)

        # --- Tier 2: System domains (from config.domains) ---
        for domain_name in sorted(self._domain_names):
            domain_config = self._config.load_domain(domain_name)
            if domain_config:
                tier2 = self._extract_domain_sections(domain_config)
                merged = _deep_merge(
                    merged, tier2, "", attribution, ConfigSource.SYSTEM_DOMAIN
                )

        # --- Tier 3: User config ---
        tier3 = self._load_user_tier()
        if tier3:
            merged = _deep_merge(merged, tier3, "", attribution, ConfigSource.USER)

        # --- Tier 4: User domains ---
        for domain_name in sorted(self._domain_names):
            tier4 = self._load_user_domain_tier(domain_name)
            if tier4:
                merged = _deep_merge(
                    merged, tier4, "", attribution, ConfigSource.USER_DOMAIN
                )

        # --- Tier 5: Session overrides ---
        if self._session_overrides:
            merged = _deep_merge(
                merged, self._session_overrides, "", attribution, ConfigSource.SESSION
            )

        # Build ResolvedConfig from merged dict
        return self._build_resolved(merged, attribution)

    def _load_system_tier(self) -> dict:
        """Extract tier-1 sections from system config."""
        data = {}
        # Data sources
        if self._config.databases:
            data["databases"] = {
                name: cfg.model_dump() for name, cfg in self._config.databases.items()
            }
        if self._config.apis:
            data["apis"] = {
                name: cfg.model_dump() for name, cfg in self._config.apis.items()
            }
        if self._config.documents:
            data["documents"] = {
                name: cfg.model_dump() for name, cfg in self._config.documents.items()
            }
        # First-class sections
        if self._config.facts:
            data["facts"] = dict(self._config.facts)
        if self._config.glossary:
            data["glossary"] = dict(self._config.glossary)
        if self._config.relationships:
            data["relationships"] = dict(self._config.relationships)
        if self._config.rights:
            data["rights"] = dict(self._config.rights)
        if self._config.skills:
            data["skills"] = {"paths": list(self._config.skills.paths)}
        if self._config.system_prompt:
            data["system_prompt"] = self._config.system_prompt
        if self._config.databases_description:
            data["databases_description"] = self._config.databases_description
        return data

    @staticmethod
    def _extract_domain_sections(domain: DomainConfig) -> dict:
        """Extract mergeable sections from a DomainConfig."""
        data = {}
        if domain.databases:
            data["databases"] = {
                name: cfg.model_dump() for name, cfg in domain.databases.items()
            }
        if domain.apis:
            data["apis"] = {
                name: cfg.model_dump() for name, cfg in domain.apis.items()
            }
        if domain.documents:
            data["documents"] = {
                name: cfg.model_dump() for name, cfg in domain.documents.items()
            }
        if domain.glossary:
            data["glossary"] = dict(domain.glossary)
        if domain.relationships:
            data["relationships"] = dict(domain.relationships)
        if domain.rights:
            data["rights"] = dict(domain.rights)
        if domain.facts:
            data["facts"] = dict(domain.facts)
        if domain.learnings:
            data["learnings"] = dict(domain.learnings)
        if domain.system_prompt:
            data["system_prompt"] = domain.system_prompt
        if domain.databases_description:
            data["databases_description"] = domain.databases_description
        return data

    def _load_user_tier(self) -> dict:
        """Load tier-3 user config from .constat/<user_id>/config.yaml."""
        user_config_path = self._base_dir / self._user_id / "config.yaml"
        if not user_config_path.exists():
            return {}
        data = _load_yaml_file(user_config_path)
        return _extract_mergeable_sections(data)

    def _load_user_domain_tier(self, domain_name: str) -> dict:
        """Load tier-4 user domain config."""
        domain_path = self._base_dir / self._user_id / "domains" / domain_name / "config.yaml"
        if not domain_path.exists():
            return {}
        data = _load_yaml_file(domain_path)
        return _extract_mergeable_sections(data)

    def _build_resolved(self, merged: dict, attribution: dict) -> ResolvedConfig:
        """Build a ResolvedConfig from the merged dict."""
        sources = SourcesConfig(
            databases=merged.get("databases", {}),
            apis=merged.get("apis", {}),
            documents=merged.get("documents", {}),
        )

        return ResolvedConfig(
            sources=sources,
            rights=merged.get("rights", {}),
            facts=merged.get("facts", {}),
            learnings=merged.get("learnings", {}),
            glossary=merged.get("glossary", {}),
            relationships=merged.get("relationships", {}),
            skills=merged.get("skills", {}),
            llm=self._config.llm,
            system_prompt=merged.get("system_prompt", ""),
            databases_description=merged.get("databases_description", ""),
            active_domains=list(self._domain_names),
            _attribution=attribution,
        )
