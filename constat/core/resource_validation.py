# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Resource validation for skill and golden question moves."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from constat.core.config import DomainConfig


def _domain_resource_patterns(domain: DomainConfig) -> set[str]:
    """Build resource patterns from a DomainConfig's data sources."""
    patterns: set[str] = set()
    for db_name in domain.databases:
        patterns.add(f"schema:{db_name}")
    for api_name in domain.apis:
        patterns.add(f"api:{api_name}")
    for doc_name in domain.documents:
        patterns.add(f"document:{doc_name}")
    return patterns


def _resource_available(pattern: str, domain_patterns: set[str]) -> bool:
    """Check if a resource pattern is available via prefix matching.

    ``schema:chinook.breeds`` matches ``schema:chinook``.
    """
    if pattern in domain_patterns:
        return True
    for dp in domain_patterns:
        if pattern.startswith(dp + "."):
            return True
    return False


def validate_resource_compatibility(
    required_resources: list[str],
    target_domain: "DomainConfig",
    target_domain_name: str,
) -> list[str]:
    """Return warning strings for resources missing from *target_domain*."""
    if not required_resources:
        return []
    domain_patterns = _domain_resource_patterns(target_domain)
    warnings: list[str] = []
    for resource in required_resources:
        if not _resource_available(resource, domain_patterns):
            warnings.append(
                f"Resource '{resource}' is not available in domain '{target_domain_name}'"
            )
    return warnings


def extract_resources_from_grounding(grounding: list[dict]) -> list[str]:
    """Extract and deduplicate resource patterns from golden question grounding assertions."""
    from constat.testing.grounding import build_source_patterns

    seen: set[str] = set()
    result: list[str] = []
    for assertion in grounding:
        for pattern in build_source_patterns(assertion):
            if pattern not in seen:
                seen.add(pattern)
                result.append(pattern)
    return result
