# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.

"""Tests for constat.core.resource_validation."""

import pytest

from constat.core.config import DomainConfig
from constat.core.resource_validation import (
    _domain_resource_patterns,
    _resource_available,
    extract_resources_from_grounding,
    validate_resource_compatibility,
)


def _make_domain(**kwargs) -> DomainConfig:
    """Build a minimal DomainConfig for testing."""
    defaults = {"name": "test"}
    defaults.update(kwargs)
    return DomainConfig(**defaults)


# ---- _domain_resource_patterns ----

def test_domain_resource_patterns():
    from constat.core.config import DatabaseConfig, APIConfig, DocumentConfig

    dc = _make_domain(
        databases={"chinook": DatabaseConfig(uri="sqlite:///chinook.db")},
        apis={"weather": APIConfig(type="rest", url="https://api.weather.com")},
        documents={"glossary": DocumentConfig(path="./glossary.md")},
    )
    patterns = _domain_resource_patterns(dc)
    assert patterns == {"schema:chinook", "api:weather", "document:glossary"}


def test_domain_resource_patterns_empty():
    dc = _make_domain()
    assert _domain_resource_patterns(dc) == set()


# ---- _resource_available ----

def test_resource_available_exact():
    assert _resource_available("schema:chinook", {"schema:chinook"})


def test_resource_available_prefix_match():
    assert _resource_available("schema:chinook.breeds", {"schema:chinook"})


def test_resource_not_available():
    assert not _resource_available("schema:postgres.orders", {"schema:chinook"})


def test_resource_not_available_partial_name():
    # "schema:chin" should NOT match "schema:chinook" — only dotted children match
    assert not _resource_available("schema:chin", {"schema:chinook"})


# ---- validate_resource_compatibility ----

def test_validate_missing_resource():
    dc = _make_domain()
    warnings = validate_resource_compatibility(["schema:chinook"], dc, "empty-domain")
    assert len(warnings) == 1
    assert "schema:chinook" in warnings[0]
    assert "empty-domain" in warnings[0]


def test_validate_all_present():
    from constat.core.config import DatabaseConfig

    dc = _make_domain(databases={"chinook": DatabaseConfig(uri="sqlite:///x")})
    warnings = validate_resource_compatibility(["schema:chinook"], dc, "has-chinook")
    assert warnings == []


def test_validate_empty_resources():
    dc = _make_domain()
    warnings = validate_resource_compatibility([], dc, "any")
    assert warnings == []


# ---- extract_resources_from_grounding ----

def test_extract_resources_from_grounding():
    grounding = [
        {"source": "database:chinook", "source_name": "chinook", "table_name": "breeds"},
        {"source": "api:weather", "source_name": "weather", "api_endpoint": "forecast"},
        {"source": "document:glossary", "source_name": "glossary"},
    ]
    resources = extract_resources_from_grounding(grounding)
    assert "schema:chinook.breeds" in resources
    assert "api:forecast" in resources
    assert "document:glossary" in resources


def test_extract_resources_deduplicates():
    grounding = [
        {"source": "database:chinook", "source_name": "chinook", "table_name": "breeds"},
        {"source": "database:chinook", "source_name": "chinook", "table_name": "breeds"},
    ]
    resources = extract_resources_from_grounding(grounding)
    assert resources.count("schema:chinook.breeds") == 1


def test_extract_resources_empty():
    assert extract_resources_from_grounding([]) == []
