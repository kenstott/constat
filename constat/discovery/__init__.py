# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Discovery tools for on-demand schema, API, and document discovery.

This module provides tools that the LLM can use to discover relevant
resources without loading everything upfront into the system prompt.

Usage:
    from constat.discovery import DiscoveryTools, DISCOVERY_SYSTEM_PROMPT

    # Create tools with your components
    tools = DiscoveryTools(
        schema_manager=schema_manager,
        api_catalog=api_catalog,
        config=config,
        fact_resolver=fact_resolver,
    )

    # Get tool schemas for LLM
    schemas = tools.get_tool_schemas()

    # Execute a tool call
    result = tools.execute("search_tables", {"query": "customer purchases"})
"""

from .api_tools import APIDiscoveryTools, API_TOOL_SCHEMAS
from .doc_tools import DocumentDiscoveryTools, DOC_TOOL_SCHEMAS
from .fact_tools import FactResolutionTools, FACT_TOOL_SCHEMAS
from .schema_tools import SchemaDiscoveryTools, SCHEMA_TOOL_SCHEMAS
from .tools import (
    DiscoveryTools,
    PromptBuilder,
    DISCOVERY_TOOL_SCHEMAS,
    DISCOVERY_SYSTEM_PROMPT,
)

__all__ = [
    # Main interface
    "DiscoveryTools",
    "PromptBuilder",
    "DISCOVERY_TOOL_SCHEMAS",
    "DISCOVERY_SYSTEM_PROMPT",
    # Individual tool classes
    "SchemaDiscoveryTools",
    "APIDiscoveryTools",
    "DocumentDiscoveryTools",
    "FactResolutionTools",
    # Tool schemas by category
    "SCHEMA_TOOL_SCHEMAS",
    "API_TOOL_SCHEMAS",
    "DOC_TOOL_SCHEMAS",
    "FACT_TOOL_SCHEMAS",
]
