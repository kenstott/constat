# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Tool schemas for LLM document discovery tools."""

# Tool schemas for LLM
DOC_TOOL_SCHEMAS = [
    {
        "name": "list_documents",
        "description": "List all available reference documents with descriptions and tags. Use this to see what documentation is available.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_document",
        "description": "Get the full content of a reference document. Use this to read business rules, data dictionaries, or other documentation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the document to retrieve",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "search_documents",
        "description": "Search across all documents for relevant content using semantic search. Use this to find specific information without reading entire documents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query describing what information you're looking for",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_document_section",
        "description": "Get a specific section from a document by header/title. Use this when you know which section contains the information you need.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the document",
                },
                "section": {
                    "type": "string",
                    "description": "Section header/title to retrieve",
                },
            },
            "required": ["name", "section"],
        },
    },
    {
        "name": "explore_entity",
        "description": "Find all document chunks that mention a specific entity (table, column, API endpoint, concept, or business term). Use this to gather additional context about an entity discovered in search results. This follows entity links to find related documentation across all sources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_name": {
                    "type": "string",
                    "description": "Name of the entity to explore (e.g., 'customers', 'revenue', 'OrderAPI')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of chunks to return",
                    "default": 5,
                },
            },
            "required": ["entity_name"],
        },
    },
]
