# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""API discovery and execution tools for GraphQL and REST endpoints.

These tools allow the LLM to:
1. Discover API operations on-demand (without loading everything upfront)
2. Execute queries against external APIs to fetch data
"""

from typing import Any, Optional

from constat.catalog.api_catalog import APICatalog, OperationType
from constat.catalog.api_executor import APIExecutor, APIExecutionError
from constat.core.config import Config


class APIDiscoveryTools:
    """Tools for discovering and executing API operations."""

    def __init__(self, api_catalog: APICatalog, config: Optional[Config] = None):
        self.api_catalog = api_catalog
        self.config = config
        self._executor: Optional[APIExecutor] = None

    @property
    def executor(self) -> Optional[APIExecutor]:
        """Lazy-initialize API executor."""
        if self._executor is None and self.config:
            self._executor = APIExecutor(self.config)
        return self._executor

    def list_apis(self) -> list[dict]:
        """
        List all available APIs with their types and descriptions.

        Returns:
            List of API info dicts with name, type, description, operation_count
        """
        results = []

        if self.config:
            # Get API info from config
            for api_name, api_config in self.config.apis.items():
                # Count operations for this API
                operation_count = sum(
                    1 for op in self.api_catalog.operations.values()
                    if api_name in op.tags or api_name.lower() in op.full_name.lower()
                )

                results.append({
                    "name": api_name,
                    "type": api_config.type,
                    "url": api_config.url or api_config.spec_url or "",
                    "description": api_config.description or f"API: {api_name}",
                    "operation_count": operation_count,
                })
        else:
            # Derive from operations if no config
            # Group operations by first tag or infer from structure
            api_names = set()
            for op in self.api_catalog.operations.values():
                if op.tags:
                    api_names.add(op.tags[0])

            for api_name in api_names:
                operations = [
                    op for op in self.api_catalog.operations.values()
                    if api_name in op.tags
                ]
                if operations:
                    # Infer type from operations
                    op_type = "graphql" if any(
                        op.operation_type in [OperationType.QUERY, OperationType.MUTATION, OperationType.SUBSCRIPTION]
                        for op in operations
                    ) else "openapi"

                    results.append({
                        "name": api_name,
                        "type": op_type,
                        "description": f"API with {len(operations)} operations",
                        "operation_count": len(operations),
                    })

        return results

    def list_operations(self, api: Optional[str] = None) -> list[dict]:
        """
        List operations, optionally filtered by API name.

        Args:
            api: Optional API name to filter by

        Returns:
            List of operation info dicts with name, type, description
        """
        results = []

        for op in self.api_catalog.operations.values():
            # Filter by API if specified
            if api:
                if api not in op.tags and api.lower() not in op.full_name.lower():
                    continue

            results.append({
                "name": op.name,
                "full_name": op.full_name,
                "type": op.operation_type.value,
                "description": op.description or "",
                "return_type": op.return_type,
                "argument_count": len(op.arguments),
            })

        # Sort by type then name
        results.sort(key=lambda x: (x["type"], x["name"]))
        return results

    def get_operation_details(self, operation: str) -> dict:
        """
        Get detailed schema for an API operation.

        Args:
            operation: Operation name (with or without type prefix)

        Returns:
            Dict with full operation details including arguments, return type, examples
        """
        op = self.api_catalog.get_operation(operation)

        if not op:
            return {"error": f"Operation not found: {operation}"}

        return {
            "name": op.name,
            "full_name": op.full_name,
            "type": op.operation_type.value,
            "description": op.description,
            "arguments": [
                {
                    "name": arg.name,
                    "type": arg.type,
                    "required": arg.required,
                    "description": arg.description or "",
                    **({"default": arg.default} if arg.default is not None else {}),
                }
                for arg in op.arguments
            ],
            "return_type": op.return_type,
            "return_fields": op.return_fields,
            "response_schema": op.response_schema,
            "use_cases": op.use_cases,
            "tags": op.tags,
        }

    def search_operations(self, query: str, limit: int = 5, operation_type: Optional[str] = None) -> list[dict]:
        """
        Find operations relevant to a natural language query using vector search.

        Args:
            query: Natural language description (e.g., "get user by ID", "create new order")
            limit: Maximum number of results
            operation_type: Optional filter: "query", "mutation", or "subscription"

        Returns:
            List of relevant operations with relevance scores
        """
        # Convert string type to enum
        op_type_enum = None
        if operation_type:
            type_map = {
                "query": OperationType.QUERY,
                "mutation": OperationType.MUTATION,
                "subscription": OperationType.SUBSCRIPTION,
            }
            op_type_enum = type_map.get(operation_type.lower())

        matches = self.api_catalog.find_relevant_operations(
            query=query,
            top_k=limit,
            operation_type=op_type_enum,
        )

        results = []
        for m in matches:
            result = {
                "name": m.operation,
                "type": m.operation_type.value,
                "relevance": m.relevance,
                "summary": m.summary,
                "use_cases": m.use_cases,
                # Protocol identification - tells LLM how to call this operation
                "protocol": m.protocol.value,  # "graphql" or "rest"
                "api_name": m.api_name,  # Which API this operation belongs to
            }
            # Add REST-specific fields only for REST operations
            if m.protocol.value == "rest":
                result["http_method"] = m.http_method  # GET, POST, PUT, PATCH, DELETE
                result["path"] = m.path  # URL path template, e.g., "/users/{userId}"
            results.append(result)
        return results

    def execute_graphql(
        self,
        api: str,
        query: str,
        variables: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Execute a GraphQL query against an external API.

        Args:
            api: Name of the API (as configured in config.yaml)
            query: GraphQL query string (e.g., "{ countries { name code } }")
            variables: Optional query variables

        Returns:
            Query result data, or error dict if execution fails
        """
        if not self.executor:
            return {"error": "No API executor configured. Check that APIs are defined in config."}

        try:
            return self.executor.execute_graphql(api, query, variables)
        except APIExecutionError as e:
            return {
                "error": str(e),
                "status_code": e.status_code,
                "response": e.response_body,
            }

    def execute_rest(
        self,
        api: str,
        path: str,
        method: str = "GET",
        path_params: Optional[dict[str, Any]] = None,
        query_params: Optional[dict[str, Any]] = None,
        body: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Execute a REST API call.

        Args:
            api: Name of the API (as configured in config.yaml)
            path: API path or operation ID (e.g., "/users/{userId}" or "getUser")
            method: HTTP method (GET, POST, PUT, PATCH, DELETE)
            path_params: Parameters to substitute in path
            query_params: Query string parameters
            body: Request body for POST/PUT/PATCH

        Returns:
            API response data, or error dict if execution fails
        """
        if not self.executor:
            return {"error": "No API executor configured. Check that APIs are defined in config."}

        try:
            return self.executor.execute_rest(
                api, path, path_params, query_params, body, method
            )
        except APIExecutionError as e:
            return {
                "error": str(e),
                "status_code": e.status_code,
                "response": e.response_body,
            }


# Tool schemas for LLM
API_TOOL_SCHEMAS = [
    {
        "name": "list_apis",
        "description": "List all available APIs with their types and descriptions. Use this first to see what APIs are configured.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "list_api_operations",
        "description": "List all operations available in an API. Use after list_apis to explore a specific API.",
        "input_schema": {
            "type": "object",
            "properties": {
                "api": {
                    "type": "string",
                    "description": "Name of the API to list operations from. Omit to list all operations.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_operation_details",
        "description": "Get detailed schema for an API operation including arguments, return type, and usage examples. Use this before calling an API.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Name of the operation (e.g., 'countries', 'getPetById')",
                },
            },
            "required": ["operation"],
        },
    },
    {
        "name": "search_operations",
        "description": "Find API operations relevant to a natural language query using semantic search. Use this when you're not sure which operation to use.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what you want to do (e.g., 'get user by ID', 'list all products')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5,
                },
                "operation_type": {
                    "type": "string",
                    "enum": ["query", "mutation", "subscription"],
                    "description": "Filter by operation type (optional)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "execute_graphql",
        "description": "Execute a GraphQL query against an external API to fetch data. Use this after discovering the API and its operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "api": {
                    "type": "string",
                    "description": "Name of the API (as shown in list_apis)",
                },
                "query": {
                    "type": "string",
                    "description": "GraphQL query string (e.g., '{ countries { name code } }')",
                },
                "variables": {
                    "type": "object",
                    "description": "Optional query variables",
                },
            },
            "required": ["api", "query"],
        },
    },
    {
        "name": "execute_rest",
        "description": "Execute a REST API call to fetch or modify data. Use this after discovering the API and its operations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "api": {
                    "type": "string",
                    "description": "Name of the API (as shown in list_apis)",
                },
                "path": {
                    "type": "string",
                    "description": "API path or operation ID (e.g., '/users/{userId}' or 'getUser')",
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                    "description": "HTTP method (default: GET)",
                },
                "path_params": {
                    "type": "object",
                    "description": "Parameters to substitute in path (e.g., {\"userId\": \"123\"})",
                },
                "query_params": {
                    "type": "object",
                    "description": "Query string parameters",
                },
                "body": {
                    "type": "object",
                    "description": "Request body for POST/PUT/PATCH",
                },
            },
            "required": ["api", "path"],
        },
    },
]
