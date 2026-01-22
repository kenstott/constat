# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""API executor for querying external GraphQL and REST APIs as data sources.

This module provides the actual execution layer that was missing - it can
execute queries against external APIs configured in the config.

Usage:
    from constat.catalog.api_executor import APIExecutor
    from constat.core.config import Config

    config = Config.from_yaml("config.yaml")
    executor = APIExecutor(config)

    # Execute a GraphQL query
    result = executor.execute_graphql(
        api_name="countries",
        query="{ countries { name code } }"
    )

    # Execute a REST call
    result = executor.execute_rest(
        api_name="petstore",
        operation="getPetById",
        params={"petId": 123}
    )
"""

import json
import logging
from typing import Any, Optional
from urllib.parse import urljoin, urlencode

import httpx

from constat.core.config import Config, APIConfig


class APIExecutionError(Exception):
    """Raised when an API call fails."""
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        retryable: bool = True,
        retry_hint: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.retryable = retryable
        self.retry_hint = retry_hint


def classify_http_error(status_code: int, response_body: str = "") -> tuple[bool, str]:
    """
    Classify HTTP error by status code to determine retry strategy.

    Returns:
        (retryable, retry_hint) tuple with guidance for LLM code generation.
    """
    # Auth errors - NOT retryable (credentials won't change)
    if status_code == 401:
        return False, "Authentication failed. Check API credentials in config."
    if status_code == 403:
        return False, "Permission denied. The API key may lack required permissions."

    # Client errors that might be fixable with different params
    if status_code == 400:
        return True, "Bad request - check query parameters, request body format, or field names."
    if status_code == 404:
        return True, "Resource not found - check path parameters, IDs, or endpoint path. The resource may not exist."
    if status_code == 405:
        return False, "HTTP method not allowed. Check the correct method (GET/POST/PUT/DELETE) for this endpoint."
    if status_code == 422:
        return True, "Validation error - check field types, required fields, or data constraints."

    # Rate limiting - retryable but may need delay
    if status_code == 429:
        return True, "Rate limited. Consider reducing request frequency or adding delays."

    # Server errors - potentially transient, worth one retry
    if status_code >= 500:
        if status_code == 500:
            return True, "Internal server error (transient). Retry once - if it persists, the API may have issues."
        if status_code == 502:
            return True, "Bad gateway (transient). The upstream server may be temporarily unavailable."
        if status_code == 503:
            return True, "Service unavailable (transient). The API may be under maintenance or overloaded."
        if status_code == 504:
            return True, "Gateway timeout (transient). The request took too long - try simplifying the query."
        # Other 5xx errors
        return True, f"Server error {status_code} (possibly transient). Retry once."

    # Other 4xx errors - generally retryable with fixes
    if status_code >= 400:
        return True, f"Client error {status_code}. Check the request parameters and format."

    # Shouldn't reach here for errors, but default to retryable
    return True, f"Unexpected status {status_code}."


class APIExecutor:
    """
    Executes queries against external GraphQL and REST APIs.

    This is the actual execution layer for API data sources. It handles:
    - GraphQL query execution
    - REST/OpenAPI call execution
    - Authentication (bearer, basic, api_key, custom headers)
    - Error handling and response parsing
    """

    def __init__(self, config: Config, timeout: float = 30.0):
        """
        Initialize the API executor.

        Args:
            config: Configuration containing API definitions
            timeout: Request timeout in seconds
        """
        self.config = config
        self.timeout = timeout
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_api_config(self, api_name: str) -> APIConfig:
        """Get API config by name."""
        if api_name not in self.config.apis:
            available = list(self.config.apis.keys())
            raise APIExecutionError(
                f"API '{api_name}' not found. Available APIs: {available}"
            )
        return self.config.apis[api_name]

    def _build_headers(self, api_config: APIConfig) -> dict[str, str]:
        """Build request headers including authentication."""
        headers = {"Content-Type": "application/json"}

        # Add custom headers from config
        headers.update(api_config.headers)

        # Add authentication
        if api_config.auth_type == "bearer" and api_config.auth_token:
            headers["Authorization"] = f"Bearer {api_config.auth_token}"
        elif api_config.auth_type == "basic" and api_config.auth_username:
            import base64
            credentials = f"{api_config.auth_username}:{api_config.auth_password or ''}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"
        elif api_config.auth_type == "api_key" and api_config.api_key:
            headers[api_config.api_key_header] = api_config.api_key

        return headers

    def execute_graphql(
        self,
        api_name: str,
        query: str,
        variables: Optional[dict[str, Any]] = None,
        operation_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Execute a GraphQL query against an external API.

        Args:
            api_name: Name of the API (as configured)
            query: GraphQL query string
            variables: Optional query variables
            operation_name: Optional operation name (for queries with multiple operations)

        Returns:
            The 'data' portion of the GraphQL response

        Raises:
            APIExecutionError: If the request fails or returns errors
        """
        api_config = self._get_api_config(api_name)

        if api_config.type != "graphql":
            raise APIExecutionError(
                f"API '{api_name}' is type '{api_config.type}', not 'graphql'"
            )

        if not api_config.url:
            raise APIExecutionError(f"API '{api_name}' has no URL configured")

        headers = self._build_headers(api_config)

        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        if operation_name:
            payload["operationName"] = operation_name

        try:
            response = self.client.post(
                api_config.url,
                headers=headers,
                json=payload,
            )
        except httpx.RequestError as e:
            raise APIExecutionError(f"Request to {api_config.url} failed: {e}")

        if response.status_code != 200:
            retryable, retry_hint = classify_http_error(response.status_code, response.text)
            raise APIExecutionError(
                f"GraphQL request failed with status {response.status_code}. {retry_hint}",
                status_code=response.status_code,
                response_body=response.text,
                retryable=retryable,
                retry_hint=retry_hint,
            )

        try:
            result = response.json()
        except json.JSONDecodeError:
            raise APIExecutionError(
                "Invalid JSON response from GraphQL API",
                response_body=response.text,
            )

        # Check for GraphQL errors per spec (https://graphql.org/learn/response/)
        # Response can have: data only (success), errors only (request error),
        # or both (partial response with field errors)
        data = result.get("data")
        errors = result.get("errors")

        if errors:
            error_messages = [e.get("message", str(e)) for e in errors]
            error_str = "; ".join(error_messages)

            # Check if data is empty/null (complete failure) vs has content (partial success)
            data_is_empty = (
                data is None
                or data == {}
                or (isinstance(data, dict) and all(
                    v is None or v == [] or v == {}
                    for v in data.values()
                ))
            )

            if data_is_empty:
                # Complete failure: no usable data returned, raise for retry
                raise APIExecutionError(
                    f"GraphQL errors (no data returned): {error_str}",
                    response_body=json.dumps(result),
                )
            else:
                # Partial success: some data returned with field errors
                # Log warning but return the data - let caller decide
                logging.warning(f"GraphQL partial response - errors present but data returned: {error_str}")

        return data if data is not None else {}

    def execute_rest(
        self,
        api_name: str,
        operation: str,
        path_params: Optional[dict[str, Any]] = None,
        query_params: Optional[dict[str, Any]] = None,
        body: Optional[dict[str, Any]] = None,
        method: Optional[str] = None,
    ) -> Any:
        """
        Execute a REST API call.

        Args:
            api_name: Name of the API (as configured)
            operation: Operation ID or path pattern (e.g., "/users/{userId}" or "getUser")
            path_params: Parameters to substitute in path (e.g., {"userId": "123"})
            query_params: Query string parameters
            body: Request body for POST/PUT/PATCH
            method: HTTP method (auto-detected from operation if not specified)

        Returns:
            Parsed JSON response (or raw text if not JSON)

        Raises:
            APIExecutionError: If the request fails
        """
        api_config = self._get_api_config(api_name)

        if api_config.type not in ("openapi", "rest"):
            raise APIExecutionError(
                f"API '{api_name}' is type '{api_config.type}', expected 'openapi' or 'rest'"
            )

        if not api_config.url:
            raise APIExecutionError(f"API '{api_name}' has no base URL configured")

        # Resolve operation to path and method
        path, http_method = self._resolve_operation(api_config, operation, method)

        # Substitute path parameters
        if path_params:
            for key, value in path_params.items():
                path = path.replace(f"{{{key}}}", str(value))

        # Build full URL
        url = urljoin(api_config.url.rstrip("/") + "/", path.lstrip("/"))

        # Add query parameters
        if query_params:
            url = f"{url}?{urlencode(query_params)}"

        headers = self._build_headers(api_config)

        try:
            if http_method.upper() == "GET":
                response = self.client.get(url, headers=headers)
            elif http_method.upper() == "POST":
                response = self.client.post(url, headers=headers, json=body)
            elif http_method.upper() == "PUT":
                response = self.client.put(url, headers=headers, json=body)
            elif http_method.upper() == "PATCH":
                response = self.client.patch(url, headers=headers, json=body)
            elif http_method.upper() == "DELETE":
                response = self.client.delete(url, headers=headers)
            else:
                raise APIExecutionError(f"Unsupported HTTP method: {http_method}")
        except httpx.RequestError as e:
            raise APIExecutionError(f"Request to {url} failed: {e}")

        if response.status_code >= 400:
            retryable, retry_hint = classify_http_error(response.status_code, response.text)
            raise APIExecutionError(
                f"REST request failed with status {response.status_code}. {retry_hint}",
                status_code=response.status_code,
                response_body=response.text,
                retryable=retryable,
                retry_hint=retry_hint,
            )

        # Parse response
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                return response.json()
            except json.JSONDecodeError:
                return response.text
        else:
            return response.text

    def _resolve_operation(
        self,
        api_config: APIConfig,
        operation: str,
        method: Optional[str],
    ) -> tuple[str, str]:
        """
        Resolve an operation to a path and HTTP method.

        Supports multiple formats:
        - "/path" - path directly, method defaults to GET
        - "GET /path" - method and path together
        - "operationId" - look up in OpenAPI spec
        """
        # Handle "METHOD /path" format (e.g., "GET /breeds")
        if " /" in operation:
            parts = operation.split(" ", 1)
            if len(parts) == 2 and parts[1].startswith("/"):
                extracted_method, path = parts
                return path, method or extracted_method.upper()

        # If it looks like a path, use it directly
        if operation.startswith("/"):
            return operation, method or "GET"

        # Look up in OpenAPI spec by operationId
        spec = self._get_openapi_spec(api_config)
        if not spec:
            raise APIExecutionError(
                f"Cannot resolve operation '{operation}' without OpenAPI spec"
            )

        # Search for operationId in paths
        for path, path_item in spec.get("paths", {}).items():
            for http_method, operation_def in path_item.items():
                if http_method in ("get", "post", "put", "patch", "delete"):
                    if operation_def.get("operationId") == operation:
                        return path, method or http_method.upper()

        raise APIExecutionError(
            f"Operation '{operation}' not found in OpenAPI spec"
        )

    def _get_openapi_spec(self, api_config: APIConfig) -> Optional[dict]:
        """Get OpenAPI spec from config (inline, file, or URL)."""
        if api_config.spec_inline:
            return api_config.spec_inline

        if api_config.spec_path:
            import yaml
            from pathlib import Path
            spec_path = Path(api_config.spec_path)
            if spec_path.exists():
                with open(spec_path) as f:
                    if spec_path.suffix in (".yaml", ".yml"):
                        return yaml.safe_load(f)
                    else:
                        return json.load(f)

        if api_config.spec_url:
            try:
                response = self.client.get(api_config.spec_url)
                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "")
                    if "yaml" in content_type or api_config.spec_url.endswith((".yaml", ".yml")):
                        import yaml
                        return yaml.safe_load(response.text)
                    else:
                        return response.json()
            except Exception:
                pass

        return None

    def list_available_apis(self) -> list[dict[str, Any]]:
        """List all configured APIs with their details."""
        result = []
        for name, api_config in self.config.apis.items():
            result.append({
                "name": name,
                "type": api_config.type,
                "url": api_config.url or api_config.spec_url or "",
                "description": api_config.description,
                "has_auth": bool(
                    api_config.auth_token or
                    api_config.auth_username or
                    api_config.api_key or
                    api_config.headers
                ),
            })
        return result

    def introspect_graphql(self, api_name: str) -> dict[str, Any]:
        """
        Introspect a GraphQL API to get its schema.

        Returns a simplified schema with queries, mutations, and their types.
        """
        api_config = self._get_api_config(api_name)

        if api_config.type != "graphql":
            raise APIExecutionError(f"API '{api_name}' is not a GraphQL API")

        # Check cache first
        if api_config._schema_cache is not None:
            return api_config._schema_cache

        # Full introspection query
        introspection_query = """
        query IntrospectionQuery {
          __schema {
            queryType { name }
            mutationType { name }
            types {
              name
              kind
              description
              fields {
                name
                description
                args {
                  name
                  description
                  type {
                    name
                    kind
                    ofType {
                      name
                      kind
                      ofType {
                        name
                        kind
                      }
                    }
                  }
                  defaultValue
                }
                type {
                  name
                  kind
                  ofType {
                    name
                    kind
                    ofType {
                      name
                      kind
                    }
                  }
                }
              }
              inputFields {
                name
                description
                type {
                  name
                  kind
                  ofType {
                    name
                    kind
                  }
                }
                defaultValue
              }
              enumValues {
                name
                description
              }
            }
          }
        }
        """

        result = self.execute_graphql(api_name, introspection_query)
        schema = result.get("__schema", {})

        # Cache the result
        api_config._schema_cache = schema
        return schema

    def get_schema_overview(self, api_name: str) -> dict[str, Any]:
        """
        Get a high-level overview of an API's schema.

        For GraphQL: Lists queries and mutations with their descriptions.
        For OpenAPI: Lists endpoints with their methods and descriptions.
        """
        api_config = self._get_api_config(api_name)

        if api_config.type == "graphql":
            schema = self.introspect_graphql(api_name)
            return self._summarize_graphql_schema(schema)
        else:
            spec = self._get_openapi_spec(api_config)
            return self._summarize_openapi_spec(spec) if spec else {}

    def _summarize_graphql_schema(self, schema: dict) -> dict[str, Any]:
        """Extract a summary of queries and mutations from GraphQL schema."""
        result = {"queries": [], "mutations": [], "types": {}}

        query_type_name = schema.get("queryType", {}).get("name", "Query")
        mutation_type_name = schema.get("mutationType", {}).get("name")

        types = {t["name"]: t for t in schema.get("types", []) if t.get("name")}

        # Get queries
        if query_type_name and query_type_name in types:
            query_type = types[query_type_name]
            for field in query_type.get("fields", []):
                if not field["name"].startswith("_"):
                    result["queries"].append({
                        "name": field["name"],
                        "description": field.get("description", ""),
                        "args": [a["name"] for a in field.get("args", [])],
                        "returns": self._format_type(field.get("type", {})),
                    })

        # Get mutations
        if mutation_type_name and mutation_type_name in types:
            mutation_type = types[mutation_type_name]
            for field in mutation_type.get("fields", []):
                if not field["name"].startswith("_"):
                    result["mutations"].append({
                        "name": field["name"],
                        "description": field.get("description", ""),
                        "args": [a["name"] for a in field.get("args", [])],
                        "returns": self._format_type(field.get("type", {})),
                    })

        return result

    def _summarize_openapi_spec(self, spec: dict) -> dict[str, Any]:
        """Extract a summary of endpoints from OpenAPI spec."""
        result = {"endpoints": [], "schemas": {}}

        for path, path_item in spec.get("paths", {}).items():
            for method, operation in path_item.items():
                if method in ("get", "post", "put", "patch", "delete"):
                    result["endpoints"].append({
                        "path": path,
                        "method": method.upper(),
                        "operationId": operation.get("operationId", ""),
                        "summary": operation.get("summary", ""),
                        "parameters": [
                            p.get("name") for p in operation.get("parameters", [])
                        ],
                    })

        return result

    def get_query_schema(self, api_name: str, query_name: str) -> dict[str, Any]:
        """
        Get detailed schema for a specific query/mutation (GraphQL) or endpoint (OpenAPI).

        For GraphQL: Returns arguments with their types, and return type fields.
        For OpenAPI: Returns parameters with their types, and response schema.
        """
        api_config = self._get_api_config(api_name)

        if api_config.type == "graphql":
            return self._get_graphql_query_detail(api_name, query_name)
        else:
            return self._get_openapi_endpoint_detail(api_config, query_name)

    def _get_graphql_query_detail(self, api_name: str, query_name: str) -> dict[str, Any]:
        """Get detailed schema for a GraphQL query or mutation."""
        schema = self.introspect_graphql(api_name)

        query_type_name = schema.get("queryType", {}).get("name", "Query")
        mutation_type_name = schema.get("mutationType", {}).get("name")

        types = {t["name"]: t for t in schema.get("types", []) if t.get("name")}

        # Search in queries first, then mutations
        for type_name in [query_type_name, mutation_type_name]:
            if type_name and type_name in types:
                for field in types[type_name].get("fields", []):
                    if field["name"] == query_name:
                        return self._build_query_detail(field, types)

        raise APIExecutionError(f"Query/mutation '{query_name}' not found in schema")

    def _build_query_detail(self, field: dict, types: dict) -> dict[str, Any]:
        """Build detailed query info including argument types and return fields."""
        result = {
            "name": field["name"],
            "description": field.get("description", ""),
            "args": [],
            "returns": self._format_type(field.get("type", {})),
            "return_fields": [],
        }

        # Build detailed argument info
        for arg in field.get("args", []):
            arg_info = {
                "name": arg["name"],
                "description": arg.get("description", ""),
                "type": self._format_type(arg.get("type", {})),
                "required": self._is_required(arg.get("type", {})),
                "default": arg.get("defaultValue"),
            }

            # If it's an input type, expand its fields
            type_name = self._get_base_type_name(arg.get("type", {}))
            if type_name and type_name in types:
                input_type = types[type_name]
                if input_type.get("kind") == "INPUT_OBJECT":
                    arg_info["fields"] = [
                        {
                            "name": f["name"],
                            "type": self._format_type(f.get("type", {})),
                            "description": f.get("description", ""),
                        }
                        for f in input_type.get("inputFields", [])
                    ]

            result["args"].append(arg_info)

        # Get return type fields
        return_type_name = self._get_base_type_name(field.get("type", {}))
        if return_type_name and return_type_name in types:
            return_type = types[return_type_name]
            if return_type.get("fields"):
                result["return_fields"] = [
                    {
                        "name": f["name"],
                        "type": self._format_type(f.get("type", {})),
                        "description": f.get("description", ""),
                    }
                    for f in return_type.get("fields", [])
                    if not f["name"].startswith("_")
                ]

        return result

    def _get_openapi_endpoint_detail(self, api_config: APIConfig, operation: str) -> dict[str, Any]:
        """Get detailed schema for an OpenAPI endpoint."""
        spec = self._get_openapi_spec(api_config)
        if not spec:
            raise APIExecutionError("No OpenAPI spec available")

        # Search by operationId or path
        for path, path_item in spec.get("paths", {}).items():
            for method, op in path_item.items():
                if method in ("get", "post", "put", "patch", "delete"):
                    if op.get("operationId") == operation or path == operation:
                        return {
                            "path": path,
                            "method": method.upper(),
                            "operationId": op.get("operationId", ""),
                            "summary": op.get("summary", ""),
                            "description": op.get("description", ""),
                            "parameters": [
                                {
                                    "name": p.get("name"),
                                    "in": p.get("in"),
                                    "required": p.get("required", False),
                                    "type": p.get("schema", {}).get("type", "string"),
                                    "description": p.get("description", ""),
                                }
                                for p in op.get("parameters", [])
                            ],
                            "requestBody": op.get("requestBody", {}),
                            "responses": op.get("responses", {}),
                        }

        raise APIExecutionError(f"Operation '{operation}' not found in OpenAPI spec")

    def _format_type(self, type_info: dict) -> str:
        """Format a GraphQL type as a string (e.g., '[Country!]!')."""
        if not type_info:
            return "unknown"

        kind = type_info.get("kind")
        name = type_info.get("name")
        of_type = type_info.get("ofType")

        if kind == "NON_NULL":
            return f"{self._format_type(of_type)}!"
        elif kind == "LIST":
            return f"[{self._format_type(of_type)}]"
        elif name:
            return name
        else:
            return "unknown"

    def _get_base_type_name(self, type_info: dict) -> Optional[str]:
        """Get the base type name (unwrapping LIST and NON_NULL)."""
        if not type_info:
            return None

        name = type_info.get("name")
        if name:
            return name

        of_type = type_info.get("ofType")
        if of_type:
            return self._get_base_type_name(of_type)

        return None

    def _is_required(self, type_info: dict) -> bool:
        """Check if a type is non-nullable (required)."""
        return type_info.get("kind") == "NON_NULL"

    def execute(
        self,
        api_name: str,
        query_or_operation: str,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Execute a query against an API (auto-detects GraphQL vs REST).

        Args:
            api_name: Name of the API
            query_or_operation: GraphQL query string or REST operation/path
            params: Variables (GraphQL) or parameters (REST)

        Returns:
            API response data
        """
        api_config = self._get_api_config(api_name)

        if api_config.type == "graphql":
            return self.execute_graphql(api_name, query_or_operation, variables=params)
        else:
            # For REST, try to split params into path and query params
            path_params = {}
            query_params = {}
            body = None

            if params:
                # Simple heuristic: params with names matching {param} in path go to path_params
                for key, value in params.items():
                    if f"{{{key}}}" in query_or_operation:
                        path_params[key] = value
                    elif isinstance(value, dict):
                        body = value
                    else:
                        query_params[key] = value

            return self.execute_rest(
                api_name,
                query_or_operation,
                path_params=path_params or None,
                query_params=query_params or None,
                body=body,
            )
