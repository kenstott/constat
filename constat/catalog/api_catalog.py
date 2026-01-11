"""API operation catalog with vector search for GraphQL/REST discovery.

Provides the same pattern as SchemaManager but for API operations:
- Introspect GraphQL schemas or OpenAPI specs
- Cache operation metadata with embeddings
- Semantic search for relevant operations
- Token-optimized overview for system prompt
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer


# Standard GraphQL introspection query
INTROSPECTION_QUERY = """
query IntrospectionQuery {
  __schema {
    queryType { name }
    mutationType { name }
    subscriptionType { name }
    types {
      kind
      name
      description
      fields(includeDeprecated: true) {
        name
        description
        args {
          name
          description
          type {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
                ofType {
                  kind
                  name
                }
              }
            }
          }
          defaultValue
        }
        type {
          kind
          name
          ofType {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
              }
            }
          }
        }
        isDeprecated
        deprecationReason
      }
    }
  }
}
"""


class OperationType(Enum):
    """Type of API operation."""
    QUERY = "query"           # Read operations
    MUTATION = "mutation"     # Write operations
    SUBSCRIPTION = "subscription"  # Real-time updates


class ArgumentType(Enum):
    """Argument requirement level."""
    REQUIRED = "required"
    OPTIONAL = "optional"


@dataclass
class OperationArgument:
    """Metadata for an operation argument."""
    name: str
    type: str  # e.g., "String!", "Int", "[ID!]!"
    description: str = ""
    default_value: Optional[Any] = None
    requirement: ArgumentType = ArgumentType.REQUIRED

    def to_signature(self) -> str:
        """Generate argument signature for display."""
        if self.requirement == ArgumentType.OPTIONAL:
            return f"{self.name}: {self.type} = {self.default_value}"
        return f"{self.name}: {self.type}"


@dataclass
class OperationField:
    """A field in the operation's return type."""
    name: str
    type: str
    description: str = ""
    is_list: bool = False
    nested_fields: list["OperationField"] = field(default_factory=list)


@dataclass
class OperationMetadata:
    """Full metadata for an API operation."""
    name: str
    operation_type: OperationType
    description: str
    arguments: list[OperationArgument] = field(default_factory=list)
    return_type: str = ""
    return_fields: list[OperationField] = field(default_factory=list)

    # Rich metadata for semantic search
    use_cases: list[str] = field(default_factory=list)  # When to use this
    examples: list[str] = field(default_factory=list)   # Example queries
    related_operations: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)  # e.g., ["session", "artifact", "realtime"]

    # Deprecation info
    deprecated: bool = False
    deprecation_reason: str = ""

    @property
    def full_name(self) -> str:
        """Full operation name with type prefix."""
        return f"{self.operation_type.value}.{self.name}"

    def to_dict(self) -> dict:
        """Convert to dict for LLM tool response."""
        return {
            "name": self.name,
            "type": self.operation_type.value,
            "description": self.description,
            "arguments": [
                {
                    "name": arg.name,
                    "type": arg.type,
                    "description": arg.description,
                    "required": arg.requirement == ArgumentType.REQUIRED,
                }
                for arg in self.arguments
            ],
            "return_type": self.return_type,
            "use_cases": self.use_cases,
            "tags": self.tags,
        }

    def to_embedding_text(self) -> str:
        """
        Generate rich text representation for embedding.

        Includes description, use cases, argument semantics, and examples
        to enable high-quality semantic search.
        """
        parts = [
            f"Operation: {self.name} ({self.operation_type.value})",
            f"Description: {self.description}",
        ]

        # Use cases are highly valuable for semantic search
        if self.use_cases:
            parts.append(f"Use cases: {'; '.join(self.use_cases)}")

        # Arguments provide context about what data is needed
        if self.arguments:
            arg_descriptions = [
                f"{arg.name}: {arg.description}" if arg.description else arg.name
                for arg in self.arguments
            ]
            parts.append(f"Arguments: {', '.join(arg_descriptions)}")

        # Return type tells what you get back
        if self.return_type:
            parts.append(f"Returns: {self.return_type}")

        # Examples help with semantic matching
        if self.examples:
            parts.append(f"Examples: {'; '.join(self.examples[:3])}")

        # Tags for categorical matching
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")

        # Related operations for graph-like discovery
        if self.related_operations:
            parts.append(f"Related: {', '.join(self.related_operations)}")

        return "\n".join(parts)

    def to_signature(self) -> str:
        """Generate a compact signature for display."""
        args = ", ".join(arg.to_signature() for arg in self.arguments)
        return f"{self.name}({args}): {self.return_type}"


@dataclass
class OperationMatch:
    """Result from vector search."""
    operation: str
    operation_type: OperationType
    relevance: float
    summary: str
    use_cases: list[str]


class APICatalog:
    """
    Catalog of API operations with semantic search.

    Similar to SchemaManager but for API operations:
    1. Register operations from GraphQL schema or manually
    2. Build vector index for semantic search
    3. Generate token-optimized overview for system prompt
    4. Provide tools for LLM to discover operations
    """

    # Same embedding model as SchemaManager for consistency
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(self):
        self.operations: dict[str, OperationMetadata] = {}  # key: "type.name"

        # Vector index components
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_keys: list[str] = []
        self._model: Optional[SentenceTransformer] = None

        # Cached overview
        self._overview: Optional[str] = None

    def register_operation(self, operation: OperationMetadata) -> None:
        """Register an API operation."""
        self.operations[operation.full_name] = operation
        # Invalidate caches
        self._embeddings = None
        self._overview = None

    def register_operations(self, operations: list[OperationMetadata]) -> None:
        """Register multiple operations at once."""
        for op in operations:
            self.operations[op.full_name] = op
        # Invalidate caches
        self._embeddings = None
        self._overview = None

    def build_index(self) -> None:
        """Build vector embeddings for all operations."""
        if not self.operations:
            return

        # Load embedding model (lazy, shared with SchemaManager)
        if self._model is None:
            self._model = SentenceTransformer(self.EMBEDDING_MODEL)

        # Generate texts for embedding
        texts = []
        self._embedding_keys = []

        for full_name, operation in self.operations.items():
            texts.append(operation.to_embedding_text())
            self._embedding_keys.append(full_name)

        # Generate embeddings
        self._embeddings = self._model.encode(texts, convert_to_numpy=True)

    def find_relevant_operations(
        self,
        query: str,
        top_k: int = 5,
        operation_type: Optional[OperationType] = None,
        tags: Optional[list[str]] = None,
    ) -> list[OperationMatch]:
        """
        Find operations relevant to a natural language query.

        Args:
            query: Natural language description of what's needed
            top_k: Maximum number of results
            operation_type: Filter by operation type (query/mutation/subscription)
            tags: Filter by tags

        Returns:
            List of matching operations with relevance scores
        """
        if self._model is None or self._embeddings is None:
            self.build_index()

        if self._embeddings is None:
            return []

        # Embed the query
        query_embedding = self._model.encode([query], convert_to_numpy=True)

        # Compute cosine similarity
        similarities = np.dot(self._embeddings, query_embedding.T).flatten()

        # Get all indices sorted by similarity
        sorted_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in sorted_indices:
            if len(results) >= top_k:
                break

            full_name = self._embedding_keys[idx]
            operation = self.operations[full_name]
            relevance = float(similarities[idx])

            # Apply filters
            if operation_type and operation.operation_type != operation_type:
                continue
            if tags and not any(t in operation.tags for t in tags):
                continue

            # Generate summary
            args_preview = ", ".join(a.name for a in operation.arguments[:3])
            if len(operation.arguments) > 3:
                args_preview += ", ..."
            summary = f"{operation.name}({args_preview}) → {operation.return_type}"

            results.append(OperationMatch(
                operation=operation.name,
                operation_type=operation.operation_type,
                relevance=round(relevance, 3),
                summary=summary,
                use_cases=operation.use_cases[:2],  # Top 2 use cases
            ))

        return results

    def get_operation(self, name: str) -> Optional[OperationMetadata]:
        """
        Get full metadata for an operation.

        Args:
            name: Operation name (with or without type prefix)

        Returns:
            OperationMetadata or None if not found
        """
        # Try exact match first
        if name in self.operations:
            return self.operations[name]

        # Try with type prefixes
        for prefix in ["query.", "mutation.", "subscription."]:
            if prefix + name in self.operations:
                return self.operations[prefix + name]

        # Try finding by name only (if unambiguous)
        matches = [
            op for op in self.operations.values()
            if op.name == name
        ]
        if len(matches) == 1:
            return matches[0]

        return None

    def get_overview(self) -> str:
        """
        Return token-optimized API overview for system prompt.

        Groups operations by type and provides compact signatures.
        """
        if self._overview is not None:
            return self._overview

        if not self.operations:
            self._overview = "No API operations registered."
            return self._overview

        lines = ["Available API Operations:"]

        # Group by operation type
        by_type: dict[OperationType, list[OperationMetadata]] = {}
        for op in self.operations.values():
            by_type.setdefault(op.operation_type, []).append(op)

        for op_type in [OperationType.QUERY, OperationType.MUTATION, OperationType.SUBSCRIPTION]:
            ops = by_type.get(op_type, [])
            if not ops:
                continue

            type_names = {
                OperationType.QUERY: "Queries",
                OperationType.MUTATION: "Mutations",
                OperationType.SUBSCRIPTION: "Subscriptions",
            }
            lines.append(f"\n## {type_names[op_type]}")
            for op in sorted(ops, key=lambda x: x.name):
                # Compact signature with description
                args = ", ".join(f"{a.name}: {a.type}" for a in op.arguments[:2])
                if len(op.arguments) > 2:
                    args += ", ..."
                sig = f"  {op.name}({args}) → {op.return_type}"

                # Add brief description if it fits
                if op.description and len(op.description) < 60:
                    sig += f"  # {op.description}"

                lines.append(sig)

        self._overview = "\n".join(lines)
        return self._overview

    def list_operations(
        self,
        operation_type: Optional[OperationType] = None,
        tags: Optional[list[str]] = None,
    ) -> list[str]:
        """List all operation names, optionally filtered."""
        result = []
        for op in self.operations.values():
            if operation_type and op.operation_type != operation_type:
                continue
            if tags and not any(t in op.tags for t in tags):
                continue
            result.append(op.full_name)
        return sorted(result)

    def get_tools(self) -> list[dict]:
        """
        Get tool definitions for LLM to discover API operations.

        Returns tools in Anthropic format.
        """
        return [
            {
                "name": "find_api_operations",
                "description": "Search for API operations relevant to a task using semantic similarity. Returns operations that can fetch data, modify state, or subscribe to updates.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of what you need to do"
                        },
                        "operation_type": {
                            "type": "string",
                            "enum": ["query", "mutation", "subscription"],
                            "description": "Filter by operation type (optional)"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum results to return (default 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_api_operation",
                "description": "Get detailed information about a specific API operation including arguments, return type, and usage examples.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Operation name (e.g., 'createSession', 'query.session')"
                        }
                    },
                    "required": ["name"]
                }
            }
        ]

    def get_tool_handlers(self) -> dict:
        """Get tool handler functions for LLM discovery."""
        return {
            "find_api_operations": lambda query, operation_type=None, top_k=5: [
                m.__dict__ for m in self.find_relevant_operations(
                    query,
                    top_k=top_k,
                    operation_type=OperationType(operation_type) if operation_type else None,
                )
            ],
            "get_api_operation": lambda name: (
                self.get_operation(name).to_dict()
                if self.get_operation(name) else {"error": f"Operation not found: {name}"}
            ),
        }


def introspect_graphql_endpoint(
    url: str,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 30.0,
) -> APICatalog:
    """
    Introspect a GraphQL endpoint and create an APICatalog.

    Args:
        url: GraphQL endpoint URL
        headers: Optional headers (e.g., for authentication)
        timeout: Request timeout in seconds

    Returns:
        APICatalog populated with operations from the schema

    Example:
        catalog = introspect_graphql_endpoint("https://countries.trevorblades.com/graphql")
        results = catalog.find_relevant_operations("find countries in Europe")
    """
    # Run introspection query
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            url,
            json={"query": INTROSPECTION_QUERY},
            headers=headers or {},
        )
        response.raise_for_status()
        data = response.json()

    if "errors" in data:
        raise ValueError(f"GraphQL introspection failed: {data['errors']}")

    schema = data["data"]["__schema"]

    # Get root type names (handle None values from API)
    query_type = schema.get("queryType") or {}
    mutation_type = schema.get("mutationType") or {}
    subscription_type = schema.get("subscriptionType") or {}

    query_type_name = query_type.get("name", "Query")
    mutation_type_name = mutation_type.get("name")
    subscription_type_name = subscription_type.get("name")

    # Build type lookup
    types_by_name = {t["name"]: t for t in schema["types"]}

    catalog = APICatalog()

    def parse_type_ref(type_ref: dict) -> str:
        """Convert GraphQL type reference to string like 'String!' or '[Country!]!'"""
        if type_ref is None:
            return "Unknown"

        kind = type_ref.get("kind")
        name = type_ref.get("name")
        of_type = type_ref.get("ofType")

        if kind == "NON_NULL":
            inner = parse_type_ref(of_type)
            return f"{inner}!"
        elif kind == "LIST":
            inner = parse_type_ref(of_type)
            return f"[{inner}]"
        elif name:
            return name
        else:
            return "Unknown"

    def extract_operations(type_name: str, op_type: OperationType) -> list[OperationMetadata]:
        """Extract operations from a root type."""
        if type_name is None or type_name not in types_by_name:
            return []

        root_type = types_by_name[type_name]
        operations = []

        for field_data in root_type.get("fields") or []:
            # Parse arguments
            arguments = []
            for arg in field_data.get("args") or []:
                arg_type = parse_type_ref(arg.get("type"))
                is_required = arg_type.endswith("!")

                arguments.append(OperationArgument(
                    name=arg["name"],
                    type=arg_type,
                    description=arg.get("description") or "",
                    default_value=arg.get("defaultValue"),
                    requirement=ArgumentType.REQUIRED if is_required else ArgumentType.OPTIONAL,
                ))

            # Parse return type
            return_type = parse_type_ref(field_data.get("type"))

            # Create operation
            operations.append(OperationMetadata(
                name=field_data["name"],
                operation_type=op_type,
                description=field_data.get("description") or f"{op_type.value} {field_data['name']}",
                arguments=arguments,
                return_type=return_type,
                deprecated=field_data.get("isDeprecated", False),
                deprecation_reason=field_data.get("deprecationReason") or "",
                # Generate use cases from description
                use_cases=_generate_use_cases(field_data["name"], field_data.get("description")),
                # Add tags based on operation characteristics
                tags=_generate_tags(field_data["name"], return_type, arguments),
            ))

        return operations

    # Extract all operations
    query_ops = extract_operations(query_type_name, OperationType.QUERY)
    mutation_ops = extract_operations(mutation_type_name, OperationType.MUTATION) if mutation_type_name else []
    subscription_ops = extract_operations(subscription_type_name, OperationType.SUBSCRIPTION) if subscription_type_name else []

    catalog.register_operations(query_ops + mutation_ops + subscription_ops)
    catalog.build_index()

    return catalog


def _generate_use_cases(name: str, description: Optional[str]) -> list[str]:
    """Generate use cases from operation name and description."""
    use_cases = []

    # Common patterns
    name_lower = name.lower()
    if name_lower.startswith("get") or name_lower.startswith("find") or name_lower.startswith("list"):
        use_cases.append(f"Retrieve {name_lower.replace('get', '').replace('find', '').replace('list', '').strip()}")
    elif name_lower.startswith("create") or name_lower.startswith("add"):
        use_cases.append(f"Create a new {name_lower.replace('create', '').replace('add', '').strip()}")
    elif name_lower.startswith("update") or name_lower.startswith("edit"):
        use_cases.append(f"Modify existing {name_lower.replace('update', '').replace('edit', '').strip()}")
    elif name_lower.startswith("delete") or name_lower.startswith("remove"):
        use_cases.append(f"Remove {name_lower.replace('delete', '').replace('remove', '').strip()}")

    # Add description-based use case
    if description:
        use_cases.append(description[:100])

    return use_cases[:3]


def _generate_tags(name: str, return_type: str, arguments: list[OperationArgument]) -> list[str]:
    """Generate tags for an operation."""
    tags = []

    name_lower = name.lower()

    # Operation type tags
    if any(x in name_lower for x in ["get", "find", "list", "search", "query"]):
        tags.append("read")
    if any(x in name_lower for x in ["create", "add", "insert"]):
        tags.append("create")
    if any(x in name_lower for x in ["update", "edit", "modify", "set"]):
        tags.append("update")
    if any(x in name_lower for x in ["delete", "remove"]):
        tags.append("delete")

    # List vs single
    if return_type.startswith("["):
        tags.append("list")
    else:
        tags.append("single")

    # Has pagination
    arg_names = [a.name.lower() for a in arguments]
    if any(x in arg_names for x in ["limit", "offset", "first", "last", "after", "before", "cursor"]):
        tags.append("paginated")

    # Has filters
    if any(x in arg_names for x in ["filter", "where", "search", "query"]):
        tags.append("filterable")

    return tags


def introspect_openapi_spec(
    spec_url: Optional[str] = None,
    spec_path: Optional[str] = None,
    spec_inline: Optional[dict] = None,
    base_url: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 30.0,
) -> APICatalog:
    """
    Parse an OpenAPI/Swagger spec and create an APICatalog.

    Supports OpenAPI 3.x and Swagger 2.0 specifications in JSON or YAML format.

    Args:
        spec_url: URL to download the OpenAPI spec from
        spec_path: Local file path to the OpenAPI spec
        spec_inline: Inline OpenAPI spec as a dict (embedded in config)
        base_url: Override the base URL from the spec (optional)
        headers: Headers to use when downloading spec (e.g., for auth)
        timeout: Request timeout in seconds

    Returns:
        APICatalog populated with operations from the spec

    Example:
        # From URL
        catalog = introspect_openapi_spec(
            spec_url="https://petstore.swagger.io/v2/swagger.json"
        )

        # From local file
        catalog = introspect_openapi_spec(
            spec_path="./specs/my-api.yaml"
        )

        # From inline spec
        catalog = introspect_openapi_spec(
            spec_inline={
                "openapi": "3.0.0",
                "info": {"title": "My API", "version": "1.0"},
                "paths": {
                    "/users/{id}": {
                        "get": {
                            "operationId": "getUser",
                            "parameters": [
                                {"name": "id", "in": "path", "required": True}
                            ]
                        }
                    }
                }
            }
        )

        # Search for operations
        results = catalog.find_relevant_operations("create a new pet")
    """
    import yaml
    from pathlib import Path

    if not spec_url and not spec_path and not spec_inline:
        raise ValueError("One of spec_url, spec_path, or spec_inline must be provided")

    # Load the spec
    if spec_inline:
        # Use inline spec directly
        spec = spec_inline
    elif spec_url:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(spec_url, headers=headers or {})
            response.raise_for_status()
            content = response.text

            # Parse as JSON or YAML
            try:
                spec = json.loads(content)
            except json.JSONDecodeError:
                spec = yaml.safe_load(content)
    else:
        spec_file = Path(spec_path)
        if not spec_file.exists():
            raise FileNotFoundError(f"OpenAPI spec not found: {spec_path}")

        with open(spec_file) as f:
            content = f.read()

        if spec_file.suffix in (".yaml", ".yml"):
            spec = yaml.safe_load(content)
        else:
            spec = json.loads(content)

    # Determine OpenAPI version
    openapi_version = spec.get("openapi", spec.get("swagger", "2.0"))
    is_openapi3 = openapi_version.startswith("3")

    # Get base URL
    if base_url:
        api_base_url = base_url
    elif is_openapi3:
        servers = spec.get("servers", [])
        api_base_url = servers[0]["url"] if servers else ""
    else:
        # Swagger 2.0
        host = spec.get("host", "")
        base_path = spec.get("basePath", "")
        schemes = spec.get("schemes", ["https"])
        api_base_url = f"{schemes[0]}://{host}{base_path}"

    catalog = APICatalog()
    operations = []

    # Parse paths
    paths = spec.get("paths", {})
    for path, path_item in paths.items():
        for method in ["get", "post", "put", "patch", "delete", "head", "options"]:
            if method not in path_item:
                continue

            operation_data = path_item[method]
            operation_id = operation_data.get("operationId", f"{method}_{path.replace('/', '_')}")

            # Determine operation type
            if method == "get":
                op_type = OperationType.QUERY
            else:
                op_type = OperationType.MUTATION

            # Parse parameters
            arguments = []
            all_params = path_item.get("parameters", []) + operation_data.get("parameters", [])

            for param in all_params:
                # Handle $ref
                if "$ref" in param:
                    param = _resolve_openapi_ref(spec, param["$ref"])

                param_name = param.get("name", "unknown")
                param_in = param.get("in", "query")  # path, query, header, cookie
                required = param.get("required", param_in == "path")

                # Get type
                if is_openapi3:
                    schema = param.get("schema", {})
                    param_type = _openapi_type_to_string(schema)
                else:
                    param_type = param.get("type", "string")

                arguments.append(OperationArgument(
                    name=param_name,
                    type=f"{param_type}{'!' if required else ''}",
                    description=param.get("description", ""),
                    requirement=ArgumentType.REQUIRED if required else ArgumentType.OPTIONAL,
                ))

            # Parse request body (OpenAPI 3.x)
            if is_openapi3 and "requestBody" in operation_data:
                request_body = operation_data["requestBody"]
                content = request_body.get("content", {})
                required = request_body.get("required", False)

                # Get the first content type schema
                for content_type, media_type in content.items():
                    schema = media_type.get("schema", {})
                    body_type = _openapi_type_to_string(schema)
                    arguments.append(OperationArgument(
                        name="body",
                        type=f"{body_type}{'!' if required else ''}",
                        description=request_body.get("description", "Request body"),
                        requirement=ArgumentType.REQUIRED if required else ArgumentType.OPTIONAL,
                    ))
                    break  # Just use first content type

            # Parse response type
            responses = operation_data.get("responses", {})
            return_type = "void"
            for status_code in ["200", "201", "default"]:
                if status_code in responses:
                    response = responses[status_code]
                    if is_openapi3:
                        content = response.get("content", {})
                        for content_type, media_type in content.items():
                            schema = media_type.get("schema", {})
                            return_type = _openapi_type_to_string(schema)
                            break
                    else:
                        schema = response.get("schema", {})
                        return_type = _openapi_type_to_string(schema)
                    break

            # Build description
            summary = operation_data.get("summary", "")
            description = operation_data.get("description", summary)
            full_description = f"{method.upper()} {path}"
            if description:
                full_description = f"{description} ({method.upper()} {path})"

            # Generate tags
            op_tags = operation_data.get("tags", [])
            generated_tags = _generate_openapi_tags(method, path, operation_id)
            all_tags = list(set(op_tags + generated_tags))

            operations.append(OperationMetadata(
                name=operation_id,
                operation_type=op_type,
                description=full_description,
                arguments=arguments,
                return_type=return_type,
                use_cases=_generate_use_cases(operation_id, description),
                tags=all_tags,
                deprecated=operation_data.get("deprecated", False),
            ))

    catalog.register_operations(operations)
    catalog.build_index()

    return catalog


def _resolve_openapi_ref(spec: dict, ref: str) -> dict:
    """Resolve a $ref pointer in an OpenAPI spec."""
    if not ref.startswith("#/"):
        return {}

    parts = ref[2:].split("/")
    result = spec
    for part in parts:
        result = result.get(part, {})
    return result


def _openapi_type_to_string(schema: dict) -> str:
    """Convert OpenAPI schema to a type string."""
    if not schema:
        return "object"

    # Handle $ref
    if "$ref" in schema:
        ref = schema["$ref"]
        # Extract type name from ref like "#/components/schemas/Pet"
        return ref.split("/")[-1]

    schema_type = schema.get("type", "object")

    if schema_type == "array":
        items = schema.get("items", {})
        item_type = _openapi_type_to_string(items)
        return f"[{item_type}]"

    if schema_type == "object":
        # Check for additionalProperties (map type)
        if "additionalProperties" in schema:
            value_type = _openapi_type_to_string(schema["additionalProperties"])
            return f"Map<string, {value_type}>"
        # Check for title or return generic object
        return schema.get("title", "object")

    # Simple types
    type_map = {
        "string": "string",
        "integer": "int",
        "number": "float",
        "boolean": "boolean",
        "null": "null",
    }
    return type_map.get(schema_type, schema_type)


def _generate_openapi_tags(method: str, path: str, operation_id: str) -> list[str]:
    """Generate tags for an OpenAPI operation."""
    tags = []

    # Method-based tags
    if method == "get":
        tags.append("read")
        if "{" not in path:
            tags.append("list")
        else:
            tags.append("single")
    elif method == "post":
        tags.append("create")
    elif method in ("put", "patch"):
        tags.append("update")
    elif method == "delete":
        tags.append("delete")

    # Path-based tags
    path_lower = path.lower()
    if "search" in path_lower or "query" in path_lower:
        tags.append("search")
    if "upload" in path_lower:
        tags.append("upload")
    if "download" in path_lower or "export" in path_lower:
        tags.append("download")

    return tags


def create_constat_api_catalog() -> APICatalog:
    """
    Create APICatalog with Constat's GraphQL operations.

    This defines all the operations available in the GraphQL API
    with rich metadata for semantic search.
    """
    catalog = APICatalog()

    # ========== QUERIES ==========

    catalog.register_operations([
        OperationMetadata(
            name="session",
            operation_type=OperationType.QUERY,
            description="Get a session by ID with its full state including plan, steps, artifacts, and tables",
            arguments=[
                OperationArgument("id", "ID!", "The session ID to retrieve"),
            ],
            return_type="Session",
            use_cases=[
                "Retrieve a previous session to continue work",
                "Load session state for UI restoration",
                "Check the status of a running analysis",
                "Get artifacts and results from a completed session",
            ],
            examples=[
                "Get session abc123 to see its results",
                "Load my previous analysis session",
                "Check if the revenue analysis is complete",
            ],
            tags=["session", "read", "state"],
        ),

        OperationMetadata(
            name="sessions",
            operation_type=OperationType.QUERY,
            description="List sessions with optional filtering by status, pagination support",
            arguments=[
                OperationArgument("status", "SessionStatus", "Filter by session status", requirement=ArgumentType.OPTIONAL),
                OperationArgument("limit", "Int", "Maximum sessions to return", default_value=20, requirement=ArgumentType.OPTIONAL),
                OperationArgument("cursor", "String", "Pagination cursor", requirement=ArgumentType.OPTIONAL),
            ],
            return_type="[SessionSummary!]!",
            use_cases=[
                "List all recent sessions",
                "Find completed sessions",
                "Browse session history",
                "Find failed sessions to retry",
            ],
            examples=[
                "Show my last 10 sessions",
                "List all completed analyses",
                "Find sessions that failed",
            ],
            tags=["session", "list", "history"],
        ),

        OperationMetadata(
            name="databases",
            operation_type=OperationType.QUERY,
            description="List all configured databases with their schemas and connection status",
            arguments=[],
            return_type="[Database!]!",
            use_cases=[
                "See what databases are available",
                "Check database connectivity",
                "Get schema information for planning",
            ],
            tags=["database", "schema", "config"],
        ),

        OperationMetadata(
            name="tables",
            operation_type=OperationType.QUERY,
            description="Search for tables across databases using semantic similarity",
            arguments=[
                OperationArgument("query", "String!", "Natural language search query"),
                OperationArgument("database", "String", "Filter to specific database", requirement=ArgumentType.OPTIONAL),
                OperationArgument("topK", "Int", "Maximum results", default_value=5, requirement=ArgumentType.OPTIONAL),
            ],
            return_type="[TableMatch!]!",
            use_cases=[
                "Find tables containing customer data",
                "Search for revenue-related tables",
                "Discover tables with order information",
            ],
            examples=[
                "Find tables with customer information",
                "Search for sales data tables",
            ],
            tags=["database", "schema", "search", "tables"],
        ),

        OperationMetadata(
            name="artifacts",
            operation_type=OperationType.QUERY,
            description="Get artifacts from a session, optionally filtered by type",
            arguments=[
                OperationArgument("sessionId", "ID!", "Session to get artifacts from"),
                OperationArgument("type", "ArtifactType", "Filter by artifact type (html, chart, svg, etc.)", requirement=ArgumentType.OPTIONAL),
                OperationArgument("stepNumber", "Int", "Filter by step number", requirement=ArgumentType.OPTIONAL),
            ],
            return_type="[Artifact!]!",
            use_cases=[
                "Get all charts from an analysis",
                "Retrieve HTML reports",
                "Get diagrams generated during analysis",
                "List images created by a session",
            ],
            examples=[
                "Get all charts from session xyz",
                "Show HTML artifacts from step 3",
            ],
            tags=["artifact", "output", "visualization"],
        ),
    ])

    # ========== MUTATIONS ==========

    catalog.register_operations([
        OperationMetadata(
            name="createSession",
            operation_type=OperationType.MUTATION,
            description="Create a new analysis session with a natural language problem statement",
            arguments=[
                OperationArgument("problem", "String!", "The problem to solve in natural language"),
                OperationArgument("config", "SessionConfigInput", "Optional configuration overrides", requirement=ArgumentType.OPTIONAL),
            ],
            return_type="Session!",
            use_cases=[
                "Start a new data analysis",
                "Begin a new investigation",
                "Create a session to answer a business question",
            ],
            examples=[
                "Create session to analyze revenue by region",
                "Start analysis of customer churn",
                "Begin investigation of sales trends",
            ],
            tags=["session", "create", "start"],
            related_operations=["executeStep", "followUp"],
        ),

        OperationMetadata(
            name="executeStep",
            operation_type=OperationType.MUTATION,
            description="Execute the next step in a session's plan, generating and running code",
            arguments=[
                OperationArgument("sessionId", "ID!", "Session ID"),
                OperationArgument("stepNumber", "Int", "Specific step to execute (default: next pending)", requirement=ArgumentType.OPTIONAL),
            ],
            return_type="StepResult!",
            use_cases=[
                "Run the next step in the analysis",
                "Execute a specific step",
                "Continue session execution",
            ],
            examples=[
                "Execute step 2 of session abc",
                "Run the next step",
            ],
            tags=["session", "execute", "step"],
            related_operations=["createSession", "retryStep"],
        ),

        OperationMetadata(
            name="executeAll",
            operation_type=OperationType.MUTATION,
            description="Execute all remaining steps in a session's plan",
            arguments=[
                OperationArgument("sessionId", "ID!", "Session ID"),
                OperationArgument("stopOnError", "Boolean", "Stop execution if a step fails", default_value=True, requirement=ArgumentType.OPTIONAL),
            ],
            return_type="SessionResult!",
            use_cases=[
                "Run entire analysis to completion",
                "Execute all steps automatically",
                "Complete the session",
            ],
            tags=["session", "execute", "batch"],
        ),

        OperationMetadata(
            name="retryStep",
            operation_type=OperationType.MUTATION,
            description="Retry a failed step, optionally with modified code or feedback",
            arguments=[
                OperationArgument("sessionId", "ID!", "Session ID"),
                OperationArgument("stepNumber", "Int!", "Step number to retry"),
                OperationArgument("feedback", "String", "Guidance for the retry", requirement=ArgumentType.OPTIONAL),
            ],
            return_type="StepResult!",
            use_cases=[
                "Retry a step that failed",
                "Re-run with different approach",
                "Fix an error and continue",
            ],
            tags=["session", "retry", "error"],
        ),

        OperationMetadata(
            name="followUp",
            operation_type=OperationType.MUTATION,
            description="Ask a follow-up question that builds on the session's existing context and results",
            arguments=[
                OperationArgument("sessionId", "ID!", "Session ID to continue"),
                OperationArgument("question", "String!", "Follow-up question or request"),
            ],
            return_type="Session!",
            use_cases=[
                "Ask additional questions about results",
                "Drill down into specific findings",
                "Request additional analysis",
                "Extend previous work with new questions",
            ],
            examples=[
                "Now break this down by quarter",
                "Which region had the highest growth?",
                "Create a chart of these results",
            ],
            tags=["session", "followup", "continuation"],
            related_operations=["createSession"],
        ),

        OperationMetadata(
            name="saveArtifact",
            operation_type=OperationType.MUTATION,
            description="Save a rich artifact (chart, HTML, diagram) to a session",
            arguments=[
                OperationArgument("sessionId", "ID!", "Session ID"),
                OperationArgument("input", "ArtifactInput!", "Artifact data including type and content"),
            ],
            return_type="Artifact!",
            use_cases=[
                "Save a generated chart",
                "Store an HTML report",
                "Persist a diagram",
            ],
            tags=["artifact", "save", "output"],
        ),

        OperationMetadata(
            name="replan",
            operation_type=OperationType.MUTATION,
            description="Revise the session's plan based on feedback while preserving completed work",
            arguments=[
                OperationArgument("sessionId", "ID!", "Session ID"),
                OperationArgument("feedback", "String!", "What should change about the plan"),
            ],
            return_type="Plan!",
            use_cases=[
                "Change the analysis approach",
                "Add steps to the plan",
                "Remove unnecessary steps",
                "Adjust based on intermediate results",
            ],
            tags=["session", "plan", "modify"],
        ),
    ])

    # ========== SUBSCRIPTIONS ==========

    catalog.register_operations([
        OperationMetadata(
            name="stepProgress",
            operation_type=OperationType.SUBSCRIPTION,
            description="Subscribe to real-time progress updates during step execution",
            arguments=[
                OperationArgument("sessionId", "ID!", "Session to monitor"),
            ],
            return_type="StepProgress!",
            use_cases=[
                "Show live execution progress",
                "Display code generation in real-time",
                "Monitor long-running steps",
            ],
            tags=["session", "realtime", "progress"],
        ),

        OperationMetadata(
            name="sessionUpdates",
            operation_type=OperationType.SUBSCRIPTION,
            description="Subscribe to all updates for a session including step completions and artifacts",
            arguments=[
                OperationArgument("sessionId", "ID!", "Session to monitor"),
            ],
            return_type="SessionUpdate!",
            use_cases=[
                "Keep UI in sync with session state",
                "Get notified when steps complete",
                "Receive artifact updates",
            ],
            tags=["session", "realtime", "updates"],
        ),
    ])

    # Build the vector index
    catalog.build_index()

    return catalog
