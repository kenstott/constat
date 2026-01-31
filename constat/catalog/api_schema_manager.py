# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""API schema introspection, caching, and vector search.

Handles GraphQL introspection and REST API metadata indexing for semantic search.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests

from constat.core.config import Config, APIConfig
from constat.embedding_loader import EmbeddingModelLoader

logger = logging.getLogger(__name__)


@dataclass
class APIFieldMetadata:
    """Metadata for a single API field/parameter."""
    name: str
    type: str
    description: Optional[str] = None
    is_required: bool = False


@dataclass
class APIEndpointMetadata:
    """Metadata for an API endpoint or GraphQL type."""
    api_name: str
    endpoint_name: str  # Resource name (e.g., "breeds", "users")
    api_type: str  # "graphql" or "rest"
    description: Optional[str] = None
    fields: list[APIFieldMetadata] = field(default_factory=list)
    # Original path/method for REST endpoints (e.g., "GET /breeds")
    http_method: Optional[str] = None
    http_path: Optional[str] = None
    operation_id: Optional[str] = None

    @property
    def full_name(self) -> str:
        """Unique identifier for this endpoint."""
        if self.http_method and self.http_path:
            return f"{self.api_name}.{self.http_method} {self.http_path}"
        return f"{self.api_name}.{self.endpoint_name}"

    @property
    def display_name(self) -> str:
        """Human-readable name for display."""
        from constat.discovery.models import normalize_entity_name
        return normalize_entity_name(self.endpoint_name)

    def to_embedding_text(self) -> str:
        """Generate text for embedding."""
        parts = [
            f"API: {self.api_name}",
            f"Resource: {self.display_name}",
            f"Type: {self.api_type}",
        ]
        if self.http_method and self.http_path:
            parts.append(f"Endpoint: {self.http_method} {self.http_path}")
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.fields:
            field_names = [f.name for f in self.fields]
            parts.append(f"Fields: {', '.join(field_names)}")
        return " | ".join(parts)


class APISchemaManager:
    """Manages API schema introspection and vector search.

    Handles:
    - GraphQL schema introspection via __schema query
    - REST API metadata from config descriptions
    - OpenAPI spec parsing for REST APIs
    - Vector embeddings for semantic search (stored in shared vectors.duckdb)
    - Caching for performance
    """

    CACHE_DIR = Path(".constat")
    CACHE_FILENAME = "api_schema_cache.json"

    def __init__(self, config: Config):
        self.config = config
        self.metadata_cache: dict[str, APIEndpointMetadata] = {}

        # Vector store for embeddings (shared DuckDB)
        from constat.discovery.vector_store import DuckDBVectorStore
        self._vector_store: Optional[DuckDBVectorStore] = None
        self._model = None

        # Progress callback
        self._progress_callback: Optional[Callable] = None

    def initialize(self, progress_callback: Optional[Callable] = None) -> None:
        """Initialize the API schema manager.

        Args:
            progress_callback: Optional callback for progress updates
        """
        self._progress_callback = progress_callback

        if not self.config.apis:
            logger.debug("No APIs configured, skipping API schema initialization")
            return

        # Initialize vector store
        if self._vector_store is None:
            from constat.discovery.vector_store import DuckDBVectorStore
            self._vector_store = DuckDBVectorStore()

        config_hash = self._compute_config_hash()

        # Try to load metadata from cache
        if self._load_metadata_cache(config_hash):
            # Check if embeddings are also cached
            cached_hash = self._vector_store.get_catalog_config_hash('api')
            if cached_hash == config_hash and self._vector_store.count_catalog_entities(source='api') > 0:
                logger.debug("Loaded API schema and embeddings from cache")
                return

        # Introspect APIs
        self._introspect_apis()

        # Build vector index in DuckDB
        self._build_vector_index(config_hash)

        # Save metadata cache
        self._save_metadata_cache(config_hash)

        self._progress_callback = None

    def _compute_config_hash(self) -> str:
        """Compute hash of API config for cache invalidation."""
        if not self.config.apis:
            return "empty"

        config_data = {
            name: {
                "type": api.type,
                "url": api.url,
                "description": api.description,
            }
            for name, api in self.config.apis.items()
        }
        return hashlib.md5(json.dumps(config_data, sort_keys=True).encode()).hexdigest()[:12]

    def _load_metadata_cache(self, config_hash: str) -> bool:
        """Load schema metadata from cache (embeddings are in DuckDB)."""
        cache_dir = self.CACHE_DIR
        schema_file = cache_dir / self.CACHE_FILENAME

        if not schema_file.exists():
            return False

        try:
            # Load schema cache
            with open(schema_file) as f:
                cache_data = json.load(f)

            if cache_data.get("config_hash") != config_hash:
                logger.debug(f"API cache hash mismatch: {cache_data.get('config_hash')} != {config_hash}")
                return False

            # Reconstruct metadata
            for key, data in cache_data.get("endpoints", {}).items():
                fields = [APIFieldMetadata(**f) for f in data.get("fields", [])]
                self.metadata_cache[key] = APIEndpointMetadata(
                    api_name=data["api_name"],
                    endpoint_name=data["endpoint_name"],
                    api_type=data["api_type"],
                    description=data.get("description"),
                    fields=fields,
                )

            return True
        except Exception as e:
            logger.debug(f"Failed to load API metadata cache: {e}")
            return False

    def _save_metadata_cache(self, config_hash: str) -> None:
        """Save schema metadata to cache (embeddings are in DuckDB)."""
        cache_dir = self.CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save schema metadata only (embeddings stored in vectors.duckdb)
        cache_data = {
            "config_hash": config_hash,
            "endpoints": {
                key: {
                    "api_name": meta.api_name,
                    "endpoint_name": meta.endpoint_name,
                    "api_type": meta.api_type,
                    "description": meta.description,
                    "fields": [
                        {"name": f.name, "type": f.type, "description": f.description, "is_required": f.is_required}
                        for f in meta.fields
                    ],
                }
                for key, meta in self.metadata_cache.items()
            },
        }

        with open(cache_dir / self.CACHE_FILENAME, "w") as f:
            json.dump(cache_data, f, indent=2)

    def _introspect_apis(self) -> None:
        """Introspect all configured APIs."""
        if not self.config.apis:
            return

        for name, api_config in self.config.apis.items():
            try:
                if api_config.type == "graphql":
                    self._introspect_graphql(name, api_config)
                else:
                    self._introspect_rest(name, api_config)
            except Exception as e:
                logger.warning(f"Failed to introspect API '{name}': {e}")
                # Still add basic metadata from config
                self._add_basic_metadata(name, api_config)

    def _introspect_graphql(self, name: str, api_config: APIConfig) -> None:
        """Introspect a GraphQL API using __schema query."""
        introspection_query = """
        query IntrospectionQuery {
            __schema {
                types {
                    name
                    kind
                    description
                    fields {
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
                    }
                }
            }
        }
        """

        headers = {"Content-Type": "application/json"}
        if api_config.headers:
            headers.update(api_config.headers)

        try:
            response = requests.post(
                api_config.url,
                json={"query": introspection_query},
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.warning(f"GraphQL introspection failed for '{name}': {e}")
            self._add_basic_metadata(name, api_config)
            return

        # Parse schema types
        schema = data.get("data", {}).get("__schema", {})
        types = schema.get("types", [])

        # Filter to user-defined types (skip __* internal types and scalars)
        for type_def in types:
            type_name = type_def.get("name", "")
            kind = type_def.get("kind", "")

            # Skip internal types and scalars
            if type_name.startswith("__") or kind in ("SCALAR", "ENUM", "INPUT_OBJECT"):
                continue

            # Skip common wrapper types
            if type_name in ("Query", "Mutation", "Subscription"):
                # But extract their fields as top-level endpoints
                for field_def in type_def.get("fields") or []:
                    endpoint_name = field_def.get("name", "")
                    fields = self._extract_return_type_fields(field_def, types)

                    meta = APIEndpointMetadata(
                        api_name=name,
                        endpoint_name=endpoint_name,
                        api_type="graphql",
                        description=field_def.get("description") or api_config.description,
                        fields=fields,
                    )
                    self.metadata_cache[meta.full_name] = meta
                continue

            # Add object types as searchable entities
            fields = []
            for field_def in type_def.get("fields") or []:
                field_type = self._get_type_name(field_def.get("type", {}))
                fields.append(APIFieldMetadata(
                    name=field_def.get("name", ""),
                    type=field_type,
                    description=field_def.get("description"),
                ))

            meta = APIEndpointMetadata(
                api_name=name,
                endpoint_name=type_name,
                api_type="graphql",
                description=type_def.get("description"),
                fields=fields,
            )
            self.metadata_cache[meta.full_name] = meta

        logger.info(f"Introspected GraphQL API '{name}': {len([k for k in self.metadata_cache if k.startswith(name)])} types/endpoints")

    def _extract_return_type_fields(self, field_def: dict, all_types: list) -> list[APIFieldMetadata]:
        """Extract fields from the return type of a query/mutation."""
        type_info = field_def.get("type", {})
        type_name = self._get_type_name(type_info)

        # Find the type definition
        for type_def in all_types:
            if type_def.get("name") == type_name:
                fields = []
                for f in type_def.get("fields") or []:
                    fields.append(APIFieldMetadata(
                        name=f.get("name", ""),
                        type=self._get_type_name(f.get("type", {})),
                        description=f.get("description"),
                    ))
                return fields

        return []

    def _get_type_name(self, type_info: dict) -> str:
        """Extract type name from GraphQL type info (handling wrappers)."""
        if not type_info:
            return "Unknown"

        name = type_info.get("name")
        if name:
            return name

        # Handle NON_NULL and LIST wrappers
        of_type = type_info.get("ofType")
        if of_type:
            return self._get_type_name(of_type)

        return "Unknown"

    def _introspect_rest(self, name: str, api_config: APIConfig) -> None:
        """Introspect REST API from OpenAPI spec if available."""
        spec = None

        # Try to get OpenAPI spec from various sources
        if api_config.spec_inline:
            spec = api_config.spec_inline
        elif api_config.spec_url:
            spec = self._fetch_openapi_spec(api_config.spec_url, api_config.headers)
        elif api_config.spec_path:
            spec = self._load_openapi_spec_file(api_config.spec_path)

        if spec:
            self._parse_openapi_spec(name, api_config, spec)
        else:
            # No spec available, use basic metadata
            self._add_basic_metadata(name, api_config)

    def _fetch_openapi_spec(self, spec_url: str, headers: dict = None) -> Optional[dict]:
        """Fetch OpenAPI spec from URL."""
        try:
            req_headers = {"Accept": "application/json"}
            if headers:
                req_headers.update(headers)

            response = requests.get(spec_url, headers=req_headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch OpenAPI spec from {spec_url}: {e}")
            return None

    def _load_openapi_spec_file(self, spec_path: str) -> Optional[dict]:
        """Load OpenAPI spec from local file."""
        import yaml
        from pathlib import Path

        try:
            path = Path(spec_path)
            if not path.exists():
                logger.warning(f"OpenAPI spec file not found: {spec_path}")
                return None

            with open(path) as f:
                if path.suffix in (".yaml", ".yml"):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load OpenAPI spec from {spec_path}: {e}")
            return None

    def _parse_openapi_spec(self, name: str, api_config: APIConfig, spec: dict) -> None:
        """Parse OpenAPI spec and extract endpoint metadata."""
        paths = spec.get("paths", {})
        components = spec.get("components", {})
        schemas = components.get("schemas", {})

        for path, path_item in paths.items():
            for method, operation in path_item.items():
                if method in ("get", "post", "put", "patch", "delete"):
                    http_method = method.upper()
                    operation_id = operation.get("operationId")

                    # Extract resource name from path
                    from constat.discovery.models import extract_resource_from_path
                    resource_name = extract_resource_from_path(path)

                    # Extract parameters as fields
                    fields = []
                    for param in operation.get("parameters", []):
                        fields.append(APIFieldMetadata(
                            name=param.get("name", ""),
                            type=param.get("schema", {}).get("type", "string"),
                            description=param.get("description"),
                            is_required=param.get("required", False),
                        ))

                    # Extract response schema fields
                    response_fields = self._extract_response_fields(operation, schemas)
                    fields.extend(response_fields)

                    # Build description from summary and operation description
                    desc_parts = []
                    if operation.get("summary"):
                        desc_parts.append(operation["summary"])
                    if operation.get("description"):
                        desc_parts.append(operation["description"])
                    if not desc_parts and api_config.description:
                        desc_parts.append(api_config.description)

                    meta = APIEndpointMetadata(
                        api_name=name,
                        endpoint_name=resource_name,
                        api_type="rest",
                        description=" - ".join(desc_parts) if desc_parts else None,
                        fields=fields,
                        http_method=http_method,
                        http_path=path,
                        operation_id=operation_id,
                    )
                    self.metadata_cache[meta.full_name] = meta

        # Also add schema types as searchable entities
        for schema_name, schema_def in schemas.items():
            fields = []
            for prop_name, prop_def in schema_def.get("properties", {}).items():
                fields.append(APIFieldMetadata(
                    name=prop_name,
                    type=prop_def.get("type", "object"),
                    description=prop_def.get("description"),
                ))

            meta = APIEndpointMetadata(
                api_name=name,
                endpoint_name=schema_name,
                api_type="rest/schema",
                description=schema_def.get("description") or schema_def.get("title"),
                fields=fields,
            )
            self.metadata_cache[meta.full_name] = meta

        logger.info(f"Parsed OpenAPI spec '{name}': {len([k for k in self.metadata_cache if k.startswith(name)])} endpoints/schemas")

    def _extract_response_fields(self, operation: dict, schemas: dict) -> list[APIFieldMetadata]:
        """Extract fields from the response schema."""
        fields = []
        responses = operation.get("responses", {})

        # Look at 200/201 response
        for code in ("200", "201"):
            if code in responses:
                content = responses[code].get("content", {})
                json_content = content.get("application/json", {})
                schema = json_content.get("schema", {})

                # Resolve $ref if present
                if "$ref" in schema:
                    ref_name = schema["$ref"].split("/")[-1]
                    schema = schemas.get(ref_name, {})

                # Handle array of items
                if schema.get("type") == "array":
                    items = schema.get("items", {})
                    if "$ref" in items:
                        ref_name = items["$ref"].split("/")[-1]
                        schema = schemas.get(ref_name, {})
                    else:
                        schema = items

                # Extract properties
                for prop_name, prop_def in schema.get("properties", {}).items():
                    fields.append(APIFieldMetadata(
                        name=prop_name,
                        type=prop_def.get("type", "object"),
                        description=prop_def.get("description"),
                    ))
                break

        return fields

    def _add_basic_metadata(self, name: str, api_config: APIConfig) -> None:
        """Add basic metadata from config when introspection fails."""
        meta = APIEndpointMetadata(
            api_name=name,
            endpoint_name="api",
            api_type=api_config.type,
            description=api_config.description,
            fields=[],
        )
        self.metadata_cache[meta.full_name] = meta

    def _build_vector_index(self, config_hash: str) -> None:
        """Build vector embeddings for API endpoints.

        Stores embeddings in the shared vectors.duckdb as unified entities.
        """
        if not self.metadata_cache:
            return

        # Get shared model
        self._model = EmbeddingModelLoader.get_instance().get_model()

        # Generate entity records for unified catalog
        entities = []
        texts = []

        for full_name in sorted(self.metadata_cache.keys()):
            meta = self.metadata_cache[full_name]
            embedding_text = meta.to_embedding_text()
            texts.append(embedding_text)

            # Determine entity type based on api_type
            if meta.api_type == "rest/schema":
                entity_type = "api_schema"
            else:
                entity_type = "api_endpoint"

            entities.append({
                "id": full_name,
                "name": meta.display_name,
                "type": entity_type,
                "parent_id": None,  # API endpoints don't have parents for now
                "metadata": {
                    "api_name": meta.api_name,
                    "api_type": meta.api_type,
                    "description": meta.description,
                    "fields": [f.name for f in meta.fields],
                    "original_name": meta.endpoint_name,
                    "http_method": meta.http_method,
                    "http_path": meta.http_path,
                    "operation_id": meta.operation_id,
                },
            })

        # Generate embeddings
        embeddings = self._model.encode(texts, convert_to_numpy=True)

        # Clear old and save to DuckDB
        self._vector_store.clear_catalog_entities('api')
        self._vector_store.add_catalog_entities(entities, embeddings, 'api', config_hash)

        # Extract entities from API descriptions
        self._extract_entities_from_descriptions()

        logger.info(f"Built API vector index with {len(texts)} endpoints")

    def _extract_entities_from_descriptions(self) -> None:
        """Extract entities from API endpoint metadata using spaCy NER.

        Creates chunks for ALL endpoints and fields (not just those with descriptions)
        so that entity extraction can find and link endpoint/field names.

        Steps:
        1. Collect chunks for ALL endpoints and fields
        2. Generate embeddings and store chunks in vector store
        3. Extract entities and chunk links
        4. Store entities and links for proper reference tracking
        """
        if not hasattr(self._vector_store, 'add_entities'):
            return

        # Lazy imports to avoid circular dependency
        from constat.discovery.models import DocumentChunk, ChunkEntity
        from constat.discovery.entity_extractor import EntityExtractor, ExtractionConfig

        # Collect chunks for ALL endpoints and fields
        chunks: list[DocumentChunk] = []
        for full_name, meta in self.metadata_cache.items():
            field_names = [f.name for f in meta.fields]

            # Endpoint chunk - use description if available, otherwise structured text
            if meta.description:
                endpoint_content = f"{meta.endpoint_name} endpoint: {meta.description}"
            else:
                endpoint_content = f"{meta.endpoint_name} endpoint in {meta.api_name} API"
                if meta.http_method and meta.http_path:
                    endpoint_content += f" ({meta.http_method} {meta.http_path})"
                if field_names:
                    endpoint_content += f" with fields: {', '.join(field_names)}"

            chunks.append(DocumentChunk(
                document_name=f"api:{full_name}",
                content=endpoint_content,
                section="api_endpoint",
                chunk_index=0,
            ))

            # Field chunks - with or without descriptions
            for i, field_meta in enumerate(meta.fields):
                if field_meta.description:
                    field_content = f"{field_meta.name} field in {meta.endpoint_name}: {field_meta.description}"
                else:
                    field_type = field_meta.type if field_meta.type else "unknown type"
                    field_content = f"{field_meta.name} field ({field_type}) in {meta.endpoint_name} endpoint"

                chunks.append(DocumentChunk(
                    document_name=f"api:{full_name}.{field_meta.name}",
                    content=field_content,
                    section="field_description",
                    chunk_index=i,
                ))

        if not chunks:
            logger.debug("No API descriptions to extract entities from")
            return

        # Step 1: Generate embeddings and store chunks
        if self._model is not None:
            try:
                texts = [c.content for c in chunks]
                embeddings = self._model.encode(texts, convert_to_numpy=True)
                self._vector_store.add_chunks(chunks, embeddings)
                logger.debug(f"Stored {len(chunks)} API description chunks")
            except Exception as e:
                logger.warning(f"Failed to store API description chunks: {e}")

        # Step 2: Configure extractor with NER only
        config = ExtractionConfig(
            extract_schema=False,
            extract_ner=True,
        )
        extractor = EntityExtractor(config)

        # Step 3: Extract entities and collect links
        all_links: list[ChunkEntity] = []
        for chunk in chunks:
            extractions = extractor.extract(chunk)
            for entity, link in extractions:
                all_links.append(link)

        # Step 4: Store entities
        entities = extractor.get_all_entities()
        if entities:
            logger.debug(f"Extracted {len(entities)} entities from API descriptions")
            self._vector_store.add_entities(entities, source="api")

        # Step 5: Store chunk-entity links
        if all_links:
            # Deduplicate links by (chunk_id, entity_id)
            unique_links: dict[tuple[str, str], ChunkEntity] = {}
            for link in all_links:
                key = (link.chunk_id, link.entity_id)
                if key not in unique_links:
                    unique_links[key] = link
                else:
                    existing = unique_links[key]
                    unique_links[key] = ChunkEntity(
                        chunk_id=link.chunk_id,
                        entity_id=link.entity_id,
                        mention_count=existing.mention_count + link.mention_count,
                        confidence=max(existing.confidence, link.confidence),
                        mention_text=existing.mention_text or link.mention_text,
                    )
            self._vector_store.link_chunk_entities(list(unique_links.values()))
            logger.debug(f"Created {len(unique_links)} chunk-entity links for API descriptions")

    def find_relevant_apis(
        self,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.3,
    ) -> list[dict]:
        """Find APIs relevant to a query using semantic search.

        Queries embeddings directly from DuckDB using array_cosine_similarity.

        Args:
            query: Natural language query
            limit: Maximum number of results
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of dicts with api_name, endpoint, type, description, fields, similarity
        """
        if self._vector_store is None or self._vector_store.count_catalog_entities(source='api') == 0:
            return []

        # Lazy load model if needed
        if self._model is None:
            self._model = EmbeddingModelLoader.get_instance().get_model()

        # Encode query
        query_embedding = self._model.encode([query], convert_to_numpy=True)

        # Search unified catalog entities for API endpoints
        results = self._vector_store.search_catalog_entities(
            query_embedding, source='api', limit=limit, min_similarity=min_similarity
        )

        # Transform results to match expected format
        return [
            {
                "api_name": r["metadata"].get("api_name", ""),
                "endpoint": r["name"],
                "type": r["metadata"].get("api_type", ""),
                "description": r["metadata"].get("description"),
                "fields": r["metadata"].get("fields", []),
                "similarity": r["similarity"],
            }
            for r in results
        ]

    def get_api_schema(self, api_name: str) -> list[APIEndpointMetadata]:
        """Get all endpoints for a specific API."""
        return [
            meta for key, meta in self.metadata_cache.items()
            if meta.api_name == api_name
        ]

    def get_endpoint_metadata(self, api_name: str, endpoint: str) -> Optional[APIEndpointMetadata]:
        """Get metadata for a specific endpoint."""
        key = f"{api_name}.{endpoint}"
        return self.metadata_cache.get(key)

    def get_description_text(self) -> list[tuple[str, str]]:
        """Return all metadata text from API schema for NER processing.

        Includes endpoint names, field names, and their descriptions.

        Returns:
            List of (source_name, text) tuples for NER extraction
        """
        results = []

        for key, meta in self.metadata_cache.items():
            # Endpoint name
            results.append((f"api:{key}", meta.endpoint_name))

            # Endpoint description
            if meta.description:
                results.append((f"api:{key}:desc", meta.description))

            # Field names and descriptions
            for field in meta.fields:
                results.append((f"api:{key}.{field.name}", field.name))
                if field.description:
                    results.append((f"api:{key}.{field.name}:desc", field.description))

        return results

    def get_entity_names(self) -> list[str]:
        """Return all API endpoint and field names for entity extraction.

        Returns:
            List of unique entity names (endpoints and fields)
        """
        entities = set()

        for meta in self.metadata_cache.values():
            # Add endpoint name
            entities.add(meta.endpoint_name)

            # Add field names
            for field in meta.fields:
                entities.add(field.name)

        return list(entities)

    def get_overview(self) -> str:
        """Get a brief overview of indexed APIs."""
        if not self.metadata_cache:
            return "No APIs indexed."

        # Group by API
        by_api: dict[str, list[str]] = {}
        for key, meta in self.metadata_cache.items():
            if meta.api_name not in by_api:
                by_api[meta.api_name] = []
            by_api[meta.api_name].append(meta.endpoint_name)

        lines = ["Indexed APIs:"]
        for api_name, endpoints in by_api.items():
            api_config = self.config.apis.get(api_name) if self.config.apis else None
            api_type = api_config.type if api_config else "unknown"
            lines.append(f"  {api_name} ({api_type}): {len(endpoints)} endpoints")

        return "\n".join(lines)
