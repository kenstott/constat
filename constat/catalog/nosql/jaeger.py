# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Jaeger distributed tracing connector."""

from datetime import datetime, timezone
from typing import Any, Optional

import requests

from .base import NoSQLConnector, NoSQLType, CollectionMetadata, FieldInfo


class JaegerConnector(NoSQLConnector):
    """Connector for Jaeger distributed tracing.

    Services are exposed as "collections". Each service's spans have
    different tag schemas, so schema inference samples spans per service.

    Usage:
        connector = JaegerConnector(
            uri="http://localhost:16686",
            name="tracing",
        )
        connector.connect()

        # List services.
        services = connector.get_collections()

        # Get span schema for a service.
        schema = connector.get_collection_schema("my-service")

        # Query spans.
        results = connector.query("my-service", {"operation": "GET /api/users"})

        # Get a specific trace.
        spans = connector.get_trace("abc123")
    """

    # Fixed span fields always present in flattened output
    FIXED_FIELDS = [
        FieldInfo(name="traceID", data_type="string", is_indexed=True, is_unique=True),
        FieldInfo(name="spanID", data_type="string"),
        FieldInfo(name="parentSpanID", data_type="string", nullable=True),
        FieldInfo(name="operationName", data_type="string", is_indexed=True),
        FieldInfo(name="serviceName", data_type="string", is_indexed=True),
        FieldInfo(name="startTime", data_type="datetime"),
        FieldInfo(name="duration", data_type="integer", description="microseconds"),
        FieldInfo(name="error", data_type="boolean"),
        FieldInfo(name="statusCode", data_type="integer", nullable=True),
    ]

    def __init__(
        self,
        uri: str = "http://localhost:16686",
        username: Optional[str] = None,
        password: Optional[str] = None,
        name: Optional[str] = None,
        description: str = "",
        sample_size: int = 50,
    ) -> None:
        super().__init__(name=name or "jaeger", description=description)
        self.uri = uri.rstrip("/")
        self.username = username
        self.password = password
        self.sample_size = sample_size
        self._session: Optional[requests.Session] = None

    @property
    def nosql_type(self) -> NoSQLType:
        return NoSQLType.OBSERVABILITY

    def connect(self) -> None:
        """Verify connectivity via GET /api/services."""
        session = requests.Session()
        if self.username and self.password:
            session.auth = (self.username, self.password)

        resp = session.get(f"{self.uri}/api/services")
        resp.raise_for_status()

        self._session = session
        self._connected = True

    def disconnect(self) -> None:
        """Close the HTTP session."""
        if self._session:
            self._session.close()
            self._session = None
        self._connected = False

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        """HTTP GET with base URL and auth."""
        if not self._connected or not self._session:
            raise RuntimeError("Not connected to Jaeger")
        resp = self._session.get(f"{self.uri}{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    def get_collections(self) -> list[str]:
        """Return Jaeger service names."""
        data = self._get("/api/services")
        return sorted(data.get("data", []))

    def get_collection_schema(self, collection: str) -> CollectionMetadata:
        """Sample spans from a service and infer tag schema."""
        if collection in self._metadata_cache:
            return self._metadata_cache[collection]

        # Fetch sample traces
        data = self._get("/api/traces", params={
            "service": collection,
            "limit": self.sample_size,
        })

        spans = []
        for trace in data.get("data", []):
            processes = trace.get("processes", {})
            for span in trace.get("spans", []):
                process = processes.get(span.get("processID", ""), {})
                spans.append(self._flatten_span(span, process))

        metadata = self._infer_schema_from_spans(collection, spans)
        self._metadata_cache[collection] = metadata
        return metadata

    def query(
        self, collection: str, query: dict, limit: int = 100,
    ) -> list[dict]:
        """Query spans for a service.

        Supported query keys:
            operation, tags (JSON string), minDuration, maxDuration,
            start (epoch micros), end (epoch micros)
        """
        params: dict[str, Any] = {"service": collection, "limit": limit}

        for key in ("operation", "tags", "minDuration", "maxDuration", "start", "end"):
            if key in query:
                params[key] = query[key]

        data = self._get("/api/traces", params=params)

        results = []
        for trace in data.get("data", []):
            processes = trace.get("processes", {})
            for span in trace.get("spans", []):
                process = processes.get(span.get("processID", ""), {})
                results.append(self._flatten_span(span, process))

        return results[:limit]

    def get_trace(self, trace_id: str) -> list[dict]:
        """Get all spans for a specific trace ID."""
        data = self._get(f"/api/traces/{trace_id}")

        results = []
        for trace in data.get("data", []):
            processes = trace.get("processes", {})
            for span in trace.get("spans", []):
                process = processes.get(span.get("processID", ""), {})
                results.append(self._flatten_span(span, process))

        return results

    def get_operations(self, service: str) -> list[str]:
        """Get operations for a service."""
        data = self._get(f"/api/services/{service}/operations")
        return data.get("data", [])

    def get_overview(self) -> str:
        """Token-optimized overview listing services and operation counts."""
        services = self.get_collections()

        lines = [f"  {self.name} (observability/jaeger): "]
        summaries = []
        for svc in services[:10]:
            try:
                ops = self.get_operations(svc)
                summaries.append(f"{svc} [{len(ops)} ops]")
            except Exception:
                summaries.append(svc)

        lines[0] += ", ".join(summaries)
        if len(services) > 10:
            lines[0] += f" (+{len(services) - 10} more services)"

        return "\n".join(lines)

    @staticmethod
    def _flatten_span(span: dict, process: dict) -> dict:
        """Flatten a Jaeger span + process into a flat dict."""
        # Extract tags into a dict
        tags = {}
        has_error = False
        status_code = None

        for tag in span.get("tags", []):
            key = tag.get("key", "")
            value = tag.get("value")
            tags[f"tags.{key}"] = value
            if key == "error" and value is True:
                has_error = True
            if key == "http.status_code":
                status_code = value

        # Process tags (service-level tags)
        for tag in process.get("tags", []):
            key = tag.get("key", "")
            value = tag.get("value")
            tags[f"tags.{key}"] = value

        # Build references to find parent
        parent_span_id = None
        for ref in span.get("references", []):
            if ref.get("refType") == "CHILD_OF":
                parent_span_id = ref.get("spanID")
                break

        # Convert startTime (epoch micros) to ISO datetime
        start_micros = span.get("startTime", 0)
        start_time = datetime.fromtimestamp(
            start_micros / 1_000_000, tz=timezone.utc,
        ).isoformat()

        result = {
            "traceID": span.get("traceID", ""),
            "spanID": span.get("spanID", ""),
            "parentSpanID": parent_span_id,
            "operationName": span.get("operationName", ""),
            "serviceName": process.get("serviceName", ""),
            "startTime": start_time,
            "duration": span.get("duration", 0),
            "error": has_error,
            "statusCode": status_code,
        }
        result.update(tags)
        return result

    def _infer_schema_from_spans(
        self, service: str, spans: list[dict],
    ) -> CollectionMetadata:
        """Build CollectionMetadata from fixed fields + dynamic tag fields."""
        # Start with fixed fields
        fields = list(self.FIXED_FIELDS)

        # Collect dynamic tag fields from samples
        tag_values: dict[str, list[Any]] = {}
        for span in spans:
            for key, value in span.items():
                if key.startswith("tags."):
                    if key not in tag_values:
                        tag_values[key] = []
                    tag_values[key].append(value)

        # Add dynamic tag fields
        for tag_name in sorted(tag_values.keys()):
            values = tag_values[tag_name]
            fields.append(FieldInfo(
                name=tag_name,
                data_type=self.infer_field_type(values),
                nullable=True,
                sample_values=values[:3],
            ))

        return CollectionMetadata(
            name=service,
            database=self.name,
            nosql_type=NoSQLType.OBSERVABILITY,
            fields=fields,
            document_count=len(spans),
            indexes=["traceID", "operationName", "serviceName"],
        )
