# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Consolidated session resources.

Provides a single source of truth for all available resources in a session.
Computed once at session start and updated on project changes.
"""

from dataclasses import dataclass, field


@dataclass
class ResourceInfo:
    """Basic info about a resource."""
    name: str
    description: str = ""
    source: str = "config"  # "config" | "project:<filename>" | "session"


@dataclass
class DatabaseInfo(ResourceInfo):
    """Database resource info."""
    db_type: str = "sql"  # sql | mongodb | cassandra | etc.


@dataclass
class APIInfo(ResourceInfo):
    """API resource info."""
    api_type: str = "graphql"  # graphql | openapi


@dataclass
class DocumentInfo(ResourceInfo):
    """Document resource info."""
    doc_type: str = "file"  # file | http | inline | etc.


@dataclass
class SessionResources:
    """Consolidated view of all available session resources.

    Single source of truth for databases, APIs, and documents.
    Computed at session creation, updated on project load/unload.

    Usage:
        resources = SessionResources()
        resources.add_database("sales", "Sales database", source="config")
        resources.add_document("policy", "HR Policy", source="project:hr.yaml")

        # Get lists for prompts/tools
        db_names = resources.database_names
        doc_names = resources.document_names
    """

    databases: dict[str, DatabaseInfo] = field(default_factory=dict)
    apis: dict[str, APIInfo] = field(default_factory=dict)
    documents: dict[str, DocumentInfo] = field(default_factory=dict)

    def add_database(
        self,
        name: str,
        description: str = "",
        db_type: str = "sql",
        source: str = "config",
    ) -> None:
        """Add a database to available resources."""
        self.databases[name] = DatabaseInfo(
            name=name,
            description=description,
            db_type=db_type,
            source=source,
        )

    def add_api(
        self,
        name: str,
        description: str = "",
        api_type: str = "graphql",
        source: str = "config",
    ) -> None:
        """Add an API to available resources."""
        self.apis[name] = APIInfo(
            name=name,
            description=description,
            api_type=api_type,
            source=source,
        )

    def add_document(
        self,
        name: str,
        description: str = "",
        doc_type: str = "file",
        source: str = "config",
    ) -> None:
        """Add a document to available resources."""
        self.documents[name] = DocumentInfo(
            name=name,
            description=description,
            doc_type=doc_type,
            source=source,
        )

    def remove_database(self, name: str) -> None:
        """Remove a database from available resources."""
        self.databases.pop(name, None)

    def remove_api(self, name: str) -> None:
        """Remove an API from available resources."""
        self.apis.pop(name, None)

    def remove_document(self, name: str) -> None:
        """Remove a document from available resources."""
        self.documents.pop(name, None)

    def remove_by_source(self, source: str) -> None:
        """Remove all resources from a specific source (e.g., a project)."""
        self.databases = {k: v for k, v in self.databases.items() if v.source != source}
        self.apis = {k: v for k, v in self.apis.items() if v.source != source}
        self.documents = {k: v for k, v in self.documents.items() if v.source != source}

    @property
    def database_names(self) -> list[str]:
        """Get list of database names."""
        return list(self.databases.keys())

    @property
    def api_names(self) -> list[str]:
        """Get list of API names."""
        return list(self.apis.keys())

    @property
    def document_names(self) -> list[str]:
        """Get list of document names."""
        return list(self.documents.keys())

    def get_database_descriptions(self) -> list[tuple[str, str]]:
        """Get list of (name, description) for databases."""
        return [(info.name, info.description) for info in self.databases.values()]

    def get_api_descriptions(self) -> list[tuple[str, str]]:
        """Get list of (name, description) for APIs."""
        return [(info.name, info.description) for info in self.apis.values()]

    def get_document_descriptions(self) -> list[tuple[str, str]]:
        """Get list of (name, description) for documents."""
        return [(info.name, info.description) for info in self.documents.values()]

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization (e.g., session.json)."""
        return {
            "databases": self.database_names,
            "apis": self.api_names,
            "documents": self.document_names,
        }

    def has_documents(self) -> bool:
        """Check if any documents are available."""
        return len(self.documents) > 0

    def has_apis(self) -> bool:
        """Check if any APIs are available."""
        return len(self.apis) > 0

    def has_databases(self) -> bool:
        """Check if any databases are available."""
        return len(self.databases) > 0
