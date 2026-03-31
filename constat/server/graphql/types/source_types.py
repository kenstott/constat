# Copyright (c) 2025 Kenneth Stott
# Canary: bcbd09fd-0e84-4825-bccb-ecf131f65b9d
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Strawberry GraphQL types for data sources (Phase 5)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import strawberry
from strawberry.scalars import JSON


@strawberry.type
class UploadedFileType:
    id: str
    filename: str
    file_uri: str
    size_bytes: int
    content_type: str
    uploaded_at: datetime


@strawberry.type
class UploadedFileListType:
    files: list[UploadedFileType]
    total: int = 0


@strawberry.type
class FileRefType:
    name: str
    uri: str
    has_auth: bool
    description: Optional[str] = None
    added_at: Optional[datetime] = None
    session_id: Optional[str] = None


@strawberry.type
class FileRefListType:
    file_refs: list[FileRefType]
    total: int = 0


@strawberry.type
class SessionDatabaseType:
    name: str
    type: str
    dialect: Optional[str] = None
    description: Optional[str] = None
    connected: bool = False
    table_count: Optional[int] = None
    added_at: Optional[datetime] = None
    is_dynamic: bool = False
    file_id: Optional[str] = None
    source: str = "config"
    tier: Optional[str] = None


@strawberry.type
class SessionDatabaseListType:
    databases: list[SessionDatabaseType]
    total: int = 0


@strawberry.type
class DatabaseTestResultType:
    name: str
    connected: bool
    table_count: int = 0
    error: Optional[str] = None


@strawberry.type
class DatabaseTablePreviewType:
    database: str
    table_name: str
    columns: list[str]
    data: JSON
    page: int
    page_size: int
    total_rows: int
    has_more: bool


@strawberry.type
class SessionApiType:
    name: str
    type: Optional[str] = None
    description: Optional[str] = None
    base_url: Optional[str] = None
    connected: bool = False
    from_config: bool = False
    source: str = "config"
    is_dynamic: bool = False
    tier: Optional[str] = None


@strawberry.type
class SessionDocumentType:
    name: str
    type: Optional[str] = None
    description: Optional[str] = None
    path: Optional[str] = None
    indexed: bool = False
    from_config: bool = False
    source: str = "config"
    tier: Optional[str] = None


@strawberry.type
class DataSourcesType:
    databases: list[SessionDatabaseType]
    apis: list[SessionApiType]
    documents: list[SessionDocumentType]


@strawberry.type
class DocumentResultType:
    name: str
    content: Optional[str] = None
    metadata: Optional[JSON] = None


@strawberry.type
class UploadDocumentResultItem:
    filename: str
    status: str
    name: Optional[str] = None
    reason: Optional[str] = None
    path: Optional[str] = None


@strawberry.type
class UploadDocumentsResultType:
    status: str
    accepted_count: int
    database_count: int
    total_files: int
    results: list["UploadDocumentResultItem"]


@strawberry.type
class UserSourcesType:
    databases: JSON
    documents: JSON
    apis: JSON


@strawberry.type
class UserSourceResultType:
    status: str
    name: str
    source_type: Optional[str] = None


@strawberry.type
class MoveSourceResultType:
    status: str
    name: str
    source_type: str


# Source input types

@strawberry.input
class DatabaseAddInput:
    name: str
    type: str = "sqlalchemy"
    uri: Optional[str] = None
    file_id: Optional[str] = None
    description: Optional[str] = None


@strawberry.input
class ApiAddInput:
    name: str
    type: str = "rest"
    base_url: str = ""
    description: Optional[str] = None
    auth_type: Optional[str] = None
    auth_header: Optional[str] = None


@strawberry.input
class FileRefInput:
    name: str
    uri: str
    auth: Optional[str] = None
    description: Optional[str] = None


@strawberry.input
class DocumentUriInput:
    name: str
    url: str
    description: str = ""
    headers: Optional[JSON] = None
    follow_links: bool = False
    max_depth: int = 2
    max_documents: int = 20
    same_domain_only: bool = True
    exclude_patterns: list[str] = strawberry.field(default_factory=list)
    type: str = "auto"


@strawberry.input
class EmailSourceInput:
    name: str
    url: str
    username: str
    password: Optional[str] = None
    auth_type: str = "basic"
    mailbox: str = "INBOX"
    since: Optional[str] = None
    max_messages: int = 500
    include_headers: bool = True
    extract_attachments: bool = True
    oauth2_client_id: Optional[str] = None
    oauth2_client_secret: Optional[str] = None
    oauth2_tenant_id: Optional[str] = None
    oauth2_refresh_token: Optional[str] = None
