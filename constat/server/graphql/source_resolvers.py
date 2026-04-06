# Copyright (c) 2025 Kenneth Stott
# Canary: 82e1602f-0135-477c-865e-ebca45eb9d84
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for data sources: files, file refs, and shared helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
import strawberry
from strawberry.file_uploads import Upload

from constat.server.graphql.session_context import GqlInfo as Info
from constat.server.graphql.types import (
    DatabaseAddInput,
    DatabaseTablePreviewType,
    DatabaseTestResultType,
    DataSourcesType,
    DeleteResultType,
    DocumentResultType,
    EmailSourceInput,
    DocumentUriInput,
    ApiAddInput,
    FileRefInput,
    FileRefListType,
    FileRefType,
    MoveSourceResultType,
    SessionApiType,
    SessionDatabaseListType,
    SessionDatabaseType,
    SessionDocumentType,
    UploadDocumentResultItem,
    UploadDocumentsResultType,
    UploadedFileListType,
    UploadedFileType,
    UserSourceResultType,
    UserSourcesType,
)

import os
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

from typing import Optional


@strawberry.type
class UriValidationResult:
    reachable: bool
    error: str | None = None


@strawberry.type
class UnifiedSourceType:
    """Unified view of a data source from the registry."""
    name: str
    kind: str
    type: str
    description: Optional[str] = None
    state: str = "connected"
    error: Optional[str] = None
    queryable: bool = False
    ingestible: bool = False
    refreshable: bool = False
    viewable: bool = False
    item_count: Optional[int] = None
    indexed_count: Optional[int] = None


def _get_managed(info: Info, session_id: str):
    sm = info.context.session_manager
    managed = sm.get_session_or_none(session_id)
    if not managed:
        raise ValueError(f"Session {session_id} not found")
    return managed


def _get_tier(managed, resource_type: str, name: str) -> str | None:
    rc = getattr(managed, "resolved_config", None)
    if not rc or not rc._attribution:
        return None
    source = rc._attribution.get(f"{resource_type}.{name}")
    return source.value if source else None


def _build_databases(managed, session_id: str) -> list[SessionDatabaseType]:
    databases = []
    seen_names: set[str] = set()

    # Config databases
    for name, db_config in managed.session.config.databases.items():
        connected = False
        table_count = 0
        try:
            if managed.session.schema_manager:
                tables = managed.session.schema_manager.get_tables_for_db(name)
                table_count = len(tables)
                connected = True
        except (KeyError, ValueError):
            pass

        databases.append(SessionDatabaseType(
            name=name,
            type=db_config.type,
            description=db_config.description,
            connected=connected,
            table_count=table_count,
            added_at=managed.created_at,
            is_dynamic=False,
            source="config",
            tier=_get_tier(managed, "databases", name),
        ))
        seen_names.add(name)

    # Domain databases
    for domain_filename in managed.active_domains:
        domain = managed.session.config.load_domain(domain_filename)
        if domain:
            for name, db_config in domain.databases.items():
                if name in seen_names:
                    continue
                connected = name in getattr(managed, "_domain_databases", set())
                table_count = 0
                if connected and managed.session.schema_manager:
                    try:
                        tables = managed.session.schema_manager.get_tables_for_db(name)
                        table_count = len(tables)
                    except (KeyError, ValueError):
                        pass

                databases.append(SessionDatabaseType(
                    name=name,
                    type=db_config.type,
                    description=db_config.description,
                    connected=connected,
                    table_count=table_count,
                    added_at=managed.created_at,
                    is_dynamic=False,
                    source=domain_filename,
                    tier=_get_tier(managed, "databases", name),
                ))
                seen_names.add(name)

    # Dynamic databases
    for db in managed._dynamic_dbs:
        if db["name"] in seen_names:
            continue
        databases.append(SessionDatabaseType(
            name=db["name"],
            type=db["type"],
            dialect=db.get("dialect"),
            description=db.get("description"),
            connected=db.get("connected", False),
            table_count=db.get("table_count", 0),
            added_at=datetime.fromisoformat(db["added_at"]),
            is_dynamic=True,
            file_id=db.get("file_id"),
            source="session",
            tier=_get_tier(managed, "databases", db["name"]),
        ))

    return databases


def _build_apis(managed) -> list[SessionApiType]:
    apis: list[SessionApiType] = []
    seen: set[str] = set()

    for name, api_config in managed.session.config.apis.items():
        apis.append(SessionApiType(
            name=name,
            type=api_config.type,
            description=api_config.description,
            base_url=api_config.url,
            connected=True,
            from_config=True,
            source="config",
            tier=_get_tier(managed, "apis", name),
        ))
        seen.add(name)

    for domain_filename in managed.active_domains:
        domain = managed.session.config.load_domain(domain_filename)
        if domain:
            for name, api_config in domain.apis.items():
                if name in seen:
                    continue
                apis.append(SessionApiType(
                    name=name,
                    type=api_config.type,
                    description=api_config.description,
                    base_url=api_config.url,
                    connected=True,
                    from_config=False,
                    source=domain_filename,
                    tier=_get_tier(managed, "apis", name),
                ))
                seen.add(name)

    for api in managed._dynamic_apis:
        if api["name"] in seen:
            continue
        apis.append(SessionApiType(
            name=api["name"],
            type=api.get("type"),
            description=api.get("description"),
            base_url=api.get("base_url"),
            connected=api.get("connected", True),
            from_config=False,
            source="session",
            is_dynamic=True,
            tier=_get_tier(managed, "apis", api["name"]),
        ))

    return apis


def _build_documents(managed) -> list[SessionDocumentType]:
    documents: list[SessionDocumentType] = []
    seen: set[str] = set()
    config = managed.session.config

    for name, doc_config in config.documents.items():
        documents.append(SessionDocumentType(
            name=name,
            type=doc_config.type,
            description=doc_config.description,
            path=doc_config.path,
            indexed=True,
            from_config=True,
            source="config",
            tier=_get_tier(managed, "documents", name),
        ))
        seen.add(name)

    for domain_filename in managed.active_domains:
        domain = config.load_domain(domain_filename)
        if domain:
            for name, doc_config in domain.documents.items():
                if name in seen:
                    continue
                documents.append(SessionDocumentType(
                    name=name,
                    type=doc_config.type,
                    description=doc_config.description,
                    path=doc_config.path,
                    indexed=True,
                    from_config=False,
                    source=domain_filename,
                    tier=_get_tier(managed, "documents", name),
                ))
                seen.add(name)

    for ref in managed._file_refs:
        if ref["name"] in seen:
            continue
        documents.append(SessionDocumentType(
            name=ref["name"],
            type=ref.get("uri", "").split(".")[-1] if ref.get("uri") else None,
            description=ref.get("description"),
            path=ref.get("uri"),
            indexed=True,
            source="session",
            from_config=False,
            tier=_get_tier(managed, "documents", ref["name"]),
        ))
        seen.add(ref["name"])

    # Personal accounts (calendar, drive, etc.) connected via OAuth
    user_id = getattr(managed, "user_id", None)
    if user_id:
        from constat.server.accounts import load_user_accounts
        data_dir = getattr(config, "data_dir", None)
        personal_accounts = load_user_accounts(user_id, data_dir)
        indexed_names: set[str] = set(
            managed.session.resources.document_names
        ) if managed.session.resources else set()
        for acct_name, acct in personal_accounts.items():
            if not acct.active or acct.type not in ("calendar", "drive", "sharepoint"):
                continue
            if acct_name in seen:
                continue
            documents.append(SessionDocumentType(
                name=acct_name,
                type=acct.type,
                description=acct.display_name,
                path=None,
                indexed=acct_name in indexed_names,
                from_config=False,
                source="personal",
                tier=None,
            ))
            seen.add(acct_name)

    return documents


@strawberry.type
class Query:
    @strawberry.field
    async def files(self, info: Info, session_id: str) -> UploadedFileListType:
        managed = _get_managed(info, session_id)
        from constat.server.routes.files import _get_uploaded_files_for_session
        file_list = _get_uploaded_files_for_session(managed)
        return UploadedFileListType(
            files=[
                UploadedFileType(
                    id=f["id"],
                    filename=f["filename"],
                    file_uri=f["file_uri"],
                    size_bytes=f["size_bytes"],
                    content_type=f["content_type"],
                    uploaded_at=datetime.fromisoformat(f["uploaded_at"]),
                )
                for f in file_list
            ],
            total=len(file_list),
        )

    @strawberry.field
    async def file_refs(self, info: Info, session_id: str) -> FileRefListType:
        managed = _get_managed(info, session_id)
        refs = managed._file_refs
        return FileRefListType(
            file_refs=[
                FileRefType(
                    name=ref["name"],
                    uri=ref["uri"],
                    has_auth=ref["has_auth"],
                    description=ref.get("description"),
                    added_at=datetime.fromisoformat(ref["added_at"]),
                    session_id=session_id,
                )
                for ref in refs
            ],
            total=len(refs),
        )

    @strawberry.field
    async def databases(self, info: Info, session_id: str) -> SessionDatabaseListType:
        managed = _get_managed(info, session_id)
        dbs = _build_databases(managed, session_id)
        return SessionDatabaseListType(databases=dbs, total=len(dbs))

    @strawberry.field
    async def data_sources(self, info: Info, session_id: str) -> DataSourcesType:
        managed = _get_managed(info, session_id)
        return DataSourcesType(
            databases=_build_databases(managed, session_id),
            apis=_build_apis(managed),
            documents=_build_documents(managed),
        )

    @strawberry.field
    async def unified_sources(self, info: Info, session_id: str) -> list[UnifiedSourceType]:
        """List all data sources via the unified registry."""
        managed = _get_managed(info, session_id)
        if not managed.registry:
            return []
        sources = managed.registry.list_all()
        return [
            UnifiedSourceType(
                name=s.name, kind=s.kind.value, type=s.type,
                description=s.description, state=s.state, error=s.error,
                queryable=s.queryable, ingestible=s.ingestible,
                refreshable=s.refreshable, viewable=s.viewable,
                item_count=s.item_count, indexed_count=s.indexed_count,
            )
            for s in sources
        ]

    @strawberry.field
    async def database_table_preview(
        self, info: Info, session_id: str, db_name: str, table_name: str,
        page: int = 1, page_size: int = 100,
    ) -> DatabaseTablePreviewType:
        import pandas as pd
        from constat.server.routes.data import _sanitize_df_for_json

        managed = _get_managed(info, session_id)
        if not managed.has_database(db_name):
            raise ValueError(f"Database not found: {db_name}")

        db_connection = managed.get_database_connection(db_name)
        if not db_connection:
            raise ValueError(f"Database '{db_name}' is not connected")

        # Validate table_name against known tables to prevent SQL injection
        if not managed.session.schema_manager:
            raise ValueError("Schema manager not available for table validation")

        known_keys = set(managed.session.schema_manager.metadata_cache.keys())
        qualified = f"{db_name}.{table_name}"
        if qualified not in known_keys and not any(
            k.split(".", 1)[-1] == table_name for k in known_keys if k.startswith(f"{db_name}.")
        ):
            raise ValueError(f"Table not found: {table_name}")

        offset = (page - 1) * page_size

        from constat.catalog.file.connector import FileConnector
        if isinstance(db_connection, FileConnector):
            import duckdb
            from pathlib import Path
            conn = duckdb.connect(":memory:")
            # Validate file_path is a real path (not injected SQL)
            file_path = str(Path(db_connection.path).resolve())
            ft = db_connection.file_type.value
            read_fns = {
                'csv': 'read_csv_auto',
                'tsv': 'read_csv_auto',
                'json': 'read_json_auto',
                'jsonl': 'read_json_auto',
                'parquet': 'read_parquet',
                'arrow': 'read_parquet',
                'feather': 'read_parquet',
            }
            fn_name = read_fns.get(ft, 'read_csv_auto')
            extra_args = ""
            if ft == 'tsv':
                extra_args = ", delim='\\t'"
            elif ft == 'jsonl':
                extra_args = ", format='newline_delimited'"
            read_fn = f"{fn_name}(?{extra_args})"
            df = conn.execute(f"SELECT * FROM {read_fn} LIMIT ? OFFSET ?", [file_path, page_size, offset]).df()
            total_rows = conn.execute(f"SELECT COUNT(*) as cnt FROM {read_fn}", [file_path]).fetchone()[0]
            conn.close()
        else:
            # Use identifier quoting via schema_manager validation above
            safe_name = table_name.replace('"', '""')
            query = f'SELECT * FROM "{safe_name}" LIMIT {int(page_size)} OFFSET {int(offset)}'
            df = pd.read_sql(query, db_connection)
            count_query = f'SELECT COUNT(*) as cnt FROM "{safe_name}"'
            count_df = pd.read_sql(count_query, db_connection)
            total_rows = int(count_df.iloc[0]["cnt"])

        return DatabaseTablePreviewType(
            database=db_name,
            table_name=table_name,
            columns=list(df.columns),
            data=_sanitize_df_for_json(df),
            page=page,
            page_size=page_size,
            total_rows=total_rows,
            has_more=offset + len(df) < total_rows,
        )

    @strawberry.field
    async def document(self, info: Info, session_id: str, name: str) -> DocumentResultType:
        managed = _get_managed(info, session_id)
        if not managed.session.doc_tools:
            raise ValueError("Document tools not available")
        result = managed.session.doc_tools.get_document(name)
        if "error" in result:
            raise ValueError(result["error"])
        return DocumentResultType(
            name=name,
            content=result.get("content"),
            metadata=result.get("metadata"),
        )

    @strawberry.field
    async def user_sources(self, info: Info) -> UserSourcesType:
        user_id = info.context.user_id
        if not user_id:
            raise ValueError("Authentication required")
        from constat.server.routes.user_sources import _load_user_config
        config = _load_user_config(user_id)
        return UserSourcesType(
            databases={k: v for k, v in config.get("databases", {}).items() if v.get("source") == "user"},
            documents={k: v for k, v in config.get("documents", {}).items() if v.get("source") == "user"},
            apis={k: v for k, v in config.get("apis", {}).items() if v.get("source") == "user"},
        )


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def upload_file(
        self, info: Info, session_id: str, file: Upload,
    ) -> UploadedFileType:
        import os
        import uuid
        managed = _get_managed(info, session_id)
        from constat.server.routes.files import (
            _get_upload_dir_for_session,
            _get_uploaded_files_for_session,
            _save_uploaded_files_for_session,
        )

        file_id = f"f_{uuid.uuid4().hex[:12]}"
        upload_dir = _get_upload_dir_for_session(managed)
        safe_filename = os.path.basename(getattr(file, "filename", None) or "upload")
        file_path = upload_dir / f"{file_id}_{safe_filename}"

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        now = datetime.now(timezone.utc)
        file_info = {
            "id": file_id,
            "filename": safe_filename,
            "file_path": str(file_path),
            "file_uri": f"file://{file_path}",
            "size_bytes": len(content),
            "content_type": getattr(file, "content_type", None) or "application/octet-stream",
            "uploaded_at": now.isoformat(),
        }

        files = _get_uploaded_files_for_session(managed)
        files.append(file_info)
        _save_uploaded_files_for_session(managed, files)

        return UploadedFileType(
            id=file_id,
            filename=safe_filename,
            file_uri=file_info["file_uri"],
            size_bytes=file_info["size_bytes"],
            content_type=file_info["content_type"],
            uploaded_at=now,
        )

    @strawberry.mutation
    async def upload_file_data_uri(
        self, info: Info, session_id: str, filename: str, data_uri: str,
    ) -> UploadedFileType:
        import base64
        import os
        import uuid
        managed = _get_managed(info, session_id)
        from constat.server.routes.files import (
            _get_upload_dir_for_session,
            _get_uploaded_files_for_session,
            _save_uploaded_files_for_session,
        )

        data = data_uri
        if data.startswith("data:"):
            parts = data.split(",", 1)
            if len(parts) != 2:
                raise ValueError("Invalid data URI format")
            encoded = parts[1]
            # Extract content type from data URI header
            header = parts[0]  # e.g. "data:text/csv;base64"
            content_type = header.split(":")[1].split(";")[0] if ":" in header else "application/octet-stream"
        else:
            encoded = data
            content_type = "application/octet-stream"

        try:
            content = base64.b64decode(encoded)
        except Exception as e:
            raise ValueError(f"Invalid base64 data: {e}")

        file_id = f"f_{uuid.uuid4().hex[:12]}"
        upload_dir = _get_upload_dir_for_session(managed)
        safe_filename = os.path.basename(filename)
        file_path = upload_dir / f"{file_id}_{safe_filename}"

        with open(file_path, "wb") as f:
            f.write(content)

        now = datetime.now(timezone.utc)
        file_info = {
            "id": file_id,
            "filename": safe_filename,
            "file_path": str(file_path),
            "file_uri": f"file://{file_path}",
            "size_bytes": len(content),
            "content_type": content_type,
            "uploaded_at": now.isoformat(),
        }

        files = _get_uploaded_files_for_session(managed)
        files.append(file_info)
        _save_uploaded_files_for_session(managed, files)

        return UploadedFileType(
            id=file_id,
            filename=safe_filename,
            file_uri=file_info["file_uri"],
            size_bytes=file_info["size_bytes"],
            content_type=file_info["content_type"],
            uploaded_at=now,
        )

    @strawberry.mutation
    async def delete_file(
        self, info: Info, session_id: str, file_id: str,
    ) -> DeleteResultType:
        from pathlib import Path
        managed = _get_managed(info, session_id)
        sm = info.context.session_manager
        from constat.server.routes.files import (
            _get_uploaded_files_for_session,
            _save_uploaded_files_for_session,
        )

        files = _get_uploaded_files_for_session(managed)
        file_info = next((f for f in files if f["id"] == file_id), None)
        if not file_info:
            raise ValueError(f"File not found: {file_id}")

        file_path = Path(file_info["file_path"])
        if file_path.exists():
            file_path.unlink()

        files = [f for f in files if f["id"] != file_id]
        _save_uploaded_files_for_session(managed, files)
        sm.refresh_entities_async(session_id)

        return DeleteResultType(status="deleted", name=file_id)

    @strawberry.mutation
    async def add_file_ref(
        self, info: Info, session_id: str, input: FileRefInput,
    ) -> FileRefType:
        managed = _get_managed(info, session_id)
        sm = info.context.session_manager

        if hasattr(managed.session, "add_file"):
            managed.session.add_file(
                name=input.name,
                uri=input.uri,
                auth=input.auth,
                description=input.description,
            )

        now = datetime.now(timezone.utc)
        managed._file_refs.append({
            "name": input.name,
            "uri": input.uri,
            "has_auth": input.auth is not None,
            "description": input.description,
            "added_at": now.isoformat(),
        })

        sm.refresh_entities_async(session_id)
        managed.save_resources()

        return FileRefType(
            name=input.name,
            uri=input.uri,
            has_auth=input.auth is not None,
            description=input.description,
            added_at=now,
            session_id=session_id,
        )

    @strawberry.mutation
    async def delete_file_ref(
        self, info: Info, session_id: str, name: str,
    ) -> DeleteResultType:
        managed = _get_managed(info, session_id)
        sm = info.context.session_manager

        original_len = len(managed._file_refs)
        managed._file_refs = [ref for ref in managed._file_refs if ref["name"] != name]
        if len(managed._file_refs) == original_len:
            raise ValueError(f"File reference not found: {name}")

        if hasattr(managed.session, 'session_files') and name in managed.session.session_files:
            del managed.session.session_files[name]

        chunks_deleted = 0
        if managed.session.doc_tools and hasattr(managed.session.doc_tools, '_vector_store'):
            vector_store = managed.session.doc_tools._vector_store
            doc_name = f"session:{name}"
            chunks_deleted = vector_store.delete_document(doc_name, session_id)

        sm.refresh_entities_async(session_id)
        managed.save_resources()
        managed._remove_doc_from_user_config(managed.user_id, name)

        return DeleteResultType(status="deleted", name=name)

    @strawberry.field
    async def validate_uri(self, info: Info, uri: str) -> UriValidationResult:
        """Check whether a URI is reachable (web URL) or exists on disk (file path)."""
        if uri.startswith("http://") or uri.startswith("https://"):
            try:
                req = urllib.request.Request(uri, method="HEAD",
                                             headers={"User-Agent": "constat/1.0"})
                with urllib.request.urlopen(req, timeout=8):
                    pass
                return UriValidationResult(reachable=True)
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    # 4xx means the server responded — URI is reachable even if auth-gated
                    return UriValidationResult(reachable=True)
                return UriValidationResult(reachable=False, error=f"HTTP {e.code}: {e.reason}")
            except urllib.error.URLError as e:
                return UriValidationResult(reachable=False, error=str(e.reason))
            except Exception as e:
                return UriValidationResult(reachable=False, error=str(e))
        else:
            exists = os.path.exists(uri)
            return UriValidationResult(
                reachable=exists,
                error=None if exists else f"Path does not exist: {uri}",
            )
