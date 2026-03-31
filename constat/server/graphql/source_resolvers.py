# Copyright (c) 2025 Kenneth Stott
# Canary: 82e1602f-0135-477c-865e-ebca45eb9d84
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for data sources: files, databases, APIs, documents, user sources."""

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

logger = logging.getLogger(__name__)


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

    @strawberry.mutation
    async def add_database(
        self, info: Info, session_id: str, input: DatabaseAddInput,
    ) -> SessionDatabaseType:
        managed = _get_managed(info, session_id)
        sm = info.context.session_manager

        uri = input.uri
        if input.file_id:
            from constat.server.routes.files import _get_uploaded_files_for_session
            files = _get_uploaded_files_for_session(managed)
            file_info = next((f for f in files if f["id"] == input.file_id), None)
            if not file_info:
                raise ValueError(f"File not found: {input.file_id}")
            uri = file_info["file_uri"]

        if not uri:
            raise ValueError("Either uri or file_id is required")

        uri_lower = uri.lower()
        file_path = uri[7:] if uri_lower.startswith("file://") else uri

        if file_path.endswith('.xlsx'):
            raise ValueError(
                "Excel files (.xlsx) cannot be added as databases. "
                "Use 'Add Document' or convert to CSV/Parquet."
            )

        if file_path.endswith('.json'):
            import json
            from pathlib import Path
            json_path = Path(file_path)
            if json_path.exists():
                try:
                    data = json.loads(json_path.read_text())
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON file: {e}")
                if not isinstance(data, list):
                    raise ValueError("JSON file must contain an array of objects")
                if data and not isinstance(data[0], dict):
                    raise ValueError("JSON array must contain objects")

        connected = False
        table_count = 0
        dialect = None

        effective_type = input.type
        file_extensions = {'.csv': 'csv', '.tsv': 'csv', '.parquet': 'parquet', '.json': 'json', '.jsonl': 'jsonl'}
        for ext, ftype in file_extensions.items():
            if file_path.lower().endswith(ext):
                effective_type = ftype
                break

        if hasattr(managed.session, "add_database"):
            try:
                managed.session.add_database(
                    name=input.name,
                    uri=uri,
                    db_type=effective_type,
                    description=input.description,
                )
                connected = True

                if managed.session.schema_manager:
                    from constat.core.config import DatabaseConfig
                    is_file_source = effective_type in ('csv', 'json', 'jsonl', 'parquet', 'arrow', 'feather')
                    db_config = DatabaseConfig(
                        type=effective_type,
                        path=uri if is_file_source else None,
                        uri=uri if not is_file_source else None,
                        description=input.description,
                    )
                    managed.session.schema_manager.add_database_dynamic(input.name, db_config)
                    table_count = sum(1 for k in managed.session.schema_manager.metadata_cache if k.startswith(f"{input.name}."))
            except Exception as e:
                raise ValueError(f"Failed to add database '{input.name}': {e}")

        if "postgresql" in uri or "postgres" in uri:
            dialect = "postgresql"
        elif "mysql" in uri:
            dialect = "mysql"
        elif "sqlite" in uri:
            dialect = "sqlite"
        elif "duckdb" in uri:
            dialect = "duckdb"
        elif "mssql" in uri:
            dialect = "mssql"

        now = datetime.now(timezone.utc)
        managed._dynamic_dbs.append({
            "name": input.name,
            "type": input.type,
            "dialect": dialect,
            "description": input.description,
            "uri": uri,
            "connected": connected,
            "table_count": table_count,
            "added_at": now.isoformat(),
            "is_dynamic": True,
            "file_id": input.file_id,
        })

        sm.resolve_config(session_id)
        if connected:
            sm.refresh_entities_async(session_id)
        managed.save_resources()

        return SessionDatabaseType(
            name=input.name,
            type=input.type,
            dialect=dialect,
            description=input.description,
            connected=connected,
            table_count=table_count,
            added_at=now,
            is_dynamic=True,
            file_id=input.file_id,
        )

    @strawberry.mutation
    async def remove_database(
        self, info: Info, session_id: str, name: str,
    ) -> DeleteResultType:
        managed = _get_managed(info, session_id)
        sm = info.context.session_manager

        if name in managed.session.config.databases:
            raise ValueError("Cannot remove config-defined database")

        for domain_filename in managed.active_domains:
            domain = managed.session.config.load_domain(domain_filename)
            if domain and name in domain.databases:
                raise ValueError(f"Cannot remove domain-defined database (from {domain_filename})")

        dynamic_dbs = managed._dynamic_dbs
        db_to_remove = next((db for db in dynamic_dbs if db["name"] == name), None)
        if not db_to_remove:
            raise ValueError(f"Database not found: {name}")

        uri = db_to_remove.get("uri", "")
        file_path = None
        if uri and not uri.startswith(("postgresql", "mysql", "sqlite", "mssql", "mongodb")):
            file_path = uri[7:] if uri.startswith("file://") else uri

        managed._dynamic_dbs = [db for db in dynamic_dbs if db["name"] != name]

        if name in managed.session.session_databases:
            del managed.session.session_databases[name]

        if managed.session.schema_manager:
            managed.session.schema_manager.remove_database_dynamic(name)

        if file_path:
            from pathlib import Path
            fp = Path(file_path)
            if fp.exists():
                try:
                    fp.unlink()
                except OSError as e:
                    raise ValueError(f"Failed to delete file {file_path}: {e}")

        from constat.server.session_manager import ManagedSession
        ManagedSession._remove_db_from_user_config(managed.user_id, name)

        sm.resolve_config(session_id)
        sm.refresh_entities_async(session_id)
        managed.save_resources()

        return DeleteResultType(status="deleted", name=name)

    @strawberry.mutation
    async def test_database(
        self, info: Info, session_id: str, name: str,
    ) -> DatabaseTestResultType:
        managed = _get_managed(info, session_id)
        if not managed.has_database(name):
            raise ValueError(f"Database not found: {name}")

        connected = False
        table_count = 0
        error = None

        try:
            if managed.session.schema_manager:
                tables = managed.session.schema_manager.get_tables_for_db(name)
                table_count = len(tables)
                connected = True
        except Exception as e:
            error = str(e)

        return DatabaseTestResultType(
            name=name,
            connected=connected,
            table_count=table_count,
            error=error,
        )

    @strawberry.mutation
    async def add_api(
        self, info: Info, session_id: str, input: ApiAddInput,
    ) -> SessionApiType:
        managed = _get_managed(info, session_id)
        sm = info.context.session_manager

        if any(api["name"] == input.name for api in managed._dynamic_apis):
            raise ValueError(f"API already exists: {input.name}")
        if input.name in managed.session.config.apis:
            raise ValueError(f"API already exists in config: {input.name}")

        now = datetime.now(timezone.utc)
        managed._dynamic_apis.append({
            "name": input.name,
            "type": input.type,
            "base_url": input.base_url,
            "description": input.description,
            "auth_type": input.auth_type,
            "auth_header": input.auth_header,
            "connected": True,
            "added_at": now.isoformat(),
            "is_dynamic": True,
        })

        managed.session.resources.add_api(
            name=input.name,
            description=input.description or "",
            api_type=input.type,
            source="session",
        )

        if managed.session.api_schema_manager:
            from constat.core.config import APIConfig
            api_config = APIConfig(
                type=input.type,
                url=input.base_url,
                description=input.description or "",
            )
            if input.auth_type and input.auth_header:
                api_config.headers = {input.auth_header: ""}
            managed.session.api_schema_manager.add_api_dynamic(input.name, api_config)

        sm.resolve_config(session_id)
        sm.refresh_entities_async(session_id)
        managed.save_resources()

        return SessionApiType(
            name=input.name,
            type=input.type,
            description=input.description,
            base_url=input.base_url,
            connected=True,
            from_config=False,
            source="session",
            is_dynamic=True,
        )

    @strawberry.mutation
    async def remove_api(
        self, info: Info, session_id: str, name: str,
    ) -> DeleteResultType:
        managed = _get_managed(info, session_id)
        sm = info.context.session_manager

        if name in managed.session.config.apis:
            raise ValueError("Cannot remove config-defined API")

        dynamic_apis = managed._dynamic_apis
        api_to_remove = next((api for api in dynamic_apis if api["name"] == name), None)
        if not api_to_remove:
            raise ValueError(f"API not found: {name}")

        managed._dynamic_apis = [api for api in dynamic_apis if api["name"] != name]
        managed.session.resources.remove_api(name)

        sm.resolve_config(session_id)
        sm.refresh_entities_async(session_id)
        managed.save_resources()

        return DeleteResultType(status="deleted", name=name)

    @strawberry.mutation
    async def add_document_uri(
        self, info: Info, session_id: str, input: DocumentUriInput,
    ) -> UserSourceResultType:
        managed = _get_managed(info, session_id)
        sm = info.context.session_manager

        session = managed.session
        if not hasattr(session, "doc_tools") or not session.doc_tools:
            raise ValueError("Document tools not available")

        from constat.core.config import DocumentConfig
        doc_config = DocumentConfig(
            url=input.url,
            description=input.description,
            headers=dict(input.headers) if input.headers else {},
            follow_links=input.follow_links,
            max_depth=input.max_depth,
            max_documents=input.max_documents,
            same_domain_only=input.same_domain_only,
            exclude_patterns=input.exclude_patterns,
            type=input.type,
        )

        managed._file_refs.append({
            "name": input.name,
            "uri": input.url,
            "has_auth": bool(input.headers),
            "description": input.description,
            "added_at": datetime.now(timezone.utc).isoformat(),
            "document_config": doc_config.model_dump(exclude_defaults=True),
        })
        managed.save_resources()

        from constat.server.routes.files import _ingest_source_async
        _ingest_source_async(sm, managed, input.name, doc_config, session_id)

        return UserSourceResultType(status="accepted", name=input.name)

    @strawberry.mutation
    async def add_email_source(
        self, info: Info, session_id: str, input: EmailSourceInput,
    ) -> UserSourceResultType:
        managed = _get_managed(info, session_id)
        sm = info.context.session_manager

        session = managed.session
        if not hasattr(session, "doc_tools") or not session.doc_tools:
            raise ValueError("Document tools not available")

        from constat.core.config import DocumentConfig
        doc_config = DocumentConfig(
            url=input.url,
            username=input.username,
            password=input.password,
            auth_type=input.auth_type,
            mailbox=input.mailbox,
            since=input.since,
            max_messages=input.max_messages,
            include_headers=input.include_headers,
            extract_attachments=input.extract_attachments,
            extract_images=True,
        )

        if input.oauth2_refresh_token:
            server_config = info.context.server_config
            if input.oauth2_tenant_id or (input.url and any(k in input.url.lower() for k in ('outlook', 'office365', 'microsoft'))):
                doc_config.oauth2_client_id = server_config.microsoft_email_client_id
                doc_config.oauth2_client_secret = input.oauth2_refresh_token
                doc_config.oauth2_tenant_id = input.oauth2_tenant_id or server_config.microsoft_email_tenant_id
            else:
                doc_config.oauth2_client_id = server_config.google_email_client_id
                doc_config.oauth2_client_secret = input.oauth2_refresh_token
                doc_config.password = server_config.google_email_client_secret
            doc_config.auth_type = "oauth2_refresh"

        managed._file_refs.append({
            "name": input.name,
            "uri": input.url,
            "has_auth": True,
            "description": f"IMAP email source ({input.mailbox})",
            "added_at": datetime.now(timezone.utc).isoformat(),
            "document_config": doc_config.model_dump(exclude_defaults=True),
        })
        managed.save_resources()

        from constat.server.routes.files import _ingest_source_async
        _ingest_source_async(sm, managed, input.name, doc_config, session_id)

        return UserSourceResultType(status="accepted", name=input.name)

    @strawberry.mutation
    async def upload_documents(
        self, info: Info, session_id: str, files: list[Upload],
    ) -> UploadDocumentsResultType:
        from pathlib import Path
        managed = _get_managed(info, session_id)
        sm = info.context.session_manager
        from constat.server.routes.files import _get_upload_dir_for_session

        doc_extensions = {'.md', '.txt', '.pdf', '.docx', '.html', '.htm', '.pptx', '.xlsx',
                          '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.webp', '.bmp', '.gif', '.svg'}
        data_extensions = {'.csv', '.tsv', '.parquet', '.json'}

        upload_dir = _get_upload_dir_for_session(managed)
        results = []

        for file in files:
            filename = getattr(file, "filename", None) or "upload"
            suffix = Path(filename).suffix.lower()
            is_document = suffix in doc_extensions
            is_data_file = suffix in data_extensions

            if not is_document and not is_data_file:
                results.append(UploadDocumentResultItem(filename=filename, status="skipped", reason=f"Unsupported format: {suffix}"))
                continue

            try:
                safe_name = Path(filename).name
                file_path = upload_dir / safe_name
                counter = 1
                original_stem = file_path.stem
                while file_path.exists():
                    file_path = upload_dir / f"{original_stem}_{counter}{suffix}"
                    counter += 1

                content = await file.read()
                file_path.write_bytes(content)
                name = file_path.stem
                uri = f"file://{file_path}"
                now = datetime.now(timezone.utc)

                if is_data_file:
                    if suffix == '.json':
                        import json as json_module
                        data = json_module.loads(content.decode('utf-8'))
                        if not isinstance(data, list):
                            results.append(UploadDocumentResultItem(filename=filename, status="error", reason="JSON must be an array of objects"))
                            continue
                        if data and not isinstance(data[0], dict):
                            results.append(UploadDocumentResultItem(filename=filename, status="error", reason="JSON array must contain objects"))
                            continue

                    file_type = suffix.lstrip('.')
                    if file_type == 'tsv':
                        file_type = 'csv'

                    managed.session.add_database(
                        name=name,
                        uri=str(file_path),
                        db_type=file_type,
                        description=f"Uploaded data file: {filename}",
                    )

                    table_count = 1
                    if managed.session.schema_manager:
                        from constat.core.config import DatabaseConfig
                        db_config = DatabaseConfig(
                            type=file_type,
                            uri=str(file_path),
                            description=f"Uploaded data file: {filename}",
                        )
                        managed.session.schema_manager.add_database_dynamic(name, db_config)
                        table_count = sum(1 for k in managed.session.schema_manager.metadata_cache if k.startswith(f"{name}."))

                    managed._dynamic_dbs.append({
                        "name": name,
                        "type": file_type,
                        "dialect": "duckdb",
                        "description": f"Uploaded data file: {filename}",
                        "uri": str(file_path),
                        "connected": True,
                        "table_count": table_count,
                        "added_at": now.isoformat(),
                        "is_dynamic": True,
                        "file_id": None,
                    })

                    results.append(UploadDocumentResultItem(filename=filename, name=name, status="database", path=str(file_path)))
                else:
                    managed._file_refs.append({
                        "name": name,
                        "uri": uri,
                        "has_auth": False,
                        "description": f"Uploaded: {filename}",
                        "added_at": now.isoformat(),
                    })

                    from constat.core.config import DocumentConfig
                    doc_config = DocumentConfig(url=uri, description=f"Uploaded: {filename}")
                    from constat.server.routes.files import _ingest_source_async
                    _ingest_source_async(sm, managed, name, doc_config, session_id)

                    results.append(UploadDocumentResultItem(filename=filename, name=name, status="accepted", path=str(file_path)))

            except (OSError, json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
                logger.error(f"Error uploading {filename}: {e}")
                results.append(UploadDocumentResultItem(filename=filename, status="error", reason=str(e)))

        if not results:
            raise ValueError("No files provided")

        database_count = sum(1 for r in results if r.status == "database")
        accepted_count = sum(1 for r in results if r.status == "accepted")

        if database_count > 0:
            managed.save_resources()
            sm.refresh_entities_async(session_id)

        return UploadDocumentsResultType(
            status="success",
            accepted_count=accepted_count,
            database_count=database_count,
            total_files=len(files),
            results=results,
        )

    @strawberry.mutation
    async def refresh_documents(
        self, info: Info, session_id: str,
    ) -> UserSourceResultType:
        managed = _get_managed(info, session_id)
        sm = info.context.session_manager

        if not managed._file_refs:
            return UserSourceResultType(status="skipped", name="no_sources")

        import asyncio
        from constat.server.source_refresher import refresh_session_sources

        for ref in managed._file_refs:
            ref.pop("last_refreshed", None)

        asyncio.get_event_loop().run_in_executor(
            None,
            refresh_session_sources,
            managed,
            sm,
            0,
        )

        return UserSourceResultType(status="started", name="all")

    @strawberry.mutation
    async def remove_user_source(
        self, info: Info, source_type: str, source_name: str,
    ) -> UserSourceResultType:
        user_id = info.context.user_id
        if not user_id:
            raise ValueError("Authentication required")
        if source_type not in ("databases", "documents", "apis"):
            raise ValueError(f"Invalid source type: {source_type}")

        from constat.server.routes.user_sources import _load_user_config, _save_user_config
        config = _load_user_config(user_id)
        section = config.get(source_type, {})
        if source_name not in section:
            raise ValueError(f"Source '{source_name}' not found in {source_type}")

        del section[source_name]
        config[source_type] = section
        _save_user_config(user_id, config)
        return UserSourceResultType(status="removed", name=source_name, source_type=source_type)

    @strawberry.mutation
    async def move_source(
        self, info: Info, source_type: str, source_name: str,
        from_domain: str, to_domain: str, session_id: str | None = None,
    ) -> MoveSourceResultType:
        user_id = info.context.user_id
        if not user_id:
            raise ValueError("Authentication required")
        if source_type not in ("databases", "documents", "apis"):
            raise ValueError(f"Invalid source type: {source_type}")

        from constat.server.routes.user_sources import _load_user_config, _save_user_config
        config = _load_user_config(user_id)
        section = config.get(source_type, {})

        if source_name not in section:
            section[source_name] = {}

        section[source_name]["source"] = "user"
        section[source_name]["domain"] = to_domain
        config[source_type] = section
        _save_user_config(user_id, config)
        return MoveSourceResultType(status="moved", name=source_name, source_type=source_type)
