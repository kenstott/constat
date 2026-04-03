# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for document source mutations."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
import strawberry
from strawberry.file_uploads import Upload

from constat.server.graphql.session_context import GqlInfo as Info
from constat.server.graphql.source_resolvers import _get_managed
from constat.server.graphql.types import (
    DocumentUriInput,
    EmailSourceInput,
    UploadDocumentResultItem,
    UploadDocumentsResultType,
    UserSourceResultType,
)

logger = logging.getLogger(__name__)


@strawberry.type
class Query:
    @strawberry.field
    async def source_document_placeholder(self) -> str:
        return ""


@strawberry.type
class Mutation:
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
