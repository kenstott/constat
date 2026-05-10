# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""GraphQL resolvers for database and API source mutations."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
import strawberry

from constat.server.graphql.session_context import GqlInfo as Info
from constat.server.graphql.source_resolvers import _get_managed
from constat.server.graphql.types import (
    ApiAddInput,
    ApiUpdateInput,
    DatabaseAddInput,
    DatabaseUpdateInput,
    DatabaseTestResultType,
    DeleteResultType,
    MoveSourceResultType,
    SessionApiType,
    SessionDatabaseType,
)

logger = logging.getLogger(__name__)


@strawberry.type
class Query:
    @strawberry.field
    async def source_database_placeholder(self) -> str:
        return ""


@strawberry.type
class Mutation:
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

        is_jdbc = input.type == "jdbc" or (input.extra_config or {}).get("jdbc_driver")
        if not uri and not is_jdbc:
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
                    uri=uri or "",
                    db_type=effective_type,
                    description=input.description,
                )
                connected = True

                if managed.session.schema_manager:
                    from constat.core.config import DatabaseConfig
                    is_file_source = effective_type in ('csv', 'json', 'jsonl', 'parquet', 'arrow', 'feather')
                    extra = dict(input.extra_config) if input.extra_config else {}
                    # Normalize type: 'sqlalchemy' is a UI alias for 'sql'
                    config_type = "sql" if effective_type in ("sqlalchemy", "sql") else effective_type
                    db_config = DatabaseConfig(
                        type=config_type,
                        path=uri if is_file_source else None,
                        uri=uri if (not is_file_source and config_type != "jdbc") else None,
                        description=input.description or "",
                        **extra,
                    )
                    managed.session.schema_manager.add_database_dynamic(input.name, db_config)
                    table_count = sum(1 for k in managed.session.schema_manager.metadata_cache if k.startswith(f"{input.name}."))
            except Exception as e:
                raise ValueError(f"Failed to add database '{input.name}': {e}")

        if effective_type == "jdbc":
            dialect = "jdbc"
        elif "postgresql" in (uri or "") or "postgres" in (uri or ""):
            dialect = "postgresql"
        elif "mysql" in (uri or ""):
            dialect = "mysql"
        elif "sqlite" in (uri or ""):
            dialect = "sqlite"
        elif "duckdb" in (uri or ""):
            dialect = "duckdb"
        elif "mssql" in (uri or ""):
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
            if input.auth_type == "bearer" and input.auth_token:
                api_config.headers = {"Authorization": f"Bearer {input.auth_token}"}
            elif input.auth_type == "basic" and input.auth_username:
                import base64
                creds = base64.b64encode(f"{input.auth_username}:{input.auth_password or ''}".encode()).decode()
                api_config.headers = {"Authorization": f"Basic {creds}"}
            elif input.auth_type == "api_key" and input.auth_header and input.auth_token:
                api_config.headers = {input.auth_header: input.auth_token}
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
    async def update_database(
        self, info: Info, session_id: str, input: DatabaseUpdateInput,
    ) -> SessionDatabaseType:
        managed = _get_managed(info, session_id)
        sm = info.context.session_manager

        if input.name in managed.session.config.databases:
            raise ValueError("Cannot update config-defined database")

        for domain_filename in managed.active_domains:
            domain = managed.session.config.load_domain(domain_filename)
            if domain and input.name in domain.databases:
                raise ValueError(f"Cannot update domain-defined database (from {domain_filename})")

        db = next((d for d in managed._dynamic_dbs if d["name"] == input.name), None)
        if not db:
            raise ValueError(f"Database not found: {input.name}")

        if input.new_name is not None:
            db["name"] = input.new_name
        if input.description is not None:
            db["description"] = input.description
        if input.uri is not None:
            db["uri"] = input.uri
        if input.extra_config is not None:
            db["extra_config"] = dict(input.extra_config)

        sm.resolve_config(session_id)
        managed.save_resources()

        return SessionDatabaseType(
            name=db["name"],
            type=db["type"],
            dialect=db.get("dialect"),
            description=db.get("description"),
            uri=db.get("uri"),
            connected=db.get("connected", False),
            table_count=db.get("table_count", 0),
            added_at=datetime.fromisoformat(db["added_at"]),
            is_dynamic=True,
            file_id=db.get("file_id"),
            source="session",
        )

    @strawberry.mutation
    async def update_api(
        self, info: Info, session_id: str, input: ApiUpdateInput,
    ) -> SessionApiType:
        managed = _get_managed(info, session_id)
        sm = info.context.session_manager

        if input.name in managed.session.config.apis:
            raise ValueError("Cannot update config-defined API")

        for domain_filename in managed.active_domains:
            domain = managed.session.config.load_domain(domain_filename)
            if domain and input.name in domain.apis:
                raise ValueError(f"Cannot update domain-defined API (from {domain_filename})")

        api = next((a for a in managed._dynamic_apis if a["name"] == input.name), None)
        if not api:
            raise ValueError(f"API not found: {input.name}")

        if input.new_name is not None:
            api["name"] = input.new_name
        if input.description is not None:
            api["description"] = input.description
        if input.base_url is not None:
            api["base_url"] = input.base_url
        if input.auth_type is not None:
            api["auth_type"] = input.auth_type
        if input.auth_header is not None:
            api["auth_header"] = input.auth_header

        if managed.session.api_schema_manager:
            from constat.core.config import APIConfig
            api_config = APIConfig(
                type=api.get("type", "rest"),
                url=api["base_url"],
                description=api.get("description") or "",
            )
            effective_auth_type = input.auth_type if input.auth_type is not None else api.get("auth_type")
            effective_token = input.auth_token if input.auth_token is not None else api.get("auth_token")
            effective_username = input.auth_username if input.auth_username is not None else api.get("auth_username")
            effective_password = input.auth_password if input.auth_password is not None else api.get("auth_password")
            effective_header = input.auth_header if input.auth_header is not None else api.get("auth_header")

            if effective_auth_type == "bearer" and effective_token:
                api_config.headers = {"Authorization": f"Bearer {effective_token}"}
            elif effective_auth_type == "basic" and effective_username:
                import base64
                creds = base64.b64encode(f"{effective_username}:{effective_password or ''}".encode()).decode()
                api_config.headers = {"Authorization": f"Basic {creds}"}
            elif effective_auth_type == "api_key" and effective_header and effective_token:
                api_config.headers = {effective_header: effective_token}

            managed.session.api_schema_manager.add_api_dynamic(api["name"], api_config)

        sm.resolve_config(session_id)
        managed.save_resources()

        return SessionApiType(
            name=api["name"],
            type=api.get("type"),
            description=api.get("description"),
            base_url=api.get("base_url"),
            connected=api.get("connected", True),
            from_config=False,
            source="session",
            is_dynamic=True,
        )

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
