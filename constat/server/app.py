# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""FastAPI application factory for the Constat API server."""

import logging
import warnings
from contextlib import asynccontextmanager

# Configure logging for server module
# Only add handler if not already configured, and prevent duplicate logs
_constat_logger = logging.getLogger('constat')
if not any(isinstance(h, logging.StreamHandler) for h in _constat_logger.handlers):
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    _constat_logger.addHandler(_console_handler)
    _constat_logger.setLevel(logging.INFO)
# Always prevent propagation to root logger to avoid duplicate messages
_constat_logger.propagate = False

# Suppress multiprocessing resource_tracker warnings at shutdown
warnings.filterwarnings("ignore", message="resource_tracker:")
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from constat.core.config import Config
from constat.server.config import ServerConfig
from constat.server.session_manager import SessionManager
from constat.server.routes.queries import shutdown_executor_async

logger = logging.getLogger(__name__)


def _compute_config_hash(data: dict) -> str:
    """Compute a hash for a configuration dict."""
    import hashlib
    import json
    config_json = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_json.encode()).hexdigest()[:16]


# =============================================================================
# Source-Level Hashing (combined hash for all resources of a type)
# =============================================================================

def _compute_db_config_hash(databases: dict) -> str:
    """Compute a combined hash for all database configurations."""
    db_data = {}
    if databases:
        for db_name, db_config in sorted(databases.items()):
            db_data[db_name] = {
                "type": db_config.type or "",
                "uri": db_config.uri or "",
                "database": db_config.database or "",
                "path": db_config.path or "",
            }
    return _compute_config_hash(db_data)


def _compute_api_config_hash(apis: dict) -> str:
    """Compute a combined hash for all API configurations."""
    api_data = {}
    if apis:
        for api_name, api_config in sorted(apis.items()):
            api_data[api_name] = {
                "type": api_config.type or "",
                "url": api_config.url or "",
                "spec_url": api_config.spec_url or "",
                "spec_path": api_config.spec_path or "",
            }
    return _compute_config_hash(api_data)


def _compute_doc_config_hash(documents: dict) -> str:
    """Compute a combined hash for all document configurations."""
    doc_data = {}
    if documents:
        for doc_name, doc_config in sorted(documents.items()):
            doc_data[doc_name] = {
                "path": doc_config.path or "",
                "description": doc_config.description or "",
                "format": doc_config.format or "",
            }
    return _compute_config_hash(doc_data)


# =============================================================================
# Resource-Level Hashing (individual hash per resource for incremental updates)
# =============================================================================

def _compute_db_resource_hash(db_name: str, db_config) -> str:
    """Compute a content hash for a single database resource.

    Includes connection config. Schema introspection would be added here
    for detecting table/column changes.
    """
    data = {
        "name": db_name,
        "type": db_config.type or "",
        "uri": db_config.uri or "",
        "database": db_config.database or "",
        "path": db_config.path or "",
    }
    return _compute_config_hash(data)


def _compute_api_resource_hash(api_name: str, api_config) -> str:
    """Compute a content hash for a single API resource.

    For spec-based APIs, could include spec file hash for change detection.
    """
    data = {
        "name": api_name,
        "type": api_config.type or "",
        "url": api_config.url or "",
        "spec_url": api_config.spec_url or "",
        "spec_path": api_config.spec_path or "",
    }
    return _compute_config_hash(data)


def _compute_doc_resource_hash(doc_name: str, doc_config, config_dir: str | None) -> str:
    """Compute a content hash for a single document resource.

    Includes file modification time for fast change detection.
    """
    import os
    from pathlib import Path

    data = {
        "name": doc_name,
        "path": doc_config.path or "",
        "description": doc_config.description or "",
        "format": doc_config.format or "",
    }

    # Add file mtime for change detection
    if doc_config.path:
        doc_path = Path(doc_config.path)
        if not doc_path.is_absolute() and config_dir:
            doc_path = (Path(config_dir) / doc_config.path).resolve()

        if doc_path.exists():
            try:
                stat = os.stat(doc_path)
                data["mtime"] = stat.st_mtime
                data["size"] = stat.st_size
            except OSError:
                pass

    return _compute_config_hash(data)


def _warmup_vector_store(config: Config) -> None:
    """Pre-index all resources (schemas, APIs, documents) at server startup.

    Uses hash-based invalidation per source (base + domains) with three hash types:
    - db_hash: database schema configuration
    - api_hash: API configuration
    - doc_hash: document configuration
    """
    from pathlib import Path
    from constat.discovery.doc_tools import DocumentDiscoveryTools
    from constat.catalog.schema_manager import SchemaManager
    from constat.catalog.api_schema_manager import APISchemaManager

    # Create doc_tools early to access vector_store for hash checks
    doc_tools = DocumentDiscoveryTools(config, skip_auto_index=True)
    vector_store = doc_tools._vector_store

    # === BASE CONFIG ===
    source_id = "__base__"

    # Base databases
    if config.databases:
        db_hash = _compute_db_config_hash(config.databases)
        cached_hash = vector_store.get_source_hash(source_id, 'db')

        if cached_hash == db_hash:
            logger.info(f"  Base databases: {len(config.databases)} unchanged, skipping")
        else:
            logger.info(f"  Base databases: building chunks...")
            schema_manager = SchemaManager(config)
            schema_manager.build_chunks()
            vector_store.set_source_hash(source_id, 'db', db_hash)
            logger.info(f"  Base databases: {len(config.databases)} indexed")

    # Base APIs
    if config.apis:
        api_hash = _compute_api_config_hash(config.apis)
        cached_hash = vector_store.get_source_hash(source_id, 'api')

        if cached_hash == api_hash:
            logger.info(f"  Base APIs: {len(config.apis)} unchanged, skipping")
        else:
            logger.info(f"  Base APIs: building chunks...")
            api_manager = APISchemaManager(config)
            api_manager.build_chunks()
            vector_store.set_source_hash(source_id, 'api', api_hash)
            logger.info(f"  Base APIs: {len(config.apis)} indexed")

    # === DOMAIN CONFIGS ===
    for domain_name, domain in config.projects.items():
        source_id = domain_name

        # Domain databases
        if domain.databases:
            db_hash = _compute_db_config_hash(domain.databases)
            cached_hash = vector_store.get_source_hash(source_id, 'db')

            if cached_hash == db_hash:
                logger.info(f"  Domain {domain_name} databases: unchanged, skipping")
            else:
                logger.info(f"  Domain {domain_name} databases: building chunks...")
                domain_config = Config(
                    config_dir=config.config_dir,
                    databases=domain.databases,
                )
                domain_schema_manager = SchemaManager(domain_config)
                domain_schema_manager.build_chunks()
                vector_store.set_source_hash(source_id, 'db', db_hash)
                logger.info(f"  Domain {domain_name} databases: {len(domain.databases)} indexed")

        # Domain APIs
        if domain.apis:
            api_hash = _compute_api_config_hash(domain.apis)
            cached_hash = vector_store.get_source_hash(source_id, 'api')

            if cached_hash == api_hash:
                logger.info(f"  Domain {domain_name} APIs: unchanged, skipping")
            else:
                logger.info(f"  Domain {domain_name} APIs: building chunks...")
                domain_api_config = Config(
                    config_dir=config.config_dir,
                    apis=domain.apis,
                )
                domain_api_manager = APISchemaManager(domain_api_config)
                domain_api_manager.build_chunks()
                vector_store.set_source_hash(source_id, 'api', api_hash)
                logger.info(f"  Domain {domain_name} APIs: {len(domain.apis)} indexed")

    # === BASE DOCUMENTS (two-level hashing) ===
    if config.documents:
        doc_hash = _compute_doc_config_hash(config.documents)
        cached_hash = vector_store.get_source_hash("__base__", 'doc')
        config_dir = Path(config.config_dir) if config.config_dir else Path.cwd()

        if cached_hash == doc_hash:
            # Fast path: source-level hash unchanged, skip all documents
            logger.info(f"  Base documents: {len(config.documents)} unchanged, skipping")
        else:
            # Slow path: source changed, check each document individually
            logger.info(f"  Base documents: checking {len(config.documents)} documents...")
            cached_resource_hashes = vector_store.get_resource_hashes_for_source("__base__", "document")
            indexed_count = 0
            skipped_count = 0

            for doc_name, doc_config in config.documents.items():
                try:
                    # Compute resource-level hash
                    resource_hash = _compute_doc_resource_hash(doc_name, doc_config, config.config_dir)
                    cached_resource_hash = cached_resource_hashes.get(doc_name)

                    if cached_resource_hash == resource_hash:
                        # Document unchanged, skip
                        skipped_count += 1
                        continue

                    # Document changed or new - delete old chunks and re-index
                    if cached_resource_hash:
                        vector_store.delete_resource_chunks("__base__", "document", doc_name)
                        logger.debug(f"  Base: deleted old chunks for {doc_name}")

                    if doc_config.path:
                        doc_path = Path(doc_config.path)
                        if not doc_path.is_absolute():
                            doc_path = (config_dir / doc_config.path).resolve()

                        if doc_path.exists():
                            success, msg = doc_tools.add_document_from_file(
                                str(doc_path),
                                name=doc_name,
                                description=doc_config.description or "",
                                domain_id="__base__",
                                skip_entity_extraction=True,  # NER done at session creation
                            )
                            if success:
                                vector_store.set_resource_hash("__base__", "document", doc_name, resource_hash)
                                indexed_count += 1
                                logger.info(f"  Base: vectorized {doc_name}")
                            else:
                                logger.warning(f"  Base: failed to vectorize {doc_name}: {msg}")
                        else:
                            logger.warning(f"  Base: file not found: {doc_path}")
                except Exception as e:
                    logger.warning(f"  Base: error indexing {doc_name}: {e}")

            # Handle removed documents (in cache but not in config)
            for old_doc_name in cached_resource_hashes.keys():
                if old_doc_name not in config.documents:
                    vector_store.delete_resource_chunks("__base__", "document", old_doc_name)
                    vector_store.delete_resource_hash("__base__", "document", old_doc_name)
                    logger.info(f"  Base: removed deleted document {old_doc_name}")

            vector_store.set_source_hash("__base__", 'doc', doc_hash)
            logger.info(f"  Base documents: {indexed_count} indexed, {skipped_count} unchanged")
    else:
        logger.info(f"  Base documents: 0 configured")

    # === DOMAIN DOCUMENTS (two-level hashing) ===
    for filename, domain in config.projects.items():
        if not domain.documents:
            continue

        doc_hash = _compute_doc_config_hash(domain.documents)
        cached_hash = vector_store.get_source_hash(filename, 'doc')

        if cached_hash == doc_hash:
            # Fast path: source-level hash unchanged, skip all documents
            logger.info(f"  Domain {filename} documents: unchanged, skipping")
            continue

        # Slow path: source changed, check each document individually
        logger.info(f"  Domain {filename} documents: checking {len(domain.documents)} documents...")
        cached_resource_hashes = vector_store.get_resource_hashes_for_source(filename, "document")
        config_dir = Path(config.config_dir) if config.config_dir else Path.cwd()
        indexed_count = 0
        skipped_count = 0

        for doc_name, doc_config in domain.documents.items():
            try:
                # Compute resource-level hash
                resource_hash = _compute_doc_resource_hash(doc_name, doc_config, config.config_dir)
                cached_resource_hash = cached_resource_hashes.get(doc_name)

                if cached_resource_hash == resource_hash:
                    # Document unchanged, skip
                    skipped_count += 1
                    continue

                # Document changed or new - delete old chunks and re-index
                if cached_resource_hash:
                    vector_store.delete_resource_chunks(filename, "document", doc_name)
                    logger.debug(f"  Domain {filename}: deleted old chunks for {doc_name}")

                if doc_config.path:
                    doc_path = Path(doc_config.path)
                    if not doc_path.is_absolute():
                        doc_path = (config_dir / doc_config.path).resolve()

                    if doc_path.exists():
                        success, msg = doc_tools.add_document_from_file(
                            str(doc_path),
                            name=doc_name,
                            description=doc_config.description or "",
                            domain_id=filename,
                            skip_entity_extraction=True,  # NER done at session creation
                        )
                        if success:
                            vector_store.set_resource_hash(filename, "document", doc_name, resource_hash)
                            indexed_count += 1
                            logger.info(f"  Domain {filename}: vectorized {doc_name}")
                        else:
                            logger.warning(f"  Domain {filename}: failed to vectorize {doc_name}: {msg}")
                    else:
                        logger.warning(f"  Domain {filename}: file not found: {doc_path}")
            except Exception as e:
                logger.warning(f"  Domain {filename}: error indexing {doc_name}: {e}")

        # Handle removed documents (in cache but not in config)
        for old_doc_name in cached_resource_hashes.keys():
            if old_doc_name not in domain.documents:
                vector_store.delete_resource_chunks(filename, "document", old_doc_name)
                vector_store.delete_resource_hash(filename, "document", old_doc_name)
                logger.info(f"  Domain {filename}: removed deleted document {old_doc_name}")

        vector_store.set_source_hash(filename, 'doc', doc_hash)
        logger.info(f"  Domain {filename} documents: {indexed_count} indexed, {skipped_count} unchanged")

    logger.info("  Pre-indexing complete")


def create_app(config: Config, server_config: ServerConfig) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Main Constat configuration
        server_config: Server-specific configuration

    Returns:
        Configured FastAPI application
    """
    # Create session manager
    session_manager = SessionManager(config, server_config)

    @asynccontextmanager
    async def lifespan(_fastapi_app: FastAPI):
        """Manage application lifecycle."""
        try:
            # Startup: Pre-load embedding model (blocking - we need it for vectorization)
            from constat.embedding_loader import EmbeddingModelLoader
            logger.info("Loading embedding model...")
            EmbeddingModelLoader.get_instance().start_loading()
            EmbeddingModelLoader.get_instance().get_model()  # Wait for completion
            logger.info("Embedding model loaded")

            # Startup: Pre-index all documents from config and domains
            # This warms up the vector store so first session doesn't pay the cost
            logger.info("Pre-indexing documents from config and domains...")
            _warmup_vector_store(config)
            logger.info("Document pre-indexing complete")

            # Startup: Start cleanup task
            await session_manager.start_cleanup_task()
            logger.info("Constat API server started")
        except Exception as e:
            logger.error(f"FATAL: Server startup failed: {e}")
            logger.exception("Full traceback:")
            raise

        yield

        # Shutdown: Stop cleanup task and cleanup sessions
        try:
            logger.info("Shutting down Constat API server...")
            await shutdown_executor_async()  # Stop thread pool to allow clean exit
            await session_manager.stop_cleanup_task()
            for managed in session_manager.list_sessions():
                session_manager.delete_session(managed.session_id)
            logger.info("Constat API server stopped cleanly")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            logger.exception("Shutdown error traceback:")

    fastapi_app = FastAPI(
        title="Constat API",
        description="Multi-step AI reasoning engine API for data analysis",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store config and session manager on app state
    # noinspection PyUnresolvedReferences
    fastapi_app.state.config = config
    # noinspection PyUnresolvedReferences
    fastapi_app.state.server_config = server_config
    # noinspection PyUnresolvedReferences
    fastapi_app.state.session_manager = session_manager

    # Add CORS middleware
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=server_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @fastapi_app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle request validation errors with detailed logging."""
        logger.error(f"[VALIDATION ERROR] Path: {request.url.path}, Method: {request.method}")
        logger.error(f"[VALIDATION ERROR] Details: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )

    @fastapi_app.exception_handler(KeyError)
    async def key_error_handler(request, exc: KeyError) -> JSONResponse:
        """Handle KeyError (session not found, etc.)."""
        import traceback
        logger.error(f"KeyError in {request.method} {request.url.path}: {exc}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "message": str(exc)},
        )

    @fastapi_app.exception_handler(RuntimeError)
    async def runtime_error_handler(_request, exc: RuntimeError) -> JSONResponse:
        """Handle RuntimeError (limit exceeded, etc.)."""
        return JSONResponse(
            status_code=400,
            content={"error": "runtime_error", "message": str(exc)},
        )

    @fastapi_app.exception_handler(ValueError)
    async def value_error_handler(_request, exc: ValueError) -> JSONResponse:
        """Handle ValueError (invalid input, etc.)."""
        return JSONResponse(
            status_code=400,
            content={"error": "validation_error", "message": str(exc)},
        )

    # Health check endpoint
    @fastapi_app.get("/health")
    async def health() -> dict[str, Any]:
        """Health check endpoint.

        Returns:
            Health status and basic stats
        """
        return {
            "status": "ok",
            "sessions": session_manager.get_stats(),
            "auth": {
                "auth_disabled": server_config.auth_disabled,
                "firebase_project_id": server_config.firebase_project_id,
            },
        }

    # Debug auth endpoint
    from fastapi import Request as FastAPIRequest
    @fastapi_app.get("/debug/auth")
    async def debug_auth(request: FastAPIRequest) -> dict[str, Any]:
        """Debug endpoint to check auth configuration."""
        from constat.server.auth import FIREBASE_AVAILABLE

        # Get auth header if present
        auth_header = request.headers.get("Authorization", "")
        has_token = auth_header.startswith("Bearer ")

        return {
            "server_config": {
                "auth_disabled": server_config.auth_disabled,
                "firebase_project_id": server_config.firebase_project_id,
            },
            "request": {
                "has_auth_header": bool(auth_header),
                "has_bearer_token": has_token,
                "token_preview": auth_header[:50] + "..." if len(auth_header) > 50 else auth_header,
            },
            "firebase_available": FIREBASE_AVAILABLE,
        }

    # Import and include routers
    from constat.server.routes.sessions import router as sessions_router
    from constat.server.routes.queries import router as queries_router
    from constat.server.routes.data import router as data_router
    from constat.server.routes.schema import router as schema_router
    from constat.server.routes.files import router as files_router
    from constat.server.routes.databases import router as databases_router
    from constat.server.routes.learnings import router as learnings_router
    from constat.server.routes.users import router as users_router
    from constat.server.routes.roles import router as roles_router
    from constat.server.routes.skills import router as skills_router

    # IMPORTANT: Register routers with specific paths BEFORE routers with /{session_id} wildcards
    # Otherwise the wildcard routes will match paths like /roles, /skills, etc.
    fastapi_app.include_router(
        roles_router,
        prefix="/api/sessions",
        tags=["roles"],
    )
    fastapi_app.include_router(
        skills_router,
        prefix="/api",
        tags=["skills"],
    )
    fastapi_app.include_router(
        sessions_router,
        prefix="/api/sessions",
        tags=["sessions"],
    )
    fastapi_app.include_router(
        queries_router,
        prefix="/api/sessions",
        tags=["queries"],
    )
    fastapi_app.include_router(
        data_router,
        prefix="/api/sessions",
        tags=["data"],
    )
    fastapi_app.include_router(
        files_router,
        prefix="/api/sessions",
        tags=["files"],
    )
    fastapi_app.include_router(
        databases_router,
        prefix="/api/sessions",
        tags=["databases"],
    )
    fastapi_app.include_router(
        schema_router,
        prefix="/api/schema",
        tags=["schema"],
    )
    fastapi_app.include_router(
        learnings_router,
        prefix="/api",
        tags=["learnings"],
    )
    fastapi_app.include_router(
        users_router,
        prefix="/api/users",
        tags=["users"],
    )

    return fastapi_app


# Module-level app instance for uvicorn reload mode
# This is only used when running with --reload flag
_app: FastAPI | None = None


def get_app() -> FastAPI:
    """Get or create the FastAPI application.

    Used by uvicorn in reload mode.
    """
    global _app
    if _app is None:
        # Load default config for development
        import os
        import yaml
        from pathlib import Path
        from dotenv import load_dotenv

        config_path = os.environ.get("CONSTAT_CONFIG", "config.yaml")
        try:
            # Load .env file (same logic as Config.from_yaml)
            config_dir = Path(config_path).parent.resolve()
            search_dir = config_dir
            while search_dir != search_dir.parent:
                env_file = search_dir / ".env"
                if env_file.exists():
                    load_dotenv(env_file)
                    break
                search_dir = search_dir.parent
            else:
                load_dotenv()  # Fallback to current directory

            config = Config.from_yaml(config_path)

            # Load server config with $ref resolution
            from constat.core.config import _resolve_refs, _substitute_env_vars
            with open(config_path) as f:
                raw_content = f.read()

            substituted = _substitute_env_vars(raw_content)
            raw_data = yaml.safe_load(substituted)
            raw_data = _resolve_refs(raw_data, config_dir)
            server_data = raw_data.get("server") if raw_data else None
            server_config = ServerConfig.from_yaml_data(server_data)
        except FileNotFoundError:
            config = Config()
            server_config = ServerConfig()
        _app = create_app(config, server_config)
    return _app


# For uvicorn reload: `uvicorn constat.server.app:app --reload`
app = get_app()
