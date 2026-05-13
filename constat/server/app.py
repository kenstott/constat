# Copyright (c) 2025 Kenneth Stott
# Canary: d6ce0510-1451-48dc-964c-452ffad6c5e8
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""FastAPI application factory for the Constat API server."""

import asyncio
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


from constat.server._warmup_hashes import (
    compute_db_config_hash as _compute_db_config_hash,
    compute_api_config_hash as _compute_api_config_hash,
    compute_doc_config_hash as _compute_doc_config_hash,
    compute_db_resource_hash as _compute_db_resource_hash,
    compute_api_resource_hash as _compute_api_resource_hash,
    compute_er_config_hash as _compute_er_config_hash,
    compute_doc_resource_hash as _compute_doc_resource_hash,
)


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

    # Create LLM router for vision descriptions on image documents
    from constat.providers.router import TaskRouter
    router = TaskRouter(config.llm)

    # Create doc_tools early to access vector_store for hash checks
    # Warmup writes to .constat/system.duckdb (the system DB)
    doc_tools = DocumentDiscoveryTools(config, skip_auto_index=True, router=router)
    vector_store = doc_tools._vector_store

    # Accumulate FK relationship triples from all schema managers built during warmup.
    # Used for graph_first search mode when available.
    from chonk.graph import RelationshipIndex as _RelIdx
    _merged_rel_idx = _RelIdx()

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
            schema_manager.build_chunks(domain_id="__base__", vector_store=vector_store)
            for triples in schema_manager.relationship_index._by_subject.values():
                for ts in triples.values():
                    for t in ts:
                        _merged_rel_idx.add(t)
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
            api_manager.build_chunks(domain_id="__base__", vector_store=vector_store)
            vector_store.set_source_hash(source_id, 'api', api_hash)
            logger.info(f"  Base APIs: {len(config.apis)} indexed")

    # === DOMAIN CONFIGS ===
    for domain_name, domain in list(config.projects.items()):
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
                domain_schema_manager.build_chunks(domain_id=domain_name, vector_store=vector_store)
                for triples in domain_schema_manager.relationship_index._by_subject.values():
                    for ts in triples.values():
                        for t in ts:
                            _merged_rel_idx.add(t)
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
                domain_api_manager.build_chunks(domain_id=domain_name, vector_store=vector_store)
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
            cached_resource_hashes = vector_store.list_doc_hashes("__base__")
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
                            if doc_path.is_dir():
                                # Expand directory to individual files
                                from constat.discovery.doc_tools._schema_inference import _expand_file_paths
                                expanded = _expand_file_paths(str(doc_path))
                                dir_success = False
                                for filename_e, filepath_e in expanded:
                                    child_name = f"{doc_name}:{filename_e}"
                                    success, msg = doc_tools.add_document_from_file(
                                        str(filepath_e),
                                        name=child_name,
                                        description=doc_config.description or "",
                                        domain_id="__base__",
                                        skip_entity_extraction=True,
                                    )
                                    if success:
                                        indexed_count += 1
                                        dir_success = True
                                        logger.info(f"  Base: vectorized {child_name}")
                                    else:
                                        logger.warning(f"  Base: failed to vectorize {child_name}: {msg}")
                                if dir_success:
                                    vector_store.set_resource_hash("__base__", "document", doc_name, resource_hash)
                            else:
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
                    vector_store.delete_doc_hash("__base__", old_doc_name)
                    logger.info(f"  Base: removed deleted document {old_doc_name}")

            if indexed_count > 0:
                vector_store.set_source_hash("__base__", 'doc', doc_hash)
            logger.info(f"  Base documents: {indexed_count} indexed, {skipped_count} unchanged")

        # Persist image labels for NER (survives across doc_tools instances)
        # Merge with existing labels (from cached/unchanged images)
        if doc_tools._image_labels:
            existing = vector_store.get_entity_resolution_names(["__image_labels__"]).get("LABEL", [])
            all_labels = list(set(existing + doc_tools._image_labels))
            vector_store.store_entity_resolution_names("__image_labels__", {"LABEL": all_labels})
            logger.info(f"  Base: persisted {len(all_labels)} image labels for NER")
    else:
        logger.info(f"  Base documents: 0 configured")

    # === DOMAIN DOCUMENTS (two-level hashing) ===
    for filename, domain in list(config.projects.items()):
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
                        if doc_path.is_dir():
                            from constat.discovery.doc_tools._schema_inference import _expand_file_paths
                            expanded = _expand_file_paths(str(doc_path))
                            for filename_e, filepath_e in expanded:
                                child_name = f"{doc_name}:{filename_e}"
                                success, msg = doc_tools.add_document_from_file(
                                    str(filepath_e),
                                    name=child_name,
                                    description=doc_config.description or "",
                                    domain_id=filename,
                                    skip_entity_extraction=True,
                                )
                                if success:
                                    indexed_count += 1
                                    logger.info(f"  Domain {filename}: vectorized {child_name}")
                                else:
                                    logger.warning(f"  Domain {filename}: failed to vectorize {child_name}: {msg}")
                            vector_store.set_resource_hash(filename, "document", doc_name, resource_hash)
                        else:
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
                elif doc_config.url:
                    success, msg = doc_tools.add_document_from_config(
                        doc_name, doc_config,
                        domain_id=filename,
                        skip_entity_extraction=True,
                    )
                    if success:
                        vector_store.set_resource_hash(filename, "document", doc_name, resource_hash)
                        indexed_count += 1
                        logger.info(f"  Domain {filename}: vectorized {doc_name} (url)")
                    else:
                        logger.warning(f"  Domain {filename}: failed to vectorize {doc_name}: {msg}")
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

    # Persist any additional image labels from domain documents
    if doc_tools._image_labels:
        existing = vector_store.get_entity_resolution_names(["__image_labels__"]).get("LABEL", [])
        all_labels = list(set(existing + doc_tools._image_labels))
        if len(all_labels) > len(existing):
            vector_store.store_entity_resolution_names("__image_labels__", {"LABEL": all_labels})
            logger.info(f"  Persisted {len(all_labels)} total image labels for NER")

    # === ENTITY RESOLUTION ===
    # Base entity resolution
    if config.entity_resolution:
        er_hash = _compute_er_config_hash(config.entity_resolution, config.apis)
        cached_hash = vector_store.get_source_hash("__base__", 'er')
        if cached_hash != er_hash:
            logger.info("  Base entity resolution: extracting...")
            base_er_mgr = SchemaManager(config)
            try:
                base_er_mgr._connect_all()
            except Exception as e:
                logger.warning(f"  Base entity resolution: some DB connections failed (OK for API sources): {e}")
            entity_terms, entity_details = base_er_mgr.extract_entity_values(
                config.entity_resolution, api_configs=config.apis,
            )
            if entity_terms:
                doc_tools.embed_entity_values(
                    entity_terms, config.entity_resolution,
                    session_id=None, domain_id=None,
                    entity_details=entity_details,
                    api_configs=config.apis,
                )
                vector_store.store_entity_resolution_names("__base__", entity_terms)
                logger.info(f"  Base entity resolution: {sum(len(v) for v in entity_terms.values())} values cached")
            vector_store.set_source_hash("__base__", 'er', er_hash)
        else:
            logger.info("  Base entity resolution: unchanged, skipping")

    # Domain entity resolution
    for domain_name, domain in config.projects.items():
        if not domain.entity_resolution:
            continue
        er_hash = _compute_er_config_hash(domain.entity_resolution, domain.apis)
        cached_hash = vector_store.get_source_hash(domain_name, 'er')
        if cached_hash != er_hash:
            logger.info(f"  Domain {domain_name} entity resolution: extracting...")
            dm_config = Config(config_dir=config.config_dir, databases=domain.databases)
            dm_mgr = SchemaManager(dm_config)
            try:
                dm_mgr._connect_all()
            except Exception as e:
                logger.warning(f"  Domain {domain_name} entity resolution: some DB connections failed: {e}")
            entity_terms, entity_details = dm_mgr.extract_entity_values(
                domain.entity_resolution, api_configs=domain.apis,
            )
            if entity_terms:
                doc_tools.embed_entity_values(
                    entity_terms, domain.entity_resolution,
                    session_id=None, domain_id=domain_name,
                    entity_details=entity_details,
                    api_configs=domain.apis,
                )
                vector_store.store_entity_resolution_names(domain_name, entity_terms)
                logger.info(f"  Domain {domain_name} entity resolution: {sum(len(v) for v in entity_terms.values())} values cached")
            vector_store.set_source_hash(domain_name, 'er', er_hash)
        else:
            logger.info(f"  Domain {domain_name} entity resolution: unchanged, skipping")

    # Wire merged FK relationship index into doc_tools for graph_first search mode.
    if _merged_rel_idx._by_subject:
        doc_tools._relationship_index = _merged_rel_idx
        logger.info(f"  Relationship index: {len(_merged_rel_idx._by_subject)} subjects wired for graph_first mode")

    logger.info("  Pre-indexing complete")

    from constat.server._chonk_warmup import warmup_chonk_index
    warmup_chonk_index(config)


async def _shutdown_tasks(session_manager) -> None:
    """Run shutdown tasks with a bounded time budget."""
    await shutdown_executor_async()
    await session_manager.stop_cleanup_task()
    for managed in session_manager.list_sessions():
        session_manager.delete_session(managed.session_id)


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
            # Startup: Kick off embedding model load (non-blocking — loads in background thread)
            from constat.embedding_loader import EmbeddingModelLoader
            EmbeddingModelLoader.get_instance().start_loading()

            # Fine-tune manager is created lazily on first request
            # (avoids opening user vault DB at startup)
            _fastapi_app.state.fine_tune_manager = None

            # Startup: Start cleanup task
            await session_manager.start_cleanup_task()

            # Startup: Warmup (embedding model + document pre-indexing) runs in background
            # so the server accepts connections immediately
            _fastapi_app.state.warmup_complete = False
            _fastapi_app.state.warmup_error = None

            async def _warmup_task():
                try:
                    logger.info("Background warmup: loading embedding model...")
                    await asyncio.to_thread(EmbeddingModelLoader.get_instance().get_model)
                    logger.info("Background warmup: embedding model loaded")
                    logger.info("Background warmup: pre-indexing documents...")
                    await asyncio.to_thread(_warmup_vector_store, config)
                    logger.info("Background warmup: document pre-indexing complete")
                    _fastapi_app.state.warmup_complete = True
                except Exception as e:
                    logger.exception(f"Background warmup failed: {e}")
                    _fastapi_app.state.warmup_error = str(e)

            warmup_task = asyncio.create_task(_warmup_task())

            # Startup: Start fine-tune polling task
            async def _fine_tune_poll_loop():
                while True:
                    try:
                        await asyncio.sleep(60)
                        manager = getattr(_fastapi_app.state, "fine_tune_manager", None)
                        if manager is None:
                            continue  # No manager yet — skip until first request creates one
                        updated = await asyncio.to_thread(manager.check_all_training)
                        for model in updated:
                            if model.status in ("ready", "failed"):
                                logger.info(f"Fine-tune {model.name}: {model.status}")
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        logger.error(f"Fine-tune poll error: {e}")

            ft_poll_task = asyncio.create_task(_fine_tune_poll_loop())

            # Startup: Start source refresh loop
            from constat.server.source_refresher import source_refresh_loop
            refresh_interval = server_config.source_refresh_interval_seconds
            source_refresh_task = asyncio.create_task(
                source_refresh_loop(session_manager, refresh_interval)
            )

            logger.info("Constat API server started (warmup running in background)")
        except Exception as e:
            logger.error(f"FATAL: Server startup failed: {e}")
            logger.exception("Full traceback:")
            raise

        yield

        # Shutdown: Stop background tasks and cleanup sessions
        try:
            warmup_task.cancel()
            ft_poll_task.cancel()
            source_refresh_task.cancel()
            logger.info("Shutting down Constat API server...")
            await asyncio.wait_for(_shutdown_tasks(session_manager), timeout=5.0)
            logger.info("Constat API server stopped cleanly")
        except asyncio.TimeoutError:
            logger.warning("Shutdown timed out after 5s, forcing exit")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            logger.exception("Shutdown error traceback:")
        finally:
            from constat.storage.duckdb_pool import close_all_pools
            close_all_pools()
            try:
                import jpype
                if jpype.isJVMStarted():
                    jpype.shutdownJVM()
            except Exception:
                pass

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

    # Load persona definitions
    from constat.server.persona_config import load_personas_config
    fastapi_app.state.personas_config = load_personas_config()

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
            "warmup_complete": getattr(fastapi_app.state, "warmup_complete", False),
            "warmup_error": getattr(fastapi_app.state, "warmup_error", None),
            "sessions": session_manager.get_stats(),
            "auth": {
                "auth_disabled": server_config.auth_disabled,
                "firebase_project_id": server_config.firebase_project_id,
                "auth_methods": (
                    (["local"] if server_config.local_users else [])
                    + (["firebase"] if server_config.firebase_project_id else [])
                    + (["microsoft"] if server_config.microsoft_auth_client_id else [])
                ),
                **({"microsoft_auth_client_id": server_config.microsoft_auth_client_id,
                    "microsoft_auth_tenant_id": server_config.microsoft_auth_tenant_id}
                   if server_config.microsoft_auth_client_id else {}),
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
    from constat.server.routes.agents import router as agents_router
    from constat.server.routes.skills import router as skills_router
    from constat.server.routes.tier_management import router as tier_management_router
    from constat.server.routes.feedback import router as feedback_router
    from constat.server.routes.testing import router as testing_router
    from constat.server.routes.fine_tune import router as fine_tune_router
    from constat.server.routes.mcp_catalog import router as mcp_catalog_router

    from constat.server.routes.oauth_email import router as oauth_email_router
    from constat.server.routes.oauth import router as oauth_router
    from constat.server.routes.accounts import router as accounts_router
    from constat.server.routes.vault import router as vault_router
    from constat.server.graphql import graphql_router
    from constat.server.routes.graphql_sse import router as graphql_sse_router

    # IMPORTANT: Register routers with specific paths BEFORE routers with /{session_id} wildcards
    # Otherwise the wildcard routes will match paths like /agents, /skills, etc.
    fastapi_app.include_router(
        oauth_email_router,
        prefix="/api/oauth/email",
        tags=["oauth-email"],
    )
    fastapi_app.include_router(
        oauth_router,
        prefix="/api/oauth",
        tags=["oauth"],
    )
    fastapi_app.include_router(
        accounts_router,
        prefix="/api/accounts",
        tags=["accounts"],
    )
    fastapi_app.include_router(
        vault_router,
        prefix="/api/vault",
        tags=["vault"],
    )
    fastapi_app.include_router(
        graphql_router,
        prefix="/api/graphql",
        tags=["graphql"],
    )
    fastapi_app.include_router(
        graphql_sse_router,
        prefix="/api/graphql",
        tags=["graphql-sse"],
    )
    fastapi_app.include_router(
        agents_router,
        prefix="/api/sessions",
        tags=["agents"],
    )
    fastapi_app.include_router(
        skills_router,
        prefix="/api",
        tags=["skills"],
    )
    fastapi_app.include_router(
        tier_management_router,
        prefix="/api/sessions",
        tags=["tier-management"],
    )
    fastapi_app.include_router(
        feedback_router,
        prefix="/api/sessions",
        tags=["feedback"],
    )
    fastapi_app.include_router(
        testing_router,
        prefix="/api/sessions",
        tags=["testing"],
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
    fastapi_app.include_router(
        fine_tune_router,
        prefix="/api",
        tags=["fine-tune"],
    )
    fastapi_app.include_router(
        mcp_catalog_router,
        prefix="/api/mcp",
        tags=["mcp-catalog"],
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

        config_path = os.environ.get("CONSTAT_CONFIG")
        if not config_path:
            # Recover from persisted file (env var lost on uvicorn reload)
            marker = Path(".constat/server_config_path")
            if marker.exists():
                config_path = marker.read_text().strip()
                logger.info(f"Recovered config path from marker: {config_path}")
            else:
                config_path = "config.yaml"
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
