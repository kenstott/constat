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


def _compute_doc_config_hash(documents: dict) -> str:
    """Compute a hash for a document configuration dict."""
    import hashlib
    import json

    doc_data = {}
    if documents:
        for doc_name, doc_config in sorted(documents.items()):
            doc_data[doc_name] = {
                "path": doc_config.path or "",
                "description": doc_config.description or "",
                "format": doc_config.format or "",
            }
    config_json = json.dumps(doc_data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(config_json.encode()).hexdigest()[:16]


def _warmup_vector_store(config: Config) -> None:
    """Pre-index all documents from config and projects at server startup.

    Uses hash-based invalidation to only re-index when config changes.
    """
    from pathlib import Path
    from constat.discovery.doc_tools import DocumentDiscoveryTools

    # Create doc_tools with skip_auto_index since we handle indexing here
    doc_tools = DocumentDiscoveryTools(config, skip_auto_index=True)
    vector_store = doc_tools._vector_store

    # Index base (config) documents with hash-based invalidation
    if config.documents:
        base_hash = _compute_doc_config_hash(config.documents)
        logger.debug(f"  Base documents: current_hash={base_hash}")

        if hasattr(vector_store, 'get_project_config_hash'):
            # Use special project_id "__base__" for config documents
            cached_hash = vector_store.get_project_config_hash("__base__")
            logger.debug(f"  Base documents: cached_hash={cached_hash}")

            if cached_hash == base_hash:
                logger.info(f"  Base documents: {len(config.documents)} unchanged, skipping")
            else:
                if cached_hash:
                    logger.info(f"  Base documents: config changed, re-indexing")
                    vector_store.clear_project_embeddings("__base__")

                # Index each config document
                config_dir = Path(config.config_dir) if config.config_dir else Path.cwd()
                for doc_name, doc_config in config.documents.items():
                    try:
                        if doc_config.path:
                            doc_path = Path(doc_config.path)
                            if not doc_path.is_absolute():
                                doc_path = (config_dir / doc_config.path).resolve()

                            if doc_path.exists():
                                success, msg = doc_tools.add_document_from_file(
                                    str(doc_path),
                                    name=doc_name,
                                    description=doc_config.description or "",
                                    project_id="__base__",
                                    config_hash=base_hash,
                                    skip_entity_extraction=True,  # NER done at session creation
                                )
                                if success:
                                    logger.info(f"  Base: vectorized {doc_name}")
                                else:
                                    logger.warning(f"  Base: failed to vectorize {doc_name}: {msg}")
                    except Exception as e:
                        logger.warning(f"  Base: error indexing {doc_name}: {e}")

                logger.info(f"  Base documents: {len(config.documents)} indexed")
    else:
        logger.info(f"  Base documents: 0 configured")

    # Get vector store for hash checks
    vector_store = doc_tools._vector_store

    # Index project documents with hash-based invalidation
    for filename, project in config.projects.items():
        if not project.documents:
            continue

        # Compute hash for this project's document config
        current_hash = _compute_doc_config_hash(project.documents)
        logger.debug(f"  Project {filename}: current_hash={current_hash}")

        # Check if hash matches existing
        if hasattr(vector_store, 'get_project_config_hash'):
            cached_hash = vector_store.get_project_config_hash(filename)
            logger.debug(f"  Project {filename}: cached_hash={cached_hash}")
            if cached_hash == current_hash:
                logger.info(f"  Project {filename}: documents unchanged, skipping")
                continue

            # Hash changed - clear old data
            if cached_hash:
                logger.info(f"  Project {filename}: config changed, re-indexing")
                vector_store.clear_project_embeddings(filename)

        # Index all documents for this project
        for doc_name, doc_config in project.documents.items():
            try:
                if doc_config.path:
                    doc_path = Path(doc_config.path)

                    # Resolve relative paths from config directory (where data/, docs/ live)
                    if not doc_path.is_absolute():
                        config_dir = Path(config.config_dir) if config.config_dir else Path.cwd()
                        doc_path = (config_dir / doc_config.path).resolve()

                    if doc_path.exists():
                        success, msg = doc_tools.add_document_from_file(
                            str(doc_path),
                            name=doc_name,
                            description=doc_config.description or "",
                            project_id=filename,
                            config_hash=current_hash,
                            skip_entity_extraction=True,  # NER done at session creation
                        )
                        if success:
                            logger.info(f"  Project {filename}: vectorized {doc_name}")
                        else:
                            logger.warning(f"  Project {filename}: failed to vectorize {doc_name}: {msg}")
                    else:
                        logger.warning(f"  Project {filename}: file not found: {doc_path}")
            except Exception as e:
                logger.warning(f"  Project {filename}: error indexing {doc_name}: {e}")

    # Force checkpoint to ensure all writes are visible to other connections
    # DuckDB WAL mode can cause stale reads without this
    try:
        vector_store._conn.execute("CHECKPOINT")
        logger.debug("  Vector store checkpoint completed")
    except Exception as e:
        logger.debug(f"  Vector store checkpoint failed (may be expected): {e}")


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
    async def lifespan(app: FastAPI):
        """Manage application lifecycle."""
        try:
            # Startup: Pre-load embedding model (blocking - we need it for vectorization)
            from constat.embedding_loader import EmbeddingModelLoader
            logger.info("Loading embedding model...")
            EmbeddingModelLoader.get_instance().start_loading()
            EmbeddingModelLoader.get_instance().get_model()  # Wait for completion
            logger.info("Embedding model loaded")

            # Startup: Pre-index all documents from config and projects
            # This warms up the vector store so first session doesn't pay the cost
            logger.info("Pre-indexing documents from config and projects...")
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

    app = FastAPI(
        title="Constat API",
        description="Multi-step AI reasoning engine API for data analysis",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store config and session manager on app state
    app.state.config = config
    app.state.server_config = server_config
    app.state.session_manager = session_manager

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Handle request validation errors with detailed logging."""
        logger.error(f"[VALIDATION ERROR] Path: {request.url.path}, Method: {request.method}")
        logger.error(f"[VALIDATION ERROR] Details: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )

    @app.exception_handler(KeyError)
    async def key_error_handler(request, exc: KeyError) -> JSONResponse:
        """Handle KeyError (session not found, etc.)."""
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "message": str(exc)},
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request, exc: RuntimeError) -> JSONResponse:
        """Handle RuntimeError (limit exceeded, etc.)."""
        return JSONResponse(
            status_code=400,
            content={"error": "runtime_error", "message": str(exc)},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc: ValueError) -> JSONResponse:
        """Handle ValueError (invalid input, etc.)."""
        return JSONResponse(
            status_code=400,
            content={"error": "validation_error", "message": str(exc)},
        )

    # Health check endpoint
    @app.get("/health")
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
    @app.get("/debug/auth")
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
    app.include_router(
        roles_router,
        prefix="/api/sessions",
        tags=["roles"],
    )
    app.include_router(
        skills_router,
        prefix="/api",
        tags=["skills"],
    )
    app.include_router(
        sessions_router,
        prefix="/api/sessions",
        tags=["sessions"],
    )
    app.include_router(
        queries_router,
        prefix="/api/sessions",
        tags=["queries"],
    )
    app.include_router(
        data_router,
        prefix="/api/sessions",
        tags=["data"],
    )
    app.include_router(
        files_router,
        prefix="/api/sessions",
        tags=["files"],
    )
    app.include_router(
        databases_router,
        prefix="/api/sessions",
        tags=["databases"],
    )
    app.include_router(
        schema_router,
        prefix="/api/schema",
        tags=["schema"],
    )
    app.include_router(
        learnings_router,
        prefix="/api",
        tags=["learnings"],
    )
    app.include_router(
        users_router,
        prefix="/api/users",
        tags=["users"],
    )

    return app


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
        import re
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
