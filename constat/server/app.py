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
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from constat.core.config import Config
from constat.server.config import ServerConfig
from constat.server.session_manager import SessionManager

logger = logging.getLogger(__name__)


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
        # Startup: Start cleanup task
        await session_manager.start_cleanup_task()
        logger.info("Constat API server started")

        yield

        # Shutdown: Stop cleanup task and cleanup sessions
        await session_manager.stop_cleanup_task()
        for managed in session_manager.list_sessions():
            session_manager.delete_session(managed.session_id)
        logger.info("Constat API server stopped")

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
        }

    # Import and include routers
    from constat.server.routes.sessions import router as sessions_router
    from constat.server.routes.queries import router as queries_router
    from constat.server.routes.data import router as data_router
    from constat.server.routes.schema import router as schema_router
    from constat.server.routes.files import router as files_router
    from constat.server.routes.databases import router as databases_router
    from constat.server.routes.learnings import router as learnings_router

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
        config_path = os.environ.get("CONSTAT_CONFIG", "config.yaml")
        try:
            config = Config.from_yaml(config_path)
        except FileNotFoundError:
            config = Config()
        server_config = ServerConfig()
        _app = create_app(config, server_config)
    return _app


# For uvicorn reload: `uvicorn constat.server.app:app --reload`
app = get_app()
