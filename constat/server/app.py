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
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
_constat_logger = logging.getLogger('constat')
if not _constat_logger.handlers:
    _constat_logger.addHandler(_console_handler)
    _constat_logger.setLevel(logging.INFO)

# Suppress multiprocessing resource_tracker warnings at shutdown
warnings.filterwarnings("ignore", message="resource_tracker:")
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from constat.core.config import Config
from constat.server.config import ServerConfig
from constat.server.session_manager import SessionManager
from constat.server.routes.queries import shutdown_executor_async

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
        await shutdown_executor_async()  # Stop thread pool to allow clean exit
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
