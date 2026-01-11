"""FastAPI application with GraphQL endpoint."""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter

from .schema import schema, GraphQLContext


def create_app(
    schema_manager=None,
    fact_resolver=None,
    config=None,
    cors_origins: Optional[list[str]] = None,
) -> FastAPI:
    """
    Create FastAPI application with GraphQL endpoint.

    Args:
        schema_manager: SchemaManager instance for database introspection
        fact_resolver: FactResolver instance for auditable mode
        config: Config instance
        cors_origins: List of allowed CORS origins (default: ["*"])

    Returns:
        FastAPI application

    Usage:
        from constat.api.graphql import create_app
        from constat.catalog import SchemaManager
        from constat.execution import FactResolver
        from constat.core import Config

        config = Config.from_yaml("config.yaml")
        schema_manager = SchemaManager(config)
        schema_manager.initialize()

        fact_resolver = FactResolver(
            llm=provider,
            schema_manager=schema_manager,
            config=config,
        )

        app = create_app(
            schema_manager=schema_manager,
            fact_resolver=fact_resolver,
            config=config,
        )

        # Run with: uvicorn constat.api.graphql.app:app
    """
    # Create shared context
    context = GraphQLContext(
        schema_manager=schema_manager,
        fact_resolver=fact_resolver,
        config=config,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        yield
        # Shutdown

    app = FastAPI(
        title="Constat API",
        description="Multi-step AI reasoning engine with auditable fact resolution",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # GraphQL router with context
    def get_context():
        return context

    graphql_router = GraphQLRouter(
        schema,
        context_getter=get_context,
        graphql_ide="graphiql",  # Enable GraphiQL IDE at /graphql
    )

    app.include_router(graphql_router, prefix="/graphql")

    # Health check endpoint
    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "schema_manager": schema_manager is not None,
            "fact_resolver": fact_resolver is not None,
        }

    # OpenAPI info
    @app.get("/")
    async def root():
        return {
            "name": "Constat API",
            "version": "0.1.0",
            "graphql": "/graphql",
            "graphiql": "/graphql",
            "health": "/health",
        }

    return app


# Default app instance (for uvicorn)
# Override by creating custom app with create_app()
app = create_app()


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: bool = False,
    **kwargs,
):
    """Run the GraphQL server."""
    import uvicorn
    uvicorn.run(
        "constat.api.graphql.app:app",
        host=host,
        port=port,
        reload=reload,
        **kwargs,
    )
