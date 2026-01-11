"""GraphQL API for Constat.

Provides a GraphQL endpoint for:
- Session management (create, list, delete)
- Problem solving (exploratory and auditable modes)
- Schema exploration (databases, tables, search)
- Fact resolution with provenance
- Real-time subscriptions for execution progress

Usage:
    from constat.api.graphql import create_app, schema

    # Create app with dependencies
    app = create_app(
        schema_manager=schema_manager,
        fact_resolver=fact_resolver,
        config=config,
    )

    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

Or run directly:
    python -m constat.api.graphql.app
"""

from .schema import schema, GraphQLContext, Query, Mutation, Subscription
from .app import create_app, run_server
from .types import (
    # Enums
    SessionStatus,
    StepStatus,
    StepType,
    ExecutionMode,
    FactSourceType,
    ArtifactType,
    # Types
    Session,
    SessionSummary,
    Plan,
    Step,
    Artifact,
    Fact,
    DerivationTrace,
    FactResolutionResult,
    Database,
    TableSchema,
    TableColumn,
    TableMatch,
    ForeignKey,
    SolveResult,
    # Events
    SessionEvent,
    PlanEvent,
    StepStartEvent,
    StepCompleteEvent,
    FactResolvingEvent,
    FactResolvedEvent,
    SessionCompleteEvent,
)

__all__ = [
    # App
    "create_app",
    "run_server",
    "schema",
    "GraphQLContext",
    "Query",
    "Mutation",
    "Subscription",
    # Enums
    "SessionStatus",
    "StepStatus",
    "StepType",
    "ExecutionMode",
    "FactSourceType",
    "ArtifactType",
    # Types
    "Session",
    "SessionSummary",
    "Plan",
    "Step",
    "Artifact",
    "Fact",
    "DerivationTrace",
    "FactResolutionResult",
    "Database",
    "TableSchema",
    "TableColumn",
    "TableMatch",
    "ForeignKey",
    "SolveResult",
    # Events
    "SessionEvent",
    "PlanEvent",
    "StepStartEvent",
    "StepCompleteEvent",
    "FactResolvingEvent",
    "FactResolvedEvent",
    "SessionCompleteEvent",
]
