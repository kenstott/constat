"""GraphQL schema definition with Query, Mutation, and Subscription."""

from datetime import datetime
from typing import AsyncGenerator, Optional
import strawberry

from .types import (
    Session,
    SessionSummary,
    SessionStatus,
    SolveResult,
    Plan,
    Step,
    StepStatus,
    StepType,
    Artifact,
    ArtifactType,
    Fact,
    FactSourceType,
    FactResolutionResult,
    DerivationTrace,
    Database,
    TableSchema,
    TableColumn,
    TableMatch,
    ForeignKey,
    ExecutionMode,
    SessionEvent,
    PlanEvent,
    StepStartEvent,
    StepCompleteEvent,
    FactResolvingEvent,
    FactResolvedEvent,
    SessionCompleteEvent,
)


# ============== Context ==============

class GraphQLContext:
    """Context available to all resolvers."""

    def __init__(
        self,
        session_manager=None,
        schema_manager=None,
        fact_resolver=None,
        config=None,
    ):
        self.session_manager = session_manager
        self.schema_manager = schema_manager
        self.fact_resolver = fact_resolver
        self.config = config
        self.sessions: dict[str, dict] = {}  # In-memory session store


def get_context() -> GraphQLContext:
    """Get the current context. Override in app setup."""
    return GraphQLContext()


# ============== Query ==============

@strawberry.type
class Query:
    """GraphQL Query type."""

    # Session queries
    @strawberry.field
    def session(self, id: str, info: strawberry.Info) -> Optional[Session]:
        """Get a session by ID."""
        ctx: GraphQLContext = info.context
        session_data = ctx.sessions.get(id)
        if not session_data:
            return None
        return _convert_session(session_data)

    @strawberry.field
    def sessions(
        self,
        info: strawberry.Info,
        limit: int = 20,
        status: Optional[SessionStatus] = None,
    ) -> list[SessionSummary]:
        """List recent sessions."""
        ctx: GraphQLContext = info.context
        sessions = list(ctx.sessions.values())

        if status:
            sessions = [s for s in sessions if s.get("status") == status.value]

        sessions = sorted(sessions, key=lambda s: s.get("created_at", ""), reverse=True)
        return [_convert_session_summary(s) for s in sessions[:limit]]

    # Schema exploration queries
    @strawberry.field
    def databases(self, info: strawberry.Info) -> list[Database]:
        """List configured databases."""
        ctx: GraphQLContext = info.context
        if not ctx.schema_manager:
            return []

        result = []
        for db_name, tables in ctx.schema_manager._tables_by_db.items():
            result.append(Database(
                name=db_name,
                description=_get_db_description(ctx, db_name),
                table_count=len(tables),
                tables=[t.name for t in tables],
            ))
        return result

    @strawberry.field
    def table(self, name: str, info: strawberry.Info) -> Optional[TableSchema]:
        """Get schema for a specific table."""
        ctx: GraphQLContext = info.context
        if not ctx.schema_manager:
            return None

        try:
            table_meta = ctx.schema_manager.get_table(name)
            return _convert_table_schema(table_meta)
        except KeyError:
            return None

    @strawberry.field
    def search_tables(
        self,
        query: str,
        info: strawberry.Info,
        top_k: int = 5,
    ) -> list[TableMatch]:
        """Search for tables relevant to a query."""
        ctx: GraphQLContext = info.context
        if not ctx.schema_manager:
            return []

        matches = ctx.schema_manager.find_relevant_tables(query, top_k=top_k)
        return [
            TableMatch(
                table=m["table"],
                database=m["database"],
                relevance=m["relevance"],
                summary=m["summary"],
            )
            for m in matches
        ]

    # Fact queries
    @strawberry.field
    def resolve_fact(
        self,
        fact_name: str,
        info: strawberry.Info,
        params: Optional[strawberry.scalars.JSON] = None,
    ) -> Optional[FactResolutionResult]:
        """Resolve a specific fact."""
        ctx: GraphQLContext = info.context
        if not ctx.fact_resolver:
            return None

        params = params or {}
        fact = ctx.fact_resolver.resolve(fact_name, **params)

        return FactResolutionResult(
            fact=_convert_fact(fact),
            derivation_trace=fact.derivation_trace,
        )

    @strawberry.field
    def suggest_mode(self, query: str) -> ExecutionMode:
        """Suggest execution mode for a query."""
        from constat.execution.mode import suggest_mode as _suggest_mode
        selection = _suggest_mode(query)
        return ExecutionMode(selection.mode.value)


# ============== Mutation ==============

@strawberry.input
class SolveInput:
    """Input for solving a problem."""
    problem: str
    mode: Optional[ExecutionMode] = None
    config_path: Optional[str] = None


@strawberry.input
class CreateSessionInput:
    """Input for creating a session."""
    mode: Optional[ExecutionMode] = None
    config_path: Optional[str] = None


@strawberry.type
class Mutation:
    """GraphQL Mutation type."""

    @strawberry.mutation
    def create_session(
        self,
        info: strawberry.Info,
        input: Optional[CreateSessionInput] = None,
    ) -> Session:
        """Create a new session."""
        import uuid
        ctx: GraphQLContext = info.context

        session_id = str(uuid.uuid4())[:8]
        mode = input.mode if input else ExecutionMode.AUDITABLE

        session_data = {
            "id": session_id,
            "status": "pending",
            "mode": mode.value,
            "created_at": datetime.now().isoformat(),
            "completed_steps": [],
            "artifacts": [],
            "facts_resolved": [],
        }
        ctx.sessions[session_id] = session_data

        return _convert_session(session_data)

    @strawberry.mutation
    def solve(self, info: strawberry.Info, input: SolveInput) -> SolveResult:
        """Solve a problem (synchronous)."""
        import uuid
        import time
        ctx: GraphQLContext = info.context

        start_time = time.time()
        session_id = str(uuid.uuid4())[:8]

        # Determine mode
        if input.mode:
            mode = input.mode
        else:
            from constat.execution.mode import suggest_mode
            selection = suggest_mode(input.problem)
            mode = ExecutionMode(selection.mode.value)

        # Create session
        session_data = {
            "id": session_id,
            "status": "running",
            "mode": mode.value,
            "problem": input.problem,
            "created_at": datetime.now().isoformat(),
            "completed_steps": [],
            "artifacts": [],
            "facts_resolved": [],
        }
        ctx.sessions[session_id] = session_data

        try:
            # Execute based on mode
            if mode == ExecutionMode.AUDITABLE and ctx.fact_resolver:
                result = _execute_auditable(ctx, input.problem, session_data)
            elif ctx.session_manager:
                result = _execute_exploratory(ctx, input.problem, session_data)
            else:
                result = {
                    "success": False,
                    "error": "No execution backend configured",
                }

            duration_ms = int((time.time() - start_time) * 1000)
            session_data["status"] = "completed" if result.get("success") else "failed"
            session_data["completed_at"] = datetime.now().isoformat()

            return SolveResult(
                success=result.get("success", False),
                session_id=session_id,
                mode=mode,
                plan=result.get("plan"),
                output=result.get("output"),
                facts=[_convert_fact(f) for f in result.get("facts", [])],
                artifacts=[_convert_artifact(a) for a in result.get("artifacts", [])],
                error=result.get("error"),
                duration_ms=duration_ms,
            )

        except Exception as e:
            session_data["status"] = "failed"
            session_data["error"] = str(e)
            return SolveResult(
                success=False,
                session_id=session_id,
                mode=mode,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000),
            )

    @strawberry.mutation
    def delete_session(self, id: str, info: strawberry.Info) -> bool:
        """Delete a session."""
        ctx: GraphQLContext = info.context
        if id in ctx.sessions:
            del ctx.sessions[id]
            return True
        return False

    @strawberry.mutation
    def interrupt_session(self, id: str, info: strawberry.Info) -> Optional[Session]:
        """Interrupt a running session."""
        ctx: GraphQLContext = info.context
        session_data = ctx.sessions.get(id)
        if session_data and session_data.get("status") == "running":
            session_data["status"] = "interrupted"
            return _convert_session(session_data)
        return None


# ============== Subscription ==============

@strawberry.type
class Subscription:
    """GraphQL Subscription type for real-time updates."""

    @strawberry.subscription
    async def session_events(
        self,
        session_id: str,
        info: strawberry.Info,
    ) -> AsyncGenerator[SessionEvent, None]:
        """Subscribe to events for a session."""
        import asyncio
        ctx: GraphQLContext = info.context

        # Simple polling-based implementation
        # In production, use proper pub/sub (Redis, etc.)
        last_step = -1
        while True:
            session_data = ctx.sessions.get(session_id)
            if not session_data:
                break

            status = session_data.get("status")

            # Check for new completed steps
            completed = session_data.get("completed_steps", [])
            for step_num in completed:
                if step_num > last_step:
                    last_step = step_num
                    yield StepCompleteEvent(
                        session_id=session_id,
                        step_number=step_num,
                        success=True,
                        duration_ms=0,
                    )

            # Check for completion
            if status in ("completed", "failed", "interrupted"):
                yield SessionCompleteEvent(
                    session_id=session_id,
                    success=status == "completed",
                    error=session_data.get("error"),
                    duration_ms=0,
                )
                break

            await asyncio.sleep(0.5)


# ============== Helper Functions ==============

def _get_db_description(ctx: GraphQLContext, db_name: str) -> str:
    """Get description for a database."""
    if ctx.config and db_name in ctx.config.databases:
        return ctx.config.databases[db_name].description
    return ""


def _convert_session(data: dict) -> Session:
    """Convert session data dict to Session type."""
    return Session(
        id=data["id"],
        status=SessionStatus(data.get("status", "pending")),
        mode=ExecutionMode(data.get("mode", "auditable")),
        plan=data.get("plan"),
        current_step=data.get("current_step"),
        completed_steps=data.get("completed_steps", []),
        artifacts=[_convert_artifact(a) for a in data.get("artifacts", [])],
        facts_resolved=[_convert_fact(f) for f in data.get("facts_resolved", [])],
        created_at=datetime.fromisoformat(data["created_at"]),
        completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
        error=data.get("error"),
    )


def _convert_session_summary(data: dict) -> SessionSummary:
    """Convert session data dict to SessionSummary."""
    return SessionSummary(
        id=data["id"],
        status=SessionStatus(data.get("status", "pending")),
        mode=ExecutionMode(data.get("mode", "auditable")),
        problem=data.get("problem"),
        step_count=len(data.get("plan", {}).get("steps", [])) if data.get("plan") else 0,
        completed_steps=len(data.get("completed_steps", [])),
        created_at=datetime.fromisoformat(data["created_at"]),
        duration_ms=data.get("duration_ms"),
    )


def _convert_fact(fact) -> Fact:
    """Convert internal Fact to GraphQL Fact."""
    from constat.execution.fact_resolver import Fact as InternalFact, FactSource

    if isinstance(fact, dict):
        return Fact(
            name=fact.get("name", ""),
            value=fact.get("value"),
            confidence=fact.get("confidence", 1.0),
            source=FactSourceType(fact.get("source", "database")),
            query=fact.get("query"),
            rule_name=fact.get("rule_name"),
            reasoning=fact.get("reasoning"),
            resolved_at=datetime.fromisoformat(fact["resolved_at"]) if fact.get("resolved_at") else datetime.now(),
            because=fact.get("because", []),
        )
    elif isinstance(fact, InternalFact):
        return Fact(
            name=fact.name,
            value=fact.value,
            confidence=fact.confidence,
            source=FactSourceType(fact.source.value),
            query=fact.query,
            rule_name=fact.rule_name,
            reasoning=fact.reasoning,
            resolved_at=fact.resolved_at,
            because=[f.name for f in fact.because],
        )
    else:
        raise ValueError(f"Unknown fact type: {type(fact)}")


def _convert_artifact(artifact) -> Artifact:
    """Convert internal Artifact to GraphQL Artifact."""
    if isinstance(artifact, dict):
        return Artifact(
            id=str(artifact.get("id", "")),
            name=artifact.get("name", ""),
            artifact_type=ArtifactType(artifact.get("artifact_type", "output")),
            content=artifact.get("content", ""),
            step_number=artifact.get("step_number", 0),
            title=artifact.get("title"),
            metadata=artifact.get("metadata"),
            created_at=datetime.fromisoformat(artifact["created_at"]) if artifact.get("created_at") else datetime.now(),
        )
    else:
        # Assume it's our internal Artifact type
        return Artifact(
            id=str(artifact.id),
            name=artifact.name,
            artifact_type=ArtifactType(artifact.artifact_type.value),
            content=artifact.content,
            step_number=artifact.step_number,
            title=artifact.title,
            metadata=artifact.metadata,
            created_at=datetime.now(),
        )


def _convert_table_schema(table_meta) -> TableSchema:
    """Convert internal TableMetadata to GraphQL TableSchema."""
    return TableSchema(
        name=table_meta.name,
        database=table_meta.database,
        columns=[
            TableColumn(
                name=col["name"],
                data_type=col["type"],
                nullable=col.get("nullable", True),
                primary_key=col["name"] in table_meta.primary_keys,
                description=col.get("description"),
            )
            for col in table_meta.columns
        ],
        row_count=table_meta.row_count,
        primary_keys=table_meta.primary_keys,
        foreign_keys=[
            ForeignKey(
                from_column=fk["from"],
                to_table=fk["to_table"],
                to_column=fk["to_column"],
            )
            for fk in table_meta.foreign_keys
        ],
        description=table_meta.description,
    )


def _execute_auditable(ctx: GraphQLContext, problem: str, session_data: dict) -> dict:
    """Execute in auditable mode using fact resolver."""
    # This is a simplified implementation
    # Full implementation would generate a plan with assumed facts
    # and resolve them lazily

    facts = []

    # For now, just return a placeholder
    return {
        "success": True,
        "output": f"Auditable analysis of: {problem}",
        "facts": facts,
        "artifacts": [],
    }


def _execute_exploratory(ctx: GraphQLContext, problem: str, session_data: dict) -> dict:
    """Execute in exploratory mode using session manager."""
    # This would integrate with the existing Session class
    return {
        "success": True,
        "output": f"Exploratory analysis of: {problem}",
        "facts": [],
        "artifacts": [],
    }


# ============== Schema ==============

schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription,
)
