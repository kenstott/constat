"""GraphQL type definitions using Strawberry."""

from datetime import datetime
from enum import Enum
from typing import Optional
import strawberry


# ============== Enums ==============

@strawberry.enum
class SessionStatus(Enum):
    """Status of a session."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@strawberry.enum
class StepStatus(Enum):
    """Status of a plan step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@strawberry.enum
class StepType(Enum):
    """Type of execution step."""
    PYTHON = "python"
    SQL = "sql"
    FACT_RESOLUTION = "fact_resolution"


@strawberry.enum
class ExecutionMode(Enum):
    """Execution mode for traceability."""
    EXPLORATORY = "exploratory"
    AUDITABLE = "auditable"


@strawberry.enum
class FactSourceType(Enum):
    """Source of a resolved fact."""
    DATABASE = "database"
    LLM_KNOWLEDGE = "llm_knowledge"
    LLM_HEURISTIC = "llm_heuristic"
    RULE = "rule"
    SUB_PLAN = "sub_plan"
    CONFIG = "config"
    CACHE = "cache"
    UNRESOLVED = "unresolved"


@strawberry.enum
class ArtifactType(Enum):
    """Type of artifact."""
    CODE = "code"
    OUTPUT = "output"
    TABLE = "table"
    CHART = "chart"
    HTML = "html"
    MARKDOWN = "markdown"
    SVG = "svg"
    PNG = "png"
    JSON = "json"


# ============== Core Types ==============

@strawberry.type
class TableColumn:
    """A column in a database table."""
    name: str
    data_type: str
    nullable: bool
    primary_key: bool
    description: Optional[str] = None


@strawberry.type
class ForeignKey:
    """A foreign key relationship."""
    from_column: str
    to_table: str
    to_column: str


@strawberry.type
class TableSchema:
    """Schema information for a database table."""
    name: str
    database: str
    columns: list[TableColumn]
    row_count: int
    primary_keys: list[str]
    foreign_keys: list[ForeignKey]
    description: Optional[str] = None


@strawberry.type
class TableMatch:
    """A table matching a search query."""
    table: str
    database: str
    relevance: float
    summary: str


@strawberry.type
class Database:
    """A configured database."""
    name: str
    description: str
    table_count: int
    tables: list[str]


# ============== Fact Resolution Types ==============

@strawberry.type
class Fact:
    """A resolved fact with provenance."""
    name: str
    value: strawberry.scalars.JSON
    confidence: float
    source: FactSourceType
    query: Optional[str] = None
    rule_name: Optional[str] = None
    reasoning: Optional[str] = None
    resolved_at: datetime
    because: list[str]  # Names of dependent facts


@strawberry.type
class DerivationTrace:
    """Human-readable derivation chain for a fact."""
    fact_name: str
    trace: str  # Formatted derivation trace


# ============== Plan & Step Types ==============

@strawberry.type
class Step:
    """A step in an execution plan."""
    number: int
    goal: str
    step_type: StepType
    status: StepStatus
    code: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None
    artifacts: list[str] = strawberry.field(default_factory=list)


@strawberry.type
class Plan:
    """An execution plan."""
    problem: str
    mode: ExecutionMode
    steps: list[Step]
    created_at: datetime


@strawberry.type
class Artifact:
    """An artifact produced during execution."""
    id: str
    name: str
    artifact_type: ArtifactType
    content: str
    step_number: int
    title: Optional[str] = None
    metadata: Optional[strawberry.scalars.JSON] = None
    created_at: datetime


# ============== Session Types ==============

@strawberry.type
class Session:
    """A reasoning session."""
    id: str
    status: SessionStatus
    mode: ExecutionMode
    plan: Optional[Plan] = None
    current_step: Optional[int] = None
    completed_steps: list[int] = strawberry.field(default_factory=list)
    artifacts: list[Artifact] = strawberry.field(default_factory=list)
    facts_resolved: list[Fact] = strawberry.field(default_factory=list)
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


@strawberry.type
class SessionSummary:
    """Summary of a session for listing."""
    id: str
    status: SessionStatus
    mode: ExecutionMode
    problem: Optional[str] = None
    step_count: int = 0
    completed_steps: int = 0
    created_at: datetime
    duration_ms: Optional[int] = None


# ============== Execution Result Types ==============

@strawberry.type
class SolveResult:
    """Result of solving a problem."""
    success: bool
    session_id: str
    mode: ExecutionMode
    plan: Optional[Plan] = None
    output: Optional[str] = None
    facts: list[Fact] = strawberry.field(default_factory=list)
    artifacts: list[Artifact] = strawberry.field(default_factory=list)
    error: Optional[str] = None
    duration_ms: int = 0


@strawberry.type
class FactResolutionResult:
    """Result of resolving a specific fact."""
    fact: Fact
    derivation_trace: str


# ============== Streaming Event Types ==============

@strawberry.type
class PlanEvent:
    """Event when plan is created."""
    session_id: str
    plan: Plan


@strawberry.type
class StepStartEvent:
    """Event when a step starts."""
    session_id: str
    step_number: int
    goal: str


@strawberry.type
class StepCompleteEvent:
    """Event when a step completes."""
    session_id: str
    step_number: int
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    duration_ms: int = 0


@strawberry.type
class FactResolvingEvent:
    """Event when starting to resolve a fact."""
    session_id: str
    fact_name: str


@strawberry.type
class FactResolvedEvent:
    """Event when a fact is resolved."""
    session_id: str
    fact: Fact


@strawberry.type
class SessionCompleteEvent:
    """Event when session completes."""
    session_id: str
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    duration_ms: int = 0


# Union type for subscription events
SessionEvent = strawberry.union(
    "SessionEvent",
    types=[
        PlanEvent,
        StepStartEvent,
        StepCompleteEvent,
        FactResolvingEvent,
        FactResolvedEvent,
        SessionCompleteEvent,
    ],
)
