"""Core data structures for multi-step planning and execution."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class StepType(Enum):
    """Type of step to execute."""
    PYTHON = "python"
    # Phase 2: PROLOG = "prolog"


class StepStatus(Enum):
    """Execution status of a step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ArtifactType(Enum):
    """Type of artifact produced by a step."""
    # Code and execution artifacts
    CODE = "code"
    OUTPUT = "output"
    ERROR = "error"

    # Data artifacts
    TABLE = "table"
    JSON = "json"

    # Rich content artifacts
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"

    # Chart/visualization artifacts
    CHART = "chart"           # Vega-Lite or similar spec
    PLOTLY = "plotly"         # Plotly JSON

    # Image artifacts
    SVG = "svg"
    PNG = "png"
    JPEG = "jpeg"

    # Diagram artifacts
    MERMAID = "mermaid"       # Mermaid diagram code
    GRAPHVIZ = "graphviz"     # DOT language
    DIAGRAM = "diagram"       # Generic diagram

    # Interactive artifacts
    REACT = "react"           # React/JSX component
    JAVASCRIPT = "javascript"


# MIME type mapping for artifacts
ARTIFACT_MIME_TYPES = {
    ArtifactType.CODE: "text/x-python",
    ArtifactType.OUTPUT: "text/plain",
    ArtifactType.ERROR: "text/plain",
    ArtifactType.TABLE: "application/json",
    ArtifactType.JSON: "application/json",
    ArtifactType.HTML: "text/html",
    ArtifactType.MARKDOWN: "text/markdown",
    ArtifactType.TEXT: "text/plain",
    ArtifactType.CHART: "application/vnd.vega.v5+json",
    ArtifactType.PLOTLY: "application/vnd.plotly.v1+json",
    ArtifactType.SVG: "image/svg+xml",
    ArtifactType.PNG: "image/png",
    ArtifactType.JPEG: "image/jpeg",
    ArtifactType.MERMAID: "text/x-mermaid",
    ArtifactType.GRAPHVIZ: "text/vnd.graphviz",
    ArtifactType.DIAGRAM: "text/plain",
    ArtifactType.REACT: "text/jsx",
    ArtifactType.JAVASCRIPT: "text/javascript",
}


@dataclass
class Step:
    """
    A single step in a multi-step plan.

    Each step has a goal described in natural language,
    and will be translated into executable code.
    """
    number: int
    goal: str  # Natural language description
    expected_inputs: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    step_type: StepType = StepType.PYTHON

    # Populated during execution
    status: StepStatus = StepStatus.PENDING
    code: Optional[str] = None
    result: Optional["StepResult"] = None

    # Phase 2 extensions
    prolog_code: Optional[str] = None
    derivation_trace: Optional[str] = None


@dataclass
class StepResult:
    """Result of executing a step."""
    success: bool
    stdout: str
    error: Optional[str] = None
    attempts: int = 1
    duration_ms: int = 0

    # Tables created/modified
    tables_created: list[str] = field(default_factory=list)
    tables_modified: list[str] = field(default_factory=list)

    # For downstream access
    variables: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """
    A multi-step plan for solving a problem.

    The plan is generated from natural language and contains
    steps to be executed sequentially.
    """
    problem: str  # Original user problem
    steps: list[Step]
    created_at: str = ""

    # Execution state
    current_step: int = 0
    completed_steps: list[int] = field(default_factory=list)
    failed_steps: list[int] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Check if all steps are complete."""
        return len(self.completed_steps) == len(self.steps)

    @property
    def next_step(self) -> Optional[Step]:
        """Get the next step to execute."""
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                return step
        return None

    def get_step(self, number: int) -> Optional[Step]:
        """Get a step by number (1-indexed)."""
        for step in self.steps:
            if step.number == number:
                return step
        return None

    def mark_step_completed(self, number: int, result: StepResult) -> None:
        """Mark a step as completed with its result."""
        step = self.get_step(number)
        if step:
            step.status = StepStatus.COMPLETED
            step.result = result
            if number not in self.completed_steps:
                self.completed_steps.append(number)

    def mark_step_failed(self, number: int, result: StepResult) -> None:
        """Mark a step as failed."""
        step = self.get_step(number)
        if step:
            step.status = StepStatus.FAILED
            step.result = result
            if number not in self.failed_steps:
                self.failed_steps.append(number)


@dataclass
class SessionState:
    """
    Complete state of a session for resumption.

    Captures everything needed to resume an interrupted session.
    """
    session_id: str
    config_path: str
    plan: Optional[Plan] = None
    scratchpad_content: str = ""
    tables: list[str] = field(default_factory=list)
    # Phase 2: knowledge_base facts and rules


@dataclass
class PlannerResponse:
    """Response from the planner."""
    plan: Plan
    reasoning: str = ""  # Why this plan was chosen


@dataclass
class Artifact:
    """
    A rich artifact produced by a step.

    Artifacts can be various types: charts, HTML, diagrams, images, etc.
    They are stored in the datastore and can be retrieved for rendering.
    """
    id: int
    name: str
    artifact_type: ArtifactType
    content: str  # The artifact content (may be base64 for binary)
    step_number: int = 0
    attempt: int = 1

    # Metadata
    title: Optional[str] = None
    description: Optional[str] = None
    content_type: Optional[str] = None  # MIME type override
    metadata: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: Optional[str] = None

    @property
    def mime_type(self) -> str:
        """Get the MIME type for this artifact."""
        if self.content_type:
            return self.content_type
        return ARTIFACT_MIME_TYPES.get(self.artifact_type, "application/octet-stream")

    @property
    def is_binary(self) -> bool:
        """Check if this artifact contains binary data (base64 encoded)."""
        return self.artifact_type in {ArtifactType.PNG, ArtifactType.JPEG}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.artifact_type.value,
            "content": self.content,
            "step_number": self.step_number,
            "attempt": self.attempt,
            "title": self.title,
            "description": self.description,
            "content_type": self.mime_type,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
