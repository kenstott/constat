# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Core data structures for multistep planning and execution."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class StepType(Enum):
    """Type of step to execute."""
    PYTHON = "python"
    # Phase 2: PROLOG = "prolog"


class TaskType(Enum):
    """Type of LLM task for model routing.

    Each task type maps to an ordered list of models to try.
    The router will try each model in order until success.
    """
    # Planning and orchestration
    PLANNING = "planning"              # Multistep plan generation
    REPLANNING = "replanning"          # Plan revision with feedback

    # Code generation
    SQL_GENERATION = "sql_generation"  # Generate SQL queries
    PYTHON_ANALYSIS = "python_analysis"  # Generate Python code

    # Classification and routing
    INTENT_CLASSIFICATION = "intent_classification"  # User intent
    MODE_SELECTION = "mode_selection"  # Exploratory vs auditable

    # Fact resolution (auditable mode)
    FACT_RESOLUTION = "fact_resolution"
    DERIVATION_LOGIC = "derivation_logic"

    # Schema exploration
    SCHEMA_DISCOVERY = "schema_discovery"

    # Text processing
    SUMMARIZATION = "summarization"
    ERROR_ANALYSIS = "error_analysis"  # Analyzing execution errors for retry
    SYNTHESIS = "synthesis"  # Generate final response from resolved facts

    # Glossary generation
    GLOSSARY_GENERATION = "glossary_generation"

    # Relationship extraction
    RELATIONSHIP_EXTRACTION = "relationship_extraction"

    # Default fallback
    GENERAL = "general"


class StepStatus(Enum):
    """Execution status of a step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ValidationOnFail(Enum):
    """Action to take when a post-validation fails."""
    RETRY = "retry"        # Re-codegen the step with validation error as context
    CLARIFY = "clarify"    # Ask user via existing clarification system
    WARN = "warn"          # Log warning, continue execution


@dataclass
class PostValidation:
    """A post-execution validation assertion for a step."""
    expression: str          # Python expression to eval (e.g., "len(df) > 0")
    description: str         # Human-readable description
    on_fail: ValidationOnFail = ValidationOnFail.RETRY
    clarify_question: str = ""  # Question for user if on_fail=CLARIFY


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
    A single step in a multistep plan.

    Each step has a goal described in natural language,
    and will be translated into executable code.

    Dependencies can be declared explicitly via `depends_on` (list of step numbers),
    or inferred from expected_inputs/expected_outputs overlap.
    Steps without dependencies can execute in parallel.

    Role Provenance:
    Each step can optionally be associated with a role via `role_id`.
    When set, facts created during step execution are tagged with that role_id
    for provenance tracking. All facts remain globally accessible - role_id
    is metadata for attribution and UI grouping, not access control.
    """
    number: int
    goal: str  # Natural language description
    expected_inputs: list[str] = field(default_factory=list)
    expected_outputs: list[str] = field(default_factory=list)
    depends_on: list[int] = field(default_factory=list)  # Explicit step dependencies
    step_type: StepType = StepType.PYTHON

    # Task type for model routing (determines which models to try)
    task_type: TaskType = TaskType.PYTHON_ANALYSIS

    # Complexity hint for model selection within a task type
    # low = simple query/transform, medium = moderate logic, high = complex operations
    complexity: str = "medium"

    # Role context for this step (None = shared context)
    # When set, facts created by this step are scoped to the role
    # Detected from patterns like "as a [role]" or "acting as [role]"
    role_id: Optional[str] = None

    # Skills to apply for this step (None = use query-level skills)
    # Skills provide domain-specific instructions for code generation
    skill_ids: Optional[list[str]] = None

    # Post-execution validations (assertions checked after successful execution)
    post_validations: list["PostValidation"] = field(default_factory=list)

    # Populated during execution
    status: StepStatus = StepStatus.PENDING
    code: Optional[str] = None
    result: Optional["StepResult"] = None

    # Phase 2 extensions
    prolog_code: Optional[str] = None
    derivation_trace: Optional[str] = None


@dataclass
class FailureSuggestion:
    """A suggested alternative approach when a step fails."""
    id: str  # Short identifier (e.g., "rephrase", "skip", "manual")
    label: str  # Display label (e.g., "Rephrase search query")
    description: str  # Detailed description of what this option does
    action: Optional[str] = None  # Optional action hint for the system


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

    # Generated code (for replay)
    code: str = ""

    # Suggestions for alternative approaches when step fails
    suggestions: list[FailureSuggestion] = field(default_factory=list)

    # Warnings from post-validations with on_fail=WARN
    validation_warnings: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """
    A multistep plan for solving a problem.

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

    # Data sensitivity (set by planner based on data involved)
    # If True, email operations require explicit authorization
    contains_sensitive_data: bool = False

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

    def infer_dependencies(self) -> None:
        """
        Infer step dependencies from expected_inputs/expected_outputs.

        If a step's expected_inputs includes something from an earlier step's
        expected_outputs, add that step to depends_on.
        """
        # Build output -> step mapping
        output_providers: dict[str, int] = {}
        for step in self.steps:
            for output in step.expected_outputs:
                output_providers[output] = step.number

        # Infer dependencies
        for step in self.steps:
            for input_name in step.expected_inputs:
                if input_name in output_providers:
                    provider_step = output_providers[input_name]
                    if provider_step != step.number and provider_step not in step.depends_on:
                        step.depends_on.append(provider_step)

    def get_dependency_graph(self) -> dict[int, list[int]]:
        """
        Get the dependency graph as adjacency list.

        Returns dict mapping step number -> list of step numbers it depends on.
        """
        return {step.number: list(step.depends_on) for step in self.steps}

    def get_runnable_steps(self) -> list[Step]:
        """
        Get steps that can run now (pending with all dependencies satisfied).

        A step is runnable if:
        1. Status is PENDING
        2. All steps in depends_on are COMPLETED
        """
        runnable = []
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue

            # Check all dependencies are completed
            deps_satisfied = True
            for dep_num in step.depends_on:
                dep_step = self.get_step(dep_num)
                if dep_step and dep_step.status != StepStatus.COMPLETED:
                    deps_satisfied = False
                    break

            if deps_satisfied:
                runnable.append(step)

        return runnable

    def get_execution_order(self) -> list[list[int]]:
        """
        Get steps grouped by execution wave (parallel batches).

        Returns list of lists, where each inner list contains step numbers
        that can execute in parallel. Waves execute sequentially.

        Example: [[1, 2], [3, 4], [5]] means:
        - Steps 1 and 2 can run in parallel (wave 1)
        - Steps 3 and 4 can run in parallel after wave 1 (wave 2)
        - Step 5 runs after wave 2 (wave 3)
        """
        waves = []
        completed = set()
        remaining = {step.number for step in self.steps}

        while remaining:
            # Find steps whose dependencies are all in completed
            wave = []
            for step_num in remaining:
                step = self.get_step(step_num)
                if step and all(dep in completed for dep in step.depends_on):
                    wave.append(step_num)

            if not wave:
                # Circular dependency or bug - break to avoid infinite loop
                break

            waves.append(sorted(wave))
            completed.update(wave)
            remaining -= set(wave)

        return waves


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
    raw_response: str = ""  # Raw LLM output before parsing


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
    content: str  # The artifact content (maybe base64 for binary)
    step_number: int = 0
    attempt: int = 1

    # Metadata
    title: Optional[str] = None
    description: Optional[str] = None
    content_type: Optional[str] = None  # MIME type override
    metadata: dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: Optional[str] = None

    # Role provenance - which role created this artifact
    role_id: Optional[str] = None

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
            "role_id": self.role_id,
        }
