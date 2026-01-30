# Copyright (c) 2025 Kenneth Stott
#
# This source code is licensed under the Business Source License 1.1
# found in the LICENSE file in the root directory of this source tree.
#
# NOTICE: Use of this software for training artificial intelligence or
# machine learning models is strictly prohibited without explicit written
# permission from the copyright holder.

"""Pydantic models for API request/response schemas."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================


class SessionStatus(str, Enum):
    """Session lifecycle status."""

    IDLE = "idle"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class EventType(str, Enum):
    """Types of events emitted via WebSocket."""

    # Session events
    SESSION_CREATED = "session_created"
    SESSION_CLOSED = "session_closed"

    # Planning events
    PLANNING_START = "planning_start"
    PLAN_READY = "plan_ready"
    PLAN_APPROVED = "plan_approved"
    PLAN_REJECTED = "plan_rejected"

    # Execution events
    STEP_START = "step_start"
    STEP_GENERATING = "step_generating"
    STEP_EXECUTING = "step_executing"
    STEP_COMPLETE = "step_complete"
    STEP_ERROR = "step_error"
    STEP_FAILED = "step_failed"

    # Fact events
    FACTS_EXTRACTED = "facts_extracted"
    FACT_RESOLVED = "fact_resolved"

    # Progress events
    PROGRESS = "progress"

    # Clarification events
    CLARIFICATION_NEEDED = "clarification_needed"
    CLARIFICATION_RECEIVED = "clarification_received"

    # Query events
    QUERY_COMPLETE = "query_complete"
    QUERY_ERROR = "query_error"
    QUERY_CANCELLED = "query_cancelled"

    # Table/artifact events
    TABLE_CREATED = "table_created"
    ARTIFACT_CREATED = "artifact_created"

    # Synthesis events
    SYNTHESIZING = "synthesizing"
    GENERATING_INSIGHTS = "generating_insights"


# ============================================================================
# Session Models
# ============================================================================


class SessionCreate(BaseModel):
    """Request to create a new session."""

    user_id: str = Field(
        default="default",
        description="User ID for session ownership",
    )
    config_overrides: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional config overrides for this session",
    )


class SessionResponse(BaseModel):
    """Response containing session details."""

    session_id: str = Field(description="Unique session identifier")
    user_id: str = Field(description="User ID who owns the session")
    status: SessionStatus = Field(description="Current session status")
    created_at: datetime = Field(description="When the session was created")
    last_activity: datetime = Field(description="Last activity timestamp")
    current_query: Optional[str] = Field(
        default=None,
        description="Current query being processed",
    )
    active_projects: list[str] = Field(
        default_factory=list,
        description="Active project filenames (e.g., ['sales-analytics.yaml'])",
    )
    tables_count: int = Field(
        default=0,
        description="Number of tables in datastore",
    )
    artifacts_count: int = Field(
        default=0,
        description="Number of artifacts produced",
    )


class SessionListResponse(BaseModel):
    """Response containing list of sessions."""

    sessions: list[SessionResponse] = Field(description="List of sessions")
    total: int = Field(description="Total number of sessions")


# ============================================================================
# Query Models
# ============================================================================


class QueryRequest(BaseModel):
    """Request to submit a query for execution."""

    problem: str = Field(
        description="Natural language problem to solve",
        min_length=1,
    )
    is_followup: bool = Field(
        default=False,
        description="Whether this is a follow-up to the previous query",
    )


class QueryResponse(BaseModel):
    """Response after submitting a query."""

    execution_id: str = Field(description="Unique execution identifier")
    status: str = Field(description="Current status of the execution")
    message: str = Field(description="Status message")


# ============================================================================
# Plan Models
# ============================================================================


class StepResponse(BaseModel):
    """A single step in a plan."""

    number: int = Field(description="Step number (1-indexed)")
    goal: str = Field(description="Natural language goal of the step")
    status: str = Field(description="Step execution status")
    expected_inputs: list[str] = Field(
        default_factory=list,
        description="Expected input data",
    )
    expected_outputs: list[str] = Field(
        default_factory=list,
        description="Expected output data",
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description="Step numbers this step depends on",
    )
    code: Optional[str] = Field(
        default=None,
        description="Generated code (after execution)",
    )
    result: Optional[dict[str, Any]] = Field(
        default=None,
        description="Execution result (after completion)",
    )


class PlanResponse(BaseModel):
    """Response containing the execution plan."""

    problem: str = Field(description="Original problem statement")
    steps: list[StepResponse] = Field(description="Plan steps")
    current_step: int = Field(description="Current step being executed")
    completed_steps: list[int] = Field(
        default_factory=list,
        description="Completed step numbers",
    )
    failed_steps: list[int] = Field(
        default_factory=list,
        description="Failed step numbers",
    )
    is_complete: bool = Field(description="Whether plan is fully executed")


class ApprovalRequest(BaseModel):
    """Request to approve or reject a plan."""

    approved: bool = Field(description="Whether to approve the plan")
    feedback: Optional[str] = Field(
        default=None,
        description="Optional feedback (required if rejected)",
    )
    deleted_steps: Optional[list[int]] = Field(
        default=None,
        description="Step numbers to skip/delete from execution",
    )


class ApprovalResponse(BaseModel):
    """Response after approval decision."""

    status: str = Field(description="Result status")
    message: str = Field(description="Status message")


# ============================================================================
# Data Models
# ============================================================================


class TableInfo(BaseModel):
    """Information about a table in the datastore."""

    name: str = Field(description="Table name")
    row_count: int = Field(description="Number of rows")
    step_number: int = Field(description="Step that created/modified it")
    columns: list[str] = Field(
        default_factory=list,
        description="Column names",
    )
    is_starred: bool = Field(default=False, description="Whether the table is starred/promoted")


class TableListResponse(BaseModel):
    """Response containing list of tables."""

    tables: list[TableInfo] = Field(description="List of tables")


class TableDataResponse(BaseModel):
    """Response containing table data."""

    name: str = Field(description="Table name")
    columns: list[str] = Field(description="Column names")
    data: list[dict[str, Any]] = Field(description="Row data")
    total_rows: int = Field(description="Total rows in table")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Rows per page")
    has_more: bool = Field(description="Whether more pages exist")


class ArtifactInfo(BaseModel):
    """Information about an artifact."""

    id: int = Field(description="Artifact ID")
    name: str = Field(description="Artifact name")
    artifact_type: str = Field(description="Type of artifact")
    step_number: int = Field(description="Step that created it")
    title: Optional[str] = Field(default=None, description="Display title")
    description: Optional[str] = Field(default=None, description="Description")
    mime_type: str = Field(description="MIME type of content")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    is_key_result: bool = Field(default=False, description="Whether this is a key result")
    is_starred: bool = Field(default=False, description="Whether user has explicitly starred this")


class ArtifactListResponse(BaseModel):
    """Response containing list of artifacts."""

    artifacts: list[ArtifactInfo] = Field(description="List of artifacts")


class ArtifactContentResponse(BaseModel):
    """Response containing artifact content."""

    id: int = Field(description="Artifact ID")
    name: str = Field(description="Artifact name")
    artifact_type: str = Field(description="Type of artifact")
    content: str = Field(description="Artifact content (may be base64 for binary)")
    mime_type: str = Field(description="MIME type")
    is_binary: bool = Field(description="Whether content is base64-encoded binary")


class FactInfo(BaseModel):
    """Information about a resolved fact."""

    name: str = Field(description="Fact name")
    value: Any = Field(description="Resolved value")
    source: str = Field(description="Source of the fact")
    reasoning: Optional[str] = Field(default=None, description="Resolution reasoning")
    confidence: Optional[float] = Field(default=None, description="Confidence score")
    is_persisted: bool = Field(default=False, description="Whether the fact is persisted globally")
    role_id: Optional[str] = Field(default=None, description="Role that created this fact (provenance)")


class FactListResponse(BaseModel):
    """Response containing resolved facts."""

    facts: list[FactInfo] = Field(description="List of resolved facts")


# ============================================================================
# Schema Models
# ============================================================================


class DatabaseInfo(BaseModel):
    """Information about a configured database."""

    name: str = Field(description="Database name")
    description: str = Field(default="", description="Database description")
    table_count: int = Field(description="Number of tables")
    type: str = Field(default="sql", description="Database type")


class SchemaOverviewResponse(BaseModel):
    """Response containing schema overview."""

    databases: list[DatabaseInfo] = Field(description="Configured databases")
    apis: list[str] = Field(
        default_factory=list,
        description="Configured API names",
    )
    documents: list[str] = Field(
        default_factory=list,
        description="Configured document names",
    )


class TableSchemaResponse(BaseModel):
    """Response containing detailed table schema."""

    database: str = Field(description="Database name")
    table_name: str = Field(description="Table name")
    columns: list[dict[str, Any]] = Field(description="Column definitions")
    row_count: int = Field(description="Approximate row count")
    relationships: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Foreign key relationships",
    )


# ============================================================================
# WebSocket Event Models
# ============================================================================


class StepEventWS(BaseModel):
    """WebSocket event format for step events."""

    event_type: EventType = Field(description="Type of event")
    session_id: str = Field(description="Session this event belongs to")
    step_number: int = Field(default=0, description="Related step number")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data",
    )


class WebSocketMessage(BaseModel):
    """Generic WebSocket message envelope."""

    type: str = Field(description="Message type (event, command, error)")
    payload: dict[str, Any] = Field(description="Message payload")


class WebSocketCommand(BaseModel):
    """Command sent from client via WebSocket."""

    action: str = Field(description="Command action (approve, reject, cancel)")
    data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Command-specific data",
    )


# ============================================================================
# File Upload Models
# ============================================================================


class FileUploadRequest(BaseModel):
    """Request to upload a file via data URI."""

    filename: str = Field(description="Original filename")
    content_type: str = Field(description="MIME type of the file")
    data: str = Field(description="Base64-encoded file content or data URI")


class UploadedFileInfo(BaseModel):
    """Information about an uploaded file."""

    id: str = Field(description="Unique file identifier")
    filename: str = Field(description="Original filename")
    file_uri: str = Field(description="file:// URI for use in queries")
    size_bytes: int = Field(description="File size in bytes")
    content_type: str = Field(description="MIME type of the file")
    uploaded_at: datetime = Field(description="Upload timestamp")


class UploadedFileListResponse(BaseModel):
    """Response containing list of uploaded files."""

    files: list[UploadedFileInfo] = Field(description="List of uploaded files")


# ============================================================================
# File Reference Models
# ============================================================================


class FileRefRequest(BaseModel):
    """Request to add a file reference."""

    name: str = Field(description="Reference name for the file")
    uri: str = Field(description="URL or file:// path")
    auth: Optional[str] = Field(
        default=None,
        description="Optional auth header (e.g., 'Bearer token123')",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the file",
    )


class FileRefInfo(BaseModel):
    """Information about a file reference."""

    name: str = Field(description="Reference name")
    uri: str = Field(description="URL or file path")
    has_auth: bool = Field(description="Whether auth was provided")
    description: Optional[str] = Field(default=None, description="File description")
    added_at: datetime = Field(description="When the reference was added")
    session_id: Optional[str] = Field(default=None, description="Session ID for persistence")


class FileRefListResponse(BaseModel):
    """Response containing list of file references."""

    file_refs: list[FileRefInfo] = Field(description="List of file references")


# ============================================================================
# Database Connection Models
# ============================================================================


class DatabaseAddRequest(BaseModel):
    """Request to add a database connection."""

    name: str = Field(description="Database connection name")
    uri: Optional[str] = Field(
        default=None,
        description="Connection URI (SQLAlchemy format)",
    )
    file_id: Optional[str] = Field(
        default=None,
        description="ID of uploaded file for file-based databases",
    )
    type: str = Field(
        default="sqlalchemy",
        description="Database type (sqlalchemy, duckdb, sqlite, mongodb)",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional description",
    )
    options: Optional[dict[str, Any]] = Field(
        default=None,
        description="Connection options (schema, readonly, etc.)",
    )


class SessionDatabaseInfo(BaseModel):
    """Information about a session database."""

    name: str = Field(description="Database name")
    type: str = Field(description="Database type")
    dialect: Optional[str] = Field(default=None, description="SQL dialect")
    description: Optional[str] = Field(default=None, description="Description")
    connected: bool = Field(description="Whether connected")
    table_count: Optional[int] = Field(default=None, description="Number of tables")
    added_at: datetime = Field(description="When the database was added")
    is_dynamic: bool = Field(description="Whether dynamically added (vs config)")
    file_id: Optional[str] = Field(default=None, description="Uploaded file ID if any")
    source: str = Field(default="config", description="Source: 'config', project filename, or 'session'")


class SessionDatabaseListResponse(BaseModel):
    """Response containing list of session databases."""

    databases: list[SessionDatabaseInfo] = Field(description="List of databases")


class SessionApiInfo(BaseModel):
    """Information about a session API source."""

    name: str = Field(description="API name")
    type: Optional[str] = Field(default=None, description="API type (rest, graphql, openapi)")
    description: Optional[str] = Field(default=None, description="Description")
    base_url: Optional[str] = Field(default=None, description="Base URL")
    connected: bool = Field(description="Whether API is reachable")
    from_config: bool = Field(description="Whether from config (vs session-added)")
    source: str = Field(default="config", description="Source: 'config', project filename, or 'session'")


class SessionDocumentInfo(BaseModel):
    """Information about a session document source."""

    name: str = Field(description="Document name")
    type: Optional[str] = Field(default=None, description="Document type")
    description: Optional[str] = Field(default=None, description="Description")
    path: Optional[str] = Field(default=None, description="File path")
    indexed: bool = Field(description="Whether document is indexed")
    from_config: bool = Field(description="Whether from config (vs session-added)")
    source: str = Field(default="config", description="Source: 'config', project filename, or 'session'")


class SessionDataSourcesResponse(BaseModel):
    """Response containing all data sources for a session."""

    databases: list[SessionDatabaseInfo] = Field(description="Database sources")
    apis: list[SessionApiInfo] = Field(description="API sources")
    documents: list[SessionDocumentInfo] = Field(description="Document sources")


class DatabaseTestResponse(BaseModel):
    """Response from database connection test."""

    name: str = Field(description="Database name")
    connected: bool = Field(description="Connection successful")
    table_count: int = Field(default=0, description="Number of tables found")
    error: Optional[str] = Field(default=None, description="Error message if failed")


# ============================================================================
# Learnings Models
# ============================================================================


class LearningInfo(BaseModel):
    """Information about a learning."""

    id: str = Field(description="Learning ID")
    content: str = Field(description="The correction/learning text")
    category: str = Field(description="Learning category")
    source: str = Field(description="How the learning was captured")
    context: Optional[dict[str, Any]] = Field(
        default=None,
        description="Original context",
    )
    applied_count: int = Field(default=0, description="Times applied")
    created_at: datetime = Field(description="When captured")


class LearningCreateRequest(BaseModel):
    """Request to add a new learning."""

    content: str = Field(description="The learning/correction text")
    category: str = Field(
        default="user_correction",
        description="Learning category",
    )


class RuleInfo(BaseModel):
    """Information about a compacted rule."""

    id: str = Field(description="Rule ID")
    summary: str = Field(description="The rule summary")
    category: str = Field(description="Rule category")
    confidence: float = Field(description="Confidence score 0-1")
    source_count: int = Field(default=0, description="Number of source learnings")
    tags: list[str] = Field(default_factory=list, description="Tags for searching")


class RuleCreateRequest(BaseModel):
    """Request to create a new rule."""

    summary: str = Field(description="The rule text/summary")
    category: str = Field(
        default="user_correction",
        description="Rule category (user_correction, api_error, codegen_error, nl_correction)",
    )
    confidence: float = Field(default=0.9, description="Confidence score 0-1")
    tags: list[str] = Field(default_factory=list, description="Tags for searching")


class RuleUpdateRequest(BaseModel):
    """Request to update an existing rule."""

    summary: Optional[str] = Field(default=None, description="New rule text/summary")
    confidence: Optional[float] = Field(default=None, description="New confidence score")
    tags: Optional[list[str]] = Field(default=None, description="New tags")


class LearningListResponse(BaseModel):
    """Response containing list of learnings and rules."""

    learnings: list[LearningInfo] = Field(description="List of raw learnings")
    rules: list[RuleInfo] = Field(default_factory=list, description="List of compacted rules")


# ============================================================================
# Entity Models
# ============================================================================


class EntityInfo(BaseModel):
    """Information about an extracted entity."""

    id: str = Field(description="Entity ID")
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    created_at: Optional[datetime] = Field(default=None, description="When extracted")
    mention_count: int = Field(default=0, description="How often referenced")


class EntityListResponse(BaseModel):
    """Response containing list of entities."""

    entities: list[EntityInfo] = Field(description="List of entities")


# ============================================================================
# Config Models
# ============================================================================


class ConfigResponse(BaseModel):
    """Sanitized configuration response."""

    databases: list[str] = Field(description="Configured database names")
    apis: list[str] = Field(description="Configured API names")
    documents: list[str] = Field(description="Configured document names")
    llm_provider: str = Field(description="LLM provider")
    llm_model: str = Field(description="LLM model")
    execution_timeout: int = Field(description="Execution timeout seconds")


# ============================================================================
# Project Models
# ============================================================================


class ProjectInfo(BaseModel):
    """Summary info for a project."""

    filename: str = Field(description="Project YAML filename")
    name: str = Field(description="Project display name")
    description: str = Field(default="", description="Project description")


class ProjectListResponse(BaseModel):
    """List of available projects."""

    projects: list[ProjectInfo] = Field(description="Available projects")


class ProjectDetailResponse(BaseModel):
    """Full project details."""

    filename: str = Field(description="Project YAML filename")
    name: str = Field(description="Project display name")
    description: str = Field(default="", description="Project description")
    databases: list[str] = Field(description="Database names in project")
    apis: list[str] = Field(description="API names in project")
    documents: list[str] = Field(description="Document names in project")


# ============================================================================
# Error Models
# ============================================================================


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(description="Error type/code")
    message: str = Field(description="Human-readable error message")
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional error details",
    )


class ValidationErrorResponse(BaseModel):
    """Validation error response format."""

    error: str = Field(default="validation_error")
    message: str = Field(description="Validation error message")
    fields: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Field-specific errors",
    )
