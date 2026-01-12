"""Constat - Conversational Statistics with LLM-powered data analysis.

This package provides multi-step planning and execution for data analysis
across multiple databases using natural language queries.

Submodules:
- core: Models and configuration
- catalog: Schema and API discovery with vector search
- storage: Persistence layer (DuckDB datastore, session history)
- execution: Planning and code execution
- providers: LLM provider integrations (Anthropic, OpenAI, etc.)
- api: GraphQL and REST API layer

Main classes:
- Session: Main entry point for running analyses
- Config: Configuration loading from YAML
- Plan/Step: Multi-step execution plan
"""

# Core models and configuration
from constat.core.config import (
    APIConfig,
    Config,
    DatabaseConfig,
    DatabaseCredentials,
    ExecutionConfig,
    LLMConfig,
    ModelSpec,
    TaskRoutingConfig,
    TaskRoutingEntry,
)
from constat.core.models import (
    Artifact,
    ArtifactType,
    ARTIFACT_MIME_TYPES,
    Plan,
    PlannerResponse,
    SessionState,
    Step,
    StepResult,
    StepStatus,
    StepType,
    TaskType,
)

# Storage
from constat.storage.datastore import DataStore
from constat.storage.history import SessionHistory

# Execution
from constat.execution.planner import Planner
from constat.execution.executor import ExecutionResult, PythonExecutor
from constat.execution.engine import QueryEngine
from constat.execution.scratchpad import Scratchpad

# Catalog
from constat.catalog.schema_manager import SchemaManager, TableMetadata
from constat.catalog.api_catalog import APICatalog, OperationType

# Providers
from constat.providers import (
    BaseLLMProvider,
    TaskRouter,
    AnthropicProvider,
    OpenAIProvider,
    GeminiProvider,
    GrokProvider,
    LlamaProvider,
    OllamaProvider,
    TogetherProvider,
    GroqProvider,
)

# Main session orchestrator
from constat.session import Session

__version__ = "0.1.0"

__all__ = [
    # Core
    "APIConfig",
    "Config",
    "DatabaseConfig",
    "DatabaseCredentials",
    "ExecutionConfig",
    "LLMConfig",
    "ModelSpec",
    "TaskRoutingConfig",
    "TaskRoutingEntry",
    "TaskType",
    "Artifact",
    "ArtifactType",
    "ARTIFACT_MIME_TYPES",
    "Plan",
    "PlannerResponse",
    "SessionState",
    "Step",
    "StepResult",
    "StepStatus",
    "StepType",
    # Storage
    "DataStore",
    "SessionHistory",
    # Execution
    "Planner",
    "ExecutionResult",
    "PythonExecutor",
    "QueryEngine",
    "Scratchpad",
    # Catalog
    "SchemaManager",
    "TableMetadata",
    "APICatalog",
    "OperationType",
    # Providers
    "BaseLLMProvider",
    "TaskRouter",
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "GrokProvider",
    "LlamaProvider",
    "OllamaProvider",
    "TogetherProvider",
    "GroqProvider",
    # Session
    "Session",
]
