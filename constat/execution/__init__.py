"""Planning and code execution."""

# Shared retry prompt template for code generation failures
RETRY_PROMPT_TEMPLATE = """Your previous code failed to execute.

{error_details}

Previous code:
```python
{previous_code}
```

Please fix the code and try again. Return ONLY the corrected Python code wrapped in ```python ... ``` markers."""


from .planner import Planner
from .executor import ExecutionResult, PythonExecutor
from .engine import QueryEngine
from .scratchpad import Scratchpad
from .fact_resolver import Fact, FactSource, FactResolver, ResolutionStrategy
from .mode import (
    ExecutionMode,
    ModeSelection,
    ExecutionConfig,
    suggest_mode,
    get_mode_system_prompt,
    get_domain_preset,
)
from .intent import (
    FollowUpIntent,
    DetectedIntent,
    IntentClassification,
    IMPLIES_REDO,
    QUICK_INTENTS,
    EXECUTION_INTENTS,
    INTENT_SUGGESTED_ACTIONS,
    ORDER_CONFIRMATION_THRESHOLD,
    from_analysis as intent_from_analysis,
)

__all__ = [
    # Shared constants
    "RETRY_PROMPT_TEMPLATE",
    # Execution
    "ExecutionResult",
    "PythonExecutor",
    "Planner",
    "QueryEngine",
    "Scratchpad",
    # Fact resolution
    "Fact",
    "FactSource",
    "FactResolver",
    "ResolutionStrategy",
    # Execution modes
    "ExecutionMode",
    "ModeSelection",
    "ExecutionConfig",
    "suggest_mode",
    "get_mode_system_prompt",
    "get_domain_preset",
    # Intent classification (LLM-based, via session._analyze_question)
    "FollowUpIntent",
    "DetectedIntent",
    "IntentClassification",
    "IMPLIES_REDO",
    "QUICK_INTENTS",
    "EXECUTION_INTENTS",
    "INTENT_SUGGESTED_ACTIONS",
    "ORDER_CONFIRMATION_THRESHOLD",
    "intent_from_analysis",
]
