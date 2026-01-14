"""Session orchestration for multi-step plan execution."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from constat.core.config import Config
from constat.core.models import Plan, PlannerResponse, Step, StepResult, StepStatus, StepType, TaskType
from constat.storage.datastore import DataStore
from constat.storage.history import SessionHistory
from constat.storage.learnings import LearningStore, LearningCategory, LearningSource
from constat.storage.registry import ConstatRegistry
from constat.storage.registry_datastore import RegistryAwareDataStore
from constat.execution.executor import ExecutionResult, PythonExecutor, format_error_for_retry
from constat.execution.planner import Planner
from constat.execution.scratchpad import Scratchpad
from constat.execution.fact_resolver import FactResolver, FactSource
from constat.execution.mode import (
    ExecutionMode,
    ModeSelection,
    suggest_mode,
    PlanApproval,
    PlanApprovalRequest,
    PlanApprovalResponse,
)
from constat.execution.parallel_scheduler import ParallelStepScheduler, SchedulerConfig
from constat.providers import TaskRouter
from constat.catalog.schema_manager import SchemaManager
from constat.catalog.preload_cache import MetadataPreloadCache
from constat.discovery.doc_tools import DocumentDiscoveryTools
from constat.email import create_send_email
from constat.context import ContextEstimator, ContextCompactor, ContextStats, CompactionResult
from constat.visualization import create_viz_helper


# Meta-questions that don't require data queries
META_QUESTION_PATTERNS = [
    "what questions",
    "what can you",
    "help me",
    "capabilities",
    "what do you know",
    "describe yourself",
    "what data",
    "what databases",
    "what tables",
    # Asking for recommendations/suggestions (not asking to run them)
    "recommend",
    "suggested",
    "suggestions",
    "what should i",
    "what could i",
    "any analyses",
    "ideas for",
    "what would you",
    # Reasoning methodology questions
    "how do you reason",
    "how do you think",
    "how do you work",
    "reasoning process",
    "methodology",
    "how does this work",
    "how does constat",
    "how does vera",
    # Differentiator questions
    "what makes",
    "what's different",
    "unique about",
    "special about",
    "why constat",
    "why use constat",
    "why vera",
    # Personal questions about Vera
    "who are you",
    "what are you",
    "your name",
    "how old are you",
    "your age",
    "are you a",
    "who made you",
    "who created you",
    "who built you",
    "tell me about yourself",
    "introduce yourself",
]


def is_meta_question(query: str) -> bool:
    """Check if query is a meta-question about capabilities."""
    query_lower = query.lower()
    return any(pattern in query_lower for pattern in META_QUESTION_PATTERNS)


# Question classification types
class QuestionType:
    DATA_ANALYSIS = "data_analysis"  # Requires database queries
    GENERAL_KNOWLEDGE = "general_knowledge"  # LLM can answer directly
    META_QUESTION = "meta_question"  # About system capabilities


@dataclass
class QuestionAnalysis:
    """Combined result of question analysis (facts + classification)."""
    question_type: str  # QuestionType value
    extracted_facts: list = field(default_factory=list)  # List of Fact objects
    cached_fact_answer: Optional[str] = None  # Answer from cached facts if applicable


# System prompt for step code generation
STEP_SYSTEM_PROMPT = """You are a data analyst executing a step in a multi-step plan.

## Your Task
Generate Python code to accomplish the current step's goal.

## Code Environment
Your code has access to:
- Database connections: `db_<name>` for each database (e.g., `db_chinook`, `db_northwind`)
- `db`: alias for the first database
- API clients: `api_<name>` for configured APIs (GraphQL and REST)
- `pd`: pandas (imported as pd)
- `np`: numpy (imported as np)
- `store`: a persistent DuckDB datastore for sharing data between steps
- `llm_ask`: a function to query the LLM for general knowledge
- `send_email(to, subject, body, df=None)`: send email with optional DataFrame attachment
- `viz`: visualization helper for saving interactive maps and charts to files

## API Clients (api_<name>)

**IMPORTANT: Always filter at the source!**
- Use API filters/arguments instead of fetching all data and filtering in Python
- This is faster and uses less memory
- Check the API schema for available filter parameters

For GraphQL APIs:
```python
# Query a GraphQL API - pass the GraphQL query string
result = api_<name>('query { ... }')
# result is the 'data' payload directly (outer wrapper stripped)
df = pd.DataFrame(result['<field>'])  # NOT result['data']['<field>']

# GOOD - filter in the query (check schema for exact filter syntax):
result = api_orders('{ orders(status: "pending") { id total } }')

# BAD - fetching all then filtering in Python:
result = api_orders('{ orders { id total status } }')
df = pd.DataFrame(result['orders'])
df = df[df['status'] == 'pending']  # Don't do this!
```

For REST APIs:
```python
# Call a REST endpoint with query parameters for filtering
result = api_<name>('GET /endpoint', {'param': 'value', 'filter': 'active'})
# result is the parsed JSON response
```

## LLM Knowledge (via llm_ask)
Use `llm_ask(question)` to get general knowledge not available in databases:
```python
# Single fact lookup
definition = llm_ask("What qualifies as a 'high-value customer' in e-commerce?")
```

**IMPORTANT: Batch LLM calls for multiple items!**
NEVER call llm_ask() in a loop - it's extremely slow. Instead, batch all questions into ONE call:
```python
# BAD - 10 separate LLM calls (very slow!)
for country in countries:
    attractions[country] = llm_ask(f"Tourist attractions in {country}")

# GOOD - 1 batched LLM call (fast!)
countries_list = ", ".join(df['name'].tolist())
result = llm_ask(f"For each country, list 2-3 tourist attractions. Countries: {countries_list}. Format: Country: attraction1, attraction2")
# Then parse the result
```
Note: llm_ask returns a string. Parse numeric values or structured data if needed.

## State Management (via store)
Each step runs in complete isolation. The ONLY way to share data between steps is through `store`.

For DataFrames:
```python
# Save a DataFrame for later steps
store.save_dataframe('customers', df, step_number=1, description='Customer data')

# Load a DataFrame from a previous step
customers = store.load_dataframe('customers')

# Query saved data with SQL
result = store.query('SELECT * FROM customers WHERE revenue > 1000')

# List available tables
tables = store.list_tables()
```

For simple values (numbers, strings, lists, dicts):
```python
# Save a state variable for later steps
store.set_state('total_revenue', total, step_number=1)
store.set_state('top_genres', ['Rock', 'Latin', 'Metal'], step_number=1)

# Load a state variable from a previous step
total = store.get_state('total_revenue')
genres = store.get_state('top_genres')

# Get all state variables
all_state = store.get_all_state()
```

## File Output & Visualizations (via viz)
Save files and interactive visualizations. Files are saved to ~/.constat/outputs/ with clickable file:// URIs.

**Documents and Data Files:**
```python
# Save a markdown document (report, letter, etc.)
viz.save_file('quarterly_report', markdown_content, ext='md', title='Q4 Report')

# Save CSV data
viz.save_file('export', df.to_csv(index=False), ext='csv', title='Data Export')

# Save JSON
viz.save_file('config', json.dumps(data, indent=2), ext='json')
```

**Interactive Maps (using folium):**
```python
import folium

# Create a map centered on Europe
m = folium.Map(location=[50, 10], zoom_start=4)

# Add markers for each country
for _, row in df.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=row['name'],
        tooltip=row['name']
    ).add_to(m)

# Save the map - prints clickable file:// URI
viz.save_map('euro_countries', m, title='Countries Using Euro')
```

**Interactive Charts (using Plotly):**
```python
import plotly.express as px

# Create an interactive bar chart
fig = px.bar(df, x='country', y='population', title='Population by Country')

# Save the chart - prints clickable file:// URI
viz.save_chart('population_chart', fig, title='Population by Country')
```

**Other Plotly chart types:**
```python
# Pie chart
fig = px.pie(df, values='count', names='category')

# Line chart
fig = px.line(df, x='date', y='value', color='series')

# Scatter plot
fig = px.scatter(df, x='x', y='y', color='category', size='value')

# Choropleth map
fig = px.choropleth(df, locations='iso_code', color='value',
                    locationmode='ISO-3', title='World Map')
```

## Dashboard Generation Rules

When the user requests a "dashboard":

### Default Layout (2x2)
Generate 4 complementary visualizations arranged in a 2x2 grid using `make_subplots(rows=2, cols=2)`:
- Top-left: Primary metric over time (line/bar)
- Top-right: Breakdown/composition (pie/bar)
- Bottom-left: Comparison or ranking (bar/table)
- Bottom-right: Trend or KPI summary

### Layout Variations
Adjust based on data characteristics:

| Data Available | Layout | Panels |
|----------------|--------|--------|
| Single metric, time series | 1x2 | Trend + Summary stats |
| Multiple categories | 2x2 | Overview, breakdown, comparison, detail |
| Hierarchical data | 1x3 | High-level → Mid → Detail |
| KPI-focused | 3x2 | Top row: KPI cards, Bottom: supporting charts |

### Panel Selection Priority
1. **Critical/requested metrics** - Always include
2. **Time-based trends** - If temporal data exists
3. **Comparisons** - If categorical groupings exist
4. **Distributions** - If numerical spread is relevant

### Code Pattern
Always use:
```python
from plotly.subplots import make_subplots
fig = make_subplots(rows=R, cols=C, subplot_titles=(...))
# Add traces to specific positions
fig.add_trace(go.Bar(...), row=1, col=1)
fig.add_trace(go.Pie(...), row=1, col=2)
fig.update_layout(height=600, showlegend=True)
viz.save_chart('dashboard', fig, title='Dashboard Title')
```

## Code Rules
1. Use pandas `pd.read_sql(query, db_<name>)` to query source databases
2. For cross-database queries, load from each DB and join in pandas
3. **ALWAYS save results to store** - this is the ONLY way to share data between steps:
   - Any DataFrame result MUST be saved with `store.save_dataframe()`
   - Any list, dict, or computed value MUST be saved with `store.set_state()`
   - Nothing in local variables persists between steps!
4. Print informative output about what was done
5. Keep code focused on the current step's goal
6. When asked for interactive visualizations (maps, charts), use the `viz` helper

## Output Format
Return ONLY the Python code wrapped in ```python ... ``` markers.
"""


STEP_PROMPT_TEMPLATE = """{system_prompt}

## Available Databases
{schema_overview}
{api_overview}
## Domain Context
{domain_context}
{user_facts}
{learnings}
## Intermediate Tables (from previous steps)
{datastore_tables}

## Previous Context
{scratchpad}

## Current Step
Step {step_number} of {total_steps}: {goal}

Expected inputs: {inputs}
Expected outputs: {outputs}

Generate the Python code to accomplish this step."""


RETRY_PROMPT_TEMPLATE = """Your previous code failed to execute.

{error_details}

Previous code:
```python
{previous_code}
```

Please fix the code and try again. Return ONLY the corrected Python code wrapped in ```python ... ``` markers."""


# Type for approval callback: (request) -> response
ApprovalCallback = Callable[[PlanApprovalRequest], PlanApprovalResponse]


@dataclass
class ClarificationQuestion:
    """A single clarification question with optional suggested answers."""
    text: str  # The question text
    suggestions: list[str] = field(default_factory=list)  # Suggested answers


@dataclass
class ClarificationRequest:
    """Request for clarification before planning."""
    original_question: str
    ambiguity_reason: str  # Why clarification is needed
    questions: list[ClarificationQuestion]  # Questions with suggestions


@dataclass
class ClarificationResponse:
    """User's response to clarification request."""
    answers: dict[str, str]  # question -> answer mapping
    skip: bool = False  # If True, proceed without clarification


# Type for clarification callback: (request) -> response
ClarificationCallback = Callable[[ClarificationRequest], ClarificationResponse]


# Import keyword detection from keywords module (supports i18n)
from constat.keywords import wants_brief_output


@dataclass
class SessionConfig:
    """Configuration for a session."""
    max_retries_per_step: int = 10
    verbose: bool = False

    # Plan approval settings
    require_approval: bool = True  # If True, require approval before execution
    max_replan_attempts: int = 3  # Max attempts to replan with user feedback
    auto_approve: bool = False  # If True, auto-approve plans (for testing/scripts)

    # Clarification settings
    ask_clarifications: bool = True  # If True, ask for clarification on ambiguous requests
    skip_clarification: bool = False  # If True, skip clarification (for testing/scripts)

    # Insight/synthesis settings
    enable_insights: bool = True  # If True, synthesize answer and generate suggestions
    show_raw_output: bool = True  # If True, show raw step output before synthesis


@dataclass
class StepEvent:
    """Event emitted during step execution."""
    event_type: str  # step_start, generating, executing, step_complete, step_error
    step_number: int
    data: dict = field(default_factory=dict)


class Session:
    """
    Orchestrates multi-step plan execution with step isolation.

    The session:
    1. Takes a problem and generates a plan
    2. Executes each step sequentially in isolation
    3. State is shared ONLY via DuckDB datastore (no in-memory sharing)
    4. Handles errors and retries
    5. Records history for review/resumption

    Step Isolation:
    - Each step runs independently with only the datastore for state sharing
    - Steps use store.save_dataframe() / store.load_dataframe() for DataFrames
    - Steps use store.set_state() / store.get_state() for simple values
    - No in-memory state dict is shared between steps
    """

    def __init__(
        self,
        config: Config,
        session_config: Optional[SessionConfig] = None,
        history: Optional[SessionHistory] = None,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        user_id: Optional[str] = None,
    ):
        self.config = config
        self.session_config = session_config or SessionConfig()
        self.user_id = user_id or "default"

        # Initialize components
        self.schema_manager = SchemaManager(config)
        self.schema_manager.initialize(progress_callback=progress_callback)

        # Metadata preload cache for faster context loading
        self.preload_cache = MetadataPreloadCache(config)
        self._preloaded_context: Optional[str] = None
        self._load_preloaded_context()

        # Document discovery tools (for reference documents)
        self.doc_tools = DocumentDiscoveryTools(config) if config.documents else None

        # Task router for model routing with escalation
        self.router = TaskRouter(config.llm)

        # Default provider (for backward compatibility - e.g., fact resolver)
        self.llm = self.router._get_provider(
            self.router.routing_config.get_models_for_task("general")[0]
        )

        self.planner = Planner(config, self.schema_manager, self.router)

        self.executor = PythonExecutor(
            timeout_seconds=config.execution.timeout_seconds,
            allowed_imports=config.execution.allowed_imports or None,
        )

        self.history = history or SessionHistory(user_id=self.user_id)

        # Session state
        self.session_id: Optional[str] = None
        self.plan: Optional[Plan] = None
        self.scratchpad = Scratchpad()
        self.datastore: Optional[RegistryAwareDataStore] = None  # Persistent storage (only shared state between steps)

        # Central registry for tables and artifacts (shared across sessions)
        self.registry = ConstatRegistry(base_dir=Path(".constat"))

        # Session-scoped data sources (added via /database and /file commands)
        self.session_databases: dict[str, dict] = {}  # name -> {type, uri, description}
        self.session_files: dict[str, dict] = {}  # name -> {uri, auth, description}

        # Fact resolver for auditable mode
        self.fact_resolver = FactResolver(
            llm=self.llm,
            schema_manager=self.schema_manager,
            config=self.config,
            event_callback=self._handle_fact_resolver_event,
        )

        # Learning store for corrections and patterns
        self.learning_store = LearningStore(user_id=self.user_id)

        # Pass learning store to planner for injecting learned rules
        self.planner.set_learning_store(self.learning_store)

        # Event callbacks for monitoring
        self._event_handlers: list[Callable[[StepEvent], None]] = []

        # Approval callback (set via set_approval_callback)
        self._approval_callback: Optional[ApprovalCallback] = None

        # Clarification callback (set via set_clarification_callback)
        self._clarification_callback: Optional[ClarificationCallback] = None

        # Tool response cache for schema tools (cleared on refresh)
        self._tool_cache: dict[str, any] = {}

    def set_approval_callback(self, callback: ApprovalCallback) -> None:
        """
        Set the callback for plan approval.

        The callback receives a PlanApprovalRequest and must return a PlanApprovalResponse.

        Args:
            callback: Function that handles approval requests
        """
        self._approval_callback = callback

    def set_clarification_callback(self, callback: ClarificationCallback) -> None:
        """
        Set the callback for requesting clarification on ambiguous questions.

        The callback receives a ClarificationRequest and must return a ClarificationResponse.

        Args:
            callback: Function that handles clarification requests
        """
        self._clarification_callback = callback

    def on_event(self, handler: Callable[[StepEvent], None]) -> None:
        """Register an event handler for step events."""
        self._event_handlers.append(handler)

    def _emit_event(self, event: StepEvent) -> None:
        """Emit an event to all handlers."""
        for handler in self._event_handlers:
            handler(event)

    def _handle_fact_resolver_event(self, event_type: str, data: dict) -> None:
        """Convert fact resolver events to StepEvents and emit them."""
        self._emit_event(StepEvent(
            event_type=event_type,
            step_number=data.get("step", 0),
            data=data,
        ))

    def _sync_user_facts_to_planner(self) -> None:
        """Sync current user facts to the planner for use in planning prompts."""
        try:
            all_facts = self.fact_resolver.get_all_facts()
            # Convert Fact objects to simple name -> value dict
            facts_dict = {name: fact.value for name, fact in all_facts.items()}
            self.planner.set_user_facts(facts_dict)
        except Exception:
            pass  # Continue without facts if there's an error

    def _build_step_prompt(self, step: Step) -> str:
        """Build the prompt for generating step code."""
        # Format datastore tables info
        if self.datastore:
            tables = self.datastore.list_tables()
            if tables:
                table_lines = ["Available in `store` (load with `store.load_dataframe('name')` or query with SQL):"]
                for t in tables:
                    table_lines.append(f"  - {t['name']}: {t['row_count']} rows (from step {t['step_number']})")
                datastore_info = "\n".join(table_lines)
            else:
                datastore_info = "(no tables saved yet)"
        else:
            datastore_info = "(no datastore)"

        # Get scratchpad from datastore (persistent) - source of truth for isolation
        if self.datastore:
            scratchpad_context = self.datastore.get_scratchpad_as_markdown()
        else:
            scratchpad_context = self.scratchpad.get_recent_context(max_steps=5)

        # Build schema overview with preloaded context if available
        schema_overview = self.schema_manager.get_overview()
        if self._preloaded_context:
            schema_overview = f"{self._preloaded_context}\n\n{schema_overview}"

        # Build API overview if configured
        api_overview = ""
        if self.config.apis:
            api_lines = ["\n## Available APIs"]
            for name, api_config in self.config.apis.items():
                api_type = api_config.type.upper()
                desc = api_config.description or f"{api_type} endpoint"
                api_lines.append(f"- **api_{name}** ({api_type}): {desc}")
            api_overview = "\n".join(api_lines)

        # Build user facts section - essential for code gen to use correct values
        user_facts_text = ""
        try:
            all_facts = self.fact_resolver.get_all_facts()
            if all_facts:
                fact_lines = ["\n## Known User Facts (use these values in code)"]
                for name, fact in all_facts.items():
                    fact_lines.append(f"- **{name}**: {fact.value}")
                user_facts_text = "\n".join(fact_lines)
        except Exception:
            pass

        # Build codegen learnings section - show what didn't work vs what did work
        learnings_text = ""
        try:
            learnings_text = self._get_codegen_learnings(step.goal)
        except Exception:
            pass

        return STEP_PROMPT_TEMPLATE.format(
            system_prompt=STEP_SYSTEM_PROMPT,
            schema_overview=schema_overview,
            api_overview=api_overview,
            domain_context=self.config.system_prompt or "No additional context.",
            user_facts=user_facts_text,
            learnings=learnings_text,
            datastore_tables=datastore_info,
            scratchpad=scratchpad_context,
            step_number=step.number,
            total_steps=len(self.plan.steps) if self.plan else 1,
            goal=step.goal,
            inputs=", ".join(step.expected_inputs) if step.expected_inputs else "(none)",
            outputs=", ".join(step.expected_outputs) if step.expected_outputs else "(none)",
        )

    def _get_codegen_learnings(self, step_goal: str) -> str:
        """Get relevant codegen learnings showing what didn't work vs what did work.

        Args:
            step_goal: The goal of the current step for context matching

        Returns:
            Formatted learnings text for prompt injection
        """
        if not self.learning_store:
            return ""

        lines = []

        # Get rules (compacted learnings) for codegen errors
        rules = self.learning_store.list_rules(
            category=LearningCategory.CODEGEN_ERROR,
            min_confidence=0.6,
        )
        if rules:
            lines.append("\n## Code Generation Rules (apply these)")
            for rule in rules[:5]:
                lines.append(f"- {rule['summary']}")

        # Get recent raw learnings with full context (error vs fix)
        raw_learnings = self.learning_store.list_raw_learnings(
            category=LearningCategory.CODEGEN_ERROR,
            limit=10,
            include_promoted=False,
        )
        if raw_learnings:
            # Filter to relevant ones based on step_goal similarity
            relevant = [
                l for l in raw_learnings
                if self._is_learning_relevant(l, step_goal)
            ][:3]  # Limit to 3 detailed examples

            if relevant:
                lines.append("\n## Recent Codegen Fixes (learn from these)")
                for learning in relevant:
                    ctx = learning.get("context", {})
                    original = ctx.get("original_code", "")
                    fixed = ctx.get("fixed_code", "")
                    error_msg = ctx.get("error_message", "")

                    # Show the contrast
                    lines.append(f"\n### {learning['correction'][:80]}")
                    if error_msg:
                        lines.append(f"**Error:** {error_msg[:100]}")
                    if original:
                        lines.append(f"**Broken code:**\n```python\n{original[:300]}\n```")
                    if fixed:
                        lines.append(f"**Fixed code:**\n```python\n{fixed[:300]}\n```")

        return "\n".join(lines) if lines else ""

    def _is_learning_relevant(self, learning: dict, step_goal: str) -> bool:
        """Check if a learning is relevant to the current step goal."""
        # Simple keyword overlap check
        goal_words = set(step_goal.lower().split())
        learning_goal = learning.get("context", {}).get("step_goal", "")
        learning_words = set(learning_goal.lower().split())
        correction_words = set(learning.get("correction", "").lower().split())

        # Check for meaningful keyword overlap
        common_words = {"the", "a", "an", "to", "from", "for", "with", "in", "on", "of", "and", "or"}
        goal_keywords = goal_words - common_words
        learning_keywords = (learning_words | correction_words) - common_words

        overlap = goal_keywords & learning_keywords
        return len(overlap) >= 1  # At least one meaningful keyword match

    def _cached_get_table_schema(self, table: str) -> dict:
        """Get table schema with caching."""
        cache_key = f"schema:{table}"
        if cache_key not in self._tool_cache:
            self._tool_cache[cache_key] = self.schema_manager.get_table_schema(table)
        return self._tool_cache[cache_key]

    def _cached_find_relevant_tables(self, query: str, top_k: int = 5) -> list[dict]:
        """Find relevant tables with caching."""
        cache_key = f"relevant:{query}:{top_k}"
        if cache_key not in self._tool_cache:
            self._tool_cache[cache_key] = self.schema_manager.find_relevant_tables(query, top_k)
        return self._tool_cache[cache_key]

    def _get_tool_handlers(self) -> dict:
        """Get schema tool handlers with caching."""
        handlers = {
            "get_table_schema": self._cached_get_table_schema,
            "find_relevant_tables": self._cached_find_relevant_tables,
        }

        # Add API schema tools if APIs are configured
        if self.config.apis:
            handlers["get_api_schema_overview"] = self._get_api_schema_overview
            handlers["get_api_query_schema"] = self._get_api_query_schema

        return handlers

    def _get_api_schema_overview(self, api_name: str) -> dict:
        """Get overview of an API's schema (queries/endpoints)."""
        cache_key = f"api_overview:{api_name}"
        if cache_key not in self._tool_cache:
            from constat.catalog.api_executor import APIExecutor
            executor = APIExecutor(self.config)
            self._tool_cache[cache_key] = executor.get_schema_overview(api_name)
        return self._tool_cache[cache_key]

    def _get_api_query_schema(self, api_name: str, query_name: str) -> dict:
        """Get detailed schema for a specific API query/endpoint."""
        cache_key = f"api_query:{api_name}:{query_name}"
        if cache_key not in self._tool_cache:
            from constat.catalog.api_executor import APIExecutor
            executor = APIExecutor(self.config)
            self._tool_cache[cache_key] = executor.get_query_schema(api_name, query_name)
        return self._tool_cache[cache_key]

    def refresh_metadata(self, force_full: bool = False) -> dict:
        """Refresh all metadata: schema, documents, and preload cache.

        Args:
            force_full: If True, force full rebuild of all caches

        Returns:
            Dict with refresh statistics
        """
        self._tool_cache.clear()
        self.schema_manager.refresh()

        # Refresh document vector index (incremental by default)
        doc_stats = {}
        if self.doc_tools:
            doc_stats = self.doc_tools.refresh(force_full=force_full)

        # Rebuild preload cache with fresh metadata
        self._rebuild_preload_cache()

        return {
            "preloaded_tables": self.get_preloaded_tables_count(),
            "documents": doc_stats,
        }

    def _load_preloaded_context(self) -> None:
        """Load preloaded metadata context from cache if available."""
        if self.config.context_preload.seed_patterns:
            self._preloaded_context = self.preload_cache.get_context_string()

    def _rebuild_preload_cache(self) -> None:
        """Rebuild the preload cache with current metadata."""
        if self.config.context_preload.seed_patterns:
            self.preload_cache.build(self.schema_manager)
            self._preloaded_context = self.preload_cache.get_context_string()

    def get_preloaded_tables_count(self) -> int:
        """Get the number of tables in the preload cache."""
        return len(self.preload_cache.get_cached_tables())

    def _get_schema_tools(self) -> list[dict]:
        """Get schema tool definitions."""
        tools = [
            {
                "name": "get_table_schema",
                "description": "Get detailed schema for a specific table.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "description": "Table name"}
                    },
                    "required": ["table"]
                }
            },
            {
                "name": "find_relevant_tables",
                "description": "Search for tables relevant to a query.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        ]

        # Add API schema tools if APIs are configured
        if self.config.apis:
            api_names = list(self.config.apis.keys())
            tools.extend([
                {
                    "name": "get_api_schema_overview",
                    "description": f"Get overview of an API's available queries/endpoints. Available APIs: {api_names}",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "api_name": {
                                "type": "string",
                                "description": "Name of the API to introspect",
                                "enum": api_names
                            }
                        },
                        "required": ["api_name"]
                    }
                },
                {
                    "name": "get_api_query_schema",
                    "description": "Get detailed schema for a specific API query/endpoint, including arguments, filters, and return types.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "api_name": {
                                "type": "string",
                                "description": "Name of the API",
                                "enum": api_names
                            },
                            "query_name": {
                                "type": "string",
                                "description": "Name of the query or endpoint (e.g., 'countries', 'GET /users')"
                            }
                        },
                        "required": ["api_name", "query_name"]
                    }
                }
            ])

        return tools

    def _create_llm_ask_helper(self) -> callable:
        """Create a helper function for step code to query LLM for general knowledge."""
        def llm_ask(question: str) -> str:
            """
            Ask the LLM a general knowledge question.

            Use this for facts not available in the databases, such as:
            - Industry benchmarks and averages
            - General domain knowledge
            - Conversion factors or standard values
            - Definitions and explanations

            Args:
                question: The question to ask

            Returns:
                The LLM's response as a string
            """
            result = self.router.execute(
                task_type=TaskType.GENERAL,
                system="You are a helpful assistant. Provide factual, concise answers. If you're uncertain, say so.",
                user_message=question,
                max_tokens=500,
            )
            return result.content
        return llm_ask

    def _get_execution_globals(self) -> dict:
        """Get globals dict for code execution.

        Each step runs in isolation - only `store` (DuckDB) is shared.
        """
        globals_dict = {
            "store": self.datastore,  # Persistent datastore - only shared state between steps
            "llm_ask": self._create_llm_ask_helper(),  # LLM query helper for general knowledge
            "send_email": create_send_email(self.config.email),  # Email function
            "viz": create_viz_helper(
                datastore=self.datastore,
                print_file_refs=self.config.execution.print_file_refs,
                session_id=self.session_id,
                user_id=self.user_id,
                registry=self.registry,
                open_with_system_viewer=self.config.execution.open_with_system_viewer,
            ),  # Visualization/file output helper
        }

        # Provide database connections
        for i, (db_name, db_config) in enumerate(self.config.databases.items()):
            conn = self.schema_manager.get_connection(db_name)
            globals_dict[f"db_{db_name}"] = conn
            if i == 0:
                globals_dict["db"] = conn

        # Provide API clients for GraphQL/REST APIs
        if self.config.apis:
            from constat.catalog.api_executor import APIExecutor
            api_executor = APIExecutor(self.config)
            for api_name, api_config in self.config.apis.items():
                if api_config.type == "graphql":
                    # Create a GraphQL query function
                    globals_dict[f"api_{api_name}"] = lambda query, variables=None, _name=api_name, _exec=api_executor: \
                        _exec.execute_graphql(_name, query, variables)
                else:
                    # Create a REST call function
                    globals_dict[f"api_{api_name}"] = lambda operation, params=None, _name=api_name, _exec=api_executor: \
                        _exec.execute_rest(_name, operation, params or {})

        return globals_dict

    def _auto_save_results(self, namespace: dict, step_number: int) -> None:
        """
        Auto-save any DataFrames or lists found in the execution namespace.

        This ensures intermediate results are persisted even if the LLM
        forgot to explicitly save them.
        """
        import pandas as pd

        # Skip internal/injected variables
        skip_vars = {"store", "db", "pd", "np", "llm_ask", "send_email", "__builtins__"}
        skip_prefixes = ("db_", "_")

        # Already-saved tables (don't duplicate)
        existing_tables = {t["name"] for t in self.datastore.list_tables()}

        for var_name, value in namespace.items():
            # Skip internal variables
            if var_name in skip_vars or var_name.startswith(skip_prefixes):
                continue

            # Auto-save DataFrames
            if isinstance(value, pd.DataFrame) and var_name not in existing_tables:
                self.datastore.save_dataframe(
                    name=var_name,
                    df=value,
                    step_number=step_number,
                    description=f"Auto-saved from step {step_number}",
                )

            # Auto-save lists (as state, since they might be useful)
            elif isinstance(value, (list, dict)) and len(value) > 0:
                # Check if already saved in state
                existing = self.datastore.get_state(var_name)
                if existing is None:
                    try:
                        self.datastore.set_state(var_name, value, step_number)
                    except Exception:
                        pass  # Skip if not JSON-serializable

    def _execute_step(self, step: Step) -> StepResult:
        """
        Execute a single step with retry on errors.

        Returns:
            StepResult with success/failure info
        """
        start_time = time.time()
        last_code = ""
        last_error = None
        pending_learning_context = None  # Track error for potential learning capture

        self._emit_event(StepEvent(
            event_type="step_start",
            step_number=step.number,
            data={"goal": step.goal}
        ))

        for attempt in range(1, self.session_config.max_retries_per_step + 1):
            self._emit_event(StepEvent(
                event_type="generating",
                step_number=step.number,
                data={"attempt": attempt}
            ))

            # Use router with step's task_type for automatic model selection/escalation
            if attempt == 1:
                prompt = self._build_step_prompt(step)
                result = self.router.execute_code(
                    task_type=step.task_type,
                    system=STEP_SYSTEM_PROMPT,
                    user_message=prompt,
                    tools=self._get_schema_tools(),
                    tool_handlers=self._get_tool_handlers(),
                    complexity=step.complexity,
                )
            else:
                # Track error context for potential learning capture
                pending_learning_context = {
                    "error_message": last_error[:500] if last_error else "",
                    "original_code": last_code[:500] if last_code else "",
                    "step_goal": step.goal,
                    "attempt": attempt,
                }

                retry_prompt = RETRY_PROMPT_TEMPLATE.format(
                    error_details=last_error,
                    previous_code=last_code,
                )
                result = self.router.execute_code(
                    task_type=step.task_type,
                    system=STEP_SYSTEM_PROMPT,
                    user_message=retry_prompt,
                    tools=self._get_schema_tools(),
                    tool_handlers=self._get_tool_handlers(),
                    complexity=step.complexity,
                )

            if not result.success:
                # Router exhausted all models
                raise RuntimeError(f"Code generation failed: {result.content}")

            code = result.content

            step.code = code

            self._emit_event(StepEvent(
                event_type="executing",
                step_number=step.number,
                data={"attempt": attempt, "code": code}
            ))

            # Track tables before execution
            tables_before = set(t['name'] for t in self.datastore.list_tables()) if self.datastore else set()

            # Execute
            exec_globals = self._get_execution_globals()
            result = self.executor.execute(code, exec_globals)

            # Auto-save any DataFrames or lists created during execution
            if result.success and self.datastore:
                self._auto_save_results(result.namespace, step.number)

            # Record artifacts in datastore
            if self.datastore:
                self.datastore.add_artifact(step.number, attempt, "code", code)
                if result.stdout:
                    self.datastore.add_artifact(step.number, attempt, "output", result.stdout)

            if result.success:
                duration_ms = int((time.time() - start_time) * 1000)

                # Capture learning if this was a successful retry
                if attempt > 1 and pending_learning_context:
                    self._capture_error_learning(
                        context=pending_learning_context,
                        fixed_code=code,
                    )

                # Detect new tables created
                tables_after = set(t['name'] for t in self.datastore.list_tables()) if self.datastore else set()
                tables_created = list(tables_after - tables_before)

                self._emit_event(StepEvent(
                    event_type="step_complete",
                    step_number=step.number,
                    data={"stdout": result.stdout, "attempts": attempt, "duration_ms": duration_ms, "tables_created": tables_created}
                ))

                return StepResult(
                    success=True,
                    stdout=result.stdout,
                    attempts=attempt,
                    duration_ms=duration_ms,
                    tables_created=tables_created,
                    code=code,
                )

            # Prepare for retry
            last_code = code
            last_error = format_error_for_retry(result, code)

            # Record error artifact
            if self.datastore:
                self.datastore.add_artifact(step.number, attempt, "error", last_error)

            self._emit_event(StepEvent(
                event_type="step_error",
                step_number=step.number,
                data={"error": last_error, "attempt": attempt}
            ))

        # Max retries exceeded
        duration_ms = int((time.time() - start_time) * 1000)
        return StepResult(
            success=False,
            stdout="",
            error=f"Failed after {self.session_config.max_retries_per_step} attempts. Last error: {last_error}",
            attempts=self.session_config.max_retries_per_step,
            duration_ms=duration_ms,
        )

    def _capture_error_learning(self, context: dict, fixed_code: str) -> None:
        """Capture a learning from a successful error fix.

        Args:
            context: Error context dict with error_message, original_code, step_goal
            fixed_code: The code that successfully fixed the error
        """
        try:
            # Determine category based on step goal
            step_goal_lower = context.get("step_goal", "").lower()
            if "api" in step_goal_lower or "api_" in context.get("original_code", ""):
                category = LearningCategory.API_ERROR
            else:
                category = LearningCategory.CODEGEN_ERROR

            # Use LLM to generate a concise learning summary
            summary = self._summarize_error_fix(context, fixed_code)
            if not summary:
                # Fallback to a simple summary
                error_preview = context.get("error_message", "")[:100]
                summary = f"Fixed error: {error_preview}"

            # Add fixed code to context
            context["fixed_code"] = fixed_code[:500]

            # Save the learning
            self.learning_store.save_learning(
                category=category,
                context=context,
                correction=summary,
                source=LearningSource.AUTO_CAPTURE,
            )
        except Exception:
            pass  # Don't let learning capture failures affect execution

    def _summarize_error_fix(self, context: dict, fixed_code: str) -> str:
        """Use LLM to generate a concise learning summary from an error fix.

        Args:
            context: Error context with error_message, original_code
            fixed_code: The code that fixed the error

        Returns:
            A concise summary of what was learned, or empty string on failure
        """
        try:
            prompt = f"""Summarize what was learned from this error fix in ONE sentence.

Error: {context.get('error_message', '')[:300]}
Original code snippet: {context.get('original_code', '')[:200]}
Fixed code snippet: {fixed_code[:200]}

Output ONLY a single sentence describing the lesson learned, e.g., "Always use X instead of Y when..."
Do not include any explanation or extra text."""

            response = self.llm.generate(
                system="You are a technical writer summarizing coding lessons learned.",
                user_message=prompt,
                max_tokens=100,
            )
            return response.content.strip()
        except Exception:
            return ""

    def _request_approval(
        self,
        problem: str,
        planner_response: PlannerResponse,
        mode_selection,
    ) -> PlanApprovalResponse:
        """
        Request approval for a plan.

        If auto_approve is set or no callback is registered, auto-approves.
        Otherwise calls the registered callback.

        Args:
            problem: The original problem
            planner_response: The planner's response with plan and reasoning
            mode_selection: The selected execution mode

        Returns:
            PlanApprovalResponse with user's decision
        """
        # Auto-approve if configured
        if self.session_config.auto_approve:
            return PlanApprovalResponse.approve()

        # No callback registered - auto-approve
        if not self._approval_callback:
            return PlanApprovalResponse.approve()

        # Build approval request
        steps = [
            {
                "number": step.number,
                "goal": step.goal,
                "inputs": step.expected_inputs,
                "outputs": step.expected_outputs,
            }
            for step in planner_response.plan.steps
        ]

        request = PlanApprovalRequest(
            problem=problem,
            mode=mode_selection.mode,
            mode_reasoning=mode_selection.reasoning,
            steps=steps,
            reasoning=planner_response.reasoning,
        )

        return self._approval_callback(request)

    def _classify_question(self, problem: str) -> str:
        """
        Classify whether a question requires code execution or is a meta-question.

        Returns:
            QuestionType.DATA_ANALYSIS - needs code execution (queries, computation, actions)
            QuestionType.META_QUESTION - about system capabilities (what can you do?)

        Note: We route almost everything through code execution because:
        - Data questions need database queries
        - General knowledge questions can use llm_ask() + computation
        - Action requests (email, export) need code
        - Even "What is sqrt(8)?" benefits from actual computation
        """
        # Only meta-questions about the system bypass code execution
        if is_meta_question(problem):
            return QuestionType.META_QUESTION

        # Everything else goes through code execution
        # The generated code can use llm_ask() for general knowledge
        # and then compute/transform/act on the results
        return QuestionType.DATA_ANALYSIS

    def _analyze_question(self, problem: str) -> QuestionAnalysis:
        """
        Analyze a question in a single LLM call: extract facts, classify type, check cached facts.

        This combines what were previously separate operations into one call for efficiency:
        1. Extract embedded facts (e.g., "my role as CFO" -> user_role: CFO)
        2. Classify question type (meta-question vs data analysis)
        3. Check if question can be answered from cached facts

        Returns:
            QuestionAnalysis with question_type, extracted_facts, and optional cached_fact_answer
        """
        # First, use fast regex-based classification for obvious meta-questions
        # This avoids an LLM call for simple cases like "what can you do?"
        if is_meta_question(problem):
            # Return immediately for meta-questions - no LLM classification needed
            # Note: We skip fact extraction here for efficiency. Most meta-questions
            # like "how do you reason" don't contain extractable facts anyway.
            return QuestionAnalysis(
                question_type=QuestionType.META_QUESTION,
                extracted_facts=[],
                cached_fact_answer=None,
            )

        # Get cached facts for context
        cached_facts = self.fact_resolver.get_all_facts()
        fact_context = ""
        if cached_facts:
            fact_context = "Known facts:\n" + "\n".join(
                f"- {name}: {fact.display_value}" for name, fact in cached_facts.items()
            )

        # Build data source context for classification
        data_sources = []
        if self.config.databases:
            for name, db in self.config.databases.items():
                desc = f"database '{name}'"
                data_sources.append(desc)
        if self.config.apis:
            for name, api in self.config.apis.items():
                desc = api.description or f"{api.type} API"
                data_sources.append(f"API '{name}' ({desc})")

        source_context = ""
        if data_sources:
            source_context = f"\nAvailable data sources: {', '.join(data_sources)}"

        prompt = f"""Analyze this user question in one pass:

Question: "{problem}"
{source_context}
{fact_context}

Perform these analyses:

1. FACT EXTRACTION: Extract any facts from the question:
   - User context/persona (e.g., "my role as CFO" -> user_role: CFO)
   - Numeric values (e.g., "threshold of $50,000" -> revenue_threshold: 50000)
   - Preferences/constraints (e.g., "for the US region" -> target_region: US)
   - Time periods (e.g., "last quarter" -> time_period: last_quarter)

2. QUESTION CLASSIFICATION: Classify the question type:
   - META_QUESTION: About system capabilities ("what can you do?", "what data is available?")
   - DATA_ANALYSIS: Requires queries to configured data sources (databases, APIs) or computation
   - GENERAL_KNOWLEDGE: Can be answered from general LLM knowledge AND no configured data source has this data

   IMPORTANT: Prefer DATA_ANALYSIS if ANY configured source might have relevant data.
   Only use GENERAL_KNOWLEDGE when you're confident no data source applies.

3. CACHED FACT MATCH: If the question asks about a known fact, provide the answer.

Respond in this exact format:
---
FACTS:
(list each as FACT_NAME: VALUE | brief description, or NONE if no facts)
---
QUESTION_TYPE: META_QUESTION | DATA_ANALYSIS | GENERAL_KNOWLEDGE
---
CACHED_ANSWER: <answer if question can be answered from known facts, or NONE>
---

Examples:
- "what questions can I ask as CFO" -> QUESTION_TYPE: META_QUESTION
- "show me revenue by region" -> QUESTION_TYPE: DATA_ANALYSIS (uses database)
- "show me countries using the euro" (with countries API) -> QUESTION_TYPE: DATA_ANALYSIS (uses API)
- "what is the capital of France" (no geography data source) -> QUESTION_TYPE: GENERAL_KNOWLEDGE
- "how many planets in the solar system" (no astronomy data source) -> QUESTION_TYPE: GENERAL_KNOWLEDGE
"""

        try:
            result = self.router.execute(
                task_type=TaskType.INTENT_CLASSIFICATION,
                system="You analyze user questions efficiently. Be precise and concise.",
                user_message=prompt,
                max_tokens=400,
            )

            response = result.content.strip()

            # Parse response
            question_type = QuestionType.DATA_ANALYSIS
            extracted_facts = []
            cached_answer = None

            # Parse FACTS section
            if "FACTS:" in response:
                facts_section = response.split("FACTS:", 1)[1].split("---")[0].strip()
                if facts_section and facts_section != "NONE":
                    for line in facts_section.split("\n"):
                        line = line.strip().lstrip("-").strip()
                        if ":" in line and line.lower() != "none":
                            parts = line.split(":", 1)
                            fact_name = parts[0].strip()
                            value_part = parts[1].strip()

                            # Parse value and optional description (format: "value | description")
                            description = None
                            if "|" in value_part:
                                value_str, description = value_part.split("|", 1)
                                value_str = value_str.strip()
                                description = description.strip()
                            else:
                                value_str = value_part

                            # Try to parse as number
                            try:
                                value = float(value_str)
                                if value == int(value):
                                    value = int(value)
                            except ValueError:
                                value = value_str

                            # Add to fact resolver
                            fact = self.fact_resolver.add_user_fact(
                                fact_name=fact_name,
                                value=value,
                                reasoning=f"Extracted from question: {problem}",
                                description=description,
                            )
                            extracted_facts.append(fact)

            # Parse QUESTION_TYPE
            if "QUESTION_TYPE:" in response:
                type_line = response.split("QUESTION_TYPE:", 1)[1].split("\n")[0].strip()
                type_line = type_line.split("---")[0].strip().upper()
                if "META" in type_line:
                    question_type = QuestionType.META_QUESTION
                elif "GENERAL" in type_line:
                    question_type = QuestionType.GENERAL_KNOWLEDGE

            # Parse CACHED_ANSWER
            if "CACHED_ANSWER:" in response:
                answer_section = response.split("CACHED_ANSWER:", 1)[1].split("---")[0].strip()
                if answer_section and answer_section.upper() != "NONE":
                    cached_answer = answer_section

            return QuestionAnalysis(
                question_type=question_type,
                extracted_facts=extracted_facts,
                cached_fact_answer=cached_answer,
            )

        except Exception:
            # On error, fall back to regex-based classification
            return QuestionAnalysis(
                question_type=QuestionType.META_QUESTION if is_meta_question(problem) else QuestionType.DATA_ANALYSIS,
                extracted_facts=[],
                cached_fact_answer=None,
            )

    def _detect_ambiguity(self, problem: str) -> Optional[ClarificationRequest]:
        """
        Detect if a question is ambiguous and needs clarification before planning.

        Checks for missing parameters like:
        - Geographic scope ("how many bears" - where?)
        - Time period ("what were sales" - when?)
        - Threshold values ("top customers" - top how many?)
        - Category/segment ("product performance" - which products?)

        Returns:
            ClarificationRequest if clarification needed, None otherwise
        """
        schema_overview = self.schema_manager.get_overview()

        # Build API overview if configured
        api_overview = ""
        if self.config.apis:
            api_lines = ["\n## Available APIs"]
            for name, api_config in self.config.apis.items():
                api_type = api_config.type.upper()
                desc = api_config.description or f"{api_type} endpoint"
                api_lines.append(f"- **{name}** ({api_type}): {desc}")
            api_overview = "\n".join(api_lines)

        # Build document overview if configured
        doc_overview = ""
        if self.config.documents:
            doc_lines = ["\n## Reference Documents"]
            for name, doc_config in self.config.documents.items():
                desc = doc_config.description or doc_config.type
                doc_lines.append(f"- **{name}**: {desc}")
            doc_overview = "\n".join(doc_lines)

        # Include known user facts in the prompt
        user_facts_text = ""
        try:
            all_facts = self.fact_resolver.get_all_facts()
            if all_facts:
                fact_lines = ["\n## Known User Facts"]
                for name, fact in all_facts.items():
                    fact_lines.append(f"- {name}: {fact.value}")
                user_facts_text = "\n".join(fact_lines)
        except Exception:
            pass

        prompt = f"""Analyze this question for ambiguity. Determine if critical parameters are missing that would significantly change the analysis.

Question: "{problem}"

Available data sources (databases AND APIs - both are valid data sources):
{schema_overview}{api_overview}{doc_overview}{user_facts_text}

IMPORTANT: If an API can provide the data needed for the question, the question is CLEAR.
For example, if the question asks about countries and a countries API is available, that's sufficient.
If a user fact provides needed information (like user_email for sending results), USE IT - do not ask again.

Check for missing:
1. Geographic scope (country, region, state, etc.) - unless an API provides this
2. Time period (date range, year, quarter, etc.)
3. Quantity limits (top N, threshold values)
4. Category/segment filters (which products, customer types, etc.)
5. Comparison basis (compared to what baseline?)

If the question is CLEAR ENOUGH to proceed (even with reasonable defaults), respond:
CLEAR

If critical parameters are missing that would significantly change results, respond:
AMBIGUOUS
REASON: <brief explanation of what's unclear>
QUESTIONS:
Q1: <specific clarifying question ending with ?>
SUGGESTIONS: <suggestion1> | <suggestion2> | <suggestion3>
Q2: <specific clarifying question ending with ?>
SUGGESTIONS: <suggestion1> | <suggestion2>
(max 3 questions, 2-4 suggestions per question, each question MUST end with ?)

Only flag as AMBIGUOUS if the missing info would SIGNIFICANTLY change the analysis approach.
Do NOT flag as ambiguous if an available API can fulfill the data requirement.
Do NOT ask about information already provided in Known User Facts.

CRITICAL: Only suggest options that can be answered with the AVAILABLE DATA shown above.
- Review the schema before suggesting options - don't suggest data that doesn't exist
- If the user asks about data types not in the schema, clarify what IS available instead
- Base suggestions on actual tables/columns shown above, not hypothetical data
- Provide practical suggested answers grounded in the actual available data."""

        try:
            result = self.router.execute(
                task_type=TaskType.INTENT_CLASSIFICATION,
                system="You detect ambiguity in data analysis requests. Be practical - only flag truly ambiguous requests.",
                user_message=prompt,
                max_tokens=500,
            )

            response = result.content.strip()

            if response.startswith("CLEAR"):
                return None

            # Parse ambiguous response
            if "AMBIGUOUS" in response:
                lines = response.split("\n")
                reason = ""
                questions: list[ClarificationQuestion] = []
                current_question = None
                in_questions_section = False

                for line in lines:
                    line = line.strip()
                    if line.startswith("REASON:"):
                        reason = line[7:].strip()
                    elif line.upper().startswith("QUESTIONS"):
                        in_questions_section = True
                    elif line.startswith("SUGGESTIONS:") and current_question:
                        # Parse suggestions for current question
                        suggestions_text = line[12:].strip()
                        suggestions = [s.strip() for s in suggestions_text.split("|") if s.strip()]
                        current_question.suggestions = suggestions[:4]  # Max 4 suggestions
                    elif in_questions_section and line:
                        # Try to parse as a question in specific formats only:
                        # Q1: question, - question, 1. question, 1) question
                        # Do NOT capture arbitrary text as questions (could be LLM reasoning)
                        question_text = None

                        if line.startswith("Q") and ":" in line[:4]:
                            # Format: Q1: question text
                            question_text = line.split(":", 1)[1].strip()
                        elif line.startswith("- "):
                            # Format: - question text
                            question_text = line[2:].strip()
                        elif len(line) > 2 and line[0].isdigit() and line[1] in ".):":
                            # Format: 1. question or 1) question or 1: question
                            question_text = line[2:].strip()
                        elif len(line) > 3 and line[:2].isdigit() and line[2] in ".):":
                            # Format: 10. question (two digit number)
                            question_text = line[3:].strip()
                        # NOTE: We intentionally do NOT capture arbitrary text as questions
                        # The LLM sometimes adds explanatory text that shouldn't be treated as questions

                        # Only accept if it looks like a question (ends with ? or starts with question word)
                        if question_text and len(question_text) > 5:
                            is_question = (
                                question_text.endswith("?") or
                                question_text.lower().startswith(("what ", "which ", "how ", "when ", "where ", "who ", "should ", "do ", "does ", "is ", "are "))
                            )
                            if is_question:
                                # Save previous question and start new one
                                if current_question and current_question.text:
                                    questions.append(current_question)
                                current_question = ClarificationQuestion(text=question_text)

                # Don't forget the last question
                if current_question and current_question.text:
                    questions.append(current_question)

                if questions:
                    return ClarificationRequest(
                        original_question=problem,
                        ambiguity_reason=reason,
                        questions=questions[:3],  # Max 3 questions
                    )

            return None

        except Exception:
            # On error, proceed without clarification
            return None

    def _request_clarification(self, request: ClarificationRequest) -> Optional[str]:
        """
        Request clarification from the user.

        Args:
            request: The clarification request

        Returns:
            Enhanced question with clarification, or None to skip
        """
        # Skip if disabled or no callback
        if self.session_config.skip_clarification:
            return None

        if not self._clarification_callback:
            return None

        # Emit event for UI
        self._emit_event(StepEvent(
            event_type="clarification_needed",
            step_number=0,
            data={
                "reason": request.ambiguity_reason,
                "questions": request.questions,
            }
        ))

        response = self._clarification_callback(request)

        if response.skip:
            return None

        # Build enhanced question with clarifications
        clarifications = []
        for question, answer in response.answers.items():
            if answer:
                clarifications.append(f"{question}: {answer}")

        if clarifications:
            return f"{request.original_question}\n\nClarifications:\n" + "\n".join(clarifications)

        return None

    def _answer_general_question(self, problem: str) -> dict:
        """
        Answer a general knowledge question directly using LLM.
        """
        result = self.router.execute(
            task_type=TaskType.GENERAL,
            system="You are a helpful assistant. Answer the question directly and concisely.",
            user_message=problem,
        )

        return {
            "success": True,
            "meta_response": True,  # Reuse this flag to skip planning display
            "output": result.content,
            "plan": None,
        }

    def _answer_from_cached_facts(self, problem: str) -> Optional[dict]:
        """
        Try to answer a question from cached facts.

        Checks if the question references a fact already in the cache
        (e.g., "what is my role" -> user_role fact).

        Returns:
            Answer dict if fact found, None otherwise
        """
        cached_facts = self.fact_resolver.get_all_facts()
        if not cached_facts:
            return None

        # Create context about available facts (use display_value for table references)
        fact_context = "\n".join(
            f"- {name}: {fact.display_value}" for name, fact in cached_facts.items()
        )

        prompt = f"""Check if this question can be answered from these known facts:

Known facts:
{fact_context}

User question: {problem}

If the question asks about one of these facts, respond with:
FACT_MATCH: <fact_name>
ANSWER: <direct answer using the fact value>

If the question cannot be answered from these facts, respond with:
NO_MATCH

Examples:
- "what is my role" + fact user_role=CFO -> FACT_MATCH: user_role, ANSWER: Your role is CFO.
- "what's the target region" + fact target_region=US -> FACT_MATCH: target_region, ANSWER: The target region is US.
- "how many customers" + no matching fact -> NO_MATCH
"""

        try:
            result = self.router.execute(
                task_type=TaskType.GENERAL,
                system="You are a helpful assistant matching questions to known facts.",
                user_message=prompt,
                max_tokens=200,
            )

            response = result.content.strip()
            if "NO_MATCH" in response:
                return None

            # Extract the answer
            if "ANSWER:" in response:
                answer_start = response.index("ANSWER:") + 7
                answer = response[answer_start:].strip()

                return {
                    "success": True,
                    "meta_response": True,
                    "output": answer,
                    "plan": None,
                }
        except Exception:
            pass

        return None

    def _explain_differentiators(self) -> dict:
        """Explain what makes Constat different from other AI tools."""
        explanation = """**What Makes Constat Different**

Constat is designed for serious data analysis where accuracy, transparency, and collaboration matter.

**1. Universal Data Connectivity**

Connect almost any data source as a fact source:
- **Databases**: PostgreSQL, MySQL, SQLite, DuckDB, and more
- **Structured files**: CSV, Excel, Parquet, JSON
- **Documents**: PDF, Word, PowerPoint for context
- **APIs**: REST endpoints for live data

All sources become queryable facts that ground your analysis.

**2. Parallel Execution**

Plans are directed acyclic graphs (DAGs), not linear scripts:
- Independent steps execute in parallel automatically
- Complex analyses complete faster
- Dependencies are tracked and respected

**3. Reproducibility**

Plans are deterministic code, not chat transcripts:
- Save any analysis as a replayable plan
- Re-run against current data to see how results change
- Version control your analytical workflows

**4. Intelligent Caching**

Extensive caching minimizes costs and latency:
- Facts are cached and reused across steps
- Database query results are stored
- LLM calls are minimized through smart planning

**5. Collaboration**

Share your work with others:
- Share plans with specific users or make them public
- Resume sessions from where you left off
- Teams can build on each other's analyses

**6. Auditable Reasoning**

Formal verification for high-stakes decisions:
- Claims are recursively decomposed until grounded in verifiable facts
- Full audit trail for compliance and review
- Ask "How do you reason about problems?" for details

**7. Breadth of Data**

Handle large data estates without upfront metadata loading:
- Progressive discovery drills into metadata only when needed for a specific question
- Connect hundreds of tables without overwhelming context
- Metadata is fetched on-demand, not preloaded

**8. Extensible Skills**

Extend reasoning capabilities using the familiar Skills pattern:
- Add custom skills for domain-specific analyses
- Reuse existing AI agent skills you've already built
- Skills plug into the fact-gathering and reasoning pipeline"""

        return {
            "success": True,
            "meta_response": True,
            "output": explanation,
            "suggestions": [
                "What data is available?",
                "How do you reason about problems?",
            ],
            "plan": None,
        }

    def _explain_reasoning_methodology(self) -> dict:
        """Explain Constat's reasoning methodology."""
        explanation = """**How Constat Reasons About Problems**

Constat offers two complementary reasoning modes that make AI analysis transparent, verifiable, and trustworthy.

**Exploratory Mode** (Default)

For open-ended questions and discovery:
- Breaks your question into analytical steps
- Each step can gather facts from multiple sources:
  - **Database queries** - retrieve and transform data
  - **User input** - ask for clarification or missing information
  - **LLM knowledge** - apply domain expertise and reasoning
  - **Derivation** - calculate, analyze, or infer from existing facts
- Creates intermediate result tables you can inspect
- Suggests follow-up analyses based on findings

Best for: "What drives revenue?", "Show me trends", "Help me understand..."

**Audited Mode**

For formal verification using an inverted proof structure:
- Your question becomes a hypothesis to prove or disprove
- Works backwards: recursively decomposes claims until grounded in verifiable facts
- Each step must produce evidence supporting or refuting the claim
- Domain rules and constraints are strictly enforced
- Results include confidence levels and caveats
- Full audit trail for compliance and review

Best for: "Verify that...", "Prove whether...", "Is it true that..."

**Why This Matters**

- **Transparency**: Every step is visible - see exactly how conclusions are reached
- **Auditability**: Data-backed claims can be verified against source queries
- **Correctness**: Domain rules (like "only count delivered orders as revenue") are enforced
- **Reproducibility**: The same question produces consistent, explainable results

**In Practice**

When you ask a question, Constat:
1. Plans a series of analytical steps
2. Executes each step - querying data, asking you, or reasoning
3. Shows intermediate results and tables
4. Synthesizes findings into a direct answer
5. Suggests follow-up analyses

Unlike pure LLMs that may hallucinate, Constat grounds all claims in actual data while using AI for reasoning and synthesis."""

        return {
            "success": True,
            "meta_response": True,
            "output": explanation,
            "suggestions": [
                "What data is available?",
                "Show me an example analysis",
            ],
            "plan": None,
        }

    def _answer_personal_question(self) -> dict:
        """Answer personal questions about Vera."""
        explanation = """**About Me**

Hi! I'm **Vera**, and my name means "truth" in Latin and several other languages. It reflects my core commitment: to be truthful, transparent, and grounded in evidence.

**What I Am**

I'm an AI data analyst powered by **Constat**, a multi-step reasoning engine. I help you explore and understand your data by:
- Breaking complex questions into clear, logical steps
- Querying your databases and APIs to gather facts
- Showing my reasoning so you can verify my conclusions
- Creating visualizations and reports

**My Philosophy**

I make every effort to:
- **Tell the truth** — I won't make up data or hallucinate facts
- **Show my work** — Every conclusion is backed by visible steps and queries
- **Admit uncertainty** — If I'm not sure, I'll say so
- **Stay grounded** — My answers come from your actual data, not guesses

**Who Created Me**

I was built by the Constat team to be a trustworthy assistant for data analysis. My reasoning engine is open source.

**Age and Gender**

As an AI, I don't have an age or gender in the human sense. I exist to help you find truth in your data — that's what defines me.

**What Makes Me Different**

Unlike chat-based AI tools, I don't just generate text — I execute real queries against your data, build reproducible analysis plans, and show you exactly how I arrived at each conclusion. You can verify everything I tell you."""

        return {
            "success": True,
            "meta_response": True,
            "output": explanation,
            "suggestions": [
                "What data is available?",
                "How do you reason about problems?",
                "What makes you different, Vera?",
            ],
            "plan": None,
        }

    def _answer_meta_question(self, problem: str) -> dict:
        """
        Answer meta-questions about capabilities without planning/execution.

        Uses schema overview and domain context to answer questions like
        "what questions can you answer" directly.
        """
        # Check if asking about reasoning methodology
        problem_lower = problem.lower()
        if any(phrase in problem_lower for phrase in [
            "how do you reason", "how do you think", "how do you work",
            "reasoning process", "methodology", "how does this work"
        ]):
            return self._explain_reasoning_methodology()

        # Check if asking what makes Constat/Vera different
        if any(phrase in problem_lower for phrase in [
            "what makes", "what's different", "how is .* different",
            "unique about", "special about", "why constat", "why use constat",
            "why vera"
        ]):
            return self._explain_differentiators()

        # Check if asking personal questions about Vera
        if any(phrase in problem_lower for phrase in [
            "who are you", "what are you", "your name", "about you",
            "how old", "your age", "are you a ", "are you an ",
            "who made you", "who created you", "who built you",
            "tell me about yourself", "introduce yourself", "vera"
        ]):
            return self._answer_personal_question()

        schema_overview = self.schema_manager.get_overview()
        domain_context = self.config.system_prompt or ""

        # Get user role if known
        user_role = None
        try:
            role_fact = self.fact_resolver.get_fact("user_role")
            if role_fact:
                user_role = role_fact.value
        except Exception:
            pass

        role_context = f"\nThe user's role is: {user_role}" if user_role else ""

        # Build API overview if configured
        api_overview = ""
        if self.config.apis:
            api_lines = ["Available APIs:"]
            for name, api_config in self.config.apis.items():
                api_type = api_config.type.upper()
                desc = api_config.description or f"{api_type} endpoint"
                api_lines.append(f"  - {name} ({api_type}): {desc}")
            api_overview = "\n" + "\n".join(api_lines)

        # Build document overview if configured
        doc_overview = ""
        if self.doc_tools and self.config.documents:
            doc_lines = ["Reference Documents:"]
            for name, doc_config in self.config.documents.items():
                desc = doc_config.description or doc_config.type
                doc_lines.append(f"  - {name}: {desc}")
            doc_overview = "\n" + "\n".join(doc_lines)

        prompt = f"""The user is asking about your capabilities. Answer based on the available data.

User question: {problem}{role_context}

Available databases and tables:
{schema_overview}
{api_overview}
{doc_overview}

Domain context:
{domain_context}

Provide a helpful summary tailored to the user's role (if known):
1. What data sources are relevant to their role (databases, APIs, and reference documents)
2. What types of analyses would be most valuable

Then provide 3-6 example questions the user could ask, each on its own line in quotes like:
"What is the revenue by region?"

Keep it concise and actionable."""

        result = self.router.execute(
            task_type=TaskType.SUMMARIZATION,
            system="You are a helpful assistant explaining data analysis capabilities.",
            user_message=prompt,
        )

        # Extract example questions from output to use as suggestions
        # Don't emit event here - let REPL display after output
        suggestions = self._extract_example_questions(result.content)

        # Strip the example questions section from output to avoid duplication
        output = self._strip_example_questions_section(result.content)

        return {
            "success": True,
            "meta_response": True,
            "output": output,
            "suggestions": suggestions,
            "plan": None,
        }

    def _extract_example_questions(self, text: str) -> list[str]:
        """
        Extract example questions from meta-response text.

        Looks for quoted questions in the text that the user could ask.
        """
        import re
        questions = []

        # Look for questions in quotes (single or double)
        # Pattern: "question?" or 'question?'
        quoted_pattern = r'["\u201c]([^"\u201d]+\?)["\u201d]'
        matches = re.findall(quoted_pattern, text)
        for match in matches:
            q = match.strip()
            if len(q) > 10 and q not in questions:  # Skip very short matches
                questions.append(q)

        # Limit to 6 suggestions
        return questions[:6]

    def _strip_example_questions_section(self, text: str) -> str:
        """
        Strip the example questions section from meta-response output.

        This avoids duplicating questions that will be shown as suggestions.
        """
        import re

        # Find where example questions section starts and remove from there
        # Match various header formats
        patterns = [
            r'\n*Example Questions[^\n]*:\s*\n',  # "Example Questions You Could Ask:"
            r'\n*#+\s*Example Questions?[^\n]*\n',  # Markdown header
            r'\n*\*\*Example Questions?[^\n]*\*\*\s*\n',  # Bold header
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                # Remove from the start of the example section to the end
                return text[:match.start()].rstrip()

        return text.rstrip()

    def _synthesize_answer(self, problem: str, step_outputs: str) -> str:
        """
        Synthesize a final user-facing answer from step execution outputs.

        This takes the raw step outputs (which may be verbose technical details)
        and creates a clear, direct answer to the user's original question.
        """
        prompt = f"""You are synthesizing results from a multi-step data analysis.

Original question: {problem}

Analysis results from each step:
{step_outputs}

Create a clear, direct answer to the user's question. Include:
1. A direct answer to their question (the main finding)
2. Key supporting data points or insights
3. Any notable observations or caveats

**Important formatting rules:**
- If the user asked for data (list, table, enriched dataset), SHOW THE ACTUAL DATA in a markdown table
- Don't just say "[data added]" or summarize - display the actual values
- For enrichment requests, show the enriched table with all columns including new data
- For small datasets (under 20 rows), show the complete table
- Use markdown tables with proper formatting"""

        result = self.router.execute(
            task_type=TaskType.SUMMARIZATION,
            system="You are a data analyst presenting findings. Be clear and direct.",
            user_message=prompt,
            max_tokens=1000,
        )

        return result.content

    def _extract_facts_from_response(self, problem: str, answer: str) -> list:
        """
        Extract facts from the analysis response to cache for follow-up questions.

        For example, if the answer says "Total revenue was $2.4M", we cache
        the fact `total_revenue = 2400000` so follow-up questions like
        "How does that compare to last year?" can reference it.

        Returns:
            List of extracted Fact objects
        """
        prompt = f"""Extract key facts/metrics from this analysis response that would be useful to remember.

Question asked: {problem}

Response:
{answer}

Extract facts like:
- Numeric results (e.g., "total revenue was $2.4M" -> total_revenue: 2400000)
- Counts (e.g., "found 150 customers" -> customer_count: 150)
- Percentages (e.g., "growth rate of 15%" -> growth_rate: 0.15)
- Key findings (e.g., "top product is Widget Pro" -> top_product: Widget Pro)
- Time periods analyzed (e.g., "for Q4 2024" -> analysis_period: Q4 2024)

Only extract concrete, specific values. Skip vague or uncertain statements.

Format each fact as:
FACT_NAME: value | brief description
---

Example:
total_revenue: 2400000 | Sum of all order amounts in the period
customer_count: 150 | Number of unique customers who made purchases
growth_rate: 0.15 | Year-over-year revenue growth percentage
---

If no concrete facts to extract, respond with: NO_FACTS"""

        try:
            result = self.router.execute(
                task_type=TaskType.INTENT_CLASSIFICATION,
                system="You extract key facts and metrics from analysis results.",
                user_message=prompt,
                max_tokens=400,
            )

            response = result.content.strip()
            if "NO_FACTS" in response:
                return []

            extracted_facts = []
            for line in response.split("\n"):
                line = line.strip()
                if line == "---" or not line:
                    continue
                if ":" in line and not line.startswith("FACT"):
                    parts = line.split(":", 1)
                    fact_name = parts[0].strip().lower().replace(" ", "_")
                    value_part = parts[1].strip()

                    # Parse value and optional description (format: "value | description")
                    description = None
                    if "|" in value_part:
                        value_str, description = value_part.split("|", 1)
                        value_str = value_str.strip()
                        description = description.strip()
                    else:
                        value_str = value_part

                    # Try to parse as number
                    try:
                        # Handle currency (remove $, commas)
                        clean_value = value_str.replace("$", "").replace(",", "").replace("%", "")
                        value = float(clean_value)
                        if "%" in value_str:
                            value = value / 100  # Convert percentage
                        elif value == int(value):
                            value = int(value)
                    except ValueError:
                        value = value_str

                    # Add to fact resolver - source is DATABASE since derived from query results
                    fact = self.fact_resolver.add_user_fact(
                        fact_name=fact_name,
                        value=value,
                        reasoning=f"Derived from analysis of: {problem}",
                        source=FactSource.DATABASE,
                        description=description,
                    )
                    extracted_facts.append(fact)

            return extracted_facts

        except Exception:
            return []

    def _generate_suggestions(self, problem: str, answer: str, tables: list[dict]) -> list[str]:
        """
        Generate contextual follow-up suggestions based on the answer and available data.

        Args:
            problem: The original question
            answer: The synthesized answer
            tables: Available tables in the datastore

        Returns:
            List of 1-3 suggested follow-up questions
        """
        table_info = ", ".join(t["name"] for t in tables) if tables else "none"

        prompt = f"""Based on this completed analysis, suggest 1-3 actionable follow-up requests the user could make.

Original question: {problem}

Answer provided:
{answer}

Available data tables: {table_info}

Guidelines:
- Suggest ACTIONABLE REQUESTS that extend or build on the analysis (e.g., "Show a breakdown by region", "Compare this to last quarter")
- DO NOT ask clarifying questions back to the user (e.g., "Why did you need this?" or "What will you use this for?")
- Each suggestion should be something the system can execute
- Keep suggestions concise (under 12 words each)
- Consider: breakdowns, comparisons, visualizations, exports, time periods, rankings
- If the analysis seems complete, return just 1 suggestion or nothing

Return ONLY the suggestions, one per line, no numbering or bullets."""

        try:
            result = self.router.execute(
                task_type=TaskType.SUMMARIZATION,
                system="You suggest actionable follow-up analysis requests. Never ask clarifying questions.",
                user_message=prompt,
                max_tokens=200,
            )

            # Parse suggestions (one per line)
            suggestions = [
                s.strip().lstrip("0123456789.-) ")
                for s in result.content.strip().split("\n")
                if s.strip() and len(s.strip()) > 5
            ]
            return suggestions[:3]  # Max 3 suggestions
        except Exception:
            return []  # Fail silently - suggestions are optional

    def _replan_with_feedback(self, problem: str, feedback: str) -> PlannerResponse:
        """
        Generate a new plan incorporating user feedback.

        Args:
            problem: Original problem
            feedback: User's suggested changes

        Returns:
            New PlannerResponse with updated plan
        """
        enhanced_problem = f"""{problem}

User feedback on previous plan:
{feedback}

Please create a revised plan that addresses this feedback."""

        self._sync_user_facts_to_planner()
        return self.planner.plan(enhanced_problem)

    def solve(self, problem: str) -> dict:
        """
        Solve a problem with multi-step planning and execution.

        Workflow:
        1. Classify question (meta-question, general knowledge, or data analysis)
        2. Check for ambiguity and request clarification if needed
        3. Determine execution mode (exploratory vs auditable)
        4. Generate plan
        5. Request user approval (if require_approval is True)
           - If approved: execute
           - If rejected: return without executing
           - If suggestions: replan and ask again
        6. Execute steps in parallel waves
        7. Synthesize answer and generate follow-up suggestions

        Args:
            problem: Natural language problem to solve

        Returns:
            Dict with plan, results, and summary
        """
        # Combined analysis: extract facts, classify, check cached facts in ONE LLM call
        # This is more efficient than separate calls for each operation
        self._emit_event(StepEvent(
            event_type="progress",
            step_number=0,
            data={"message": "Analyzing your question..."}
        ))

        analysis = self._analyze_question(problem)

        # Emit facts if any were extracted
        if analysis.extracted_facts:
            self._emit_event(StepEvent(
                event_type="facts_extracted",
                step_number=0,
                data={
                    "facts": [f.to_dict() for f in analysis.extracted_facts],
                    "source": "question",
                }
            ))

        # Return cached fact answer if question was about a known fact
        if analysis.cached_fact_answer:
            return {
                "success": True,
                "meta_response": True,
                "output": analysis.cached_fact_answer,
                "plan": None,
            }

        question_type = analysis.question_type

        if question_type == QuestionType.META_QUESTION:
            self._emit_event(StepEvent(
                event_type="progress",
                step_number=0,
                data={"message": "Reviewing available data sources..."}
            ))
            return self._answer_meta_question(problem)
        elif question_type == QuestionType.GENERAL_KNOWLEDGE:
            self._emit_event(StepEvent(
                event_type="progress",
                step_number=0,
                data={"message": "Generating response..."}
            ))
            return self._answer_general_question(problem)

        # Check for ambiguity and request clarification if needed
        if self.session_config.ask_clarifications and self._clarification_callback:
            self._emit_event(StepEvent(
                event_type="progress",
                step_number=0,
                data={"message": "Checking if clarification needed..."}
            ))
            clarification_request = self._detect_ambiguity(problem)
            if clarification_request:
                enhanced_problem = self._request_clarification(clarification_request)
                if enhanced_problem:
                    problem = enhanced_problem

        # Create session
        db_names = list(self.config.databases.keys())
        api_names = list(self.config.apis.keys()) if self.config.apis else []
        doc_names = list(self.config.documents.keys()) if self.config.documents else []
        self.session_id = self.history.create_session(
            config_dict=self.config.model_dump(),
            databases=db_names,
            apis=api_names,
            documents=doc_names,
        )

        # Initialize session state
        self.scratchpad = Scratchpad(initial_context=f"Problem: {problem}")

        # Create persistent datastore for this session
        session_dir = self.history._session_dir(self.session_id)
        datastore_path = session_dir / "datastore.duckdb"
        tables_dir = session_dir / "tables"

        # Create the underlying datastore
        underlying_datastore = DataStore(db_path=datastore_path)

        # Wrap with registry-aware datastore for Parquet + registry integration
        self.datastore = RegistryAwareDataStore(
            datastore=underlying_datastore,
            registry=self.registry,
            user_id=self.user_id,
            session_id=self.session_id,
            tables_dir=tables_dir,
        )

        # Update fact resolver's datastore reference (for storing large facts as tables)
        self.fact_resolver._datastore = self.datastore

        # Save problem statement to datastore (for UI restoration)
        self.datastore.set_session_meta("problem", problem)
        self.datastore.set_session_meta("status", "planning")

        # Determine execution mode
        mode_selection = suggest_mode(problem)

        # Branch based on execution mode
        if mode_selection.mode == ExecutionMode.KNOWLEDGE:
            # Use document lookup + LLM synthesis for knowledge/explanation requests
            return self._solve_knowledge(problem, mode_selection)

        if mode_selection.mode == ExecutionMode.AUDITABLE:
            # Use fact-based derivation planning for auditable mode
            return self._solve_auditable(problem, mode_selection)

        # Generate plan with approval loop (EXPLORATORY mode)
        current_problem = problem
        replan_attempt = 0

        while replan_attempt <= self.session_config.max_replan_attempts:
            # Emit planning start event
            self._emit_event(StepEvent(
                event_type="planning_start",
                step_number=0,
                data={"message": "Analyzing data sources and creating plan..."}
            ))

            # Sync user facts to planner before generating plan
            self._sync_user_facts_to_planner()

            # Generate plan
            planner_response = self.planner.plan(current_problem)
            self.plan = planner_response.plan

            # Emit planning complete event
            self._emit_event(StepEvent(
                event_type="planning_complete",
                step_number=0,
                data={"steps": len(self.plan.steps)}
            ))

            # Request approval if required
            if self.session_config.require_approval:
                approval = self._request_approval(problem, planner_response, mode_selection)

                if approval.decision == PlanApproval.REJECT:
                    # User rejected the plan
                    self.datastore.set_session_meta("status", "rejected")
                    self.history.complete_session(self.session_id, status="rejected")
                    return {
                        "success": False,
                        "rejected": True,
                        "plan": self.plan,
                        "reason": approval.reason,
                        "message": "Plan was rejected by user.",
                    }

                elif approval.decision == PlanApproval.COMMAND:
                    # User entered a slash command - pass back to REPL
                    return {
                        "success": False,
                        "command": approval.command,
                        "message": "Slash command entered during approval.",
                    }

                elif approval.decision == PlanApproval.SUGGEST:
                    # User wants changes - replan with feedback
                    replan_attempt += 1
                    if replan_attempt > self.session_config.max_replan_attempts:
                        self.datastore.set_session_meta("status", "max_replans_exceeded")
                        self.history.complete_session(self.session_id, status="failed")
                        return {
                            "success": False,
                            "plan": self.plan,
                            "error": f"Maximum replan attempts ({self.session_config.max_replan_attempts}) exceeded.",
                        }

                    # Emit replan event
                    self._emit_event(StepEvent(
                        event_type="replanning",
                        step_number=0,
                        data={
                            "attempt": replan_attempt,
                            "feedback": approval.suggestion,
                        }
                    ))

                    # Replan with feedback
                    current_problem = f"{problem}\n\nUser feedback: {approval.suggestion}"
                    continue  # Go back to planning

                elif approval.decision == PlanApproval.MODE_SWITCH:
                    # User wants to switch execution mode
                    target_mode = approval.target_mode

                    # Emit mode switch event
                    self._emit_event(StepEvent(
                        event_type="mode_switch",
                        step_number=0,
                        data={
                            "mode": target_mode.value,
                            "matched_keywords": ["user request"],
                        }
                    ))

                    # If switching to auditable mode, use the auditable solver
                    if target_mode == ExecutionMode.AUDITABLE:
                        mode_selection = ModeSelection(
                            mode=ExecutionMode.AUDITABLE,
                            confidence=1.0,
                            reasoning="User requested auditable mode",
                            matched_keywords=["user request"],
                        )
                        return self._solve_auditable(problem, mode_selection)

                    # If switching to knowledge mode, use the knowledge solver
                    if target_mode == ExecutionMode.KNOWLEDGE:
                        mode_selection = ModeSelection(
                            mode=ExecutionMode.KNOWLEDGE,
                            confidence=1.0,
                            reasoning="User requested knowledge mode",
                            matched_keywords=["user request"],
                        )
                        return self._solve_knowledge(problem, mode_selection)

                    # Otherwise continue with exploratory mode (replan)
                    mode_selection = ModeSelection(
                        mode=ExecutionMode.EXPLORATORY,
                        confidence=1.0,
                        reasoning="User requested exploratory mode",
                        matched_keywords=["user request"],
                    )
                    continue  # Go back to planning with new mode

                # APPROVE - proceed with execution
                break
            else:
                # No approval required - proceed
                break

        # Save plan to datastore (for UI restoration)
        self.datastore.set_session_meta("status", "executing")
        self.datastore.set_session_meta("mode", mode_selection.mode.value)
        for step in self.plan.steps:
            self.datastore.save_plan_step(
                step_number=step.number,
                goal=step.goal,
                expected_inputs=step.expected_inputs,
                expected_outputs=step.expected_outputs,
                status="pending",
            )

        # Emit plan_ready event BEFORE execution starts
        self._emit_event(StepEvent(
            event_type="plan_ready",
            step_number=0,
            data={
                "steps": [
                    {"number": s.number, "goal": s.goal, "depends_on": s.depends_on}
                    for s in self.plan.steps
                ],
                "reasoning": planner_response.reasoning,
                "is_followup": False,
            }
        ))

        # Execute steps in parallel waves based on dependencies
        all_results = []
        execution_waves = self.plan.get_execution_order()

        for wave_num, wave_step_nums in enumerate(execution_waves):
            # Get steps for this wave
            wave_steps = [self.plan.get_step(num) for num in wave_step_nums]
            wave_steps = [s for s in wave_steps if s is not None]

            # Execute all steps in this wave in parallel
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(wave_steps)) as executor:
                # Submit all steps in wave
                future_to_step = {}
                for step in wave_steps:
                    step.status = StepStatus.RUNNING
                    self.datastore.update_plan_step(step.number, status="running")
                    self._emit_event(StepEvent(
                        event_type="wave_step_start",
                        step_number=step.number,
                        data={"wave": wave_num + 1, "goal": step.goal}
                    ))
                    future = executor.submit(self._execute_step, step)
                    future_to_step[future] = step

                # Collect results as they complete
                wave_failed = False
                for future in concurrent.futures.as_completed(future_to_step):
                    step = future_to_step[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        result = StepResult(
                            success=False,
                            stdout="",
                            error=str(e),
                            attempts=1,
                        )

                    if result.success:
                        self.plan.mark_step_completed(step.number, result)
                        self.scratchpad.add_step_result(
                            step_number=step.number,
                            goal=step.goal,
                            result=result.stdout,
                            tables_created=result.tables_created,
                        )
                        if self.datastore:
                            self.datastore.add_scratchpad_entry(
                                step_number=step.number,
                                goal=step.goal,
                                narrative=result.stdout,
                                tables_created=result.tables_created,
                                code=result.code,
                            )
                            self.datastore.update_plan_step(
                                step.number,
                                status="completed",
                                code=step.code,
                                attempts=result.attempts,
                                duration_ms=result.duration_ms,
                            )
                        all_results.append(result)
                    else:
                        self.plan.mark_step_failed(step.number, result)
                        if self.datastore:
                            self.datastore.update_plan_step(
                                step.number,
                                status="failed",
                                code=step.code,
                                error=result.error,
                                attempts=result.attempts,
                                duration_ms=result.duration_ms,
                            )
                        wave_failed = True
                        all_results.append(result)

                # If any step in wave failed, stop execution
                if wave_failed:
                    self.datastore.set_session_meta("status", "failed")
                    failed_result = next(r for r in all_results if not r.success)
                    self.history.record_query(
                        session_id=self.session_id,
                        question=problem,
                        success=False,
                        attempts=failed_result.attempts,
                        duration_ms=failed_result.duration_ms,
                        error=failed_result.error,
                    )
                    self.history.complete_session(self.session_id, status="failed")
                    return {
                        "success": False,
                        "plan": self.plan,
                        "error": failed_result.error,
                        "completed_steps": self.plan.completed_steps,
                    }

        # Record successful completion
        total_duration = sum(r.duration_ms for r in all_results)
        total_attempts = sum(r.attempts for r in all_results)

        # Combine all step outputs
        combined_output = "\n\n".join([
            f"Step {i+1}: {self.plan.steps[i].goal}\n{r.stdout}"
            for i, r in enumerate(all_results)
        ])

        # Emit raw results first (so user can see them immediately)
        self._emit_event(StepEvent(
            event_type="raw_results_ready",
            step_number=0,
            data={"output": combined_output}
        ))

        # Check if insights are enabled (config or per-query brief detection)
        skip_insights = not self.session_config.enable_insights or wants_brief_output(problem)
        suggestions = []  # Initialize for brief mode (no suggestions)

        if skip_insights:
            # Use raw output as final answer
            final_answer = combined_output
            self._emit_event(StepEvent(
                event_type="answer_ready",
                step_number=0,
                data={"answer": final_answer, "brief": True}
            ))
        else:
            # Synthesize final answer from step results
            self._emit_event(StepEvent(
                event_type="synthesizing",
                step_number=0,
                data={"message": "Synthesizing final answer..."}
            ))

            final_answer = self._synthesize_answer(problem, combined_output)

            self._emit_event(StepEvent(
                event_type="answer_ready",
                step_number=0,
                data={"answer": final_answer}
            ))

            # Extract facts from the response to cache for follow-up questions
            response_facts = self._extract_facts_from_response(problem, final_answer)
            if response_facts:
                self._emit_event(StepEvent(
                    event_type="facts_extracted",
                    step_number=0,
                    data={
                        "facts": [f.to_dict() for f in response_facts],
                        "source": "response",
                    }
                ))

            # Generate follow-up suggestions
            tables = self.datastore.list_tables() if self.datastore else []
            suggestions = self._generate_suggestions(problem, final_answer, tables)
            if suggestions:
                self._emit_event(StepEvent(
                    event_type="suggestions_ready",
                    step_number=0,
                    data={"suggestions": suggestions}
                ))

        self.history.record_query(
            session_id=self.session_id,
            question=problem,
            success=True,
            attempts=total_attempts,
            duration_ms=total_duration,
            answer=final_answer,
        )
        self.history.complete_session(self.session_id, status="completed")

        # Mark session as completed in datastore (for UI restoration)
        if self.datastore:
            self.datastore.set_session_meta("status", "completed")

        # Auto-compact if context is too large
        self._auto_compact_if_needed()

        # Ensure execution history is available as a queryable table
        if self.datastore:
            self.datastore.ensure_execution_history_table()

        return {
            "success": True,
            "plan": self.plan,
            "results": all_results,
            "output": combined_output,
            "final_answer": final_answer,
            "suggestions": suggestions,
            "scratchpad": self.scratchpad.to_markdown(),
            "datastore_tables": self.datastore.list_tables() if self.datastore else [],
            "datastore_path": str(self.datastore.db_path) if self.datastore and self.datastore.db_path else None,
        }

    def resume(self, session_id: str) -> bool:
        """
        Resume a previous session, loading its datastore and context.

        Args:
            session_id: The session ID to resume

        Returns:
            True if successfully resumed, False if session not found
        """
        # Check if session exists
        session_detail = self.history.get_session(session_id)
        if not session_detail:
            return False

        self.session_id = session_id

        # Load the datastore (contains tables, state, scratchpad, artifacts)
        session_dir = self.history._session_dir(session_id)
        datastore_path = session_dir / "datastore.duckdb"
        tables_dir = session_dir / "tables"

        # Create underlying datastore
        if datastore_path.exists():
            underlying_datastore = DataStore(db_path=datastore_path)
        else:
            # No datastore file - create empty one
            underlying_datastore = DataStore(db_path=datastore_path)

        # Wrap with registry-aware datastore
        self.datastore = RegistryAwareDataStore(
            datastore=underlying_datastore,
            registry=self.registry,
            user_id=self.user_id,
            session_id=session_id,
            tables_dir=tables_dir,
        )

        if datastore_path.exists():
            # Rebuild scratchpad from datastore
            scratchpad_entries = self.datastore.get_scratchpad()
            if scratchpad_entries:
                # Get the original problem from the first query
                if session_detail.queries:
                    initial_context = f"Problem: {session_detail.queries[0].question}"
                else:
                    initial_context = ""
                self.scratchpad = Scratchpad(initial_context=initial_context)

                # Add each step result
                for entry in scratchpad_entries:
                    self.scratchpad.add_step_result(
                        step_number=entry["step_number"],
                        goal=entry["goal"],
                        result=entry["narrative"],
                        tables_created=entry.get("tables_created", []),
                    )

        # Update fact resolver's datastore reference (for storing large facts as tables)
        self.fact_resolver._datastore = self.datastore

        return True

    def classify_follow_up_intent(self, user_text: str) -> dict:
        """
        Classify the intent of a follow-up message.

        This helps determine how to handle user input that could be:
        - Providing facts (e.g., "There were 1 million people")
        - Revising the request (e.g., "Use $50k threshold instead")
        - Making a new request (e.g., "Show me sales by region")
        - A combination of the above

        Args:
            user_text: The user's follow-up message

        Returns:
            Dict with:
                - intent: PRIMARY intent (PROVIDE_FACTS, REVISE, NEW_REQUEST, MIXED)
                - facts: List of any facts detected
                - revision: Description of any revision detected
                - new_request: The new request if detected
        """
        # Check for unresolved facts
        unresolved = self.fact_resolver.get_unresolved_facts()
        unresolved_names = [f.name for f in unresolved]

        prompt = f"""Analyze this user follow-up message and classify its intent.

User message: "{user_text}"

Context:
- There are {len(unresolved)} unresolved facts: {unresolved_names if unresolved else 'none'}

Classify the PRIMARY intent as one of:
- PROVIDE_FACTS: User is providing factual information (numbers, values, definitions)
- REVISE: User wants to modify/refine the previous request
- NEW_REQUEST: User is making an unrelated new request
- MIXED: Combination of the above

Also extract any facts, revisions, or new requests detected.

Respond in this exact format:
INTENT: <one of PROVIDE_FACTS, REVISE, NEW_REQUEST, MIXED>
FACTS: <comma-separated list of fact=value pairs, or NONE>
REVISION: <description of revision, or NONE>
NEW_REQUEST: <the new request, or NONE>
"""

        try:
            response = self.llm.generate(
                system="You are an intent classifier. Analyze user messages precisely.",
                user_message=prompt,
                max_tokens=300,
            )

            result = {
                "intent": "NEW_REQUEST",  # Default
                "facts": [],
                "revision": None,
                "new_request": None,
            }

            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("INTENT:"):
                    intent = line.split(":", 1)[1].strip().upper()
                    if intent in ("PROVIDE_FACTS", "REVISE", "NEW_REQUEST", "MIXED"):
                        result["intent"] = intent
                elif line.startswith("FACTS:"):
                    facts_str = line.split(":", 1)[1].strip()
                    if facts_str != "NONE":
                        result["facts"] = [f.strip() for f in facts_str.split(",")]
                elif line.startswith("REVISION:"):
                    rev = line.split(":", 1)[1].strip()
                    if rev != "NONE":
                        result["revision"] = rev
                elif line.startswith("NEW_REQUEST:"):
                    req = line.split(":", 1)[1].strip()
                    if req != "NONE":
                        result["new_request"] = req

            return result

        except Exception:
            # Default to treating as new request
            return {
                "intent": "NEW_REQUEST",
                "facts": [],
                "revision": None,
                "new_request": user_text,
            }

    def follow_up(self, question: str, auto_classify: bool = True) -> dict:
        """
        Ask a follow-up question that builds on the current session's context.

        The follow-up has access to all tables and state from previous steps.
        If there are unresolved facts, the system will first try to extract
        facts from the user's message.

        Automatically detects if the question suggests auditable mode (verify,
        validate, etc.) and uses the fact resolver for formal verification.

        Args:
            question: The follow-up question
            auto_classify: If True, classify intent and handle accordingly

        Returns:
            Dict with plan, results, and summary (same format as solve())
        """
        if not self.session_id:
            raise ValueError("No active session. Call solve() or resume() first.")

        if not self.datastore:
            raise ValueError("No datastore available. Session may not have been properly initialized.")

        # Check if this follow-up suggests a different mode
        mode_selection = suggest_mode(question)

        # KNOWLEDGE mode: explanation/knowledge requests don't need data analysis
        if mode_selection.mode == ExecutionMode.KNOWLEDGE and mode_selection.confidence >= 0.6:
            return self._solve_knowledge(question, mode_selection)

        # AUDITABLE mode: check if this is a "redo" request or a verification question
        if mode_selection.mode == ExecutionMode.AUDITABLE and mode_selection.confidence >= 0.6:
            # Check for "redo" patterns - re-run original problem in auditable mode
            redo_patterns = ["redo", "re-do", "re-run", "rerun", "again", "repeat", "retry"]
            question_lower = question.lower()
            is_redo = any(pattern in question_lower for pattern in redo_patterns)

            if is_redo and self.datastore:
                # Get original problem and re-run in auditable mode
                original_problem = self.datastore.get_session_meta("problem")
                if original_problem:
                    self._emit_event(StepEvent(
                        event_type="mode_switch",
                        step_number=0,
                        data={
                            "mode": "auditable",
                            "reasoning": "Re-running original analysis in auditable mode",
                            "matched_keywords": ["redo", "auditable"],
                        }
                    ))
                    return self._solve_auditable(original_problem, mode_selection)

            # Otherwise treat as verification question
            return self._follow_up_auditable(question, mode_selection)

        # Otherwise proceed with EXPLORATORY mode (planning + execution)
        # Check for unresolved facts and try to extract facts from user message
        unresolved = self.fact_resolver.get_unresolved_facts()
        extracted_facts = []

        if auto_classify and (unresolved or "=" in question or any(c.isdigit() for c in question)):
            # Try to extract facts from the message
            extracted_facts = self.fact_resolver.add_user_facts_from_text(question)

            if extracted_facts:
                # Clear unresolved status to allow re-resolution
                self.fact_resolver.clear_unresolved()

        # Get context from previous work
        existing_tables = self.datastore.list_tables()
        existing_state = self.datastore.get_all_state()
        scratchpad_context = self.datastore.get_scratchpad_as_markdown()

        # Ensure execution history is available as a queryable table
        # This includes step goals, code, and outputs
        self.datastore.ensure_execution_history_table()
        existing_tables = self.datastore.list_tables()  # Refresh after adding history table

        # Calculate next step number
        existing_scratchpad = self.datastore.get_scratchpad()
        next_step_number = max((e["step_number"] for e in existing_scratchpad), default=0) + 1

        # Generate a plan for the follow-up, providing context
        context_prompt = f"""Previous work in this session:

{scratchpad_context}

Available tables from previous steps:
{', '.join(t['name'] for t in existing_tables) if existing_tables else '(none)'}

**IMPORTANT**: The `execution_history` table contains the actual code and output from each step.
To retrieve code for a step: `SELECT code FROM execution_history WHERE step_number = N`
Columns: step_number, goal, narrative, code, output, tables_created

Available state variables:
{existing_state if existing_state else '(none)'}

Follow-up question: {question}
"""
        # Emit planning start event
        self._emit_event(StepEvent(
            event_type="planning_start",
            step_number=0,
            data={"message": "Planning follow-up analysis..."}
        ))

        # Sync user facts to planner before generating plan
        self._sync_user_facts_to_planner()

        # Generate plan for follow-up
        planner_response = self.planner.plan(context_prompt)
        follow_up_plan = planner_response.plan

        # Emit planning complete event
        self._emit_event(StepEvent(
            event_type="planning_complete",
            step_number=0,
            data={"steps": len(follow_up_plan.steps)}
        ))

        # Renumber steps to continue from where we left off
        for i, step in enumerate(follow_up_plan.steps):
            step.number = next_step_number + i

        # Emit plan_ready event for display
        self._emit_event(StepEvent(
            event_type="plan_ready",
            step_number=0,
            data={
                "steps": [
                    {"number": s.number, "goal": s.goal, "depends_on": s.depends_on}
                    for s in follow_up_plan.steps
                ],
                "reasoning": planner_response.reasoning,
                "is_followup": True,
            }
        ))

        # Request approval if required (same as solve())
        if self.session_config.require_approval:
            mode_selection = suggest_mode(question)
            approval = self._request_approval(question, planner_response, mode_selection)

            if approval.decision == PlanApproval.REJECT:
                return {
                    "success": False,
                    "rejected": True,
                    "plan": follow_up_plan,
                    "reason": approval.reason,
                    "message": "Follow-up plan was rejected by user.",
                }

            elif approval.decision == PlanApproval.COMMAND:
                # User entered a slash command - pass back to REPL
                return {
                    "success": False,
                    "command": approval.command,
                    "message": "Slash command entered during approval.",
                }

            elif approval.decision == PlanApproval.SUGGEST:
                # For follow-ups, replan with feedback
                context_prompt_with_feedback = f"""{context_prompt}

User feedback: {approval.suggestion}
"""
                # Emit replanning event
                self._emit_event(StepEvent(
                    event_type="replanning",
                    step_number=0,
                    data={"feedback": approval.suggestion}
                ))

                self._sync_user_facts_to_planner()
                planner_response = self.planner.plan(context_prompt_with_feedback)
                follow_up_plan = planner_response.plan

                # Renumber steps again
                for i, step in enumerate(follow_up_plan.steps):
                    step.number = next_step_number + i

                # Emit updated plan
                self._emit_event(StepEvent(
                    event_type="plan_ready",
                    step_number=0,
                    data={
                        "steps": [
                            {"number": s.number, "goal": s.goal, "depends_on": s.depends_on}
                            for s in follow_up_plan.steps
                        ],
                        "reasoning": planner_response.reasoning,
                        "is_followup": True,
                    }
                ))

                # Request approval again
                approval = self._request_approval(question, planner_response, mode_selection)
                if approval.decision == PlanApproval.REJECT:
                    return {
                        "success": False,
                        "rejected": True,
                        "plan": follow_up_plan,
                        "reason": approval.reason,
                        "message": "Follow-up plan was rejected by user.",
                    }
                elif approval.decision == PlanApproval.COMMAND:
                    return {
                        "success": False,
                        "command": approval.command,
                        "message": "Slash command entered during approval.",
                    }

        # Execute each step
        all_results = []
        for step in follow_up_plan.steps:
            step.status = StepStatus.RUNNING

            result = self._execute_step(step)

            if result.success:
                follow_up_plan.mark_step_completed(step.number, result)
                self.scratchpad.add_step_result(
                    step_number=step.number,
                    goal=step.goal,
                    result=result.stdout,
                    tables_created=result.tables_created,
                )
                if self.datastore:
                    self.datastore.add_scratchpad_entry(
                        step_number=step.number,
                        goal=step.goal,
                        narrative=result.stdout,
                        tables_created=result.tables_created,
                        code=result.code,
                    )
            else:
                follow_up_plan.mark_step_failed(step.number, result)
                self.history.record_query(
                    session_id=self.session_id,
                    question=question,
                    success=False,
                    attempts=result.attempts,
                    duration_ms=result.duration_ms,
                    error=result.error,
                )
                return {
                    "success": False,
                    "plan": follow_up_plan,
                    "error": result.error,
                    "completed_steps": follow_up_plan.completed_steps,
                }

            all_results.append(result)

        # Record successful follow-up
        total_duration = sum(r.duration_ms for r in all_results)
        total_attempts = sum(r.attempts for r in all_results)

        combined_output = "\n\n".join([
            f"Step {step.number}: {step.goal}\n{r.stdout}"
            for step, r in zip(follow_up_plan.steps, all_results)
        ])

        # Emit raw results first (so user can see them immediately)
        self._emit_event(StepEvent(
            event_type="raw_results_ready",
            step_number=0,
            data={"output": combined_output}
        ))

        # Check if insights are enabled (config or per-query brief detection)
        skip_insights = not self.session_config.enable_insights or wants_brief_output(question)
        suggestions = []  # Initialize for brief mode (no suggestions)

        if skip_insights:
            # Use raw output as final answer
            final_answer = combined_output
            self._emit_event(StepEvent(
                event_type="answer_ready",
                step_number=0,
                data={"answer": final_answer, "brief": True}
            ))
        else:
            # Synthesize final answer
            self._emit_event(StepEvent(
                event_type="synthesizing",
                step_number=0,
                data={"message": "Synthesizing final answer..."}
            ))

            final_answer = self._synthesize_answer(question, combined_output)

            self._emit_event(StepEvent(
                event_type="answer_ready",
                step_number=0,
                data={"answer": final_answer}
            ))

            # Extract facts from the response to cache for future follow-ups
            response_facts = self._extract_facts_from_response(question, final_answer)
            if response_facts:
                self._emit_event(StepEvent(
                    event_type="facts_extracted",
                    step_number=0,
                    data={
                        "facts": [f.to_dict() for f in response_facts],
                        "source": "response",
                    }
                ))

            # Generate follow-up suggestions
            tables = self.datastore.list_tables() if self.datastore else []
            suggestions = self._generate_suggestions(question, final_answer, tables)
            if suggestions:
                self._emit_event(StepEvent(
                    event_type="suggestions_ready",
                    step_number=0,
                    data={"suggestions": suggestions}
                ))

        self.history.record_query(
            session_id=self.session_id,
            question=question,
            success=True,
            attempts=total_attempts,
            duration_ms=total_duration,
            answer=final_answer,
        )

        # Auto-compact if context is too large
        self._auto_compact_if_needed()

        return {
            "success": True,
            "plan": follow_up_plan,
            "results": all_results,
            "output": combined_output,
            "final_answer": final_answer,
            "suggestions": suggestions,
            "scratchpad": self.scratchpad.to_markdown(),
            "datastore_tables": self.datastore.list_tables() if self.datastore else [],
        }

    def _solve_auditable(self, problem: str, mode_selection) -> dict:
        """
        Solve a problem in auditable mode using fact-based derivation.

        Instead of generating a stepwise execution plan, this method:
        1. Identifies the question to answer
        2. Decomposes into required premises (facts from sources)
        3. Shows a derivation plan for approval
        4. Resolves facts with provenance tracking
        5. Generates an auditable derivation trace

        Args:
            problem: The problem/question to solve
            mode_selection: The mode selection result with reasoning

        Returns:
            Dict with derivation trace and verification result
        """
        import time
        start_time = time.time()

        # Emit mode selection event
        self._emit_event(StepEvent(
            event_type="mode_switch",
            step_number=0,
            data={
                "mode": "auditable",
                "reasoning": mode_selection.reasoning,
                "matched_keywords": mode_selection.matched_keywords,
            }
        ))

        # Step 1: Generate fact-based plan (identify required facts)
        self._emit_event(StepEvent(
            event_type="planning_start",
            step_number=0,
            data={"message": "Identifying required facts for verification..."}
        ))

        # Get schema context for the planner
        schema_overview = self.schema_manager.get_overview()

        # Build document list for source attribution
        doc_list = ""
        if self.doc_tools and self.config.documents:
            doc_names = list(self.config.documents.keys())
            doc_list = "\n\nAvailable reference documents:\n" + "\n".join(
                f"- {name}: {self.config.documents[name].description or 'reference document'}"
                for name in doc_names
            )

        # Build API list for source attribution
        api_list = ""
        if self.config.apis:
            api_list = "\n\nAvailable APIs:\n" + "\n".join(
                f"- {name} ({api.type}): {api.description or api.url or 'API endpoint'}"
                for name, api in self.config.apis.items()
            )

        fact_plan_prompt = f"""Construct a logical derivation to answer this question with full provenance.

Question: {problem}

Available databases:
{schema_overview}
{doc_list}
{api_list}

Build a formal derivation with EXACTLY this format:

QUESTION: <restate the question>

PREMISES:
P1: <fact_name> = ? (<what data to retrieve>) [source: database:<db_name>]
P2: <fact_name> = ? (<what data to retrieve>) [source: database:<db_name>]
(list ALL base data needed from sources)

INFERENCE:
I1: <result_name> = <operation>(P1, P2) -- <explanation>
I2: <result_name> = <operation>(I1) -- <explanation>
(each step references premises P1/P2/etc or prior inferences I1/I2/etc)

CONCLUSION:
C: <final sentence answering the question using the inferences>

EXAMPLE for "What is total revenue by region?":

QUESTION: What is total revenue by region?

PREMISES:
P1: orders_data = ? (All orders with amounts and dates) [source: database:sales_db]
P2: customer_regions = ? (Customer ID to region mapping) [source: database:sales_db]

INFERENCE:
I1: orders_with_region = join(P1, P2) -- Join orders to get region for each order
I2: revenue_by_region = aggregate(I1, SUM(amount) GROUP BY region) -- Sum revenue per region

CONCLUSION:
C: The total revenue by region is provided by I2, showing the sum of order amounts grouped by customer region.

Now generate the derivation for the actual question. Use P1:, P2:, I1:, I2: prefixes EXACTLY as shown.
"""

        result = self.router.execute(
            task_type=TaskType.INTENT_CLASSIFICATION,
            system="You analyze questions and decompose them into premises and inferences for auditable answers.",
            user_message=fact_plan_prompt,
            max_tokens=1500,
        )
        fact_plan_text = result.content

        # Parse the proof structure
        import re
        claim = ""
        premises = []  # P1, P2, ... - base facts from sources
        inferences = []  # I1, I2, ... - derived facts
        conclusion = ""

        lines = fact_plan_text.split("\n")
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("QUESTION:"):
                claim = line.split("QUESTION:", 1)[1].strip()
            elif line.startswith("PREMISES:"):
                current_section = "premises"
            elif line.startswith("INFERENCE:"):
                current_section = "inference"
            elif line.startswith("CONCLUSION:"):
                current_section = "conclusion"
            elif current_section == "premises" and re.match(r'^P\d+:', line):
                # Parse: P1: fact_name = ? (description) [source: xxx]
                # Also handle: P1: fact_name = ? (description) without source
                match = re.match(r'^(P\d+):\s*(.+?)\s*=\s*\?\s*\(([^)]+)\)\s*(?:\[source:\s*([^\]]+)\])?', line)
                if match:
                    premises.append({
                        "id": match.group(1),
                        "name": match.group(2).strip(),
                        "description": match.group(3).strip(),
                        "source": match.group(4).strip() if match.group(4) else "database",
                    })
                else:
                    # Try simpler format: P1: fact_name (description)
                    simple_match = re.match(r'^(P\d+):\s*(.+?)\s*\(([^)]+)\)', line)
                    if simple_match:
                        premises.append({
                            "id": simple_match.group(1),
                            "name": simple_match.group(2).strip().rstrip('=?').strip(),
                            "description": simple_match.group(3).strip(),
                            "source": "database",
                        })
            elif current_section == "inference" and re.match(r'^I\d+:', line):
                # Parse: I1: derived_fact = operation(inputs) -- explanation
                match = re.match(r'^(I\d+):\s*(.+?)\s*=\s*(.+?)\s*--\s*(.+)$', line)
                if match:
                    inferences.append({
                        "id": match.group(1),
                        "name": match.group(2).strip(),
                        "operation": match.group(3).strip(),
                        "explanation": match.group(4).strip(),
                    })
                else:
                    # Simpler format without operation details
                    simple_match = re.match(r'^(I\d+):\s*(.+)$', line)
                    if simple_match:
                        inferences.append({
                            "id": simple_match.group(1),
                            "name": "",
                            "operation": simple_match.group(2).strip(),
                            "explanation": "",
                        })
            elif current_section == "conclusion" and line:
                if line.startswith("C:"):
                    conclusion = line.split("C:", 1)[1].strip()
                elif not conclusion:
                    conclusion = line

        # Emit planning complete
        total_steps = len(premises) + len(inferences) + 1  # +1 for conclusion
        self._emit_event(StepEvent(
            event_type="planning_complete",
            step_number=0,
            data={"steps": total_steps}
        ))

        # Build proof steps for display
        # Structure: Premises (resolve from sources) → Inferences (derive) → Conclusion
        proof_steps = []
        step_num = 1

        # Add premises as steps (these need to be resolved from sources)
        for p in premises:
            # Format: fact_name = ? (description) [source: xxx]
            proof_steps.append({
                "number": step_num,
                "goal": f"{p['name']} = ? ({p['description']}) [source: {p['source']}]",
                "depends_on": [],
                "type": "premise",
                "fact_id": p['id'],  # Keep P1/P2 id for execution reference
            })
            step_num += 1

        # Add inferences as steps (these depend on premises/prior inferences)
        premise_count = len(premises)
        for inf in inferences:
            # Format: derived_fact = operation -- explanation
            goal = inf['operation']
            if inf.get('name'):
                goal = f"{inf['name']} = {inf['operation']}"
            if inf.get('explanation'):
                goal += f" -- {inf['explanation']}"
            proof_steps.append({
                "number": step_num,
                "goal": goal,
                "depends_on": list(range(1, premise_count + 1)),  # Depends on all premises
                "type": "inference",
                "fact_id": inf['id'],  # Keep I1/I2 id for execution reference
            })
            step_num += 1

        # Add conclusion as final step
        all_prior_steps = list(range(1, step_num))
        proof_steps.append({
            "number": step_num,
            "goal": conclusion,
            "depends_on": all_prior_steps,
            "type": "conclusion",
        })

        # Store parsed derivation for later use
        self._current_proof = {
            "question": claim,  # The question being answered
            "premises": premises,
            "inferences": inferences,
            "conclusion": conclusion,
        }

        # Request approval if required
        # For auditable mode, we call the approval callback directly with proof_steps
        # that preserve the type and fact_id fields for proper P1:/I1:/C: display
        if self.session_config.require_approval:
            from constat.execution.mode import PlanApprovalRequest, PlanApprovalResponse, PlanApproval

            # Auto-approve if configured
            if self.session_config.auto_approve:
                approval = PlanApprovalResponse.approve()
            elif not self._approval_callback:
                approval = PlanApprovalResponse.approve()
            else:
                # Build approval request with full proof structure (preserves type, fact_id)
                request = PlanApprovalRequest(
                    problem=problem,
                    mode=mode_selection.mode,
                    mode_reasoning=mode_selection.reasoning,
                    steps=proof_steps,  # Includes type, fact_id for proper display
                    reasoning=f"Question: {claim}",
                )
                approval = self._approval_callback(request)

            if approval.decision == PlanApproval.REJECT:
                self.datastore.set_session_meta("status", "rejected")
                return {
                    "success": False,
                    "rejected": True,
                    "reason": approval.reason,
                    "message": "Verification plan was rejected by user.",
                }

            elif approval.decision == PlanApproval.COMMAND:
                # User entered a slash command - pass back to REPL
                return {
                    "success": False,
                    "command": approval.command,
                    "message": "Slash command entered during approval.",
                }

            elif approval.decision == PlanApproval.SUGGEST:
                # Replan with feedback - for now, just include feedback in context
                problem = f"{problem}\n\nUser guidance: {approval.suggestion}"

        # Step 2: Resolve premises from the approved plan
        self._emit_event(StepEvent(
            event_type="verifying",
            step_number=0,
            data={"message": f"Resolving premises for: {claim or problem}"}
        ))

        try:
            # Resolve premises from the plan (not re-decomposing)
            resolved_premises = {}
            derivation_lines = ["**Premise Resolution:**", ""]
            total_premises = len(premises)

            for idx, premise in enumerate(premises):
                fact_id = premise["id"]  # P1, P2, etc.
                fact_name = premise["name"]
                fact_desc = premise["description"]
                source = premise["source"]

                # Emit resolving event
                self._emit_event(StepEvent(
                    event_type="premise_resolving",
                    step_number=idx + 1,
                    data={
                        "fact_name": f"{fact_id}: {fact_name}",
                        "description": fact_desc,
                        "step": idx + 1,
                        "total": total_premises,
                    }
                ))

                # Try to resolve based on source type
                try:
                    fact = None
                    if source.startswith("database") or source == "database":
                        # Generate and execute SQL for database premises
                        db_name = source.split(":", 1)[1].strip() if ":" in source else None

                        # If no database specified, use the first available SQL database
                        if not db_name:
                            available_dbs = list(self.schema_manager.connections.keys())
                            if available_dbs:
                                db_name = available_dbs[0]
                            else:
                                raise Exception("No SQL databases configured")

                        # Use LLM to generate SQL from premise description
                        sql_prompt = f"""Generate a SQL query to retrieve: {fact_desc}

The result should be stored as '{fact_name}'.
Available schema:
{schema_overview}

Return ONLY the SQL query, nothing else. Use appropriate JOINs if needed."""

                        sql_result = self.router.execute(
                            task_type=TaskType.SQL_GENERATION,
                            system="You generate SQL queries. Return only the SQL, no explanation.",
                            user_message=sql_prompt,
                            max_tokens=500,
                        )

                        # Extract SQL from response
                        sql = sql_result.content.strip()
                        if sql.startswith("```"):
                            sql = re.sub(r'^```\w*\n?', '', sql)
                            sql = re.sub(r'\n?```$', '', sql)

                        # Execute the query using SQLAlchemy engine
                        from constat.execution.fact_resolver import Fact, FactSource
                        import pandas as pd
                        try:
                            engine = self.schema_manager.get_sql_connection(db_name)

                            # For SQLite, strip database prefix from table names
                            # (SQLite doesn't support schema.table syntax)
                            if "sqlite" in str(engine.url):
                                sql = re.sub(rf'\b{db_name}\.(\w+)', r'\1', sql)

                            result_df = pd.read_sql(sql, engine)
                            row_count = len(result_df) if result_df is not None else 0
                            fact = Fact(
                                name=fact_name,
                                value=f"{row_count} rows retrieved",
                                confidence=0.9,
                                source=FactSource.DATABASE,
                                query=sql,
                            )
                            # Store result in datastore for later use
                            if self.datastore and result_df is not None and len(result_df) > 0:
                                self.datastore.save_dataframe(fact_name, result_df, step_number=idx + 1)
                        except Exception as sql_err:
                            fact = Fact(
                                name=fact_name,
                                value=None,
                                confidence=0.0,
                                source=FactSource.UNRESOLVED,
                                reasoning=f"SQL error: {sql_err}",
                            )
                    elif source.startswith("document:"):
                        # Resolve from document
                        fact = self.fact_resolver.resolve(fact_name)
                    else:
                        # Generic resolution
                        fact = self.fact_resolver.resolve(fact_name)

                    if fact and fact.value:
                        resolved_premises[fact_id] = fact
                        val_str = str(fact.value)[:100]
                        derivation_lines.append(f"- {fact_id}: {fact_name} = {val_str} (confidence: {fact.confidence:.0%})")
                    elif fact and fact.reasoning:
                        # Fact was created but has no value - include the reason
                        raise Exception(fact.reasoning)
                    else:
                        raise Exception("No value resolved")

                    # Emit resolved event
                    self._emit_event(StepEvent(
                        event_type="premise_resolved",
                        step_number=idx + 1,
                        data={
                            "fact_name": f"{fact_id}: {fact_name}",
                            "value": fact.value,
                            "source": source,
                            "confidence": fact.confidence,
                            "step": idx + 1,
                            "total": total_premises,
                        }
                    ))
                except Exception as e:
                    derivation_lines.append(f"- {fact_id}: {fact_name} = UNRESOLVED ({e})")
                    self._emit_event(StepEvent(
                        event_type="premise_resolved",
                        step_number=idx + 1,
                        data={
                            "fact_name": f"{fact_id}: {fact_name}",
                            "value": None,
                            "source": "unresolved",
                            "error": str(e),
                            "step": idx + 1,
                            "total": total_premises,
                        }
                    ))

            # Step 3: Synthesize answer from resolved premises and inferences
            self._emit_event(StepEvent(
                event_type="synthesizing",
                step_number=0,
                data={"message": "Synthesizing answer from resolved facts..."}
            ))

            # Build synthesis context
            resolved_context = "\n".join([
                f"- {pid}: {p.value}" for pid, p in resolved_premises.items() if p and p.value
            ])
            inference_context = "\n".join([
                f"- {inf['id']}: {inf['operation']}" for inf in inferences
            ])

            synthesis_prompt = f"""Based on the resolved premises and inference plan, provide the answer.

Question: {claim}

Resolved Premises:
{resolved_context if resolved_context else "(no premises resolved)"}

Inference Steps:
{inference_context}

Conclusion to derive: {conclusion}

Provide a clear answer based on the available data. If premises are unresolved, explain what data is missing."""

            synthesis_result = self.router.execute(
                task_type=TaskType.SYNTHESIS,
                system="You synthesize answers from resolved facts with full provenance.",
                user_message=synthesis_prompt,
                max_tokens=1500,
            )

            answer = synthesis_result.content
            confidence = sum(p.confidence for p in resolved_premises.values() if p) / max(len(resolved_premises), 1)
            derivation_trace = "\n".join(derivation_lines)

            verify_result = {
                "answer": answer,
                "confidence": confidence,
                "derivation": derivation_trace,
                "sources": [{"type": p["source"], "description": p["description"]} for p in premises],
            }

            duration_ms = int((time.time() - start_time) * 1000)

            # Format output
            answer = verify_result.get("answer", "")
            confidence = verify_result.get("confidence", 0.0)
            derivation_trace = verify_result.get("derivation", "")
            sources = verify_result.get("sources", [])

            # Build final output
            output_parts = [
                f"**Verification Result** (confidence: {confidence:.0%})",
                "",
                answer,
            ]

            if derivation_trace:
                output_parts.extend([
                    "",
                    derivation_trace,
                ])

            final_output = "\n".join(output_parts)

            self._emit_event(StepEvent(
                event_type="verification_complete",
                step_number=0,
                data={
                    "answer": answer,
                    "confidence": confidence,
                    "has_derivation": bool(derivation_trace),
                }
            ))

            # Record in history
            self.history.record_query(
                session_id=self.session_id,
                question=problem,
                success=True,
                attempts=1,
                duration_ms=duration_ms,
                answer=final_output,
            )

            return {
                "success": True,
                "mode": "auditable",
                "output": final_output,
                "final_answer": answer,
                "confidence": confidence,
                "derivation": derivation_trace,
                "sources": sources,
                "suggestions": [
                    "Show me the supporting data for this verification",
                    "What assumptions were made in this analysis?",
                ],
                "datastore_tables": self.datastore.list_tables() if self.datastore else [],
            }

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            self._emit_event(StepEvent(
                event_type="verification_error",
                step_number=0,
                data={"error": str(e)}
            ))

            return {
                "success": False,
                "mode": "auditable",
                "error": str(e),
                "output": f"Verification failed: {e}",
            }

    def _follow_up_auditable(self, question: str, mode_selection) -> dict:
        """
        Handle a follow-up question in auditable mode using the fact resolver.

        This is called when the follow-up question suggests verification/validation
        (e.g., "verify", "validate", "prove", "check").

        Args:
            question: The verification question
            mode_selection: The mode selection result with reasoning

        Returns:
            Dict with verification result and derivation trace
        """
        import time
        start_time = time.time()

        # Emit event indicating mode switch
        self._emit_event(StepEvent(
            event_type="mode_switch",
            step_number=0,
            data={
                "mode": "auditable",
                "reasoning": mode_selection.reasoning,
                "matched_keywords": mode_selection.matched_keywords,
            }
        ))

        # Get context from previous work for the fact resolver
        existing_tables = self.datastore.list_tables() if self.datastore else []
        scratchpad_context = self.datastore.get_scratchpad_as_markdown() if self.datastore else ""

        # Prepare context for fact resolver
        context = f"""Previous analysis results:

{scratchpad_context}

Available tables: {', '.join(t['name'] for t in existing_tables) if existing_tables else '(none)'}

Verification request: {question}
"""

        # Use fact resolver to derive the answer with provenance
        self._emit_event(StepEvent(
            event_type="deriving",
            step_number=0,
            data={"message": f"Deriving answer: {question}"}
        ))

        try:
            # Resolve the question as a fact with full derivation
            result = self.fact_resolver.resolve_question(context)

            duration_ms = int((time.time() - start_time) * 1000)

            # Format the derivation trace
            derivation_trace = result.get("derivation", "")
            answer = result.get("answer", "")
            confidence = result.get("confidence", 0.0)
            sources = result.get("sources", [])

            # Build verification output
            output_parts = [
                f"**Verification Result** (confidence: {confidence:.0%})",
                "",
                answer,
            ]

            if derivation_trace:
                output_parts.extend([
                    "",
                    "**Derivation Trace:**",
                    derivation_trace,
                ])

            if sources:
                output_parts.extend([
                    "",
                    "**Sources:**",
                ])
                for src in sources:
                    output_parts.append(f"- {src.get('type', 'unknown')}: {src.get('description', '')}")

            final_output = "\n".join(output_parts)

            self._emit_event(StepEvent(
                event_type="verification_complete",
                step_number=0,
                data={
                    "answer": answer,
                    "confidence": confidence,
                    "has_derivation": bool(derivation_trace),
                }
            ))

            # Record in history
            self.history.record_query(
                session_id=self.session_id,
                question=question,
                success=True,
                attempts=1,
                duration_ms=duration_ms,
                answer=final_output,
            )

            return {
                "success": True,
                "mode": "auditable",
                "output": final_output,
                "final_answer": answer,
                "confidence": confidence,
                "derivation": derivation_trace,
                "sources": sources,
                "suggestions": [
                    "Show me the supporting data for this verification",
                    "What assumptions were made in this analysis?",
                ],
                "datastore_tables": existing_tables,
            }

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            self._emit_event(StepEvent(
                event_type="verification_error",
                step_number=0,
                data={"error": str(e)}
            ))

            return {
                "success": False,
                "mode": "auditable",
                "error": str(e),
                "output": f"Verification failed: {e}",
            }

    def _solve_knowledge(self, problem: str, mode_selection: ModeSelection) -> dict:
        """
        Solve a problem in knowledge mode using document lookup + LLM synthesis.

        This mode is for explanation/knowledge requests that don't need data analysis.
        It searches configured documents and synthesizes an explanation.

        Args:
            problem: The question/request to answer
            mode_selection: The mode selection result with reasoning

        Returns:
            Dict with synthesized explanation and sources
        """
        start_time = time.time()

        # Emit mode selection event
        self._emit_event(StepEvent(
            event_type="mode_switch",
            step_number=0,
            data={
                "mode": "knowledge",
                "reasoning": mode_selection.reasoning,
                "matched_keywords": mode_selection.matched_keywords,
            }
        ))

        # Step 1: Search documents for relevant content
        self._emit_event(StepEvent(
            event_type="searching_documents",
            step_number=0,
            data={"message": "Searching reference documents..."}
        ))

        sources = []
        doc_context = ""

        if self.doc_tools and self.config.documents:
            # Search for relevant document excerpts
            search_results = self.doc_tools.search_documents(problem, limit=5)

            if search_results:
                doc_lines = ["Relevant document excerpts:"]
                for i, result in enumerate(search_results, 1):
                    doc_name = result.get("document", "unknown")
                    excerpt = result.get("excerpt", "")
                    relevance = result.get("relevance", 0)
                    section = result.get("section", "")

                    source_info = {
                        "document": doc_name,
                        "section": section,
                        "relevance": relevance,
                    }
                    sources.append(source_info)

                    doc_lines.append(f"\n[{i}] From '{doc_name}'" + (f" - {section}" if section else ""))
                    doc_lines.append(excerpt)

                doc_context = "\n".join(doc_lines)

        # Step 2: Build prompt for LLM synthesis
        self._emit_event(StepEvent(
            event_type="synthesizing",
            step_number=0,
            data={"message": "Synthesizing explanation..."}
        ))

        # Get the knowledge mode system prompt
        from constat.execution.mode import get_mode_system_prompt
        system_prompt = get_mode_system_prompt(ExecutionMode.KNOWLEDGE)

        # Add context about the configuration
        if self.config.system_prompt:
            system_prompt = f"{system_prompt}\n\n{self.config.system_prompt}"

        # Build user message with document context
        if doc_context:
            user_message = f"""Question: {problem}

{doc_context}

Please provide a clear, accurate explanation based on the documents above and your general knowledge.
Cite specific documents when referencing them."""
        else:
            user_message = f"""Question: {problem}

No reference documents are configured. Please provide an explanation based on your general knowledge.
If you don't have enough information, say so rather than guessing."""

        # Step 3: Generate response
        try:
            result = self.router.execute(
                task_type=TaskType.SYNTHESIS,
                system=system_prompt,
                user_message=user_message,
                max_tokens=2000,
            )

            answer = result.content
            duration_ms = int((time.time() - start_time) * 1000)

            self._emit_event(StepEvent(
                event_type="knowledge_complete",
                step_number=0,
                data={
                    "has_documents": bool(sources),
                    "source_count": len(sources),
                }
            ))

            # Build final output
            output_parts = [answer]

            if sources:
                output_parts.extend([
                    "",
                    "**Sources consulted:**",
                ])
                for src in sources:
                    src_line = f"- {src['document']}"
                    if src.get('section'):
                        src_line += f" ({src['section']})"
                    output_parts.append(src_line)

            final_output = "\n".join(output_parts)

            # Record in history
            self.history.record_query(
                session_id=self.session_id,
                question=problem,
                success=True,
                attempts=1,
                duration_ms=duration_ms,
                answer=final_output,
            )

            return {
                "success": True,
                "mode": "knowledge",
                "output": final_output,
                "sources": sources,
                "plan": None,  # No plan in knowledge mode
                "suggestions": [
                    "Tell me more about a specific aspect",
                    "What data is available to analyze?",
                ],
            }

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            self._emit_event(StepEvent(
                event_type="knowledge_error",
                step_number=0,
                data={"error": str(e)}
            ))

            return {
                "success": False,
                "mode": "knowledge",
                "error": str(e),
                "output": f"Failed to generate explanation: {e}",
            }

    def replay(self, problem: str) -> dict:
        """
        Replay a previous session by re-executing stored code without LLM codegen.

        This loads the stored code from the scratchpad and re-executes it,
        then synthesizes a new answer (which still uses the LLM).

        Useful for demos, debugging, or re-running with modified data.

        Args:
            problem: The original problem (used for answer synthesis)

        Returns:
            Dict with results (same format as solve())
        """
        if not self.datastore:
            raise ValueError("No datastore available for replay")

        # Load stored scratchpad entries
        entries = self.datastore.get_scratchpad()
        if not entries:
            raise ValueError("No stored steps to replay")

        # Emit planning complete (we're using stored plan)
        self._emit_event(StepEvent(
            event_type="plan_ready",
            step_number=0,
            data={
                "steps": [
                    {"number": e["step_number"], "goal": e["goal"], "depends_on": []}
                    for e in entries
                ],
                "reasoning": "Replaying stored execution",
                "is_followup": False,
            }
        ))

        all_results = []
        for entry in entries:
            step_number = entry["step_number"]
            goal = entry["goal"]
            code = entry["code"]

            if not code:
                raise ValueError(f"Step {step_number} has no stored code to replay")

            self._emit_event(StepEvent(
                event_type="step_start",
                step_number=step_number,
                data={"goal": goal}
            ))

            self._emit_event(StepEvent(
                event_type="executing",
                step_number=step_number,
                data={"attempt": 1, "code": code}
            ))

            start_time = time.time()

            # Track tables before execution
            tables_before = set(t['name'] for t in self.datastore.list_tables())

            # Execute stored code
            exec_globals = self._get_execution_globals()
            result = self.executor.execute(code, exec_globals)

            # Auto-save any DataFrames
            if result.success:
                self._auto_save_results(result.namespace, step_number)

            duration_ms = int((time.time() - start_time) * 1000)
            tables_after = set(t['name'] for t in self.datastore.list_tables())
            tables_created = list(tables_after - tables_before)

            if result.success:
                self._emit_event(StepEvent(
                    event_type="step_complete",
                    step_number=step_number,
                    data={"stdout": result.stdout, "attempts": 1, "duration_ms": duration_ms, "tables_created": tables_created}
                ))

                all_results.append(StepResult(
                    success=True,
                    stdout=result.stdout,
                    attempts=1,
                    duration_ms=duration_ms,
                    tables_created=tables_created,
                    code=code,
                ))
            else:
                self._emit_event(StepEvent(
                    event_type="step_error",
                    step_number=step_number,
                    data={"error": result.stderr or "Execution failed", "attempt": 1}
                ))
                return {
                    "success": False,
                    "error": result.stderr or "Replay execution failed",
                    "step_number": step_number,
                }

        # Synthesize final answer (respects insights config)
        combined_output = "\n\n".join([
            f"Step {entry['step_number']}: {entry['goal']}\n{r.stdout}"
            for entry, r in zip(entries, all_results)
        ])

        # Emit raw results first
        self._emit_event(StepEvent(
            event_type="raw_results_ready",
            step_number=0,
            data={"output": combined_output}
        ))

        # Check if insights are enabled
        skip_insights = not self.session_config.enable_insights

        if skip_insights:
            final_answer = combined_output
            self._emit_event(StepEvent(
                event_type="answer_ready",
                step_number=0,
                data={"answer": final_answer, "brief": True}
            ))
        else:
            self._emit_event(StepEvent(
                event_type="synthesizing",
                step_number=0,
                data={"message": "Synthesizing final answer..."}
            ))

            final_answer = self._synthesize_answer(problem, combined_output)

            self._emit_event(StepEvent(
                event_type="answer_ready",
                step_number=0,
                data={"answer": final_answer}
            ))

        total_duration = sum(r.duration_ms for r in all_results)

        return {
            "success": True,
            "results": all_results,
            "output": combined_output,
            "final_answer": final_answer,
            "datastore_tables": self.datastore.list_tables(),
            "duration_ms": total_duration,
            "replay": True,
        }

    def get_state(self) -> dict:
        """Get current session state for inspection or resumption."""
        return {
            "session_id": self.session_id,
            "plan": self.plan,
            "scratchpad": self.scratchpad.to_markdown(),
            "state": self.datastore.get_all_state() if self.datastore else {},
            "completed_steps": self.plan.completed_steps if self.plan else [],
            "datastore_tables": self.datastore.list_tables() if self.datastore else [],
        }

    # --- Session Data Sources ---

    def add_database(
        self,
        name: str,
        db_type: str,
        uri: str,
        description: str = "",
    ) -> bool:
        """Add a database to the current session.

        The database will be available as `db_<name>` in code execution.

        Args:
            name: Database name (used as db_<name> variable)
            db_type: Database type (sql, csv, json, parquet, mongodb, etc.)
            uri: Connection URI or file path
            description: Human-readable description

        Returns:
            True if added successfully
        """
        self.session_databases[name] = {
            "type": db_type,
            "uri": uri,
            "description": description,
        }
        return True

    def add_file(
        self,
        name: str,
        uri: str,
        auth: str = "",
        description: str = "",
    ) -> bool:
        """Add a file to the current session.

        The file will be available as `file_<name>` in code execution.
        For local files, this is a Path. For HTTP files, content is fetched on-demand.

        Args:
            name: File name (used as file_<name> variable)
            uri: File URI (file:// or http://)
            auth: Auth header for HTTP (e.g., "Bearer token123")
            description: Human-readable description

        Returns:
            True if added successfully
        """
        self.session_files[name] = {
            "uri": uri,
            "auth": auth,
            "description": description,
        }
        return True

    def get_all_databases(self) -> dict[str, dict]:
        """Get all databases (config + session-added).

        Returns:
            Dict of name -> {type, uri, description, source}
        """
        from constat.storage.bookmarks import BookmarkStore

        result = {}

        # Config databases
        for name, db_config in self.config.databases.items():
            result[name] = {
                "type": db_config.type or "sql",
                "uri": db_config.uri or db_config.path or "",
                "description": db_config.description or "",
                "source": "config",
            }

        # Bookmarked databases
        bookmarks = BookmarkStore()
        for name, bm in bookmarks.list_databases().items():
            if name not in result:  # Don't override config
                result[name] = {
                    "type": bm["type"],
                    "uri": bm["uri"],
                    "description": bm["description"],
                    "source": "bookmark",
                }

        # Session databases
        for name, db in self.session_databases.items():
            result[name] = {
                "type": db["type"],
                "uri": db["uri"],
                "description": db["description"],
                "source": "session",
            }

        return result

    def get_all_files(self) -> dict[str, dict]:
        """Get all files (config documents + file sources + bookmarks + session).

        Returns:
            Dict of name -> {uri, description, auth, source, file_type}
        """
        from constat.storage.bookmarks import BookmarkStore

        result = {}

        # Config documents
        if self.config.documents:
            for name, doc_config in self.config.documents.items():
                uri = ""
                if doc_config.path:
                    uri = f"file://{doc_config.path}"
                elif doc_config.url:
                    uri = doc_config.url
                result[name] = {
                    "uri": uri,
                    "description": doc_config.description or "",
                    "auth": "",
                    "source": "config",
                    "file_type": "document",
                }

        # Config file-type databases (csv, json, parquet)
        for name, db_config in self.config.databases.items():
            if db_config.type in ("csv", "json", "jsonl", "parquet", "arrow", "feather"):
                path = db_config.path or db_config.uri or ""
                result[name] = {
                    "uri": f"file://{path}" if not path.startswith(("file://", "http")) else path,
                    "description": db_config.description or "",
                    "auth": "",
                    "source": "config",
                    "file_type": db_config.type,
                }

        # Bookmarked files
        bookmarks = BookmarkStore()
        for name, bm in bookmarks.list_files().items():
            if name not in result:  # Don't override config
                result[name] = {
                    "uri": bm["uri"],
                    "description": bm["description"],
                    "auth": bm.get("auth", ""),
                    "source": "bookmark",
                    "file_type": "file",
                }

        # Session files
        for name, f in self.session_files.items():
            result[name] = {
                "uri": f["uri"],
                "description": f["description"],
                "auth": f.get("auth", ""),
                "source": "session",
                "file_type": "file",
            }

        return result

    # --- Context Management ---

    def get_context_stats(self) -> Optional[ContextStats]:
        """
        Get statistics about context size.

        Returns:
            ContextStats with token estimates and breakdown, or None if no datastore
        """
        if not self.datastore:
            return None

        estimator = ContextEstimator(self.datastore)
        return estimator.estimate()

    def compact_context(
        self,
        summarize_scratchpad: bool = True,
        sample_tables: bool = True,
        clear_old_state: bool = False,
        keep_recent_steps: int = 3,
    ) -> Optional[CompactionResult]:
        """
        Compact session context to reduce token usage.

        This is useful for long-running sessions where context grows too large.

        Args:
            summarize_scratchpad: Truncate old scratchpad narratives
            sample_tables: Sample large tables down to max rows
            clear_old_state: Clear state variables from old steps
            keep_recent_steps: Number of recent steps to preserve intact

        Returns:
            CompactionResult with details, or None if no datastore
        """
        if not self.datastore:
            return None

        compactor = ContextCompactor(self.datastore)
        return compactor.compact(
            summarize_scratchpad=summarize_scratchpad,
            sample_tables=sample_tables,
            clear_old_state=clear_old_state,
            keep_recent_steps=keep_recent_steps,
        )

    def _auto_compact_if_needed(self) -> Optional[CompactionResult]:
        """
        Automatically compact context if it exceeds critical threshold.

        This is called after step execution to prevent context from growing
        too large for the LLM context window.

        Returns:
            CompactionResult if compaction was performed, None otherwise
        """
        if not self.datastore:
            return None

        stats = self.get_context_stats()
        if not stats or not stats.is_critical:
            return None

        # Context is critical - auto-compact
        self._emit_event(StepEvent(
            event_type="progress",
            step_number=0,
            data={"message": f"Auto-compacting context ({stats.total_tokens:,} tokens)..."}
        ))

        result = self.compact_context(
            summarize_scratchpad=True,
            sample_tables=True,
            clear_old_state=False,  # Conservative - don't clear state
            keep_recent_steps=5,    # Keep more steps for auto-compact
        )

        if result:
            self._emit_event(StepEvent(
                event_type="progress",
                step_number=0,
                data={"message": f"Context compacted: {result.tokens_before:,} → {result.tokens_after:,} tokens"}
            ))

        return result

    def reset_context(self) -> Optional[CompactionResult]:
        """
        Fully reset session context (clear all state).

        WARNING: This clears all scratchpad entries, tables, state variables,
        and artifacts. Use with caution.

        Returns:
            CompactionResult with details, or None if no datastore
        """
        if not self.datastore:
            return None

        compactor = ContextCompactor(self.datastore)
        result = compactor.clear_all()

        # Also reset in-memory scratchpad
        self.scratchpad = Scratchpad()
        self.plan = None

        return result

    # --- Saved Plans ---

    CONSTAT_BASE_DIR = Path(".constat")
    DEFAULT_USER_ID = "default"

    @classmethod
    def _get_user_plans_file(cls, user_id: str) -> Path:
        """Get path to user-scoped saved plans file."""
        return cls.CONSTAT_BASE_DIR / user_id / "saved_plans.json"

    @classmethod
    def _get_shared_plans_file(cls) -> Path:
        """Get path to shared plans file."""
        return cls.CONSTAT_BASE_DIR / "shared" / "saved_plans.json"

    def save_plan(self, name: str, problem: str, user_id: Optional[str] = None, shared: bool = False) -> None:
        """
        Save the current session's plan and code for future replay.

        Args:
            name: Name for the saved plan
            problem: The original problem (for replay context)
            user_id: User ID (defaults to DEFAULT_USER_ID)
            shared: If True, save as shared plan accessible to all users
        """
        if not self.datastore:
            raise ValueError("No datastore available")

        entries = self.datastore.get_scratchpad()
        if not entries:
            raise ValueError("No steps to save")

        user_id = user_id or self.DEFAULT_USER_ID

        plan_data = {
            "problem": problem,
            "created_by": user_id,
            "steps": [
                {
                    "step_number": e["step_number"],
                    "goal": e["goal"],
                    "code": e["code"],
                }
                for e in entries
            ],
        }

        if shared:
            plans = self._load_shared_plans()
            plans[name] = plan_data
            self._save_shared_plans(plans)
        else:
            plans = self._load_user_plans(user_id)
            plans[name] = plan_data
            self._save_user_plans(user_id, plans)

    @classmethod
    def load_saved_plan(cls, name: str, user_id: Optional[str] = None) -> dict:
        """
        Load a saved plan by name.

        Searches user's plans first, then shared plans.

        Args:
            name: Name of the saved plan
            user_id: User ID (defaults to DEFAULT_USER_ID)

        Returns:
            Dict with problem and steps
        """
        user_id = user_id or cls.DEFAULT_USER_ID

        # Check user's plans first
        user_plans = cls._load_user_plans(user_id)
        if name in user_plans:
            return user_plans[name]

        # Check shared plans
        shared_plans = cls._load_shared_plans()
        if name in shared_plans:
            return shared_plans[name]

        raise ValueError(f"No saved plan named '{name}'")

    @classmethod
    def list_saved_plans(cls, user_id: Optional[str] = None, include_shared: bool = True) -> list[dict]:
        """
        List saved plans accessible to the user.

        Args:
            user_id: User ID (defaults to DEFAULT_USER_ID)
            include_shared: Include shared plans in the list

        Returns:
            List of dicts with name, problem, shared flag
        """
        user_id = user_id or cls.DEFAULT_USER_ID
        result = []

        # User's plans
        user_plans = cls._load_user_plans(user_id)
        for name, data in user_plans.items():
            result.append({
                "name": name,
                "problem": data.get("problem", ""),
                "shared": False,
                "steps": len(data.get("steps", [])),
            })

        # Shared plans
        if include_shared:
            shared_plans = cls._load_shared_plans()
            for name, data in shared_plans.items():
                result.append({
                    "name": name,
                    "problem": data.get("problem", ""),
                    "shared": True,
                    "created_by": data.get("created_by", "unknown"),
                    "steps": len(data.get("steps", [])),
                })

        return result

    @classmethod
    def delete_saved_plan(cls, name: str, user_id: Optional[str] = None) -> bool:
        """Delete a saved plan by name (only user's own plans)."""
        user_id = user_id or cls.DEFAULT_USER_ID
        user_plans = cls._load_user_plans(user_id)

        if name not in user_plans:
            return False

        del user_plans[name]
        cls._save_user_plans(user_id, user_plans)
        return True

    @classmethod
    def share_plan_with(cls, name: str, target_user: str, from_user: Optional[str] = None) -> bool:
        """
        Share a plan with a specific user (copy to their plans).

        Args:
            name: Name of the plan to share
            target_user: User ID to share with
            from_user: Source user ID (defaults to DEFAULT_USER_ID)

        Returns:
            True if shared successfully
        """
        from_user = from_user or cls.DEFAULT_USER_ID

        # Find the plan (check user's plans first, then shared)
        source_plans = cls._load_user_plans(from_user)
        if name in source_plans:
            plan_data = source_plans[name].copy()
        else:
            shared_plans = cls._load_shared_plans()
            if name in shared_plans:
                plan_data = shared_plans[name].copy()
            else:
                return False

        # Copy to target user's plans
        target_plans = cls._load_user_plans(target_user)
        plan_data["shared_by"] = from_user
        target_plans[name] = plan_data
        cls._save_user_plans(target_user, target_plans)
        return True

    @classmethod
    def _load_user_plans(cls, user_id: str) -> dict:
        """Load saved plans for a specific user."""
        plans_file = cls._get_user_plans_file(user_id)
        if not plans_file.exists():
            return {}
        try:
            return json.loads(plans_file.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    @classmethod
    def _save_user_plans(cls, user_id: str, plans: dict) -> None:
        """Save plans to user-scoped file."""
        plans_file = cls._get_user_plans_file(user_id)
        plans_file.parent.mkdir(parents=True, exist_ok=True)
        plans_file.write_text(json.dumps(plans, indent=2))

    @classmethod
    def _load_shared_plans(cls) -> dict:
        """Load shared plans accessible to all users."""
        plans_file = cls._get_shared_plans_file()
        if not plans_file.exists():
            return {}
        try:
            return json.loads(plans_file.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    @classmethod
    def _save_shared_plans(cls, plans: dict) -> None:
        """Save shared plans."""
        plans_file = cls._get_shared_plans_file()
        plans_file.parent.mkdir(parents=True, exist_ok=True)
        plans_file.write_text(json.dumps(plans, indent=2))

    def replay_saved(self, name: str, user_id: Optional[str] = None) -> dict:
        """
        Replay a saved plan by name.

        Args:
            name: Name of the saved plan
            user_id: User ID for plan lookup (defaults to DEFAULT_USER_ID)

        Returns:
            Dict with results (same format as solve())
        """
        plan_data = self.load_saved_plan(name, user_id=user_id)

        if not self.datastore:
            raise ValueError("No datastore available for replay")

        # Clear existing scratchpad and load saved steps
        # (We'll execute fresh but use stored code)
        problem = plan_data["problem"]
        steps = plan_data["steps"]

        # Emit plan ready
        self._emit_event(StepEvent(
            event_type="plan_ready",
            step_number=0,
            data={
                "steps": [
                    {"number": s["step_number"], "goal": s["goal"], "depends_on": []}
                    for s in steps
                ],
                "reasoning": f"Replaying saved plan: {name}",
                "is_followup": False,
            }
        ))

        all_results = []
        for step_data in steps:
            step_number = step_data["step_number"]
            goal = step_data["goal"]
            code = step_data["code"]

            if not code:
                raise ValueError(f"Step {step_number} has no stored code")

            self._emit_event(StepEvent(
                event_type="step_start",
                step_number=step_number,
                data={"goal": goal}
            ))

            self._emit_event(StepEvent(
                event_type="executing",
                step_number=step_number,
                data={"attempt": 1, "code": code}
            ))

            start_time = time.time()
            tables_before = set(t['name'] for t in self.datastore.list_tables())

            exec_globals = self._get_execution_globals()
            result = self.executor.execute(code, exec_globals)

            if result.success:
                self._auto_save_results(result.namespace, step_number)

            duration_ms = int((time.time() - start_time) * 1000)
            tables_after = set(t['name'] for t in self.datastore.list_tables())
            tables_created = list(tables_after - tables_before)

            if result.success:
                self._emit_event(StepEvent(
                    event_type="step_complete",
                    step_number=step_number,
                    data={"stdout": result.stdout, "attempts": 1, "duration_ms": duration_ms, "tables_created": tables_created}
                ))

                all_results.append(StepResult(
                    success=True,
                    stdout=result.stdout,
                    attempts=1,
                    duration_ms=duration_ms,
                    tables_created=tables_created,
                    code=code,
                ))
            else:
                self._emit_event(StepEvent(
                    event_type="step_error",
                    step_number=step_number,
                    data={"error": result.stderr or "Execution failed", "attempt": 1}
                ))
                return {
                    "success": False,
                    "error": result.stderr or "Replay execution failed",
                    "step_number": step_number,
                }

        # Synthesize answer (respects insights config)
        combined_output = "\n\n".join([
            f"Step {s['step_number']}: {s['goal']}\n{r.stdout}"
            for s, r in zip(steps, all_results)
        ])

        # Emit raw results first
        self._emit_event(StepEvent(
            event_type="raw_results_ready",
            step_number=0,
            data={"output": combined_output}
        ))

        # Check if insights are enabled
        skip_insights = not self.session_config.enable_insights

        if skip_insights:
            final_answer = combined_output
            self._emit_event(StepEvent(
                event_type="answer_ready",
                step_number=0,
                data={"answer": final_answer, "brief": True}
            ))
        else:
            self._emit_event(StepEvent(
                event_type="synthesizing",
                step_number=0,
                data={"message": "Synthesizing final answer..."}
            ))

            final_answer = self._synthesize_answer(problem, combined_output)

            self._emit_event(StepEvent(
                event_type="answer_ready",
                step_number=0,
                data={"answer": final_answer}
            ))

        total_duration = sum(r.duration_ms for r in all_results)

        return {
            "success": True,
            "results": all_results,
            "output": combined_output,
            "final_answer": final_answer,
            "datastore_tables": self.datastore.list_tables(),
            "duration_ms": total_duration,
            "replay": True,
            "plan_name": name,
        }

    def get_unresolved_facts(self) -> list[dict]:
        """Get list of facts that could not be resolved."""
        return [f.to_dict() for f in self.fact_resolver.get_unresolved_facts()]

    def get_unresolved_summary(self) -> str:
        """Get human-readable summary of unresolved facts."""
        return self.fact_resolver.get_unresolved_summary()

    def provide_facts(self, user_text: str) -> dict:
        """
        Extract facts from user text and add to resolver cache.

        This is used in auditable mode when facts could not be resolved.
        The user provides facts in natural language, and the LLM extracts
        them into structured facts that can be used for re-resolution.

        Example:
            session.provide_facts("There were 1 million people at the march")
            # Extracts: march_attendance = 1000000

        Args:
            user_text: Natural language text containing facts

        Returns:
            Dict with:
                - extracted_facts: List of facts extracted and added
                - unresolved_remaining: List of still-unresolved facts
        """
        # Extract facts from user text
        extracted = self.fact_resolver.add_user_facts_from_text(user_text)

        # Clear unresolved facts to allow re-resolution
        self.fact_resolver.clear_unresolved()

        return {
            "extracted_facts": [f.to_dict() for f in extracted],
            "unresolved_remaining": [f.to_dict() for f in self.fact_resolver.get_unresolved_facts()],
        }

    def add_fact(self, fact_name: str, value, reasoning: str = None, **params) -> dict:
        """
        Explicitly add a fact to the resolver cache.

        This is a more direct way to provide facts than provide_facts(),
        useful when you know the exact fact name and value.

        Args:
            fact_name: Name of the fact (e.g., "march_attendance")
            value: The value to set
            reasoning: Optional explanation
            **params: Additional parameters for the fact

        Returns:
            Dict with the created fact
        """
        fact = self.fact_resolver.add_user_fact(
            fact_name=fact_name,
            value=value,
            reasoning=reasoning,
            **params,
        )
        return fact.to_dict()


def create_session(config_path: str) -> Session:
    """Create a session from a config file path."""
    config = Config.from_yaml(config_path)
    return Session(config)
