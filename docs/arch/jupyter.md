# Jupyter Notebook Interface

**Status:** Design
**Approach:** HTTP client SDK (not local lib)

## Decision: HTTP over Local Lib

The notebook client talks to a running constat server. Rationale:

- Server already exposes 30+ REST endpoints + WebSocket event streaming
- No need to configure LLM keys, DB connections, or catalog in notebook env
- Sessions are shared with web UI (start in notebook, view in browser)
- Execution runs in server process; kernel crash doesn't kill queries
- Multi-user/RBAC already built into server layer
- Only gap: no Arrow/Parquet download endpoint (add one)

## Package Location

`constat/notebook/` within the main package. Shares types with server.

```
constat/notebook/
    __init__.py          # Public API: ConstatClient, configure
    client.py            # HTTP + WebSocket client
    display.py           # IPython rich display formatters
    widgets.py           # ipywidgets for plan approval, clarification
    progress.py          # Execution progress rendering
    config.py            # Connection config (server URL, auth token)
```

## Public API Surface

```python
from constat.notebook import ConstatClient

# Connect to server
client = ConstatClient("http://localhost:8000", token="...")

# Session management
session = client.create_session()
session = client.get_session("session-id")
sessions = client.list_sessions()

# Core query (blocks, shows live progress, returns result)
result = session.solve("What are the top 10 items by value?")
result = session.follow_up("Break that down by region")

# Auditable mode — reasoning chain with full provenance
result = session.prove("Calculate raises based on performance guidelines")
result.proof_tree          # ProofTree DAG
result.proof_tree.display()  # Visual DAG rendering in notebook

# Result access
result.answer           # str - synthesized answer
result.tables           # dict[str, polars.DataFrame]
result.artifacts        # list[Artifact] with .display() method
result.suggestions      # list[str]
result.proof_tree       # ProofTree (if auditable mode)

# Direct table access
df = session.table("results")                    # -> polars.DataFrame
df = session.table("results", pandas=True)       # -> pandas.DataFrame
session.tables()                                 # list all tables

# Artifacts
artifacts = session.artifacts()
artifacts[0].display()     # Renders inline (chart, HTML, markdown)
artifacts[0].download("output.csv")

# Schema browsing
client.databases()
client.tables("my_database")
client.table_schema("my_database", "events")

# Domain management
client.domains()                              # list configured domains
client.domain_questions("hr-reporting")       # golden questions for domain

# Glossary
session.glossary()                            # list glossary terms
session.glossary_term("employees")            # term details + relationships

# Facts
session.facts()
session.remember("fiscal_year_start", "April 1")
session.forget("fiscal_year_start")

# Skills
client.skills()                               # list available skills
client.skill_info("data-quality")             # skill details

# Regression testing
test_result = session.run_tests(domains=["hr-reporting"], include_e2e=True)
test_result.summary()                          # pass/fail counts
test_result.failures()                         # detailed failure info
```

## Execution Flow

### solve() / follow_up()

```
1. POST /api/sessions/{id}/query
2. Connect WebSocket /api/sessions/{id}/ws
3. Stream events -> render progress (see Progress section)
4. If plan_ready event -> show PlanApprovalWidget (see Widgets section)
5. If clarification_needed -> show ClarificationWidget
6. On query_complete -> fetch tables, build SolveResult, return
7. On query_error -> raise ConstatError with details
```

### prove()

```
1. POST /api/sessions/{id}/prove
2. Connect WebSocket /api/sessions/{id}/ws
3. Stream events -> render DAG progress (premise resolution, inference execution)
   - proof_start -> show DAG skeleton
   - fact_start/fact_resolved/fact_failed -> update premise nodes
   - inference_executing/inference_complete -> update inference nodes
   - proof_complete -> finalize
4. On proof_summary_ready -> build ProveResult with proof tree
5. On query_error -> raise ConstatError
```

### Key: nest_asyncio

Jupyter runs its own event loop. Use `nest_asyncio` to allow the sync
`session.solve()` call to run async WebSocket consumption internally.

```python
import nest_asyncio
nest_asyncio.apply()
```

The client's public API is synchronous (blocking). Internally it runs
an asyncio event loop for WebSocket streaming.

## Progress Rendering

Use `IPython.display` for live-updating output during execution.

### Exploratory mode (solve)

```
Planning...
Plan: "Analyze items by category" (3 steps)
  [Approve] [Reject] [Edit]          <- ipywidgets buttons

Step 1/3: Query events table ████████░░ generating...
Step 2/3: Aggregate by category ████████████ complete
Step 3/3: Join with metadata ░░░░░░░░░░ pending

Synthesizing answer...
```

### Auditable mode (prove)

```
Reasoning Chain: "Calculate raises based on guidelines"

Premises:
  P1: employees ████████████ resolved (database:hr)
  P2: performance_reviews ████████████ resolved (database:hr)
  P3: raise_guidelines ████████░░ resolving... (document:business_rules)

Inferences:
  I1: employee_reviews ░░░░░░░░░░ pending
  I2: raise_recommendations ░░░░░░░░░░ pending

Synthesizing answer...
```

Implementation:
- `ipywidgets.Output` context for each step/node
- `ipywidgets.IntProgress` for step progress bars
- Status updates via WebSocket events (`step_start`, `step_generating`,
  `step_executing`, `step_complete`, `fact_start`, `fact_resolved`,
  `inference_executing`, `inference_complete`)
- Clear and redraw on each event (using `IPython.display.clear_output`)

### Fallback (no ipywidgets)

If ipywidgets is not installed, fall back to `print()`-based progress:

```
[planning] Generating plan...
[plan_ready] 3 steps (auto-approved)
[step 1/3] Query events table... done (1.2s)
[step 2/3] Aggregate by category... done (0.8s)
[step 3/3] Join with metadata... done (0.5s)
[synthesizing] Generating answer...
```

## Plan Approval Widget

When `require_approval=True` (default), show an interactive widget:

```python
class PlanApprovalWidget:
    """ipywidgets-based plan approval UI."""

    def __init__(self, plan_data: dict, callback: Callable):
        # Display plan steps as numbered list
        # Three buttons: Approve, Reject, Edit
        # Reject shows textarea for feedback
        # Edit shows editable step list
        # Calls callback with PlanApprovalResponse
```

If ipywidgets not available, use `input()` prompt:
```
Plan: "Analyze items by value"
  1. Query events table for all records
  2. Aggregate metric by group_id
  3. Join with metadata, sort descending

Approve? [Y/n/feedback]:
```

## SolveResult Object

```python
@dataclass
class SolveResult:
    success: bool
    answer: str                              # Synthesized answer
    tables: dict[str, polars.DataFrame]      # All tables created
    artifacts: list[Artifact]                # Charts, code, HTML, etc.
    suggestions: list[str]                   # Follow-up suggestions
    steps: list[StepInfo]                    # Execution details
    error: str | None
    raw_output: str | None

    def _repr_markdown_(self) -> str:
        """IPython rich display: render answer as markdown."""

    def display(self):
        """Full rich display: answer + tables + artifacts."""
```

```python
@dataclass
class ProveResult(SolveResult):
    proof_tree: ProofTree | None             # DAG with premises + inferences
    proof_nodes: list[ProofNode]             # Flat list of resolved nodes

    def display(self):
        """Rich display: answer + proof tree visualization + tables."""
```

`_repr_markdown_` gives automatic rich rendering when the result is the
last expression in a cell.

## Artifact Display

Artifacts use IPython's display system for inline rendering:

```python
class Artifact:
    id: int
    name: str
    artifact_type: str    # PLOTLY, HTML, MARKDOWN, TABLE, PNG, etc.
    content: str
    mime_type: str

    def display(self):
        """Render inline using IPython.display."""
        if self.artifact_type == "PLOTLY":
            # plotly.io.from_json -> fig.show()
        elif self.artifact_type in ("PNG", "JPEG"):
            display(Image(data=base64.b64decode(self.content)))
        elif self.artifact_type == "HTML":
            display(HTML(self.content))
        elif self.artifact_type == "MARKDOWN":
            display(Markdown(self.content))
        elif self.artifact_type == "TABLE":
            # Parse as DataFrame, display
            display(polars.read_csv(StringIO(self.content)))

    def _repr_html_(self) -> str:
        """Auto-display in notebook output."""
```

## DataFrame Transport

### Current: JSON via REST

`GET /api/sessions/{id}/tables/{name}` returns paginated JSON.
Fine for small tables, poor for large ones.

### New: Arrow IPC endpoint

Add a single new endpoint to the server:

```python
@router.get("/{session_id}/tables/{table_name}/arrow")
async def get_table_arrow(session_id: str, table_name: str):
    """Return table as Arrow IPC stream for efficient DataFrame transport."""
    # DuckDB -> Arrow -> IPC bytes
    arrow_table = datastore.conn.execute(
        f"SELECT * FROM {table_name}"
    ).fetch_arrow_table()

    sink = io.BytesIO()
    writer = pa.ipc.new_stream(sink, arrow_table.schema)
    writer.write_table(arrow_table)
    writer.close()

    return Response(
        content=sink.getvalue(),
        media_type="application/vnd.apache.arrow.stream",
    )
```

Client side:

```python
def table(self, name: str, pandas: bool = False):
    resp = self._get(f"/tables/{name}/arrow")
    reader = pa.ipc.open_stream(resp.content)
    arrow_table = reader.read_all()
    if pandas:
        return arrow_table.to_pandas()
    return polars.from_arrow(arrow_table)
```

This gives zero-copy DataFrame construction from DuckDB -> Arrow -> Polars.

## Dependencies

Required:
- `httpx` - HTTP client (async + sync)
- `websockets` - WebSocket client
- `nest_asyncio` - Allow sync calls in Jupyter event loop
- `polars` - Primary DataFrame library

Optional (enhanced experience):
- `ipywidgets` - Interactive plan approval, progress bars
- `pandas` - Alternative DataFrame output
- `plotly` - Inline chart rendering from Plotly artifacts
- `pyarrow` - Arrow IPC transport (falls back to JSON without it)

## Configuration

```python
# Option 1: Constructor
client = ConstatClient("http://localhost:8000", token="bearer-token")

# Option 2: Environment variables
#   CONSTAT_SERVER_URL=http://localhost:8000
#   CONSTAT_AUTH_TOKEN=bearer-token
client = ConstatClient()

# Option 3: Config file (~/.constat/notebook.yaml)
client = ConstatClient()  # Auto-discovers config
```

## Implementation Phases

### Phase 1: Core Client (MVP)

Files: `client.py`, `config.py`, `__init__.py`

- `ConstatClient` with session CRUD
- `Session.solve()` and `follow_up()` (blocking, print-based progress)
- `Session.table()` returning Polars DataFrame (JSON transport)
- `SolveResult` with `_repr_markdown_`
- Basic print-based progress (no ipywidgets dependency)
- Auto-approve plans (no widget yet)

Endpoints used:
- `POST /api/sessions` / `GET /api/sessions` / `DELETE /api/sessions/{id}`
- `POST /api/sessions/{id}/query`
- `WS /api/sessions/{id}/ws`
- `GET /api/sessions/{id}/tables` / `GET /api/sessions/{id}/tables/{name}`
- `POST /api/sessions/{id}/plan/approve`

### Phase 2: Rich Display + Widgets + Prove

Files: `display.py`, `widgets.py`, `progress.py`

- `PlanApprovalWidget` (ipywidgets)
- `ClarificationWidget` (ipywidgets)
- Step progress bars (ipywidgets)
- `Artifact.display()` with type-specific rendering
- `SolveResult.display()` full rich output
- `Session.prove()` with DAG progress rendering
- `ProveResult` with proof tree visualization
- Proof node progress (premise/inference status updates)

### Phase 3: Performance + Domain Features

Files: server-side `data.py` addition, client `client.py` update

- Arrow IPC endpoint on server
- Arrow-based DataFrame transport in client
- Schema browsing methods
- Domain management (`client.domains()`, `client.domain_questions()`)
- Glossary browsing (`session.glossary()`, `session.glossary_term()`)
- Skills listing (`client.skills()`)
- Facts/learnings management
- Session resume/replay
- `session.download_code()` for generated step code

### Phase 4: Regression Testing

Files: `testing.py` in client

- `session.run_tests()` with SSE streaming progress
- `TestResult` with summary/failures display
- Domain-scoped and tag-filtered test execution
- Expected output validation results
- Integration with golden questions CRUD

## Server Changes Required

1. **Arrow IPC endpoint** (Phase 3): `GET /{session_id}/tables/{table_name}/arrow`
   - Returns Arrow IPC stream bytes
   - ~10 lines in `constat/server/routes/data.py`

2. **No other server changes needed for Phase 1-2.** The existing REST + WebSocket
   API covers all required operations.

3. **Phase 3-4 use existing endpoints:**
   - `GET /{session_id}/glossary` — glossary terms
   - `GET /{session_id}/tests/domains` — testable domains
   - `POST /{session_id}/tests/run` — SSE test execution stream
   - `GET /{session_id}/tests/{domain}/questions` — golden questions

## WebSocket Events

The client must handle these event types for progress rendering:

### Exploratory mode
- `planning_start`, `plan_ready`, `plan_approved`
- `step_start`, `step_generating`, `step_executing`, `step_complete`, `step_error`
- `clarification_needed`, `clarification_received`
- `synthesizing`, `query_complete`, `query_error`

### Auditable mode (additional)
- `proof_start`, `proof_complete`, `proof_summary_ready`
- `fact_start`, `fact_resolved`, `fact_failed`, `fact_blocked`
- `dag_execution_start`
- `inference_code`, `inference_executing`, `inference_complete`

## Example Notebook

```python
from constat.notebook import ConstatClient

client = ConstatClient()
session = client.create_session()

# Ask a question - shows live progress, plan approval widget
result = session.solve("What are our top 10 items by total value?")

# Answer renders as markdown automatically
result
```

```python
# Access the data as a DataFrame
df = result.tables["top_items"]
df.head()
```

```python
# Follow up
result2 = session.follow_up("Show me their frequency over time")

# Display chart artifact inline
result2.artifacts[0].display()
```

```python
# Auditable mode — reasoning chain with provenance
result3 = session.prove("Calculate raises based on performance review guidelines")

# View the proof tree
result3.proof_tree.display()

# Access computed tables
raises = result3.tables["raise_recommendations"]
raises.head()
```

```python
# Direct table access
events = session.table("aggregated_events")
events.filter(polars.col("value") > 10000).sort("value", descending=True)
```

```python
# Run regression tests
test_result = session.run_tests(domains=["hr-reporting"], include_e2e=True)
test_result.summary()
```
