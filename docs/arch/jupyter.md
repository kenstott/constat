# Jupyter Notebook Interface

**Status:** Implemented
**Approach:** HTTP client SDK (not local lib)

## Decision: HTTP over Local Lib

The notebook client talks to a running constat server. Rationale:

- Server already exposes 30+ REST endpoints + WebSocket event streaming
- No need to configure LLM keys, DB connections, or catalog in notebook env
- Sessions are shared with web UI (start in notebook, view in browser)
- Execution runs in server process; kernel crash doesn't kill queries
- Multi-user/RBAC already built into server layer

## Package Location

`constat-jupyter/` — separate pip-installable package alongside the main constat repo.

```
constat-jupyter/constat_jupyter/
    __init__.py          # Public API: ConstatClient, configure, load_ipython_extension
    client.py            # HTTP + WebSocket client
    magic.py             # IPython magics (%constat, %%constat)
    widgets.py           # ipywidgets for plan approval, clarification
    progress.py          # Execution progress rendering
    config.py            # Connection config (server URL, auth token)
```

## Magics Interface

The primary notebook experience. Zero Python knowledge required.

### Setup

```python
%load_ext constat_jupyter
%constat connect sales-analytics,hr-reporting
```

On first connect, a sidecar file (`<notebook>.constat.json`) is written with the session ID and domains. On subsequent connects (after kernel restart), the magic restores the session and enables per-cell replay.

### Asking Questions

```python
%%constat
What are the top 10 products by revenue?
```

```python
%%constat
Filter to Q4 only
```

First question uses `solve()`, subsequent questions automatically use `follow_up()`.

### Cell Magic Flags

| Flag | Behavior |
|---|---|
| (none) | `solve()` first time, `follow_up()` after. Conditional approval (server auto-approves simple plans, widget for complex). |
| `new` | New session (same domains), then `solve()` |
| `published` | Ask + display starred tables/artifacts only |
| `auto` | Auto-approve all plans (no approval widget) |
| `approve` | Force approval widget for every plan |
| `code` | Also display code/sql artifacts |
| `output` | Also display step output artifacts |
| `verbose` | Display all artifacts (code + output) |
| `include:md,html` | Only show these artifact types |
| `exclude:table` | Hide these artifact types |

Flags combine freely:

```python
%%constat new
Something completely different

%%constat published
Show me the final report

%%constat auto include:md
Summarize the results

%%constat approve verbose
Complex query I want to review before execution
```

### Approval Modes

| Mode | Magic flag | Behavior |
|---|---|---|
| **Conditional** (default) | (none) | Server auto-approves simple plans; shows widget for complex ones |
| **Auto** | `auto` | Client auto-approves everything, no widget |
| **Always ask** | `approve` | Forces approval widget for every plan |

### Line Magic Subcommands

| Subcommand | Action |
|---|---|
| `connect [d1,d2,...]` | Create client+session, optionally set domains |
| `login` | Firebase email/password auth |
| `status` | Show session ID, active domains |
| `domains` | List available domains on server |
| `domains active` | Show currently active domains |
| `domains add <name>` | Add a domain to current session |
| `domains drop <name>` | Remove a domain from current session |
| `sources` | Show all data sources by domain |
| `add database <uri> [name]` | Add a database to session |
| `add api <spec_url> [name]` | Add an API to session |
| `add document <uri>` | Add a document to session |
| `tables` | List tables from current session |
| `table <name>` | Fetch and display a table (iTables) |
| `artifacts` | List artifacts (id, name, type, starred) |
| `artifact <id>` | Display a specific artifact |
| (no args) | Show usage/help |

### Async Handling

Cell magic generates Python code with `await` and passes to `self.shell.run_cell()`. IPython's autoawait handles the event loop. Widget callbacks (plan approval, clarification) work because the kernel loop keeps running.

### Error Display

Errors are displayed as styled HTML boxes, not raw tracebacks:

```
Error: Connection refused — is the server running?
```

### Power User Globals

After `%constat connect`, `_constat_client` and `_constat_session` are injected into the notebook namespace:

```python
# Power users can use the full Python API
await _constat_session.table("results")
await _constat_session.command("/compact")
```

### Firebase Login

`%constat login` uses the Firebase REST API `signInWithPassword` endpoint. Requires `CONSTAT_FIREBASE_API_KEY` environment variable. Collects email via `input()`, password via `getpass.getpass()`. Stores token and allows subsequent `%constat connect` to authenticate.

## Power User API

For users who prefer full Python control.

```python
from constat_jupyter import ConstatClient

# Connect to server
client = ConstatClient("http://localhost:8000", token="...")

# Session management
session = client.create_session()
session = client.get_session("session-id")
sessions = client.list_sessions()

# Core query (blocks, shows live progress, returns result)
result = await session.solve("What are the top 10 items by value?")
result = await session.follow_up("Break that down by region")

# Auditable mode — reasoning chain with full provenance
result = await session.reason_chain("Calculate raises based on performance guidelines")

# Result access
result.answer           # str - synthesized answer
result.tables           # dict[str, polars.DataFrame]
result.artifacts        # list[Artifact] with .display() method
result.suggestions      # list[str]

# Direct table access
df = session.table("results")                    # -> polars.DataFrame
df = session.table("results", pandas=True)       # -> pandas.DataFrame
session.tables()                                 # list all tables

# Artifacts
artifacts = session.artifacts()
artifacts[0].display()     # Renders inline (chart, HTML, markdown)

# Schema browsing
client.databases()
client.table_schema("my_database", "events")

# Domain management
client.domains()                              # list configured domains
session.set_domains(["sales-analytics"])

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
results = session.run_tests(domains=["hr-reporting"])
```

## Session Replay

Exploratory sessions can be replayed — stored scratchpad code is re-executed without LLM codegen. This is useful for demos, testing with updated data, or resuming work after a break.

### Automatic Per-Cell Replay (Kernel Restart + Run All)

The notebook magic makes replay seamless. A per-notebook sidecar file (`<notebook>.constat.json`) persists the session ID and domains across kernel restarts.

When `%constat connect` detects a stored session with scratchpad data:
1. It restores the session and fetches the stored cell records from the server (distinct `objective_index` values from scratchpad entries)
2. Each subsequent `%%constat` cell replays only its own objective's steps via `session.replay(question, objective_index=N)`
3. Once all stored cells are replayed, new `%%constat` cells fall through to normal `follow_up()` — so you can extend the analysis after a restart

No approval widget is shown during replay (plans are auto-approved since they use stored code).

### Manual Replay (Python API)

```python
# Replay entire session
result = await session.replay("Original question")

# Replay a single objective
result = await session.replay("Original question", objective_index=0)
```

### Server-Side

`Session.replay(problem, objective_index=None)` loads scratchpad entries and re-executes stored code. When `objective_index` is provided, only entries with that index are replayed. The `plan_ready` event is emitted with `auto_approved: True` so clients skip the approval widget.

The `POST /api/sessions/{id}/query` endpoint accepts `replay: true` and optional `objective_index` in the request body.

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

### Async in Jupyter

IPython's built-in `autoawait` handles the event loop. The client's public API is async (using `await`). No external event loop patching required.

## Progress Rendering

Use `IPython.display` for live-updating output during execution.

### Step Number Continuity

Follow-up queries continue step numbering from where the previous query left off.
The server renumbers follow-up steps (e.g., if query 1 had steps 1–6, query 2
shows steps 7–9). The progress widget tracks `_base_step` from the `plan_ready`
event to correctly index into its arrays and display server-provided step numbers.

### Agent/Skill Tags

When a step has an associated agent (`role_id`) or skills (`skill_ids`), the
`step_start` event includes them. The progress widget renders these as grey tags
next to the step goal: `✓ Step 3: Load employee data [hr-analyst, data-quality]`.

### With ipywidgets

```
Planning...
Plan: "Analyze items by category" (steps 1–3)
  [Approve] [Reject] [Edit]          <- ipywidgets buttons

Step 1/3: Query events table ████████░░ generating...
Step 2/3: Aggregate by category ████████████ complete
Step 3/3: Join with metadata ░░░░░░░░░░ pending

Synthesizing answer...
```

### Fallback (no ipywidgets)

```
[planning] Generating plan...
[plan] Analyze items (steps 1–3)
  1. Query events table
  2. Aggregate by category
  3. Join with metadata
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

    def _repr_html_(self) -> str:
        """IPython rich display: render answer + tables as HTML."""

    def display(self, published: bool = False):
        """Full rich display: answer + tables + artifacts.
        If published=True, only show starred items."""
```

`_repr_html_` gives automatic rich rendering when the result is the
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

### Current: Parquet via REST

`GET /api/sessions/{id}/tables/{name}/download?format=parquet` returns Parquet bytes.
Efficient binary transport with zero-copy DataFrame construction via Polars.

## Dependencies

Required:
- `httpx` - HTTP client (async + sync)
- `websockets` - WebSocket client
- `polars` - Primary DataFrame library
- `ipython` - Magics support

Optional (enhanced experience):
- `ipywidgets` - Interactive plan approval, progress bars
- `itables` - Interactive DataTable rendering
- `pandas` - Alternative DataFrame output
- `plotly` - Inline chart rendering from Plotly artifacts

## Configuration

```python
# Option 1: Constructor
client = ConstatClient("http://localhost:8000", token="bearer-token")

# Option 2: Environment variables
#   CONSTAT_SERVER_URL=http://localhost:8000
#   CONSTAT_AUTH_TOKEN=bearer-token
client = ConstatClient()
```

## Example Notebook

### Using Magics (recommended)

```python
# Cell 1
%load_ext constat_jupyter
```

```python
# Cell 2
%constat connect sales-analytics,hr-reporting
```

```python
# Cell 3
%%constat
What are our top 10 items by total value?
```

```python
# Cell 4
%%constat
Break that down by region
```

```python
# Cell 5
%constat tables
```

```python
# Cell 6
%constat table top_items
```

```python
# Cell 7
%%constat new
Something completely different
```

### Using Python API

```python
from constat_jupyter import ConstatClient

client = ConstatClient()
session = client.create_session()

# Ask a question - shows live progress, plan approval widget
result = await session.solve("What are our top 10 items by total value?")
result
```

```python
# Access the data as a DataFrame
df = result.tables["top_items"]
df.head()
```

```python
# Follow up
result2 = await session.follow_up("Show me their frequency over time")

# Display chart artifact inline
result2.artifacts[0].display()
```

```python
# Direct table access
events = session.table("aggregated_events")
events.filter(polars.col("value") > 10000).sort("value", descending=True)
```

## WebSocket Events

The client handles these event types for progress rendering:

### Exploratory mode
- `planning_start`, `plan_ready`, `plan_approved`
- `step_start` (includes `agent`, `skills` when present), `step_generating`, `step_executing`, `step_complete`, `step_error`
- `clarification_needed`, `clarification_received`
- `synthesizing`, `query_complete`, `query_error`

### Auditable mode (additional)
- `chain_start`, `chain_complete`, `chain_summary_ready`
- `fact_start`, `fact_resolved`, `fact_failed`, `fact_blocked`
- `dag_execution_start`
- `inference_code`, `inference_executing`, `inference_complete`
