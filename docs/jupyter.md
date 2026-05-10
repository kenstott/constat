# Jupyter Notebook Guide

The Jupyter interface is Constat's notebook experience — reproducible analysis, per-cell replay on kernel restart, DataFrame ecosystem, and zero-code magics.

## Installation

```bash
pip install -e constat-jupyter/
```

The notebook client talks to a running Constat server (not a local library). This means:
- No LLM keys, DB connections, or catalog configuration in the notebook environment
- Sessions shared with the web UI (start in notebook, view in browser)
- Execution runs in the server process; kernel crash doesn't kill queries

## Setup

```python
# Cell 1: Load the extension
%load_ext constat_jupyter

# Cell 2: Connect to domains
%constat connect sales-analytics,hr-reporting
```

On first connect, a sidecar file (`<notebook>.constat.json`) is written with the session ID and domains. On subsequent connects (after kernel restart), the magic restores the session and enables per-cell replay.

## Magics Interface

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
| (none) | `solve()` first time, `follow_up()` after. Conditional approval. |
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

## Approval Modes

| Mode | Magic flag | Behavior |
|---|---|---|
| **Conditional** (default) | (none) | Server auto-approves simple plans; shows widget for complex ones |
| **Auto** | `auto` | Client auto-approves everything, no widget |
| **Always ask** | `approve` | Forces approval widget for every plan |

## Session Replay

### Automatic Per-Cell Replay (Kernel Restart + Run All)

The notebook magic makes replay seamless. When `%constat connect` detects a stored session with scratchpad data:
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

## Power User Python API

After `%constat connect`, `_constat_client` and `_constat_session` are injected into the notebook namespace:

```python
# Power users can use the full Python API
await _constat_session.table("results")
await _constat_session.command("/compact")
```

### Full API

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

## SolveResult

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

`_repr_html_` gives automatic rich rendering when the result is the last expression in a cell.

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

## Configuration

```python
# Option 1: Constructor
client = ConstatClient("http://localhost:8000", token="bearer-token")

# Option 2: Environment variables
#   CONSTAT_SERVER_URL=http://localhost:8000
#   CONSTAT_AUTH_TOKEN=bearer-token
client = ConstatClient()
```

### Firebase Login

`%constat login` uses the Firebase REST API `signInWithPassword` endpoint. Requires `CONSTAT_FIREBASE_API_KEY` environment variable. Collects email via `input()`, password via `getpass.getpass()`. Stores token and allows subsequent `%constat connect` to authenticate.

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
