# Artifact Visibility and Inline References Design

## Problem Statement

The artifacts panel is becoming cluttered with intermediate outputs created during multi-step execution. Every table, file, or generated artifact gets promoted to artifact visibility, even when:

- It's a temporary intermediate step
- It's superseded by a later, more complete output
- The user only cares about the final deliverable

This applies to all artifact types: tables, Excel files, Word documents, markdown files, charts, etc.

Additionally, when a request is purely to **produce** an artifact (vs. analyze data), generating a "significant insight" feels excessive. The response should be minimal: confirmation + clickable reference.

## Current Architecture

### Storage Layers

| Layer | Storage | Purpose |
|-------|---------|---------|
| Tables | Parquet files in `.constat/<user>/sessions/<session>/tables/` | Data created during execution |
| Rich Artifacts | SQLAlchemy datastore (DuckDB default, supports PostgreSQL/SQLite) | Charts, code, errors, HTML, etc. |
| Registry | DuckDB at `.constat/registry.duckdb` | Cross-session discovery |

### Current Flow

1. **Execution**: Steps run, artifacts auto-detected (tables via `_detect_new_tables()`, files via explicit saves)
2. **Storage**: Tables saved to Parquet; files/charts saved to datastore
3. **Response**: `_synthesize_answer()` generates narrative with name-based references
4. **Display**: All artifacts shown in artifacts panel + available via `/tables`, `/artifacts`

### Current Reference Pattern

```python
# session.py lines 6746-6785
artifact_reference = f"\n\nArtifact tables created: {', '.join(artifact_tables)}"
artifact_reference += "\n(User can view these via /tables command)"
```

LLM instructed to reference by name: "see the `budget_validated_raises` table"

This pattern doesn't distinguish consequential outputs from intermediate results.

## Proposed Design

### Core Principle

**The narrative becomes the navigation.** Intermediate tables are discoverable through inline references in the response, not through a separate cluttered list.

### Three-Tier Visibility

| Tier | What's Shown | Access Method |
|------|--------------|---------------|
| **Artifacts Panel** | Only consequential outputs | Always visible in UI |
| **Inline References** | Intermediate artifacts | Clickable links in response prose |
| **Exhaustive List** | All artifacts | `/tables`, `/artifacts` commands |

### What Makes an Artifact "Consequential"

An artifact is promoted to the artifacts panel if ANY of:

1. **Final step output**: Created in the last execution step of a request
2. **Explicitly published**: LLM calls `publish(name, title?, description?)`

Artifacts NOT promoted remain accessible via inline links and exhaustive commands.

### Artifact Types Covered

This design applies to all artifact types:
- **Tables**: Parquet datasets created during analysis
- **Documents**: Excel (.xlsx), Word (.docx), markdown (.md) files
- **Visualizations**: Charts, plots, diagrams
- **Exports**: CSV, JSON, HTML outputs

### Inline Clickable References

#### Link Format

LLM generates simple inline syntax:

```
Filtered to active customers → `active_customers` (234 rows), then joined with orders...
Exported results to `quarterly_report.xlsx`
```

The UI renders artifact names as clickable links that open the artifact on click.

#### Implementation Options

**Option A: Custom URI Scheme**
```
constat://artifact/{name}
constat://table/{table_name}
constat://file/{file_path}
```

**Option B: Markdown Extension**
```markdown
[[table:active_customers]]
[[file:quarterly_report.xlsx]]
[[artifact:spending_chart]]
```

**Option C: Rich Text Tokens** (for Textual UI)
```python
@dataclass
class ArtifactReference:
    name: str
    artifact_type: str  # table, file, chart, etc.
    metadata: dict      # row_count, file_size, etc.
```

Rendered inline, clickable in the response widget.

### Production vs. Analytical Requests

#### Production Requests
"Create a table with customers who spent over $1000"
"Export the results to Excel"
"Generate a summary document"

Response should be minimal:
```
Created `high_value_customers` (1,247 rows)
```
```
Exported to `quarterly_report.xlsx`
```

No elaborate insight. The artifact IS the answer.

#### Analytical Requests
"What's the distribution of customer spending?"

Response includes analysis:
```
Customer spending follows a long-tail distribution:
- 80% of customers spend under $100
- Top 5% account for 45% of revenue
- See `spending_distribution` for the full breakdown
```

#### Detection Heuristic

Classify request intent:
- **Production**: Contains "create", "build", "generate", "export", "make a table/dataset/file"
- **Analytical**: Contains "what", "why", "how", "analyze", "compare", "distribution"

Use `_classify_intent()` (already exists) to determine response verbosity.

## Implementation Plan

### Phase 1: Inline Artifact References

**Files to modify:**
- `session.py`: Update synthesis prompt to generate clickable references
- `textual_repl.py`: Render artifact references as clickable in response display
- `feedback.py`: Handle artifact reference clicks

**Changes:**

1. Modify `_synthesize_answer()` prompt (session.py ~6780):
```python
"""
When referencing artifacts in your response:
- Use format: `artifact_name` for inline references (these become clickable links)
- For tables, include row count: `table_name` (N rows)
- For files, just the name: `report.xlsx`
- Only describe the artifact's purpose, not its full contents
"""
```

2. Add response post-processing to convert artifact names to clickable links:
```python
def _linkify_artifact_references(text: str, artifacts: list[ArtifactInfo]) -> str:
    """Convert `artifact_name` references to clickable links."""
    for artifact in artifacts:
        text = text.replace(f"`{artifact.name}`", f"[[{artifact.type}:{artifact.name}]]")
    return text
```

3. Render `[[type:name]]` as clickable in Textual:
```python
class ArtifactLink(ClickableText):
    def on_click(self):
        self.app.open_artifact(self.artifact_type, self.artifact_name)
```

### Phase 2: Artifact Panel Filtering

**Files to modify:**
- `storage/parquet_store.py`: Add `is_published` flag to table metadata
- `storage/datastore.py`: Add `is_published` flag to artifact metadata
- `session.py`: Track which artifacts are "final" vs "intermediate"
- `textual_repl.py`: Filter artifacts panel to show only consequential outputs

**Changes:**

1. Add `publish()` function available to generated code:
```python
def publish(name: str, title: str = None, description: str = None):
    """Mark an artifact as a consequential output for the artifacts panel."""
    _constat_runtime.mark_published(name, title, description)
```

2. Auto-publish artifacts from final step:
```python
# In step execution
if step.is_final:
    for artifact in new_artifacts:
        self.mark_published(artifact)
```

3. Filter artifacts panel:
```python
def _get_visible_artifacts(self) -> list[ArtifactInfo]:
    return [a for a in self.artifacts if a.is_published]
```

### Phase 3: Production Request Detection

**Files to modify:**
- `session.py`: Adjust synthesis based on intent classification

**Changes:**

1. Use existing intent classification:
```python
intent = self._classify_intent(problem)
if intent.is_production_request:
    # Minimal response
    return self._format_production_response(tables_created)
else:
    # Full synthesis
    return self._synthesize_answer(...)
```

2. Production response format:
```python
def _format_production_response(self, artifacts: list[ArtifactInfo]) -> str:
    lines = ["Created:"]
    for a in artifacts:
        if a.artifact_type == "table":
            lines.append(f"- `{a.name}` ({a.row_count:,} rows)")
        else:
            lines.append(f"- `{a.name}`")
    return "\n".join(lines)
```

## Data Model Changes

### Artifact Metadata Extension

```python
@dataclass
class ArtifactMetadata:
    name: str
    artifact_type: str                # table, file, chart, etc.
    step_number: int
    created_at: datetime
    # New fields
    is_published: bool = False        # Explicitly marked as output
    is_final_step: bool = False       # Created in last step
    title: str = None                 # Human-friendly name
    description: str = None           # What this artifact contains
    # Type-specific metadata
    row_count: int = None             # For tables
    file_size: int = None             # For files
    file_path: str = None             # For files
```

### Registry Schema Update

```sql
-- For tables
ALTER TABLE constat_tables ADD COLUMN is_published BOOLEAN DEFAULT FALSE;
ALTER TABLE constat_tables ADD COLUMN is_final_step BOOLEAN DEFAULT FALSE;
ALTER TABLE constat_tables ADD COLUMN title TEXT;
ALTER TABLE constat_tables ADD COLUMN description TEXT;

-- For other artifacts
ALTER TABLE constat_artifacts ADD COLUMN is_published BOOLEAN DEFAULT FALSE;
ALTER TABLE constat_artifacts ADD COLUMN is_final_step BOOLEAN DEFAULT FALSE;
```

## UI Mockups

### Artifacts Panel (After)

```
┌─ Artifacts ──────────────────────┐
│                                  │
│  Tables                          │
│  ├─ high_value_customers (1,247) │
│  └─ monthly_summary (12)         │
│                                  │
│  Files                           │
│  └─ quarterly_report.xlsx        │
│                                  │
│  Charts                          │
│  └─ spending_distribution.png    │
│                                  │
│  ─────────────────────────────── │
│  [Show all (12 intermediate)]    │
│                                  │
└──────────────────────────────────┘
```

### Response with Inline Links

```
Filtered customers by activity status → `active_customers` (234 rows),
then joined with order history → `customer_orders` (1,892 rows).

Final analysis shows 15% of active customers have no orders.
See `high_value_customers` for the top spenders.

Exported summary to `quarterly_report.xlsx`.
```

Each backticked artifact name is clickable.

## Migration

1. Existing artifacts default to `is_published=True` (preserve current behavior)
2. New artifacts default to `is_published=False` unless:
   - Created in final step
   - Explicitly published
3. Users can manually publish/unpublish via `/publish <name>` command

## Open Questions

1. **Session vs. request scope**: Should "final step" mean final step of the current request, or final step of the session? (Likely: current request)

2. **Link syntax**: Which inline link format works best for the Textual UI rendering?

## Success Metrics

- Artifacts panel shows 50-70% fewer items (only consequential ones)
- All artifacts remain discoverable via inline links or exhaustive commands
- Production requests get responses <50 words
- User can navigate from response text directly to any referenced artifact

## Related

- [Federated Query Optimizer](../plans/query-optimizer.md) - Cost-based optimization for heterogeneous data sources, including authority hierarchy and data movement optimization