# React UI Implementation Plan

> **Last Updated**: 2026-01-23
> **Verified Against**: Current codebase (session.py, core/models.py, proof_tree.py, discovery/models.py, storage/learnings.py)
> **Status**: Phase 1 Complete - API Server Mode implemented

## Implementation Progress

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: API Server Mode | ✅ Complete | All endpoints implemented, 53 tests passing |
| Phase 2: React Frontend | ✅ Complete | Project structure, components, and stores implemented |
| Phase 3: Implementation Steps | ✅ Complete | Steps 1-7 complete |

## Overview

This plan outlines the implementation of a React-based web UI for Constat, including an API server mode to expose the existing functionality via HTTP/WebSocket.

### Existing Infrastructure

The following components already exist and will be wrapped by the API:

- **Session class** (`constat/session.py`): Core orchestrator with methods for `solve()`, `follow_up()`, `add_file()`, `add_database()`
- **StepEvent system**: Event emission via `_emit_event()` for real-time progress updates
- **Approval/Clarification callbacks**: `set_approval_callback()`, `set_clarification_callback()`
- **ProofNode** (`constat/proof_tree.py`): Hierarchical fact resolution tree for auditable mode
- **Artifact system** (`constat/core/models.py`): ArtifactType enum with 17 artifact types
- **Entity extraction** (`constat/discovery/models.py`): Entity and EntityType classes
- **Learning storage** (`constat/storage/learnings.py`): Two-tier learning/rule system
- **Dependencies**: FastAPI and uvicorn already in `pyproject.toml`

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Status Bar                                      │
├────────┬────────────────────────────────────────────────┬───────────────────┤
│        │                                                │                   │
│ Hamburger│           Conversation Panel                 │  Artifact Panel   │
│  Menu   │                                                │   (Accordion)     │
│        │  - Query input                                 │                   │
│ Global  │  - Plan display                               │  - Charts         │
│Commands │  - Step execution                             │  - Tables         │
│        │  - Results/Output                              │  - Code           │
│        │  - Follow-up queries                           │  - HTML           │
│        │                                                │  - Facts          │
│        │                                                │                   │
├────────┴────────────────────────────────────────────────┴───────────────────┤
│                              Toolbar                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: API Server Mode

### 1.1 Server Framework

**Technology**: FastAPI with uvicorn (already in pyproject.toml dependencies)

**Location**: `constat/server/` (new module)

**Files**:
- `constat/server/__init__.py`
- `constat/server/app.py` - FastAPI application
- `constat/server/routes/` - Route modules
- `constat/server/websocket.py` - WebSocket handlers for real-time updates
- `constat/server/models.py` - Pydantic request/response models
- `constat/server/session_manager.py` - Server-side session management

### 1.2 CLI Integration

Add `serve` command to CLI:

```bash
constat serve --port 8000 --host 0.0.0.0 --config config.yaml
```

**Options**:
- `--port` - Server port (default: 8000)
- `--host` - Bind address (default: 127.0.0.1)
- `--config` - Configuration file path
- `--cors-origins` - Allowed CORS origins
- `--reload` - Enable hot reload for development

### 1.3 API Endpoints

#### Session Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sessions` | Create new session |
| GET | `/api/sessions` | List sessions (paginated) |
| GET | `/api/sessions/{id}` | Get session details |
| DELETE | `/api/sessions/{id}` | Close/delete session |
| POST | `/api/sessions/{id}/resume` | Resume existing session |

#### Query Execution
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sessions/{id}/query` | Submit query for execution |
| GET | `/api/sessions/{id}/plan` | Get current plan |
| POST | `/api/sessions/{id}/plan/approve` | Approve plan for execution |
| POST | `/api/sessions/{id}/plan/reject` | Reject plan with feedback |
| POST | `/api/sessions/{id}/cancel` | Cancel running execution |

#### Data Access
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/sessions/{id}/artifacts` | List artifacts |
| GET | `/api/sessions/{id}/artifacts/{aid}` | Get artifact content |
| GET | `/api/sessions/{id}/tables` | List session tables |
| GET | `/api/sessions/{id}/tables/{name}` | Get table data (paginated) |
| GET | `/api/sessions/{id}/facts` | Get resolved facts (auditable mode) |
| POST | `/api/sessions/{id}/facts/{fid}/persist` | Cache fact for future use |
| POST | `/api/sessions/{id}/facts/{fid}/forget` | Invalidate cached fact |
| PUT | `/api/sessions/{id}/facts/{fid}` | Edit fact value |
| GET | `/api/sessions/{id}/proof-tree` | Get proof tree (auditable mode) |
| GET | `/api/sessions/{id}/entities` | List extracted entities |
| GET | `/api/sessions/{id}/entities?type=` | Filter entities by type |
| POST | `/api/sessions/{id}/entities/{eid}/glossary` | Add entity to glossary |

#### Schema Discovery
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/schema/databases` | List configured databases |
| GET | `/api/schema/databases/{name}/tables` | List tables in database |
| GET | `/api/schema/tables/{db}/{table}` | Get table schema |
| GET | `/api/schema/search?q=` | Search tables semantically |

#### File Upload
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sessions/{id}/files` | Upload file (multipart/form-data or data URI) |
| GET | `/api/sessions/{id}/files` | List uploaded files |
| GET | `/api/sessions/{id}/files/{fid}` | Download/retrieve file |
| DELETE | `/api/sessions/{id}/files/{fid}` | Remove uploaded file |

**File upload flow:**
1. Client sends file as multipart/form-data or base64 data URI
2. Server stores in session temp directory (e.g., `~/.constat/sessions/{id}/uploads/`)
3. Server returns `file://` URI for use in queries
4. Files are cleaned up when session is deleted/expired

**Request body (data URI option):**
```json
{
  "filename": "report.pdf",
  "content_type": "application/pdf",
  "data": "data:application/pdf;base64,JVBERi0xLjQK..."
}
```

**Response:**
```json
{
  "id": "f_abc123",
  "filename": "report.pdf",
  "file_uri": "file:///Users/.../.constat/sessions/sess_xyz/uploads/report.pdf",
  "size_bytes": 102400,
  "content_type": "application/pdf",
  "uploaded_at": "2024-01-15T10:30:00Z"
}
```

#### File Reference (wraps REPL `/file` command)

*Note: These endpoints wrap the existing `Session.add_file()` method. Different from File Upload - this adds references to existing files/URLs.*

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sessions/{id}/file-refs` | Add file reference (wraps `session.add_file()`) |
| GET | `/api/sessions/{id}/file-refs` | List session file references |
| DELETE | `/api/sessions/{id}/file-refs/{name}` | Remove file reference |

**Request body:**
```json
{
  "name": "sales_report",
  "uri": "https://example.com/reports/sales.csv",
  "auth": "Bearer token123",
  "description": "Monthly sales report CSV"
}
```

**Response:**
```json
{
  "name": "sales_report",
  "uri": "https://example.com/reports/sales.csv",
  "has_auth": true,
  "description": "Monthly sales report CSV",
  "added_at": "2024-01-15T10:30:00Z"
}
```

#### Dynamic Database Connection

*Note: These endpoints wrap the existing `Session.add_database()` method and `/database` REPL command.*

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/sessions/{id}/databases` | Add database to session (wraps `session.add_database()`) |
| GET | `/api/sessions/{id}/databases` | List session databases (config + dynamic) |
| DELETE | `/api/sessions/{id}/databases/{name}` | Remove dynamic database |
| POST | `/api/sessions/{id}/databases/{name}/test` | Test database connection |

**Supported database types:**

1. **File-based databases** (uploaded or local path):
   - SQLite: `sqlite:///path/to/db.sqlite`
   - DuckDB: `duckdb:///path/to/db.duckdb`
   - Can upload `.sqlite` or `.duckdb` files via `/add`, then connect

2. **SQLAlchemy databases** (connection string):
   - PostgreSQL: `postgresql://user:pass@host:5432/dbname`
   - MySQL: `mysql+pymysql://user:pass@host:3306/dbname`
   - SQL Server: `mssql+pyodbc://user:pass@host/dbname`

3. **NoSQL databases**:
   - MongoDB: `mongodb://user:pass@host:27017/dbname`
   - Elasticsearch: `elasticsearch://host:9200`

**Request body:**
```json
{
  "name": "sales_db",
  "uri": "postgresql://user:pass@localhost:5432/sales",
  "description": "Production sales database",
  "type": "sqlalchemy",
  "options": {
    "schema": "public",
    "readonly": true
  }
}
```

**For file-based (using previously uploaded file):**
```json
{
  "name": "local_analytics",
  "file_id": "f_abc123",
  "description": "Uploaded DuckDB analytics file",
  "type": "duckdb"
}
```

**Response:**
```json
{
  "name": "sales_db",
  "type": "sqlalchemy",
  "dialect": "postgresql",
  "connected": true,
  "table_count": 45,
  "added_at": "2024-01-15T10:30:00Z",
  "is_dynamic": true
}
```

#### Settings & Learnings
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/learnings` | Get captured learnings |
| POST | `/api/learnings` | Add new learning |
| DELETE | `/api/learnings/{id}` | Remove learning |
| GET | `/api/config` | Get current config (sanitized) |

### 1.4 WebSocket Events

**Endpoint**: `ws://host/api/sessions/{id}/ws`

**Server → Client Events**:
```typescript
// Plan generated
{ type: "plan_generated", plan: Plan }

// Step lifecycle
{ type: "step_started", step_number: number, goal: string }
{ type: "step_progress", step_number: number, message: string }
{ type: "step_completed", step_number: number, result: StepResult }
{ type: "step_failed", step_number: number, error: string, retry_count: number }

// Proof tree (auditable mode) - real-time node updates
{ type: "proof_tree_created", root: ProofNode }
{ type: "proof_node_resolving", node_path: string[], name: string }
{ type: "proof_node_resolved", node_path: string[], value: any, source: string, confidence?: number }
{ type: "proof_node_failed", node_path: string[], error: string }
{ type: "proof_node_cached", node_path: string[], value: any }
{ type: "proof_node_added", parent_path: string[], node: ProofNode }  // New child node discovered

// Artifacts
{ type: "artifact_created", artifact: Artifact }

// Entities (as discovered during execution)
{ type: "entity_extracted", entity: Entity }

// Execution
{ type: "execution_complete", success: boolean, output: string }
{ type: "execution_error", error: string }

// Session
{ type: "session_state_changed", state: SessionState }
```

**Client → Server Events**:
```typescript
{ type: "cancel_execution" }
{ type: "approve_plan", plan_id: string }
{ type: "reject_plan", feedback: string }
```

### 1.5 Authentication (Future)

- JWT-based authentication
- API key support for programmatic access
- Session isolation per user

---

## Phase 2: React Frontend

### 2.1 Technology Stack

- **Framework**: React 18+ with TypeScript
- **Build Tool**: Vite
- **State Management**: Zustand (lightweight, TypeScript-friendly)
- **UI Components**: Tailwind CSS + Headless UI (or shadcn/ui)
- **Data Fetching**: TanStack Query (React Query)
- **WebSocket**: Native WebSocket with reconnection logic
- **Charts**: Re-render Plotly artifacts (plotly.js-dist)
- **Code Display**: Monaco Editor or react-syntax-highlighter
- **Tables**: TanStack Table

### 2.2 Project Structure

```
constat-ui/
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── api/
│   │   ├── client.ts          # Axios/fetch client
│   │   ├── sessions.ts        # Session API calls
│   │   ├── queries.ts         # Query API calls
│   │   └── websocket.ts       # WebSocket manager
│   ├── components/
│   │   ├── layout/
│   │   │   ├── StatusBar.tsx
│   │   │   ├── Toolbar.tsx
│   │   │   ├── HamburgerMenu.tsx
│   │   │   └── MainLayout.tsx
│   │   ├── conversation/
│   │   │   ├── ConversationPanel.tsx
│   │   │   ├── QueryInput.tsx
│   │   │   ├── PlanView.tsx
│   │   │   ├── StepCard.tsx
│   │   │   ├── ResultView.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   └── renderers/           # Inline conversation renderers
│   │   │       ├── ProofTreeRenderer.tsx
│   │   │       └── DFDRenderer.tsx
│   │   ├── artifacts/
│   │   │   ├── ArtifactPanel.tsx
│   │   │   ├── ArtifactAccordion.tsx
│   │   │   ├── ChartViewer.tsx
│   │   │   ├── TableViewer.tsx
│   │   │   ├── CodeViewer.tsx
│   │   │   ├── HtmlViewer.tsx
│   │   │   ├── FactTableViewer.tsx   # Persist/forget/edit actions
│   │   │   ├── EntityViewer.tsx      # Filter/explore/add to glossary
│   │   │   └── LearningViewer.tsx    # Edit/delete/promote actions
│   │   └── common/
│   │       ├── LoadingSpinner.tsx
│   │       ├── ErrorBoundary.tsx
│   │       └── Accordion.tsx
│   ├── hooks/
│   │   ├── useSession.ts
│   │   ├── useWebSocket.ts
│   │   ├── useArtifacts.ts
│   │   └── useExecutionStatus.ts
│   ├── store/
│   │   ├── sessionStore.ts
│   │   ├── artifactStore.ts
│   │   └── uiStore.ts
│   ├── types/
│   │   ├── api.ts
│   │   ├── session.ts
│   │   ├── plan.ts
│   │   └── artifact.ts
│   └── utils/
│       ├── formatters.ts
│       └── constants.ts
├── public/
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

### 2.3 Component Specifications

#### Status Bar
**Location**: Top of viewport (fixed)

**Content**:
- App logo/name (left)
- Session indicator (session ID, status badge)
- Connection status (WebSocket connected/disconnected)
- Current execution mode (Exploratory / Auditable)
- User/settings dropdown (right)

#### Hamburger Menu
**Location**: Left side, collapsible drawer

**Global Commands** (mirrors CLI REPL commands):
- `/tables` - View session tables
- `/show <table>` - Display table contents
- `/code [step]` - Show generated code (all or specific step)
- `/query <sql>` - Run SQL query on datastore
- `/facts` - View resolved facts
- `/learnings [category]` - View/manage learnings
- `/history`, `/sessions` - Session history
- `/remember <fact>` - Persist a session fact
- `/forget <name>` - Forget a remembered fact
- `/export <table> [file]` - Export table to CSV/XLSX
- `/artifacts` - Show saved artifacts with file:// URIs
- `/audit` - Re-derive last result with full audit trail
- `/prove` - Switch to auditable mode derivation
- `/add` - Upload file from browser (opens file picker, uploads as data URI → stored as temp file → returns file:// URI) *[new for web UI]*
- `/file` - Add file reference (wraps existing REPL command `session.add_file()` - for URLs or local paths with optional auth)
- `/database`, `/db` - Add/manage database connections (wraps `session.add_database()`)

**Settings**:
- Theme toggle (light/dark)
- Execution mode preference
- Auto-approve plans toggle
- Notification preferences

#### Conversation Panel
**Location**: Center, main content area

**Components**:
1. **Message History**: Scrollable list of exchanges
   - User queries (right-aligned bubbles)
   - System responses (left-aligned)
   - Plan cards (expandable)
   - Step execution progress
   - Final output/synthesis

2. **Query Input**: Bottom of panel
   - Multi-line text input
   - Submit button
   - Mode indicator
   - Quick action buttons (clear, retry)

3. **Plan View**: Inline card showing:
   - Steps with goals
   - Dependency graph (optional visualization)
   - Approve/Reject buttons (if manual approval)
   - Execution progress per step

4. **Step Card**: Expandable card per step
   - Goal description
   - Status indicator (pending/running/success/failed)
   - Generated code (collapsible)
   - Output/stdout (collapsible)
   - Tables created
   - Error details (if failed)

5. **Inline Renderers** (embedded in conversation flow):
   - **Proof Tree Renderer** - Hierarchical tree visualization for auditable mode derivations
     - Collapsible nodes showing fact → source relationships
     - Status indicators (resolved, pending, failed, cached)
     - Click to expand provenance details
   - **DFD Renderer** - Data Flow Diagram visualization
     - Interactive diagram showing data transformations
     - Nodes for data sources, transformations, outputs
     - Hover for details, click to inspect

#### Artifact Panel (Right)
**Location**: Right side, resizable

**Accordion Sections**:
1. **Charts** - Interactive Plotly visualizations
2. **Tables** - Paginated data tables with sort/filter
3. **Code** - Generated code snippets with syntax highlighting
4. **HTML/Markdown** - Rendered HTML content
5. **Files** - Downloadable files (CSV, JSON, etc.)

**Automated/Live Sections** (system-managed, not step-generated):
6. **Facts** - Resolved facts table (auditable mode) with action handlers:
   - Persist fact to cache
   - Forget/invalidate cached fact
   - Edit fact value
   - View fact source/provenance
7. **Entities** - Extracted entities from session with actions:
   - Filter by type (table, column, concept, business_term, api_endpoint, api_field, api_schema)
   - View related documents/chunks
   - Add to glossary/business terms
   - Explore entity (trigger entity search)
8. **Learnings** - Session-captured corrections and rules:
   - Mark as permanent/temporary
   - Edit rule text
   - Delete learning
   - Promote to global rule

**Behavior**:
- Multiple sections can be open simultaneously
- Each artifact shows timestamp and step source
- Click artifact to expand/focus
- Download button for exportable artifacts

#### Toolbar
**Location**: Bottom of viewport (fixed)

**Content**:
- Quick actions: New Query, Cancel, Clear Session
- Mode toggle (Exploratory / Auditable)
- Status indicators (queries run, tables created)
- Help/keyboard shortcuts button

---

## Phase 3: Implementation Steps

### Step 1: API Server Foundation
1. Create `constat/server/` module structure
2. Implement FastAPI app with CORS configuration
3. Add `serve` command to CLI
4. Implement session management endpoints
5. Add basic health check endpoint

### Step 2: Core API Endpoints
1. Implement query submission endpoint
2. Add WebSocket handler for real-time updates
3. Integrate with existing Session class
4. Implement artifact retrieval endpoints
5. Add table data pagination

### Step 3: React Project Setup
1. Initialize Vite + React + TypeScript project
2. Configure Tailwind CSS
3. Set up project structure
4. Create API client with type definitions
5. Implement WebSocket connection manager

### Step 4: Layout Components
1. Build MainLayout with grid structure
2. Implement StatusBar component
3. Create HamburgerMenu with drawer
4. Build Toolbar component
5. Set up responsive breakpoints

### Step 5: Conversation Panel
1. Implement QueryInput with submit handling
2. Create MessageBubble components
3. Build PlanView with step display
4. Add StepCard with expandable sections
5. Implement auto-scroll behavior

### Step 6: Artifact Panel
1. Create ArtifactAccordion container
2. Implement ChartViewer (Plotly integration)
3. Build TableViewer with pagination
4. Add CodeViewer with syntax highlighting
5. Implement FactTableViewer with action handlers (persist, forget, edit, view provenance)

### Step 7: State Management & Integration
1. Set up Zustand stores
2. Implement React Query hooks
3. Connect WebSocket to state updates
4. Add optimistic updates for UX
5. Implement error handling

### Step 8: Polish & Testing
1. Add loading states and skeletons
2. Implement error boundaries
3. Add keyboard shortcuts
4. Write component tests
5. End-to-end testing

---

## API Models (TypeScript)

```typescript
// Session
interface Session {
  id: string;
  created_at: string;
  last_activity: string;
  status: "active" | "idle" | "executing" | "closed";
  query_count: number;
  mode: "exploratory" | "auditable";
}

// Plan
// Note: Maps to constat/core/models.py Plan dataclass
interface Plan {
  problem: string;                        // Original user problem
  steps: Step[];
  created_at: string;
  // Execution state
  current_step: number;
  completed_steps: number[];
  failed_steps: number[];
  is_complete: boolean;                   // Computed: all steps complete
  contains_sensitive_data: boolean;       // If true, email ops need auth
  // Added for API (not in Python model)
  id?: string;                            // Server-assigned ID
  status?: "pending_approval" | "approved" | "executing" | "completed" | "failed";
}

// Note: Maps to constat/core/models.py Step dataclass
interface Step {
  number: number;
  goal: string;                           // Natural language description
  expected_inputs: string[];
  expected_outputs: string[];
  depends_on: number[];                   // Explicit step dependencies
  step_type: "python";                    // Currently only Python supported
  task_type: TaskType;
  complexity: "low" | "medium" | "high";  // Hint for model selection
  status: StepStatus;
  code?: string;                          // Populated during execution
  result?: StepResult;
}

type TaskType =
  | "planning" | "replanning"
  | "sql_generation" | "python_analysis"
  | "intent_classification" | "mode_selection"
  | "fact_resolution" | "summarization"
  | "embedding" | "quick_response";

type StepStatus = "pending" | "running" | "completed" | "failed" | "skipped";

// Note: Maps to constat/core/models.py StepResult dataclass
interface StepResult {
  success: boolean;
  stdout: string;
  error?: string;
  attempts: number;
  duration_ms: number;
  tables_created: string[];
  tables_modified: string[];
  variables: Record<string, any>;
  code: string;                           // Generated code (for replay)
  suggestions?: FailureSuggestion[];      // Alternative approaches on failure
}

interface FailureSuggestion {
  approach: string;
  reason: string;
}

// Artifact
// Note: Maps to constat/core/models.py ArtifactType enum
interface Artifact {
  id: string;
  name: string;
  type: ArtifactType;
  content: string;
  mime_type: string;
  is_binary: boolean;
  step_number: number;
  created_at: string;
  metadata?: Record<string, any>;
}

type ArtifactType =
  // Code and execution artifacts
  | "code" | "output" | "error"
  // Data artifacts
  | "table" | "json"
  // Rich content artifacts
  | "html" | "markdown" | "text"
  // Chart/visualization artifacts
  | "chart" | "plotly"
  // Image artifacts
  | "svg" | "png" | "jpeg"
  // Diagram artifacts
  | "mermaid" | "graphviz" | "diagram"
  // Interactive artifacts
  | "react" | "javascript";

// Fact (auditable mode) - displayed in FactTableViewer artifact
interface Fact {
  id: string;
  name: string;
  description: string;
  value: any;
  source: "cache" | "database" | "document" | "api" | "llm" | "derived" | "user";
  confidence?: number;
  cached: boolean;
  query?: string;
  resolved_at: string;
}

// Fact actions (for FactTableViewer)
interface FactActions {
  persist: (factId: string) => Promise<void>;    // Cache fact for future use
  forget: (factId: string) => Promise<void>;     // Invalidate cached fact
  edit: (factId: string, newValue: any) => Promise<void>;  // Manual override
  viewProvenance: (factId: string) => void;      // Show derivation details
}

// Entity (for EntityViewer)
// Note: Maps to constat/discovery/models.py Entity and EntityType
interface Entity {
  id: string;
  name: string;
  type: EntityType;
  metadata: Record<string, any>;
  created_at: string;
  mention_count?: number;        // How often referenced in session
  related_chunks?: string[];     // Document chunks mentioning this entity
}

type EntityType =
  // Schema entities
  | "table" | "column"
  // Semantic entities
  | "concept" | "business_term"
  // API entities (from API catalog)
  | "api_endpoint" | "api_field" | "api_schema";

// Learning (for LearningViewer)
// Note: Maps to constat/storage/learnings.py LearningStore structure
interface Learning {
  id: string;
  content: string;               // The correction text
  category: LearningCategory;
  source: LearningSource;
  context?: Record<string, any>; // Original context when captured
  applied_count: number;         // Times this learning was applied
  promoted_to?: string;          // Rule ID if promoted
  created_at: string;
}

// Matches constat/storage/learnings.py LearningCategory
type LearningCategory =
  | "user_correction"  // Explicit user corrections
  | "api_error"        // API-related errors/fixes
  | "codegen_error"    // Code generation errors/fixes
  | "nl_correction";   // Natural language detected corrections

// Matches constat/storage/learnings.py LearningSource
type LearningSource =
  | "auto_capture"      // Automatically captured from errors
  | "explicit_command"  // Via /correct or /remember command
  | "nl_detection";     // Detected from natural language

// Compacted rules (promoted from learnings)
interface Rule {
  id: string;
  category: LearningCategory;
  summary: string;               // Generalized rule text
  confidence: number;            // 0.0-1.0
  source_learnings: string[];    // IDs of learnings that formed this rule
  tags: string[];
  applied_count: number;
  created_at: string;
}

// Proof Tree Node (for ProofTreeRenderer - inline in conversation)
// Note: Maps to constat/proof_tree.py ProofNode dataclass
interface ProofNode {
  name: string;
  description: string;
  status: NodeStatus;
  value?: any;
  source: string;                // "cache", "database", "config", "llm", "derived", "user"
  confidence: number;            // 0.0-1.0
  children: ProofNode[];
  query?: string;                // SQL query, code snippet, or other context
  error?: string;
  // Additional fields from actual implementation
  result_summary?: string;       // Brief summary for intermediate display
  depth: number;                 // Depth in tree for indentation
  all_dependencies: string[];    // All dependencies (may differ from visual tree)
  visual_parent?: string;        // Parent name shown in tree
}

type NodeStatus = "pending" | "resolving" | "resolved" | "failed" | "cached";

// DFD Node (for DFDRenderer - inline in conversation)
interface DFDNode {
  id: string;
  label: string;
  type: "source" | "transform" | "output" | "store";
  metadata?: Record<string, any>;
}

interface DFDEdge {
  from: string;
  to: string;
  label?: string;
}

interface DataFlowDiagram {
  nodes: DFDNode[];
  edges: DFDEdge[];
}

// Uploaded File (for /add command - browser upload)
interface UploadedFile {
  id: string;
  filename: string;
  file_uri: string;              // file:// URI for use in queries
  size_bytes: number;
  content_type: string;
  uploaded_at: string;
}

// File Reference (for /file command - wraps session.add_file())
interface FileReference {
  name: string;
  uri: string;                   // URL or file:// path
  has_auth: boolean;             // Auth provided (not exposed for security)
  description?: string;
  added_at: string;
}

// Session Database (for /database command)
interface SessionDatabase {
  name: string;
  type: "sqlalchemy" | "mongodb" | "elasticsearch" | "duckdb" | "sqlite";
  dialect?: string;              // postgresql, mysql, etc. (for sqlalchemy)
  description?: string;
  connected: boolean;
  table_count?: number;
  added_at: string;
  is_dynamic: boolean;           // true if added via /connect, false if from config
  file_id?: string;              // If backed by uploaded file
  options?: {
    schema?: string;
    readonly?: boolean;
    [key: string]: any;
  };
}

// WebSocket Events
// Note: These map to StepEvent from constat/session.py, transformed for WebSocket transport
// The server adapter converts internal StepEvent.event_type to these WebSocket event types

type WSEvent =
  // Planning phase
  | { type: "planning_start"; data: Record<string, any> }
  | { type: "plan_generated"; plan: Plan }
  | { type: "clarification_needed"; data: { question: string; options?: string[] } }

  // Step lifecycle (maps to StepEvent event_types: step_start, generating, executing, step_complete, step_error, step_failed)
  | { type: "step_start"; step_number: number; goal: string }
  | { type: "generating"; step_number: number; data: Record<string, any> }
  | { type: "executing"; step_number: number; data: Record<string, any> }
  | { type: "step_complete"; step_number: number; result: StepResult }
  | { type: "step_error"; step_number: number; error: string; data: Record<string, any> }
  | { type: "step_failed"; step_number: number; error: string }

  // SQL-specific events (for SQL steps)
  | { type: "sql_generating"; step_number: number; data: Record<string, any> }
  | { type: "sql_executing"; step_number: number; data: Record<string, any> }
  | { type: "sql_error"; step_number: number; error: string }

  // Progress and display
  | { type: "progress"; step_number: number; message: string }
  | { type: "quick_display"; data: { content: string; format?: string } }

  // Proof Tree (auditable mode) - fact resolver events
  | { type: "proof_tree_created"; root: ProofNode }
  | { type: "proof_node_resolving"; node_path: string[]; name: string }
  | { type: "proof_node_resolved"; node_path: string[]; value: any; source: string; confidence?: number }
  | { type: "proof_node_failed"; node_path: string[]; error: string }
  | { type: "proof_node_cached"; node_path: string[]; value: any }
  | { type: "proof_node_added"; parent_path: string[]; node: ProofNode }

  // Data extraction events
  | { type: "facts_extracted"; data: { facts: Fact[] } }
  | { type: "correction_saved"; data: { correction: string } }

  // Artifacts & Entities
  | { type: "artifact_created"; artifact: Artifact }
  | { type: "entity_extracted"; entity: Entity }

  // Execution control
  | { type: "execution_cancelled"; data: Record<string, any> }
  | { type: "execution_complete"; success: boolean; output: string }
  | { type: "execution_error"; error: string }

  // Session
  | { type: "session_state_changed"; state: SessionState };
```

---

## Configuration

### Server Configuration (config.yaml addition)

```yaml
server:
  host: "127.0.0.1"
  port: 8000
  cors_origins:
    - "http://localhost:5173"  # Vite dev server
    - "http://localhost:3000"
  session_timeout_minutes: 60
  max_concurrent_sessions: 10
  require_plan_approval: false  # Auto-approve plans

ui:
  theme: "system"  # light | dark | system
  default_mode: "exploratory"
  show_code_by_default: false
  artifact_panel_width: 400
```

---

## Build & Deployment

### Development
```bash
# Start API server (with hot reload)
constat serve --reload --port 8000

# Start React dev server (separate terminal)
cd constat-ui && npm run dev
```

### Production
```bash
# Build React app
cd constat-ui && npm run build

# Serve with embedded UI
constat serve --port 8000 --static-dir ./constat-ui/dist
```

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
RUN cd constat-ui && npm ci && npm run build
EXPOSE 8000
CMD ["constat", "serve", "--host", "0.0.0.0", "--static-dir", "./constat-ui/dist"]
```

---

## Implementation Notes

### Session Event Adaptation

The existing `Session` class emits `StepEvent` objects with these actual event types (from `session.py`):

| Internal Event Type | WebSocket Mapping | Notes |
|---------------------|-------------------|-------|
| `step_start` | `step_start` | Step execution begins |
| `generating` | `generating` | Code generation in progress |
| `executing` | `executing` | Code execution in progress |
| `step_complete` | `step_complete` | Step finished successfully |
| `step_error` | `step_error` | Recoverable error (will retry) |
| `step_failed` | `step_failed` | Step failed after retries |
| `sql_generating` | `sql_generating` | SQL-specific generation |
| `sql_executing` | `sql_executing` | SQL execution |
| `sql_error` | `sql_error` | SQL error |
| `planning_start` | `planning_start` | Planning phase begins |
| `progress` | `progress` | General progress message |
| `quick_display` | `quick_display` | Fast inline display |
| `facts_extracted` | `facts_extracted` | Facts discovered |
| `correction_saved` | `correction_saved` | Learning captured |
| `execution_cancelled` | `execution_cancelled` | User cancelled |
| `clarification_needed` | `clarification_needed` | Need user input |

The server adapter layer should:
1. Register as an event handler via `session.add_event_handler()`
2. Transform `StepEvent` to WebSocket JSON format
3. Broadcast to connected clients

### Key Session Methods to Expose

| Method | API Endpoint | Notes |
|--------|--------------|-------|
| `Session.solve(problem)` | `POST /api/sessions/{id}/query` | Main entry point |
| `Session.follow_up(question)` | `POST /api/sessions/{id}/query` | With `is_followup: true` |
| `Session.add_file(name, uri, auth, description)` | `POST /api/sessions/{id}/file-refs` | Wraps existing method |
| `Session.add_database(name, type, uri, description)` | `POST /api/sessions/{id}/databases` | Wraps existing method |
| `Session.set_approval_callback(cb)` | WebSocket approve/reject | Hook for manual plan approval |
| `Session.set_clarification_callback(cb)` | WebSocket clarification | Hook for user clarifications |
| `Session.cancel()` | `POST /api/sessions/{id}/cancel` | Cancel running execution |
| `Session.get_context_stats()` | Include in session details | Token usage info |

### Proof Tree WebSocket Events

For auditable mode, the FactResolver emits events that are forwarded through `_handle_fact_resolver_event()`. The server should:

1. Subscribe to these events when session is in auditable mode
2. Transform `ProofNode` updates to incremental WebSocket events
3. Support full tree retrieval via `GET /api/sessions/{id}/proof-tree`

---

## Future Enhancements

1. **Multi-user support**: Authentication, user-scoped sessions
2. **Collaborative sessions**: Multiple users viewing same session
3. **Mobile responsive**: Adaptive layout for tablets/phones
4. **Offline mode**: PWA with cached sessions
5. **Plugins**: Custom artifact renderers
6. **Theming**: Customizable color schemes
7. **Export**: PDF/HTML report generation
8. **Notifications**: Browser notifications for long-running queries