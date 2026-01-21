# React UI Implementation Plan

## Overview

This plan outlines the implementation of a React-based web UI for Constat, including an API server mode to expose the existing functionality via HTTP/WebSocket.

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

**Technology**: FastAPI with uvicorn

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

**Global Commands**:
- `/tables` - View session tables
- `/code` - Show recent generated code
- `/facts` - View resolved facts
- `/learnings` - View/manage learnings
- `/schema` - Browse database schemas
- `/history` - Session history
- `/remember <note>` - Add learning
- `/mode` - Switch execution mode
- `/export` - Export session data
- `/add` - Upload file from browser (opens file picker, uploads as data URI → stored as temp file → returns file:// URI) *[new for web UI]*
- `/file` - Add file reference (wraps existing REPL command `session.add_file()` - for URLs or local paths with optional auth)
- `/database` - Add database connection (wraps existing REPL command `session.add_database()` - opens dialog for name, type, URI)

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
   - Filter by type (table, column, concept, business_term)
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
interface Plan {
  id: string;
  problem: string;
  steps: Step[];
  reasoning: string;
  created_at: string;
  status: "pending_approval" | "approved" | "executing" | "completed" | "failed";
}

interface Step {
  number: number;
  goal: string;
  task_type: string;
  depends_on: number[];
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  code?: string;
  result?: StepResult;
}

interface StepResult {
  success: boolean;
  stdout?: string;
  error?: string;
  tables_created: string[];
  variables: Record<string, any>;
  execution_time_ms: number;
  attempts: number;
}

// Artifact
interface Artifact {
  id: string;
  name: string;
  type: "CODE" | "OUTPUT" | "HTML" | "CHART" | "PLOTLY" | "TABLE" | "FACT_TABLE" | "SVG" | "PNG" | "MARKDOWN";
  content: string;
  mime_type: string;
  is_binary: boolean;
  step_number: number;
  created_at: string;
  metadata?: Record<string, any>;
}

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
interface Entity {
  id: string;
  name: string;
  type: "table" | "column" | "concept" | "business_term";
  metadata: Record<string, any>;
  created_at: string;
  mention_count?: number;        // How often referenced in session
  related_chunks?: string[];     // Document chunks mentioning this entity
}

// Learning (for LearningViewer)
interface Learning {
  id: string;
  content: string;
  category: "business_rule" | "correction" | "code_pattern" | "schema_mapping";
  source: "user_explicit" | "auto_detected" | "compaction";
  is_global: boolean;            // Applies to all sessions or just current
  session_id?: string;           // If session-specific
  created_at: string;
  last_used_at?: string;
}

// Proof Tree Node (for ProofTreeRenderer - inline in conversation)
interface ProofNode {
  name: string;
  description: string;
  status: "pending" | "resolving" | "resolved" | "failed" | "cached";
  value?: any;
  source: "cache" | "database" | "document" | "api" | "llm" | "derived" | "user";
  confidence?: number;
  children: ProofNode[];
  query?: string;
  error?: string;
}

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
type WSEvent =
  // Plan & Steps
  | { type: "plan_generated"; plan: Plan }
  | { type: "step_started"; step_number: number; goal: string }
  | { type: "step_progress"; step_number: number; message: string }
  | { type: "step_completed"; step_number: number; result: StepResult }
  | { type: "step_failed"; step_number: number; error: string; retry_count: number }
  // Proof Tree (auditable mode)
  | { type: "proof_tree_created"; root: ProofNode }
  | { type: "proof_node_resolving"; node_path: string[]; name: string }
  | { type: "proof_node_resolved"; node_path: string[]; value: any; source: string; confidence?: number }
  | { type: "proof_node_failed"; node_path: string[]; error: string }
  | { type: "proof_node_cached"; node_path: string[]; value: any }
  | { type: "proof_node_added"; parent_path: string[]; node: ProofNode }
  // Artifacts & Entities
  | { type: "artifact_created"; artifact: Artifact }
  | { type: "entity_extracted"; entity: Entity }
  // Execution
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

## Future Enhancements

1. **Multi-user support**: Authentication, user-scoped sessions
2. **Collaborative sessions**: Multiple users viewing same session
3. **Mobile responsive**: Adaptive layout for tablets/phones
4. **Offline mode**: PWA with cached sessions
5. **Plugins**: Custom artifact renderers
6. **Theming**: Customizable color schemes
7. **Export**: PDF/HTML report generation
8. **Notifications**: Browser notifications for long-running queries