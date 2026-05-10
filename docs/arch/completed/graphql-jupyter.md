# GraphQL Migration: Jupyter Client

## Goal
Migrate `constat-jupyter` to be completely GraphQL-driven. No REST. No WebSockets. All operations use GraphQL queries, mutations, and HTTP-SSE or async polling for execution events.

## Constraint: No WebSockets
Jupyter notebook environments (JupyterLab, VS Code, Colab, Kaggle) have inconsistent WebSocket support. The Jupyter client MUST NOT use WebSockets for any purpose — not even GraphQL subscriptions over `graphql-ws`. Instead, use one of:
- **HTTP SSE (Server-Sent Events)** — server pushes events over a long-lived HTTP GET
- **Polling** — client polls a query at intervals
- **GraphQL `@stream`/`@defer`** — if Strawberry supports it

This is the key difference from `graphql.md` (the UI migration), which uses `graphql-ws` WebSocket subscriptions.

---

## Current State

### Jupyter Client Architecture
| File | Lines | Purpose |
|------|-------|---------|
| `client.py` | 1170 | `ConstatClient` (HTTP) + `Session` (HTTP+WS) |
| `widgets.py` | ~200 | ipywidgets for plan approval, clarification |
| `magic.py` | ~150 | `%constat` / `%%constat` Jupyter magics |
| `progress.py` | ~100 | Progress bars (widget + print) |
| `entity_cache.py` | ~80 | Entity state with JSON patch |
| `models.py` | ~60 | `SolveResult`, `Artifact`, `StepInfo`, `ConstatError` |
| `config.py` | ~50 | Server URL + token resolution |

### Current Transport Usage

#### REST Endpoints Used (→ GraphQL queries/mutations)
```
POST   /api/sessions                           → mutation createSession
GET    /api/sessions                           → query sessions
GET    /api/sessions/{id}                      → query session
DELETE /api/sessions/{id}                      → mutation deleteSession
POST   /api/sessions/{id}/query                → mutation submitQuery
POST   /api/sessions/{id}/cancel               → mutation cancelExecution
GET    /api/sessions/{id}/tables               → query tables
GET    /api/sessions/{id}/tables/{n}/download  → REST KEPT (binary)
GET    /api/sessions/{id}/artifacts            → query artifacts
GET    /api/sessions/{id}/artifacts/{id}       → query artifact
GET    /api/sessions/{id}/sources              → query dataSources
GET    /api/sessions/{id}/databases            → query databases
GET    /api/sessions/{id}/glossary             → query glossary (already GQL)
GET    /api/sessions/{id}/glossary/{name}      → query glossaryTerm
POST   /api/sessions/{id}/glossary             → mutation createGlossaryTerm
DELETE /api/sessions/{id}/glossary/{name}      → mutation deleteGlossaryTerm
POST   /api/sessions/{id}/glossary/{n}/refine  → mutation refineGlossaryTerm
POST   /api/sessions/{id}/glossary/generate    → mutation generateGlossary
GET    /api/sessions/{id}/facts                → query facts
POST   /api/sessions/{id}/facts                → mutation addFact
POST   /api/sessions/{id}/facts/{n}/forget     → mutation forgetFact
POST   /api/sessions/{id}/domains              → mutation setActiveDomains
POST   /api/sessions/{id}/reset-context        → mutation resetContext
GET    /api/sessions/{id}/prompt-context       → query promptContext
GET    /api/sessions/{id}/plan                 → query executionPlan
GET    /api/sessions/{id}/steps                → query steps
GET    /api/sessions/{id}/inference-codes      → query inferenceCodes
GET    /api/sessions/{id}/ddl                  → query sessionDDL
GET    /api/sessions/{id}/output               → query executionOutput
GET    /api/sessions/{id}/scratchpad           → query scratchpad
GET    /api/sessions/{id}/proof-tree           → query proofTree
GET    /api/sessions/{id}/entities             → query entities
GET    /api/sessions/{id}/messages             → query messages
GET    /api/sessions/{id}/files                → query files
POST   /api/sessions/{id}/files                → mutation uploadFile
DELETE /api/sessions/{id}/files/{id}           → mutation deleteFile
POST   /api/sessions/{id}/documents/upload     → mutation uploadDocuments
POST   /api/sessions/{id}/documents/add-uri    → mutation addDocumentUri
POST   /api/sessions/{id}/databases            → mutation addDatabase
DELETE /api/sessions/{id}/databases/{name}     → mutation removeDatabase
POST   /api/sessions/{id}/apis                 → mutation addApi
DELETE /api/sessions/{id}/apis/{name}          → mutation removeApi
POST   /api/sessions/{id}/feedback/flag        → mutation flagAnswer
GET    /api/sessions/{id}/tables/{n}/star      → mutation toggleTableStar
DELETE /api/sessions/{id}/tables/{name}        → mutation deleteTable
POST   /api/sessions/{id}/artifacts/{id}/star  → mutation toggleArtifactStar
DELETE /api/sessions/{id}/artifacts/{id}       → mutation deleteArtifact
GET    /api/sessions/{id}/agents               → query agents
GET    /api/sessions/{id}/agents/{name}        → query agent
PUT    /api/sessions/{id}/agents/current       → mutation setActiveAgent
POST   /api/sessions/{id}/agents               → mutation createAgent
PUT    /api/sessions/{id}/agents/{name}        → mutation editAgent
DELETE /api/sessions/{id}/agents/{name}        → mutation deleteAgent
POST   /api/sessions/{id}/agents/draft         → mutation draftAgent
GET    /api/sessions/{id}/tests/domains        → query testableDomains
GET    /api/sessions/{id}/tests/{d}/questions  → query goldenQuestions
POST   /api/sessions/{id}/tests/{d}/questions  → mutation createGoldenQuestion
PUT    /api/sessions/{id}/tests/{d}/questions/{i} → mutation updateGoldenQuestion
DELETE /api/sessions/{id}/tests/{d}/questions/{i} → mutation deleteGoldenQuestion
POST   /api/sessions/{id}/tests/run            → REST KEPT (SSE streaming)
GET    /api/domains                            → query domains
GET    /api/skills                             → query skills
GET    /api/skills/{name}                      → query skill
POST   /api/skills                             → mutation createSkill
PUT    /api/skills/{name}                      → mutation editSkill
DELETE /api/skills/{name}                      → mutation deleteSkill
POST   /api/skills/draft                       → mutation draftSkill
GET    /api/skills/{name}/download             → REST KEPT (binary)
GET    /api/schema                             → query databaseSchema
GET    /api/schema/databases/{db}/tables/{t}   → query tableSchema
GET    /api/schema/search                      → query schemaSearch
GET    /api/learnings                          → query learnings
POST   /api/learnings/compact                  → mutation compactLearnings
POST   /api/rules                              → mutation createRule
PUT    /api/rules/{id}                         → mutation updateRule
DELETE /api/rules/{id}                         → mutation deleteRule
```

#### WebSocket Actions (→ GraphQL mutations)
```
action: approve                               → mutation approvePlan
action: reject                                → mutation approvePlan(approved: false)
action: clarify                               → mutation answerClarification (NEW)
action: skip_clarification                    → mutation skipClarification (NEW)
action: entity_seed                           → query entityState (NEW) or param on subscription
action: replan_from (edit/delete/redo)        → mutation replanFrom
action: edit_objective                        → mutation editObjective
action: delete_objective                      → mutation deleteObjective
```

#### WebSocket Events (→ SSE stream or polling)
```
plan_ready                                    → SSE event or poll executionStatus
step_event / step_complete / step_error       → SSE event or poll executionStatus
query_complete                                → SSE event or poll executionStatus
query_error / query_cancelled                 → SSE event or poll executionStatus
clarification_needed                          → SSE event or poll executionStatus
entity_state / entity_patch                   → SSE event or poll entities
planning_start                                → SSE event or poll executionStatus
```

---

## Execution Event Transport: SSE vs Polling

### Recommended: GraphQL over HTTP + SSE for Subscriptions

Use the [GraphQL over SSE](https://the-guild.dev/graphql/sse) pattern:

```
GET /api/graphql/stream?query=subscription{queryExecution(sessionId:"...")}
Accept: text/event-stream
Authorization: Bearer <token>
```

Server sends SSE events:
```
event: next
data: {"data":{"queryExecution":{"__typename":"PlanReady","plan":{...}}}}

event: next
data: {"data":{"queryExecution":{"__typename":"StepEvent","stepNumber":1,...}}}

event: complete
data:
```

**Why SSE over polling:**
- Lower latency (real-time push, no polling interval)
- Works in all Jupyter environments (plain HTTP)
- Lower server load than polling
- Clean completion signal (`event: complete`)
- `httpx` supports SSE via `stream("GET", ...)` + `iter_lines()`

### Server-Side Implementation

Add an SSE subscription endpoint alongside the existing `graphql-ws` WebSocket endpoint:

```python
# constat/server/graphql/sse.py
@router.get("/api/graphql/stream")
async def graphql_sse(request: Request, query: str, variables: str = "{}"):
    """Execute a GraphQL subscription over SSE (no WebSocket required)."""
    parsed_vars = json.loads(variables)
    # Execute subscription, yield SSE events
    async def event_generator():
        async for result in schema.subscribe(query, variable_values=parsed_vars, context_value=ctx):
            yield {"event": "next", "data": json.dumps({"data": result.data})}
        yield {"event": "complete", "data": ""}
    return EventSourceResponse(event_generator())
```

### Client-Side Implementation

```python
# constat-jupyter client — replaces websockets.connect()
async def _sse_event_loop(self, session_id: str, auto_approve: bool, timeout: float) -> SolveResult:
    query = 'subscription { queryExecution(sessionId: "%s") { __typename ... } }' % session_id
    url = f"{self._client._base_url}/api/graphql/stream"
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url, params={"query": query}, headers=self._headers(), timeout=timeout) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    # Map GraphQL event types to existing handler logic
                    ...
```

---

## Exceptions: Endpoints That Stay REST

| Endpoint | Reason |
|----------|--------|
| `GET /api/sessions/{id}/tables/{name}/download` | Binary Parquet/CSV download |
| `GET /api/skills/{name}/download` | Binary file download |
| `POST /api/sessions/{id}/tests/run` | SSE streaming (already non-WS) |
| `GET /api/sessions/{id}/download-code` | Text file download |
| `GET /api/sessions/{id}/download-inference-code` | Text file download |

---

## New Server-Side Requirements

### 1. SSE Subscription Endpoint
New endpoint: `GET /api/graphql/stream` — executes GraphQL subscriptions over SSE.

Must support:
- All existing subscriptions (`queryExecution`, `glossaryChanged`, `entityUpdates`)
- Auth via Bearer token header
- Clean shutdown on client disconnect
- Event format: `event: next\ndata: {json}\n\n`

### 2. New Mutations (not in graphql.md)

These WS actions used by the Jupyter client need dedicated mutations:

```graphql
# Clarification handling (currently WS-only actions)
mutation answerClarification(sessionId: String!, answers: JSON!) → Boolean
mutation skipClarification(sessionId: String!) → Boolean

# Entity seed (currently WS-only action)  
query entityState(sessionId: String!, sinceVersion: Int) → EntityStateType
```

### 3. Skills & Agents CRUD Mutations (not yet in GraphQL)

The Jupyter client uses skill/agent CRUD that may not have GraphQL equivalents yet:

```graphql
mutation createSkill(name: String!, content: String!) → SkillType
mutation editSkill(name: String!, content: String!) → SkillType
mutation deleteSkill(name: String!) → Boolean
mutation draftSkill(name: String!, description: String!) → SkillType

mutation createAgent(sessionId: String!, name: String!, content: String!) → AgentType
mutation editAgent(sessionId: String!, name: String!, content: String!) → AgentType
mutation deleteAgent(sessionId: String!, name: String!) → Boolean
mutation draftAgent(sessionId: String!, name: String!, description: String!) → AgentType
mutation setActiveAgent(sessionId: String!, name: String!) → AgentType
query agents(sessionId: String!) → [AgentType]
query agent(sessionId: String!, name: String!) → AgentType
```

---

## Implementation Phases

### Phase 1: SSE Infrastructure
**Scope:** SSE subscription endpoint + client-side SSE consumer.

**Work:**
- Add `GET /api/graphql/stream` endpoint (SSE transport for subscriptions)
- Add `sse-starlette` dependency (or use Starlette's `StreamingResponse`)
- Client: create `_sse_event_loop()` replacing `_ws_event_loop()`
- Client: create `_execute_async_graphql()` replacing `_execute_async()`
- Verify SSE works in JupyterLab, VS Code, and plain Python

**Tests:**
- Unit: SSE endpoint streams events for a mock subscription
- Integration: submit query via mutation → receive events via SSE → complete
- Notebook: manual test in JupyterLab

**Files:**
```
constat/server/graphql/sse.py                  (new)
constat/server/app.py                          (modify — mount SSE route)
constat-jupyter/constat_jupyter/graphql.py      (new — SSE client + GQL operations)
tests/test_graphql_sse.py                      (new)
```

---

### Phase 2: Session + Execution Mutations
**Scope:** Migrate `ConstatClient` session CRUD + `Session.solve/follow_up` to GraphQL.

**Work:**
- Client: `ConstatClient.create_session()` → `mutation createSession`
- Client: `ConstatClient.list_sessions()` → `query sessions`
- Client: `ConstatClient.get_session()` → `query session`
- Client: `ConstatClient.delete_session()` → `mutation deleteSession`
- Client: `Session.solve()` → `mutation submitQuery` + SSE subscription
- Client: `Session.cancel()` → `mutation cancelExecution`
- Server: `mutation answerClarification` + `mutation skipClarification` (new)
- Client: plan approval → `mutation approvePlan` (exists)
- Client: clarification → `mutation answerClarification` (new)

**Tests:**
- Unit: each client method sends correct GraphQL operation
- Integration: create session → solve query → receive SSE events → result

**Done when:** `solve()`, `follow_up()`, `replay()`, `reason_chain()` work via GraphQL+SSE. No `websockets` import in execution path.

---

### Phase 3: Read-Only Queries
**Scope:** Migrate all `Session` read methods to GraphQL queries.

**Work:**
- `Session.tables()` → `query tables`
- `Session.artifacts()` / `Session.artifact()` → `query artifacts` / `query artifact`
- `Session.facts()` → `query facts`
- `Session.entities()` → `query entities`
- `Session.steps()` → `query steps`
- `Session.plan()` → `query executionPlan`
- `Session.scratchpad()` → `query scratchpad`
- `Session.ddl()` → `query sessionDDL`
- `Session.output()` → `query executionOutput`
- `Session.proof_tree()` → `query proofTree`
- `Session.messages()` → `query messages`
- `Session.sources()` → `query dataSources`
- `Session.databases()` → `query databases`
- `Session.context()` → `query promptContext`
- `Session.status` → `query session`
- `Session.inference_codes()` → `query inferenceCodes`
- `Session.glossary()` → already GraphQL (or use existing GQL query)

**Tests:**
- Unit: each method sends correct query, parses response correctly
- Integration: after solving a query, verify all read methods return expected data via GraphQL

**Done when:** All `Session` read methods use GraphQL. No REST `_http.get()` calls for these operations.

---

### Phase 4: Write Mutations
**Scope:** Migrate all `Session` write methods to GraphQL mutations.

**Work:**
- Facts: `remember()` → `mutation addFact`, `forget()` → `mutation forgetFact`
- Glossary: `define()` → `mutation createGlossaryTerm`, `undefine()` → `mutation deleteGlossaryTerm`, `refine()`, `generate_glossary()`
- Tables: `star_table()` → `mutation toggleTableStar`, `delete_table()` → `mutation deleteTable`
- Artifacts: `star_artifact()` → `mutation toggleArtifactStar`, `delete_artifact()` → `mutation deleteArtifact`
- Data sources: `add_database()`, `remove_database()`, `add_api()`, `remove_api()`, `add_document()`, `upload_document()`, `upload_file()`, `delete_file()`
- Domains: `set_domains()` → `mutation setActiveDomains`
- Session: `reset()` → `mutation resetContext`
- Feedback: `correct()` → `mutation flagAnswer`

**Tests:**
- Unit: each mutation sends correct operation with correct variables
- Integration: add database → list databases → remove → verify gone

**Done when:** All write operations use GraphQL mutations. Only binary download endpoints use REST.

---

### Phase 5: WS Actions → Mutations
**Scope:** Migrate `_ws_action_async` methods to GraphQL mutations + SSE.

**Work:**
- `step_edit()` → `mutation replanFrom(mode: "edit")` + SSE
- `step_delete()` → `mutation replanFrom(mode: "delete")` + SSE
- `step_redo()` → `mutation replanFrom(mode: "redo")` + SSE
- `edit_objective()` → `mutation editObjective` + SSE
- `delete_objective()` → `mutation deleteObjective` + SSE
- Entity cache: `entity_seed` → `query entityState(sinceVersion)` before subscribing
- Remove `_ws_action_async()`, `_ws_event_loop()`, `_ws_url()`, `_ws_headers()`, `_ws_send_entity_seed()`

**Tests:**
- Unit: each method sends correct mutation + subscribes to SSE
- Integration: edit step → receive replan events via SSE → new result

**Done when:** `_ws_action_async` deleted. `websockets` removed from dependencies.

---

### Phase 6: ConstatClient Methods
**Scope:** Migrate `ConstatClient` non-session methods to GraphQL.

**Work:**
- `domains()` → `query domains`
- `skills()` / `skill_info()` → `query skills` / `query skill`
- `create_skill()`, `edit_skill()`, `delete_skill()`, `draft_skill()` → mutations
- `search_schema()` → `query schemaSearch`
- `learnings()` → `query learnings`
- `compact_learnings()` → `mutation compactLearnings`
- `add_rule()`, `edit_rule()`, `delete_rule()` → mutations

**Server:** Add any missing skill/agent CRUD mutations.

**Tests:**
- Unit: each ConstatClient method sends correct GraphQL operation
- Integration: create skill → list → edit → delete round-trip via GraphQL

**Done when:** `ConstatClient` uses GraphQL for all operations except binary downloads.

---

### Phase 7: Agent CRUD + Testing
**Scope:** Migrate `Session` agent and testing methods to GraphQL.

**Work:**
- Agents: `agents()`, `agent()`, `set_agent()`, `create_agent()`, `edit_agent()`, `delete_agent()`, `draft_agent()` → queries/mutations
- Testing: `test_domains()`, `test_questions()`, `create_test_question()`, `update_test_question()`, `delete_test_question()` → queries/mutations
- `run_tests()` stays REST (SSE streaming)

**Tests:**
- Unit: correct operations sent for each method
- Integration: create agent → set active → solve query → delete agent

**Done when:** All agent and testing operations use GraphQL.

---

### Phase 8: Cleanup + File Split
**Scope:** Remove all REST/WS code, split `client.py` (<1000 lines), update dependencies.

**Work:**
- Delete all `_http.get()`/`_http.post()` calls except binary downloads and SSE test runner
- Remove `websockets` from `pyproject.toml` dependencies
- Split `client.py` into:
  - `client.py` — `ConstatClient` (HTTP client, session factory, non-session GraphQL operations)
  - `session.py` — `Session` (session-scoped GraphQL operations, SSE execution)
  - `graphql.py` — GraphQL operation documents + SSE transport
- Update `__init__.py` exports
- Update `magic.py` imports
- Final audit: grep for `_http`, `websockets`, `/api/sessions/` REST paths

**Tests:**
- All existing tests pass with new module structure
- `websockets` not importable by any code path
- `client.py` < 500 lines, `session.py` < 500 lines

**Done when:** Zero WebSocket code. Zero REST calls (except 3-5 binary endpoints + SSE test runner). `websockets` removed from dependencies. All files < 1000 lines.

---

## Dependency Changes

### Remove
```
websockets>=11       → deleted (WS transport eliminated)
```

### Add
```
# None required — httpx already supports SSE via stream + iter_lines
```

### Keep
```
httpx>=0.24          (HTTP + SSE transport)
polars>=0.20         (DataFrame)
ipython>=7.0         (Magics)
jsonpatch>=1.32      (Entity state patches — still useful for entity delta queries)
ipywidgets>=8        (Optional: interactive widgets — unchanged)
itables>=2.0         (Optional: table display — unchanged)
```

---

## GraphQL Client Pattern

The Jupyter client uses `httpx` (already a dependency) for all GraphQL operations:

```python
# constat-jupyter/constat_jupyter/graphql.py

class GraphQLClient:
    """Thin GraphQL-over-HTTP client using httpx."""

    def __init__(self, base_url: str, token: str | None = None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30)
        self._sync_http = httpx.Client(base_url=base_url, headers=headers, timeout=30)

    def query(self, operation: str, variables: dict | None = None) -> dict:
        """Synchronous GraphQL query/mutation."""
        resp = self._sync_http.post("/api/graphql", json={"query": operation, "variables": variables or {}})
        resp.raise_for_status()
        result = resp.json()
        if result.get("errors"):
            raise ConstatError(result["errors"][0]["message"])
        return result["data"]

    async def subscribe_sse(self, operation: str, variables: dict | None = None) -> AsyncGenerator[dict, None]:
        """Subscribe to a GraphQL subscription via SSE."""
        params = {"query": operation, "variables": json.dumps(variables or {})}
        async with self._http.stream("GET", "/api/graphql/stream", params=params) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    yield event.get("data", {})
```

---

## Migration Summary

| Category | REST Calls | → GraphQL Ops | Transport |
|----------|-----------|---------------|-----------|
| Session CRUD | 4 | 2Q + 2M | HTTP POST to `/api/graphql` |
| Query execution | 2 + WS | 2M + SSE subscription | HTTP POST + SSE GET |
| WS actions (replan, objectives) | 5 WS actions | 5M + SSE | HTTP POST + SSE GET |
| WS approval/clarification | 4 WS actions | 4M | HTTP POST |
| Read-only session state | 17 | 17Q | HTTP POST |
| Write mutations (facts, glossary, etc.) | 15 | 15M | HTTP POST |
| Data source management | 10 | 10M | HTTP POST |
| Agent CRUD | 7 | 4Q + 3M | HTTP POST |
| Skills (ConstatClient) | 6 | 2Q + 4M | HTTP POST |
| Learnings/Rules | 5 | 1Q + 4M | HTTP POST |
| Schema/Domains | 4 | 4Q | HTTP POST |
| Testing | 5 + SSE | 2Q + 3M | HTTP POST (SSE stays) |
| Binary downloads | 5 | — | REST KEPT |
| **Total** | **~89 REST + WS** | **~32Q + 52M + SSE** | **Zero WebSockets** |
