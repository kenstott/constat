# GraphQL Migration Plan

## Goal
Move the UI to be completely GraphQL-driven. No REST.

## Current State

### Already GraphQL
| Area | Operations | Subscriptions |
|------|-----------|---------------|
| Glossary terms | query, create, update, delete, batch delete, batch status update | `glossaryChanged` (CRUD + generation progress) |
| Glossary generation | generate, draft definition/aliases/tags, refine, suggest taxonomy, rename | via `glossaryChanged` |
| Relationships | create, update, delete, approve | via `glossaryChanged` |

### Already WebSocket (needs GraphQL subscription replacement)
| Event | Current Transport |
|-------|-------------------|
| `step_complete`, `step_event` | WS |
| `plan_ready`, `approval_request` | WS |
| `query_complete`, `error` | WS |
| `clarification_request` | WS |
| `entity_update` | WS |
| `artifact_created`, `table_created` | WS |
| `source_ingest_start/complete/error` | WS |
| `autocomplete_response` | WS |

### WebSocket Actions (bidirectional, needs mutation replacement)
| Action | Data |
|--------|------|
| `approve` | — |
| `reject` | `{feedback}` |
| `cancel` | — |
| `replan_from` | `{step_number, mode, edited_goal}` |
| `edit_objective` | `{objective_index, new_text}` |
| `delete_objective` | `{objective_index}` |
| `autocomplete` | `{context, prefix, parent, request_id}` |
| `heartbeat` | `{since}` |

---

## Gaps: REST Endpoints Requiring GraphQL Migration

### 1. Session Management (9 endpoints)
```
POST   /sessions                              → mutation createSession
GET    /sessions                              → query sessions
GET    /sessions/{id}                         → query session(id)
DELETE /sessions/{id}                         → mutation deleteSession
POST   /sessions/{id}/public                  → mutation togglePublicSharing
GET    /sessions/{id}/shares                  → query sessionShares(id)
POST   /sessions/{id}/share                   → mutation shareSession
DELETE /sessions/{id}/share/{userId}          → mutation removeShare
POST   /sessions/{id}/reset-context           → mutation resetContext
```

### 2. Query Execution (4 endpoints → mutations + subscription)
```
POST   /sessions/{id}/query                   → mutation submitQuery
POST   /sessions/{id}/cancel                  → mutation cancelExecution
GET    /sessions/{id}/plan                    → query executionPlan(id)
POST   /sessions/{id}/plan/approve            → mutation approvePlan
```
Plus all WS events become a `subscription queryExecution(sessionId)`.

### 3. Domain Management (16 endpoints)
```
GET    /domains                               → query domains
POST   /domains                               → mutation createDomain
GET    /domains/tree                          → query domainTree
GET    /domains/{f}                           → query domain(filename)
GET    /domains/{f}/content                   → query domainContent(filename)
PUT    /domains/{f}/content                   → mutation updateDomainContent
PATCH  /domains/{f}                           → mutation updateDomain
DELETE /domains/{f}                           → mutation deleteDomain
POST   /domains/{f}/promote                   → mutation promoteDomain
GET    /domains/{f}/skills                    → query domainSkills(filename)
GET    /domains/{f}/agents                    → query domainAgents(filename)
GET    /domains/{f}/rules                     → query domainRules(filename)
POST   /domains/move-source                   → mutation moveDomainSource
POST   /domains/move-skill                    → mutation moveDomainSkill
POST   /domains/move-agent                    → mutation moveDomainAgent
POST   /domains/move-rule                     → mutation moveDomainRule
```
Session-scoped domain selection:
```
POST   /sessions/{id}/domains                 → mutation setActiveDomains
GET    /sessions/{id}/domains                 → query activeDomains(sessionId)
```

### 4. Tables (6 endpoints)
```
GET    /sessions/{id}/tables                  → query tables(sessionId)
GET    /sessions/{id}/tables/{name}           → query tableData(sessionId, name, page, pageSize)
DELETE /sessions/{id}/tables/{name}           → mutation deleteTable
GET    /sessions/{id}/tables/{name}/versions  → query tableVersions(sessionId, name)
GET    /sessions/{id}/tables/{name}/version/N → query tableVersionData(sessionId, name, version, page, pageSize)
POST   /sessions/{id}/tables/{name}/star      → mutation toggleTableStar
```

### 5. Artifacts (5 endpoints)
```
GET    /sessions/{id}/artifacts               → query artifacts(sessionId)
GET    /sessions/{id}/artifacts/{aid}         → query artifact(sessionId, artifactId)
DELETE /sessions/{id}/artifacts/{aid}         → mutation deleteArtifact
POST   /sessions/{id}/artifacts/{aid}/star    → mutation toggleArtifactStar
GET    /sessions/{id}/artifacts/{aid}/versions → query artifactVersions(sessionId, artifactId)
```

### 6. Facts (6 endpoints)
```
GET    /sessions/{id}/facts                   → query facts(sessionId)
POST   /sessions/{id}/facts                   → mutation addFact
POST   /sessions/{id}/facts/{name}            → mutation editFact
POST   /sessions/{id}/facts/{name}/persist    → mutation persistFact
POST   /sessions/{id}/facts/{name}/forget     → mutation forgetFact
POST   /sessions/{id}/facts/{name}/move       → mutation moveFact
```

### 7. Entities (2 endpoints)
```
GET    /sessions/{id}/entities                → query entities(sessionId, entityType)
POST   /sessions/{id}/entities/{eid}/glossary → mutation addEntityToGlossary
```

### 8. File Management (8 endpoints)
```
POST   /sessions/{id}/files                   → mutation uploadFile (multipart*)
POST   /sessions/{id}/files/data-uri          → mutation uploadFileDataUri
GET    /sessions/{id}/files                   → query files(sessionId)
GET    /sessions/{id}/files/{fid}             → REST KEPT (binary download)
DELETE /sessions/{id}/files/{fid}             → mutation deleteFile
POST   /sessions/{id}/file-refs               → mutation addFileRef
GET    /sessions/{id}/file-refs               → query fileRefs(sessionId)
DELETE /sessions/{id}/file-refs/{name}        → mutation deleteFileRef
```
*File upload: GraphQL multipart upload spec or keep REST for binary.

### 9. Documents (6 endpoints)
```
POST   /sessions/{id}/documents/upload        → mutation uploadDocuments (multipart*)
POST   /sessions/{id}/documents/add-uri       → mutation addDocumentUri
POST   /sessions/{id}/documents/add-email     → mutation addEmailSource
POST   /sessions/{id}/documents/refresh       → mutation refreshDocuments
PUT    /sessions/{id}/documents/{name}        → mutation updateDocument
GET    /sessions/{id}/document                → REST KEPT (binary/HTML content)
GET    /sessions/{id}/file                    → REST KEPT (binary file serving)
```
Plus `subscription documentIngestion(sessionId)` replacing WS events.

### 10. Database Management (7 endpoints)
```
POST   /sessions/{id}/databases               → mutation addDatabase
GET    /sessions/{id}/databases               → query databases(sessionId)
GET    /sessions/{id}/databases/{db}/tables/{t}/preview → query databaseTablePreview(...)
GET    /sessions/{id}/databases/{db}/tables/{t}/download → REST KEPT (binary download)
POST   /sessions/{id}/databases/{db}/test     → mutation testDatabase
DELETE /sessions/{id}/databases/{db}           → mutation removeDatabase
PUT    /sessions/{id}/databases/{name}        → mutation updateDatabase
```

### 11. API Management (2 endpoints)
```
POST   /sessions/{id}/apis                    → mutation addApi
DELETE /sessions/{id}/apis/{name}             → mutation removeApi
```

### 12. Data Sources (1 endpoint)
```
GET    /sessions/{id}/sources                 → query dataSources(sessionId)
```

### 13. Proof/Tracing (3 endpoints)
```
GET    /sessions/{id}/proof-tree              → query proofTree(sessionId)
POST   /sessions/{id}/proof-facts             → mutation saveProofFacts
GET    /sessions/{id}/proof-facts             → query proofFacts(sessionId)
```

### 14. Execution State (6 endpoints)
```
GET    /sessions/{id}/steps                   → query steps(sessionId)
GET    /sessions/{id}/inference-codes         → query inferenceCodes(sessionId)
GET    /sessions/{id}/scratchpad              → query scratchpad(sessionId)
GET    /sessions/{id}/ddl                     → query sessionDDL(sessionId)
GET    /sessions/{id}/output                  → query executionOutput(sessionId)
GET    /sessions/{id}/routing                 → query sessionRouting(sessionId)
```

### 15. User Preferences / Messages (5 endpoints)
```
GET    /sessions/{id}/messages                → query messages(sessionId)
POST   /sessions/{id}/messages                → mutation saveMessages
GET    /sessions/{id}/objectives              → query objectives(sessionId)
GET    /sessions/{id}/prompt-context          → query promptContext(sessionId)
PUT    /sessions/{id}/system-prompt           → mutation updateSystemPrompt
```

### 16. Learning/Rules (7 endpoints)
```
GET    /learnings                             → query learnings(category)
POST   /learnings/compact                     → mutation compactLearnings
DELETE /learnings/{id}                        → mutation deleteLearning
POST   /rules                                 → mutation createRule
PUT    /rules/{id}                            → mutation updateRule
DELETE /rules/{id}                            → mutation deleteRule
GET    /learnings/exemplars/simple            → REST KEPT (file download)
```

### 17. Skills & Agents (3 endpoints)
```
GET    /skills                                → query skills
GET    /skills/{name}                         → query skill(name)
POST   /agents/{id}/agents/{name}             → mutation activateAgent
```

### 18. Schema (2 endpoints)
```
GET    /schema/databases/{db}/tables          → query databaseSchema(dbName)
GET    /schema/apis/{api}                     → query apiSchema(apiName)
```

### 19. Authentication (4 endpoints)
```
POST   /auth/login                            → mutation login
POST   /auth/logout                           → mutation logout
POST   /auth/passkey/register                 → mutation registerPasskey
POST   /auth/passkey/authenticate             → mutation authenticatePasskey
```

### 20. User / Permissions (3 endpoints)
```
GET    /users/me/permissions                  → query myPermissions
GET    /users/{uid}/sources                   → query userSources(userId)
DELETE /users/{uid}/sources/{type}/{name}     → mutation removeUserSource
POST   /sessions/{id}/sources/{type}/{name}/promote → mutation promoteSource
```

### 21. Config (1 endpoint)
```
GET    /config                                → query config
```

### 22. Feedback (1 endpoint)
```
POST   /sessions/{id}/feedback                → mutation submitFeedback
```

### 23. Testing (3 endpoints)
```
GET    /sessions/{id}/tests/golden            → query goldenTests(sessionId)
POST   /sessions/{id}/tests/run               → mutation runTest
GET    /sessions/{id}/tests/results           → query testResults(sessionId)
```

### 24. Fine-Tuning (5 endpoints)
```
GET    /fine-tune/jobs                        → query fineTuneJobs
POST   /fine-tune/jobs                        → mutation startFineTuneJob
GET    /fine-tune/jobs/{modelId}              → query fineTuneJob(modelId)
POST   /fine-tune/jobs/{modelId}/cancel       → mutation cancelFineTuneJob
DELETE /fine-tune/jobs/{modelId}              → mutation deleteFineTuneJob
GET    /fine-tune/providers                   → query fineTuneProviders
```

### 25. Public/Shared Session (7 endpoints)
```
GET    /public/{id}                           → query publicSession(id)
GET    /public/{id}/messages                  → query publicMessages(id)
GET    /public/{id}/artifacts                 → query publicArtifacts(id)
GET    /public/{id}/tables                    → query publicTables(id)
GET    /public/{id}/tables/{name}             → query publicTableData(id, name, page, pageSize)
GET    /public/{id}/artifacts/{aid}           → query publicArtifact(id, artifactId)
GET    /public/{id}/proof-facts               → query publicProofFacts(id)
```

### 26. OAuth (2 endpoints)
```
GET    /oauth/email/providers                 → query emailOAuthProviders
POST   /oauth/email/callback                  → mutation emailOAuthCallback
```

---

## Exceptions: Endpoints That Stay REST

| Endpoint | Reason |
|----------|--------|
| `GET /sessions/{id}/files/{fid}` | Binary file download |
| `GET /sessions/{id}/document` | HTML/binary document content |
| `GET /sessions/{id}/file` | Binary file serving |
| `GET /sessions/{id}/databases/{db}/tables/{t}/download` | Binary CSV/Parquet download |
| `GET /learnings/exemplars/simple` | File download |

These 5 endpoints serve binary content — GraphQL is not suited for binary streams. Keep as REST behind a `/api/download/` prefix.

---

## Subscriptions Replacing WebSocket

| New Subscription | Replaces WS Events |
|-----------------|---------------------|
| `queryExecution(sessionId)` | `step_event`, `step_complete`, `plan_ready`, `approval_request`, `query_complete`, `error`, `clarification_request` |
| `entityUpdates(sessionId)` | `entity_update` |
| `artifactChanges(sessionId)` | `artifact_created` |
| `tableChanges(sessionId)` | `table_created` |
| `documentIngestion(sessionId)` | `source_ingest_start`, `source_ingest_complete`, `source_ingest_error` |
| `autocomplete(sessionId)` | `autocomplete_response` |
| `glossaryChanged(sessionId)` | Already exists |

WS bidirectional actions (`approve`, `reject`, `cancel`, `replan_from`, `edit_objective`, `delete_objective`) become mutations.
`autocomplete` request becomes a mutation (or query with `@defer`).
`heartbeat` is handled at the transport layer by the GraphQL WS protocol (`graphql-ws` ping/pong).

---

## Summary: Migration Scope

| Category | REST Endpoints | → GraphQL Ops | Notes |
|----------|---------------|---------------|-------|
| Already done (glossary) | 0 | 2Q + 17M + 1S | Complete |
| Session management | 9 | 2Q + 7M | |
| Query execution | 4 + WS | 1Q + 4M + 1S | Largest subscription |
| Domains | 18 | 5Q + 13M | |
| Tables | 6 | 3Q + 2M | |
| Artifacts | 5 | 3Q + 2M | |
| Facts | 6 | 1Q + 5M | |
| Entities | 2 | 1Q + 1M | |
| Files | 8 | 2Q + 5M | 1 stays REST |
| Documents | 6 | 0Q + 5M + 1S | 2 stay REST |
| Databases | 7 | 2Q + 4M | 1 stays REST |
| APIs | 2 | 0Q + 2M | |
| Data sources | 1 | 1Q | |
| Proof/tracing | 3 | 2Q + 1M | |
| Execution state | 6 | 6Q | Read-only |
| Messages/prefs | 5 | 3Q + 2M | |
| Learning/rules | 7 | 2Q + 4M | 1 stays REST |
| Skills/agents | 3 | 2Q + 1M | |
| Schema | 2 | 2Q | |
| Auth | 4 | 0Q + 4M | |
| Users/permissions | 4 | 2Q + 2M | |
| Config | 1 | 1Q | |
| Feedback | 1 | 0Q + 1M | |
| Testing | 3 | 2Q + 1M | |
| Fine-tuning | 6 | 3Q + 3M | |
| Public sessions | 7 | 5Q | Unauthenticated |
| OAuth | 2 | 1Q + 1M | |
| WS → subscriptions | 7 events | 6S | |
| WS → mutations | 7 actions | 7M | |
| **Total** | **~131 REST + WS** | **~52Q + 87M + 8S** | **5 stay REST (binary)** |

---

## Frontend Modernization

The GraphQL migration is also a frontend architecture reset. Current state is poor — Zustand stores mix data fetching with UI state, components call REST APIs directly, Apollo is configured but never wired into the React tree, TanStack Query is a dead dependency, and full-store re-renders degrade performance.

### Current Problems

| Problem | Where | Impact |
|---------|-------|--------|
| No `ApolloProvider` in tree | `main.tsx` — Apollo client exists but never provided | GraphQL hooks silently broken or bypassed |
| Dead `QueryClientProvider` | `main.tsx` — TanStack Query wired but zero `useQuery` calls | Dead dependency; confusing |
| Zustand stores do data fetching | `artifactStore` has 40+ fetch-then-setState actions | No deduplication, no cache, manual invalidation |
| Full-store re-renders | `useSessionStore()` returns entire store | Any state change re-renders all consumers |
| `useEffect` for data fetching | `ArtifactPanel`, `App.tsx` | Race conditions, no cleanup, N+1 fetches |
| Dual-source truth for glossary | `glossaryStore` (REST) + `useGlossaryData` hook (GraphQL) | Data divergence, manual mapping |
| Single `error` string per store | `artifactStore.error` covers 40+ operations | Can't identify which operation failed |
| 4+ boolean loading fields | `loading`, `factsLoading`, `sourcesLoading`, `learningsLoading` | Inconsistent UX, no per-item granularity |
| Prop drilling 5+ levels | `ConversationPanel` → `MessageBubble` → nested buttons | Hard to refactor, brittle |
| Manual cache invalidation | `updateTerm` → `get().fetchTerms()` | Stale data, extra requests |
| Inline API calls in components | `GlossaryPanel` calls `getGlossarySuggestions()` directly | Bypasses stores/hooks entirely |
| No auth token caching | `getToken()` called on every HTTP + WS request | Unnecessary async overhead |

### Target Architecture

```
<StrictMode>
  <BrowserRouter>
    <ApolloProvider>                    ← NEW: provides cache + subscriptions
      <AuthProvider>                    ← NEW: context for auth state + token
        <SessionProvider sessionId={}>  ← NEW: context for active session
          <App>
            <MainLayout>
              <ConversationPanel />
              <ArtifactPanel />
              <GlossaryPanel />
              <ProofPanel />
            </MainLayout>
          </App>
        </SessionProvider>
      </AuthProvider>
    </ApolloProvider>
  </BrowserRouter>
</StrictMode>
```

### Principle: Apollo Is The Only Store

**Delete Zustand entirely.** Server data uses queries/mutations/subscriptions. UI state uses Apollo reactive variables (`makeVar`) and local-only fields (`@client`).

| Concern | Before (Zustand) | After (Apollo) |
|---------|-------------------|----------------|
| Tables list | `artifactStore.tables` + `fetchTables()` | `useTables(sessionId)` → Apollo cache |
| Artifact content | `artifactStore.fetchArtifact()` + `set()` | `useArtifact(sessionId, id)` → cache-first |
| Facts | `artifactStore.facts` + `fetchFacts()` | `useFacts(sessionId)` → cache + subscription |
| Glossary terms | `glossaryStore.terms` + REST fallback | `useGlossary(sessionId)` → already Apollo |
| Entities | `artifactStore.entities` + `fetchEntities()` | `useEntities(sessionId)` → Apollo cache |
| Session metadata | `sessionStore.session` | `useSession(sessionId)` → Apollo cache |
| Messages | `sessionStore.messages` + debounced save | `useMessages(sessionId)` → Apollo + auto-persist mutation |
| Proof facts | `proofStore.facts` | `useProofFacts(sessionId)` → subscription-driven |
| Execution events | `sessionStore` WS handler | `useQueryExecution(sessionId)` → subscription |
| Panel open/closed | `uiStore.showArtifactPanel` | `showArtifactPanelVar` → reactive variable |
| Deep link state | `uiStore.activeDeepLink` | `activeDeepLinkVar` → reactive variable |
| Toast notifications | `toastStore` | `toastsVar` → reactive variable |
| Theme / preferences | `uiStore.theme` | `themeVar` → reactive variable + `localStorage` persist |

### Reactive Variables for UI State

Apollo reactive variables are read via `useReactiveVar()` — same hook pattern as queries, zero new concepts.

```typescript
// constat-ui/src/graphql/ui-state.ts
import { makeVar } from '@apollo/client'

// Panel visibility
export const showArtifactPanelVar = makeVar<boolean>(true)
export const showProofPanelVar = makeVar<boolean>(false)
export const showGlossaryPanelVar = makeVar<boolean>(false)

// Deep links
export const activeDeepLinkVar = makeVar<DeepLink | null>(null)

// Toast queue
export const toastsVar = makeVar<Toast[]>([])
export function addToast(toast: Omit<Toast, 'id'>) {
  toastsVar([...toastsVar(), { ...toast, id: crypto.randomUUID() }])
}
export function dismissToast(id: string) {
  toastsVar(toastsVar().filter(t => t.id !== id))
}

// Theme (persisted to localStorage)
const stored = localStorage.getItem('theme') as 'light' | 'dark' | null
export const themeVar = makeVar<'light' | 'dark'>(stored ?? 'light')
export function setTheme(theme: 'light' | 'dark') {
  themeVar(theme)
  localStorage.setItem('theme', theme)
}

// Expanded/collapsed accordion state
export const expandedSectionsVar = makeVar<Set<string>>(new Set())
export function toggleSection(id: string) {
  const s = new Set(expandedSectionsVar())
  s.has(id) ? s.delete(id) : s.add(id)
  expandedSectionsVar(s)
}
```

```typescript
// Component usage — identical hook pattern to data queries
function ArtifactPanel() {
  const showPanel = useReactiveVar(showArtifactPanelVar)
  const { artifacts, loading } = useArtifacts()

  if (!showPanel) return null
  if (loading) return <Skeleton />
  return <ArtifactList artifacts={artifacts} />
}
```

**Why this works better than Zustand:**
- One mental model: everything is Apollo (`useQuery`, `useMutation`, `useSubscription`, `useReactiveVar`)
- Reactive variables trigger re-renders only in components that read them (same granularity as Zustand selectors, without the selector boilerplate)
- UI state can be mixed into GraphQL queries via `@client` directive if needed (e.g., augmenting server data with local UI flags)
- No second state library to learn, configure, or debug
- `localStorage` persistence is a 2-line wrapper (see `setTheme` above)

### Local-Only Fields (optional, for mixed server + UI state)

When UI state is logically attached to a server entity, use `@client` fields:

```typescript
// Example: "isExpanded" is local-only, attached to each Table in cache
const typePolicies = {
  Table: {
    fields: {
      isExpanded: { read: () => false },  // default
    },
  },
}

// Query mixes server + local
const TABLES_QUERY = gql`
  query Tables($sessionId: String!) {
    tables(sessionId: $sessionId) {
      name
      rowCount
      starred
      isExpanded @client   # local-only, no server round-trip
    }
  }
`
```

### Contexts

#### `AuthContext`
```typescript
interface AuthContextValue {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  isAuthDisabled: boolean
  login: (email: string, password: string) => Promise<void>
  logout: () => Promise<void>
  permissions: UserPermissions
}
```
Wraps Firebase subscription. Provides token to Apollo link. Replaces `authStore` for data (keep `authStore` only if needed for non-React code).

#### `SessionContext`
```typescript
interface SessionContextValue {
  sessionId: string
  session: Session            // from useSession() query
  activeDomains: string[]
  setActiveDomains: (domains: string[]) => void
}
```
Set once on session selection. All child hooks read `sessionId` from context — no prop drilling.

### Custom Hooks (one per data domain)

Each hook encapsulates a GraphQL query/mutation pair and returns `{ data, loading, error, mutate }`.

```typescript
// Tables
function useTables(): { tables: Table[]; loading: boolean; error?: Error }
function useTableData(name: string, page: number): { rows: Row[]; total: number; loading: boolean }
function useTableMutations(): { deleteTable, toggleStar }

// Artifacts
function useArtifacts(): { artifacts: Artifact[]; loading: boolean }
function useArtifact(id: string): { content: ArtifactContent; loading: boolean }
function useArtifactMutations(): { deleteArtifact, toggleStar }

// Facts
function useFacts(): { facts: Fact[]; loading: boolean }
function useFactMutations(): { addFact, editFact, persistFact, forgetFact, moveFact }

// Entities
function useEntities(type?: string): { entities: Entity[]; loading: boolean }

// Session state (read-only)
function useSteps(): { steps: Step[]; loading: boolean }
function useScratchpad(): { narrative: string; loading: boolean }
function useProofTree(): { tree: ProofTree; loading: boolean }
function useProofFacts(): { facts: ProofFact[]; loading: boolean }

// Execution (subscription-driven)
function useQueryExecution(): {
  submit: (problem: string, isFollowup: boolean) => void
  cancel: () => void
  approvePlan: (approved: boolean, feedback?: string) => void
  replanFrom: (step: number, mode: string, goal?: string) => void
  events: ExecutionEvent[]
  status: 'idle' | 'planning' | 'executing' | 'awaiting_approval' | 'complete' | 'error'
  plan: ExecutionPlan | null
}

// Data sources
function useDatabases(): { databases: Database[]; loading: boolean }
function useDatabaseMutations(): { addDatabase, removeDatabase, updateDatabase, testDatabase }
function useFiles(): { files: UploadedFile[]; loading: boolean }
function useDocuments(): { ... }

// Domains
function useDomains(): { domains: Domain[]; tree: DomainTree; loading: boolean }
function useDomainMutations(): { createDomain, deleteDomain, promoteDomain, ... }

// Learning/Rules
function useLearnings(category?: string): { learnings: Learning[]; rules: Rule[]; loading: boolean }
function useRuleMutations(): { createRule, updateRule, deleteRule, compactLearnings }

// Config (app-level, not session-scoped)
function useConfig(): { config: ServerConfig; loading: boolean }
function usePermissions(): { permissions: UserPermissions; loading: boolean }
```

### Apollo Cache Policies

```typescript
const typePolicies: TypePolicies = {
  Query: {
    fields: {
      tables: { merge: false },        // replace on refetch
      artifacts: { merge: false },
      facts: { merge: false },
    },
  },
  Table: { keyFields: ['name'] },
  Artifact: { keyFields: ['id'] },
  Fact: { keyFields: ['name'] },
  GlossaryTermType: { keyFields: ['name'] },   // already exists
  Session: { keyFields: ['sessionId'] },
  Database: { keyFields: ['name'] },
  Domain: { keyFields: ['filename'] },
  Rule: { keyFields: ['id'] },
  Learning: { keyFields: ['id'] },
}
```

Mutations use `cache.modify()` or `refetchQueries` — no manual `fetchTerms()` calls.

### Subscription Wiring

```typescript
// In SessionProvider — auto-subscribe when session is active
function SessionProvider({ sessionId, children }) {
  const execSub = useSubscription(QUERY_EXECUTION_SUB, { variables: { sessionId } })
  const entitySub = useSubscription(ENTITY_UPDATES_SUB, { variables: { sessionId } })
  const ingestSub = useSubscription(DOCUMENT_INGESTION_SUB, { variables: { sessionId } })
  // glossaryChanged already wired in useGlossaryData

  // Update Apollo cache on subscription events
  useEffect(() => {
    if (execSub.data?.queryExecution) {
      handleExecutionEvent(client.cache, execSub.data.queryExecution)
    }
  }, [execSub.data])

  return (
    <SessionContext.Provider value={{ sessionId, ... }}>
      {children}
    </SessionContext.Provider>
  )
}
```

### Error + Loading Patterns

Standardized across all hooks:

```typescript
// Shared error boundary for GraphQL errors
function GraphQLErrorBoundary({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary fallback={<ErrorPanel />}>
      {children}
    </ErrorBoundary>
  )
}

// Per-hook loading pattern (all hooks return same shape)
interface QueryResult<T> {
  data: T | undefined
  loading: boolean
  error: ApolloError | undefined
}

// Component usage — no try/catch, no manual error strings
function FactsPanel() {
  const { facts, loading, error } = useFacts()
  if (error) return <ErrorMessage error={error} />
  if (loading) return <Skeleton variant="list" count={5} />
  return <FactsList facts={facts} />
}
```

### What Gets Deleted

| File | Reason |
|------|--------|
| `store/sessionStore.ts` | Data → Apollo hooks; WS → subscription; UI bits → reactive vars |
| `store/artifactStore.ts` | All data fetching → Apollo hooks. 40+ actions deleted. |
| `store/glossaryStore.ts` | Already migrating to `useGlossaryData`; delete REST bridge |
| `store/proofStore.ts` | → `useProofFacts` subscription hook |
| `store/testStore.ts` | → `useGoldenTests` + `useTestResults` hooks |
| `store/uiStore.ts` | → `graphql/ui-state.ts` reactive variables |
| `store/toastStore.ts` | → `toastsVar` reactive variable |
| `store/authStore.ts` | → `AuthContext` + reactive variable for non-React access |
| `store/entityCache.ts` | Entity state → Apollo normalized cache |
| `store/entityCacheKeys.ts` | Dead (already deleted in git status) |
| `api/sessions.ts` (~1238 lines) | All REST calls → GraphQL mutations in hooks |
| `api/queries.ts` | → `useQueryExecution` hook |
| `api/websocket.ts` | → GraphQL subscription transport |
| `api/client.ts` | Kept only for 5 binary download endpoints |
| **`zustand` dependency** | Removed from `package.json` entirely |

### New Files

| File | Purpose |
|------|--------|
| `graphql/ui-state.ts` | All reactive variables (panels, theme, toasts, deep links, expanded sections) |
| `graphql/cache-policies.ts` | `typePolicies` with `keyFields` and local-only field defaults |
| `contexts/AuthContext.tsx` | Auth provider wrapping app; provides user/token/permissions |
| `contexts/SessionContext.tsx` | Session provider; auto-subscribes to execution/entity/ingestion events |
| `hooks/useTables.ts` | `useTables`, `useTableData`, `useTableMutations` |
| `hooks/useArtifacts.ts` | `useArtifacts`, `useArtifact`, `useArtifactMutations` |
| `hooks/useFacts.ts` | `useFacts`, `useFactMutations` |
| `hooks/useEntities.ts` | `useEntities` |
| `hooks/useSession.ts` | `useSession`, `useSessions`, `useSessionMutations` |
| `hooks/useQueryExecution.ts` | `useQueryExecution` (submit, cancel, approve, subscription) |
| `hooks/useDatabases.ts` | `useDatabases`, `useDatabaseMutations` |
| `hooks/useFiles.ts` | `useFiles`, `useDocuments`, mutation hooks |
| `hooks/useDomains.ts` | `useDomains`, `useDomainMutations` |
| `hooks/useLearnings.ts` | `useLearnings`, `useRuleMutations` |
| `hooks/useProof.ts` | `useProofTree`, `useProofFacts` |
| `hooks/useConfig.ts` | `useConfig`, `usePermissions` |

### Zustand → Apollo Migration Per Phase

Each implementation phase (below) includes the hook + context + reactive variable work for that domain:

| Phase | Zustand Stores Killed | New Hooks | New Context / Reactive Vars |
|-------|----------------------|-----------|----------------------------|
| 0 | Remove `QueryClientProvider` (dead) | — | `ApolloProvider` wired; `graphql/ui-state.ts` created |
| 1 | `authStore` | `useAuth` | `AuthContext`; `isAuthDisabledVar` |
| 2 | `sessionStore` (session CRUD) | `useSession`, `useSessions` | `SessionContext` |
| 3 | `sessionStore` (read-only state), `proofStore` | `useSteps`, `useScratchpad`, `useProofTree`, `useProofFacts`, `useMessages`, `useObjectives` | — |
| 4 | `artifactStore` (tables/artifacts/facts/entities) | `useTables`, `useTableData`, `useArtifacts`, `useArtifact`, `useFacts`, `useEntities` + mutation hooks | — |
| 5 | `artifactStore` (sources) | `useDatabases`, `useFiles`, `useDocuments`, `useDataSources` + mutation hooks | — |
| 6 | domain state in `sessionStore` | `useDomains`, `useDomainMutations` | — |
| 7 | `sessionStore` (WS handler) | `useQueryExecution` | Subscriptions in `SessionProvider` |
| 8 | `artifactStore` (learnings) | `useLearnings`, `useRuleMutations`, `useSkills` | — |
| 9 | `testStore` | `useGoldenTests`, `useTestResults`, `useFineTuneJobs`, `useUserSources` | — |
| 10 | `uiStore`, `toastStore`, `glossaryStore` — delete all remaining | — | Reactive vars replace: `showArtifactPanelVar`, `activeDeepLinkVar`, `toastsVar`, `themeVar`, `expandedSectionsVar` |
| **Done** | `zustand` removed from `package.json` | | **Zero external state libraries** |

### Performance Wins

| Problem | Current | After |
|---------|---------|-------|
| Full-store re-render | Any `sessionStore` change re-renders all consumers | `useQuery` / `useReactiveVar` only re-render on relevant data changes |
| Duplicate fetches | 10+ `useEffect` fetch calls on mount in `ArtifactPanel` | Apollo deduplicates identical in-flight queries |
| No cache | Every panel mount refetches all data | Apollo normalized cache — read from cache first |
| Manual invalidation | `fetchTerms()` after every mutation | Mutations update cache directly via `cache.modify()` or `refetchQueries` |
| N+1 subscription handlers | WS events parsed in 3+ stores | Single subscription → Apollo cache update → all consumers see new data |
| Two state systems | Zustand + Apollo + dead TanStack Query | Apollo only — one cache, one mental model |
| Bundle size | `zustand` + `@tanstack/react-query` + `@apollo/client` | `@apollo/client` only |

---

## Implementation Phases

Each phase: implement backend schema + resolvers → wire frontend → write tests → delete REST route → next phase.

### Test Strategy Per Phase

Each phase produces three test layers before moving on:

| Layer | Backend (pytest) | Frontend (vitest) |
|-------|-----------------|-------------------|
| **Unit** | Resolver functions with mocked session manager. Type instantiation. Schema introspection (fields, args, return types exist). | GraphQL query/mutation documents parse correctly. Store actions dispatch correct operations. |
| **Integration** | `httpx.AsyncClient` against real Strawberry schema with test session. Verify resolver → session manager → storage round-trip. Auth/permission checks. | Apollo MockedProvider rendering components with canned GraphQL responses. Subscription event → store update → re-render. |
| **E2E** | Full server (`TestClient`) hitting `/api/graphql` with real DuckDB session. Mutation → query round-trip (create then read back). Subscription event delivery. | Playwright against dev server. User action → GraphQL call → UI update. WebSocket subscription connection and event handling. |

### Ordering Rationale

1. **Auth + Config + Session first** — everything else depends on having a session
2. **Read-only state next** — low risk, high coverage, builds out query patterns
3. **Core interactive loop** — tables/artifacts/facts are the main UI
4. **Query execution + WS** — hardest phase, deferred until patterns are established
5. **Admin/secondary features last** — lower traffic, can coexist with REST longer

---

### Phase 0: Infrastructure
**Scope:** Shared plumbing for all subsequent phases.

**Work:**
- Extend `constat/server/graphql/__init__.py` with module-based schema stitching (one file per domain)
- Add `SessionContext` to GraphQL context (session_manager + authenticated user + session_id)
- Configure `graphql-ws` subscription transport on the existing `/api/graphql` endpoint
- Frontend: configure Apollo link chain (HTTP + WS split link, auth header injection)
- Add `constat/server/graphql/session_context.py` — shared context type for resolvers
- Establish test fixtures: `graphql_client` (pytest), `MockedProvider` wrapper (vitest), Playwright GraphQL intercept helpers

**Tests:**
- Unit: context getter returns correct session_manager and user
- Integration: `/api/graphql` accepts query `{ __typename }` with auth header
- E2E: Apollo client connects, subscription WS handshake completes

**Files:**
```
constat/server/graphql/__init__.py          (modify)
constat/server/graphql/session_context.py   (new)
constat-ui/src/graphql/client.ts            (modify)
tests/test_graphql_infra.py                 (new)
constat-ui/src/graphql/__tests__/client.test.ts (new)
```

---

### Phase 1: Auth + Config + Permissions
**Scope:** 4 auth + 1 config + 1 permissions = **6 endpoints → 2Q + 4M**

**Gaps → GraphQL:**
```
mutation login(email, password)             → AuthPayload {token, user}
mutation logout                             → Boolean
mutation registerPasskey(...)               → PasskeyRegistration
mutation authenticatePasskey(...)           → AuthPayload
query config                                → ServerConfig
query myPermissions                         → UserPermissions
```

**Tests:**
- Unit: login resolver returns token on valid creds, raises on invalid. Config resolver returns expected shape.
- Integration: `login` mutation → subsequent query with Bearer token succeeds. Unauthenticated query rejected.
- E2E: Playwright login flow → token stored → authenticated page loads.

**Done when:** All auth/config REST routes removed. Frontend `authStore` uses GraphQL only.

---

### Phase 2: Session CRUD + Domain Selection
**Scope:** 9 session + 2 domain selection = **11 endpoints → 4Q + 7M**

**Gaps → GraphQL:**
```
mutation createSession(userId)              → Session
query sessions(userId)                      → [Session]
query session(id)                           → Session
mutation deleteSession(id)                  → Boolean
mutation togglePublicSharing(id, isPublic)  → Session
query sessionShares(id)                     → [Share]
mutation shareSession(id, email)            → Share
mutation removeShare(id, userId)            → Boolean
mutation resetContext(id)                    → Boolean
mutation setActiveDomains(id, domains)      → [String]
query activeDomains(id)                     → [String]
```

**Tests:**
- Unit: createSession resolver calls session_manager.create. Domain selection validates against known domains.
- Integration: create → list → get → delete round-trip. Share flow: share → list shares → remove.
- E2E: Playwright create session → see in sidebar → delete → gone.

**Done when:** `sessionStore` no longer imports from `api/sessions.ts` for these operations.

---

### Phase 3: Read-Only Session State
**Scope:** 6 execution state + 3 proof + 5 messages/prefs + 2 schema = **16 endpoints → 16Q**

**Gaps → GraphQL:**
```
query steps(sessionId)                      → [Step]
query inferenceCodes(sessionId)             → [InferenceCode]
query scratchpad(sessionId)                 → String
query sessionDDL(sessionId)                 → String
query executionOutput(sessionId)            → ExecutionOutput
query sessionRouting(sessionId)             → RoutingInfo
query proofTree(sessionId)                  → ProofTree
query proofFacts(sessionId)                 → [ProofFact]
query messages(sessionId)                   → [Message]
query objectives(sessionId)                 → [Objective]
query promptContext(sessionId)              → PromptContext
query databaseSchema(dbName)               → [TableSchema]
query apiSchema(apiName)                    → ApiSchema
mutation saveProofFacts(sessionId, facts)   → Boolean
mutation saveMessages(sessionId, messages)  → Boolean
mutation updateSystemPrompt(sessionId, p)   → Boolean
```

**Tests:**
- Unit: each resolver returns correct type from session manager mock.
- Integration: after a real query execution, verify steps/scratchpad/output populated via GraphQL.
- E2E: run query in Playwright → open proof panel → facts display.

**Done when:** All read-only session data fetched via GraphQL.

---

### Phase 4: Tables + Artifacts + Facts + Entities
**Scope:** 6 + 5 + 6 + 2 = **19 endpoints → 8Q + 10M**

**Gaps → GraphQL:**
```
# Tables
query tables(sessionId)                     → [Table]
query tableData(sessionId, name, p, ps)     → TablePage {rows, total, columns}
query tableVersions(sessionId, name)        → [TableVersion]
query tableVersionData(sessionId, n, v, p)  → TablePage
mutation deleteTable(sessionId, name)       → Boolean
mutation toggleTableStar(sessionId, name)   → Table

# Artifacts
query artifacts(sessionId)                  → [Artifact]
query artifact(sessionId, id)               → ArtifactContent
query artifactVersions(sessionId, id)       → [ArtifactVersion]
mutation deleteArtifact(sessionId, id)      → Boolean
mutation toggleArtifactStar(sessionId, id)  → Artifact

# Facts
query facts(sessionId)                      → [Fact]
mutation addFact(sessionId, name, value)    → Fact
mutation editFact(sessionId, name, value)   → Fact
mutation persistFact(sessionId, name)       → Fact
mutation forgetFact(sessionId, name)        → Boolean
mutation moveFact(sessionId, name, domain)  → Fact

# Entities
query entities(sessionId, type)             → [Entity]
mutation addEntityToGlossary(sessionId, id) → GlossaryTermType
```

**Tests:**
- Unit: pagination args validated (page >= 1, pageSize capped). Star toggle flips boolean.
- Integration: addFact → listFacts → moveFact → verify domain changed. deleteTable → listTables → gone.
- E2E: Playwright star artifact → refresh → still starred. Add fact → see in panel → persist → reload → still there.

**Done when:** Artifact panel, table accordion, fact panel, entity list all GraphQL-driven.

---

### Phase 5: Data Sources (Files, Documents, Databases, APIs)
**Scope:** 8 + 6 + 7 + 2 + 1 = **24 endpoints → 5Q + 16M + 1S**

**Gaps → GraphQL:**
```
# Files
query files(sessionId)                      → [UploadedFile]
mutation uploadFile(sessionId, file)        → UploadedFile      # multipart
mutation uploadFileDataUri(sessionId, d)    → UploadedFile
mutation deleteFile(sessionId, id)          → Boolean
query fileRefs(sessionId)                   → [FileRef]
mutation addFileRef(sessionId, input)       → FileRef
mutation deleteFileRef(sessionId, name)     → Boolean

# Documents
mutation uploadDocuments(sessionId, files)  → [DocumentResult]  # multipart
mutation addDocumentUri(sessionId, input)   → DocumentResult
mutation addEmailSource(sessionId, input)   → DocumentResult
mutation refreshDocuments(sessionId)        → Boolean
mutation updateDocument(sessionId, n, inp)  → DocumentResult

# Databases
query databases(sessionId)                  → [Database]
query databaseTablePreview(sid, db, t, p)   → TablePage
mutation addDatabase(sessionId, input)      → Database
mutation testDatabase(sessionId, name)      → TestResult
mutation removeDatabase(sessionId, name)    → Boolean
mutation updateDatabase(sessionId, n, inp)  → Database

# APIs
mutation addApi(sessionId, input)           → ApiSource
mutation removeApi(sessionId, name)        → Boolean

# Combined
query dataSources(sessionId)                → DataSources {databases, apis, documents, files}

# Subscription
subscription documentIngestion(sessionId)   → IngestionEvent
```

**Multipart uploads:** Use `graphql-upload` spec (strawberry supports it via `Upload` scalar).

**Tests:**
- Unit: upload resolver validates file type/size. addDatabase resolver validates connection string format.
- Integration: uploadFile → listFiles → deleteFile round-trip. addDatabase → testDatabase → preview.
- E2E: Playwright drag-and-drop file upload → see in file list. Add database URL → test connection → see tables.

**Done when:** All source management panels use GraphQL. Document ingestion progress via subscription.

---

### Phase 6: Domain Management
**Scope:** 16 domain endpoints = **16 endpoints → 5Q + 13M** (including moves)

**Gaps → GraphQL:**
```
query domains                               → [Domain]
query domainTree                            → DomainTree
query domain(filename)                      → Domain
query domainContent(filename)               → String (YAML)
query domainSkills(filename)                → [Skill]
query domainAgents(filename)                → [Agent]
query domainRules(filename)                 → [Rule]
mutation createDomain(input)                → Domain
mutation updateDomainContent(f, yaml)       → Domain
mutation updateDomain(f, input)             → Domain
mutation deleteDomain(f)                    → Boolean
mutation promoteDomain(f, target)           → Domain
mutation moveDomainSource(input)            → Boolean
mutation moveDomainSkill(input)             → Boolean
mutation moveDomainAgent(input)             → Boolean
mutation moveDomainRule(input)              → Boolean
```

**Tests:**
- Unit: promoteDomain validates tier ordering. moveDomainSource validates source exists.
- Integration: createDomain → getDomainTree → see new node → deleteDomain → node gone.
- E2E: Playwright domain admin panel → create → promote → move skill into it → delete.

**Done when:** Domain admin panel fully GraphQL-driven.

---

### Phase 7: Query Execution + WebSocket Replacement
**Scope:** 4 execution endpoints + all WS events/actions = **4 endpoints + WS → 1Q + 11M + 1S**

This is the hardest phase. The custom WS protocol becomes GraphQL mutations + a single execution subscription.

**Gaps → GraphQL:**
```
# Mutations (replacing REST + WS actions)
mutation submitQuery(sessionId, problem, isFollowup) → QuerySubmission {queryId}
mutation cancelExecution(sessionId)                  → Boolean
mutation approvePlan(sessionId, approved, feedback, deletedSteps, editedSteps) → Boolean
mutation replanFrom(sessionId, step, mode, goal)     → Boolean
mutation editObjective(sessionId, index, text)       → Boolean
mutation deleteObjective(sessionId, index)           → Boolean
mutation requestAutocomplete(sessionId, context, prefix, parent) → AutocompleteResult

# Query
query executionPlan(sessionId)                       → ExecutionPlan

# Subscription (replacing ALL WS events)
subscription queryExecution(sessionId) → ExecutionEvent
  # Union type:
  | StepEvent {stepNumber, status, code, narrative}
  | StepComplete {stepNumber, result, tableName, artifactId}
  | PlanReady {plan}
  | ApprovalRequest {message}
  | ClarificationRequest {question, options}
  | QueryComplete {summary}
  | ExecutionError {message, stepNumber}
  | EntityUpdate {entities}
  | ArtifactCreated {artifact}
  | TableCreated {table}
```

**Migration strategy:**
1. Implement subscription alongside existing WS (both active)
2. Wire frontend to subscription
3. Verify parity with shadow testing (both transports, compare events)
4. Remove WS endpoint

**Tests:**
- Unit: each event type serializes correctly. submitQuery resolver creates execution task.
- Integration: submitQuery → subscribe → receive step events → approve plan → receive completion.
- E2E: Playwright full query flow: type question → see plan → approve → watch steps execute → see result table.

**Done when:** `constat-ui/src/api/websocket.ts` deleted. All streaming via GraphQL subscription.

---

### Phase 8: Learning/Rules + Skills/Agents
**Scope:** 7 + 3 = **10 endpoints → 4Q + 5M**

**Gaps → GraphQL:**
```
query learnings(category)                   → LearningsAndRules
query skills                                → [Skill]
query skill(name)                           → Skill
mutation compactLearnings                   → CompactionResult
mutation deleteLearning(id)                 → Boolean
mutation createRule(input)                  → Rule
mutation updateRule(id, input)              → Rule
mutation deleteRule(id)                     → Boolean
mutation activateAgent(sessionId, name)     → Agent
```

**Tests:**
- Unit: compactLearnings resolver triggers compaction. Rule validation (required fields).
- Integration: createRule → listLearnings → see rule → updateRule → verify change → deleteRule.
- E2E: Playwright rule management UI → create → edit → compact.

**Done when:** Learning panel and skill/agent activation use GraphQL.

---

### Phase 9: User Sources + Fine-Tuning + Feedback + Testing + OAuth
**Scope:** 4 + 6 + 1 + 3 + 2 = **16 endpoints → 8Q + 8M**

**Gaps → GraphQL:**
```
# User sources
query userSources(userId)                   → [UserSource]
mutation removeUserSource(uid, type, name)  → Boolean
mutation promoteSource(sessionId, type, n)  → UserSource

# Fine-tuning
query fineTuneJobs(params)                  → [FineTuneJob]
query fineTuneJob(modelId)                  → FineTuneJob
query fineTuneProviders                     → [FineTuneProvider]
mutation startFineTuneJob(input)            → FineTuneJob
mutation cancelFineTuneJob(modelId)         → FineTuneJob
mutation deleteFineTuneJob(modelId)         → Boolean

# Feedback
mutation submitFeedback(sessionId, input)   → Boolean

# Testing
query goldenTests(sessionId)                → [GoldenTest]
query testResults(sessionId)                → [TestResult]
mutation runTest(sessionId, input)           → TestRun

# OAuth
query emailOAuthProviders                   → [OAuthProvider]
mutation emailOAuthCallback(input)          → OAuthResult
```

**Tests:**
- Unit: promoteSource validates source exists in session. Fine-tune job state machine.
- Integration: startFineTuneJob → getFineTuneJob → cancelFineTuneJob lifecycle.
- E2E: Playwright golden test runner → run test → see results.

**Done when:** All secondary panels use GraphQL.

---

### Phase 10: Public Sessions + Cleanup
**Scope:** 7 public endpoints + REST route deletion + cleanup = **7 endpoints → 7Q**

**Gaps → GraphQL:**
```
query publicSession(id)                     → PublicSession
query publicMessages(id)                    → [Message]
query publicArtifacts(id)                   → [Artifact]
query publicTables(id)                      → [Table]
query publicTableData(id, name, page, ps)   → TablePage
query publicArtifact(id, artifactId)        → ArtifactContent
query publicProofFacts(id)                  → [ProofFact]
```

Public queries skip auth — use separate schema entry point or `@public` directive.

**Cleanup work:**
- Delete all REST route files except binary download endpoints
- Delete `constat-ui/src/api/sessions.ts`, `queries.ts`, `websocket.ts`
- Consolidate remaining binary endpoints under `/api/download/`
- Remove `constat-ui/src/api/client.ts` REST client (keep only for binary fetches)
- Final audit: grep for any remaining `fetch(` or REST client usage

**Tests:**
- Unit: public queries work without auth token. Private queries reject without auth.
- Integration: publicSession returns data for shared sessions, 404 for private ones.
- E2E: Playwright open public share link → see read-only session with tables/artifacts/proof.

**Done when:** No REST routes remain (except 5 binary endpoints). `api/client.ts` only used for binary downloads. WebSocket manager deleted.

---

## Phase Summary

| Phase | Section | Endpoints | GraphQL Ops | Risk |
|-------|---------|-----------|-------------|------|
| 0 | Infrastructure | — | — | Low |
| 1 | Auth + Config | 6 | 2Q + 4M | Low |
| 2 | Session CRUD | 11 | 4Q + 7M | Low |
| 3 | Read-Only State | 16 | 16Q | Low |
| 4 | Tables/Artifacts/Facts/Entities | 19 | 8Q + 10M | Medium |
| 5 | Data Sources | 24 | 5Q + 16M + 1S | Medium |
| 6 | Domain Management | 16 | 5Q + 13M | Medium |
| 7 | Query Execution + WS | 4 + WS | 1Q + 11M + 1S | **High** |
| 8 | Learning/Rules/Skills | 10 | 4Q + 5M | Low |
| 9 | Secondary Features | 16 | 8Q + 8M | Low |
| 10 | Public + Cleanup | 7 + cleanup | 7Q | Low |
| **Total** | | **~131** | **52Q + 87M + 8S** | |
