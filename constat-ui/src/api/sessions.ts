// Session API calls

import { get, post, put, patch, del } from './client'
import type {
  Session,
  SessionListResponse,
  TableInfo,
  TableData,
  TableVersionsResponse,
  Artifact,
  ArtifactContent,
  ArtifactVersionsResponse,
  Fact,
  Entity,
  GlossaryTerm,
  GlossaryListResponse,
  UploadedFile,
  FileReference,
  SessionDatabase,
  Config,
  Learning,
  Rule,
  FineTuneJob,
  FineTuneProvider,
  ModelRouteInfo,
} from '@/types/api'

// Session ID persistence (matches key used in App.tsx)
const SESSION_ID_BASE_KEY = 'constat-session-id'

function getSessionKey(userId?: string): string {
  // Per-user keys when auth is enabled, base key when disabled
  return userId && userId !== 'default'
    ? `${SESSION_ID_BASE_KEY}-${userId}`
    : SESSION_ID_BASE_KEY
}

export function getStoredSessionId(userId?: string): string | null {
  return localStorage.getItem(getSessionKey(userId))
}

export function storeSessionId(sessionId: string, userId?: string): void {
  localStorage.setItem(getSessionKey(userId), sessionId)
}

export function clearStoredSessionId(userId?: string): void {
  localStorage.removeItem(getSessionKey(userId))
}

export function createNewSessionId(userId?: string): string {
  const sessionId = crypto.randomUUID()
  storeSessionId(sessionId, userId)
  return sessionId
}

export function getOrCreateSessionId(userId?: string): string {
  const existing = getStoredSessionId(userId)
  if (existing) {
    return existing
  }
  return createNewSessionId(userId)
}

// Session CRUD
export async function createSession(userId = 'default', sessionId?: string): Promise<Session> {
  // Use provided session_id or get/create from localStorage
  const effectiveSessionId = sessionId ?? getOrCreateSessionId()
  return post<Session>('/sessions', { user_id: userId, session_id: effectiveSessionId })
}

export async function listSessions(userId?: string): Promise<SessionListResponse> {
  const query = userId ? `?user_id=${encodeURIComponent(userId)}` : ''
  return get<SessionListResponse>(`/sessions${query}`)
}

export async function getSession(sessionId: string): Promise<Session> {
  return get<Session>(`/sessions/${sessionId}`)
}

export async function deleteSession(sessionId: string): Promise<{ status: string }> {
  return del<{ status: string }>(`/sessions/${sessionId}`)
}

// Tables
export async function listTables(sessionId: string): Promise<{ tables: TableInfo[] }> {
  return get<{ tables: TableInfo[] }>(`/sessions/${sessionId}/tables`)
}

export async function getTableData(
  sessionId: string,
  tableName: string,
  page = 1,
  pageSize = 100
): Promise<TableData> {
  return get<TableData>(
    `/sessions/${sessionId}/tables/${tableName}?page=${page}&page_size=${pageSize}`
  )
}

// Table Versions
export async function getTableVersions(
  sessionId: string,
  tableName: string
): Promise<TableVersionsResponse> {
  return get<TableVersionsResponse>(
    `/sessions/${sessionId}/tables/${encodeURIComponent(tableName)}/versions`
  )
}

export async function getTableVersionData(
  sessionId: string,
  tableName: string,
  version: number,
  page = 1,
  pageSize = 100
): Promise<TableData> {
  return get<TableData>(
    `/sessions/${sessionId}/tables/${encodeURIComponent(tableName)}/version/${version}?page=${page}&page_size=${pageSize}`
  )
}

// Database source table preview (for viewing underlying data source tables)
export interface DatabaseTablePreview {
  database: string
  table_name: string
  columns: string[]
  data: Record<string, unknown>[]
  page: number
  page_size: number
  total_rows: number
  has_more: boolean
}

export async function getDatabaseTablePreview(
  sessionId: string,
  dbName: string,
  tableName: string,
  page = 1,
  pageSize = 100
): Promise<DatabaseTablePreview> {
  return get<DatabaseTablePreview>(
    `/sessions/${sessionId}/databases/${dbName}/tables/${tableName}/preview?page=${page}&page_size=${pageSize}`
  )
}

// Database schema (tables per database)
export interface DatabaseTableInfo {
  name: string
  row_count: number | null
  column_count: number
}

export async function listDatabaseTables(
  sessionId: string,
  dbName: string,
): Promise<{ database: string; tables: DatabaseTableInfo[] }> {
  return get<{ database: string; tables: DatabaseTableInfo[] }>(
    `/schema/databases/${dbName}/tables?session_id=${sessionId}`
  )
}

// API schema (endpoints per API)
export interface ApiEndpointField {
  name: string
  type: string
  description?: string
  is_required: boolean
}

export interface ApiEndpointInfo {
  name: string
  kind?: string  // "graphql_query", "graphql_type", "rest", etc.
  return_type?: string  // e.g., "[Breed!]!", "User"
  description?: string
  http_method?: string
  http_path?: string
  fields: ApiEndpointField[]
}

export interface ApiSchemaResponse {
  name: string
  type: string
  description?: string
  endpoints: ApiEndpointInfo[]
}

export async function getApiSchema(
  sessionId: string,
  apiName: string,
): Promise<ApiSchemaResponse> {
  return get<ApiSchemaResponse>(
    `/schema/apis/${apiName}?session_id=${sessionId}`
  )
}

// Artifacts
export async function listArtifacts(sessionId: string): Promise<{ artifacts: Artifact[] }> {
  return get<{ artifacts: Artifact[] }>(`/sessions/${sessionId}/artifacts`)
}

export async function getArtifact(
  sessionId: string,
  artifactId: number
): Promise<ArtifactContent> {
  return get<ArtifactContent>(`/sessions/${sessionId}/artifacts/${artifactId}`)
}

// Artifact Versions
export async function getArtifactVersions(
  sessionId: string,
  artifactId: number
): Promise<ArtifactVersionsResponse> {
  return get<ArtifactVersionsResponse>(
    `/sessions/${sessionId}/artifacts/${artifactId}/versions`
  )
}

// Facts
export async function listFacts(sessionId: string): Promise<{ facts: Fact[] }> {
  return get<{ facts: Fact[] }>(`/sessions/${sessionId}/facts`)
}

export async function addFact(
  sessionId: string,
  name: string,
  value: unknown,
  persist: boolean = false
): Promise<{ status: string; fact: Fact }> {
  return post<{ status: string; fact: Fact }>(`/sessions/${sessionId}/facts`, { name, value, persist })
}

export async function persistFact(
  sessionId: string,
  factName: string
): Promise<{ status: string }> {
  return post<{ status: string }>(`/sessions/${sessionId}/facts/${factName}/persist`)
}

export async function forgetFact(
  sessionId: string,
  factName: string
): Promise<{ status: string }> {
  return post<{ status: string }>(`/sessions/${sessionId}/facts/${factName}/forget`)
}

export async function editFact(
  sessionId: string,
  factName: string,
  value: unknown
): Promise<{ status: string }> {
  return post<{ status: string }>(`/sessions/${sessionId}/facts/${factName}`, { value })
}

export async function moveFact(
  sessionId: string,
  factName: string,
  toDomain: string
): Promise<{ status: string }> {
  return post<{ status: string }>(`/sessions/${sessionId}/facts/${encodeURIComponent(factName)}/move`, { to_domain: toDomain })
}

// Star/Promote
export async function toggleArtifactStar(
  sessionId: string,
  artifactId: number
): Promise<{ artifact_id: number; is_starred: boolean }> {
  return post<{ artifact_id: number; is_starred: boolean }>(
    `/sessions/${sessionId}/artifacts/${artifactId}/star`
  )
}

export async function deleteArtifact(
  sessionId: string,
  artifactId: number
): Promise<{ status: string; artifact_id: number }> {
  return del<{ status: string; artifact_id: number }>(
    `/sessions/${sessionId}/artifacts/${artifactId}`
  )
}

export async function deleteTable(
  sessionId: string,
  tableName: string
): Promise<{ status: string; table_name: string }> {
  return del<{ status: string; table_name: string }>(
    `/sessions/${sessionId}/tables/${tableName}`
  )
}

// Public Sharing
export async function togglePublicSharing(
  sessionId: string,
  isPublic: boolean
): Promise<{ status: string; public: boolean; share_url: string }> {
  return post<{ status: string; public: boolean; share_url: string }>(
    `/sessions/${sessionId}/public`,
    { public: isPublic }
  )
}

export async function publicGetSession(
  sessionId: string
): Promise<{ session_id: string; summary: string | null; status: string }> {
  const resp = await fetch(`/api/public/${sessionId}`)
  if (!resp.ok) throw new Error('Not found')
  return resp.json()
}

export async function publicGetMessages(
  sessionId: string
): Promise<{ messages: StoredMessage[] }> {
  const resp = await fetch(`/api/public/${sessionId}/messages`)
  if (!resp.ok) throw new Error('Not found')
  return resp.json()
}

export async function publicListArtifacts(
  sessionId: string
): Promise<{ artifacts: Artifact[] }> {
  const resp = await fetch(`/api/public/${sessionId}/artifacts`)
  if (!resp.ok) throw new Error('Not found')
  return resp.json()
}

export async function publicListTables(
  sessionId: string
): Promise<{ tables: TableInfo[] }> {
  const resp = await fetch(`/api/public/${sessionId}/tables`)
  if (!resp.ok) throw new Error('Not found')
  return resp.json()
}

export async function publicGetTableData(
  sessionId: string,
  tableName: string,
  page = 1,
  pageSize = 100
): Promise<TableData> {
  const resp = await fetch(`/api/public/${sessionId}/tables/${encodeURIComponent(tableName)}?page=${page}&page_size=${pageSize}`)
  if (!resp.ok) throw new Error('Not found')
  return resp.json()
}

export async function publicGetArtifact(
  sessionId: string,
  artifactId: number
): Promise<ArtifactContent> {
  const resp = await fetch(`/api/public/${sessionId}/artifacts/${artifactId}`)
  if (!resp.ok) throw new Error('Not found')
  return resp.json()
}

export async function publicGetProofFacts(
  sessionId: string
): Promise<{ facts: StoredProofFact[]; summary: string | null }> {
  const resp = await fetch(`/api/public/${sessionId}/proof-facts`)
  if (!resp.ok) throw new Error('Not found')
  return resp.json()
}

// Session Sharing
export async function shareSession(
  sessionId: string,
  email: string
): Promise<{ status: string; share_url: string }> {
  return post<{ status: string; share_url: string }>(
    `/sessions/${sessionId}/share`,
    { email }
  )
}

export async function getShares(
  sessionId: string
): Promise<{ shared_with: string[] }> {
  return get<{ shared_with: string[] }>(`/sessions/${sessionId}/shares`)
}

export async function removeShare(
  sessionId: string,
  userId: string
): Promise<{ status: string }> {
  return del<{ status: string }>(`/sessions/${sessionId}/share/${encodeURIComponent(userId)}`)
}

export async function toggleTableStar(
  sessionId: string,
  tableName: string
): Promise<{ table_name: string; is_starred: boolean }> {
  return post<{ table_name: string; is_starred: boolean }>(
    `/sessions/${sessionId}/tables/${encodeURIComponent(tableName)}/star`
  )
}

// Entities
export async function listEntities(
  sessionId: string,
  entityType?: string
): Promise<{ entities: Entity[] }> {
  const query = entityType ? `?entity_type=${encodeURIComponent(entityType)}` : ''
  return get<{ entities: Entity[] }>(`/sessions/${sessionId}/entities${query}`)
}

export async function addEntityToGlossary(
  sessionId: string,
  entityId: string
): Promise<{ status: string }> {
  return post<{ status: string }>(`/sessions/${sessionId}/entities/${entityId}/glossary`)
}

// Glossary
export async function getGlossary(
  sessionId: string,
  scope: string = 'all',
  domain?: string
): Promise<GlossaryListResponse> {
  const params = new URLSearchParams({ scope })
  if (domain) params.set('domain', domain)
  return get<GlossaryListResponse>(`/sessions/${sessionId}/glossary?${params}`)
}

export async function getGlossaryTerm(
  sessionId: string,
  name: string
): Promise<GlossaryTerm & { grounded: boolean; connected_resources: unknown[] }> {
  return get(`/sessions/${sessionId}/glossary/${encodeURIComponent(name)}`)
}

export async function addDefinition(
  sessionId: string,
  name: string,
  definition: string,
  domain?: string,
  aliases?: string[]
): Promise<{ status: string; name: string }> {
  return post(`/sessions/${sessionId}/glossary`, {
    name,
    definition,
    domain,
    aliases: aliases || [],
  })
}

export async function updateGlossaryTerm(
  sessionId: string,
  name: string,
  updates: Record<string, unknown>
): Promise<{ status: string }> {
  return put(`/sessions/${sessionId}/glossary/${encodeURIComponent(name)}`, updates)
}

export async function deleteGlossaryByStatus(
  sessionId: string,
  status: string = 'draft'
): Promise<{ status: string; count: number }> {
  return del(`/sessions/${sessionId}/glossary?status=${encodeURIComponent(status)}`)
}

export async function deleteGlossaryTerm(
  sessionId: string,
  name: string
): Promise<{ status: string; deleted?: string; reparented?: string[]; deprecated?: string[] }> {
  return del(`/sessions/${sessionId}/glossary/${encodeURIComponent(name)}`)
}

export async function renameTerm(
  sessionId: string,
  name: string,
  newName: string
): Promise<{ status: string; old_name: string; new_name: string; display_name: string; relationships_updated: number }> {
  return post(`/sessions/${sessionId}/glossary/${encodeURIComponent(name)}/rename`, { new_name: newName })
}

export async function reconnectTerm(
  sessionId: string,
  name: string,
  updates: { parent_id?: string; domain?: string }
): Promise<{ status: string; name: string; still_deprecated: boolean }> {
  return post(`/sessions/${sessionId}/glossary/${encodeURIComponent(name)}/reconnect`, updates)
}

export interface DomainTreeNode {
  filename: string
  name: string
  path: string
  description: string
  tier: string
  active: boolean
  owner: string
  steward: string
  databases: string[]
  apis: string[]
  documents: string[]
  skills: string[]
  agents: string[]
  rules: string[]
  facts: string[]
  system_prompt: string
  domains: string[]
  children: DomainTreeNode[]
}

export async function getDomainTree(): Promise<DomainTreeNode[]> {
  return get('/domains/tree')
}

export async function moveDomainSource(body: {
  source_type: 'databases' | 'apis' | 'documents'
  source_name: string
  from_domain: string
  to_domain: string
  session_id?: string
}): Promise<{ status: string }> {
  return post('/domains/move-source', body)
}

export async function draftGlossaryDefinition(
  sessionId: string,
  name: string
): Promise<{ status: string; name: string; draft: string }> {
  return post(`/sessions/${sessionId}/glossary/${encodeURIComponent(name)}/draft-definition`)
}

export async function draftGlossaryAliases(
  sessionId: string,
  name: string
): Promise<{ status: string; name: string; aliases: string[] }> {
  return post(`/sessions/${sessionId}/glossary/${encodeURIComponent(name)}/draft-aliases`)
}

export async function draftGlossaryTags(
  sessionId: string,
  name: string
): Promise<{ status: string; name: string; tags: string[] }> {
  return post(`/sessions/${sessionId}/glossary/${encodeURIComponent(name)}/draft-tags`)
}

export async function refineGlossaryTerm(
  sessionId: string,
  name: string
): Promise<{ status: string; before: string; after: string }> {
  return post(`/sessions/${sessionId}/glossary/${encodeURIComponent(name)}/refine`)
}

export async function generateGlossary(
  sessionId: string
): Promise<{ status: string }> {
  return post(`/sessions/${sessionId}/glossary/generate`)
}

export async function suggestTaxonomy(
  sessionId: string
): Promise<{ suggestions: Array<{ child: string; parent: string; confidence: string; reason: string }> }> {
  return post(`/sessions/${sessionId}/glossary/suggest-taxonomy`)
}

export async function bulkUpdateStatus(
  sessionId: string,
  names: string[],
  status: string
): Promise<{ status: string; updated: string[]; failed: string[]; count: number }> {
  return fetch(`/api/sessions/${sessionId}/glossary/bulk-status`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ names, status }),
  }).then(r => r.json())
}

// Relationships
export async function createRelationship(
  sessionId: string,
  subjectName: string,
  verb: string,
  objectName: string
): Promise<{ status: string; id: string }> {
  return post(`/sessions/${sessionId}/relationships`, {
    subject_name: subjectName,
    verb,
    object_name: objectName,
  })
}

export async function updateRelationshipVerb(
  sessionId: string,
  relId: string,
  verb: string
): Promise<{ status: string }> {
  return put(`/sessions/${sessionId}/relationships/${relId}`, { verb })
}

export async function approveRelationship(
  sessionId: string,
  relId: string
): Promise<{ status: string }> {
  return put(`/sessions/${sessionId}/relationships/${relId}/approve`, {})
}

export async function deleteRelationship(
  sessionId: string,
  relId: string
): Promise<{ status: string }> {
  return del(`/sessions/${sessionId}/relationships/${relId}`)
}

export async function persistGlossary(
  sessionId: string,
  domain?: string
): Promise<{ status: string; count: number }> {
  return post(`/sessions/${sessionId}/glossary/persist`, { domain })
}

// Proof Tree
export async function getProofTree(
  sessionId: string
): Promise<{ facts: unknown[]; execution_trace: unknown[] }> {
  return get<{ facts: unknown[]; execution_trace: unknown[] }>(
    `/sessions/${sessionId}/proof-tree`
  )
}

// Step Codes
export interface StepCode {
  step_number: number
  goal: string
  code: string
  prompt?: string
  model?: string
}

export async function listStepCodes(
  sessionId: string
): Promise<{ steps: StepCode[]; total: number }> {
  return get<{ steps: StepCode[]; total: number }>(
    `/sessions/${sessionId}/steps`
  )
}

// Inference Codes (auditable mode)
export interface InferenceCode {
  inference_id: string
  name: string
  operation: string
  code: string
  attempt: number
  prompt?: string
  model?: string
}

export async function listInferenceCodes(
  sessionId: string
): Promise<{ inferences: InferenceCode[]; total: number }> {
  return get<{ inferences: InferenceCode[]; total: number }>(
    `/sessions/${sessionId}/inference-codes`
  )
}

// Scratchpad (execution narrative per step)
export interface ScratchpadEntry {
  step_number: number
  goal: string
  narrative: string
  tables_created: string[]
  code: string
  user_query: string
  objective_index: number | null
}

export async function getScratchpad(
  sessionId: string
): Promise<{ entries: ScratchpadEntry[]; total: number }> {
  return get<{ entries: ScratchpadEntry[]; total: number }>(
    `/sessions/${sessionId}/scratchpad`
  )
}

// DDL (session store schema)
export async function getDDL(
  sessionId: string
): Promise<{ ddl: string }> {
  return get<{ ddl: string }>(`/sessions/${sessionId}/ddl`)
}

// Output
export async function getOutput(
  sessionId: string
): Promise<{ output: string; suggestions: string[]; current_query?: string }> {
  return get<{ output: string; suggestions: string[]; current_query?: string }>(
    `/sessions/${sessionId}/output`
  )
}

// Files
export async function listFiles(sessionId: string): Promise<{ files: UploadedFile[] }> {
  return get<{ files: UploadedFile[] }>(`/sessions/${sessionId}/files`)
}

export async function deleteFile(
  sessionId: string,
  fileId: string
): Promise<{ status: string }> {
  return del<{ status: string }>(`/sessions/${sessionId}/files/${fileId}`)
}

// File References
export async function listFileRefs(
  sessionId: string
): Promise<{ file_refs: FileReference[] }> {
  return get<{ file_refs: FileReference[] }>(`/sessions/${sessionId}/file-refs`)
}

export async function addFileRef(
  sessionId: string,
  data: { name: string; uri: string; auth?: string; description?: string }
): Promise<FileReference> {
  return post<FileReference>(`/sessions/${sessionId}/file-refs`, data)
}

export async function deleteFileRef(
  sessionId: string,
  name: string
): Promise<{ status: string }> {
  return del<{ status: string }>(`/sessions/${sessionId}/file-refs/${name}`)
}

interface UploadResult {
  filename: string
  name?: string
  status: string
  reason?: string
  path?: string
}

interface UploadDocumentsResponse {
  status: string
  indexed_count: number
  total_files: number
  results: UploadResult[]
}

export async function uploadDocuments(
  sessionId: string,
  files: File[]
): Promise<UploadDocumentsResponse> {
  const formData = new FormData()
  for (const file of files) {
    formData.append('files', file)
  }

  const response = await fetch(`/api/sessions/${sessionId}/documents/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }))
    throw new Error(error.detail || 'Upload failed')
  }

  return response.json()
}

// Get document content
export interface DocumentContent {
  name: string
  type?: 'file' | 'content'  // 'file' = open with system app, 'content' = render in modal
  content?: string
  format?: string
  url?: string
  sections?: string[]
  path?: string  // For type='file', the local file path
  metadata?: Record<string, unknown>
}

export async function getDocument(
  sessionId: string,
  documentName: string
): Promise<DocumentContent> {
  const response = await fetch(`/api/sessions/${sessionId}/document?name=${encodeURIComponent(documentName)}`)

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Document not found' }))
    throw new Error(error.detail || 'Failed to fetch document')
  }

  return response.json()
}

// Data Sources (combined: databases, APIs, documents)
export interface SessionApiSource {
  name: string
  type?: string
  description?: string
  base_url?: string
  connected: boolean
  from_config: boolean
  source: string  // 'config', project filename, or 'session'
}

export interface SessionDocumentSource {
  name: string
  type?: string
  description?: string
  path?: string
  indexed: boolean
  from_config: boolean
  source: string  // 'config', project filename, or 'session'
}

export interface DataSourcesResponse {
  databases: SessionDatabase[]
  apis: SessionApiSource[]
  documents: SessionDocumentSource[]
}

export async function listDataSources(
  sessionId: string
): Promise<DataSourcesResponse> {
  return get<DataSourcesResponse>(`/sessions/${sessionId}/sources`)
}

// Databases
export async function listDatabases(
  sessionId: string
): Promise<{ databases: SessionDatabase[] }> {
  return get<{ databases: SessionDatabase[] }>(`/sessions/${sessionId}/databases`)
}

export async function addDatabase(
  sessionId: string,
  data: {
    name: string
    uri?: string
    file_id?: string
    type?: string
    description?: string
  }
): Promise<SessionDatabase> {
  return post<SessionDatabase>(`/sessions/${sessionId}/databases`, data)
}

export async function removeDatabase(
  sessionId: string,
  name: string
): Promise<{ status: string }> {
  return del<{ status: string }>(`/sessions/${sessionId}/databases/${name}`)
}

export async function testDatabase(
  sessionId: string,
  name: string
): Promise<{ name: string; connected: boolean; table_count: number; error?: string }> {
  return post<{ name: string; connected: boolean; table_count: number; error?: string }>(
    `/sessions/${sessionId}/databases/${name}/test`
  )
}

// APIs
export async function addApi(
  sessionId: string,
  data: {
    name: string
    type?: string
    base_url: string
    description?: string
    auth_type?: string
    auth_header?: string
  }
): Promise<SessionApiSource> {
  return post<SessionApiSource>(`/sessions/${sessionId}/apis`, data)
}

export async function removeApi(
  sessionId: string,
  name: string
): Promise<{ status: string }> {
  return del<{ status: string }>(`/sessions/${sessionId}/apis/${name}`)
}

// Config (global, not per-session)
export async function getConfig(): Promise<Config> {
  return get<Config>(`/config`)
}

export async function getSessionRouting(
  sessionId: string
): Promise<Record<string, Record<string, ModelRouteInfo[]>>> {
  return get<Record<string, Record<string, ModelRouteInfo[]>>>(`/sessions/${sessionId}/routing`)
}

// Learnings (global, not per-session)
export async function listLearnings(
  category?: string
): Promise<{ learnings: Learning[]; rules: Rule[] }> {
  const query = category ? `?category=${encodeURIComponent(category)}` : ''
  return get<{ learnings: Learning[]; rules: Rule[] }>(`/learnings${query}`)
}

export async function compactLearnings(): Promise<{
  status: string
  message?: string
  rules_created: number
  learnings_archived: number
}> {
  return post<{
    status: string
    message?: string
    rules_created: number
    learnings_archived: number
  }>('/learnings/compact', {})
}

// Rules (global, not per-session)
export async function addRule(data: {
  summary: string
  category?: string
  confidence?: number
  tags?: string[]
}): Promise<Rule> {
  return post<Rule>('/rules', data)
}

export async function updateRule(
  ruleId: string,
  data: {
    summary?: string
    confidence?: number
    tags?: string[]
  }
): Promise<Rule> {
  return put<Rule>(`/rules/${ruleId}`, data)
}

export async function deleteRule(ruleId: string): Promise<{ status: string; id: string }> {
  return del<{ status: string; id: string }>(`/rules/${ruleId}`)
}

export async function deleteLearning(learningId: string): Promise<{ status: string; id: string }> {
  return del<{ status: string; id: string }>(`/learnings/${learningId}`)
}

export async function downloadSimpleExemplars(params: {
  format?: 'messages' | 'alpaca' | 'sharegpt'
  include?: string[]
  domain?: string
  min_confidence?: number
  since?: string
}): Promise<void> {
  const query = new URLSearchParams()
  if (params.format) query.set('format', params.format)
  if (params.include) query.set('include', params.include.join(','))
  if (params.domain) query.set('domain', params.domain)
  if (params.min_confidence) query.set('min_confidence', String(params.min_confidence))
  if (params.since) query.set('since', params.since)

  // Build headers with auth
  const headers: Record<string, string> = {}
  const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
  if (!isAuthDisabled) {
    const token = await useAuthStore.getState().getToken()
    if (token) headers['Authorization'] = `Bearer ${token}`
  }

  const response = await fetch(`/api/learnings/exemplars/simple?${query}`, { headers })
  if (!response.ok) throw new Error(`Download failed: ${response.statusText}`)
  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `exemplars_${params.format || 'messages'}.jsonl`
  a.click()
  URL.revokeObjectURL(url)
}

// Document URI addition
export async function addDocumentURI(
  sessionId: string,
  body: {
    name: string
    url: string
    description?: string
    headers?: Record<string, string>
    follow_links?: boolean
    max_depth?: number
    max_documents?: number
    same_domain_only?: boolean
    exclude_patterns?: string[]
    type?: string
  }
): Promise<{ status: string; name: string; message: string }> {
  return post<{ status: string; name: string; message: string }>(
    `/sessions/${sessionId}/documents/add-uri`,
    body
  )
}

// Messages (for session restoration)
export interface StoredMessage {
  id: string
  type: 'user' | 'system' | 'plan' | 'step' | 'output' | 'error' | 'thinking'
  content: string
  timestamp: string
  stepNumber?: number
  isFinalInsight?: boolean
  stepDurationMs?: number
  role?: string
  skills?: string[]
}

export async function getMessages(sessionId: string): Promise<{ messages: StoredMessage[] }> {
  return get<{ messages: StoredMessage[] }>(`/sessions/${sessionId}/messages`)
}

export async function saveMessages(
  sessionId: string,
  messages: StoredMessage[]
): Promise<{ status: string; count: number }> {
  return post<{ status: string; count: number }>(`/sessions/${sessionId}/messages`, { messages })
}

// Domains
export interface DomainInfo {
  filename: string
  name: string
  description: string
  tier: string
  active: boolean
  owner: string
}

export async function listDomains(): Promise<{ domains: DomainInfo[] }> {
  return get<{ domains: DomainInfo[] }>('/domains')
}

export async function getDomain(filename: string): Promise<{
  filename: string
  name: string
  description: string
  databases: string[]
  apis: string[]
  documents: string[]
}> {
  return get(`/domains/${encodeURIComponent(filename)}`)
}

export async function setActiveDomains(
  sessionId: string,
  domains: string[]
): Promise<{ status: string; session_id: string; active_domains: string[] }> {
  return post(`/sessions/${sessionId}/domains`, { domains })
}

export async function getDomainContent(
  filename: string
): Promise<{ content: string; path: string; filename: string }> {
  return get(`/domains/${encodeURIComponent(filename)}/content`)
}

export async function createDomain(
  name: string,
  description: string = '',
  parentDomain: string = '',
  initialDomains: string[] = [],
  systemPrompt: string = ''
): Promise<{ status: string; filename: string; name: string; description: string }> {
  return post('/domains', { name, description, parent_domain: parentDomain, initial_domains: initialDomains, system_prompt: systemPrompt })
}

export async function updateDomainContent(
  filename: string,
  content: string
): Promise<{ status: string; filename: string; path: string }> {
  return put(`/domains/${encodeURIComponent(filename)}/content`, { content })
}

export async function updateDomain(
  filename: string,
  data: { name?: string; description?: string; order?: number; active?: boolean }
): Promise<{ status: string; filename: string }> {
  return patch(`/domains/${encodeURIComponent(filename)}`, data)
}

export async function deleteDomain(
  filename: string
): Promise<{ status: string; filename: string }> {
  return del(`/domains/${encodeURIComponent(filename)}`)
}

// Domain-scoped content
export interface DomainSkillInfo {
  name: string
  description: string
  domain: string
}

export interface DomainAgentInfo {
  name: string
  description: string
  domain: string
}

export interface DomainRuleInfo {
  id: string
  summary: string
  category: string
  confidence: number
  domain: string
}

export async function listDomainSkills(
  filename: string
): Promise<{ skills: DomainSkillInfo[] }> {
  return get(`/domains/${encodeURIComponent(filename)}/skills`)
}

export async function listDomainAgents(
  filename: string
): Promise<{ agents: DomainAgentInfo[] }> {
  return get(`/domains/${encodeURIComponent(filename)}/agents`)
}

export async function listDomainRules(
  filename: string
): Promise<{ rules: DomainRuleInfo[] }> {
  return get(`/domains/${encodeURIComponent(filename)}/rules`)
}

export async function moveSkill(body: {
  skill_name: string
  from_domain: string
  to_domain: string
}): Promise<{ status: string }> {
  return post('/domains/move-skill', body)
}

export async function moveAgent(body: {
  agent_name: string
  from_domain: string
  to_domain: string
}): Promise<{ status: string }> {
  return post('/domains/move-agent', body)
}

export async function moveRule(body: {
  rule_id: string
  to_domain: string
}): Promise<{ status: string }> {
  return post('/domains/move-rule', body)
}

export async function promoteDomain(
  filename: string,
  targetName?: string
): Promise<{ status: string; filename: string; new_tier: string }> {
  return post(`/domains/${encodeURIComponent(filename)}/promote`, { target_name: targetName })
}

// Prompt Context
export interface ActiveAgent {
  name: string
  prompt: string
}

export interface ActiveSkill {
  name: string
  prompt: string
  description: string
}

export interface PromptContext {
  system_prompt: string
  active_agent: ActiveAgent | null
  active_skills: ActiveSkill[]
}

export async function getPromptContext(sessionId: string): Promise<PromptContext> {
  return get<PromptContext>(`/sessions/${sessionId}/prompt-context`)
}

export async function updateSystemPrompt(
  sessionId: string,
  systemPrompt: string
): Promise<{ status: string; system_prompt: string }> {
  return put<{ status: string; system_prompt: string }>(
    `/sessions/${sessionId}/system-prompt`,
    { system_prompt: systemPrompt }
  )
}

// User permissions
export interface UserPermissions {
  user_id: string
  email: string | null
  persona: string
  domains: string[]
  databases: string[]
  documents: string[]
  apis: string[]
  visibility: Record<string, boolean>
  writes: Record<string, boolean>
  feedback: Record<string, boolean>
}

export async function getMyPermissions(): Promise<UserPermissions> {
  return get<UserPermissions>('/users/me/permissions')
}

// Reset context for new query (clears session state, keeps user settings)
export async function resetContext(sessionId: string): Promise<{ status: string }> {
  return post<{ status: string }>(`/sessions/${sessionId}/reset-context`, {})
}

// Proof facts storage (for restoring proof panel on session resume)
export interface StoredProofFact {
  id: string
  name: string
  description?: string
  status: 'pending' | 'planning' | 'executing' | 'resolved' | 'failed' | 'blocked'
  value?: unknown
  source?: string
  confidence?: number
  tier?: number
  strategy?: string
  formula?: string
  reason?: string
  dependencies: string[]
  elapsed_ms?: number
}

export async function saveProofFacts(
  sessionId: string,
  facts: StoredProofFact[],
  summary?: string | null
): Promise<{ status: string; count: number }> {
  return post<{ status: string; count: number }>(
    `/sessions/${sessionId}/proof-facts`,
    { facts, summary }
  )
}

export async function getProofFacts(
  sessionId: string
): Promise<{ facts: StoredProofFact[]; summary: string | null }> {
  return get<{ facts: StoredProofFact[]; summary: string | null }>(
    `/sessions/${sessionId}/proof-facts`
  )
}

// Fine-Tuning
export async function listFineTuneJobs(params?: {
  status?: string
  domain?: string
}): Promise<FineTuneJob[]> {
  const query = new URLSearchParams()
  if (params?.status) query.set('status', params.status)
  if (params?.domain) query.set('domain', params.domain)
  const qs = query.toString()
  return get<FineTuneJob[]>(`/fine-tune/jobs${qs ? `?${qs}` : ''}`)
}

export async function startFineTuneJob(body: {
  name: string
  provider: string
  base_model: string
  task_types: string[]
  domain?: string
  include?: string[]
  min_confidence?: number
  hyperparams?: Record<string, unknown>
}): Promise<FineTuneJob> {
  return post<FineTuneJob>('/fine-tune/jobs', body)
}

export async function getFineTuneJob(modelId: string): Promise<FineTuneJob> {
  return get<FineTuneJob>(`/fine-tune/jobs/${modelId}`)
}

export async function cancelFineTuneJob(modelId: string): Promise<void> {
  await post(`/fine-tune/jobs/${modelId}/cancel`)
}

export async function deleteFineTuneJob(modelId: string): Promise<void> {
  await del(`/fine-tune/jobs/${modelId}`)
}

export async function listFineTuneProviders(): Promise<FineTuneProvider[]> {
  return get<FineTuneProvider[]>('/fine-tune/providers')
}