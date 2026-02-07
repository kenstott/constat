// Session API calls

import { get, post, put, del } from './client'
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
  UploadedFile,
  FileReference,
  SessionDatabase,
  Config,
  Learning,
  Rule,
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

// Star/Promote
export async function toggleArtifactStar(
  sessionId: string,
  artifactId: number
): Promise<{ artifact_id: number; is_starred: boolean }> {
  return post<{ artifact_id: number; is_starred: boolean }>(
    `/sessions/${sessionId}/artifacts/${artifactId}/star`
  )
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
}

export async function listInferenceCodes(
  sessionId: string
): Promise<{ inferences: InferenceCode[]; total: number }> {
  return get<{ inferences: InferenceCode[]; total: number }>(
    `/sessions/${sessionId}/inference-codes`
  )
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

// Messages (for session restoration)
export interface StoredMessage {
  id: string
  type: 'user' | 'system' | 'plan' | 'step' | 'output' | 'error' | 'thinking'
  content: string
  timestamp: string
  stepNumber?: number
  isFinalInsight?: boolean
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

// Projects
export interface ProjectInfo {
  filename: string
  name: string
  description: string
}

export async function listProjects(): Promise<{ projects: ProjectInfo[] }> {
  return get<{ projects: ProjectInfo[] }>('/projects')
}

export async function getProject(filename: string): Promise<{
  filename: string
  name: string
  description: string
  databases: string[]
  apis: string[]
  documents: string[]
}> {
  return get(`/projects/${encodeURIComponent(filename)}`)
}

export async function setActiveProjects(
  sessionId: string,
  projects: string[]
): Promise<{ status: string; session_id: string; active_projects: string[] }> {
  return post(`/sessions/${sessionId}/projects`, { projects })
}

export async function getProjectContent(
  filename: string
): Promise<{ content: string; path: string; filename: string }> {
  return get(`/projects/${encodeURIComponent(filename)}/content`)
}

export async function updateProjectContent(
  filename: string,
  content: string
): Promise<{ status: string; filename: string; path: string }> {
  return put(`/projects/${encodeURIComponent(filename)}/content`, { content })
}

// Prompt Context
export interface ActiveRole {
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
  active_role: ActiveRole | null
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
  admin: boolean
  projects: string[]
  databases: string[]
  documents: string[]
  apis: string[]
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