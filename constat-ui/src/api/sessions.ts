// Session API calls

import { get, post, put, del } from './client'
import type {
  Session,
  SessionListResponse,
  TableInfo,
  TableData,
  Artifact,
  ArtifactContent,
  Fact,
  Entity,
  UploadedFile,
  FileReference,
  SessionDatabase,
  Config,
  Learning,
  Rule,
} from '@/types/api'

// Session CRUD
export async function createSession(userId = 'default'): Promise<Session> {
  return post<Session>('/sessions', { user_id: userId })
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