// Session API calls

import { get, post, del } from './client'
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