// API Types - mirrors server models

export type SessionStatus =
  | 'idle'
  | 'planning'
  | 'awaiting_approval'
  | 'executing'
  | 'completed'
  | 'error'
  | 'cancelled'

export type StepStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped'

export type TaskType =
  | 'planning'
  | 'replanning'
  | 'sql_generation'
  | 'python_analysis'
  | 'intent_classification'
  | 'mode_selection'
  | 'fact_resolution'
  | 'summarization'
  | 'embedding'
  | 'quick_response'

export type ArtifactType =
  | 'code'
  | 'output'
  | 'error'
  | 'table'
  | 'json'
  | 'html'
  | 'markdown'
  | 'text'
  | 'chart'
  | 'plotly'
  | 'vega'
  | 'svg'
  | 'png'
  | 'jpeg'
  | 'mermaid'
  | 'graphviz'
  | 'diagram'
  | 'react'
  | 'javascript'

export type FactSource = 'cache' | 'database' | 'document' | 'api' | 'llm' | 'derived' | 'user'

export type EntityType =
  | 'table'
  | 'column'
  | 'concept'
  | 'business_term'
  | 'api_endpoint'
  | 'api_field'
  | 'api_schema'

export type NodeStatus = 'pending' | 'resolving' | 'resolved' | 'failed' | 'cached'

// Session
export interface Session {
  session_id: string
  user_id: string
  status: SessionStatus
  created_at: string
  last_activity: string
  current_query?: string
  summary?: string // LLM-generated session summary
  tables_count: number
  artifacts_count: number
}

export interface SessionListResponse {
  sessions: Session[]
  total: number
}

// Plan & Steps
export interface StepResult {
  success: boolean
  stdout: string
  error?: string
  attempts: number
  duration_ms: number
  tables_created: string[]
  tables_modified: string[]
  variables?: Record<string, unknown>
  code: string
  suggestions?: FailureSuggestion[]
}

export interface FailureSuggestion {
  approach: string
  reason: string
}

export interface Step {
  number: number
  goal: string
  status: StepStatus
  expected_inputs: string[]
  expected_outputs: string[]
  depends_on: number[]
  code?: string
  result?: StepResult
}

export interface Plan {
  problem: string
  steps: Step[]
  current_step: number
  completed_steps: number[]
  failed_steps: number[]
  is_complete: boolean
}

// Query
export interface QueryRequest {
  problem: string
  is_followup?: boolean
}

export interface QueryResponse {
  execution_id: string
  status: string
  message: string
}

// Tables
export interface TableInfo {
  name: string
  row_count: number
  step_number: number
  columns: string[]
  is_starred?: boolean
}

export interface TableData {
  name: string
  columns: string[]
  data: Record<string, unknown>[]
  total_rows: number
  page: number
  page_size: number
  has_more: boolean
}

// Artifacts
export interface Artifact {
  id: number
  name: string
  artifact_type: ArtifactType
  step_number: number
  title?: string
  description?: string
  mime_type: string
  created_at?: string
  is_key_result?: boolean
  is_starred?: boolean
}

export interface ArtifactContent extends Artifact {
  content: string
  is_binary: boolean
}

// Facts
export interface Fact {
  name: string
  value: unknown
  source: FactSource
  reasoning?: string
  confidence?: number
  is_persisted: boolean
}

// Entity Reference
export interface EntityReference {
  document: string
  section?: string
  mentions: number
  mention_text?: string  // Exact text as it appeared in the source
}

// Entities
export interface Entity {
  id: string
  name: string  // Normalized display name
  type: EntityType
  sources: string[]
  metadata: Record<string, unknown>
  references: EntityReference[]
  created_at?: string
  mention_count: number
  original_name?: string  // Original name before normalization (if different)
}

// Proof Tree
export interface ProofNode {
  name: string
  description: string
  status: NodeStatus
  value?: unknown
  source: string
  confidence: number
  children: ProofNode[]
  query?: string
  error?: string
  result_summary?: string
  depth: number
  all_dependencies: string[]
  visual_parent?: string
}

// Files
export interface UploadedFile {
  id: string
  filename: string
  file_uri: string
  size_bytes: number
  content_type: string
  uploaded_at: string
}

export interface FileReference {
  name: string
  uri: string
  has_auth: boolean
  description?: string
  added_at: string
  session_id?: string
}

// Databases
export interface SessionDatabase {
  name: string
  type: string
  dialect?: string
  description?: string
  connected: boolean
  table_count?: number
  added_at: string
  is_dynamic: boolean
  file_id?: string
}

// Learnings
export interface Learning {
  id: string
  content: string
  category: string
  source: string
  context?: Record<string, unknown>
  applied_count: number
  created_at: string
}

// Rules (compacted learnings)
export interface Rule {
  id: string
  summary: string
  category: string
  confidence: number
  source_count: number
  tags: string[]
}

// Config
export interface Config {
  databases: string[]
  apis: string[]
  documents: string[]
  llm_provider: string
  llm_model: string
  execution_timeout: number
}

// API Source Info
export interface ApiSourceInfo {
  name: string
  type?: string
  description?: string
  base_url?: string
  connected: boolean
}

// Document Source Info
export interface DocumentSourceInfo {
  name: string
  type?: string
  description?: string
  path?: string
  indexed: boolean
}

// Autocomplete
export interface CompletionItem {
  label: string
  value: string
  description?: string
  category?: string
}

// WebSocket Events
export type EventType =
  | 'welcome'
  | 'session_created'
  | 'session_closed'
  | 'planning_start'
  | 'plan_ready'
  | 'plan_approved'
  | 'plan_rejected'
  | 'step_start'
  | 'step_generating'
  | 'step_executing'
  | 'step_complete'
  | 'step_error'
  | 'step_failed'
  | 'facts_extracted'
  | 'fact_resolved'
  | 'progress'
  | 'query_complete'
  | 'query_error'
  | 'query_cancelled'
  | 'table_created'
  | 'artifact_created'
  | 'clarification_needed'
  | 'clarification_received'
  | 'autocomplete_response'
  | 'synthesizing'
  | 'generating_insights'

export interface WSEvent {
  event_type: EventType
  session_id: string
  step_number: number
  timestamp: string
  data: Record<string, unknown>
}

export interface WSMessage {
  type: 'event' | 'ack' | 'error'
  payload: Record<string, unknown>
}