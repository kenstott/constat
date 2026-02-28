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
  | 'csv'
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

export type FactSource = 'cache' | 'database' | 'document' | 'api' | 'llm' | 'derived' | 'user' | 'config'

export type EntityType =
  | 'table'
  | 'column'
  | 'concept'
  | 'action'
  | 'business_term'
  | 'api'
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
  active_domains?: string[] // Active domain filenames
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
  role_id?: string | null
  skill_ids?: string[] | null
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
  role_id?: string | null  // Role provenance - which role created this table
  version?: number
  version_count?: number
}

export interface TableVersionInfo {
  version: number
  step_number?: number
  row_count: number
  created_at?: string
}

export interface TableVersionsResponse {
  name: string
  current_version: number
  versions: TableVersionInfo[]
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
  metadata?: Record<string, unknown>
  role_id?: string | null  // Role provenance - which role created this artifact
  version?: number
  version_count?: number
}

export interface ArtifactVersionInfo {
  id: number
  version: number
  step_number: number
  attempt: number
  created_at?: string
}

export interface ArtifactVersionsResponse {
  name: string
  current_version: number
  versions: ArtifactVersionInfo[]
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
  role_id?: string | null  // Role provenance - which role created this fact
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

// Glossary
export type GlossaryStatus = 'defined' | 'self_describing'
export type GlossaryEditorialStatus = 'draft' | 'reviewed' | 'approved'
export type GlossaryProvenance = 'llm' | 'human' | 'hybrid'

export interface GlossaryTerm {
  name: string
  display_name: string
  definition?: string | null
  domain?: string | null
  domain_path?: string | null
  parent_id?: string | null
  parent_verb?: 'HAS_ONE' | 'HAS_KIND' | 'HAS_MANY'
  parent?: { name: string; display_name: string } | null
  aliases: string[]
  semantic_type?: string | null
  cardinality: string
  status?: GlossaryEditorialStatus | null
  provenance?: GlossaryProvenance | null
  glossary_status: GlossaryStatus
  entity_id?: string | null
  glossary_id?: string | null
  ner_type?: string | null
  ignored?: boolean
  connected_resources: Array<{
    entity_name: string
    entity_type: string
    sources: Array<{ document_name: string; source: string; section?: string; url?: string }>
  }>
  children?: Array<{ name: string; display_name: string; parent_verb?: 'HAS_ONE' | 'HAS_KIND' | 'HAS_MANY' }>
  relationships?: Array<{
    id: string
    subject: string
    verb: string
    object: string
    confidence: number
    user_edited?: boolean
  }>
  cluster_siblings?: string[]
}

export interface GlossaryListResponse {
  terms: GlossaryTerm[]
  total_defined: number
  total_self_describing: number
}

export interface GlossaryFilter {
  scope?: 'all' | 'defined' | 'self_describing'
  status?: GlossaryEditorialStatus
  type?: string
  search?: string
  domain?: string
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
  source?: string  // 'config', project filename, or 'session'
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
  from_config?: boolean  // true if from config file (cannot be removed)
  source?: string  // 'config', project filename, or 'session'
}

// Document Source Info
export interface DocumentSourceInfo {
  name: string
  type?: string
  description?: string
  path?: string
  indexed: boolean
  from_config?: boolean  // true if from config file (cannot be removed)
  source?: string  // 'config', project filename, or 'session'
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
  | 'proof_start'
  | 'replanning'
  | 'plan_ready'
  | 'plan_approved'
  | 'plan_rejected'
  | 'dynamic_context'
  | 'step_start'
  | 'step_generating'
  | 'step_executing'
  | 'step_complete'
  | 'step_error'
  | 'step_failed'
  | 'validation_retry'
  | 'validation_warnings'
  | 'facts_extracted'
  | 'fact_resolved'
  | 'fact_start'
  | 'fact_planning'
  | 'fact_executing'
  | 'fact_failed'
  | 'fact_blocked'
  | 'dag_execution_start'
  | 'inference_code'
  | 'proof_complete'
  | 'proof_summary_ready'
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
  | 'entity_rebuild_start'
  | 'entity_rebuild_complete'
  | 'glossary_rebuild_start'
  | 'glossary_rebuild_complete'
  | 'glossary_terms_added'
  | 'glossary_generation_progress'
  | 'relationships_extracted'

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