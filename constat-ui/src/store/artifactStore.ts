// Copyright (c) 2025 Kenneth Stott
// Canary: f8ab86d2-744a-4793-8fd5-745bcd2cba8e
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// Artifact state store

import { create } from './createStore'
import type { Artifact, ArtifactContent, TableInfo, Fact, Entity, SessionDatabase, ApiSourceInfo, DocumentSourceInfo, Learning, Rule, ModelRouteInfo } from '@/types/api'
import * as sessionsApi from '@/api/sessions'
import * as skillsApi from '@/api/skills'
import { apolloClient } from '@/graphql/client'
import {
  STEPS_QUERY, INFERENCE_CODES_QUERY, SCRATCHPAD_QUERY, SESSION_DDL_QUERY,
  SESSION_ROUTING_QUERY, PROMPT_CONTEXT_QUERY, UPDATE_SYSTEM_PROMPT,
  toStepCode, toInferenceCode, toScratchpadEntry, toPromptContext,
} from '@/graphql/operations/state'
import {
  TABLES_QUERY, ARTIFACTS_QUERY, FACTS_QUERY, ENTITIES_QUERY,
  ARTIFACT_QUERY,
  DELETE_TABLE, DELETE_ARTIFACT, TOGGLE_TABLE_STAR, TOGGLE_ARTIFACT_STAR,
  PERSIST_FACT, FORGET_FACT,
  toTableInfo, toArtifact, toArtifactContent, toFact, toEntity,
} from '@/graphql/operations/data'
import {
  DATABASES_QUERY, DATA_SOURCES_QUERY,
  toSessionDatabase, toSessionApi, toSessionDocument,
} from '@/graphql/operations/sources'
import {
  LEARNINGS_QUERY, SKILLS_QUERY, ACTIVATE_AGENT, SET_ACTIVE_SKILLS,
  CREATE_RULE, UPDATE_RULE as UPDATE_RULE_MUTATION, DELETE_RULE as DELETE_RULE_MUTATION,
  DELETE_LEARNING as DELETE_LEARNING_MUTATION,
  toLearningInfo, toRuleInfo, toSkillInfo,
} from '@/graphql/operations/learnings'

// Step code from execution (matches API response)
interface StepCode {
  step_number: number
  goal: string
  code: string
  prompt?: string
  model?: string
}

// Scratchpad entry (execution narrative per step)
interface ScratchpadEntry {
  step_number: number
  goal: string
  narrative: string
  tables_created: string[]
  code: string
  user_query: string
  objective_index: number | null
}

// Inference code from auditable mode (matches API response)
interface InferenceCode {
  inference_id: string
  name: string
  operation: string
  code: string
  attempt: number
  prompt?: string
  model?: string
}

// Prompt context types
interface PromptContext {
  systemPrompt: string
  activeAgent: { name: string; prompt: string } | null
  activeSkills: Array<{ name: string; prompt: string; description: string }>
}

// Skill info (all skills, not just active)
interface SkillInfo {
  name: string
  prompt: string
  description: string
  filename: string
  is_active: boolean
  domain: string
  source: string
}

// Agent info (all agents, not just active)
interface AgentInfo {
  name: string
  prompt: string
  is_active: boolean
  domain: string
  source: string
}

// User permissions
interface UserPermissions {
  isAdmin: boolean
  persona: string
  visibility: Record<string, boolean>
  writes: Record<string, boolean>
}

interface ArtifactState {
  // Data
  artifacts: Artifact[]
  tables: TableInfo[]
  facts: Fact[]
  entities: Entity[]
  learnings: Learning[]
  rules: Rule[]
  databases: SessionDatabase[]
  apis: ApiSourceInfo[]
  documents: DocumentSourceInfo[]
  stepCodes: StepCode[]
  inferenceCodes: InferenceCode[]
  scratchpadEntries: ScratchpadEntry[]
  sessionDDL: string
  promptContext: PromptContext | null
  taskRouting: Record<string, Record<string, ModelRouteInfo[]>> | null
  allSkills: SkillInfo[]
  allAgents: AgentInfo[]
  userPermissions: UserPermissions

  // Selected items
  selectedArtifact: ArtifactContent | null
  selectedTable: string | null

  // Loading states
  loading: boolean
  sourcesLoading: boolean
  factsLoading: boolean
  learningsLoading: boolean
  configLoading: boolean
  error: string | null
  ingestingSource: string | null
  ingestProgress: { current: number; total: number } | null

  // Actions
  fetchArtifacts: (sessionId: string) => Promise<void>
  fetchTables: (sessionId: string) => Promise<void>
  fetchFacts: (sessionId: string) => Promise<void>
  fetchEntities: (sessionId: string, entityType?: string) => Promise<void>
  fetchLearnings: () => Promise<void>
  fetchStepCodes: (sessionId: string) => Promise<void>
  fetchInferenceCodes: (sessionId: string) => Promise<void>
  fetchScratchpad: (sessionId: string) => Promise<void>
  fetchDDL: (sessionId: string) => Promise<void>
  fetchDatabases: (sessionId: string) => Promise<void>
  fetchDataSources: (sessionId: string) => Promise<void>
  fetchPromptContext: (sessionId: string) => Promise<void>
  fetchTaskRouting: (sessionId: string) => Promise<void>
  fetchAllSkills: () => Promise<void>
  createSkill: (name: string, prompt: string, description?: string) => Promise<void>
  updateSkill: (name: string, content: string) => Promise<void>
  deleteSkill: (name: string) => Promise<void>
  toggleSkillActive: (name: string, sessionId: string) => Promise<void>
  draftSkill: (sessionId: string, name: string, description: string) => Promise<{ content: string; description: string }>
  fetchAllAgents: (sessionId: string) => Promise<void>
  setActiveAgent: (agentName: string | null, sessionId: string) => Promise<void>
  fetchPermissions: () => Promise<void>
  updateSystemPrompt: (sessionId: string, systemPrompt: string) => Promise<void>
  selectArtifact: (sessionId: string, artifactId: number) => Promise<void>
  selectTable: (tableName: string | null) => void
  persistFact: (sessionId: string, factName: string) => Promise<void>
  forgetFact: (sessionId: string, factName: string) => Promise<void>
  // Real-time update methods
  addTable: (table: TableInfo) => void
  updateTable: (tableName: string, updates: Partial<TableInfo>) => void
  addArtifact: (artifact: Artifact) => void
  addFact: (fact: Fact) => void
  addStepCode: (stepNumber: number, goal: string, code: string, model?: string) => void
  // Delete artifact/table
  removeArtifact: (sessionId: string, artifactId: number) => Promise<void>
  removeTable: (sessionId: string, tableName: string) => Promise<void>
  // Star/promote actions (persist to server)
  toggleArtifactStar: (sessionId: string, artifactId: number) => Promise<void>
  toggleTableStar: (sessionId: string, tableName: string) => Promise<void>
  // Rule management
  addRule: (summary: string, tags?: string[]) => Promise<void>
  updateRule: (ruleId: string, summary: string, tags?: string[]) => Promise<void>
  deleteRule: (ruleId: string) => Promise<void>
  deleteLearning: (learningId: string) => Promise<void>
  patchEntities: (diff: { added: Array<{name: string, type: string}>, removed: Array<{name: string, type: string}> }) => void
  addInferenceCode: (ic: InferenceCode) => void
  supersededStepNumbers: Set<number>
  markStepsSuperseded: () => void  // Mark all current step numbers as superseded (on redo)
  setIngestingSource: (name: string | null) => void
  setIngestProgress: (progress: { current: number; total: number } | null) => void
  clear: () => void
  clearQueryResults: () => void  // Clear artifacts/tables/facts/stepCodes but keep data sources/entities/learnings
  clearInferenceCodes: () => void  // Clear inference codes on proof re-run
  truncateFromStep: (fromStep: number, tablesDropped?: string[]) => void  // Remove artifacts/tables/codes from step N onwards
}

export const useArtifactStore = create<ArtifactState>((set, get) => ({
  artifacts: [],
  tables: [],
  facts: [],
  stepCodes: [],
  inferenceCodes: [],
  scratchpadEntries: [],
  sessionDDL: '',
  entities: [],
  learnings: [],
  rules: [],
  databases: [],
  apis: [],
  documents: [],
  promptContext: null,
  taskRouting: null,
  allSkills: [],
  allAgents: [],
  userPermissions: { isAdmin: false, persona: 'viewer', visibility: {}, writes: {} },
  selectedArtifact: null,
  selectedTable: null,
  supersededStepNumbers: new Set<number>(),
  loading: false,
  sourcesLoading: true,
  factsLoading: true,
  learningsLoading: true,
  configLoading: true,
  error: null,
  ingestingSource: null,
  ingestProgress: null,

  fetchArtifacts: async (sessionId) => {
    set({ loading: true, error: null })
    try {
      const { data } = await apolloClient.query({ query: ARTIFACTS_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
      set({ artifacts: data.artifacts.artifacts.map(toArtifact), loading: false })
    } catch (error) {
      set({ error: String(error), loading: false })
    }
  },

  fetchTables: async (sessionId) => {
    set({ loading: true, error: null })
    try {
      const { data } = await apolloClient.query({ query: TABLES_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
      set({ tables: data.tables.tables.map(toTableInfo), loading: false })
    } catch (error) {
      set({ error: String(error), loading: false })
    }
  },

  fetchFacts: async (sessionId) => {
    set({ factsLoading: true, error: null })
    try {
      const { data } = await apolloClient.query({ query: FACTS_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
      set({ facts: data.facts.facts.map(toFact), factsLoading: false })
    } catch (error) {
      set({ error: String(error), factsLoading: false })
    }
  },

  fetchEntities: async (sessionId, entityType) => {
    set({ loading: true, error: null })
    try {
      const { data } = await apolloClient.query({ query: ENTITIES_QUERY, variables: { sessionId, entityType: entityType ?? null }, fetchPolicy: 'network-only' })
      set({ entities: data.entities.entities.map(toEntity), loading: false })
    } catch (error) {
      set({ error: String(error), loading: false })
    }
  },

  fetchLearnings: async () => {
    set({ learningsLoading: true, error: null })
    try {
      const { data } = await apolloClient.query({ query: LEARNINGS_QUERY, fetchPolicy: 'network-only' })
      const learnings = data.learnings.learnings.map(toLearningInfo)
      const rules = data.learnings.rules.map(toRuleInfo)
      set({ learnings, rules, learningsLoading: false })
    } catch (error) {
      set({ error: String(error), learningsLoading: false })
    }
  },

  fetchStepCodes: async (sessionId) => {
    try {
      const { data } = await apolloClient.query({ query: STEPS_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
      const sorted = data.steps.steps.map(toStepCode).sort((a: any, b: any) => a.step_number - b.step_number)
      set({ stepCodes: sorted })
    } catch (error) {
      console.warn('Failed to fetch step codes:', error)
    }
  },

  fetchInferenceCodes: async (sessionId) => {
    try {
      const { data } = await apolloClient.query({ query: INFERENCE_CODES_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
      set({ inferenceCodes: data.inferenceCodes.inferences.map(toInferenceCode).filter((ic: any) => ic.inference_id) })
    } catch (error) {
      console.warn('Failed to fetch inference codes:', error)
    }
  },

  fetchScratchpad: async (sessionId) => {
    try {
      const { data } = await apolloClient.query({ query: SCRATCHPAD_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
      set({ scratchpadEntries: data.scratchpad.entries.map(toScratchpadEntry) })
    } catch (error) {
      console.warn('Failed to fetch scratchpad:', error)
    }
  },

  fetchDDL: async (sessionId) => {
    try {
      const { data } = await apolloClient.query({ query: SESSION_DDL_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
      set({ sessionDDL: data.sessionDdl })
    } catch (error) {
      console.warn('Failed to fetch DDL:', error)
    }
  },

  fetchDatabases: async (sessionId) => {
    set({ loading: true, error: null })
    try {
      const { data } = await apolloClient.query({ query: DATABASES_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
      set({ databases: data.databases.databases.map(toSessionDatabase), loading: false })
    } catch (error) {
      set({ error: String(error), loading: false })
    }
  },

  fetchDataSources: async (sessionId) => {
    set({ sourcesLoading: true, error: null })
    try {
      const { data } = await apolloClient.query({ query: DATA_SOURCES_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
      const ds = data.dataSources
      set({
        databases: ds.databases.map(toSessionDatabase),
        apis: ds.apis.map(toSessionApi),
        documents: ds.documents.map(toSessionDocument),
        sourcesLoading: false,
      })
    } catch (error) {
      set({ error: String(error), sourcesLoading: false })
    }
  },

  fetchPromptContext: async (sessionId) => {
    try {
      const { data } = await apolloClient.query({ query: PROMPT_CONTEXT_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
      const ctx = toPromptContext(data.promptContext)
      set({
        promptContext: {
          systemPrompt: ctx.system_prompt,
          activeAgent: ctx.active_agent,
          activeSkills: ctx.active_skills,
        },
      })
    } catch (error) {
      console.warn('Failed to fetch prompt context:', error)
    }
  },

  fetchTaskRouting: async (sessionId: string) => {
    try {
      const { data } = await apolloClient.query({ query: SESSION_ROUTING_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
      set({ taskRouting: data.sessionRouting })
    } catch (error) {
      console.warn('Failed to fetch task routing:', error)
    }
  },

  fetchAllSkills: async () => {
    try {
      const { data } = await apolloClient.query({ query: SKILLS_QUERY, fetchPolicy: 'network-only' })
      set({
        allSkills: data.skills.skills.map(toSkillInfo),
        configLoading: false,
      })
    } catch (error) {
      console.warn('Failed to fetch skills:', error)
      set({ configLoading: false })
    }
  },

  createSkill: async (name, prompt, description = '') => {
    try {
      await skillsApi.createSkill(name, prompt, description)
      get().fetchAllSkills()
    } catch (error) {
      set({ error: String(error) })
      throw error
    }
  },

  updateSkill: async (name, content) => {
    try {
      await skillsApi.updateSkillContent(name, content)
      get().fetchAllSkills()
    } catch (error) {
      set({ error: String(error) })
      throw error
    }
  },

  deleteSkill: async (name) => {
    const skill = get().allSkills.find(s => s.name === name)
    const domain = skill?.domain
    set(state => ({ allSkills: state.allSkills.filter(s => s.name !== name) }))
    try {
      await skillsApi.deleteSkill(name, domain)
      get().fetchAllSkills()
    } catch (error) {
      get().fetchAllSkills()
      set({ error: String(error) })
      throw error
    }
  },

  toggleSkillActive: async (name, sessionId) => {
    const { allSkills } = get()
    const skill = allSkills.find((s) => s.name === name)
    if (!skill) return

    // Build new active list
    const currentActive = allSkills.filter((s) => s.is_active).map((s) => s.name)
    const newActive = skill.is_active
      ? currentActive.filter((n) => n !== name)
      : [...currentActive, name]

    try {
      await apolloClient.mutate({ mutation: SET_ACTIVE_SKILLS, variables: { skillNames: newActive } })
      get().fetchAllSkills()
      get().fetchPromptContext(sessionId)
    } catch (error) {
      set({ error: String(error) })
    }
  },

  draftSkill: async (sessionId, name, description) => {
    try {
      const result = await skillsApi.draftSkill(sessionId, name, description)
      return { content: result.content, description: result.description }
    } catch (error) {
      set({ error: String(error) })
      throw error
    }
  },

  fetchAllAgents: async (sessionId) => {
    try {
      const { getAuthHeaders } = await import('@/config/auth-helpers')
      const headers = await getAuthHeaders()

      const response = await fetch(`/api/sessions/agents?session_id=${sessionId}`, {
        headers,
        credentials: 'include',
      })
      if (response.ok) {
        const data = await response.json()
        set({
          allAgents: (data.agents || []).map((r: { name: string; prompt: string; is_active: boolean; domain?: string; source?: string }) => ({
            name: r.name,
            prompt: r.prompt,
            is_active: r.is_active,
            domain: r.domain || '',
            source: r.source || '',
          })),
        })
      } else {
        console.warn('Failed to fetch agents:', response.status, response.statusText)
      }
    } catch (error) {
      console.warn('Failed to fetch agents:', error)
    }
  },

  setActiveAgent: async (agentName, sessionId) => {
    try {
      await apolloClient.mutate({ mutation: ACTIVATE_AGENT, variables: { sessionId, agentName } })
      get().fetchAllAgents(sessionId)
      get().fetchPromptContext(sessionId)
    } catch (error) {
      set({ error: String(error) })
    }
  },

  fetchPermissions: async () => {
    const { isAuthDisabled, getToken } = await import('@/config/auth-helpers')
    if (isAuthDisabled) {
      set({ userPermissions: {
        isAdmin: true, persona: 'platform_admin',
        visibility: { results: true, databases: true, apis: true, documents: true, system_prompt: true, agents: true, skills: true, learnings: true, code: true, inference_code: true, facts: true, glossary: true },
        writes: { sources: true, glossary: true, skills: true, agents: true, facts: true, learnings: true, system_prompt: true, tier_promote: true },
      } })
      return
    }
    const token = await getToken()
    if (!token) {
      return
    }
    try {
      const perms = await sessionsApi.getMyPermissions()
      set({ userPermissions: { isAdmin: perms.persona === 'platform_admin', persona: perms.persona, visibility: perms.visibility, writes: perms.writes } })
    } catch (error) {
      console.warn('Failed to fetch permissions:', error)
    }
  },

  updateSystemPrompt: async (sessionId, systemPrompt) => {
    try {
      await apolloClient.mutate({ mutation: UPDATE_SYSTEM_PROMPT, variables: { sessionId, systemPrompt } })
      get().fetchPromptContext(sessionId)
    } catch (error) {
      set({ error: String(error) })
      throw error
    }
  },

  selectArtifact: async (sessionId, artifactId) => {
    set({ loading: true, error: null })
    try {
      const { data } = await apolloClient.query({ query: ARTIFACT_QUERY, variables: { sessionId, id: artifactId }, fetchPolicy: 'network-only' })
      set({ selectedArtifact: toArtifactContent(data.artifact), loading: false })
    } catch (error) {
      set({ error: String(error), loading: false })
    }
  },

  selectTable: (tableName) => set({ selectedTable: tableName }),

  persistFact: async (sessionId, factName) => {
    await apolloClient.mutate({ mutation: PERSIST_FACT, variables: { sessionId, factName } })
    get().fetchFacts(sessionId)
  },

  forgetFact: async (sessionId, factName) => {
    await apolloClient.mutate({ mutation: FORGET_FACT, variables: { sessionId, factName } })
    get().fetchFacts(sessionId)
  },

  // Real-time update methods for WebSocket events
  addTable: (table) => {
    set((state) => {
      // Check if table already exists, update if so
      const existingIndex = state.tables.findIndex((t) => t.name === table.name)
      if (existingIndex >= 0) {
        const updated = [...state.tables]
        updated[existingIndex] = table
        return { tables: updated }
      }
      return { tables: [...state.tables, table] }
    })
  },

  updateTable: (tableName, updates) => {
    set((state) => ({
      tables: state.tables.map((t) =>
        t.name === tableName ? { ...t, ...updates } : t
      ),
    }))
  },

  addArtifact: (artifact) => {
    set((state) => {
      // Check if artifact already exists
      const existingIndex = state.artifacts.findIndex((a) => a.id === artifact.id)
      if (existingIndex >= 0) {
        const updated = [...state.artifacts]
        updated[existingIndex] = artifact
        return { artifacts: updated }
      }
      return { artifacts: [...state.artifacts, artifact] }
    })
  },

  addFact: (fact) => {
    set((state) => {
      // Check if fact already exists
      const existingIndex = state.facts.findIndex((f) => f.name === fact.name)
      if (existingIndex >= 0) {
        const updated = [...state.facts]
        updated[existingIndex] = fact
        return { facts: updated }
      }
      return { facts: [...state.facts, fact] }
    })
  },

  addStepCode: (stepNumber, goal, code, model?) => {
    set((state) => {
      // Check if step code already exists
      const existingIndex = state.stepCodes.findIndex((s) => s.step_number === stepNumber)
      if (existingIndex >= 0) {
        const updated = [...state.stepCodes]
        updated[existingIndex] = { step_number: stepNumber, goal, code, model }
        return { stepCodes: updated }
      }
      const updated = [...state.stepCodes, { step_number: stepNumber, goal, code, model }]
      updated.sort((a, b) => a.step_number - b.step_number)
      return { stepCodes: updated }
    })
  },

  removeArtifact: async (sessionId, artifactId) => {
    const { artifacts } = get()
    const artifact = artifacts.find((a) => a.id === artifactId)
    const isTable = artifact?.artifact_type === 'table'

    // Optimistic removal
    set((state) => ({
      artifacts: state.artifacts.filter((a) => a.id !== artifactId),
      ...(isTable && artifact ? { tables: state.tables.filter((t) => t.name !== artifact.name) } : {}),
    }))

    try {
      await apolloClient.mutate({ mutation: DELETE_ARTIFACT, variables: { sessionId, id: artifactId } })
    } catch {
      // Revert on error — refetch both lists
      const { data } = await apolloClient.query({ query: ARTIFACTS_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
      set({ artifacts: data.artifacts.artifacts.map(toArtifact) })
      if (isTable) {
        const { data: tablesData } = await apolloClient.query({ query: TABLES_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
        set({ tables: tablesData.tables.tables.map(toTableInfo) })
      }
    }
  },

  removeTable: async (sessionId, tableName) => {
    // Optimistic removal
    set((state) => ({
      tables: state.tables.filter((t) => t.name !== tableName),
      artifacts: state.artifacts.filter((a) => !(a.artifact_type === 'table' && a.name === tableName)),
    }))

    try {
      await apolloClient.mutate({ mutation: DELETE_TABLE, variables: { sessionId, name: tableName } })
    } catch {
      // Revert on error — refetch both lists
      const [tablesResult, artifactsResult] = await Promise.all([
        apolloClient.query({ query: TABLES_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' }),
        apolloClient.query({ query: ARTIFACTS_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' }),
      ])
      set({ tables: tablesResult.data.tables.tables.map(toTableInfo), artifacts: artifactsResult.data.artifacts.artifacts.map(toArtifact) })
    }
  },

  toggleArtifactStar: async (sessionId, artifactId) => {
    // Optimistic update - toggle is_starred
    set((state) => ({
      artifacts: state.artifacts.map((a) =>
        a.id === artifactId ? { ...a, is_starred: !a.is_starred } : a
      ),
    }))

    try {
      await apolloClient.mutate({ mutation: TOGGLE_ARTIFACT_STAR, variables: { sessionId, id: artifactId } })
      // Refresh artifacts list to reflect changes (is_key_result may change too)
      const { data } = await apolloClient.query({ query: ARTIFACTS_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
      set({ artifacts: data.artifacts.artifacts.map(toArtifact) })
    } catch (error) {
      // Revert on error
      set((state) => ({
        artifacts: state.artifacts.map((a) =>
          a.id === artifactId ? { ...a, is_starred: !a.is_starred } : a
        ),
        error: String(error),
      }))
    }
  },

  toggleTableStar: async (sessionId, tableName) => {
    // Optimistic update for tables list
    set((state) => ({
      tables: state.tables.map((t) =>
        t.name === tableName ? { ...t, is_starred: !t.is_starred } : t
      ),
    }))

    try {
      await apolloClient.mutate({ mutation: TOGGLE_TABLE_STAR, variables: { sessionId, name: tableName } })
      // Refresh artifacts list to reflect starred table changes
      const { data } = await apolloClient.query({ query: ARTIFACTS_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
      set({ artifacts: data.artifacts.artifacts.map(toArtifact) })
    } catch (error) {
      // Revert on error
      set((state) => ({
        tables: state.tables.map((t) =>
          t.name === tableName ? { ...t, is_starred: !t.is_starred } : t
        ),
        error: String(error),
      }))
    }
  },

  addRule: async (summary, tags = []) => {
    try {
      await apolloClient.mutate({ mutation: CREATE_RULE, variables: { input: { summary, tags } } })
      get().fetchLearnings()
    } catch (error) {
      set({ error: String(error) })
    }
  },

  updateRule: async (ruleId, summary, tags) => {
    try {
      await apolloClient.mutate({ mutation: UPDATE_RULE_MUTATION, variables: { ruleId, input: { summary, tags } } })
      get().fetchLearnings()
    } catch (error) {
      set({ error: String(error) })
    }
  },

  deleteRule: async (ruleId) => {
    // Optimistic update
    set((state) => ({
      rules: state.rules.filter((r) => r.id !== ruleId),
    }))

    try {
      await apolloClient.mutate({ mutation: DELETE_RULE_MUTATION, variables: { ruleId } })
    } catch (error) {
      // Refresh to restore state on error
      get().fetchLearnings()
      set({ error: String(error) })
    }
  },

  deleteLearning: async (learningId) => {
    // Optimistic update
    set((state) => ({
      learnings: state.learnings.filter((l) => l.id !== learningId),
    }))

    try {
      await apolloClient.mutate({ mutation: DELETE_LEARNING_MUTATION, variables: { learningId } })
    } catch (error) {
      // Refresh to restore state on error
      get().fetchLearnings()
      set({ error: String(error) })
    }
  },

  setIngestingSource: (name) => set({ ingestingSource: name, ingestProgress: name ? get().ingestProgress : null }),
  setIngestProgress: (progress) => set({ ingestProgress: progress }),

  clear: () =>
    set({
      artifacts: [],
      tables: [],
      facts: [],
      entities: [],
      learnings: [],
      rules: [],
      databases: [],
      apis: [],
      documents: [],
      stepCodes: [],
  inferenceCodes: [],
      scratchpadEntries: [],
      sessionDDL: '',
      promptContext: null,
      allSkills: [],
      allAgents: [],
      userPermissions: { isAdmin: false, persona: 'viewer', visibility: {}, writes: {} },
      selectedArtifact: null,
      selectedTable: null,
      supersededStepNumbers: new Set<number>(),
      sourcesLoading: true,
      factsLoading: true,
      learningsLoading: true,
      configLoading: true,
      error: null,
      ingestingSource: null,
      ingestProgress: null,
    }),

  // Clear only query-produced results, keep session context (data sources, entities, learnings)
  clearQueryResults: () =>
    set({
      artifacts: [],
      tables: [],
      facts: [],
      stepCodes: [],
  inferenceCodes: [],
      scratchpadEntries: [],
      sessionDDL: '',
      selectedArtifact: null,
      selectedTable: null,
      error: null,
    }),
  markStepsSuperseded: () =>
    set((state) => {
      const superseded = new Set(state.supersededStepNumbers)
      for (const sc of state.stepCodes) {
        superseded.add(sc.step_number)
      }
      return { supersededStepNumbers: superseded }
    }),

  patchEntities: (diff) =>
    set((state) => {
      let entities = state.entities
      if (diff.removed.length > 0) {
        const removeSet = new Set(diff.removed.map(e => `${e.name}::${e.type}`))
        entities = entities.filter(e => !removeSet.has(`${e.name}::${e.type}`))
      }
      if (diff.added.length > 0) {
        const existing = new Set(entities.map(e => `${e.name}::${e.type}`))
        const newEntities = diff.added
          .filter(e => !existing.has(`${e.name}::${e.type}`))
          .map(e => ({ name: e.name, type: e.type } as Entity))
        entities = [...entities, ...newEntities]
      }
      return { entities }
    }),

  addInferenceCode: (ic) =>
    set((state) => {
      // Replace if same inference_id exists (retry), otherwise append
      const filtered = state.inferenceCodes.filter(x => x.inference_id !== ic.inference_id)
      return { inferenceCodes: [...filtered, ic] }
    }),
  clearInferenceCodes: () => set({ inferenceCodes: [] }),
  truncateFromStep: (fromStep, _tablesDropped) =>
    set((state) => ({
      tables: state.tables.filter((t) => (t.step_number ?? 0) < fromStep),
      artifacts: state.artifacts.filter((a) => (a.step_number ?? 0) < fromStep),
      stepCodes: state.stepCodes.filter((sc) => sc.step_number < fromStep),
      scratchpadEntries: state.scratchpadEntries.filter((e) => (e.step_number ?? 0) < fromStep),
    })),
}))