// Artifact state store

import { create } from 'zustand'
import type { Artifact, ArtifactContent, TableInfo, Fact, Entity, SessionDatabase, ApiSourceInfo, DocumentSourceInfo, Learning, Rule, ModelRouteInfo } from '@/types/api'
import * as sessionsApi from '@/api/sessions'
import * as skillsApi from '@/api/skills'

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
  addInferenceCode: (ic: InferenceCode) => void
  supersededStepNumbers: Set<number>
  markStepsSuperseded: () => void  // Mark all current step numbers as superseded (on redo)
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

  fetchArtifacts: async (sessionId) => {
    set({ loading: true, error: null })
    try {
      const response = await sessionsApi.listArtifacts(sessionId)
      set({ artifacts: response.artifacts, loading: false })
    } catch (error) {
      set({ error: String(error), loading: false })
    }
  },

  fetchTables: async (sessionId) => {
    set({ loading: true, error: null })
    try {
      const response = await sessionsApi.listTables(sessionId)
      set({ tables: response.tables, loading: false })
    } catch (error) {
      set({ error: String(error), loading: false })
    }
  },

  fetchFacts: async (sessionId) => {
    set({ factsLoading: true, error: null })
    try {
      const response = await sessionsApi.listFacts(sessionId)
      set({ facts: response.facts, factsLoading: false })
    } catch (error) {
      set({ error: String(error), factsLoading: false })
    }
  },

  fetchEntities: async (sessionId, entityType) => {
    set({ loading: true, error: null })
    try {
      const response = await sessionsApi.listEntities(sessionId, entityType)
      set({ entities: response.entities, loading: false })
    } catch (error) {
      set({ error: String(error), loading: false })
    }
  },

  fetchLearnings: async () => {
    set({ learningsLoading: true, error: null })
    try {
      console.log('[LEARNINGS] Fetching learnings...')
      const response = await sessionsApi.listLearnings()
      console.log('[LEARNINGS] Received:', response.learnings.length, 'learnings,', response.rules?.length || 0, 'rules')
      set({ learnings: response.learnings, rules: response.rules || [], learningsLoading: false })
    } catch (error) {
      console.error('[LEARNINGS] Error fetching learnings:', error)
      set({ error: String(error), learningsLoading: false })
    }
  },

  fetchStepCodes: async (sessionId) => {
    try {
      const response = await sessionsApi.listStepCodes(sessionId)
      const sorted = [...response.steps].sort((a, b) => a.step_number - b.step_number)
      set({ stepCodes: sorted })
    } catch (error) {
      console.warn('Failed to fetch step codes:', error)
    }
  },

  fetchInferenceCodes: async (sessionId) => {
    try {
      const response = await sessionsApi.listInferenceCodes(sessionId)
      set({ inferenceCodes: response.inferences.filter((ic: { inference_id: string }) => ic.inference_id) })
    } catch (error) {
      console.warn('Failed to fetch inference codes:', error)
    }
  },

  fetchScratchpad: async (sessionId) => {
    try {
      const response = await sessionsApi.getScratchpad(sessionId)
      set({ scratchpadEntries: response.entries })
    } catch (error) {
      console.warn('Failed to fetch scratchpad:', error)
    }
  },

  fetchDDL: async (sessionId) => {
    try {
      const response = await sessionsApi.getDDL(sessionId)
      set({ sessionDDL: response.ddl })
    } catch (error) {
      console.warn('Failed to fetch DDL:', error)
    }
  },

  fetchDatabases: async (sessionId) => {
    set({ loading: true, error: null })
    try {
      const response = await sessionsApi.listDatabases(sessionId)
      set({ databases: response.databases, loading: false })
    } catch (error) {
      set({ error: String(error), loading: false })
    }
  },

  fetchDataSources: async (sessionId) => {
    set({ sourcesLoading: true, error: null })
    try {
      // Use unified sources endpoint that includes config + active projects + session data
      const response = await sessionsApi.listDataSources(sessionId)

      // Map to frontend types
      const apis: ApiSourceInfo[] = response.apis.map((api) => ({
        name: api.name,
        type: api.type,
        description: api.description,
        base_url: api.base_url,
        connected: api.connected,
        from_config: api.from_config,
        source: api.source,
      }))

      const docs: DocumentSourceInfo[] = response.documents.map((doc) => ({
        name: doc.name,
        type: doc.type,
        description: doc.description,
        path: doc.path,
        indexed: doc.indexed,
        from_config: doc.from_config,
        source: doc.source,
      }))

      set({
        databases: response.databases,
        apis,
        documents: docs,
        sourcesLoading: false,
      })
    } catch (error) {
      set({ error: String(error), sourcesLoading: false })
    }
  },

  fetchPromptContext: async (sessionId) => {
    try {
      const response = await sessionsApi.getPromptContext(sessionId)
      set({
        promptContext: {
          systemPrompt: response.system_prompt,
          activeAgent: response.active_agent,
          activeSkills: response.active_skills,
        },
      })
    } catch (error) {
      console.warn('Failed to fetch prompt context:', error)
    }
  },

  fetchTaskRouting: async (sessionId: string) => {
    try {
      const layers = await sessionsApi.getSessionRouting(sessionId)
      set({ taskRouting: layers })
    } catch (error) {
      console.warn('Failed to fetch task routing:', error)
    }
  },

  fetchAllSkills: async () => {
    try {
      const response = await skillsApi.listSkills()
      set({
        allSkills: response.skills.map((s) => ({
          name: s.name,
          prompt: s.prompt,
          description: s.description,
          filename: s.filename,
          is_active: s.is_active,
          domain: s.domain || '',
          source: s.source || '',
        })),
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
      await skillsApi.setActiveSkills(newActive)
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
      // Import auth helpers
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = {}
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) {
          headers['Authorization'] = `Bearer ${token}`
        }
      }

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
      // Import auth helpers
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) {
          headers['Authorization'] = `Bearer ${token}`
        }
      }

      const response = await fetch(`/api/sessions/agents/current?session_id=${sessionId}`, {
        method: 'PUT',
        headers,
        credentials: 'include',
        body: JSON.stringify({ agent_name: agentName }),
      })
      if (response.ok) {
        get().fetchAllAgents(sessionId)
        get().fetchPromptContext(sessionId)
      }
    } catch (error) {
      set({ error: String(error) })
    }
  },

  fetchPermissions: async () => {
    // When auth disabled, grant full access (matches authStore behavior)
    const { isAuthDisabled, useAuthStore } = await import('@/store/authStore')
    if (isAuthDisabled) {
      set({ userPermissions: {
        isAdmin: true, persona: 'platform_admin',
        visibility: { results: true, databases: true, apis: true, documents: true, system_prompt: true, agents: true, skills: true, learnings: true, code: true, inference_code: true, facts: true, glossary: true },
        writes: { sources: true, glossary: true, skills: true, agents: true, facts: true, learnings: true, system_prompt: true, tier_promote: true },
      } })
      return
    }
    // Skip if auth token not yet available (avoids 401 → logout during HMR)
    const token = await useAuthStore.getState().getToken()
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
      await sessionsApi.updateSystemPrompt(sessionId, systemPrompt)
      // Refresh prompt context to show updated value
      get().fetchPromptContext(sessionId)
    } catch (error) {
      set({ error: String(error) })
      throw error
    }
  },

  selectArtifact: async (sessionId, artifactId) => {
    set({ loading: true, error: null })
    try {
      const artifact = await sessionsApi.getArtifact(sessionId, artifactId)
      set({ selectedArtifact: artifact, loading: false })
    } catch (error) {
      set({ error: String(error), loading: false })
    }
  },

  selectTable: (tableName) => set({ selectedTable: tableName }),

  persistFact: async (sessionId, factName) => {
    await sessionsApi.persistFact(sessionId, factName)
    get().fetchFacts(sessionId)
  },

  forgetFact: async (sessionId, factName) => {
    await sessionsApi.forgetFact(sessionId, factName)
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
      await sessionsApi.deleteArtifact(sessionId, artifactId)
    } catch {
      // Revert on error — refetch both lists
      const response = await sessionsApi.listArtifacts(sessionId)
      set({ artifacts: response.artifacts })
      if (isTable) {
        const tablesResponse = await sessionsApi.listTables(sessionId)
        set({ tables: tablesResponse.tables })
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
      await sessionsApi.deleteTable(sessionId, tableName)
    } catch {
      // Revert on error — refetch both lists
      const [tablesResponse, artifactsResponse] = await Promise.all([
        sessionsApi.listTables(sessionId),
        sessionsApi.listArtifacts(sessionId),
      ])
      set({ tables: tablesResponse.tables, artifacts: artifactsResponse.artifacts })
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
      // Persist to server
      await sessionsApi.toggleArtifactStar(sessionId, artifactId)
      // Refresh artifacts list to reflect changes (is_key_result may change too)
      const response = await sessionsApi.listArtifacts(sessionId)
      set({ artifacts: response.artifacts })
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
      // Persist to server
      await sessionsApi.toggleTableStar(sessionId, tableName)
      // Refresh artifacts list to reflect starred table changes
      const response = await sessionsApi.listArtifacts(sessionId)
      set({ artifacts: response.artifacts })
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
      await sessionsApi.addRule({ summary, tags })
      // Refresh learnings list
      get().fetchLearnings()
    } catch (error) {
      set({ error: String(error) })
    }
  },

  updateRule: async (ruleId, summary, tags) => {
    try {
      await sessionsApi.updateRule(ruleId, { summary, tags })
      // Refresh learnings list
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
      await sessionsApi.deleteRule(ruleId)
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
      await sessionsApi.deleteLearning(learningId)
    } catch (error) {
      // Refresh to restore state on error
      get().fetchLearnings()
      set({ error: String(error) })
    }
  },

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