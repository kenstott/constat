// Artifact state store

import { create } from 'zustand'
import type { Artifact, ArtifactContent, TableInfo, Fact, Entity, SessionDatabase, ApiSourceInfo, DocumentSourceInfo, Learning, Rule } from '@/types/api'
import * as sessionsApi from '@/api/sessions'
import * as skillsApi from '@/api/skills'

// Step code from execution (matches API response)
interface StepCode {
  step_number: number
  goal: string
  code: string
}

// Prompt context types
interface PromptContext {
  systemPrompt: string
  activeRole: { name: string; prompt: string } | null
  activeSkills: Array<{ name: string; prompt: string; description: string }>
}

// Skill info (all skills, not just active)
interface SkillInfo {
  name: string
  prompt: string
  description: string
  filename: string
  is_active: boolean
}

// Role info (all roles, not just active)
interface RoleInfo {
  name: string
  prompt: string
  is_active: boolean
}

// User permissions
interface UserPermissions {
  isAdmin: boolean
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
  promptContext: PromptContext | null
  allSkills: SkillInfo[]
  allRoles: RoleInfo[]
  userPermissions: UserPermissions

  // Selected items
  selectedArtifact: ArtifactContent | null
  selectedTable: string | null

  // Loading states
  loading: boolean
  error: string | null

  // Actions
  fetchArtifacts: (sessionId: string) => Promise<void>
  fetchTables: (sessionId: string) => Promise<void>
  fetchFacts: (sessionId: string) => Promise<void>
  fetchEntities: (sessionId: string, entityType?: string) => Promise<void>
  fetchLearnings: () => Promise<void>
  fetchStepCodes: (sessionId: string) => Promise<void>
  fetchDatabases: (sessionId: string) => Promise<void>
  fetchDataSources: (sessionId: string) => Promise<void>
  fetchPromptContext: (sessionId: string) => Promise<void>
  fetchAllSkills: () => Promise<void>
  createSkill: (name: string, prompt: string, description?: string) => Promise<void>
  updateSkill: (name: string, content: string) => Promise<void>
  deleteSkill: (name: string) => Promise<void>
  toggleSkillActive: (name: string, sessionId: string) => Promise<void>
  fetchAllRoles: (sessionId: string) => Promise<void>
  setActiveRole: (roleName: string | null, sessionId: string) => Promise<void>
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
  addStepCode: (stepNumber: number, goal: string, code: string) => void
  // Star/promote actions (persist to server)
  toggleArtifactStar: (sessionId: string, artifactId: number) => Promise<void>
  toggleTableStar: (sessionId: string, tableName: string) => Promise<void>
  // Rule management
  addRule: (summary: string, tags?: string[]) => Promise<void>
  updateRule: (ruleId: string, summary: string, tags?: string[]) => Promise<void>
  deleteRule: (ruleId: string) => Promise<void>
  deleteLearning: (learningId: string) => Promise<void>
  clear: () => void
}

export const useArtifactStore = create<ArtifactState>((set, get) => ({
  artifacts: [],
  tables: [],
  facts: [],
  stepCodes: [],
  entities: [],
  learnings: [],
  rules: [],
  databases: [],
  apis: [],
  documents: [],
  promptContext: null,
  allSkills: [],
  allRoles: [],
  userPermissions: { isAdmin: false },
  selectedArtifact: null,
  selectedTable: null,
  loading: false,
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
    set({ loading: true, error: null })
    try {
      const response = await sessionsApi.listFacts(sessionId)
      set({ facts: response.facts, loading: false })
    } catch (error) {
      set({ error: String(error), loading: false })
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
    set({ loading: true, error: null })
    try {
      console.log('[LEARNINGS] Fetching learnings...')
      const response = await sessionsApi.listLearnings()
      console.log('[LEARNINGS] Received:', response.learnings.length, 'learnings,', response.rules?.length || 0, 'rules')
      set({ learnings: response.learnings, rules: response.rules || [], loading: false })
    } catch (error) {
      console.error('[LEARNINGS] Error fetching learnings:', error)
      set({ error: String(error), loading: false })
    }
  },

  fetchStepCodes: async (sessionId) => {
    try {
      const response = await sessionsApi.listStepCodes(sessionId)
      set({ stepCodes: response.steps })
    } catch (error) {
      // Step codes endpoint might not exist on older servers
      console.warn('Failed to fetch step codes:', error)
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
    set({ loading: true, error: null })
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
        loading: false,
      })
    } catch (error) {
      set({ error: String(error), loading: false })
    }
  },

  fetchPromptContext: async (sessionId) => {
    try {
      const response = await sessionsApi.getPromptContext(sessionId)
      set({
        promptContext: {
          systemPrompt: response.system_prompt,
          activeRole: response.active_role,
          activeSkills: response.active_skills,
        },
      })
    } catch (error) {
      console.warn('Failed to fetch prompt context:', error)
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
        })),
      })
    } catch (error) {
      console.warn('Failed to fetch skills:', error)
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
    try {
      await skillsApi.deleteSkill(name)
      get().fetchAllSkills()
    } catch (error) {
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

  fetchAllRoles: async (sessionId) => {
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

      const response = await fetch(`/api/sessions/roles?session_id=${sessionId}`, {
        headers,
        credentials: 'include',
      })
      if (response.ok) {
        const data = await response.json()
        set({
          allRoles: (data.roles || []).map((r: { name: string; prompt: string; is_active: boolean }) => ({
            name: r.name,
            prompt: r.prompt,
            is_active: r.is_active,
          })),
        })
      } else {
        console.warn('Failed to fetch roles:', response.status, response.statusText)
      }
    } catch (error) {
      console.warn('Failed to fetch roles:', error)
    }
  },

  setActiveRole: async (roleName, sessionId) => {
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

      const response = await fetch(`/api/sessions/roles/current?session_id=${sessionId}`, {
        method: 'PUT',
        headers,
        credentials: 'include',
        body: JSON.stringify({ role_name: roleName }),
      })
      if (response.ok) {
        get().fetchAllRoles(sessionId)
        get().fetchPromptContext(sessionId)
      }
    } catch (error) {
      set({ error: String(error) })
    }
  },

  fetchPermissions: async () => {
    try {
      const perms = await sessionsApi.getMyPermissions()
      set({ userPermissions: { isAdmin: perms.admin } })
    } catch (error) {
      // If auth is disabled or user not logged in, default to non-admin
      console.warn('Failed to fetch permissions:', error)
      set({ userPermissions: { isAdmin: false } })
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

  addStepCode: (stepNumber, goal, code) => {
    set((state) => {
      // Check if step code already exists
      const existingIndex = state.stepCodes.findIndex((s) => s.step_number === stepNumber)
      if (existingIndex >= 0) {
        const updated = [...state.stepCodes]
        updated[existingIndex] = { step_number: stepNumber, goal, code }
        return { stepCodes: updated }
      }
      return { stepCodes: [...state.stepCodes, { step_number: stepNumber, goal, code }] }
    })
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
      promptContext: null,
      allSkills: [],
      allRoles: [],
      userPermissions: { isAdmin: false },
      selectedArtifact: null,
      selectedTable: null,
      error: null,
    }),
}))