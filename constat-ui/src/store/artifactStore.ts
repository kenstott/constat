// Artifact state store

import { create } from 'zustand'
import type { Artifact, ArtifactContent, TableInfo, Fact, Entity, SessionDatabase, ApiSourceInfo, DocumentSourceInfo, Learning, Rule } from '@/types/api'
import * as sessionsApi from '@/api/sessions'

// Step code from execution (matches API response)
interface StepCode {
  step_number: number
  goal: string
  code: string
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
      const response = await sessionsApi.listLearnings()
      set({ learnings: response.learnings, rules: response.rules || [], loading: false })
    } catch (error) {
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
      selectedArtifact: null,
      selectedTable: null,
      error: null,
    }),
}))