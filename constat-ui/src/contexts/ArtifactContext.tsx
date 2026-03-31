// Copyright (c) 2025 Kenneth Stott
// Canary: 867ee1d8-ab85-4c24-9b3e-77cc96e24d6b
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { createContext, useContext, type ReactNode } from 'react'
import { useArtifactStore } from '@/store/artifactStore'
import type { Artifact, ArtifactContent, TableInfo, Fact, Entity, SessionDatabase, ApiSourceInfo, DocumentSourceInfo, Learning, Rule, ModelRouteInfo } from '@/types/api'

interface StepCode {
  step_number: number
  goal: string
  code: string
  prompt?: string
  model?: string
}

interface InferenceCode {
  inference_id: string
  name: string
  operation: string
  code: string
  attempt: number
  prompt?: string
  model?: string
}

interface ScratchpadEntry {
  step_number: number
  goal: string
  narrative: string
  tables_created: string[]
  code: string
  user_query: string
  objective_index: number | null
}

interface PromptContext {
  systemPrompt: string
  activeAgent: { name: string; prompt: string } | null
  activeSkills: Array<{ name: string; prompt: string; description: string }>
}

interface SkillInfo {
  name: string
  prompt: string
  description: string
  filename: string
  is_active: boolean
  domain: string
  source: string
}

interface AgentInfo {
  name: string
  prompt: string
  is_active: boolean
  domain: string
  source: string
}

interface ArtifactContextValue {
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
  selectedArtifact: ArtifactContent | null
  selectedTable: string | null
  supersededStepNumbers: Set<number>

  // Loading states
  loading: boolean
  sourcesLoading: boolean
  factsLoading: boolean
  learningsLoading: boolean
  configLoading: boolean
  ingestingSource: string | null
  ingestProgress: { current: number; total: number } | null

  // Fetch actions
  fetchFacts: (sessionId: string) => Promise<void>
  fetchEntities: (sessionId: string, entityType?: string) => Promise<void>
  fetchLearnings: () => Promise<void>
  fetchDataSources: (sessionId: string) => Promise<void>
  fetchPromptContext: (sessionId: string) => Promise<void>
  fetchTaskRouting: (sessionId: string) => Promise<void>
  fetchAllSkills: () => Promise<void>
  fetchAllAgents: (sessionId: string) => Promise<void>
  fetchScratchpad: (sessionId: string) => Promise<void>
  fetchDDL: (sessionId: string) => Promise<void>

  // Mutation actions
  selectArtifact: (sessionId: string, artifactId: number) => Promise<void>
  selectTable: (tableName: string | null) => void
  removeArtifact: (sessionId: string, artifactId: number) => Promise<void>
  removeTable: (sessionId: string, tableName: string) => Promise<void>
  toggleArtifactStar: (sessionId: string, artifactId: number) => Promise<void>
  toggleTableStar: (sessionId: string, tableName: string) => Promise<void>
  persistFact: (sessionId: string, factName: string) => Promise<void>
  forgetFact: (sessionId: string, factName: string) => Promise<void>
  createSkill: (name: string, prompt: string, description?: string) => Promise<void>
  updateSkill: (name: string, content: string) => Promise<void>
  deleteSkill: (name: string) => Promise<void>
  toggleSkillActive: (name: string, sessionId: string) => Promise<void>
  draftSkill: (sessionId: string, name: string, description: string) => Promise<{ content: string; description: string }>
  updateSystemPrompt: (sessionId: string, systemPrompt: string) => Promise<void>
  addRule: (summary: string, tags?: string[]) => Promise<void>
  updateRule: (ruleId: string, summary: string, tags?: string[]) => Promise<void>
  deleteRule: (ruleId: string) => Promise<void>
  deleteLearning: (learningId: string) => Promise<void>
}

const ArtifactContext = createContext<ArtifactContextValue | null>(null)

export function ArtifactProvider({ children }: { children: ReactNode }) {
  const store = useArtifactStore()

  const value: ArtifactContextValue = {
    // Data
    artifacts: store.artifacts,
    tables: store.tables,
    facts: store.facts,
    entities: store.entities,
    learnings: store.learnings,
    rules: store.rules,
    databases: store.databases,
    apis: store.apis,
    documents: store.documents,
    stepCodes: store.stepCodes,
    inferenceCodes: store.inferenceCodes,
    scratchpadEntries: store.scratchpadEntries,
    sessionDDL: store.sessionDDL,
    promptContext: store.promptContext as PromptContext | null,
    taskRouting: store.taskRouting,
    allSkills: store.allSkills as SkillInfo[],
    allAgents: store.allAgents as AgentInfo[],
    selectedArtifact: store.selectedArtifact,
    selectedTable: store.selectedTable,
    supersededStepNumbers: store.supersededStepNumbers,

    // Loading states
    loading: store.loading,
    sourcesLoading: store.sourcesLoading,
    factsLoading: store.factsLoading,
    learningsLoading: store.learningsLoading,
    configLoading: store.configLoading,
    ingestingSource: store.ingestingSource,
    ingestProgress: store.ingestProgress,

    // Fetch actions
    fetchFacts: store.fetchFacts,
    fetchEntities: store.fetchEntities,
    fetchLearnings: store.fetchLearnings,
    fetchDataSources: store.fetchDataSources,
    fetchPromptContext: store.fetchPromptContext,
    fetchTaskRouting: store.fetchTaskRouting,
    fetchAllSkills: store.fetchAllSkills,
    fetchAllAgents: store.fetchAllAgents,
    fetchScratchpad: store.fetchScratchpad,
    fetchDDL: store.fetchDDL,

    // Mutation actions
    selectArtifact: store.selectArtifact,
    selectTable: store.selectTable,
    removeArtifact: store.removeArtifact,
    removeTable: store.removeTable,
    toggleArtifactStar: store.toggleArtifactStar,
    toggleTableStar: store.toggleTableStar,
    persistFact: store.persistFact,
    forgetFact: store.forgetFact,
    createSkill: store.createSkill,
    updateSkill: store.updateSkill,
    deleteSkill: store.deleteSkill,
    toggleSkillActive: store.toggleSkillActive,
    draftSkill: store.draftSkill,
    updateSystemPrompt: store.updateSystemPrompt,
    addRule: store.addRule,
    updateRule: store.updateRule,
    deleteRule: store.deleteRule,
    deleteLearning: store.deleteLearning,
  }

  return <ArtifactContext.Provider value={value}>{children}</ArtifactContext.Provider>
}

export function useArtifactContext(): ArtifactContextValue {
  const ctx = useContext(ArtifactContext)
  if (!ctx) throw new Error('useArtifactContext must be used within ArtifactProvider')
  return ctx
}
