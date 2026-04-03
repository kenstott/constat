// Copyright (c) 2025 Kenneth Stott
// Canary: 867ee1d8-ab85-4c24-9b3e-77cc96e24d6b
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { createContext, useContext, useCallback, type ReactNode } from 'react'
import { useQuery, useReactiveVar } from '@apollo/client'
import { apolloClient } from '@/graphql/client'
import { useSessionContext } from '@/contexts/SessionContext'
import type { ArtifactContent, ModelRouteInfo } from '@/types/api'
import {
  STEPS_QUERY, INFERENCE_CODES_QUERY, SCRATCHPAD_QUERY, SESSION_DDL_QUERY,
  SESSION_ROUTING_QUERY, PROMPT_CONTEXT_QUERY, UPDATE_SYSTEM_PROMPT,
  toStepCode, toInferenceCode, toScratchpadEntry, toPromptContext,
} from '@/graphql/operations/state'
import {
  ARTIFACT_QUERY, DELETE_TABLE, DELETE_ARTIFACT,
  TOGGLE_TABLE_STAR, TOGGLE_ARTIFACT_STAR,
  PERSIST_FACT, FORGET_FACT,
  toArtifactContent,
} from '@/graphql/operations/data'
import {
  SKILLS_QUERY, SET_ACTIVE_SKILLS,
  CREATE_SKILL, UPDATE_SKILL, DELETE_SKILL, DRAFT_SKILL,
  CREATE_RULE, UPDATE_RULE as UPDATE_RULE_MUTATION, DELETE_RULE as DELETE_RULE_MUTATION,
  DELETE_LEARNING as DELETE_LEARNING_MUTATION,
  toSkillInfo,
} from '@/graphql/operations/learnings'
import {
  stepCodesVar, inferenceCodesVar, scratchpadEntriesVar,
  selectedArtifactVar, selectedTableVar, supersededStepNumbersVar,
  ingestingSourceVar, ingestProgressVar,
} from '@/graphql/ui-state'

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

interface AgentInfo {
  name: string
  prompt: string
  is_active: boolean
  domain: string
  source: string
}

interface ArtifactContextValue {
  // Event-driven state (reactive vars)
  stepCodes: StepCode[]
  inferenceCodes: InferenceCode[]
  scratchpadEntries: ScratchpadEntry[]
  sessionDDL: string
  promptContext: PromptContext | null
  taskRouting: Record<string, Record<string, ModelRouteInfo[]>> | null
  allAgents: AgentInfo[]
  selectedArtifact: ArtifactContent | null
  selectedTable: string | null
  supersededStepNumbers: Set<number>

  // Loading states
  ingestingSource: string | null
  ingestProgress: { current: number; total: number } | null

  // Fetch actions
  fetchPromptContext: (sessionId: string) => Promise<void>
  fetchTaskRouting: (sessionId: string) => Promise<void>
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
  const { sessionId } = useSessionContext()

  // Apollo queries — data lives in Apollo cache
  const { data: stepsData } = useQuery(STEPS_QUERY, { variables: { sessionId: sessionId! }, skip: !sessionId })
  const { data: icData } = useQuery(INFERENCE_CODES_QUERY, { variables: { sessionId: sessionId! }, skip: !sessionId })
  const { data: scratchData } = useQuery(SCRATCHPAD_QUERY, { variables: { sessionId: sessionId! }, skip: !sessionId })
  const { data: ddlData } = useQuery(SESSION_DDL_QUERY, { variables: { sessionId: sessionId! }, skip: !sessionId })
  const { data: pcData } = useQuery(PROMPT_CONTEXT_QUERY, { variables: { sessionId: sessionId! }, skip: !sessionId })
  const { data: routingData } = useQuery(SESSION_ROUTING_QUERY, { variables: { sessionId: sessionId! }, skip: !sessionId })
  const { data: skillsData } = useQuery(SKILLS_QUERY, { skip: !sessionId })

  // Reactive vars — event-driven state
  const rvStepCodes = useReactiveVar(stepCodesVar)
  const rvInferenceCodes = useReactiveVar(inferenceCodesVar)
  const rvScratchpad = useReactiveVar(scratchpadEntriesVar)
  const selectedArtifact = useReactiveVar(selectedArtifactVar)
  const selectedTable = useReactiveVar(selectedTableVar)
  const supersededStepNumbers = useReactiveVar(supersededStepNumbersVar)
  const ingestingSource = useReactiveVar(ingestingSourceVar)
  const ingestProgress = useReactiveVar(ingestProgressVar)

  // Merge Apollo query data with reactive var data (reactive vars have real-time updates)
  const queryStepCodes = stepsData?.steps?.steps?.map(toStepCode)?.sort((a: StepCode, b: StepCode) => a.step_number - b.step_number) ?? []
  const stepCodes = rvStepCodes.length > 0 ? rvStepCodes : queryStepCodes
  const queryInferenceCodes = icData?.inferenceCodes?.inferences?.map(toInferenceCode)?.filter((ic: InferenceCode) => ic.inference_id) ?? []
  const inferenceCodes = rvInferenceCodes.length > 0 ? rvInferenceCodes : queryInferenceCodes
  const queryScratchpad = scratchData?.scratchpad?.entries?.map(toScratchpadEntry) ?? []
  const scratchpadEntries = rvScratchpad.length > 0 ? rvScratchpad : queryScratchpad

  const sessionDDL = ddlData?.sessionDdl ?? ''
  const promptContext = pcData?.promptContext
    ? (() => { const ctx = toPromptContext(pcData.promptContext); return { systemPrompt: ctx.system_prompt, activeAgent: ctx.active_agent, activeSkills: ctx.active_skills } })()
    : null
  const taskRouting = routingData?.sessionRouting ?? null

  // Agents — fetched via REST (no GraphQL query)
  const fetchAllAgents = useCallback(async (_sid: string) => {
    // Agents are still fetched via REST — TODO: migrate to GraphQL
    // For now, this is a no-op since agents are fetched on session_ready via event handler
  }, [])

  // Refetch helpers
  const fetchPromptContext = useCallback(async (_sid: string) => {
    await apolloClient.refetchQueries({ include: ['PromptContext'] })
  }, [])

  const fetchTaskRouting = useCallback(async (_sid: string) => {
    await apolloClient.refetchQueries({ include: ['SessionRouting'] })
  }, [])

  const fetchScratchpad = useCallback(async (_sid: string) => {
    await apolloClient.refetchQueries({ include: ['Scratchpad'] })
  }, [])

  const fetchDDL = useCallback(async (_sid: string) => {
    await apolloClient.refetchQueries({ include: ['SessionDdl'] })
  }, [])

  // Mutation actions
  const selectArtifact = useCallback(async (sid: string, artifactId: number) => {
    const { data } = await apolloClient.query({ query: ARTIFACT_QUERY, variables: { sessionId: sid, id: artifactId }, fetchPolicy: 'network-only' })
    selectedArtifactVar(toArtifactContent(data.artifact))
  }, [])

  const selectTable = useCallback((tableName: string | null) => {
    selectedTableVar(tableName)
  }, [])

  const removeArtifact = useCallback(async (sid: string, artifactId: number) => {
    await apolloClient.mutate({ mutation: DELETE_ARTIFACT, variables: { sessionId: sid, id: artifactId } })
    await apolloClient.refetchQueries({ include: ['Artifacts', 'Tables'] })
  }, [])

  const removeTable = useCallback(async (sid: string, tableName: string) => {
    await apolloClient.mutate({ mutation: DELETE_TABLE, variables: { sessionId: sid, name: tableName } })
    await apolloClient.refetchQueries({ include: ['Tables', 'Artifacts'] })
  }, [])

  const toggleArtifactStar = useCallback(async (sid: string, artifactId: number) => {
    await apolloClient.mutate({ mutation: TOGGLE_ARTIFACT_STAR, variables: { sessionId: sid, id: artifactId } })
    await apolloClient.refetchQueries({ include: ['Artifacts'] })
  }, [])

  const toggleTableStar = useCallback(async (sid: string, tableName: string) => {
    await apolloClient.mutate({ mutation: TOGGLE_TABLE_STAR, variables: { sessionId: sid, name: tableName } })
    await apolloClient.refetchQueries({ include: ['Artifacts'] })
  }, [])

  const persistFact = useCallback(async (sid: string, factName: string) => {
    await apolloClient.mutate({ mutation: PERSIST_FACT, variables: { sessionId: sid, factName } })
    await apolloClient.refetchQueries({ include: ['Facts'] })
  }, [])

  const forgetFact = useCallback(async (sid: string, factName: string) => {
    await apolloClient.mutate({ mutation: FORGET_FACT, variables: { sessionId: sid, factName } })
    await apolloClient.refetchQueries({ include: ['Facts'] })
  }, [])

  const createSkill = useCallback(async (name: string, prompt: string, description = '') => {
    await apolloClient.mutate({ mutation: CREATE_SKILL, variables: { input: { name, prompt, description } } })
    await apolloClient.refetchQueries({ include: ['Skills'] })
  }, [])

  const updateSkill = useCallback(async (name: string, content: string) => {
    await apolloClient.mutate({ mutation: UPDATE_SKILL, variables: { name, input: { content } } })
    await apolloClient.refetchQueries({ include: ['Skills'] })
  }, [])

  const deleteSkill = useCallback(async (name: string) => {
    const skill = skillsData?.skills?.skills?.map(toSkillInfo)?.find((s: { name: string }) => s.name === name)
    await apolloClient.mutate({ mutation: DELETE_SKILL, variables: { name, domain: skill?.domain } })
    await apolloClient.refetchQueries({ include: ['Skills'] })
  }, [skillsData])

  const toggleSkillActive = useCallback(async (name: string, _sid: string) => {
    const allSkills = skillsData?.skills?.skills?.map(toSkillInfo) ?? []
    const skill = allSkills.find((s: { name: string }) => s.name === name)
    if (!skill) return
    const currentActive = allSkills.filter((s: { is_active: boolean }) => s.is_active).map((s: { name: string }) => s.name)
    const newActive = skill.is_active ? currentActive.filter((n: string) => n !== name) : [...currentActive, name]
    await apolloClient.mutate({ mutation: SET_ACTIVE_SKILLS, variables: { skillNames: newActive } })
    await apolloClient.refetchQueries({ include: ['Skills', 'PromptContext'] })
  }, [skillsData])

  const draftSkill = useCallback(async (sid: string, name: string, description: string) => {
    const { data } = await apolloClient.mutate({ mutation: DRAFT_SKILL, variables: { sessionId: sid, input: { name, userDescription: description } } })
    return data?.draftSkill ?? { content: '', description: '' }
  }, [])

  const updateSystemPrompt = useCallback(async (sid: string, systemPrompt: string) => {
    await apolloClient.mutate({ mutation: UPDATE_SYSTEM_PROMPT, variables: { sessionId: sid, systemPrompt } })
    await apolloClient.refetchQueries({ include: ['PromptContext'] })
  }, [])

  const addRule = useCallback(async (summary: string, tags: string[] = []) => {
    await apolloClient.mutate({ mutation: CREATE_RULE, variables: { input: { summary, tags } } })
    await apolloClient.refetchQueries({ include: ['Learnings'] })
  }, [])

  const updateRule = useCallback(async (ruleId: string, summary: string, tags?: string[]) => {
    await apolloClient.mutate({ mutation: UPDATE_RULE_MUTATION, variables: { ruleId, input: { summary, tags } } })
    await apolloClient.refetchQueries({ include: ['Learnings'] })
  }, [])

  const deleteRule = useCallback(async (ruleId: string) => {
    await apolloClient.mutate({ mutation: DELETE_RULE_MUTATION, variables: { ruleId } })
    await apolloClient.refetchQueries({ include: ['Learnings'] })
  }, [])

  const deleteLearning = useCallback(async (learningId: string) => {
    await apolloClient.mutate({ mutation: DELETE_LEARNING_MUTATION, variables: { learningId } })
    await apolloClient.refetchQueries({ include: ['Learnings'] })
  }, [])

  const value: ArtifactContextValue = {
    stepCodes, inferenceCodes, scratchpadEntries, sessionDDL,
    promptContext, taskRouting, allAgents: [],
    selectedArtifact, selectedTable, supersededStepNumbers,
    ingestingSource, ingestProgress,
    fetchPromptContext, fetchTaskRouting, fetchAllAgents, fetchScratchpad, fetchDDL,
    selectArtifact, selectTable,
    removeArtifact, removeTable, toggleArtifactStar, toggleTableStar,
    persistFact, forgetFact,
    createSkill, updateSkill, deleteSkill, toggleSkillActive, draftSkill,
    updateSystemPrompt, addRule, updateRule, deleteRule, deleteLearning,
  }

  return <ArtifactContext.Provider value={value}>{children}</ArtifactContext.Provider>
}

export function useArtifactContext(): ArtifactContextValue {
  const ctx = useContext(ArtifactContext)
  if (!ctx) throw new Error('useArtifactContext must be used within ArtifactProvider')
  return ctx
}
