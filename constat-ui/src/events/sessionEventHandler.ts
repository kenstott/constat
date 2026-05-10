// Copyright (c) 2025 Kenneth Stott
// Canary: 91dfdc2f-2595-4d3f-8692-b297551f71f0
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// Session event handler — reducer + side effects extracted from sessionStore
import { applyPatch, type Operation } from 'fast-json-patch'
import type { SubscriptionEvent, Plan, GlossaryTerm } from '@/types/api'
import {
  fetchTerms as glossaryFetchTerms,
  setTermsFromState,
  addTerms as glossaryAddTerms,
  setEntityRebuilding,
  setGenerating,
  setProgress,
  bumpRefreshKey,
} from '@/store/glossaryState'
import { apolloClient } from '@/graphql/client'
import { SAVE_PROOF_FACTS } from '@/graphql/operations/state'
import { ARTIFACTS_QUERY, TABLES_QUERY, FACTS_QUERY, ARTIFACT_QUERY, toArtifact, toArtifactContent, toTableInfo } from '@/graphql/operations/data'
import {
  addStepCode,
  addInferenceCode,
  clearInferenceCodes,
  truncateFromStep,
  ingestingSourceVar,
  ingestProgressVar,
  selectedArtifactVar,
  expandSections,
  handleFactEvent,
  exportFacts,
  isSummaryGeneratingVar,
} from '@/graphql/ui-state'
import { getCachedEntry, setCachedEntry } from '@/store/entityCache'
import { type CompactState, inflateToGlossaryTerms } from '@/store/entityCacheKeys'
import type {
  SessionExecutionState,
  SessionAction,
  Message,
  ExecutionPhase,
  SideEffectStores,
} from '@/events/types'

// ---------------------------------------------------------------------------
// parseSourceTables
// ---------------------------------------------------------------------------

export function parseSourceTables(code: string): string[] {
  const sources = new Set<string>()
  const patterns = [
    /\bFROM\s+([a-zA-Z_]\w*\.[a-zA-Z_]\w*)/gi,
    /\bJOIN\s+([a-zA-Z_]\w*\.[a-zA-Z_]\w*)/gi,
  ]
  for (const re of patterns) {
    let m: RegExpExecArray | null
    while ((m = re.exec(code)) !== null) sources.add(m[1])
  }
  return [...sources]
}

// ---------------------------------------------------------------------------
// Pure helper functions for state transformations
// ---------------------------------------------------------------------------

function updateStepMessage(
  state: SessionExecutionState,
  stepNumber: number,
  content: string,
  isComplete = false,
): SessionExecutionState {
  const messageId = state.stepMessageIds[stepNumber]
  if (!messageId) return state
  return {
    ...state,
    messages: state.messages.map((m) =>
      m.id === messageId
        ? { ...m, content, isLive: !isComplete, isPending: false, role: m.role, skills: m.skills }
        : m
    ),
  }
}

function ensureLiveMessage(
  state: SessionExecutionState,
  content: string,
  phase: ExecutionPhase,
): SessionExecutionState {
  const existingId = state.liveMessageId || state.thinkingMessageId
  if (existingId) {
    const updated: SessionExecutionState = {
      ...state,
      messages: state.messages.map((m) =>
        m.id === existingId ? { ...m, type: 'system' as const, content } : m
      ),
    }
    if (state.thinkingMessageId && !state.liveMessageId) {
      return { ...updated, thinkingMessageId: null, liveMessageId: existingId, executionPhase: phase }
    }
    return { ...updated, executionPhase: phase }
  }
  const newId = crypto.randomUUID()
  const liveMessage: Message = {
    id: newId,
    type: 'system',
    content,
    timestamp: new Date(),
    isLive: true,
  }
  return {
    ...state,
    messages: [...state.messages, liveMessage],
    liveMessageId: newId,
    executionPhase: phase,
  }
}

function clearLiveMessage(state: SessionExecutionState): SessionExecutionState {
  let next = state
  if (next.liveMessageId) {
    next = {
      ...next,
      messages: next.messages.filter((m) => m.id !== next.liveMessageId),
      liveMessageId: null,
      executionPhase: 'idle',
    }
  }
  if (next.thinkingMessageId) {
    next = {
      ...next,
      messages: next.messages.filter((m) => m.id !== next.thinkingMessageId),
      thinkingMessageId: null,
    }
  }
  return next
}

function finalizeAllSteps(state: SessionExecutionState): SessionExecutionState {
  return {
    ...state,
    messages: state.messages.map((m) =>
      m.stepNumber !== undefined && state.stepMessageIds[m.stepNumber]
        ? { ...m, isLive: false }
        : m
    ),
    stepMessageIds: {},
  }
}

// ---------------------------------------------------------------------------
// sessionEventReducer — pure state transitions, no side effects
// ---------------------------------------------------------------------------

export function sessionEventReducer(
  state: SessionExecutionState,
  action: SessionAction,
): SessionExecutionState {
  switch (action.type) {
    case 'SUBMIT_QUERY': {
      const thinkingMessage: Message = {
        id: action.thinkingId,
        type: 'thinking',
        content: '',
        timestamp: new Date(),
        isLive: true,
      }
      return {
        ...state,
        messages: [...state.messages, thinkingMessage],
        thinkingMessageId: action.thinkingId,
        liveMessageId: null,
        stepMessageIds: {},
        currentQuery: action.query,
        querySubmittedAt: Date.now(),
        status: 'planning',
        executionPhase: 'idle',
        currentStepNumber: 0,
        stepAttempt: 1,
        isRedo: action.isRedo,
        suggestions: [],
      }
    }

    case 'CANCEL_EXECUTION':
      return {
        ...state,
        status: 'cancelled',
        executionPhase: 'idle',
        currentStepNumber: 0,
        stepAttempt: 1,
      }

    case 'APPROVE_PLAN': {
      return {
        ...state,
        messages: [
          ...state.messages.map((m) =>
            action.isRedo && m.type === 'step' && !m.isSuperseded
              ? { ...m, isSuperseded: true }
              : m
          ),
          ...action.stepMessages,
        ],
        stepMessageIds: { ...state.stepMessageIds, ...action.stepMessageIds },
        liveMessageId: null,
        status: 'executing',
        executionPhase: 'executing',
        plan: null,
      }
    }

    case 'REJECT_PLAN': {
      if (action.hasFeedback) {
        return { ...state, plan: null, status: 'planning', executionPhase: 'planning' }
      }
      return { ...state, plan: null, status: 'idle' }
    }

    case 'ANSWER_CLARIFICATION': {
      const next: SessionExecutionState = {
        ...state,
        clarification: null,
        status: action.isInputRequest ? 'executing' : 'planning',
      }
      if (action.userMessage) {
        if (action.stepMsgId) {
          const stepIdx = next.messages.findIndex((m) => m.id === action.stepMsgId)
          if (stepIdx >= 0) {
            let insertIdx = next.messages.length
            for (let i = stepIdx + 1; i < next.messages.length; i++) {
              if (next.messages[i].type === 'system') {
                insertIdx = i + 1
                break
              }
            }
            const updated = [...next.messages]
            updated.splice(insertIdx, 0, action.userMessage)
            return { ...next, messages: updated }
          }
        }
        return { ...next, messages: [...next.messages, action.userMessage] }
      }
      return next
    }

    case 'SKIP_CLARIFICATION':
      return {
        ...state,
        clarification: null,
        status: action.isInputRequest ? 'executing' : 'planning',
      }

    case 'SET_CLARIFICATION_STEP':
      return {
        ...state,
        clarification: state.clarification
          ? { ...state.clarification, currentStep: action.step }
          : null,
      }

    case 'SET_CLARIFICATION_ANSWER':
      return {
        ...state,
        clarification: state.clarification
          ? { ...state.clarification, answers: { ...state.clarification.answers, [action.step]: action.answer } }
          : null,
      }

    case 'SET_CLARIFICATION_STRUCTURED_ANSWER':
      return {
        ...state,
        clarification: state.clarification
          ? { ...state.clarification, structuredAnswers: { ...state.clarification.structuredAnswers, [action.step]: action.data } }
          : null,
      }

    case 'ADD_MESSAGE':
      return { ...state, messages: [...state.messages, action.message] }

    case 'UPDATE_MESSAGE':
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.id ? { ...m, ...action.updates } : m
        ),
      }

    case 'REMOVE_MESSAGE':
      return {
        ...state,
        messages: state.messages.filter((m) => m.id !== action.id),
        thinkingMessageId: state.thinkingMessageId === action.id ? null : state.thinkingMessageId,
      }

    case 'CLEAR_MESSAGES':
      return { ...state, messages: [], thinkingMessageId: null }

    case 'SET_MESSAGES':
      return {
        ...state,
        messages: action.messages,
        ...(action.suggestions !== undefined ? { suggestions: action.suggestions } : {}),
        ...(action.plan !== undefined ? { plan: action.plan } : {}),
      }

    case 'SET_CURRENT_QUERY':
      return { ...state, currentQuery: action.query }

    case 'SET_STATUS':
      return { ...state, status: action.status }

    case 'ADD_QUEUED_MESSAGE':
      return { ...state, queuedMessages: [...state.queuedMessages, action.message] }

    case 'REMOVE_QUEUED_MESSAGE':
      return {
        ...state,
        queuedMessages: state.queuedMessages.filter((m) => m.id !== action.id),
      }

    case 'CLEAR_QUEUE':
      return { ...state, queuedMessages: [] }

    case 'RESET':
      return {
        ...state,
        messages: action.messages ?? [],
        thinkingMessageId: null,
        liveMessageId: null,
        stepMessageIds: {},
        currentStepNumber: 0,
        stepAttempt: 1,
        plan: null,
        clarification: null,
        executionPhase: 'idle',
        status: 'idle',
        suggestions: [],
        queryContext: null,
        queuedMessages: [],
        lastQueryStartStep: 0,
        querySubmittedAt: null,
      }

    case 'SUBSCRIPTION_EVENT':
      return reduceWSEvent(state, action.event)

    default:
      return state
  }
}

// ---------------------------------------------------------------------------
// SUBSCRIPTION_EVENT reducer (session state changes only, no side effects)
// ---------------------------------------------------------------------------

function reduceWSEvent(state: SessionExecutionState, event: SubscriptionEvent): SessionExecutionState {
  switch (event.event_type) {
    case 'heartbeat_ack':
      // Side effect only — handled in executeSideEffects
      return state

    case 'session_ready': {
      const readyData = event.data as { active_domains?: string[] }
      if (readyData.active_domains) {
        // session update handled in sessionStore (needs session object) — state-only fields:
        return { ...state }
      }
      return state
    }

    case 'welcome': {
      const data = event.data as { suggestions: string[]; tagline?: string; reliable_adjective?: string; honest_adjective?: string }
      const tagline = data.reliable_adjective && data.honest_adjective
        ? `I'm **Vera**, your ${data.reliable_adjective} and ${data.honest_adjective} data analyst. _${data.tagline || ''}_`
        : ''
      return { ...state, suggestions: data.suggestions || [], welcomeTagline: tagline }
    }

    case 'planning_start':
      return ensureLiveMessage({ ...state, status: 'planning' }, 'Planning...', 'planning')

    case 'replan_start': {
      const fromStep = (event.data as Record<string, unknown>)?.from_step as number
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.stepNumber !== undefined && m.stepNumber >= fromStep
            ? { ...m, isSuperseded: true, isLive: false, isPending: false }
            : m
        ),
        status: 'executing',
      }
    }

    case 'proof_start':
      return ensureLiveMessage({ ...state, status: 'planning' }, 'Generating reasoning chain...', 'planning')

    case 'replanning':
      return ensureLiveMessage({ ...state, status: 'planning', executionPhase: 'planning' }, 'Revising plan...', 'planning')

    case 'dynamic_context': {
      const agent = event.data.agent as { name: string; similarity: number } | undefined
      const skills = event.data.skills as { name: string; similarity: number }[] | undefined
      let next = { ...state, queryContext: { agent, skills } }

      const contextParts: string[] = []
      if (agent?.name) contextParts.push(`@${agent.name}`)
      if (skills && skills.length > 0) contextParts.push(...skills.map(s => s.name))

      if (contextParts.length > 0) {
        const msgId = next.liveMessageId || next.thinkingMessageId
        if (msgId) {
          next = {
            ...next,
            messages: next.messages.map((m) =>
              m.id === msgId ? { ...m, content: `Planning... (${contextParts.join(', ')})` } : m
            ),
          }
        }
      }
      return next
    }

    case 'plan_ready': {
      const withCleared = clearLiveMessage(state)
      const planData = event.data.plan as Plan | undefined
      if (planData) {
        return { ...withCleared, status: 'awaiting_approval', plan: planData, executionPhase: 'awaiting_approval' }
      }
      const autoSteps = (event.data.steps as Array<{ number: number; goal: string; depends_on?: number[]; role_id?: string; domain?: string }>) || []
      const existingStepNumbers = new Set(
        withCleared.messages.filter((m) => m.type === 'step' && m.stepNumber !== undefined && !m.isSuperseded).map((m) => m.stepNumber)
      )
      const autoStepIds: Record<number, string> = {}
      const autoStepMsgs: Message[] = autoSteps
        .filter((step) => !existingStepNumbers.has(step.number))
        .map((step) => {
          const id = crypto.randomUUID()
          autoStepIds[step.number] = id
          return {
            id,
            type: 'step' as const,
            content: `Step ${step.number}: ${step.goal || 'Pending'}`,
            timestamp: new Date(),
            stepNumber: step.number,
            isLive: false,
            isPending: true,
            ...(step.role_id ? { role: step.domain ? `${step.domain}/${step.role_id}` : step.role_id } : {}),
          }
        })
      autoStepMsgs.sort((a, b) => (a.stepNumber || 0) - (b.stepNumber || 0))
      return {
        ...withCleared,
        messages: [...withCleared.messages, ...autoStepMsgs],
        stepMessageIds: { ...withCleared.stepMessageIds, ...autoStepIds },
        liveMessageId: null,
        status: 'executing',
        executionPhase: 'executing',
        plan: null,
      }
    }

    case 'plan_updated': {
      const freshIds = { ...state.stepMessageIds }
      const updatedSteps = (event.data.steps as Array<{ number: number; goal: string; depends_on?: number[]; role_id?: string; domain?: string }>) || []
      const newStepNumbers = new Set(updatedSteps.map((s) => s.number))
      const staleIds = state.messages
        .filter((m) => m.type === 'step' && m.stepNumber && m.isPending && !newStepNumbers.has(m.stepNumber))
        .map((m) => m.id)

      let next = state
      if (staleIds.length > 0) {
        next = { ...next, messages: next.messages.filter((m) => !staleIds.includes(m.id)) }
        for (const num of Object.keys(freshIds)) {
          if (staleIds.includes(freshIds[Number(num)])) {
            delete freshIds[Number(num)]
          }
        }
      }

      const newPendingSteps = updatedSteps.filter(
        (s) => !freshIds[s.number] && !next.messages.some((m) => m.stepNumber === s.number)
      )
      if (newPendingSteps.length > 0) {
        const newMsgs = newPendingSteps.map((step) => {
          const id = crypto.randomUUID()
          freshIds[step.number] = id
          return {
            id,
            type: 'step' as const,
            content: `Step ${step.number}: ${step.goal || 'Pending'}`,
            timestamp: new Date(),
            stepNumber: step.number,
            isLive: false,
            isPending: true,
            role: step.domain ? `${step.domain}/${step.role_id}` : step.role_id,
          }
        })
        next = { ...next, messages: [...next.messages, ...newMsgs] }
      }
      return { ...next, stepMessageIds: freshIds }
    }

    case 'step_start': {
      const goal = (event.data.goal as string) || 'Processing'
      const isFirstStep = state.currentStepNumber === 0
      let next = updateStepMessage(state, event.step_number, `Step ${event.step_number}: ${goal}...`)

      const startMsgId = next.stepMessageIds[event.step_number]
      if (startMsgId) {
        const agent = event.data.agent as string | undefined
        const stepDomain = event.data.domain as string | undefined
        const qualifiedRole = agent && stepDomain ? `${stepDomain}/${agent}` : undefined
        next = {
          ...next,
          messages: next.messages.map((m) =>
            m.id === startMsgId
              ? { ...m, stepStartedAt: Date.now(), stepAttempts: 0, ...(qualifiedRole ? { role: qualifiedRole } : agent && !m.role ? { role: agent } : {}) }
              : m
          ),
        }
      }
      return {
        ...next,
        status: 'executing',
        currentStepNumber: event.step_number,
        stepAttempt: 1,
        ...(isFirstStep ? { lastQueryStartStep: event.step_number } : {}),
      }
    }

    case 'step_generating': {
      const goal = (event.data.goal as string) || ''
      const attempt = state.stepAttempt > 1 ? ` (attempt ${state.stepAttempt})` : ''
      const goalPrefix = goal ? `${goal}. ` : ''
      const genMsgId = state.stepMessageIds[event.step_number]
      if (!genMsgId) return { ...state, executionPhase: 'generating' }
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === genMsgId
            ? { ...m, content: `Step ${event.step_number}: ${goalPrefix}Planning${attempt}...`, isLive: true, isPending: true }
            : m
        ),
        executionPhase: 'generating',
      }
    }

    case 'model_escalation': {
      const fromModel = (event.data.from_model as string) || ''
      const toModel = (event.data.to_model as string) || ''
      const reason = (event.data.reason as string) || ''
      const shortName = (m: string) => {
        const parts = m.split('/')
        return parts[parts.length - 1].replace('Meta-Llama-', 'Llama-').replace('-Instruct-Turbo', '')
      }
      const reasonShort = reason.length > 60 ? reason.slice(0, 60) + '...' : reason
      let next = updateStepMessage(
        state,
        event.step_number,
        `Step ${event.step_number}: ${shortName(fromModel)} → ${shortName(toModel)} (${reasonShort})`
      )
      const escMsgId = next.stepMessageIds[event.step_number]
      if (escMsgId) {
        next = {
          ...next,
          messages: next.messages.map((m) =>
            m.id === escMsgId ? { ...m, stepAttempts: (m.stepAttempts || 0) + 1 } : m
          ),
        }
      }
      return { ...next, executionPhase: 'retrying' }
    }

    case 'step_executing': {
      const goal = (event.data.goal as string) || ''
      const code = (event.data.code as string) || ''
      let next = updateStepMessage(state, event.step_number, `Step ${event.step_number}: Executing${goal ? ` - ${goal}` : ''}...`)
      const sources = parseSourceTables(code)
      const execMsgId = next.stepMessageIds[event.step_number]
      if (sources.length > 0 && execMsgId) {
        next = {
          ...next,
          messages: next.messages.map((m) =>
            m.id === execMsgId ? { ...m, stepSourcesRead: sources } : m
          ),
        }
      }
      return { ...next, executionPhase: 'executing' }
    }

    case 'step_error': {
      const newAttempt = state.stepAttempt + 1
      let next = updateStepMessage(state, event.step_number, `Step ${event.step_number}: Retrying (attempt ${newAttempt})...`)
      const errMsgId = next.stepMessageIds[event.step_number]
      if (errMsgId) {
        next = {
          ...next,
          messages: next.messages.map((m) =>
            m.id === errMsgId ? { ...m, stepAttempts: (m.stepAttempts || 0) + 1 } : m
          ),
        }
      }
      return { ...next, stepAttempt: newAttempt, executionPhase: 'retrying' }
    }

    case 'step_failed': {
      const errorMsg = (event.data.error as string) || 'Failed'
      return updateStepMessage(state, event.step_number, `Step ${event.step_number}: ❌ ${errorMsg}`, true)
    }

    case 'step_complete': {
      const result = event.data as {
        success?: boolean
        stdout?: string
        goal?: string
        code?: string
        tables_created?: string[]
        duration_ms?: number
        attempts?: number
        model?: string
      }
      const summary = result.goal || 'Completed'
      const outputSummary = result.stdout ? `\n\n${result.stdout}` : ''
      let next = updateStepMessage(
        state,
        event.step_number,
        `Step ${event.step_number}: ✓ ${summary}${outputSummary}`,
        true
      )
      const completeMsgId = next.stepMessageIds[event.step_number]
      const tablesCreated = result.tables_created || []
      const completeSources = parseSourceTables(result.code || '')
      if (completeMsgId) {
        next = {
          ...next,
          messages: next.messages.map((m) => {
            if (m.id !== completeMsgId) return m
            const duration = result.duration_ms ?? (m.stepStartedAt ? Date.now() - m.stepStartedAt : undefined)
            return {
              ...m,
              stepDurationMs: duration,
              stepAttempts: result.attempts != null ? Math.max(0, result.attempts - 1) : (m.stepAttempts || 0),
              stepTablesCreated: tablesCreated,
              stepSourcesRead: m.stepSourcesRead?.length ? m.stepSourcesRead : completeSources,
            }
          }),
        }
      }
      return next
    }

    case 'validation_retry': {
      const validation = (event.data.validation as string) || 'Validation failed'
      return { ...updateStepMessage(state, event.step_number, `Step ${event.step_number}: Retrying (${validation})...`), executionPhase: 'retrying' }
    }

    case 'validation_warnings': {
      const warnings = (event.data.warnings as string[]) || []
      if (warnings.length === 0) return state
      const warningText = warnings.map(w => `⚠ ${w}`).join('\n')
      const stepMsgId = state.stepMessageIds[event.step_number]
      if (!stepMsgId) return state
      const existing = state.messages.find(m => m.id === stepMsgId)
      if (!existing) return state
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === stepMsgId ? { ...m, content: `${existing.content}\n${warningText}` } : m
        ),
      }
    }

    case 'synthesizing':
    case 'generating_insights': {
      const withFinalized = finalizeAllSteps(state)
      const insightMsg = (event.data as { message?: string })?.message || 'Generating insights...'
      const existingThinking = withFinalized.messages.find(m => m.type === 'thinking' && m.isLive)
      if (existingThinking) {
        return {
          ...withFinalized,
          messages: withFinalized.messages.map((m) =>
            m.id === existingThinking.id ? { ...m, content: insightMsg } : m
          ),
          executionPhase: 'synthesizing',
          thinkingMessageId: existingThinking.id,
        }
      }
      const newId = Date.now().toString()
      const thinkingMessage: Message = {
        id: newId,
        type: 'thinking',
        content: insightMsg,
        timestamp: new Date(),
        isLive: true,
      }
      return {
        ...withFinalized,
        messages: [...withFinalized.messages, thinkingMessage],
        executionPhase: 'synthesizing',
        thinkingMessageId: newId,
      }
    }

    case 'query_complete': {
      let next = finalizeAllSteps(state)
      next = clearLiveMessage(next)
      const completeSuggestions = (event.data.suggestions as string[]) || []
      next = { ...next, status: 'completed', currentStepNumber: 0, stepAttempt: 1, suggestions: completeSuggestions, executionPhase: 'idle', queryContext: null }

      const totalElapsedMs = next.querySubmittedAt ? Date.now() - next.querySubmittedAt : undefined
      const lastStepMsg = [...next.messages].reverse().find((m) => m.type === 'step')
      const isBrief = event.data.brief as boolean
      const summaryOutput = (event.data.output as string) || ''

      if (lastStepMsg && !isBrief && summaryOutput) {
        const newOutputMsg: Message = {
          id: crypto.randomUUID(),
          type: 'output',
          content: summaryOutput,
          timestamp: new Date(),
          isFinalInsight: true,
          stepDurationMs: totalElapsedMs,
        }
        next = {
          ...next,
          messages: [
            ...next.messages.map((m) =>
              m.id === lastStepMsg.id ? { ...m, stepDurationMs: totalElapsedMs ?? m.stepDurationMs } : m
            ),
            newOutputMsg,
          ],
        }
      } else if (lastStepMsg) {
        next = {
          ...next,
          messages: next.messages.map((m) =>
            m.id === lastStepMsg.id
              ? { ...m, isFinalInsight: true, stepDurationMs: totalElapsedMs ?? m.stepDurationMs }
              : m
          ),
        }
      } else {
        const noStepOutput: Message = {
          id: crypto.randomUUID(),
          type: 'output',
          content: summaryOutput || 'Analysis complete',
          timestamp: new Date(),
          isFinalInsight: true,
          stepDurationMs: totalElapsedMs,
        }
        next = { ...next, messages: [...next.messages, noStepOutput] }
      }
      return next
    }

    case 'query_error': {
      let next = finalizeAllSteps(state)
      next = clearLiveMessage(next)
      const errorMsg = (event.data.error as string) || 'Query failed'
      const isRejection = errorMsg.toLowerCase().includes('rejected') || errorMsg.toLowerCase().includes('was rejected')
      const isCancellation = errorMsg.toLowerCase().includes('cancel') || errorMsg.toLowerCase().includes('cancelled')
      next = { ...next, status: 'error', currentStepNumber: 0, stepAttempt: 1, executionPhase: 'idle' }
      if (!isCancellation && !isRejection) {
        const errMsg: Message = {
          id: crypto.randomUUID(),
          type: 'error',
          content: errorMsg,
          timestamp: new Date(),
        }
        next = { ...next, messages: [...next.messages, errMsg] }
      }
      return next
    }

    case 'query_cancelled': {
      let next = finalizeAllSteps(state)
      next = clearLiveMessage(next)
      next = { ...next, status: 'cancelled', currentStepNumber: 0, stepAttempt: 1, executionPhase: 'idle' }
      const cancelMsg: Message = {
        id: crypto.randomUUID(),
        type: 'system',
        content: 'Query cancelled',
        timestamp: new Date(),
      }
      return { ...next, messages: [...next.messages, cancelMsg] }
    }

    case 'clarification_needed': {
      const data = event.data as {
        original_question: string
        ambiguity_reason: string
        questions: Array<{ text: string; suggestions: string[]; widget?: { type: string; config: Record<string, unknown> } }>
      }
      let next = clearLiveMessage(state)
      const allQuestions = (data.questions || []).map(q => q.text).filter(Boolean)
      const questionText = allQuestions.length > 1
        ? 'Please clarify:\n' + allQuestions.map((q, i) => `${i + 1}. ${q}`).join('\n')
        : allQuestions.length === 1
          ? (allQuestions[0].match(/^please clarify/i) ? allQuestions[0] : `Please clarify: ${allQuestions[0]}`)
          : 'Please clarify your question.'
      const stepMsgId = next.currentStepNumber ? next.stepMessageIds[next.currentStepNumber] : null
      if (stepMsgId) {
        const idx = next.messages.findIndex((m) => m.id === stepMsgId)
        const newMsg: Message = {
          id: crypto.randomUUID(),
          type: 'system',
          content: questionText,
          timestamp: new Date(),
        }
        if (idx >= 0) {
          const updated = [...next.messages]
          updated.splice(idx + 1, 0, newMsg)
          next = { ...next, messages: updated }
        } else {
          next = { ...next, messages: [...next.messages, newMsg] }
        }
      } else {
        const sysMsg: Message = {
          id: crypto.randomUUID(),
          type: 'system',
          content: questionText,
          timestamp: new Date(),
        }
        next = { ...next, messages: [...next.messages, sysMsg] }
      }
      return {
        ...next,
        status: 'awaiting_approval',
        executionPhase: 'idle',
        clarification: {
          needed: true,
          originalQuestion: data.original_question,
          ambiguityReason: data.ambiguity_reason,
          questions: data.questions || [],
          currentStep: 0,
          answers: {},
          structuredAnswers: {},
        },
      }
    }

    case 'steps_truncated': {
      const fromStep = event.step_number
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.type === 'step' && (m.stepNumber || 0) >= fromStep
            ? { ...m, isSuperseded: true }
            : m
        ),
      }
    }

    // Pure side-effect events — no session state change
    case 'table_created':
    case 'artifact_created':
    case 'facts_extracted':
    case 'fact_start':
    case 'fact_planning':
    case 'fact_executing':
    case 'fact_resolved':
    case 'fact_failed':
    case 'dag_execution_start':
    case 'inference_code':
    case 'proof_summary_ready':
    case 'proof_complete':
    case 'progress':
    case 'entity_rebuild_complete':
    case 'entity_rebuild_start':
    case 'entity_state':
    case 'entity_patch':
    case 'source_ingest_complete':
    case 'source_ingest_error':
    case 'source_ingest_progress':
    case 'source_ingest_start':
    case 'glossary_terms_added':
    case 'glossary_rebuild_complete':
    case 'glossary_rebuild_start':
    case 'glossary_generation_progress':
    case 'relationships_extracted':
      return state

    default:
      return state
  }
}

// ---------------------------------------------------------------------------
// executeSideEffects — all cross-store calls
// ---------------------------------------------------------------------------

export function executeSideEffects(
  event: SubscriptionEvent,
  sessionId: string,
  _stores: SideEffectStores,
  lastHeartbeatRef: { current: string | null },
): void {
  switch (event.event_type) {
    case 'heartbeat_ack': {
      const data = event.data as { server_time: string }
      lastHeartbeatRef.current = data.server_time
      break
    }

    case 'session_ready': {
      apolloClient.refetchQueries({ include: ['Entities', 'DataSources', 'Skills', 'Agents', 'ActiveDomains'] })
      glossaryFetchTerms(sessionId)
      break
    }

    case 'proof_start': {
      clearInferenceCodes()
      handleFactEvent(event.event_type, event.data as Record<string, unknown>)
      break
    }

    case 'step_executing': {
      const goal = (event.data.goal as string) || ''
      const code = (event.data.code as string) || ''
      const model = (event.data.model as string) || undefined
      if (code) {
        addStepCode(event.step_number, goal, code, model)
      }
      break
    }

    case 'step_complete': {
      const result = event.data as { goal?: string; code?: string; model?: string }
      if (result.code) {
        addStepCode(event.step_number, result.goal || '', result.code, result.model)
      }
      // No refetch during execution — query_complete will refresh all
      break
    }

    case 'synthesizing':
    case 'generating_insights': {
      apolloClient.refetchQueries({ include: ['Artifacts', 'Tables'] }).then(() => {
        const artifactsResult = apolloClient.readQuery({
          query: ARTIFACTS_QUERY,
          variables: { sessionId },
        })
        const tablesResult = apolloClient.readQuery({
          query: TABLES_QUERY,
          variables: { sessionId },
        })

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const artifacts = ((artifactsResult as any)?.artifacts?.artifacts ?? []).map(toArtifact)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const tables = ((tablesResult as any)?.tables?.tables ?? []).map(toTableInfo)

        const sectionsToExpand: string[] = []
        const publishedArtifacts = artifacts.filter((a: ReturnType<typeof toArtifact>) => a.is_key_result)
        const nonTableArtifacts = publishedArtifacts.filter((a: ReturnType<typeof toArtifact>) => a.artifact_type !== 'table')
        const tableArtifacts = publishedArtifacts.filter((a: ReturnType<typeof toArtifact>) => a.artifact_type === 'table')

        if (publishedArtifacts.length > 0) sectionsToExpand.push('artifacts')
        if (tables.length > 0) sectionsToExpand.push('tables')
        if (sectionsToExpand.length > 0) expandSections(sectionsToExpand)

        const candidates = nonTableArtifacts.length > 0 ? nonTableArtifacts : tableArtifacts
        if (candidates.length > 0) {
          const markdownTypes = ['markdown', 'md']
          const sortedCandidates = [...candidates].sort((a: ReturnType<typeof toArtifact>, b: ReturnType<typeof toArtifact>) => {
            const aIsMarkdown = markdownTypes.includes(a.artifact_type?.toLowerCase() || '')
            const bIsMarkdown = markdownTypes.includes(b.artifact_type?.toLowerCase() || '')
            if (aIsMarkdown && !bIsMarkdown) return -1
            if (!aIsMarkdown && bIsMarkdown) return 1
            return b.step_number - a.step_number
          })
          const best = sortedCandidates[0]
          apolloClient.query({
            query: ARTIFACT_QUERY,
            variables: { sessionId, id: best.id },
            fetchPolicy: 'network-only',
          }).then((res) => {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const content = (res.data as any)?.artifact
            if (content) {
              selectedArtifactVar(toArtifactContent(content))
            }
          }).catch(err => {
            console.error('[synthesizing] Error fetching artifact content:', err)
          })
        }
      }).catch(err => {
        console.error('[synthesizing] Error fetching artifacts/tables:', err)
      })
      break
    }

    case 'query_complete': {
      // Terminal event — refresh all query data once
      apolloClient.refetchQueries({
        include: ['Tables', 'Artifacts', 'Facts', 'InferenceCodes', 'Steps', 'Learnings', 'Scratchpad', 'SessionDDL'],
      })
      break
    }

    case 'table_created': {
      // Write directly to Apollo cache — no network call
      const tableData = event.data as { name: string; row_count?: number; columns?: string[] }
      const existing = apolloClient.readQuery({ query: TABLES_QUERY, variables: { sessionId } })
      if (existing) {
        const newTable = {
          __typename: 'TableInfoType',
          name: tableData.name,
          rowCount: tableData.row_count || 0,
          stepNumber: event.step_number,
          columns: tableData.columns || [],
          isStarred: false,
          isView: false,
          roleId: null,
          version: 1,
          versionCount: 1,
        }
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const tables = (existing as any).tables.tables
        const idx = tables.findIndex((t: { name: string }) => t.name === tableData.name)
        const updated = idx >= 0 ? tables.map((t: { name: string }) => t.name === tableData.name ? newTable : t) : [...tables, newTable]
        apolloClient.writeQuery({
          query: TABLES_QUERY,
          variables: { sessionId },
          data: { tables: { __typename: 'TablesResult', tables: updated, total: updated.length } },
        })
      }
      break
    }

    case 'artifact_created': {
      // Write directly to Apollo cache — no network call
      const artifactData = event.data as { id?: number; name?: string; artifact_type?: string; title?: string; description?: string; mime_type?: string; is_key_result?: boolean }
      if (artifactData.id && artifactData.name && artifactData.artifact_type) {
        const existing = apolloClient.readQuery({ query: ARTIFACTS_QUERY, variables: { sessionId } })
        if (existing) {
          const newArtifact = {
            __typename: 'ArtifactInfoType',
            id: artifactData.id,
            name: artifactData.name,
            artifactType: artifactData.artifact_type,
            stepNumber: event.step_number,
            title: artifactData.title || null,
            description: artifactData.description || null,
            mimeType: artifactData.mime_type || 'application/octet-stream',
            createdAt: new Date().toISOString(),
            isStarred: artifactData.is_key_result || false,
            metadata: null,
            roleId: null,
            version: 1,
            versionCount: 1,
          }
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const artifacts = (existing as any).artifacts.artifacts
          const idx = artifacts.findIndex((a: { id: number }) => a.id === artifactData.id)
          const updated = idx >= 0 ? artifacts.map((a: { id: number }) => a.id === artifactData.id ? newArtifact : a) : [...artifacts, newArtifact]
          apolloClient.writeQuery({
            query: ARTIFACTS_QUERY,
            variables: { sessionId },
            data: { artifacts: { __typename: 'ArtifactsResult', artifacts: updated, total: updated.length } },
          })
        }
      }
      break
    }

    case 'facts_extracted': {
      // Write directly to Apollo cache — no network call
      const factsData = event.data as { facts?: Array<{ name: string; value?: unknown; source?: string; reasoning?: string; confidence?: number }> }
      if (factsData.facts && factsData.facts.length > 0) {
        const existing = apolloClient.readQuery({ query: FACTS_QUERY, variables: { sessionId } })
        if (existing) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const currentFacts = [...(existing as any).facts.facts]
          for (const fact of factsData.facts) {
            const newFact = {
              __typename: 'FactInfoType',
              name: fact.name,
              value: fact.value ?? null,
              source: fact.source ?? null,
              reasoning: fact.reasoning ?? null,
              confidence: fact.confidence ?? null,
              isPersisted: false,
              roleId: null,
              domain: null,
            }
            const idx = currentFacts.findIndex((f: { name: string }) => f.name === fact.name)
            if (idx >= 0) currentFacts[idx] = newFact
            else currentFacts.push(newFact)
          }
          apolloClient.writeQuery({
            query: FACTS_QUERY,
            variables: { sessionId },
            data: { facts: { __typename: 'FactsResult', facts: currentFacts, total: currentFacts.length } },
          })
        }
      }
      break
    }

    case 'steps_truncated': {
      truncateFromStep(event.step_number)
      // Truncation changes visible state — refresh affected queries
      apolloClient.refetchQueries({ include: ['Tables', 'Artifacts', 'SessionDDL'] })
      break
    }

    case 'fact_start':
    case 'fact_planning':
    case 'fact_executing':
    case 'fact_resolved':
    case 'fact_failed':
    case 'dag_execution_start':
      handleFactEvent(event.event_type, event.data as Record<string, unknown>)
      break

    case 'inference_code': {
      handleFactEvent(event.event_type, event.data as Record<string, unknown>)
      const icData = event.data as Record<string, unknown>
      if (icData.inference_id && icData.code) {
        addInferenceCode({
          inference_id: icData.inference_id as string,
          name: (icData.name as string) || '',
          operation: (icData.operation as string) || '',
          code: icData.code as string,
          attempt: icData.attempt as number,
          model: (icData.model as string) || undefined,
        })
      }
      break
    }

    case 'proof_summary_ready': {
      handleFactEvent(event.event_type, event.data as Record<string, unknown>)
      const facts = exportFacts()
      const summary = (event.data as Record<string, unknown>).summary as string
      if (facts.length > 0) {
        apolloClient.mutate({ mutation: SAVE_PROOF_FACTS, variables: { sessionId, facts, summary } }).catch(err => {
          console.error('Failed to save proof summary:', err)
        })
      }
      break
    }

    case 'proof_complete': {
      handleFactEvent(event.event_type, event.data as Record<string, unknown>)
      setTimeout(() => {
        if (isSummaryGeneratingVar()) {
          const currentFacts = exportFacts()
          if (currentFacts.length > 0) {
            apolloClient.mutate({ mutation: SAVE_PROOF_FACTS, variables: { sessionId, facts: currentFacts, summary: null } }).catch(err => {
              console.error('Failed to save proof facts (fallback):', err)
            })
          }
          handleFactEvent('proof_summary_ready', { summary: null })
        }
      }, 30000)
      break
    }

    case 'entity_rebuild_complete': {
      apolloClient.refetchQueries({ include: ['Entities'] })
      setEntityRebuilding(false)
      break
    }

    case 'entity_rebuild_start':
      setEntityRebuilding(true)
      break

    case 'entity_state': {
      const { state, version } = event.data as { state: CompactState; version: number }
      const { terms, totalDefined, totalSelfDescribing } = inflateToGlossaryTerms(state)
      setTermsFromState(terms, totalDefined, totalSelfDescribing)
      setCachedEntry(sessionId, state, version)
      break
    }

    case 'entity_patch': {
      const { patch, version } = event.data as { patch: Operation[]; version: number }
      getCachedEntry(sessionId).then((entry) => {
        const base: CompactState = entry?.state ?? { e: {}, g: {}, r: {}, k: {} }
        try {
          const { newDocument } = applyPatch(base, patch, false, false)
          const newState = newDocument as CompactState
          const { terms, totalDefined, totalSelfDescribing } = inflateToGlossaryTerms(newState)
          setTermsFromState(terms, totalDefined, totalSelfDescribing)
          setCachedEntry(sessionId, newState, version)
        } catch (err) {
          console.warn('[entity_patch] patch failed, requesting full state', err)
        }
      })
      break
    }

    case 'source_ingest_complete':
      apolloClient.refetchQueries({ include: ['DataSources'] })
      ingestingSourceVar(null)
      ingestProgressVar(null)
      break

    case 'source_ingest_error': {
      const errData = event.data as { name?: string; error?: string }
      console.error(`[source_ingest_error] ${errData.name}: ${errData.error}`)
      break
    }

    case 'source_ingest_progress': {
      const progressData = event.data as { current: number; total: number }
      ingestProgressVar({ current: progressData.current, total: progressData.total })
      break
    }

    case 'source_ingest_start': {
      const startData = event.data as { name?: string }
      ingestingSourceVar(startData.name || null)
      ingestProgressVar(null)
      break
    }

    case 'glossary_terms_added': {
      const termsData = event.data as { terms?: GlossaryTerm[] }
      if (termsData.terms && termsData.terms.length > 0) {
        glossaryAddTerms(termsData.terms!)
      }
      break
    }

    case 'glossary_rebuild_complete':
      setGenerating(false)
      break

    case 'glossary_rebuild_start':
      setGenerating(true)
      break

    case 'glossary_generation_progress': {
      const { stage, percent } = event.data as { stage: string; percent: number }
      setProgress(stage, percent)
      break
    }

    case 'relationships_extracted':
      bumpRefreshKey()
      break

    default:
      break
  }
}
