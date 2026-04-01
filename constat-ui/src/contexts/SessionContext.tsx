// Copyright (c) 2025 Kenneth Stott
// Canary: b8d3d556-cbd7-4ec9-89f1-d417877ef209
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import {
  createContext,
  useContext,
  useCallback,
  useReducer,
  useState,
  useRef,
  useEffect,
  useMemo,
  type ReactNode,
} from 'react'
import { useReactiveVar, type ObservableSubscription } from '@apollo/client'
import { useAuth } from '@/contexts/AuthContext'
import {
  clearArtifactState,
  markStepsSuperseded,
  proofFactsVar,
  isProvingVar,
  isPlanningCompleteVar,
  isProofPanelOpenVar,
  proofSummaryVar,
  isSummaryGeneratingVar,
  hasCompletedProofVar,
  openProofPanel,
  closeProofPanel,
  clearProofFacts,
  importFacts,
} from '@/graphql/ui-state'
import { fetchTerms as glossaryFetchTerms, loadFromCache as glossaryLoadFromCache } from '@/store/glossaryState'
import { briefModeVar } from '@/graphql/ui-state'
import { apolloClient } from '@/graphql/client'
import { sessionEventReducer, executeSideEffects } from '@/events/sessionEventHandler'
import {
  type SessionExecutionState,
  type Message,
  type SideEffectStores,
  initialExecutionState,
} from '@/events/types'
import {
  CREATE_SESSION,
  SHARE_SESSION,
  SET_ACTIVE_DOMAINS,
  toSession,
} from '@/graphql/operations/sessions'
import {
  MESSAGES_QUERY,
  PROOF_FACTS_QUERY,
  toStoredMessage,
  toStoredProofFact,
} from '@/graphql/operations/state'
import {
  QUERY_EXECUTION_SUBSCRIPTION,
  SUBMIT_QUERY,
  CANCEL_EXECUTION,
  APPROVE_PLAN,
  ANSWER_CLARIFICATION,
  SKIP_CLARIFICATION,
  REPLAN_FROM,
  EDIT_OBJECTIVE,
  DELETE_OBJECTIVE,
  HEARTBEAT,
  toExecutionEvent,
} from '@/graphql/operations/execution'
import { ACTIVATE_AGENT } from '@/graphql/operations/learnings'
import { getOrCreateSessionId, createNewSessionId } from '@/api/session-id'
import type { Session, SessionStatus, Plan } from '@/types/api'
import type { ClarificationState, QueuedMessage } from '@/events/types'

// Re-export for consumers that import these types from this module
export type { Message, ClarificationState, QueuedMessage }

type ExecutionPhase =
  | 'idle'
  | 'planning'
  | 'awaiting_approval'
  | 'generating'
  | 'executing'
  | 'retrying'
  | 'summarizing'
  | 'synthesizing'

interface FactNode {
  id: string
  name: string
  description?: string
  status: 'pending' | 'planning' | 'executing' | 'resolved' | 'failed' | 'blocked'
  value?: unknown
  source?: string
  confidence?: number
  tier?: number
  strategy?: string
  formula?: string
  reason?: string
  dependencies: string[]
  elapsed_ms?: number
  attempt?: number
  code?: string
}

interface AgentInfo {
  name: string
  prompt: string
  is_active: boolean
}

interface SessionContextValue {
  // Session
  sessionId: string | null
  session: Session | null
  activeDomains: string[]
  setActiveDomains: (domains: string[]) => Promise<void>
  updateSession: (updates: Partial<Session>) => void
  setSession: (session: Session | null, options?: { preserveMessages?: boolean }) => void
  createSession: (userId?: string, forceNew?: boolean) => Promise<void>

  // Connection
  subscriptionConnected: boolean
  sessionReady: boolean
  isCreatingSession: boolean

  // Execution state
  status: SessionStatus
  executionPhase: ExecutionPhase
  plan: Plan | null
  currentQuery: string

  // Messages
  messages: Message[]
  suggestions: string[]
  welcomeTagline: string
  queuedMessages: QueuedMessage[]

  // Clarification
  clarification: ClarificationState | null

  // Agents
  agents: AgentInfo[]
  currentAgent: string | null
  fetchAgents: () => Promise<void>
  setAgent: (agentName: string | null) => Promise<void>

  // Actions
  submitQuery: (problem: string, isFollowup?: boolean) => Promise<void>
  cancelExecution: () => Promise<void>
  approvePlan: (deletedSteps?: number[], editedSteps?: Array<{ number: number; goal: string }>) => Promise<void>
  rejectPlan: (feedback: string, editedSteps?: Array<{ number: number; goal: string }>) => Promise<void>
  shareSession: (email: string) => Promise<{ share_url: string }>
  replanFromStep: (stepNumber: number, mode: 'edit' | 'delete' | 'redo', editedGoal?: string) => void
  removeQueuedMessage: (id: string) => void
  editObjective: (objectiveIndex: number, newText: string) => void
  deleteObjective: (objectiveIndex: number) => void

  // Proof state
  proofFacts: Map<string, FactNode>
  isProofPanelOpen: boolean
  isPlanningComplete: boolean
  proofSummary: string | null
  isSummaryGenerating: boolean
  isProving: boolean
  hasCompletedProof: boolean

  // Proof actions
  openProofPanel: () => void
  closeProofPanel: () => void
  clearProofFacts: () => void

  // Clarification actions
  answerClarification: (answers: Record<number, string>, structuredAnswers?: Record<number, unknown>) => void
  skipClarification: () => void
  setClarificationStep: (step: number) => void
  setClarificationAnswer: (step: number, answer: string) => void
  setClarificationStructuredAnswer: (step: number, data: unknown) => void

  // Session switching (for HamburgerMenu)
  switchSession: (sessionId: string) => Promise<void>
}

const SessionContext = createContext<SessionContextValue | null>(null)

export function SessionProvider({ children }: { children: ReactNode }) {
  const { userId } = useAuth()

  // useState — imperative, not event-driven
  const [session, setSessionState] = useState<Session | null>(null)
  const [subscriptionConnected, setSubscriptionConnected] = useState(false)
  const [sessionReady, setSessionReady] = useState(false)
  const [isCreatingSession, setIsCreatingSession] = useState(false)
  const [agents, setAgents] = useState<AgentInfo[]>([])
  const [currentAgent, setCurrentAgent] = useState<string | null>(null)

  // useReducer — event-driven execution state
  const [execState, dispatch] = useReducer(sessionEventReducer, initialExecutionState)

  // useRef — subscription lifecycle + stale-closure bridge
  const subscriptionRef = useRef<ObservableSubscription | null>(null)
  const heartbeatRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const lastHeartbeatRef = useRef<string | null>(null)
  const execStateRef = useRef<SessionExecutionState>(execState)
  const userIdRef = useRef<string>(userId)
  const sessionRef = useRef<Session | null>(session)

  // Keep refs in sync
  useEffect(() => { execStateRef.current = execState }, [execState])
  useEffect(() => { userIdRef.current = userId }, [userId])
  useEffect(() => { sessionRef.current = session }, [session])

  // Proof reactive var state
  const proofFacts = useReactiveVar(proofFactsVar)
  const isProofPanelOpen = useReactiveVar(isProofPanelOpenVar)
  const isPlanningComplete = useReactiveVar(isPlanningCompleteVar)
  const proofSummary = useReactiveVar(proofSummaryVar)
  const isSummaryGenerating = useReactiveVar(isSummaryGeneratingVar)
  const isProving = useReactiveVar(isProvingVar)
  const hasCompletedProof = useReactiveVar(hasCompletedProofVar)

  // SideEffectStores — all stores migrated to reactive vars, empty object kept for call-site compat
  const sideEffectStores: SideEffectStores = useMemo(() => ({}), [])

  // ---------------------------------------------------------------------------
  // Subscription helpers
  // ---------------------------------------------------------------------------

  const stopSubscription = useCallback(() => {
    subscriptionRef.current?.unsubscribe()
    subscriptionRef.current = null
    if (heartbeatRef.current) {
      clearInterval(heartbeatRef.current)
      heartbeatRef.current = null
    }
  }, [])

  const startSubscription = useCallback((sess: Session) => {
    subscriptionRef.current = apolloClient
      .subscribe({
        query: QUERY_EXECUTION_SUBSCRIPTION,
        variables: { sessionId: sess.session_id },
      })
      .subscribe({
        next({ data }) {
          if (!data?.queryExecution) return
          const event = toExecutionEvent(data.queryExecution)

          // session_ready: mark session as ready (active domains handled by Apollo cache)
          if (event.event_type === 'session_ready') {
            setSessionReady(true)
          }

          // Reducer handles execution state
          dispatch({ type: 'SUBSCRIPTION_EVENT', event })

          // Side effects for cross-store calls
          executeSideEffects(event, sess.session_id, sideEffectStores, lastHeartbeatRef)

          // Process queued messages on terminal events
          if (
            event.event_type === 'query_complete' ||
            event.event_type === 'query_error' ||
            event.event_type === 'query_cancelled'
          ) {
            setTimeout(() => {
              const queued = execStateRef.current.queuedMessages
              if (queued.length > 0) {
                const next = queued[0]
                dispatch({ type: 'REMOVE_QUEUED_MESSAGE', id: next.id })
                // submitQuery reads sessionRef inside — safe to call here
                submitQueryInternal(next.content, true)
              }
            }, 500)
          }
        },
        error(err) {
          console.error('[subscription] error:', err)
          setSubscriptionConnected(false)
        },
      })
    setSubscriptionConnected(true)

    heartbeatRef.current = setInterval(() => {
      apolloClient
        .mutate({
          mutation: HEARTBEAT,
          variables: { sessionId: sess.session_id, since: lastHeartbeatRef.current },
        })
        .catch((err) => console.error('[heartbeat] error:', err))
    }, 30_000)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sideEffectStores])

  // ---------------------------------------------------------------------------
  // submitQuery — declared early via ref so startSubscription can call it
  // ---------------------------------------------------------------------------

  // We use a ref indirection to avoid circular dependency with startSubscription
  const submitQueryRef = useRef<(problem: string, isFollowup?: boolean) => Promise<void>>(async () => {})

  const submitQueryInternal = useCallback(
    (problem: string, isFollowup = false) => submitQueryRef.current(problem, isFollowup),
    []
  )

  // ---------------------------------------------------------------------------
  // createSession
  // ---------------------------------------------------------------------------

  const createSession = useCallback(async (overrideUserId?: string, forceNew = false) => {
    const uid = overrideUserId ?? userIdRef.current ?? 'default'
    setIsCreatingSession(true)
    stopSubscription()
    clearArtifactState()

    const sessionId = forceNew
      ? createNewSessionId(uid)
      : getOrCreateSessionId(uid)

    const { data } = await apolloClient.mutate({
      mutation: CREATE_SESSION,
      variables: { userId: uid, sessionId },
    })
    const sess = toSession(data.createSession)

    let restoredMessages: Message[] = []
    if (!forceNew) {
      const [messagesRes, factsRes, _codesRes, _artifactsRes] = await Promise.allSettled([
        apolloClient
          .query({ query: MESSAGES_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
          .then(({ data: d }) => ({ messages: d.messages.messages.map(toStoredMessage) })),
        apolloClient
          .query({ query: PROOF_FACTS_QUERY, variables: { sessionId }, fetchPolicy: 'network-only' })
          .then(({ data: d }) => ({
            facts: d.proofFacts.facts.map(toStoredProofFact),
            summary: d.proofFacts.summary,
          })),
        apolloClient.refetchQueries({ include: ['Steps', 'InferenceCodes'] }),
        apolloClient.refetchQueries({ include: ['Tables', 'Artifacts'] }),
      ])

      if (messagesRes.status === 'fulfilled') {
        const { messages: stored } = messagesRes.value
        if (stored && stored.length > 0) {
          restoredMessages = stored.map((m: ReturnType<typeof toStoredMessage>) => ({
            id: m.id,
            type: m.type as Message['type'],
            content: m.content,
            timestamp: new Date(m.timestamp),
            stepNumber: m.stepNumber,
            isFinalInsight: m.isFinalInsight,
            stepDurationMs: m.stepDurationMs,
            role: m.role,
            skills: m.skills,
          }))
          console.log('[createSession] Restored', restoredMessages.length, 'messages')
        }
      } else {
        console.warn('[createSession] Could not restore messages:', messagesRes.reason)
      }

      if (factsRes.status === 'fulfilled') {
        const { facts: storedFacts, summary } = factsRes.value
        if (storedFacts && storedFacts.length > 0) {
          importFacts(storedFacts, summary)
          console.log('[createSession] Restored', storedFacts.length, 'proof facts')
        }
      } else {
        console.warn('[createSession] Could not restore proof facts:', factsRes.reason)
      }

      if (_codesRes.status === 'fulfilled') {
        console.log('[createSession] Restored step and inference codes')
      } else {
        console.warn('[createSession] Could not restore codes:', _codesRes.reason)
      }
    }

    const isReconnect = (sess.active_domains?.length ?? 0) > 0

    setSessionState(sess)
    setSessionReady(isReconnect)
    setIsCreatingSession(false)
    dispatch({ type: 'RESET', messages: restoredMessages })

    startSubscription(sess)

    if (isReconnect) {
      glossaryFetchTerms(sess.session_id)
    }
  }, [stopSubscription, startSubscription])

  // ---------------------------------------------------------------------------
  // setSession
  // ---------------------------------------------------------------------------

  const setSession = useCallback(
    (sess: Session | null, options?: { preserveMessages?: boolean }) => {
      stopSubscription()
      if (sess) {
        if (!options?.preserveMessages) {
          dispatch({ type: 'SET_MESSAGES', messages: [], suggestions: [], plan: null })
        }
        glossaryLoadFromCache(sess.session_id)
        startSubscription(sess)
      } else {
        setSubscriptionConnected(false)
      }
      setSessionState(sess)
      if (sess?.status) {
        dispatch({ type: 'SET_STATUS', status: sess.status })
      }
    },
    [stopSubscription, startSubscription]
  )

  // ---------------------------------------------------------------------------
  // updateSession
  // ---------------------------------------------------------------------------

  const updateSession = useCallback((updates: Partial<Session>) => {
    setSessionState((prev) => (prev ? { ...prev, ...updates } : prev))
  }, [])

  // ---------------------------------------------------------------------------
  // setActiveDomains
  // ---------------------------------------------------------------------------

  const setActiveDomains = useCallback(async (domains: string[]) => {
    const sess = sessionRef.current
    if (!sess) return
    await apolloClient.mutate({
      mutation: SET_ACTIVE_DOMAINS,
      variables: { sessionId: sess.session_id, domains },
    })
    updateSession({ active_domains: domains })
  }, [updateSession])

  // ---------------------------------------------------------------------------
  // submitQuery
  // ---------------------------------------------------------------------------

  const submitQuery = useCallback(async (problem: string, isFollowup = false) => {
    const sess = sessionRef.current
    if (!sess) return

    const { suggestions, status, executionPhase } = execStateRef.current

    // Expand number shortcuts
    let expandedProblem = problem
    const trimmed = problem.trim()
    if (/^\d+$/.test(trimmed) && suggestions.length > 0) {
      const idx = parseInt(trimmed, 10) - 1
      if (idx >= 0 && idx < suggestions.length) {
        expandedProblem = suggestions[idx]
      }
    }

    // Detect @vera learning
    const veraMatch = expandedProblem.match(/^@vera\s+(.+)/is)
    if (veraMatch) {
      expandedProblem = `/rule ${veraMatch[1].trim()}`
    }

    // Detect @username share
    const shareMatch = !veraMatch && expandedProblem.match(/^@(\S+)\s*(.*)/s)
    if (shareMatch) {
      const targetUser = shareMatch[1]
      dispatch({
        type: 'ADD_MESSAGE',
        message: {
          id: crypto.randomUUID(),
          type: 'user',
          content: problem,
          timestamp: new Date(),
          defaultExpanded: true,
        },
      })
      try {
        const result = await shareSessionFn(targetUser)
        dispatch({
          type: 'ADD_MESSAGE',
          message: {
            id: crypto.randomUUID(),
            type: 'output',
            content: `**Shared** with \`${targetUser}\`\n\n[Share link](${result.share_url})`,
            timestamp: new Date(),
            isFinalInsight: true,
          },
        })
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Share failed'
        dispatch({
          type: 'ADD_MESSAGE',
          message: { id: crypto.randomUUID(), type: 'error', content: msg, timestamp: new Date() },
        })
      }
      return
    }

    // Apply brief mode prefix
    if (briefModeVar() && !expandedProblem.startsWith('/') && !/^briefly\b/i.test(expandedProblem)) {
      expandedProblem = `briefly, ${expandedProblem}`
    }

    // Check if busy
    const isBusy =
      status === 'planning' ||
      status === 'executing' ||
      status === 'awaiting_approval' ||
      executionPhase !== 'idle'

    if (isBusy) {
      dispatch({
        type: 'ADD_QUEUED_MESSAGE',
        message: {
          id: crypto.randomUUID(),
          content: expandedProblem,
          timestamp: new Date(),
        },
      })
      return
    }

    // Not busy — add user message
    dispatch({
      type: 'ADD_MESSAGE',
      message: {
        id: crypto.randomUUID(),
        type: 'user',
        content: veraMatch ? problem : expandedProblem,
        timestamp: new Date(),
        defaultExpanded: true,
      },
    })

    const thinkingId = crypto.randomUUID()
    const isRedo = /^\/redo\b/i.test(expandedProblem.trim()) || !isFollowup
    dispatch({ type: 'SUBMIT_QUERY', query: expandedProblem, thinkingId, isRedo })

    const { data: submitData } = await apolloClient.mutate({
      mutation: SUBMIT_QUERY,
      variables: {
        sessionId: sess.session_id,
        input: { problem: expandedProblem, isFollowup },
      },
    })
    const response = submitData.submitQuery as { status: string; message?: string }

    // Slash commands return immediately
    if ((response.status === 'completed' || response.status === 'error') && response.message) {
      // Remove thinking message
      dispatch({ type: 'REMOVE_MESSAGE', id: thinkingId })

      let content = response.message
      if (veraMatch && response.status === 'completed') {
        const ruleIdMatch = response.message.match(/`(rule_[a-f0-9]+)`/)
        const ruleId = ruleIdMatch?.[1] ?? 'unknown'
        const uid = userIdRef.current
        content = `**Rule:** \`${ruleId}\`\n**Domain:** ${uid}\n**Summary:** ${veraMatch[1].trim()}\n\n*Tip: You can move this rule into a domain with* \`/move-rule ${ruleId} <domain-name>\``
      }

      dispatch({
        type: 'ADD_MESSAGE',
        message: {
          id: crypto.randomUUID(),
          type: response.status === 'error' ? 'error' : 'output',
          content,
          timestamp: new Date(),
          isFinalInsight: response.status === 'completed',
        },
      })
      dispatch({
        type: 'SET_STATUS',
        status: response.status === 'error' ? 'error' : 'completed',
      })
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Wire submitQuery into the ref (used by subscription callback and submitQueryInternal)
  useEffect(() => {
    submitQueryRef.current = submitQuery
  }, [submitQuery])

  // Stable shareSession ref for use inside submitQuery
  const shareSessionRef = useRef<(email: string) => Promise<{ share_url: string }>>(async () => {
    throw new Error('No active session')
  })

  const shareSessionFn = useCallback(
    (email: string) => shareSessionRef.current(email),
    []
  )

  const shareSession = useCallback(async (email: string): Promise<{ share_url: string }> => {
    const sess = sessionRef.current
    if (!sess) throw new Error('No active session')
    const { data } = await apolloClient.mutate({
      mutation: SHARE_SESSION,
      variables: { sessionId: sess.session_id, email },
    })
    return { share_url: data.shareSession.shareUrl }
  }, [])

  useEffect(() => {
    shareSessionRef.current = shareSession
  }, [shareSession])

  // ---------------------------------------------------------------------------
  // cancelExecution
  // ---------------------------------------------------------------------------

  const cancelExecution = useCallback(async () => {
    const sess = sessionRef.current
    if (!sess) return
    await apolloClient.mutate({
      mutation: CANCEL_EXECUTION,
      variables: { sessionId: sess.session_id },
    })
    dispatch({ type: 'CANCEL_EXECUTION' })
  }, [])

  // ---------------------------------------------------------------------------
  // approvePlan
  // ---------------------------------------------------------------------------

  const approvePlan = useCallback(
    async (
      deletedSteps?: number[],
      editedSteps?: Array<{ number: number; goal: string }>,
    ) => {
      const sess = sessionRef.current
      if (!sess) return
      const { plan, messages, isRedo } = execStateRef.current

      let stepsToShow: Array<{
        number: number
        goal: string
        role_id?: string
        domain?: string
        skill_ids?: string[]
      }>
      if (editedSteps && editedSteps.length > 0) {
        stepsToShow = editedSteps.map((s) => ({ number: s.number, goal: s.goal }))
      } else {
        const allSteps = plan?.steps || []
        const deletedSet = new Set(deletedSteps || [])
        stepsToShow = allSteps
          .filter((_step, index) => {
            const stepNum = allSteps[index].number ?? index + 1
            return !deletedSet.has(stepNum)
          })
          .map((step, index) => ({
            number: step.number ?? index + 1,
            goal: step.goal || '',
            role_id: step.role_id ?? undefined,
            domain: step.domain ?? undefined,
            skill_ids: step.skill_ids ?? undefined,
          }))
      }

      const existingStepNums = new Set(
        messages
          .filter(
            (m) => m.type === 'step' && m.stepNumber !== undefined && !m.isSuperseded,
          )
          .map((m) => m.stepNumber),
      )
      const stepMessageIds: Record<number, string> = {}
      const stepMessages: Message[] = stepsToShow
        .filter((step) => !existingStepNums.has(step.number))
        .map((step) => {
          const id = crypto.randomUUID()
          stepMessageIds[step.number] = id
          return {
            id,
            type: 'step' as const,
            content: `Step ${step.number}: ${step.goal || 'Pending'}`,
            timestamp: new Date(),
            stepNumber: step.number,
            isLive: false,
            isPending: true,
            ...(step.role_id
              ? { role: step.domain ? `${step.domain}/${step.role_id}` : step.role_id }
              : {}),
            ...(step.skill_ids?.length ? { skills: step.skill_ids } : {}),
          }
        })

      stepMessages.sort((a, b) => (a.stepNumber || 0) - (b.stepNumber || 0))

      await apolloClient.mutate({
        mutation: APPROVE_PLAN,
        variables: {
          sessionId: sess.session_id,
          input: { approved: true, deletedSteps, editedSteps },
        },
      })

      if (isRedo) {
        markStepsSuperseded()
      }

      dispatch({ type: 'APPROVE_PLAN', stepMessages, stepMessageIds, isRedo })
    },
    [],
  )

  // ---------------------------------------------------------------------------
  // rejectPlan
  // ---------------------------------------------------------------------------

  const rejectPlan = useCallback(
    async (feedback: string, editedSteps?: Array<{ number: number; goal: string }>) => {
      const sess = sessionRef.current
      if (!sess) return

      if (feedback && feedback !== 'Cancelled by user') {
        dispatch({
          type: 'ADD_MESSAGE',
          message: {
            id: crypto.randomUUID(),
            type: 'user',
            content: feedback,
            timestamp: new Date(),
          },
        })
        await apolloClient.mutate({
          mutation: APPROVE_PLAN,
          variables: {
            sessionId: sess.session_id,
            input: { approved: false, feedback, editedSteps },
          },
        })
        dispatch({ type: 'REJECT_PLAN', hasFeedback: true })
      } else {
        dispatch({
          type: 'ADD_MESSAGE',
          message: {
            id: crypto.randomUUID(),
            type: 'system',
            content: 'Plan cancelled',
            timestamp: new Date(),
          },
        })
        await apolloClient.mutate({
          mutation: APPROVE_PLAN,
          variables: {
            sessionId: sess.session_id,
            input: { approved: false, feedback },
          },
        })
        dispatch({ type: 'REJECT_PLAN', hasFeedback: false })
      }
    },
    [],
  )

  // ---------------------------------------------------------------------------
  // answerClarification
  // ---------------------------------------------------------------------------

  const answerClarification = useCallback(
    (answers: Record<number, string>, structuredAnswers?: Record<number, unknown>) => {
      const { clarification, currentStepNumber, stepMessageIds } = execStateRef.current
      if (!clarification) return

      const finalStructured: Record<string, unknown> =
        structuredAnswers || clarification.structuredAnswers || {}

      const questionTexts = clarification.questions || []
      const textKeyedAnswers: Record<string, string> = {}
      const textKeyedStructured: Record<string, unknown> = {}
      for (const [key, value] of Object.entries(answers)) {
        const idx = parseInt(String(key))
        const qText = questionTexts[idx]?.text || String(key)
        textKeyedAnswers[qText] = value as string
        if (finalStructured[key] !== undefined) {
          textKeyedStructured[qText] = finalStructured[key]
        }
      }

      const structuredPayload =
        Object.keys(textKeyedStructured).length > 0 ? textKeyedStructured : undefined
      const sess = sessionRef.current
      if (sess) {
        apolloClient
          .mutate({
            mutation: ANSWER_CLARIFICATION,
            variables: {
              sessionId: sess.session_id,
              answers: textKeyedAnswers,
              structuredAnswers: structuredPayload,
            },
          })
          .catch((err) => console.error('[answerClarification] error:', err))
      }

      const isInputRequest = clarification.ambiguityReason === 'input_request'
      const answerSummary =
        clarification.questions.length > 1
          ? clarification.questions
              .map((q, i) => `**${q.text}**\n${answers[i] || 'Skipped'}`)
              .join('\n\n')
          : answers[0] || 'Skipped'

      const userMessage: Message = {
        id: crypto.randomUUID(),
        type: 'user',
        content: answerSummary,
        timestamp: new Date(),
      }

      const stepMsgId = currentStepNumber ? stepMessageIds[currentStepNumber] : null

      dispatch({
        type: 'ANSWER_CLARIFICATION',
        isInputRequest,
        userMessage,
        stepMsgId: stepMsgId ?? null,
      })
    },
    [],
  )

  // ---------------------------------------------------------------------------
  // skipClarification
  // ---------------------------------------------------------------------------

  const skipClarification = useCallback(() => {
    const { clarification } = execStateRef.current
    const sess = sessionRef.current
    if (sess) {
      apolloClient
        .mutate({
          mutation: SKIP_CLARIFICATION,
          variables: { sessionId: sess.session_id },
        })
        .catch((err) => console.error('[skipClarification] error:', err))
    }
    const isInputRequest = clarification?.ambiguityReason === 'input_request'
    dispatch({ type: 'SKIP_CLARIFICATION', isInputRequest: isInputRequest ?? false })
  }, [])

  // ---------------------------------------------------------------------------
  // setClarification*
  // ---------------------------------------------------------------------------

  const setClarificationStep = useCallback((step: number) => {
    dispatch({ type: 'SET_CLARIFICATION_STEP', step })
  }, [])

  const setClarificationAnswer = useCallback((step: number, answer: string) => {
    dispatch({ type: 'SET_CLARIFICATION_ANSWER', step, answer })
  }, [])

  const setClarificationStructuredAnswer = useCallback((step: number, data: unknown) => {
    dispatch({ type: 'SET_CLARIFICATION_STRUCTURED_ANSWER', step, data })
  }, [])

  // ---------------------------------------------------------------------------
  // removeQueuedMessage
  // ---------------------------------------------------------------------------

  const removeQueuedMessage = useCallback((id: string) => {
    dispatch({ type: 'REMOVE_QUEUED_MESSAGE', id })
  }, [])

  // ---------------------------------------------------------------------------
  // replanFromStep
  // ---------------------------------------------------------------------------

  const replanFromStep = useCallback(
    (stepNumber: number, mode: 'edit' | 'delete' | 'redo', editedGoal?: string) => {
      // replan_start marks steps >= stepNumber as superseded and sets status: 'executing'
      dispatch({
        type: 'SUBSCRIPTION_EVENT',
        event: {
          event_type: 'replan_start',
          session_id: '',
          step_number: stepNumber,
          timestamp: '',
          data: { from_step: stepNumber },
        },
      })
      const sess = sessionRef.current
      if (sess) {
        apolloClient
          .mutate({
            mutation: REPLAN_FROM,
            variables: { sessionId: sess.session_id, stepNumber, mode, editedGoal },
          })
          .catch((err) => console.error('[replanFromStep] error:', err))
      }
    },
    [],
  )

  // ---------------------------------------------------------------------------
  // editObjective / deleteObjective
  // ---------------------------------------------------------------------------

  const editObjective = useCallback((objectiveIndex: number, newText: string) => {
    dispatch({ type: 'SET_STATUS', status: 'executing' })
    const sess = sessionRef.current
    if (sess) {
      apolloClient
        .mutate({
          mutation: EDIT_OBJECTIVE,
          variables: { sessionId: sess.session_id, objectiveIndex, newText },
        })
        .catch((err) => console.error('[editObjective] error:', err))
    }
  }, [])

  const deleteObjective = useCallback((objectiveIndex: number) => {
    dispatch({ type: 'SET_STATUS', status: 'executing' })
    const sess = sessionRef.current
    if (sess) {
      apolloClient
        .mutate({
          mutation: DELETE_OBJECTIVE,
          variables: { sessionId: sess.session_id, objectiveIndex },
        })
        .catch((err) => console.error('[deleteObjective] error:', err))
    }
  }, [])

  // ---------------------------------------------------------------------------
  // fetchAgents / setAgent
  // ---------------------------------------------------------------------------

  const fetchAgents = useCallback(async () => {
    const sess = sessionRef.current
    if (!sess) return
    try {
      const { getAuthHeaders } = await import('@/config/auth-helpers')
      const headers = await getAuthHeaders()
      const response = await fetch(
        `/api/sessions/agents?session_id=${sess.session_id}`,
        { headers, credentials: 'include' },
      )
      if (response.ok) {
        const data = await response.json()
        setAgents(data.agents || [])
        setCurrentAgent(data.current_agent || null)
      }
    } catch (error) {
      console.error('Failed to fetch agents:', error)
    }
  }, [])

  const setAgent = useCallback(async (agentName: string | null) => {
    const sess = sessionRef.current
    if (!sess) return
    try {
      const { data } = await apolloClient.mutate({
        mutation: ACTIVATE_AGENT,
        variables: { sessionId: sess.session_id, agentName },
      })
      const active = data.activateAgent.currentAgent
      setCurrentAgent(active)
      setAgents((prev) =>
        prev.map((r) => ({ ...r, is_active: r.name === active })),
      )
      apolloClient.refetchQueries({ include: ['PromptContext'] })
    } catch (error) {
      console.error('Failed to set agent:', error)
    }
  }, [])

  // ---------------------------------------------------------------------------
  // Session switching (HamburgerMenu)
  // ---------------------------------------------------------------------------

  const switchSession = useCallback(async (targetSessionId: string) => {
    const currentSessionId = sessionRef.current?.session_id
    if (targetSessionId === currentSessionId) return

    stopSubscription()
    clearArtifactState()

    // Create/restore session + fetch messages in parallel
    const [sessionResult, messagesResult] = await Promise.all([
      apolloClient.mutate({
        mutation: CREATE_SESSION,
        variables: { userId: userIdRef.current, sessionId: targetSessionId },
      }),
      apolloClient.query({ query: MESSAGES_QUERY, variables: { sessionId: targetSessionId }, fetchPolicy: 'network-only' })
        .then(({ data }: { data: { messages: { messages: unknown[] } } }) => ({
          messages: data.messages.messages.map(toStoredMessage),
        }))
        .catch(() => ({ messages: [] as unknown[] })),
    ])

    const newSession = toSession(sessionResult.data.createSession)
    const restoredMessages = (messagesResult.messages || []).map((m: any) => ({
      id: m.id,
      type: m.type as Message['type'],
      content: m.content,
      timestamp: new Date(m.timestamp),
      stepNumber: m.stepNumber,
      isFinalInsight: m.isFinalInsight,
      stepDurationMs: m.stepDurationMs,
      role: m.role,
      skills: m.skills,
    }))

    dispatch({ type: 'RESET', messages: restoredMessages })
    setSessionState(newSession)
    sessionRef.current = newSession
    setSessionReady((newSession.active_domains?.length ?? 0) > 0)

    // Subscribe to new session
    startSubscription(newSession)

    // Update localStorage
    const { storeSessionId } = await import('@/api/session-id')
    storeSessionId(targetSessionId, newSession.user_id)

    // Fetch all session data via Apollo cache
    await apolloClient.refetchQueries({
      include: ['Tables', 'Artifacts', 'Facts', 'Entities', 'DataSources',
                'Steps', 'InferenceCodes', 'Learnings', 'PromptContext', 'ActiveDomains'],
    })
  }, [stopSubscription, startSubscription])

  // ---------------------------------------------------------------------------
  // Cleanup
  // ---------------------------------------------------------------------------

  useEffect(() => {
    return () => {
      stopSubscription()
    }
  }, [stopSubscription])

  // ---------------------------------------------------------------------------
  // Context value
  // ---------------------------------------------------------------------------

  const value = useMemo<SessionContextValue>(
    () => ({
      sessionId: session?.session_id ?? null,
      session,
      activeDomains: session?.active_domains ?? [],
      setActiveDomains,
      updateSession,
      setSession,
      createSession,
      subscriptionConnected,
      sessionReady,
      isCreatingSession,
      status: execState.status,
      executionPhase: execState.executionPhase as ExecutionPhase,
      plan: execState.plan,
      currentQuery: execState.currentQuery,
      messages: execState.messages,
      suggestions: execState.suggestions,
      welcomeTagline: execState.welcomeTagline,
      queuedMessages: execState.queuedMessages,
      clarification: execState.clarification,
      agents,
      currentAgent,
      fetchAgents,
      setAgent,
      submitQuery,
      cancelExecution,
      approvePlan,
      rejectPlan,
      shareSession,
      replanFromStep,
      removeQueuedMessage,
      editObjective,
      deleteObjective,
      proofFacts,
      isProofPanelOpen,
      isPlanningComplete,
      proofSummary,
      isSummaryGenerating,
      isProving,
      hasCompletedProof,
      openProofPanel,
      closeProofPanel,
      clearProofFacts,
      answerClarification,
      skipClarification,
      setClarificationStep,
      setClarificationAnswer,
      setClarificationStructuredAnswer,
      switchSession,
    }),
    [
      session,
      subscriptionConnected,
      sessionReady,
      isCreatingSession,
      execState,
      agents,
      currentAgent,
      proofFacts,
      isProofPanelOpen,
      isPlanningComplete,
      proofSummary,
      isSummaryGenerating,
      isProving,
      hasCompletedProof,
      setActiveDomains,
      updateSession,
      setSession,
      createSession,
      fetchAgents,
      setAgent,
      submitQuery,
      cancelExecution,
      approvePlan,
      rejectPlan,
      shareSession,
      replanFromStep,
      removeQueuedMessage,
      editObjective,
      deleteObjective,
      openProofPanel,
      closeProofPanel,
      clearProofFacts,
      answerClarification,
      skipClarification,
      setClarificationStep,
      setClarificationAnswer,
      setClarificationStructuredAnswer,
      switchSession,
    ],
  )

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
}

export function useSessionContext(): SessionContextValue {
  const ctx = useContext(SessionContext)
  if (!ctx) throw new Error('useSessionContext must be used within SessionProvider')
  return ctx
}
