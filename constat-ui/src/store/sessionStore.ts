// Session state store
console.log('[sessionStore] MODULE LOADED - v2025-02-05-debug')

import { create } from 'zustand'
import type { Session, SessionStatus, Plan, WSEvent, TableInfo, Artifact, Fact, GlossaryTerm } from '@/types/api'
import { wsManager } from '@/api/websocket'
import * as sessionsApi from '@/api/sessions'
import { getOrCreateSessionId, createNewSessionId } from '@/api/sessions'
import * as queriesApi from '@/api/queries'
import { applyPatch, type Operation } from 'fast-json-patch'
import { useArtifactStore } from './artifactStore'
import { useProofStore } from './proofStore'
import { useUIStore } from './uiStore'
import { getCachedEntry, setCachedEntry } from './entityCache'
import { type CompactState, inflateToGlossaryTerms } from './entityCacheKeys'
import { useGlossaryStore } from './glossaryStore'

interface Message {
  id: string
  type: 'user' | 'system' | 'plan' | 'step' | 'output' | 'error' | 'thinking'
  content: string
  timestamp: Date
  stepNumber?: number
  plan?: Plan
  isLive?: boolean // Message that updates in place during execution
  isPending?: boolean // Step that hasn't started yet (shows pending animation)
  defaultExpanded?: boolean // Start expanded (don't collapse)
  isFinalInsight?: boolean // Final insight message with View Result button
  role?: string // Agent used for this step (e.g., "data_analyst")
  skills?: string[] // Skills used for this step
  stepStartedAt?: number // Epoch ms when step started
  stepDurationMs?: number // Final duration from backend
  stepAttempts?: number // Number of attempts (retries)
  isSuperseded?: boolean // Step from a previous run (dimmed in UI)
  stepSourcesRead?: string[]    // SQL source tables parsed from code (e.g., "hr.employees")
  stepTablesCreated?: string[]  // tables created by this step (from step_complete event)
}

// Execution phases for live status updates
type ExecutionPhase =
  | 'idle'
  | 'planning'
  | 'awaiting_approval'
  | 'generating'
  | 'executing'
  | 'retrying'
  | 'summarizing'
  | 'synthesizing'

interface ClarificationQuestion {
  text: string
  suggestions: string[]
  widget?: { type: string; config: Record<string, unknown> }
}

interface ClarificationState {
  needed: boolean
  originalQuestion: string
  ambiguityReason: string
  questions: ClarificationQuestion[]
  currentStep: number
  answers: Record<number, string>
  structuredAnswers: Record<number, unknown>
}

// Parse SQL source table references (schema.table) from code
function parseSourceTables(code: string): string[] {
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

interface QueuedMessage {
  id: string
  content: string
  timestamp: Date
}

interface AgentInfo {
  name: string
  prompt: string
  is_active: boolean
}

interface SessionState {
  // Current session
  session: Session | null
  status: SessionStatus
  wsConnected: boolean

  // Conversation
  messages: Message[]
  currentQuery: string
  thinkingMessageId: string | null

  // Execution tracking
  executionPhase: ExecutionPhase
  liveMessageId: string | null
  stepMessageIds: Record<number, string> // Maps step number to message ID
  currentStepNumber: number
  stepAttempt: number
  lastQueryStartStep: number // First step number of the current/last query (for View Result)
  querySubmittedAt: number | null // Epoch ms when query was submitted

  // Plan
  plan: Plan | null

  // Clarification
  clarification: ClarificationState | null

  // Welcome / suggestions
  welcomeTagline: string
  suggestions: string[]

  // Queued messages (submitted while busy)
  queuedMessages: QueuedMessage[]

  // Agents
  agents: AgentInfo[]
  currentAgent: string | null

  // Dynamic context for current query (agent/skills selected)
  queryContext: {
    agent?: { name: string; similarity: number }
    skills?: { name: string; similarity: number }[]
  } | null

  // Whether the current query is a redo (clears old steps) vs follow-up (keeps old steps)
  isRedo: boolean

  // Session creation state (for disabling input during new query)
  isCreatingSession: boolean

  // Whether background session init is complete (domains loaded, sources available)
  sessionReady: boolean

  // Actions
  createSession: (userId?: string, forceNew?: boolean) => Promise<void>
  setSession: (session: Session | null, options?: { preserveMessages?: boolean }) => void
  updateSession: (updates: Partial<Session>) => void
  submitQuery: (problem: string, isFollowup?: boolean) => Promise<void>
  cancelExecution: () => Promise<void>
  approvePlan: (deletedSteps?: number[], editedSteps?: Array<{ number: number; goal: string }>) => Promise<void>
  rejectPlan: (feedback: string, editedSteps?: Array<{ number: number; goal: string }>) => Promise<void>
  answerClarification: (answers: Record<number, string>, structuredAnswers?: Record<number, unknown>) => void
  skipClarification: () => void
  setClarificationStep: (step: number) => void
  setClarificationAnswer: (step: number, answer: string) => void
  setClarificationStructuredAnswer: (step: number, data: unknown) => void
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void
  updateMessage: (id: string, updates: Partial<Pick<Message, 'type' | 'content'>>) => void
  removeMessage: (id: string) => void
  clearMessages: () => void
  setCurrentQuery: (query: string) => void
  handleWSEvent: (event: WSEvent) => void
  removeQueuedMessage: (id: string) => void
  clearQueue: () => void
  fetchAgents: () => Promise<void>
  setAgent: (agentName: string | null) => Promise<void>
  shareSession: (email: string) => Promise<{ share_url: string }>
  replanFromStep: (stepNumber: number, mode: 'edit' | 'delete' | 'redo', editedGoal?: string) => void
  editObjective: (objectiveIndex: number, newText: string) => void
  deleteObjective: (objectiveIndex: number) => void
}

export const useSessionStore = create<SessionState>((set, get) => ({
  session: null,
  status: 'idle',
  wsConnected: false,
  messages: [],
  currentQuery: '',
  thinkingMessageId: null,
  executionPhase: 'idle',
  liveMessageId: null,
  stepMessageIds: {},
  currentStepNumber: 0,
  stepAttempt: 1,
  plan: null,
  clarification: null,
  welcomeTagline: '',
  suggestions: [],
  queuedMessages: [],
  lastQueryStartStep: 0,
  querySubmittedAt: null,
  agents: [],
  currentAgent: null,
  queryContext: null,
  isRedo: false,
  isCreatingSession: false,
  sessionReady: false,

  createSession: async (userId = 'default', forceNew = false) => {
    // Mark session as being created (disables input)
    set({ isCreatingSession: true })

    // Disconnect old WebSocket FIRST to prevent any events during transition
    wsManager.disconnect()

    // Clear artifact store for fresh session
    useArtifactStore.getState().clear()

    // Get or create session ID (forceNew = true for reset/new session)
    const sessionId = forceNew ? createNewSessionId(userId) : getOrCreateSessionId(userId)

    // Create session on server with client-provided session ID
    const session = await sessionsApi.createSession(userId, sessionId)

    // Try to restore messages, facts, and codes in parallel if reconnecting
    let restoredMessages: Message[] = []
    if (!forceNew) {
      const artifactStore = useArtifactStore.getState()
      const [messagesRes, factsRes, _codesRes, _artifactsRes] = await Promise.allSettled([
        sessionsApi.getMessages(sessionId),
        sessionsApi.getProofFacts(sessionId),
        Promise.all([
          artifactStore.fetchStepCodes(sessionId),
          artifactStore.fetchInferenceCodes(sessionId),
        ]),
        Promise.all([
          artifactStore.fetchTables(sessionId),
          artifactStore.fetchArtifacts(sessionId),
        ]),
      ])

      // Extract restored messages
      if (messagesRes.status === 'fulfilled') {
        const { messages: storedMessages } = messagesRes.value
        if (storedMessages && storedMessages.length > 0) {
          restoredMessages = storedMessages.map(m => ({
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

      // Extract restored proof facts
      if (factsRes.status === 'fulfilled') {
        const { facts: storedFacts, summary } = factsRes.value
        if (storedFacts && storedFacts.length > 0) {
          useProofStore.getState().importFacts(storedFacts, summary)
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

    // Reconnect: active_domains populated → session already initialized
    // New session: active_domains empty → wait for session_ready event
    const isReconnect = (session.active_domains?.length ?? 0) > 0

    // Initialize with restored messages (or empty for new session)
    set({
      session,
      status: 'idle',
      messages: restoredMessages,
      plan: null,
      suggestions: [],
      queuedMessages: [],
      clarification: null,
      executionPhase: 'idle',
      currentStepNumber: 0,
      stepAttempt: 1,
      stepMessageIds: {},
      liveMessageId: null,
      thinkingMessageId: null,
      lastQueryStartStep: 0,
      querySubmittedAt: null,
      queryContext: null,
      isCreatingSession: false,
      sessionReady: isReconnect,
    })

    // Connect WebSocket - server will send welcome message on connect
    wsManager.connect(session.session_id)
    wsManager.onStatus((connected) => set({ wsConnected: connected }))
    wsManager.onEvent((event) => get().handleWSEvent(event))

    // On reconnect, fetch glossary immediately (session already initialized)
    // On new session, session_ready event will trigger the fetch
    if (isReconnect) {
      useGlossaryStore.getState().fetchTerms(session.session_id)
    }
  },

  setSession: (session, options?: { preserveMessages?: boolean }) => {
    if (session) {
      // Clear messages for fresh session (unless preserving for restoration)
      if (!options?.preserveMessages) {
        set({ messages: [], suggestions: [], plan: null })
      }
      useGlossaryStore.getState().loadFromCache(session.session_id)
      wsManager.connect(session.session_id)
      wsManager.onStatus((connected) => set({ wsConnected: connected }))
      wsManager.onEvent((event) => get().handleWSEvent(event))
    } else {
      wsManager.disconnect()
    }
    set({ session, status: session?.status ?? 'idle' })
  },

  updateSession: (updates) => {
    // Update session properties without clearing messages or reconnecting WebSocket
    const { session } = get()
    if (session) {
      set({ session: { ...session, ...updates } })
    }
  },

  submitQuery: async (problem, isFollowup = false) => {
    const { session, addMessage, suggestions, status, executionPhase } = get()
    if (!session) return

    // Expand number shortcuts (e.g., "1" -> first suggestion)
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

    // Detect @username share (any @mention that isn't @vera)
    const shareMatch = !veraMatch && expandedProblem.match(/^@(\S+)\s*(.*)/s)
    if (shareMatch) {
      const targetUser = shareMatch[1]
      addMessage({ type: 'user', content: problem, defaultExpanded: true })
      try {
        const result = await get().shareSession(targetUser)
        addMessage({
          type: 'output',
          content: `**Shared** with \`${targetUser}\`\n\n[Share link](${result.share_url})`,
          isFinalInsight: true,
        })
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Share failed'
        addMessage({ type: 'error', content: msg })
      }
      return
    }

    // Apply brief mode prefix if enabled (server detects "briefly" keyword)
    const { briefMode } = useUIStore.getState()
    if (briefMode && !expandedProblem.startsWith('/') && !/^briefly\b/i.test(expandedProblem)) {
      expandedProblem = `briefly, ${expandedProblem}`
    }

    // Check if session is busy (planning, executing, awaiting approval)
    const isBusy = status === 'planning' || status === 'executing' || status === 'awaiting_approval' ||
      executionPhase !== 'idle'

    if (isBusy) {
      // Queue the message instead of submitting
      const queuedMessage: QueuedMessage = {
        id: crypto.randomUUID(),
        content: expandedProblem,
        timestamp: new Date(),
      }
      set((state) => ({
        queuedMessages: [...state.queuedMessages, queuedMessage],
      }))
      return
    }

    // Add user message (show original @vera text, not expanded /rule)
    addMessage({ type: 'user', content: veraMatch ? problem : expandedProblem, defaultExpanded: true })

    // Add thinking indicator (will become live message on first event)
    const thinkingId = crypto.randomUUID()
    const thinkingMessage: Message = {
      id: thinkingId,
      type: 'thinking',
      content: '',
      timestamp: new Date(),
      isLive: true,
    }
    const isRedo = /^\/redo\b/i.test(expandedProblem.trim()) || !isFollowup
    set((state) => ({
      messages: [...state.messages, thinkingMessage],
      thinkingMessageId: thinkingId,
      liveMessageId: null,
      stepMessageIds: {},
      currentQuery: expandedProblem,
      querySubmittedAt: Date.now(),
      status: 'planning',
      executionPhase: 'idle',
      currentStepNumber: 0,
      stepAttempt: 1,
      isRedo,
      suggestions: [], // Clear suggestions after use
    }))

    const response = await queriesApi.submitQuery(session.session_id, expandedProblem, isFollowup)

    // Slash commands return completed/error immediately via HTTP (no websocket event)
    if ((response.status === 'completed' || response.status === 'error') && response.message) {
      const { thinkingMessageId } = get()
      if (thinkingMessageId) {
        set((state) => ({
          messages: state.messages.filter(m => m.id !== thinkingMessageId),
          thinkingMessageId: null,
          liveMessageId: null,
        }))
      }

      // Format @vera response with rule details
      let content = response.message
      if (veraMatch && response.status === 'completed') {
        const ruleIdMatch = response.message.match(/`(rule_[a-f0-9]+)`/)
        const ruleId = ruleIdMatch?.[1] ?? 'unknown'
        const { useAuthStore } = await import('@/store/authStore')
        const userId = useAuthStore.getState().userId
        content = `**Rule:** \`${ruleId}\`\n**Domain:** ${userId}\n**Summary:** ${veraMatch[1].trim()}\n\n*Tip: You can move this rule into a domain with* \`/move-rule ${ruleId} <domain-name>\``
      }

      addMessage({
        type: response.status === 'error' ? 'error' : 'output',
        content,
        isFinalInsight: response.status === 'completed',
      })
      set({ status: response.status === 'error' ? 'error' : 'completed', currentStepNumber: 0, stepAttempt: 1, executionPhase: 'idle' })
    }
  },

  cancelExecution: async () => {
    const { session } = get()
    if (!session) return

    await queriesApi.cancelExecution(session.session_id)
    // Reset both status and executionPhase so new queries can be submitted immediately
    set({ status: 'cancelled', executionPhase: 'idle', currentStepNumber: 0, stepAttempt: 1 })
  },

  approvePlan: async (deletedSteps?: number[], editedSteps?: Array<{ number: number; goal: string }>) => {
    const { session, plan } = get()
    if (!session) return

    // If editedSteps provided, use those; otherwise use original plan steps (minus deleted)
    let stepsToShow: Array<{ number: number; goal: string; role_id?: string; domain?: string; skill_ids?: string[] }>
    if (editedSteps && editedSteps.length > 0) {
      // Use edited steps - these are already renumbered and filtered
      stepsToShow = editedSteps.map(s => ({ number: s.number, goal: s.goal }))
    } else {
      const allSteps = plan?.steps || []
      // Filter out deleted steps
      const deletedSet = new Set(deletedSteps || [])
      stepsToShow = allSteps
        .filter((step, index) => {
          const stepNum = step.number ?? index + 1
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

    // Create message bubbles for steps (pending until step_start)
    // Skip steps that already have non-superseded messages (prevents duplicates)
    const existingStepNums = new Set(
      get().messages.filter((m) => m.type === 'step' && m.stepNumber !== undefined && !m.isSuperseded).map((m) => m.stepNumber)
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
          isLive: false, // Not live until step starts
          isPending: true, // Pending animation until step starts
          ...(step.role_id ? { role: step.domain ? `${step.domain}/${step.role_id}` : step.role_id } : {}),
          ...(step.skill_ids?.length ? { skills: step.skill_ids } : {}),
        }
      })

    await queriesApi.approvePlan(session.session_id, true, undefined, deletedSteps, editedSteps)
    wsManager.approve()
    // Sort step messages by step number to ensure consistent display order
    stepMessages.sort((a, b) => (a.stepNumber || 0) - (b.stepNumber || 0))
    const { isRedo } = get()
    if (isRedo) {
      // Redo: mark old step codes as superseded (don't clear them)
      useArtifactStore.getState().markStepsSuperseded()
    }
    set((state) => ({
      // Always append. On redo, mark old step messages as superseded.
      messages: [
        ...state.messages.map((m) =>
          isRedo && m.type === 'step' && !m.isSuperseded
            ? { ...m, isSuperseded: true }
            : m
        ),
        ...stepMessages,
      ],
      stepMessageIds: { ...state.stepMessageIds, ...stepMessageIds },
      liveMessageId: null,
      status: 'executing',
      executionPhase: 'executing',
      plan: null,
    }))
  },

  rejectPlan: async (feedback, editedSteps) => {
    const { session, addMessage } = get()
    if (!session) return

    // Clear the plan UI
    set({ plan: null })

    if (feedback && feedback !== 'Cancelled by user') {
      // Add feedback as user message in conversation
      addMessage({ type: 'user', content: feedback })

      // Notify backend of rejection with feedback and edited plan via REST
      // Backend will trigger replanning with original query + edited plan structure
      await queriesApi.approvePlan(session.session_id, false, feedback, undefined, editedSteps)
      // Set status to planning - backend will send new plan_ready event
      set({ status: 'planning', executionPhase: 'planning' })
    } else {
      // Just cancelled - go back to idle
      addMessage({ type: 'system', content: 'Plan cancelled' })
      await queriesApi.approvePlan(session.session_id, false, feedback)
      wsManager.reject(feedback)
      set({ status: 'idle' })
    }
  },

  answerClarification: (answers, structuredAnswers) => {
    const { clarification, currentStepNumber } = get()

    // Merge any structured answers from clarification state
    const finalStructured: Record<string, unknown> = structuredAnswers || clarification?.structuredAnswers || {}

    // Remap numeric keys → question text for backend (plan shows "Q: A" not "0: A")
    const questionTexts = clarification?.questions || []
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

    // Dismiss dialog and send response FIRST — ensures modal closes immediately
    const payload: Record<string, unknown> = { answers: textKeyedAnswers }
    if (Object.keys(textKeyedStructured).length > 0) {
      payload.structured_answers = textKeyedStructured
    }
    wsManager.send('clarify', payload)
    const isInputRequest = clarification?.ambiguityReason === 'input_request'
    set({ clarification: null, status: isInputRequest ? 'executing' : 'planning' })

    // Add user response bubble (right side) — Q&A pairs for context
    if (clarification) {
      const answerSummary = clarification.questions.length > 1
        ? clarification.questions
            .map((q, i) => `**${q.text}**\n${answers[i] || 'Skipped'}`)
            .join('\n\n')
        : answers[0] || 'Skipped'

      const userMsg: Message = {
        id: crypto.randomUUID(),
        type: 'user',
        content: answerSummary,
        timestamp: new Date(),
      }

      // Find the question system message (last system message near the current step)
      // and insert the response after it
      const stepMsgId = currentStepNumber ? get().stepMessageIds[currentStepNumber] : null
      set((state) => {
        if (stepMsgId) {
          const stepIdx = state.messages.findIndex((m) => m.id === stepMsgId)
          if (stepIdx >= 0) {
            // Find the system message after the step (the question)
            let insertIdx = state.messages.length
            for (let i = stepIdx + 1; i < state.messages.length; i++) {
              if (state.messages[i].type === 'system') {
                insertIdx = i + 1
                break
              }
            }
            const updated = [...state.messages]
            updated.splice(insertIdx, 0, userMsg)
            return { messages: updated }
          }
        }
        return { messages: [...state.messages, userMsg] }
      })
    }
  },

  skipClarification: () => {
    const { clarification } = get()
    wsManager.send('skip_clarification')
    const isInputRequest = clarification?.ambiguityReason === 'input_request'
    set({ clarification: null, status: isInputRequest ? 'executing' : 'planning' })
  },

  setClarificationStep: (step) => {
    set((state) => ({
      clarification: state.clarification
        ? { ...state.clarification, currentStep: step }
        : null,
    }))
  },

  setClarificationAnswer: (step, answer) => {
    set((state) => ({
      clarification: state.clarification
        ? {
            ...state.clarification,
            answers: { ...state.clarification.answers, [step]: answer },
          }
        : null,
    }))
  },

  setClarificationStructuredAnswer: (step, data) => {
    set((state) => ({
      clarification: state.clarification
        ? {
            ...state.clarification,
            structuredAnswers: { ...state.clarification.structuredAnswers, [step]: data },
          }
        : null,
    }))
  },

  addMessage: (message) => {
    const newMessage: Message = {
      ...message,
      id: crypto.randomUUID(),
      timestamp: new Date(),
    }
    set((state) => ({ messages: [...state.messages, newMessage] }))
  },

  updateMessage: (id, updates) => {
    set((state) => ({
      messages: state.messages.map((m) =>
        m.id === id ? { ...m, ...updates } : m
      ),
    }))
  },

  removeMessage: (id) => {
    set((state) => ({
      messages: state.messages.filter((m) => m.id !== id),
      thinkingMessageId: state.thinkingMessageId === id ? null : state.thinkingMessageId,
    }))
  },

  clearMessages: () => set({ messages: [], thinkingMessageId: null }),

  setCurrentQuery: (query) => set({ currentQuery: query }),

  removeQueuedMessage: (id) => {
    set((state) => ({
      queuedMessages: state.queuedMessages.filter((m) => m.id !== id),
    }))
  },

  clearQueue: () => set({ queuedMessages: [] }),

  handleWSEvent: (event) => {
    const { addMessage, thinkingMessageId, liveMessageId, stepMessageIds, updateMessage, removeMessage, stepAttempt } = get()

    // Helper to update a step's message bubble
    const updateStepMessage = (stepNumber: number, content: string, isComplete = false) => {
      const messageId = stepMessageIds[stepNumber]
      if (messageId) {
        set((state) => ({
          messages: state.messages.map((m) =>
            m.id === messageId
              // Preserve role/skills when updating content
              ? { ...m, content, isLive: !isComplete, isPending: false, role: m.role, skills: m.skills }
              : m
          ),
        }))
      }
    }

    // Helper to get or create the live message (for planning phase)
    const ensureLiveMessage = (content: string, phase: ExecutionPhase): string => {
      const existingId = liveMessageId || thinkingMessageId
      if (existingId) {
        updateMessage(existingId, { type: 'system', content })
        if (thinkingMessageId && !liveMessageId) {
          set({ thinkingMessageId: null, liveMessageId: existingId, executionPhase: phase })
        } else {
          set({ executionPhase: phase })
        }
        return existingId
      } else {
        // Create new live message
        const newId = crypto.randomUUID()
        const liveMessage: Message = {
          id: newId,
          type: 'system',
          content,
          timestamp: new Date(),
          isLive: true,
        }
        set((state) => ({
          messages: [...state.messages, liveMessage],
          liveMessageId: newId,
          executionPhase: phase,
        }))
        return newId
      }
    }

    // Helper to clear live/thinking message
    const clearLiveMessage = () => {
      if (liveMessageId) {
        removeMessage(liveMessageId)
        set({ liveMessageId: null, executionPhase: 'idle' })
      }
      if (thinkingMessageId) {
        removeMessage(thinkingMessageId)
        set({ thinkingMessageId: null })
      }
    }

    // Helper to finalize all step messages (remove live state)
    const finalizeAllSteps = () => {
      set((state) => ({
        messages: state.messages.map((m) =>
          m.stepNumber !== undefined && stepMessageIds[m.stepNumber]
            ? { ...m, isLive: false }
            : m
        ),
        stepMessageIds: {},
      }))
    }

    switch (event.event_type) {
      case 'heartbeat_ack': {
        const data = event.data as { server_time: string }
        wsManager.setLastHeartbeatTime(data.server_time)
        break
      }

      case 'session_ready': {
        // Background init complete — domains loaded, sources/glossary/agents available
        const { session: s } = get()
        if (s) {
          const sid = s.session_id
          const readyData = event.data as { active_domains?: string[] }
          if (readyData.active_domains) {
            set({ session: { ...s, active_domains: readyData.active_domains }, sessionReady: true })
          } else {
            set({ sessionReady: true })
          }
          // Fetch all domain-dependent data
          useArtifactStore.getState().fetchEntities(sid)
          useArtifactStore.getState().fetchDataSources(sid)
          useArtifactStore.getState().fetchAllSkills()
          useArtifactStore.getState().fetchAllAgents(sid)
          useGlossaryStore.getState().fetchTerms(sid)
        }
        break
      }

      case 'welcome': {
        // Centered greeting is rendered by ConversationPanel when messages is empty.
        const data = event.data as { suggestions: string[]; tagline?: string; reliable_adjective?: string; honest_adjective?: string }
        const tagline = data.reliable_adjective && data.honest_adjective
          ? `I'm **Vera**, your ${data.reliable_adjective} and ${data.honest_adjective} data analyst. _${data.tagline || ''}_`
          : ''
        set({ suggestions: data.suggestions || [], welcomeTagline: tagline })
        break
      }

      case 'planning_start':
        ensureLiveMessage('Planning...', 'planning')
        set({ status: 'planning' })  // Don't clear queryContext - it was set by dynamic_context
        break

      case 'replan_start': {
        const fromStep = (event.data as Record<string, unknown>)?.from_step as number
        // Mark steps >= fromStep as superseded
        set((state) => ({
          messages: state.messages.map((m) =>
            m.stepNumber !== undefined && m.stepNumber >= fromStep
              ? { ...m, isSuperseded: true, isLive: false, isPending: false }
              : m
          ),
          status: 'executing',
        }))
        break
      }

      case 'proof_start':
        ensureLiveMessage('Generating reasoning chain...', 'planning')
        set({ status: 'planning' })
        // Clear previous inference codes (proof is a complete re-run)
        useArtifactStore.getState().clearInferenceCodes()
        // Also forward to proofStore to clear previous facts
        useProofStore.getState().handleFactEvent(event.event_type, event.data as Record<string, unknown>)
        break

      case 'replanning':
        // Plan revision in progress - show feedback acknowledgment
        ensureLiveMessage('Revising plan...', 'planning')
        set({ status: 'planning', executionPhase: 'planning' })
        break

      case 'dynamic_context': {
        // Agent and skills selected for this query
        const agent = event.data.agent as { name: string; similarity: number } | undefined
        const skills = event.data.skills as { name: string; similarity: number }[] | undefined
        set({ queryContext: { agent, skills } })

        // Update thinking message to show context
        const contextParts: string[] = []
        if (agent?.name) {
          contextParts.push(`@${agent.name}`)
        }
        if (skills && skills.length > 0) {
          contextParts.push(...skills.map(s => s.name))
        }

        if (contextParts.length > 0) {
          const { thinkingMessageId: tid, liveMessageId: lid, updateMessage: update } = get()
          const msgId = lid || tid
          if (msgId) {
            update(msgId, { content: `Planning... (${contextParts.join(', ')})` })
          }
        }
        break
      }

      case 'plan_ready': {
        clearLiveMessage()
        const planData = event.data.plan as Plan | undefined
        if (planData) {
          // Approval callback sent plan — show approval dialog
          set({ status: 'awaiting_approval', plan: planData, executionPhase: 'awaiting_approval' })
        } else {
          // Auto-approved or follow-up — create step messages and go straight to executing
          const autoSteps = (event.data.steps as Array<{ number: number; goal: string; depends_on?: number[]; role_id?: string; domain?: string }>) || []
          // Skip steps that already have messages (prevents duplicates on replan/re-emit)
          const existingStepNumbers = new Set(
            get().messages.filter((m) => m.type === 'step' && m.stepNumber !== undefined && !m.isSuperseded).map((m) => m.stepNumber)
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
          set((state) => ({
            messages: [...state.messages, ...autoStepMsgs],
            stepMessageIds: { ...state.stepMessageIds, ...autoStepIds },
            liveMessageId: null,
            status: 'executing',
            executionPhase: 'executing',
            plan: null,
          }))
        }
        break
      }

      case 'plan_updated': {
        // Mid-execution replan (e.g., after user_input step) — update step list
        // without disrupting execution state or showing approval dialog
        const updatedSteps = (event.data.steps as Array<{ number: number; goal: string; depends_on?: number[]; role_id?: string; domain?: string }>) || []
        // Build a fresh copy of stepMessageIds to avoid stale closure issues
        const freshIds = { ...get().stepMessageIds }
        // Remove pending step messages that are no longer in the plan
        const newStepNumbers = new Set(updatedSteps.map((s) => s.number))
        const staleIds = get().messages
          .filter((m) => m.type === 'step' && m.stepNumber && m.isPending && !newStepNumbers.has(m.stepNumber))
          .map((m) => m.id)
        if (staleIds.length > 0) {
          set((state) => ({
            messages: state.messages.filter((m) => !staleIds.includes(m.id)),
          }))
          for (const num of Object.keys(freshIds)) {
            if (staleIds.includes(freshIds[Number(num)])) {
              delete freshIds[Number(num)]
            }
          }
        }
        // Add new pending step messages — re-read messages AFTER stale removal
        const freshMsgs = get().messages
        const newPendingSteps = updatedSteps.filter(
          (s) => !freshIds[s.number] && !freshMsgs.some((m) => m.stepNumber === s.number)
        )
        if (newPendingSteps.length > 0) {
          const newMsgs = newPendingSteps.map((step) => ({
            id: crypto.randomUUID(),
            type: 'step' as const,
            content: `Step ${step.number}: ${step.goal || 'Pending'}`,
            timestamp: new Date(),
            stepNumber: step.number,
            isLive: false,
            isPending: true,
            role: step.domain ? `${step.domain}/${step.role_id}` : step.role_id,
          }))
          for (const msg of newMsgs) {
            freshIds[msg.stepNumber!] = msg.id
          }
          set((state) => ({
            messages: [...state.messages, ...newMsgs],
          }))
        }
        // Sync stepMessageIds back to state
        set({ stepMessageIds: freshIds })
        break
      }

      case 'step_start': {
        const goal = (event.data.goal as string) || 'Processing'
        // Track the starting step of this query for View Result
        const { currentStepNumber: prevStep } = get()
        const isFirstStep = prevStep === 0

        // Agent/skills are shown as badges in the step header (not in content)
        updateStepMessage(event.step_number, `Step ${event.step_number}: ${goal}...`)
        // Set stepStartedAt + domain-qualified role on the step message
        const startMsgId = stepMessageIds[event.step_number]
        if (startMsgId) {
          const agent = event.data.agent as string | undefined
          const stepDomain = event.data.domain as string | undefined
          console.log(`[step_start] step=${event.step_number} agent=${agent} domain=${stepDomain} data_keys=${Object.keys(event.data).join(',')}`)
          const qualifiedRole = agent && stepDomain ? `${stepDomain}/${agent}` : undefined
          set((state) => ({
            messages: state.messages.map((m) =>
              m.id === startMsgId ? { ...m, stepStartedAt: Date.now(), stepAttempts: 0, ...(qualifiedRole ? { role: qualifiedRole } : agent && !m.role ? { role: agent } : {}) } : m
            ),
          }))
        }
        set({
          status: 'executing',
          currentStepNumber: event.step_number,
          stepAttempt: 1,
          ...(isFirstStep ? { lastQueryStartStep: event.step_number } : {}),
        })
        break
      }

      case 'step_generating': {
        const goal = (event.data.goal as string) || ''
        const attempt = stepAttempt > 1 ? ` (attempt ${stepAttempt})` : ''
        const goalPrefix = goal ? `${goal}. ` : ''
        // Keep isPending=true during code generation (planning phase)
        const genMsgId = stepMessageIds[event.step_number]
        if (genMsgId) {
          set((state) => ({
            messages: state.messages.map((m) =>
              m.id === genMsgId
                ? { ...m, content: `Step ${event.step_number}: ${goalPrefix}Planning${attempt}...`, isLive: true, isPending: true }
                : m
            ),
          }))
        }
        set({ executionPhase: 'generating' })
        break
      }

      case 'model_escalation': {
        const fromModel = (event.data.from_model as string) || ''
        const toModel = (event.data.to_model as string) || ''
        const reason = (event.data.reason as string) || ''
        // Show short model names (e.g., "together/meta-llama/..." → "Llama-3.1-8B")
        const shortName = (m: string) => {
          const parts = m.split('/')
          return parts[parts.length - 1].replace('Meta-Llama-', 'Llama-').replace('-Instruct-Turbo', '')
        }
        const reasonShort = reason.length > 60 ? reason.slice(0, 60) + '...' : reason
        updateStepMessage(
          event.step_number,
          `Step ${event.step_number}: ${shortName(fromModel)} → ${shortName(toModel)} (${reasonShort})`
        )
        // Increment stepAttempts so the retry badge shows
        const escMsgId = stepMessageIds[event.step_number]
        if (escMsgId) {
          set((state) => ({
            messages: state.messages.map((m) =>
              m.id === escMsgId ? { ...m, stepAttempts: (m.stepAttempts || 0) + 1 } : m
            ),
          }))
        }
        set({ executionPhase: 'retrying' })
        break
      }

      case 'step_executing': {
        const goal = (event.data.goal as string) || ''
        const code = (event.data.code as string) || ''
        const model = (event.data.model as string) || undefined
        if (code) {
          useArtifactStore.getState().addStepCode(event.step_number, goal, code, model)
        }
        updateStepMessage(event.step_number, `Step ${event.step_number}: Executing${goal ? ` - ${goal}` : ''}...`)
        // Parse SQL sources from code and store on message
        const sources = parseSourceTables(code)
        const execMsgId = stepMessageIds[event.step_number]
        if (sources.length > 0 && execMsgId) {
          set((state) => ({
            messages: state.messages.map((m) =>
              m.id === execMsgId ? { ...m, stepSourcesRead: sources } : m
            ),
          }))
        }
        set({ executionPhase: 'executing' })
        break
      }

      case 'step_error': {
        // Show retry attempt in step message
        const newAttempt = stepAttempt + 1
        updateStepMessage(event.step_number, `Step ${event.step_number}: Retrying (attempt ${newAttempt})...`)
        // Increment stepAttempts on the message
        const errMsgId = stepMessageIds[event.step_number]
        if (errMsgId) {
          set((state) => ({
            messages: state.messages.map((m) =>
              m.id === errMsgId ? { ...m, stepAttempts: (m.stepAttempts || 0) + 1 } : m
            ),
          }))
        }
        set({ stepAttempt: newAttempt, executionPhase: 'retrying' })
        break
      }

      case 'step_failed': {
        // Mark step as failed
        const errorMsg = (event.data.error as string) || 'Failed'
        updateStepMessage(event.step_number, `Step ${event.step_number}: ❌ ${errorMsg}`, true)
        break
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
        // Update step bubble with completion status and output (code goes to Code accordion)
        const summary = result.goal || 'Completed'
        const outputSummary = result.stdout ? `\n\n${result.stdout}` : ''
        updateStepMessage(
          event.step_number,
          `Step ${event.step_number}: ✓ ${summary}${outputSummary}`,
          true
        )
        // Set final duration, attempts, tables created, and sources on the message
        const completeMsgId = stepMessageIds[event.step_number]
        const tablesCreated = result.tables_created || []
        const completeSources = parseSourceTables(result.code || '')
        if (completeMsgId) {
          set((state) => ({
            messages: state.messages.map((m) => {
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
          }))
        }
        // Store code for the Code accordion
        if (result.code) {
          useArtifactStore.getState().addStepCode(event.step_number, result.goal || '', result.code, result.model)
        }
        // Fetch artifacts/facts/tables/learnings after each step completes
        const { session } = get()
        if (session) {
          const artifactStore = useArtifactStore.getState()
          artifactStore.fetchArtifacts(session.session_id)
          artifactStore.fetchFacts(session.session_id)
          artifactStore.fetchTables(session.session_id)
          artifactStore.fetchLearnings()
          artifactStore.fetchScratchpad(session.session_id)
        }
        break
      }

      case 'validation_retry': {
        // Post-validation failed, step will retry
        const validation = (event.data.validation as string) || 'Validation failed'
        updateStepMessage(event.step_number, `Step ${event.step_number}: Retrying (${validation})...`)
        set({ executionPhase: 'retrying' })
        break
      }

      case 'validation_warnings': {
        // Post-validation warnings (non-blocking)
        const warnings = (event.data.warnings as string[]) || []
        if (warnings.length > 0) {
          const warningText = warnings.map(w => `⚠ ${w}`).join('\n')
          const { messages } = get()
          const stepMsgId = get().stepMessageIds[event.step_number]
          if (stepMsgId) {
            const existing = messages.find(m => m.id === stepMsgId)
            if (existing) {
              get().updateMessage(stepMsgId, { content: `${existing.content}\n${warningText}` })
            }
          }
        }
        break
      }

      case 'synthesizing':
      case 'generating_insights': {
        // Show "Generating insights..." with animation
        finalizeAllSteps()
        // Add or update a live thinking message
        const insightMsg = (event.data as { message?: string })?.message || 'Generating insights...'
        const existingThinking = get().messages.find(m => m.type === 'thinking' && m.isLive)
        if (existingThinking) {
          // Update existing thinking message
          set((state) => ({
            messages: state.messages.map((m) =>
              m.id === existingThinking.id ? { ...m, content: insightMsg } : m
            ),
            executionPhase: 'synthesizing',
            thinkingMessageId: existingThinking.id,  // Track it for clearLiveMessage
          }))
        } else {
          // Create new thinking message and track its ID
          const newId = Date.now().toString()
          const thinkingMessage: Message = {
            id: newId,
            type: 'thinking',
            content: insightMsg,
            timestamp: new Date(),
            isLive: true,
          }
          set((state) => ({
            messages: [...state.messages, thinkingMessage],
            executionPhase: 'synthesizing',
            thinkingMessageId: newId,  // Track it for clearLiveMessage
          }))
        }

        // Auto-expand best results in artifact panel before insights are generated
        // and select the best artifact to render its contents
        const { session } = get()
        if (session) {
          console.log('[synthesizing] Starting artifact promotion for session:', session.session_id)
          const artifactStore = useArtifactStore.getState()
          // Refresh data to get latest artifacts and tables
          Promise.all([
            artifactStore.fetchArtifacts(session.session_id),
            artifactStore.fetchTables(session.session_id),
          ]).then(() => {
            // Import uiStore dynamically to avoid circular deps
            import('./uiStore').then(({ useUIStore }) => {
              const { artifacts, tables, selectArtifact } = useArtifactStore.getState()
              console.log('[synthesizing] Fetched - Artifacts:', artifacts.length, 'Tables:', tables.length)
              console.log('[synthesizing] All artifacts:', artifacts.map(a => `${a.name} (type=${a.artifact_type}, is_key_result=${a.is_key_result})`))

              const sectionsToExpand: string[] = []

              // Find all published artifacts (is_key_result = true)
              const publishedArtifacts = artifacts.filter((a) => a.is_key_result)
              // Prefer non-tables, but tables are OK as fallback
              const nonTableArtifacts = publishedArtifacts.filter((a) => a.artifact_type !== 'table')
              const tableArtifacts = publishedArtifacts.filter((a) => a.artifact_type === 'table')
              console.log('[synthesizing] Published:', publishedArtifacts.length, 'NonTable:', nonTableArtifacts.length, 'Table:', tableArtifacts.length)
              console.log('[synthesizing] NonTable artifacts:', nonTableArtifacts.map(a => `${a.name} (${a.artifact_type})`))

              if (publishedArtifacts.length > 0) {
                sectionsToExpand.push('artifacts')
              }
              if (tables.length > 0) {
                sectionsToExpand.push('tables')
              }

              // Expand the sections
              if (sectionsToExpand.length > 0) {
                useUIStore.getState().expandArtifactSections(sectionsToExpand)
              }

              // Select the best published artifact
              // Priority: 1) markdown documents, 2) non-tables, 3) tables
              // Within each category, prefer highest step_number
              const candidates = nonTableArtifacts.length > 0 ? nonTableArtifacts : tableArtifacts
              console.log('[synthesizing v2025-02-05] Candidates before sort:', candidates.map(c => `${c.id}:${c.name}(${c.artifact_type})`))
              if (candidates.length > 0) {
                // Sort candidates: markdown first, then by step_number descending
                const markdownTypes = ['markdown', 'md']
                const sortedCandidates = [...candidates].sort((a, b) => {
                  const aType = a.artifact_type?.toLowerCase() || ''
                  const bType = b.artifact_type?.toLowerCase() || ''
                  const aIsMarkdown = markdownTypes.includes(aType)
                  const bIsMarkdown = markdownTypes.includes(bType)
                  console.log(`[sort] Comparing ${a.id}:${aType}(md=${aIsMarkdown}) vs ${b.id}:${bType}(md=${bIsMarkdown})`)
                  // Markdown comes first
                  if (aIsMarkdown && !bIsMarkdown) return -1
                  if (!aIsMarkdown && bIsMarkdown) return 1
                  // Then by step_number descending
                  return b.step_number - a.step_number
                })

                console.log('[synthesizing v2025-02-05] Candidates after sort:', sortedCandidates.map(c => `${c.id}:${c.name}(${c.artifact_type})`))
                const best = sortedCandidates[0]
                console.log('[synthesizing v2025-02-05] Best artifact:', best.name, best.id, 'type:', best.artifact_type)
                selectArtifact(session.session_id, best.id)
              }
            }).catch(err => {
              console.error('[synthesizing] Error importing uiStore:', err)
            })
          }).catch(err => {
            console.error('[synthesizing] Error fetching artifacts/tables:', err)
          })
        }
        break
      }

      case 'query_complete': {
        // Finalize all step messages
        finalizeAllSteps()
        clearLiveMessage()
        // Extract suggestions for number shortcuts
        const completeSuggestions = (event.data.suggestions as string[]) || []
        set({ status: 'completed', currentStepNumber: 0, stepAttempt: 1, suggestions: completeSuggestions, executionPhase: 'idle', queryContext: null })
        // Compute total elapsed time from query submission to now
        const { querySubmittedAt } = get()
        const totalElapsedMs = querySubmittedAt ? Date.now() - querySubmittedAt : undefined
        // Show synthesized summary (unless brief mode skipped synthesis)
        const { messages: currentMessages } = get()
        const lastStepMsg = [...currentMessages].reverse().find((m) => m.type === 'step')
        const isBrief = event.data.brief as boolean
        const summaryOutput = (event.data.output as string) || ''

        if (lastStepMsg && !isBrief && summaryOutput) {
          // Add synthesis summary as a separate message after the steps
          addMessage({
            type: 'output',
            content: summaryOutput,
            isFinalInsight: true,
            stepDurationMs: totalElapsedMs,
          })
          // Mark last step with total elapsed (for results panel timing)
          set((state) => ({
            messages: state.messages.map((m) =>
              m.id === lastStepMsg.id
                ? { ...m, stepDurationMs: totalElapsedMs ?? m.stepDurationMs }
                : m
            ),
          }))
        } else if (lastStepMsg) {
          // Brief mode — just mark last step as final
          set((state) => ({
            messages: state.messages.map((m) =>
              m.id === lastStepMsg.id
                ? { ...m, isFinalInsight: true, stepDurationMs: totalElapsedMs ?? m.stepDurationMs }
                : m
            ),
          }))
        } else {
          // No step messages (e.g. control intent) — add output bubble
          addMessage({
            type: 'output',
            content: summaryOutput || 'Analysis complete',
            isFinalInsight: true,
            stepDurationMs: totalElapsedMs,
          })
        }
        // Refresh artifact panel with final data (including learnings)
        const { session } = get()
        if (session) {
          const artifactStore = useArtifactStore.getState()
          artifactStore.fetchTables(session.session_id)
          artifactStore.fetchArtifacts(session.session_id)
          artifactStore.fetchFacts(session.session_id)
          artifactStore.fetchInferenceCodes(session.session_id)
          artifactStore.fetchStepCodes(session.session_id)
          artifactStore.fetchLearnings()
        }
        // Process queued messages after a short delay to let UI update
        setTimeout(() => {
          const { queuedMessages, submitQuery: submit } = get()
          if (queuedMessages.length > 0) {
            const nextMessage = queuedMessages[0]
            set((state) => ({
              queuedMessages: state.queuedMessages.slice(1),
            }))
            submit(nextMessage.content, true)
          }
        }, 500)
        break
      }

      case 'query_error': {
        finalizeAllSteps()
        clearLiveMessage()
        const errorMsg = (event.data.error as string) || 'Query failed'
        // Check if this is a plan rejection (user revised/modified the plan)
        const isRejection = errorMsg.toLowerCase().includes('rejected') ||
                           errorMsg.toLowerCase().includes('was rejected')
        // Check if this is a cancellation (user already saw "Plan cancelled" message)
        const isCancellation = errorMsg.toLowerCase().includes('cancel') ||
                              errorMsg.toLowerCase().includes('cancelled')
        set({ status: 'error', currentStepNumber: 0, stepAttempt: 1, executionPhase: 'idle' })
        // Don't show redundant error message for cancellations or rejections
        // (rejections immediately start a new query, so no need to show error)
        if (!isCancellation && !isRejection) {
          addMessage({
            type: 'error',
            content: errorMsg,
          })
        }
        // Process queued messages after error too
        setTimeout(() => {
          const { queuedMessages, submitQuery: submit } = get()
          if (queuedMessages.length > 0) {
            const nextMessage = queuedMessages[0]
            set((state) => ({
              queuedMessages: state.queuedMessages.slice(1),
            }))
            submit(nextMessage.content, true)
          }
        }, 500)
        break
      }

      case 'query_cancelled':
        finalizeAllSteps()
        clearLiveMessage()
        set({ status: 'cancelled', currentStepNumber: 0, stepAttempt: 1, executionPhase: 'idle' })
        addMessage({ type: 'system', content: 'Query cancelled' })
        // Process queued messages after cancellation too
        setTimeout(() => {
          const { queuedMessages, submitQuery: submit } = get()
          if (queuedMessages.length > 0) {
            const nextMessage = queuedMessages[0]
            set((state) => ({
              queuedMessages: state.queuedMessages.slice(1),
            }))
            submit(nextMessage.content, true)
          }
        }, 500)
        break

      case 'clarification_needed': {
        const data = event.data as {
          original_question: string
          ambiguity_reason: string
          questions: Array<{ text: string; suggestions: string[]; widget?: { type: string; config: Record<string, unknown> } }>
        }
        // Replace live/thinking with the question text (shown in expanded detail)
        clearLiveMessage()
        const allQuestions = (data.questions || []).map(q => q.text).filter(Boolean)
        const questionText = allQuestions.length > 1
          ? 'Please clarify:\n' + allQuestions.map((q, i) => `${i + 1}. ${q}`).join('\n')
          : allQuestions.length === 1
            ? (allQuestions[0].match(/^please clarify/i) ? allQuestions[0] : `Please clarify: ${allQuestions[0]}`)
            : 'Please clarify your question.'
        // Insert the question as a system message at the current position (near the active step)
        const { currentStepNumber } = get()
        const stepMsgId = currentStepNumber ? stepMessageIds[currentStepNumber] : null
        if (stepMsgId) {
          // Insert after the current step message
          set((state) => {
            const idx = state.messages.findIndex((m) => m.id === stepMsgId)
            const newMsg: Message = {
              id: crypto.randomUUID(),
              type: 'system',
              content: questionText,
              timestamp: new Date(),
            }
            if (idx >= 0) {
              const updated = [...state.messages]
              updated.splice(idx + 1, 0, newMsg)
              return { messages: updated }
            }
            return { messages: [...state.messages, newMsg] }
          })
        } else {
          addMessage({ type: 'system', content: questionText })
        }
        set({
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
        })
        break
      }

      case 'table_created': {
        // Add table to artifact store for real-time right panel update
        const tableData = event.data as {
          name: string
          row_count?: number
          columns?: string[]
        }
        const table: TableInfo = {
          name: tableData.name,
          row_count: tableData.row_count || 0,
          step_number: event.step_number,
          columns: tableData.columns || [],
        }
        useArtifactStore.getState().addTable(table)
        // Refresh DDL to reflect new table/view
        const sid = get().session?.session_id
        if (sid) useArtifactStore.getState().fetchDDL(sid)
        break
      }

      case 'artifact_created': {
        // Add artifact to artifact store for real-time right panel update
        const artifactData = event.data as Partial<Artifact>
        if (artifactData.id && artifactData.name && artifactData.artifact_type) {
          const artifact: Artifact = {
            id: artifactData.id,
            name: artifactData.name,
            artifact_type: artifactData.artifact_type,
            step_number: event.step_number,
            title: artifactData.title,
            description: artifactData.description,
            mime_type: artifactData.mime_type || 'application/octet-stream',
            is_key_result: artifactData.is_key_result,
          }
          useArtifactStore.getState().addArtifact(artifact)
        }
        break
      }

      case 'steps_truncated': {
        const fromStep = event.step_number
        const tablesDropped = (event.data as { tables_dropped?: string[] })?.tables_dropped
        // Remove artifacts, tables, step codes from step N onwards
        useArtifactStore.getState().truncateFromStep(fromStep, tablesDropped)
        // Mark step messages as superseded
        set((state) => ({
          messages: state.messages.map((m) =>
            m.type === 'step' && (m.stepNumber || 0) >= fromStep
              ? { ...m, isSuperseded: true }
              : m
          ),
        }))
        // Refresh DDL
        const truncSid = get().session?.session_id
        if (truncSid) useArtifactStore.getState().fetchDDL(truncSid)
        break
      }

      case 'facts_extracted': {
        // Add facts to artifact store
        const factsData = event.data as { facts?: Fact[]; fact?: Fact }
        if (factsData.facts) {
          factsData.facts.forEach((fact) => {
            useArtifactStore.getState().addFact(fact)
          })
        }
        if (factsData.fact) {
          useArtifactStore.getState().addFact(factsData.fact)
        }
        break
      }

      // Proof/auditable mode events - forward to proof store
      case 'fact_start':
      case 'fact_planning':
      case 'fact_executing':
      case 'fact_resolved':
      case 'fact_failed':
      case 'dag_execution_start':
        useProofStore.getState().handleFactEvent(event.event_type, event.data as Record<string, unknown>)
        break

      case 'inference_code': {
        useProofStore.getState().handleFactEvent(event.event_type, event.data as Record<string, unknown>)
        // Also add to artifactStore for real-time display in right panel
        const icData = event.data as Record<string, unknown>
        if (icData.inference_id && icData.code) {
          useArtifactStore.getState().addInferenceCode({
            inference_id: icData.inference_id as string,
            name: icData.name as string || '',
            operation: icData.operation as string || '',
            code: icData.code as string,
            attempt: icData.attempt as number,
            model: icData.model as string || undefined,
          })
        }
        break
      }

      case 'proof_summary_ready': {
        const proofStore = useProofStore.getState()
        proofStore.handleFactEvent(event.event_type, event.data as Record<string, unknown>)

        // Save proof facts again with summary included
        const { session } = get()
        if (session) {
          const facts = proofStore.exportFacts()
          const summary = (event.data as Record<string, unknown>).summary as string
          if (facts.length > 0) {
            sessionsApi.saveProofFacts(session.session_id, facts, summary).catch(err => {
              console.error('Failed to save proof summary:', err)
            })
          }
        }
        break
      }

      case 'proof_complete': {
        const proofStore = useProofStore.getState()
        proofStore.handleFactEvent(event.event_type, event.data as Record<string, unknown>)
        // Facts are saved on proof_summary_ready (includes summary).
        // If summary generation fails, proofStore.isSummaryGenerating will remain true
        // and we save as fallback after a timeout.
        const { session: pcSession } = get()
        if (pcSession) {
          setTimeout(() => {
            const ps = useProofStore.getState()
            if (ps.isSummaryGenerating) {
              // Summary didn't arrive — save facts without it
              const facts = ps.exportFacts()
              if (facts.length > 0) {
                sessionsApi.saveProofFacts(pcSession.session_id, facts, null).catch(err => {
                  console.error('Failed to save proof facts (fallback):', err)
                })
              }
              ps.handleFactEvent('proof_summary_ready', { summary: null })
            }
          }, 30000)
        }
        break
      }

      case 'progress':
        // Progress events could update a progress bar if needed
        break

      case 'entity_rebuild_complete': {
        // Entity extraction finished — apply diff to entities
        const { session: s } = get()
        if (s) {
          const data = event.data as { diff?: { added: Array<{name: string, type: string}>, removed: Array<{name: string, type: string}> } }
          if (data.diff) {
            useArtifactStore.getState().patchEntities(data.diff)
          }
          useGlossaryStore.getState().setEntityRebuilding(false)
        }
        break
      }

      case 'entity_rebuild_start':
        useGlossaryStore.getState().setEntityRebuilding(true)
        break

      case 'entity_state': {
        const { session: s } = get()
        if (s) {
          const { state, version } = event.data as { state: CompactState; version: number }
          const { terms, totalDefined, totalSelfDescribing } = inflateToGlossaryTerms(state)
          useGlossaryStore.getState().setTermsFromState(terms, totalDefined, totalSelfDescribing)
          setCachedEntry(s.session_id, state, version)
        }
        break
      }

      case 'entity_patch': {
        const { session: s } = get()
        if (s) {
          const { patch, version } = event.data as { patch: Operation[]; version: number }
          getCachedEntry(s.session_id).then((entry) => {
            const base: CompactState = entry?.state ?? { e: {}, g: {}, r: {}, k: {} }
            const { newDocument } = applyPatch(base, patch, false, false)
            const newState = newDocument as CompactState
            const { terms, totalDefined, totalSelfDescribing } = inflateToGlossaryTerms(newState)
            useGlossaryStore.getState().setTermsFromState(terms, totalDefined, totalSelfDescribing)
            setCachedEntry(s.session_id, newState, version)
          })
        }
        break
      }

      case 'source_ingest_complete': {
        const { session: s } = get()
        if (s) {
          useArtifactStore.getState().fetchDataSources(s.session_id)
        }
        useArtifactStore.getState().setIngestingSource(null)
        useArtifactStore.getState().setIngestProgress(null)
        break
      }

      case 'source_ingest_error': {
        const errData = event.data as { name?: string; error?: string }
        console.error(`[source_ingest_error] ${errData.name}: ${errData.error}`)
        break
      }

      case 'source_ingest_progress': {
        const progressData = event.data as { current: number; total: number }
        useArtifactStore.getState().setIngestProgress({ current: progressData.current, total: progressData.total })
        break
      }

      case 'source_ingest_start': {
        const startData = event.data as { name?: string }
        useArtifactStore.getState().setIngestingSource(startData.name || null)
        useArtifactStore.getState().setIngestProgress(null)
        break
      }

      case 'glossary_terms_added': {
        const termsData = event.data as { terms?: GlossaryTerm[] }
        if (termsData.terms && termsData.terms.length > 0) {
          useGlossaryStore.getState().addTerms(termsData.terms!)
        }
        break
      }

      case 'glossary_rebuild_complete': {
        const gStore = useGlossaryStore.getState()
        gStore.setGenerating(false)
        break
      }

      case 'glossary_rebuild_start':
        useGlossaryStore.getState().setGenerating(true)
        break

      case 'glossary_generation_progress': {
        const { stage, percent } = event.data as { stage: string; percent: number }
        useGlossaryStore.getState().setProgress(stage, percent)
        break
      }

      case 'relationships_extracted': {
        useGlossaryStore.setState((s) => ({ refreshKey: s.refreshKey + 1 }))
        break
      }
    }
  },

  fetchAgents: async () => {
    const { session } = get()
    if (!session) return

    try {
      // Import auth helpers to include Bearer token
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = {}
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) {
          headers['Authorization'] = `Bearer ${token}`
        }
      }

      const response = await fetch(`/api/sessions/agents?session_id=${session.session_id}`, {
        headers,
        credentials: 'include',
      })
      if (response.ok) {
        const data = await response.json()
        set({
          agents: data.agents || [],
          currentAgent: data.current_agent || null,
        })
      }
    } catch (error) {
      console.error('Failed to fetch agents:', error)
    }
  },

  setAgent: async (agentName: string | null) => {
    const { session } = get()
    if (!session) return

    try {
      // Import auth helpers to include Bearer token
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) {
          headers['Authorization'] = `Bearer ${token}`
        }
      }

      const response = await fetch(`/api/sessions/agents/current?session_id=${session.session_id}`, {
        method: 'PUT',
        headers,
        credentials: 'include',
        body: JSON.stringify({ agent_name: agentName }),
      })
      if (response.ok) {
        const data = await response.json()
        set({ currentAgent: data.current_agent })
        // Update agents list to reflect active state
        set((state) => ({
          agents: state.agents.map((r) => ({
            ...r,
            is_active: r.name === data.current_agent,
          })),
        }))
        // Refresh prompt context in artifact store
        useArtifactStore.getState().fetchPromptContext(session.session_id)
      }
    } catch (error) {
      console.error('Failed to set agent:', error)
    }
  },

  shareSession: async (email: string) => {
    const { session } = get()
    if (!session) throw new Error('No active session')
    const result = await sessionsApi.shareSession(session.session_id, email)
    return { share_url: result.share_url }
  },

  replanFromStep: (stepNumber: number, mode: 'edit' | 'delete' | 'redo', editedGoal?: string) => {
    // Mark steps >= stepNumber as superseded
    set((state) => ({
      messages: state.messages.map((m) =>
        m.stepNumber !== undefined && m.stepNumber >= stepNumber
          ? { ...m, isSuperseded: true }
          : m
      ),
      status: 'executing',
    }))
    wsManager.replanFrom(stepNumber, mode, editedGoal)
  },

  editObjective: (objectiveIndex: number, newText: string) => {
    set({ status: 'executing' })
    wsManager.editObjective(objectiveIndex, newText)
  },

  deleteObjective: (objectiveIndex: number) => {
    set({ status: 'executing' })
    wsManager.deleteObjective(objectiveIndex)
  },
}))