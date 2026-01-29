// Session state store

import { create } from 'zustand'
import type { Session, SessionStatus, Plan, WSEvent, TableInfo, Artifact, Fact } from '@/types/api'
import { wsManager } from '@/api/websocket'
import * as sessionsApi from '@/api/sessions'
import * as queriesApi from '@/api/queries'
import { useArtifactStore } from './artifactStore'

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
}

interface ClarificationState {
  needed: boolean
  originalQuestion: string
  ambiguityReason: string
  questions: ClarificationQuestion[]
  currentStep: number
  answers: Record<number, string>
}

interface QueuedMessage {
  id: string
  content: string
  timestamp: Date
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

  // Plan
  plan: Plan | null

  // Clarification
  clarification: ClarificationState | null

  // Suggestions (for number shortcuts)
  suggestions: string[]

  // Queued messages (submitted while busy)
  queuedMessages: QueuedMessage[]

  // Actions
  createSession: (userId?: string) => Promise<void>
  setSession: (session: Session | null) => void
  updateSession: (updates: Partial<Session>) => void
  submitQuery: (problem: string, isFollowup?: boolean) => Promise<void>
  cancelExecution: () => Promise<void>
  approvePlan: (deletedSteps?: number[]) => Promise<void>
  rejectPlan: (feedback: string) => Promise<void>
  answerClarification: (answers: Record<number, string>) => void
  skipClarification: () => void
  setClarificationStep: (step: number) => void
  setClarificationAnswer: (step: number, answer: string) => void
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void
  updateMessage: (id: string, updates: Partial<Pick<Message, 'type' | 'content'>>) => void
  removeMessage: (id: string) => void
  clearMessages: () => void
  setCurrentQuery: (query: string) => void
  handleWSEvent: (event: WSEvent) => void
  removeQueuedMessage: (id: string) => void
  clearQueue: () => void
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
  suggestions: [],
  queuedMessages: [],
  lastQueryStartStep: 0,

  createSession: async (userId = 'default') => {
    const session = await sessionsApi.createSession(userId)

    // Clear artifact store for fresh session
    useArtifactStore.getState().clear()

    // Initialize empty - welcome message will come from server via WebSocket
    set({ session, status: 'idle', messages: [], plan: null, suggestions: [] })

    // Connect WebSocket - server will send welcome message on connect
    wsManager.connect(session.session_id)
    wsManager.onStatus((connected) => set({ wsConnected: connected }))
    wsManager.onEvent((event) => get().handleWSEvent(event))
  },

  setSession: (session) => {
    if (session) {
      // Clear messages for fresh session, welcome will come from server
      set({ messages: [], suggestions: [], plan: null })
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

    // Add user message (show expanded form)
    addMessage({ type: 'user', content: expandedProblem })

    // Add thinking indicator (will become live message on first event)
    const thinkingId = crypto.randomUUID()
    const thinkingMessage: Message = {
      id: thinkingId,
      type: 'thinking',
      content: '',
      timestamp: new Date(),
      isLive: true,
    }
    set((state) => ({
      messages: [...state.messages, thinkingMessage],
      thinkingMessageId: thinkingId,
      liveMessageId: null,
      stepMessageIds: {},
      currentQuery: expandedProblem,
      status: 'planning',
      executionPhase: 'idle',
      currentStepNumber: 0,
      stepAttempt: 1,
      suggestions: [], // Clear suggestions after use
    }))

    await queriesApi.submitQuery(session.session_id, expandedProblem, isFollowup)
  },

  cancelExecution: async () => {
    const { session } = get()
    if (!session) return

    await queriesApi.cancelExecution(session.session_id)
    set({ status: 'cancelled' })
  },

  approvePlan: async (deletedSteps?: number[]) => {
    const { session, plan } = get()
    if (!session) return

    const allSteps = plan?.steps || []
    // Filter out deleted steps
    const deletedSet = new Set(deletedSteps || [])
    const steps = allSteps.filter((step, index) => {
      const stepNum = step.number ?? index + 1
      return !deletedSet.has(stepNum)
    })

    // Create message bubbles for remaining steps (pending until step_start)
    const stepMessageIds: Record<number, string> = {}
    const stepMessages: Message[] = steps.map((step) => {
      const id = crypto.randomUUID()
      const stepNum = step.number ?? allSteps.indexOf(step) + 1
      stepMessageIds[stepNum] = id
      return {
        id,
        type: 'step' as const,
        content: `Step ${stepNum}: ${step.goal || 'Pending'}`,
        timestamp: new Date(),
        stepNumber: stepNum,
        isLive: false, // Not live until step starts
        isPending: true, // Pending animation until step starts
      }
    })

    await queriesApi.approvePlan(session.session_id, true, undefined, deletedSteps)
    wsManager.approve()
    set((state) => ({
      messages: [...state.messages, ...stepMessages],
      stepMessageIds,
      liveMessageId: null,
      status: 'executing',
      executionPhase: 'executing',
      plan: null,
    }))
  },

  rejectPlan: async (feedback) => {
    const { session, addMessage } = get()
    if (!session) return

    // Add feedback as a user message
    if (feedback && feedback !== 'Cancelled by user') {
      addMessage({ type: 'user', content: feedback })
    } else {
      addMessage({ type: 'system', content: 'Plan cancelled' })
    }

    await queriesApi.approvePlan(session.session_id, false, feedback)
    wsManager.reject(feedback)
    set({ status: feedback === 'Cancelled by user' ? 'idle' : 'planning', plan: null })
  },

  answerClarification: (answers) => {
    const { clarification, addMessage } = get()

    // Add user bubble with clarification answers
    if (clarification) {
      const answerSummary = clarification.questions
        .map((q, i) => `**${q.text}**\n${answers[i] || 'Skipped'}`)
        .join('\n\n')
      addMessage({ type: 'user', content: answerSummary })
    }

    wsManager.send('clarify', { answers })
    set({ clarification: null, status: 'planning' })
  },

  skipClarification: () => {
    wsManager.send('skip_clarification')
    set({ clarification: null, status: 'planning' })
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
              ? { ...m, content, isLive: !isComplete, isPending: false }
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
      case 'welcome': {
        // Welcome message from server (unified with REPL)
        // Only add if no messages exist yet (prevents duplicate on reconnect)
        const { messages } = get()
        if (messages.length > 0) {
          // Already have messages, just update suggestions
          const data = event.data as { suggestions: string[] }
          set({ suggestions: data.suggestions || [] })
          break
        }
        const data = event.data as {
          message_markdown: string
          suggestions: string[]
        }
        const welcomeMessage: Message = {
          id: crypto.randomUUID(),
          type: 'system',
          content: data.message_markdown,
          timestamp: new Date(),
          defaultExpanded: true,
        }
        set((state) => ({
          messages: [...state.messages, welcomeMessage],
          suggestions: data.suggestions || [],
        }))
        break
      }

      case 'planning_start':
        ensureLiveMessage('Planning...', 'planning')
        set({ status: 'planning' })
        break

      case 'plan_ready':
        // Clear live message, show plan approval dialog
        clearLiveMessage()
        set({ status: 'awaiting_approval', plan: event.data.plan as Plan, executionPhase: 'awaiting_approval' })
        break

      case 'step_start': {
        const goal = (event.data.goal as string) || 'Processing'
        updateStepMessage(event.step_number, `Step ${event.step_number}: ${goal}...`)
        // Track the starting step of this query for View Result
        const { currentStepNumber: prevStep } = get()
        const isFirstStep = prevStep === 0
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
        updateStepMessage(event.step_number, `Step ${event.step_number}: ${goalPrefix}Planning${attempt}...`)
        set({ executionPhase: 'generating' })
        break
      }

      case 'step_executing': {
        const goal = (event.data.goal as string) || ''
        updateStepMessage(event.step_number, `Step ${event.step_number}: Executing${goal ? ` - ${goal}` : ''}...`)
        set({ executionPhase: 'executing' })
        break
      }

      case 'step_error': {
        // Show retry attempt in step message
        const newAttempt = stepAttempt + 1
        updateStepMessage(event.step_number, `Step ${event.step_number}: Retrying (attempt ${newAttempt})...`)
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
        }
        // Update step bubble with completion status and output (code goes to Code accordion)
        const summary = result.goal || 'Completed'
        const outputSummary = result.stdout ? `\n\n${result.stdout}` : ''
        updateStepMessage(
          event.step_number,
          `Step ${event.step_number}: ✓ ${summary}${outputSummary}`,
          true
        )
        // Store code for the Code accordion
        if (result.code) {
          useArtifactStore.getState().addStepCode(event.step_number, result.goal || '', result.code)
        }
        // Fetch artifacts/facts/tables/learnings after each step completes
        const { session } = get()
        if (session) {
          const artifactStore = useArtifactStore.getState()
          artifactStore.fetchArtifacts(session.session_id)
          artifactStore.fetchFacts(session.session_id)
          artifactStore.fetchTables(session.session_id)
          artifactStore.fetchLearnings()
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

              // Visualization types for identifying key results
              const visualizationTypes = ['chart', 'plotly', 'svg', 'png', 'jpeg', 'html', 'image', 'vega', 'markdown', 'md', 'table']

              // Find visualizations
              const visualizations = artifacts.filter((a) =>
                visualizationTypes.includes(a.artifact_type.toLowerCase())
              )
              console.log('[synthesizing] Visualizations found:', visualizations.length)
              if (visualizations.length > 0) {
                sectionsToExpand.push('visualizations')
              }

              // Find key artifacts (marked as key results)
              const keyArtifacts = artifacts.filter((a) => a.is_key_result)
              console.log('[synthesizing] Key artifacts found:', keyArtifacts.length)
              if (keyArtifacts.length > 0) {
                sectionsToExpand.push('artifacts')
              }

              // Always expand tables if we have any
              if (tables.length > 0) {
                sectionsToExpand.push('tables')
              }

              // Expand the sections
              console.log('[synthesizing] Expanding sections:', sectionsToExpand)
              if (sectionsToExpand.length > 0) {
                useUIStore.getState().expandArtifactSections(sectionsToExpand)
              }

              // Select the best artifact to render it:
              // Priority: 1) Latest visualization, 2) Latest key result artifact, 3) None
              const bestVisualization = visualizations.length > 0
                ? visualizations.reduce((best, curr) =>
                    (curr.step_number > best.step_number) ? curr : best
                  )
                : null

              const bestKeyResult = keyArtifacts.length > 0
                ? keyArtifacts.reduce((best, curr) =>
                    (curr.step_number > best.step_number) ? curr : best
                  )
                : null

              // Select visualization first (more visual impact), otherwise key result
              const bestArtifact = bestVisualization || bestKeyResult
              console.log('[synthesizing] Best artifact to open:', bestArtifact ? `${bestArtifact.name} (id=${bestArtifact.id})` : 'none')
              if (bestArtifact) {
                console.log('[synthesizing] Calling selectArtifact for:', bestArtifact.id)
                selectArtifact(session.session_id, bestArtifact.id)
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
        set({ status: 'completed', currentStepNumber: 0, stepAttempt: 1, suggestions: completeSuggestions, executionPhase: 'idle' })
        // Add final insights bubble
        const output = (event.data.output as string) || 'Analysis complete'
        addMessage({
          type: 'output',
          content: output,
          isFinalInsight: true,
        })
        // Refresh artifact panel with final data (including learnings)
        const { session } = get()
        if (session) {
          const artifactStore = useArtifactStore.getState()
          artifactStore.fetchTables(session.session_id)
          artifactStore.fetchArtifacts(session.session_id)
          artifactStore.fetchFacts(session.session_id)
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

      case 'query_error':
        finalizeAllSteps()
        clearLiveMessage()
        set({ status: 'error', currentStepNumber: 0, stepAttempt: 1, executionPhase: 'idle' })
        addMessage({
          type: 'error',
          content: (event.data.error as string) || 'Query failed',
        })
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

      case 'query_cancelled':
        finalizeAllSteps()
        clearLiveMessage()
        set({ status: 'cancelled', currentStepNumber: 0, stepAttempt: 1, executionPhase: 'idle' })
        addMessage({ type: 'system', content: 'Execution cancelled' })
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
          questions: Array<{ text: string; suggestions: string[] }>
        }
        // Replace live/thinking with clarification prompt
        clearLiveMessage()
        addMessage({ type: 'system', content: 'Please clarify your question.' })
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

      case 'facts_extracted':
      case 'fact_resolved': {
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

      case 'progress':
        // Progress events could update a progress bar if needed
        break
    }
  },
}))