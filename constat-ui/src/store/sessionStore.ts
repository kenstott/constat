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
  currentStepNumber: number
  stepAttempt: number

  // Plan
  plan: Plan | null

  // Clarification
  clarification: ClarificationState | null

  // Actions
  createSession: (userId?: string) => Promise<void>
  setSession: (session: Session | null) => void
  submitQuery: (problem: string, isFollowup?: boolean) => Promise<void>
  cancelExecution: () => Promise<void>
  approvePlan: () => Promise<void>
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
  currentStepNumber: 0,
  stepAttempt: 1,
  plan: null,
  clarification: null,

  createSession: async (userId = 'default') => {
    const session = await sessionsApi.createSession(userId)
    set({ session, status: 'idle', messages: [], plan: null })

    // Connect WebSocket
    wsManager.connect(session.session_id)
    wsManager.onStatus((connected) => set({ wsConnected: connected }))
    wsManager.onEvent((event) => get().handleWSEvent(event))
  },

  setSession: (session) => {
    if (session) {
      wsManager.connect(session.session_id)
      wsManager.onStatus((connected) => set({ wsConnected: connected }))
      wsManager.onEvent((event) => get().handleWSEvent(event))
    } else {
      wsManager.disconnect()
    }
    set({ session, status: session?.status ?? 'idle' })
  },

  submitQuery: async (problem, isFollowup = false) => {
    const { session, addMessage } = get()
    if (!session) return

    // Add user message
    addMessage({ type: 'user', content: problem })

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
      currentQuery: problem,
      status: 'planning',
      executionPhase: 'idle',
      currentStepNumber: 0,
      stepAttempt: 1,
    }))

    await queriesApi.submitQuery(session.session_id, problem, isFollowup)
  },

  cancelExecution: async () => {
    const { session } = get()
    if (!session) return

    await queriesApi.cancelExecution(session.session_id)
    set({ status: 'cancelled' })
  },

  approvePlan: async () => {
    const { session, plan, addMessage } = get()
    if (!session) return

    // Add a summary message about the approved plan
    if (plan) {
      const steps = plan.steps || []
      addMessage({
        type: 'system',
        content: `Plan approved: ${steps.length} step${steps.length !== 1 ? 's' : ''} to execute`,
      })
    }

    // Create a live message for execution progress
    const liveId = crypto.randomUUID()
    const liveMessage: Message = {
      id: liveId,
      type: 'system',
      content: 'Starting execution...',
      timestamp: new Date(),
      isLive: true,
    }

    await queriesApi.approvePlan(session.session_id, true)
    wsManager.approve()
    set((state) => ({
      messages: [...state.messages, liveMessage],
      liveMessageId: liveId,
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

  handleWSEvent: (event) => {
    const { addMessage, thinkingMessageId, liveMessageId, updateMessage, removeMessage, stepAttempt } = get()

    // Helper to get or create the live message
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

    switch (event.event_type) {
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
        ensureLiveMessage(`Step ${event.step_number}: ${goal}...`, 'generating')
        set({ status: 'executing', currentStepNumber: event.step_number, stepAttempt: 1 })
        break
      }

      case 'step_generating': {
        const attempt = stepAttempt > 1 ? ` (attempt ${stepAttempt})` : ''
        ensureLiveMessage(`Step ${event.step_number}: Generating code${attempt}...`, 'generating')
        break
      }

      case 'step_executing': {
        ensureLiveMessage(`Step ${event.step_number}: Executing...`, 'executing')
        break
      }

      case 'step_error': {
        // Show retry attempt in live message
        const newAttempt = stepAttempt + 1
        ensureLiveMessage(`Step ${event.step_number}: Retrying (attempt ${newAttempt})...`, 'retrying')
        set({ stepAttempt: newAttempt })
        break
      }

      case 'step_failed':
        // Step fully failed - add error bubble and update live message
        addMessage({
          type: 'error',
          content: `Step ${event.step_number} failed: ${event.data.error || 'Unknown error'}`,
          stepNumber: event.step_number,
        })
        break

      case 'step_complete': {
        const result = event.data as { success?: boolean; stdout?: string; goal?: string }
        // Add step completion bubble
        addMessage({
          type: 'step',
          content: `Step ${event.step_number}: ${result.goal || 'Completed'}`,
          stepNumber: event.step_number,
        })
        // Add output if there is any
        if (result.stdout) {
          addMessage({
            type: 'output',
            content: result.stdout,
            stepNumber: event.step_number,
          })
        }
        // Update live message to show we're moving on
        ensureLiveMessage('Continuing...', 'executing')
        break
      }

      case 'query_complete': {
        // Remove live message
        clearLiveMessage()
        set({ status: 'completed', currentStepNumber: 0, stepAttempt: 1 })
        // Add final insights bubble
        const output = (event.data.output as string) || 'Analysis complete'
        addMessage({
          type: 'output',
          content: output,
        })
        break
      }

      case 'query_error':
        clearLiveMessage()
        set({ status: 'error', currentStepNumber: 0, stepAttempt: 1 })
        addMessage({
          type: 'error',
          content: (event.data.error as string) || 'Query failed',
        })
        break

      case 'query_cancelled':
        clearLiveMessage()
        set({ status: 'cancelled', currentStepNumber: 0, stepAttempt: 1 })
        addMessage({ type: 'system', content: 'Execution cancelled' })
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