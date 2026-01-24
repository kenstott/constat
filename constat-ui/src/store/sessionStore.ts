// Session state store

import { create } from 'zustand'
import type { Session, SessionStatus, Plan, WSEvent } from '@/types/api'
import { wsManager } from '@/api/websocket'
import * as sessionsApi from '@/api/sessions'
import * as queriesApi from '@/api/queries'

interface Message {
  id: string
  type: 'user' | 'system' | 'plan' | 'step' | 'output' | 'error' | 'thinking'
  content: string
  timestamp: Date
  stepNumber?: number
  plan?: Plan
}

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

    // Add thinking indicator
    const thinkingId = crypto.randomUUID()
    const thinkingMessage: Message = {
      id: thinkingId,
      type: 'thinking',
      content: '',
      timestamp: new Date(),
    }
    set((state) => ({
      messages: [...state.messages, thinkingMessage],
      thinkingMessageId: thinkingId,
      currentQuery: problem,
      status: 'planning',
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

    await queriesApi.approvePlan(session.session_id, true)
    wsManager.approve()
    set({ status: 'executing' })
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
    const { addMessage, thinkingMessageId, updateMessage, removeMessage } = get()

    // Helper to remove thinking indicator
    const clearThinking = () => {
      if (thinkingMessageId) {
        removeMessage(thinkingMessageId)
      }
    }

    switch (event.event_type) {
      case 'planning_start':
        // Replace thinking with planning message
        if (thinkingMessageId) {
          updateMessage(thinkingMessageId, { type: 'system', content: 'Planning...' })
          set({ thinkingMessageId: null, status: 'planning' })
        } else {
          set({ status: 'planning' })
          addMessage({ type: 'system', content: 'Planning...' })
        }
        break

      case 'plan_ready':
        // Plan is shown in PlanApprovalDialog, not as inline message
        clearThinking()
        set({ status: 'awaiting_approval', plan: event.data.plan as Plan })
        break

      case 'step_start':
        set({ status: 'executing' })
        addMessage({
          type: 'step',
          content: `Step ${event.step_number}: ${event.data.goal || 'Starting...'}`,
          stepNumber: event.step_number,
        })
        break

      case 'step_complete': {
        const result = event.data as { success?: boolean; stdout?: string }
        if (result.stdout) {
          addMessage({
            type: 'output',
            content: result.stdout,
            stepNumber: event.step_number,
          })
        }
        break
      }

      case 'step_error':
      case 'step_failed':
        addMessage({
          type: 'error',
          content: `Step ${event.step_number} failed: ${event.data.error || 'Unknown error'}`,
          stepNumber: event.step_number,
        })
        break

      case 'query_complete':
        clearThinking()
        set({ status: 'completed' })
        addMessage({
          type: 'output',
          content: (event.data.output as string) || 'Query completed',
        })
        break

      case 'query_error':
        clearThinking()
        set({ status: 'error' })
        addMessage({
          type: 'error',
          content: (event.data.error as string) || 'Query failed',
        })
        break

      case 'query_cancelled':
        set({ status: 'cancelled' })
        addMessage({ type: 'system', content: 'Execution cancelled' })
        break

      case 'clarification_needed': {
        const data = event.data as {
          original_question: string
          ambiguity_reason: string
          questions: Array<{ text: string; suggestions: string[] }>
        }
        // Replace thinking with clarification prompt
        if (thinkingMessageId) {
          updateMessage(thinkingMessageId, {
            type: 'system',
            content: 'Please clarify your question.',
          })
          set({ thinkingMessageId: null })
        } else {
          addMessage({ type: 'system', content: 'Please clarify your question.' })
        }
        set({
          status: 'awaiting_approval',
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

      case 'progress':
        // Progress events are handled by status animations, not messages
        break
    }
  },
}))