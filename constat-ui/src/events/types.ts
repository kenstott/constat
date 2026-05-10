// Copyright (c) 2025 Kenneth Stott
// Canary: 92354268-b283-43e4-b5d6-92742af9399e
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// Session execution state types — used by the event reducer and SessionProvider

import type { SessionStatus, Plan, SubscriptionEvent } from '@/types/api'

export interface Message {
  id: string
  type: 'user' | 'system' | 'plan' | 'step' | 'output' | 'error' | 'thinking'
  content: string
  timestamp: Date
  stepNumber?: number
  plan?: Plan
  isLive?: boolean
  isPending?: boolean
  defaultExpanded?: boolean
  isFinalInsight?: boolean
  role?: string
  skills?: string[]
  stepStartedAt?: number
  stepDurationMs?: number
  stepAttempts?: number
  isSuperseded?: boolean
  stepSourcesRead?: string[]
  stepTablesCreated?: string[]
}

export type ExecutionPhase =
  | 'idle'
  | 'planning'
  | 'awaiting_approval'
  | 'generating'
  | 'executing'
  | 'retrying'
  | 'summarizing'
  | 'synthesizing'

export interface ClarificationQuestion {
  text: string
  suggestions: string[]
  widget?: { type: string; config: Record<string, unknown> }
}

export interface ClarificationState {
  needed: boolean
  originalQuestion: string
  ambiguityReason: string
  questions: ClarificationQuestion[]
  currentStep: number
  answers: Record<number, string>
  structuredAnswers: Record<number, unknown>
}

export interface QueuedMessage {
  id: string
  content: string
  timestamp: Date
}

/** The execution state managed by useReducer in SessionProvider */
export interface SessionExecutionState {
  status: SessionStatus
  executionPhase: ExecutionPhase
  messages: Message[]
  thinkingMessageId: string | null
  liveMessageId: string | null
  stepMessageIds: Record<number, string>
  currentStepNumber: number
  stepAttempt: number
  lastQueryStartStep: number
  querySubmittedAt: number | null
  plan: Plan | null
  clarification: ClarificationState | null
  suggestions: string[]
  welcomeTagline: string
  queuedMessages: QueuedMessage[]
  queryContext: {
    agent?: { name: string; similarity: number }
    skills?: { name: string; similarity: number }[]
  } | null
  currentQuery: string
  isRedo: boolean
}

export const initialExecutionState: SessionExecutionState = {
  status: 'idle',
  executionPhase: 'idle',
  messages: [],
  thinkingMessageId: null,
  liveMessageId: null,
  stepMessageIds: {},
  currentStepNumber: 0,
  stepAttempt: 1,
  lastQueryStartStep: 0,
  querySubmittedAt: null,
  plan: null,
  clarification: null,
  suggestions: [],
  welcomeTagline: '',
  queuedMessages: [],
  queryContext: null,
  currentQuery: '',
  isRedo: false,
}

/** Actions dispatched to the session event reducer */
export type SessionAction =
  | { type: 'SUBSCRIPTION_EVENT'; event: SubscriptionEvent }
  | { type: 'SUBMIT_QUERY'; query: string; thinkingId: string; isRedo: boolean }
  | { type: 'CANCEL_EXECUTION' }
  | { type: 'APPROVE_PLAN'; stepMessages: Message[]; stepMessageIds: Record<number, string>; isRedo: boolean }
  | { type: 'REJECT_PLAN'; hasFeedback: boolean }
  | { type: 'ANSWER_CLARIFICATION'; isInputRequest: boolean; userMessage: Message; stepMsgId: string | null }
  | { type: 'SKIP_CLARIFICATION'; isInputRequest: boolean }
  | { type: 'SET_CLARIFICATION_STEP'; step: number }
  | { type: 'SET_CLARIFICATION_ANSWER'; step: number; answer: string }
  | { type: 'SET_CLARIFICATION_STRUCTURED_ANSWER'; step: number; data: unknown }
  | { type: 'ADD_MESSAGE'; message: Message }
  | { type: 'UPDATE_MESSAGE'; id: string; updates: Partial<Pick<Message, 'type' | 'content'>> }
  | { type: 'REMOVE_MESSAGE'; id: string }
  | { type: 'CLEAR_MESSAGES' }
  | { type: 'SET_MESSAGES'; messages: Message[]; suggestions?: string[]; plan?: Plan | null }
  | { type: 'SET_CURRENT_QUERY'; query: string }
  | { type: 'SET_STATUS'; status: SessionStatus }
  | { type: 'ADD_QUEUED_MESSAGE'; message: QueuedMessage }
  | { type: 'REMOVE_QUEUED_MESSAGE'; id: string }
  | { type: 'CLEAR_QUEUE' }
  | { type: 'RESET'; messages?: Message[] }

/** Kept for API compatibility — all side-effect stores migrated to Apollo reactive vars */
// eslint-disable-next-line @typescript-eslint/no-empty-object-type
export interface SideEffectStores {}
