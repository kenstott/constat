// Copyright (c) 2025 Kenneth Stott
// Canary: test-coverage-sessionEventHandler
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.

import { describe, it, expect, vi, beforeEach } from 'vitest'

vi.mock('@/graphql/client', () => ({
  apolloClient: {
    query: vi.fn().mockResolvedValue({ data: {} }),
    mutate: vi.fn().mockResolvedValue({ data: {} }),
    refetchQueries: vi.fn().mockResolvedValue([]),
    readQuery: vi.fn().mockReturnValue(null),
    writeQuery: vi.fn(),
  },
}))

vi.mock('@/graphql/operations/state', () => ({
  SAVE_PROOF_FACTS: 'SAVE_PROOF_FACTS',
}))

vi.mock('@/graphql/operations/data', () => ({
  ARTIFACTS_QUERY: 'ARTIFACTS_QUERY',
  TABLES_QUERY: 'TABLES_QUERY',
  FACTS_QUERY: 'FACTS_QUERY',
  ARTIFACT_QUERY: 'ARTIFACT_QUERY',
  toArtifact: (a: Record<string, unknown>) => a,
  toArtifactContent: (c: Record<string, unknown>) => c,
  toTableInfo: (t: Record<string, unknown>) => t,
}))

vi.mock('@/graphql/ui-state', () => ({
  addStepCode: vi.fn(),
  addInferenceCode: vi.fn(),
  clearInferenceCodes: vi.fn(),
  truncateFromStep: vi.fn(),
  ingestingSourceVar: vi.fn(),
  ingestProgressVar: vi.fn(),
  selectedArtifactVar: vi.fn(),
  expandSections: vi.fn(),
  handleFactEvent: vi.fn(),
  exportFacts: vi.fn().mockReturnValue([]),
  isSummaryGeneratingVar: vi.fn().mockReturnValue(false),
}))

vi.mock('@/store/glossaryState', () => ({
  fetchTerms: vi.fn(),
  setTermsFromState: vi.fn(),
  addTerms: vi.fn(),
  setEntityRebuilding: vi.fn(),
  setGenerating: vi.fn(),
  setProgress: vi.fn(),
  bumpRefreshKey: vi.fn(),
}))

vi.mock('@/config/auth-helpers', () => ({
  isAuthDisabled: true,
  getAuthHeaders: vi.fn().mockResolvedValue({}),
  getToken: vi.fn().mockResolvedValue(null),
}))

vi.mock('@/store/entityCache', () => ({
  getCachedEntry: vi.fn().mockResolvedValue(null),
  setCachedEntry: vi.fn(),
}))

vi.mock('@/store/entityCacheKeys', () => ({
  inflateToGlossaryTerms: vi.fn().mockReturnValue({ terms: [], totalDefined: 0, totalSelfDescribing: 0 }),
}))

vi.mock('fast-json-patch', () => ({
  applyPatch: vi.fn().mockReturnValue({ newDocument: { e: {}, g: {}, r: {}, k: {} } }),
}))

import { sessionEventReducer, executeSideEffects, parseSourceTables } from '../sessionEventHandler'
import { initialExecutionState } from '../types'
import type { SessionExecutionState, SessionAction, Message } from '../types'
import type { SubscriptionEvent } from '@/types/api'

function makeEvent(event_type: string, data: Record<string, unknown> = {}, step_number = 0): SubscriptionEvent {
  return {
    event_type: event_type as SubscriptionEvent['event_type'],
    session_id: 'test-session',
    step_number,
    timestamp: new Date().toISOString(),
    data,
  }
}

function subAction(event_type: string, data: Record<string, unknown> = {}, step_number = 0): SessionAction {
  return { type: 'SUBSCRIPTION_EVENT', event: makeEvent(event_type, data, step_number) }
}

function stateWithStepMessages(stepNumbers: number[]): SessionExecutionState {
  const messages: Message[] = []
  const stepMessageIds: Record<number, string> = {}
  for (const n of stepNumbers) {
    const id = `step-msg-${n}`
    stepMessageIds[n] = id
    messages.push({
      id,
      type: 'step',
      content: `Step ${n}: Pending`,
      timestamp: new Date(),
      stepNumber: n,
      isLive: false,
      isPending: true,
    })
  }
  return { ...initialExecutionState, messages, stepMessageIds, status: 'executing', executionPhase: 'executing' }
}

// ---------------------------------------------------------------------------
// parseSourceTables
// ---------------------------------------------------------------------------

describe('parseSourceTables', () => {
  it('extracts schema.table from FROM clause', () => {
    expect(parseSourceTables('SELECT * FROM mydb.users')).toEqual(['mydb.users'])
  })

  it('extracts schema.table from JOIN clause', () => {
    expect(parseSourceTables('SELECT * FROM a.b JOIN c.d ON 1=1')).toEqual(['a.b', 'c.d'])
  })

  it('deduplicates tables referenced multiple times', () => {
    const result = parseSourceTables('SELECT * FROM s.t JOIN s.t ON 1=1')
    expect(result).toEqual(['s.t'])
  })

  it('returns empty array for code with no SQL', () => {
    expect(parseSourceTables('print("hello")')).toEqual([])
  })

  it('handles multiple FROM and JOIN in one query', () => {
    const sql = 'SELECT * FROM a.b JOIN c.d ON 1=1 FROM e.f JOIN g.h ON 1=1'
    const result = parseSourceTables(sql)
    expect(result).toContain('a.b')
    expect(result).toContain('c.d')
    expect(result).toContain('e.f')
    expect(result).toContain('g.h')
  })

  it('handles case-insensitive FROM and JOIN', () => {
    expect(parseSourceTables('select * from Schema.Table join Other.Tbl on 1=1')).toEqual(
      expect.arrayContaining(['Schema.Table', 'Other.Tbl'])
    )
  })
})

// ---------------------------------------------------------------------------
// sessionEventReducer — action types
// ---------------------------------------------------------------------------

describe('sessionEventReducer — action types', () => {
  let state: SessionExecutionState

  beforeEach(() => {
    state = { ...initialExecutionState }
  })

  describe('SUBMIT_QUERY', () => {
    it('sets up thinking message and query state', () => {
      const next = sessionEventReducer(state, {
        type: 'SUBMIT_QUERY',
        query: 'What is revenue?',
        thinkingId: 'think-1',
        isRedo: false,
      })
      expect(next.status).toBe('planning')
      expect(next.thinkingMessageId).toBe('think-1')
      expect(next.currentQuery).toBe('What is revenue?')
      expect(next.isRedo).toBe(false)
      expect(next.messages).toHaveLength(1)
      expect(next.messages[0].type).toBe('thinking')
      expect(next.querySubmittedAt).toBeGreaterThan(0)
    })
  })

  describe('CANCEL_EXECUTION', () => {
    it('sets status to cancelled and resets step info', () => {
      state.status = 'executing'
      state.currentStepNumber = 3
      const next = sessionEventReducer(state, { type: 'CANCEL_EXECUTION' })
      expect(next.status).toBe('cancelled')
      expect(next.executionPhase).toBe('idle')
      expect(next.currentStepNumber).toBe(0)
    })
  })

  describe('APPROVE_PLAN', () => {
    it('appends step messages and sets executing', () => {
      const stepMsg: Message = { id: 's1', type: 'step', content: 'Step 1', timestamp: new Date(), stepNumber: 1 }
      const next = sessionEventReducer(state, {
        type: 'APPROVE_PLAN',
        stepMessages: [stepMsg],
        stepMessageIds: { 1: 's1' },
        isRedo: false,
      })
      expect(next.status).toBe('executing')
      expect(next.executionPhase).toBe('executing')
      expect(next.messages).toHaveLength(1)
      expect(next.stepMessageIds[1]).toBe('s1')
    })

    it('marks existing step messages as superseded when isRedo', () => {
      const existing: Message = { id: 'old', type: 'step', content: 'Old step', timestamp: new Date(), stepNumber: 1 }
      state.messages = [existing]
      const newStep: Message = { id: 'new', type: 'step', content: 'New step', timestamp: new Date(), stepNumber: 1 }
      const next = sessionEventReducer(state, {
        type: 'APPROVE_PLAN',
        stepMessages: [newStep],
        stepMessageIds: { 1: 'new' },
        isRedo: true,
      })
      expect(next.messages[0].isSuperseded).toBe(true)
      expect(next.messages[1].id).toBe('new')
    })
  })

  describe('REJECT_PLAN', () => {
    it('with feedback: sets status to planning', () => {
      state.plan = { steps: [] } as any
      const next = sessionEventReducer(state, { type: 'REJECT_PLAN', hasFeedback: true })
      expect(next.plan).toBeNull()
      expect(next.status).toBe('planning')
    })

    it('without feedback: sets status to idle', () => {
      state.plan = { steps: [] } as any
      const next = sessionEventReducer(state, { type: 'REJECT_PLAN', hasFeedback: false })
      expect(next.plan).toBeNull()
      expect(next.status).toBe('idle')
    })
  })

  describe('ANSWER_CLARIFICATION', () => {
    it('clears clarification and appends user message', () => {
      state.clarification = {
        needed: true, originalQuestion: 'q', ambiguityReason: 'r',
        questions: [], currentStep: 0, answers: {}, structuredAnswers: {},
      }
      const msg: Message = { id: 'u1', type: 'user', content: 'My answer', timestamp: new Date() }
      const next = sessionEventReducer(state, {
        type: 'ANSWER_CLARIFICATION', isInputRequest: false, userMessage: msg, stepMsgId: null,
      })
      expect(next.clarification).toBeNull()
      expect(next.status).toBe('planning')
      expect(next.messages[next.messages.length - 1].content).toBe('My answer')
    })

    it('inserts user message after step message when stepMsgId provided', () => {
      const stepMsg: Message = { id: 'step-1', type: 'step', content: 'Step 1', timestamp: new Date(), stepNumber: 1 }
      const sysMsg: Message = { id: 'sys-1', type: 'system', content: 'Clarify', timestamp: new Date() }
      state.messages = [stepMsg, sysMsg]
      state.clarification = {
        needed: true, originalQuestion: 'q', ambiguityReason: 'r',
        questions: [], currentStep: 0, answers: {}, structuredAnswers: {},
      }
      const userMsg: Message = { id: 'u1', type: 'user', content: 'Answer', timestamp: new Date() }
      const next = sessionEventReducer(state, {
        type: 'ANSWER_CLARIFICATION', isInputRequest: true, userMessage: userMsg, stepMsgId: 'step-1',
      })
      expect(next.status).toBe('executing')
      // User message should be inserted after the system message (idx 2)
      expect(next.messages[2].id).toBe('u1')
    })

    it('sets executing when isInputRequest', () => {
      state.clarification = {
        needed: true, originalQuestion: 'q', ambiguityReason: 'r',
        questions: [], currentStep: 0, answers: {}, structuredAnswers: {},
      }
      const msg: Message = { id: 'u1', type: 'user', content: 'Answer', timestamp: new Date() }
      const next = sessionEventReducer(state, {
        type: 'ANSWER_CLARIFICATION', isInputRequest: true, userMessage: msg, stepMsgId: null,
      })
      expect(next.status).toBe('executing')
    })
  })

  describe('SKIP_CLARIFICATION', () => {
    it('clears clarification, sets planning when not input request', () => {
      state.clarification = {
        needed: true, originalQuestion: 'q', ambiguityReason: 'r',
        questions: [], currentStep: 0, answers: {}, structuredAnswers: {},
      }
      const next = sessionEventReducer(state, { type: 'SKIP_CLARIFICATION', isInputRequest: false })
      expect(next.clarification).toBeNull()
      expect(next.status).toBe('planning')
    })

    it('sets executing when isInputRequest', () => {
      state.clarification = {
        needed: true, originalQuestion: 'q', ambiguityReason: 'r',
        questions: [], currentStep: 0, answers: {}, structuredAnswers: {},
      }
      const next = sessionEventReducer(state, { type: 'SKIP_CLARIFICATION', isInputRequest: true })
      expect(next.status).toBe('executing')
    })
  })

  describe('SET_CLARIFICATION_STEP', () => {
    it('updates current step when clarification exists', () => {
      state.clarification = {
        needed: true, originalQuestion: 'q', ambiguityReason: 'r',
        questions: [], currentStep: 0, answers: {}, structuredAnswers: {},
      }
      const next = sessionEventReducer(state, { type: 'SET_CLARIFICATION_STEP', step: 2 })
      expect(next.clarification?.currentStep).toBe(2)
    })

    it('does nothing when no clarification', () => {
      const next = sessionEventReducer(state, { type: 'SET_CLARIFICATION_STEP', step: 2 })
      expect(next.clarification).toBeNull()
    })
  })

  describe('SET_CLARIFICATION_ANSWER', () => {
    it('stores answer for the given step', () => {
      state.clarification = {
        needed: true, originalQuestion: 'q', ambiguityReason: 'r',
        questions: [], currentStep: 0, answers: {}, structuredAnswers: {},
      }
      const next = sessionEventReducer(state, { type: 'SET_CLARIFICATION_ANSWER', step: 1, answer: 'yes' })
      expect(next.clarification?.answers[1]).toBe('yes')
    })

    it('returns null clarification when none exists', () => {
      const next = sessionEventReducer(state, { type: 'SET_CLARIFICATION_ANSWER', step: 1, answer: 'yes' })
      expect(next.clarification).toBeNull()
    })
  })

  describe('SET_CLARIFICATION_STRUCTURED_ANSWER', () => {
    it('stores structured answer', () => {
      state.clarification = {
        needed: true, originalQuestion: 'q', ambiguityReason: 'r',
        questions: [], currentStep: 0, answers: {}, structuredAnswers: {},
      }
      const next = sessionEventReducer(state, { type: 'SET_CLARIFICATION_STRUCTURED_ANSWER', step: 0, data: { foo: 'bar' } })
      expect(next.clarification?.structuredAnswers[0]).toEqual({ foo: 'bar' })
    })
  })

  describe('UPDATE_MESSAGE', () => {
    it('updates matching message fields', () => {
      state.messages = [{ id: 'm1', type: 'system', content: 'old', timestamp: new Date() }]
      const next = sessionEventReducer(state, { type: 'UPDATE_MESSAGE', id: 'm1', updates: { content: 'new' } })
      expect(next.messages[0].content).toBe('new')
    })
  })

  describe('REMOVE_MESSAGE', () => {
    it('removes message and clears thinkingMessageId if matching', () => {
      state.messages = [{ id: 't1', type: 'thinking', content: '', timestamp: new Date() }]
      state.thinkingMessageId = 't1'
      const next = sessionEventReducer(state, { type: 'REMOVE_MESSAGE', id: 't1' })
      expect(next.messages).toHaveLength(0)
      expect(next.thinkingMessageId).toBeNull()
    })
  })

  describe('CLEAR_MESSAGES', () => {
    it('clears all messages', () => {
      state.messages = [{ id: 'm1', type: 'system', content: 'hello', timestamp: new Date() }]
      state.thinkingMessageId = 'm1'
      const next = sessionEventReducer(state, { type: 'CLEAR_MESSAGES' })
      expect(next.messages).toHaveLength(0)
      expect(next.thinkingMessageId).toBeNull()
    })
  })

  describe('SET_MESSAGES', () => {
    it('replaces messages', () => {
      const msgs: Message[] = [{ id: 'a', type: 'output', content: 'hi', timestamp: new Date() }]
      const next = sessionEventReducer(state, { type: 'SET_MESSAGES', messages: msgs })
      expect(next.messages).toEqual(msgs)
    })

    it('optionally sets suggestions and plan', () => {
      const msgs: Message[] = []
      const plan = { steps: [{ number: 1, goal: 'g' }] } as any
      const next = sessionEventReducer(state, { type: 'SET_MESSAGES', messages: msgs, suggestions: ['s1'], plan })
      expect(next.suggestions).toEqual(['s1'])
      expect(next.plan).toBe(plan)
    })
  })

  describe('SET_CURRENT_QUERY', () => {
    it('sets current query', () => {
      const next = sessionEventReducer(state, { type: 'SET_CURRENT_QUERY', query: 'hello' })
      expect(next.currentQuery).toBe('hello')
    })
  })

  describe('ADD_QUEUED_MESSAGE', () => {
    it('appends queued message', () => {
      const msg = { id: 'q1', content: 'queued', timestamp: new Date() }
      const next = sessionEventReducer(state, { type: 'ADD_QUEUED_MESSAGE', message: msg })
      expect(next.queuedMessages).toHaveLength(1)
    })
  })

  describe('REMOVE_QUEUED_MESSAGE', () => {
    it('removes queued message by id', () => {
      state.queuedMessages = [{ id: 'q1', content: 'queued', timestamp: new Date() }]
      const next = sessionEventReducer(state, { type: 'REMOVE_QUEUED_MESSAGE', id: 'q1' })
      expect(next.queuedMessages).toHaveLength(0)
    })
  })

  describe('CLEAR_QUEUE', () => {
    it('clears all queued messages', () => {
      state.queuedMessages = [{ id: 'q1', content: 'queued', timestamp: new Date() }]
      const next = sessionEventReducer(state, { type: 'CLEAR_QUEUE' })
      expect(next.queuedMessages).toHaveLength(0)
    })
  })

  describe('RESET', () => {
    it('resets with no messages', () => {
      state.status = 'executing'
      const next = sessionEventReducer(state, { type: 'RESET' })
      expect(next.messages).toEqual([])
      expect(next.status).toBe('idle')
      expect(next.plan).toBeNull()
      expect(next.queuedMessages).toEqual([])
    })
  })
})

// ---------------------------------------------------------------------------
// sessionEventReducer — SUBSCRIPTION_EVENT (reduceWSEvent)
// ---------------------------------------------------------------------------

describe('sessionEventReducer — subscription events', () => {
  let state: SessionExecutionState

  beforeEach(() => {
    state = { ...initialExecutionState }
  })

  describe('heartbeat_ack', () => {
    it('returns state unchanged', () => {
      const next = sessionEventReducer(state, subAction('heartbeat_ack', { server_time: 'now' }))
      expect(next).toEqual(state)
    })
  })

  describe('session_ready', () => {
    it('returns new state object when active_domains present', () => {
      const next = sessionEventReducer(state, subAction('session_ready', { active_domains: ['user'] }))
      expect(next).not.toBe(state) // new object
      expect(next.status).toBe(state.status)
    })

    it('returns same state when no active_domains', () => {
      const next = sessionEventReducer(state, subAction('session_ready', {}))
      expect(next).toBe(state)
    })
  })

  describe('replan_start', () => {
    it('marks steps >= from_step as superseded', () => {
      state = stateWithStepMessages([1, 2, 3])
      const next = sessionEventReducer(state, subAction('replan_start', { from_step: 2 }))
      const step2 = next.messages.find(m => m.stepNumber === 2)
      const step3 = next.messages.find(m => m.stepNumber === 3)
      const step1 = next.messages.find(m => m.stepNumber === 1)
      expect(step2?.isSuperseded).toBe(true)
      expect(step3?.isSuperseded).toBe(true)
      expect(step1?.isSuperseded).toBeFalsy()
    })
  })

  describe('proof_start', () => {
    it('shows generating reasoning chain message', () => {
      const next = sessionEventReducer(state, subAction('proof_start'))
      expect(next.executionPhase).toBe('planning')
      const liveMsg = next.messages.find(m => m.isLive)
      expect(liveMsg?.content).toContain('Generating reasoning chain')
    })
  })

  describe('replanning', () => {
    it('shows revising plan message', () => {
      const next = sessionEventReducer(state, subAction('replanning'))
      expect(next.executionPhase).toBe('planning')
      const liveMsg = next.messages.find(m => m.isLive)
      expect(liveMsg?.content).toContain('Revising plan')
    })
  })

  describe('dynamic_context', () => {
    it('sets queryContext with agent and skills', () => {
      const next = sessionEventReducer(state, subAction('dynamic_context', {
        agent: { name: 'sales-bot', similarity: 0.9 },
        skills: [{ name: 'sql', similarity: 0.8 }],
      }))
      expect(next.queryContext?.agent?.name).toBe('sales-bot')
      expect(next.queryContext?.skills).toHaveLength(1)
    })

    it('updates live message content when liveMessageId exists', () => {
      const liveMsg: Message = { id: 'live-1', type: 'system', content: 'Planning...', timestamp: new Date(), isLive: true }
      state.messages = [liveMsg]
      state.liveMessageId = 'live-1'
      const next = sessionEventReducer(state, subAction('dynamic_context', {
        agent: { name: 'analyst', similarity: 0.9 },
        skills: [],
      }))
      const updated = next.messages.find(m => m.id === 'live-1')
      expect(updated?.content).toContain('@analyst')
    })

    it('updates thinking message content when only thinkingMessageId exists', () => {
      const thinkMsg: Message = { id: 'think-1', type: 'thinking', content: '', timestamp: new Date(), isLive: true }
      state.messages = [thinkMsg]
      state.thinkingMessageId = 'think-1'
      const next = sessionEventReducer(state, subAction('dynamic_context', {
        agent: { name: 'bot', similarity: 0.9 },
        skills: [{ name: 'charting', similarity: 0.7 }],
      }))
      const updated = next.messages.find(m => m.id === 'think-1')
      expect(updated?.content).toContain('@bot')
      expect(updated?.content).toContain('charting')
    })
  })

  describe('plan_ready', () => {
    it('with plan data: sets awaiting_approval', () => {
      const plan = { steps: [{ number: 1, goal: 'Do stuff' }] }
      const next = sessionEventReducer(state, subAction('plan_ready', { plan }))
      expect(next.status).toBe('awaiting_approval')
      expect(next.plan).toBe(plan)
    })

    it('with auto-approved steps: creates step messages', () => {
      const steps = [
        { number: 1, goal: 'First step', role_id: 'analyst', domain: 'sales' },
        { number: 2, goal: 'Second step' },
      ]
      const next = sessionEventReducer(state, subAction('plan_ready', { steps }))
      expect(next.status).toBe('executing')
      expect(next.plan).toBeNull()
      expect(next.messages.filter(m => m.type === 'step')).toHaveLength(2)
      expect(next.messages[0].content).toContain('First step')
      expect(next.messages[0].role).toBe('sales/analyst')
    })

    it('does not duplicate already-existing step numbers', () => {
      state = stateWithStepMessages([1])
      const steps = [{ number: 1, goal: 'Already exists' }, { number: 2, goal: 'New' }]
      const next = sessionEventReducer(state, subAction('plan_ready', { steps }))
      const stepMsgs = next.messages.filter(m => m.type === 'step' && !m.isSuperseded)
      // Original step 1 + new step 2
      expect(stepMsgs).toHaveLength(2)
    })
  })

  describe('plan_updated', () => {
    it('removes stale pending steps and adds new ones', () => {
      state = stateWithStepMessages([1, 2, 3])
      // Steps 1 and 2 remain, 3 is removed, 4 is added
      const steps = [
        { number: 1, goal: 'Step 1' },
        { number: 2, goal: 'Step 2' },
        { number: 4, goal: 'New step' },
      ]
      const next = sessionEventReducer(state, subAction('plan_updated', { steps }))
      // Step 3 (pending, not in new steps) should be removed
      const step3 = next.messages.find(m => m.stepNumber === 3)
      expect(step3).toBeUndefined()
      // Step 4 should be added
      const step4 = next.messages.find(m => m.stepNumber === 4)
      expect(step4).toBeDefined()
      expect(step4?.content).toContain('New step')
    })
  })

  describe('step_start', () => {
    it('updates step message content and sets currentStepNumber', () => {
      state = stateWithStepMessages([1])
      const next = sessionEventReducer(state, subAction('step_start', { goal: 'Compute revenue' }, 1))
      expect(next.currentStepNumber).toBe(1)
      expect(next.status).toBe('executing')
      const msg = next.messages.find(m => m.stepNumber === 1)
      expect(msg?.content).toContain('Compute revenue')
    })

    it('sets lastQueryStartStep on first step', () => {
      state = stateWithStepMessages([1])
      state.currentStepNumber = 0
      const next = sessionEventReducer(state, subAction('step_start', { goal: 'First' }, 1))
      expect(next.lastQueryStartStep).toBe(1)
    })

    it('sets qualified role from agent and domain', () => {
      state = stateWithStepMessages([1])
      const next = sessionEventReducer(state, subAction('step_start', { goal: 'Go', agent: 'analyst', domain: 'sales' }, 1))
      const msg = next.messages.find(m => m.stepNumber === 1)
      expect(msg?.role).toBe('sales/analyst')
    })
  })

  describe('step_generating', () => {
    it('sets generating phase and updates message', () => {
      state = stateWithStepMessages([1])
      const next = sessionEventReducer(state, subAction('step_generating', { goal: 'Compute' }, 1))
      expect(next.executionPhase).toBe('generating')
      const msg = next.messages.find(m => m.stepNumber === 1)
      expect(msg?.content).toContain('Planning')
    })

    it('shows attempt number when > 1', () => {
      state = stateWithStepMessages([1])
      state.stepAttempt = 2
      const next = sessionEventReducer(state, subAction('step_generating', { goal: 'Compute' }, 1))
      const msg = next.messages.find(m => m.stepNumber === 1)
      expect(msg?.content).toContain('attempt 2')
    })

    it('returns generating phase even without step message', () => {
      const next = sessionEventReducer(state, subAction('step_generating', {}, 99))
      expect(next.executionPhase).toBe('generating')
    })
  })

  describe('model_escalation', () => {
    it('updates step message with model transition', () => {
      state = stateWithStepMessages([1])
      const next = sessionEventReducer(state, subAction('model_escalation', {
        from_model: 'provider/Meta-Llama-3.1-70B-Instruct-Turbo',
        to_model: 'provider/gpt-4o',
        reason: 'Complex query needs better model',
      }, 1))
      expect(next.executionPhase).toBe('retrying')
      const msg = next.messages.find(m => m.stepNumber === 1)
      expect(msg?.content).toContain('Llama-3.1-70B')
      expect(msg?.content).toContain('gpt-4o')
    })

    it('increments stepAttempts', () => {
      state = stateWithStepMessages([1])
      const next = sessionEventReducer(state, subAction('model_escalation', {
        from_model: 'a', to_model: 'b', reason: 'x',
      }, 1))
      const msg = next.messages.find(m => m.stepNumber === 1)
      expect(msg?.stepAttempts).toBe(1)
    })
  })

  describe('step_executing', () => {
    it('updates message with executing status', () => {
      state = stateWithStepMessages([1])
      const next = sessionEventReducer(state, subAction('step_executing', { goal: 'Revenue', code: 'SELECT * FROM db.sales' }, 1))
      expect(next.executionPhase).toBe('executing')
      const msg = next.messages.find(m => m.stepNumber === 1)
      expect(msg?.content).toContain('Executing')
      expect(msg?.stepSourcesRead).toContain('db.sales')
    })
  })

  describe('step_error', () => {
    it('increments attempt and sets retrying', () => {
      state = stateWithStepMessages([1])
      state.stepAttempt = 1
      const next = sessionEventReducer(state, subAction('step_error', {}, 1))
      expect(next.stepAttempt).toBe(2)
      expect(next.executionPhase).toBe('retrying')
      const msg = next.messages.find(m => m.stepNumber === 1)
      expect(msg?.content).toContain('attempt 2')
    })
  })

  describe('step_failed', () => {
    it('marks step as failed', () => {
      state = stateWithStepMessages([1])
      const next = sessionEventReducer(state, subAction('step_failed', { error: 'Timeout' }, 1))
      const msg = next.messages.find(m => m.stepNumber === 1)
      expect(msg?.content).toContain('Timeout')
      expect(msg?.isLive).toBe(false)
    })
  })

  describe('step_complete', () => {
    it('marks step as completed with goal and duration', () => {
      state = stateWithStepMessages([1])
      const next = sessionEventReducer(state, subAction('step_complete', {
        success: true, goal: 'Computed revenue', stdout: 'Revenue: $1M',
        tables_created: ['results'], duration_ms: 500, attempts: 2,
        code: 'SELECT * FROM db.sales',
      }, 1))
      const msg = next.messages.find(m => m.stepNumber === 1)
      expect(msg?.content).toContain('Computed revenue')
      expect(msg?.content).toContain('Revenue: $1M')
      expect(msg?.isLive).toBe(false)
      expect(msg?.stepDurationMs).toBe(500)
      expect(msg?.stepAttempts).toBe(1) // attempts - 1
      expect(msg?.stepTablesCreated).toEqual(['results'])
    })
  })

  describe('validation_retry', () => {
    it('shows validation failure message', () => {
      state = stateWithStepMessages([1])
      const next = sessionEventReducer(state, subAction('validation_retry', { validation: 'Row count mismatch' }, 1))
      expect(next.executionPhase).toBe('retrying')
      const msg = next.messages.find(m => m.stepNumber === 1)
      expect(msg?.content).toContain('Row count mismatch')
    })
  })

  describe('validation_warnings', () => {
    it('appends warning text to step message', () => {
      state = stateWithStepMessages([1])
      state.messages[0].content = 'Step 1: Executing...'
      const next = sessionEventReducer(state, subAction('validation_warnings', { warnings: ['Null values detected'] }, 1))
      const msg = next.messages.find(m => m.stepNumber === 1)
      expect(msg?.content).toContain('Null values detected')
    })

    it('returns state unchanged when warnings empty', () => {
      const next = sessionEventReducer(state, subAction('validation_warnings', { warnings: [] }, 1))
      expect(next).toBe(state)
    })

    it('returns state unchanged when no step message exists', () => {
      const next = sessionEventReducer(state, subAction('validation_warnings', { warnings: ['w'] }, 99))
      expect(next).toBe(state)
    })
  })

  describe('synthesizing / generating_insights', () => {
    it('finalizes all steps and shows thinking message', () => {
      state = stateWithStepMessages([1])
      const next = sessionEventReducer(state, subAction('synthesizing', { message: 'Generating insights...' }))
      expect(next.executionPhase).toBe('synthesizing')
      const thinkMsg = next.messages.find(m => m.type === 'thinking')
      expect(thinkMsg).toBeDefined()
      expect(thinkMsg?.content).toContain('Generating insights')
    })

    it('reuses existing thinking message if present', () => {
      state.messages = [{ id: 'think-1', type: 'thinking', content: '', timestamp: new Date(), isLive: true }]
      const next = sessionEventReducer(state, subAction('generating_insights', { message: 'Almost done' }))
      expect(next.thinkingMessageId).toBe('think-1')
      const msg = next.messages.find(m => m.id === 'think-1')
      expect(msg?.content).toBe('Almost done')
    })
  })

  describe('query_complete', () => {
    it('with output and step messages: adds final insight output message', () => {
      state = stateWithStepMessages([1])
      state.querySubmittedAt = Date.now() - 1000
      const next = sessionEventReducer(state, subAction('query_complete', {
        output: 'Revenue is $1M', suggestions: ['Next question?'],
      }))
      expect(next.status).toBe('completed')
      expect(next.executionPhase).toBe('idle')
      expect(next.suggestions).toEqual(['Next question?'])
      const outputMsg = next.messages.find(m => m.type === 'output' && m.isFinalInsight)
      expect(outputMsg?.content).toBe('Revenue is $1M')
    })

    it('brief query_complete: marks last step as final insight', () => {
      state = stateWithStepMessages([1])
      const next = sessionEventReducer(state, subAction('query_complete', { brief: true }))
      const stepMsg = next.messages.find(m => m.type === 'step')
      expect(stepMsg?.isFinalInsight).toBe(true)
    })

    it('no step messages: creates output message', () => {
      const next = sessionEventReducer(state, subAction('query_complete', { output: 'Done' }))
      const outputMsg = next.messages.find(m => m.type === 'output')
      expect(outputMsg?.content).toBe('Done')
      expect(outputMsg?.isFinalInsight).toBe(true)
    })

    it('no step messages and no output: defaults to "Analysis complete"', () => {
      const next = sessionEventReducer(state, subAction('query_complete', {}))
      const outputMsg = next.messages.find(m => m.type === 'output')
      expect(outputMsg?.content).toBe('Analysis complete')
    })
  })

  describe('query_error', () => {
    it('adds error message for non-cancellation/rejection errors', () => {
      state = stateWithStepMessages([1])
      const next = sessionEventReducer(state, subAction('query_error', { error: 'SQL syntax error' }))
      expect(next.status).toBe('error')
      const errMsg = next.messages.find(m => m.type === 'error')
      expect(errMsg?.content).toBe('SQL syntax error')
    })

    it('does not add error message for cancellation errors', () => {
      const next = sessionEventReducer(state, subAction('query_error', { error: 'Query cancelled by user' }))
      expect(next.status).toBe('error')
      expect(next.messages.filter(m => m.type === 'error')).toHaveLength(0)
    })

    it('does not add error message for rejection errors', () => {
      const next = sessionEventReducer(state, subAction('query_error', { error: 'Plan was rejected' }))
      expect(next.status).toBe('error')
      expect(next.messages.filter(m => m.type === 'error')).toHaveLength(0)
    })

    it('defaults error message when not provided', () => {
      const next = sessionEventReducer(state, subAction('query_error', {}))
      const errMsg = next.messages.find(m => m.type === 'error')
      expect(errMsg?.content).toBe('Query failed')
    })
  })

  describe('query_cancelled', () => {
    it('adds cancellation system message', () => {
      state = stateWithStepMessages([1])
      const next = sessionEventReducer(state, subAction('query_cancelled'))
      expect(next.status).toBe('cancelled')
      const sysMsg = next.messages.find(m => m.type === 'system' && m.content === 'Query cancelled')
      expect(sysMsg).toBeDefined()
    })
  })

  describe('clarification_needed', () => {
    it('sets up clarification state with questions', () => {
      const next = sessionEventReducer(state, subAction('clarification_needed', {
        original_question: 'What is revenue?',
        ambiguity_reason: 'Multiple revenue types',
        questions: [
          { text: 'Gross or net?', suggestions: ['Gross', 'Net'] },
          { text: 'Which year?', suggestions: ['2024', '2025'] },
        ],
      }))
      expect(next.status).toBe('awaiting_approval')
      expect(next.clarification?.needed).toBe(true)
      expect(next.clarification?.questions).toHaveLength(2)
      expect(next.clarification?.originalQuestion).toBe('What is revenue?')
      // Should have multi-question message
      const sysMsg = next.messages.find(m => m.type === 'system')
      expect(sysMsg?.content).toContain('1. Gross or net?')
      expect(sysMsg?.content).toContain('2. Which year?')
    })

    it('single question without "please clarify" prefix gets prefix added', () => {
      const next = sessionEventReducer(state, subAction('clarification_needed', {
        original_question: 'q',
        ambiguity_reason: 'r',
        questions: [{ text: 'Which department?', suggestions: [] }],
      }))
      const sysMsg = next.messages.find(m => m.type === 'system')
      expect(sysMsg?.content).toContain('Please clarify: Which department?')
    })

    it('single question starting with "Please clarify" keeps as-is', () => {
      const next = sessionEventReducer(state, subAction('clarification_needed', {
        original_question: 'q',
        ambiguity_reason: 'r',
        questions: [{ text: 'Please clarify the time range', suggestions: [] }],
      }))
      const sysMsg = next.messages.find(m => m.type === 'system')
      expect(sysMsg?.content).toBe('Please clarify the time range')
    })

    it('no questions: default message', () => {
      const next = sessionEventReducer(state, subAction('clarification_needed', {
        original_question: 'q',
        ambiguity_reason: 'r',
        questions: [],
      }))
      const sysMsg = next.messages.find(m => m.type === 'system')
      expect(sysMsg?.content).toBe('Please clarify your question.')
    })

    it('inserts after current step message when currentStepNumber set', () => {
      state = stateWithStepMessages([1])
      state.currentStepNumber = 1
      const next = sessionEventReducer(state, subAction('clarification_needed', {
        original_question: 'q', ambiguity_reason: 'r',
        questions: [{ text: 'Which?', suggestions: [] }],
      }))
      // System message should be right after step 1
      const stepIdx = next.messages.findIndex(m => m.stepNumber === 1)
      expect(next.messages[stepIdx + 1]?.type).toBe('system')
    })
  })

  describe('steps_truncated', () => {
    it('marks steps >= step_number as superseded', () => {
      state = stateWithStepMessages([1, 2, 3])
      const next = sessionEventReducer(state, subAction('steps_truncated', {}, 2))
      expect(next.messages.find(m => m.stepNumber === 1)?.isSuperseded).toBeFalsy()
      expect(next.messages.find(m => m.stepNumber === 2)?.isSuperseded).toBe(true)
      expect(next.messages.find(m => m.stepNumber === 3)?.isSuperseded).toBe(true)
    })
  })

  describe('pure side-effect events return state unchanged', () => {
    const sideEffectOnly = [
      'table_created', 'artifact_created', 'facts_extracted', 'fact_start',
      'fact_planning', 'fact_executing', 'fact_resolved', 'fact_failed',
      'dag_execution_start', 'inference_code', 'proof_summary_ready', 'proof_complete',
      'progress', 'entity_rebuild_complete', 'entity_rebuild_start', 'entity_state',
      'entity_patch', 'source_ingest_complete', 'source_ingest_error',
      'source_ingest_progress', 'source_ingest_start', 'glossary_terms_added',
      'glossary_rebuild_complete', 'glossary_rebuild_start', 'glossary_generation_progress',
      'relationships_extracted',
    ]

    for (const eventType of sideEffectOnly) {
      it(`${eventType} returns state unchanged`, () => {
        const next = sessionEventReducer(state, subAction(eventType))
        expect(next).toBe(state)
      })
    }
  })

  describe('unknown event type', () => {
    it('returns state unchanged', () => {
      const next = sessionEventReducer(state, subAction('some_unknown_event_type'))
      expect(next).toBe(state)
    })
  })
})

// ---------------------------------------------------------------------------
// executeSideEffects
// ---------------------------------------------------------------------------

describe('executeSideEffects — comprehensive', () => {
  const stores = {} as any
  const lastHeartbeatRef = { current: null as string | null }

  beforeEach(() => {
    vi.clearAllMocks()
    lastHeartbeatRef.current = null
  })

  it('proof_start: clears inference codes and handles fact event', async () => {
    const { clearInferenceCodes } = await import('@/graphql/ui-state')
    const { handleFactEvent } = await import('@/graphql/ui-state')
    executeSideEffects(makeEvent('proof_start'), 'test-session', stores, lastHeartbeatRef)
    expect(clearInferenceCodes).toHaveBeenCalled()
    expect(handleFactEvent).toHaveBeenCalledWith('proof_start', expect.any(Object))
  })

  it('step_executing: calls addStepCode when code present', async () => {
    const { addStepCode } = await import('@/graphql/ui-state')
    executeSideEffects(makeEvent('step_executing', { goal: 'Revenue', code: 'SELECT 1', model: 'gpt-4o' }, 1), 'test-session', stores, lastHeartbeatRef)
    expect(addStepCode).toHaveBeenCalledWith(1, 'Revenue', 'SELECT 1', 'gpt-4o')
  })

  it('step_executing: does not call addStepCode without code', async () => {
    const { addStepCode } = await import('@/graphql/ui-state')
    vi.mocked(addStepCode).mockClear()
    executeSideEffects(makeEvent('step_executing', { goal: 'Revenue' }, 1), 'test-session', stores, lastHeartbeatRef)
    expect(addStepCode).not.toHaveBeenCalled()
  })

  it('step_complete: calls addStepCode when code present', async () => {
    const { addStepCode } = await import('@/graphql/ui-state')
    executeSideEffects(makeEvent('step_complete', { goal: 'Done', code: 'SELECT 2', model: 'gpt-4' }, 1), 'test-session', stores, lastHeartbeatRef)
    expect(addStepCode).toHaveBeenCalledWith(1, 'Done', 'SELECT 2', 'gpt-4')
  })

  it('query_complete: refetches all data queries', async () => {
    const { apolloClient } = await import('@/graphql/client')
    executeSideEffects(makeEvent('query_complete'), 'test-session', stores, lastHeartbeatRef)
    expect(apolloClient.refetchQueries).toHaveBeenCalledWith({
      include: ['Tables', 'Artifacts', 'Facts', 'InferenceCodes', 'Steps', 'Learnings', 'Scratchpad', 'SessionDDL'],
    })
  })

  it('table_created: writes to Apollo cache when existing data', async () => {
    const { apolloClient } = await import('@/graphql/client')
    vi.mocked(apolloClient.readQuery).mockReturnValue({
      tables: { tables: [], total: 0 },
    })
    executeSideEffects(makeEvent('table_created', { name: 'results', row_count: 10, columns: ['a', 'b'] }, 1), 'test-session', stores, lastHeartbeatRef)
    expect(apolloClient.writeQuery).toHaveBeenCalled()
  })

  it('table_created: updates existing table by name', async () => {
    const { apolloClient } = await import('@/graphql/client')
    vi.mocked(apolloClient.readQuery).mockReturnValue({
      tables: { tables: [{ name: 'results', rowCount: 5 }], total: 1 },
    })
    executeSideEffects(makeEvent('table_created', { name: 'results', row_count: 20 }, 1), 'test-session', stores, lastHeartbeatRef)
    expect(apolloClient.writeQuery).toHaveBeenCalled()
  })

  it('table_created: no-op when no existing cache', async () => {
    const { apolloClient } = await import('@/graphql/client')
    vi.mocked(apolloClient.readQuery).mockReturnValue(null)
    vi.mocked(apolloClient.writeQuery).mockClear()
    executeSideEffects(makeEvent('table_created', { name: 'x' }, 1), 'test-session', stores, lastHeartbeatRef)
    expect(apolloClient.writeQuery).not.toHaveBeenCalled()
  })

  it('artifact_created: writes to Apollo cache when data complete', async () => {
    const { apolloClient } = await import('@/graphql/client')
    vi.mocked(apolloClient.readQuery).mockReturnValue({
      artifacts: { artifacts: [], total: 0 },
    })
    executeSideEffects(makeEvent('artifact_created', {
      id: 1, name: 'chart', artifact_type: 'plotly', title: 'Revenue Chart',
    }, 1), 'test-session', stores, lastHeartbeatRef)
    expect(apolloClient.writeQuery).toHaveBeenCalled()
  })

  it('artifact_created: skips when missing required fields', async () => {
    const { apolloClient } = await import('@/graphql/client')
    vi.mocked(apolloClient.writeQuery).mockClear()
    executeSideEffects(makeEvent('artifact_created', { name: 'chart' }, 1), 'test-session', stores, lastHeartbeatRef)
    expect(apolloClient.writeQuery).not.toHaveBeenCalled()
  })

  it('artifact_created: updates existing artifact by id', async () => {
    const { apolloClient } = await import('@/graphql/client')
    vi.mocked(apolloClient.readQuery).mockReturnValue({
      artifacts: { artifacts: [{ id: 1, name: 'old' }], total: 1 },
    })
    executeSideEffects(makeEvent('artifact_created', {
      id: 1, name: 'new', artifact_type: 'plotly',
    }, 1), 'test-session', stores, lastHeartbeatRef)
    expect(apolloClient.writeQuery).toHaveBeenCalled()
  })

  it('facts_extracted: writes facts to Apollo cache', async () => {
    const { apolloClient } = await import('@/graphql/client')
    vi.mocked(apolloClient.readQuery).mockReturnValue({
      facts: { facts: [], total: 0 },
    })
    executeSideEffects(makeEvent('facts_extracted', {
      facts: [{ name: 'revenue', value: 1000000, source: 'sales', reasoning: 'sum', confidence: 0.95 }],
    }), 'test-session', stores, lastHeartbeatRef)
    expect(apolloClient.writeQuery).toHaveBeenCalled()
  })

  it('facts_extracted: updates existing fact by name', async () => {
    const { apolloClient } = await import('@/graphql/client')
    vi.mocked(apolloClient.readQuery).mockReturnValue({
      facts: { facts: [{ name: 'revenue', value: 500 }], total: 1 },
    })
    executeSideEffects(makeEvent('facts_extracted', {
      facts: [{ name: 'revenue', value: 1000 }],
    }), 'test-session', stores, lastHeartbeatRef)
    expect(apolloClient.writeQuery).toHaveBeenCalled()
  })

  it('facts_extracted: no-op when no facts', async () => {
    const { apolloClient } = await import('@/graphql/client')
    vi.mocked(apolloClient.writeQuery).mockClear()
    executeSideEffects(makeEvent('facts_extracted', { facts: [] }), 'test-session', stores, lastHeartbeatRef)
    expect(apolloClient.writeQuery).not.toHaveBeenCalled()
  })

  it('facts_extracted: no-op when no existing cache', async () => {
    const { apolloClient } = await import('@/graphql/client')
    vi.mocked(apolloClient.readQuery).mockReturnValue(null)
    vi.mocked(apolloClient.writeQuery).mockClear()
    executeSideEffects(makeEvent('facts_extracted', {
      facts: [{ name: 'x' }],
    }), 'test-session', stores, lastHeartbeatRef)
    expect(apolloClient.writeQuery).not.toHaveBeenCalled()
  })

  it('steps_truncated: calls truncateFromStep and refetches', async () => {
    const { truncateFromStep } = await import('@/graphql/ui-state')
    const { apolloClient } = await import('@/graphql/client')
    executeSideEffects(makeEvent('steps_truncated', {}, 3), 'test-session', stores, lastHeartbeatRef)
    expect(truncateFromStep).toHaveBeenCalledWith(3)
    expect(apolloClient.refetchQueries).toHaveBeenCalledWith({ include: ['Tables', 'Artifacts', 'SessionDDL'] })
  })

  it('fact lifecycle events are forwarded to handleFactEvent', async () => {
    const { handleFactEvent } = await import('@/graphql/ui-state')
    const factEvents = ['fact_start', 'fact_planning', 'fact_executing', 'fact_resolved', 'fact_failed', 'dag_execution_start']
    for (const eventType of factEvents) {
      vi.mocked(handleFactEvent).mockClear()
      executeSideEffects(makeEvent(eventType, { some: 'data' }), 'test-session', stores, lastHeartbeatRef)
      expect(handleFactEvent).toHaveBeenCalledWith(eventType, expect.objectContaining({ some: 'data' }))
    }
  })

  it('inference_code: calls handleFactEvent and addInferenceCode', async () => {
    const { handleFactEvent, addInferenceCode } = await import('@/graphql/ui-state')
    executeSideEffects(makeEvent('inference_code', {
      inference_id: 'inf-1', name: 'test', operation: 'compute', code: 'SELECT 1', attempt: 1, model: 'gpt-4',
    }), 'test-session', stores, lastHeartbeatRef)
    expect(handleFactEvent).toHaveBeenCalledWith('inference_code', expect.any(Object))
    expect(addInferenceCode).toHaveBeenCalledWith({
      inference_id: 'inf-1', name: 'test', operation: 'compute', code: 'SELECT 1', attempt: 1, model: 'gpt-4',
    })
  })

  it('inference_code: skips addInferenceCode when no inference_id or code', async () => {
    const { addInferenceCode } = await import('@/graphql/ui-state')
    vi.mocked(addInferenceCode).mockClear()
    executeSideEffects(makeEvent('inference_code', { name: 'test' }), 'test-session', stores, lastHeartbeatRef)
    expect(addInferenceCode).not.toHaveBeenCalled()
  })

  it('proof_summary_ready: saves facts when available', async () => {
    const { handleFactEvent, exportFacts } = await import('@/graphql/ui-state')
    const { apolloClient } = await import('@/graphql/client')
    vi.mocked(exportFacts).mockReturnValue([{ name: 'f1', value: 1 }] as any)
    executeSideEffects(makeEvent('proof_summary_ready', { summary: 'All good' }), 'test-session', stores, lastHeartbeatRef)
    expect(handleFactEvent).toHaveBeenCalledWith('proof_summary_ready', expect.any(Object))
    expect(apolloClient.mutate).toHaveBeenCalled()
  })

  it('proof_summary_ready: skips save when no facts', async () => {
    const { exportFacts } = await import('@/graphql/ui-state')
    const { apolloClient } = await import('@/graphql/client')
    vi.mocked(exportFacts).mockReturnValue([])
    vi.mocked(apolloClient.mutate).mockClear()
    executeSideEffects(makeEvent('proof_summary_ready', { summary: 'Empty' }), 'test-session', stores, lastHeartbeatRef)
    expect(apolloClient.mutate).not.toHaveBeenCalled()
  })

  it('proof_complete: handles fact event', async () => {
    const { handleFactEvent } = await import('@/graphql/ui-state')
    executeSideEffects(makeEvent('proof_complete'), 'test-session', stores, lastHeartbeatRef)
    expect(handleFactEvent).toHaveBeenCalledWith('proof_complete', expect.any(Object))
  })

  it('entity_rebuild_complete: refetches entities and clears rebuilding', async () => {
    const { apolloClient } = await import('@/graphql/client')
    const { setEntityRebuilding } = await import('@/store/glossaryState')
    executeSideEffects(makeEvent('entity_rebuild_complete'), 'test-session', stores, lastHeartbeatRef)
    expect(apolloClient.refetchQueries).toHaveBeenCalledWith({ include: ['Entities'] })
    expect(setEntityRebuilding).toHaveBeenCalledWith(false)
  })

  it('entity_rebuild_start: sets rebuilding true', async () => {
    const { setEntityRebuilding } = await import('@/store/glossaryState')
    executeSideEffects(makeEvent('entity_rebuild_start'), 'test-session', stores, lastHeartbeatRef)
    expect(setEntityRebuilding).toHaveBeenCalledWith(true)
  })

  it('entity_state: inflates terms and caches', async () => {
    const { setTermsFromState } = await import('@/store/glossaryState')
    const { setCachedEntry } = await import('@/store/entityCache')
    const { inflateToGlossaryTerms } = await import('@/store/entityCacheKeys')
    vi.mocked(inflateToGlossaryTerms).mockReturnValue({ terms: [{ name: 't1' }] as any, totalDefined: 1, totalSelfDescribing: 0 })
    executeSideEffects(makeEvent('entity_state', { state: { e: {}, g: {}, r: {}, k: {} }, version: 1 }), 'test-session', stores, lastHeartbeatRef)
    expect(inflateToGlossaryTerms).toHaveBeenCalled()
    expect(setTermsFromState).toHaveBeenCalled()
    expect(setCachedEntry).toHaveBeenCalled()
  })

  it('entity_patch: applies patch and updates terms', async () => {
    const { getCachedEntry, setCachedEntry } = await import('@/store/entityCache')
    const { setTermsFromState } = await import('@/store/glossaryState')
    vi.mocked(getCachedEntry).mockResolvedValue({ state: { e: {}, g: {}, r: {}, k: {} }, version: 1 })
    executeSideEffects(makeEvent('entity_patch', { patch: [{ op: 'add', path: '/e/foo', value: {} }], version: 2 }), 'test-session', stores, lastHeartbeatRef)
    // Wait for async
    await vi.waitFor(() => {
      expect(setCachedEntry).toHaveBeenCalled()
      expect(setTermsFromState).toHaveBeenCalled()
    })
  })

  it('source_ingest_complete: refetches and clears vars', async () => {
    const { apolloClient } = await import('@/graphql/client')
    const { ingestingSourceVar, ingestProgressVar } = await import('@/graphql/ui-state')
    executeSideEffects(makeEvent('source_ingest_complete'), 'test-session', stores, lastHeartbeatRef)
    expect(apolloClient.refetchQueries).toHaveBeenCalledWith({ include: ['DataSources'] })
    expect(ingestingSourceVar).toHaveBeenCalledWith(null)
    expect(ingestProgressVar).toHaveBeenCalledWith(null)
  })

  it('source_ingest_error: logs error (no state change)', async () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    executeSideEffects(makeEvent('source_ingest_error', { name: 'file.csv', error: 'Parse error' }), 'test-session', stores, lastHeartbeatRef)
    expect(spy).toHaveBeenCalled()
    spy.mockRestore()
  })

  it('source_ingest_progress: sets progress', async () => {
    const { ingestProgressVar } = await import('@/graphql/ui-state')
    executeSideEffects(makeEvent('source_ingest_progress', { current: 5, total: 10 }), 'test-session', stores, lastHeartbeatRef)
    expect(ingestProgressVar).toHaveBeenCalledWith({ current: 5, total: 10 })
  })

  it('source_ingest_start: sets ingesting source', async () => {
    const { ingestingSourceVar, ingestProgressVar } = await import('@/graphql/ui-state')
    executeSideEffects(makeEvent('source_ingest_start', { name: 'data.csv' }), 'test-session', stores, lastHeartbeatRef)
    expect(ingestingSourceVar).toHaveBeenCalledWith('data.csv')
    expect(ingestProgressVar).toHaveBeenCalledWith(null)
  })

  it('glossary_terms_added: adds terms', async () => {
    const { addTerms } = await import('@/store/glossaryState')
    const terms = [{ name: 'revenue', definition: 'Total income' }]
    executeSideEffects(makeEvent('glossary_terms_added', { terms }), 'test-session', stores, lastHeartbeatRef)
    expect(addTerms).toHaveBeenCalledWith(terms)
  })

  it('glossary_terms_added: no-op when no terms', async () => {
    const { addTerms } = await import('@/store/glossaryState')
    vi.mocked(addTerms).mockClear()
    executeSideEffects(makeEvent('glossary_terms_added', { terms: [] }), 'test-session', stores, lastHeartbeatRef)
    expect(addTerms).not.toHaveBeenCalled()
  })

  it('glossary_rebuild_complete: sets generating false', async () => {
    const { setGenerating } = await import('@/store/glossaryState')
    executeSideEffects(makeEvent('glossary_rebuild_complete'), 'test-session', stores, lastHeartbeatRef)
    expect(setGenerating).toHaveBeenCalledWith(false)
  })

  it('glossary_rebuild_start: sets generating true', async () => {
    const { setGenerating } = await import('@/store/glossaryState')
    executeSideEffects(makeEvent('glossary_rebuild_start'), 'test-session', stores, lastHeartbeatRef)
    expect(setGenerating).toHaveBeenCalledWith(true)
  })

  it('glossary_generation_progress: sets progress', async () => {
    const { setProgress } = await import('@/store/glossaryState')
    executeSideEffects(makeEvent('glossary_generation_progress', { stage: 'analyzing', percent: 50 }), 'test-session', stores, lastHeartbeatRef)
    expect(setProgress).toHaveBeenCalledWith('analyzing', 50)
  })

  it('relationships_extracted: bumps refresh key', async () => {
    const { bumpRefreshKey } = await import('@/store/glossaryState')
    executeSideEffects(makeEvent('relationships_extracted'), 'test-session', stores, lastHeartbeatRef)
    expect(bumpRefreshKey).toHaveBeenCalled()
  })

  it('synthesizing: refetches artifacts and tables', async () => {
    const { apolloClient } = await import('@/graphql/client')
    executeSideEffects(makeEvent('synthesizing'), 'test-session', stores, lastHeartbeatRef)
    expect(apolloClient.refetchQueries).toHaveBeenCalledWith({ include: ['Artifacts', 'Tables'] })
  })

  it('unknown event: does nothing', () => {
    // Should not throw
    executeSideEffects(makeEvent('totally_unknown'), 'test-session', stores, lastHeartbeatRef)
  })
})

// ---------------------------------------------------------------------------
// Helper function edge cases
// ---------------------------------------------------------------------------

describe('helper function edge cases', () => {
  it('ensureLiveMessage creates new message when no existing live/thinking', () => {
    const state = { ...initialExecutionState }
    const next = sessionEventReducer(state, subAction('planning_start'))
    expect(next.liveMessageId).toBeTruthy()
    expect(next.messages).toHaveLength(1)
    expect(next.messages[0].type).toBe('system')
  })

  it('ensureLiveMessage reuses thinkingMessageId and promotes to liveMessageId', () => {
    const state = {
      ...initialExecutionState,
      messages: [{ id: 'think-1', type: 'thinking' as const, content: '', timestamp: new Date(), isLive: true }],
      thinkingMessageId: 'think-1',
      liveMessageId: null,
    }
    const next = sessionEventReducer(state, subAction('planning_start'))
    expect(next.thinkingMessageId).toBeNull()
    expect(next.liveMessageId).toBe('think-1')
  })

  it('clearLiveMessage removes both live and thinking messages', () => {
    const state = {
      ...initialExecutionState,
      messages: [
        { id: 'live-1', type: 'system' as const, content: 'Planning...', timestamp: new Date(), isLive: true },
        { id: 'think-1', type: 'thinking' as const, content: '', timestamp: new Date(), isLive: true },
      ],
      liveMessageId: 'live-1',
      thinkingMessageId: 'think-1',
    }
    // query_complete calls clearLiveMessage internally
    const next = sessionEventReducer(state, subAction('query_complete', {}))
    expect(next.liveMessageId).toBeNull()
    expect(next.thinkingMessageId).toBeNull()
    expect(next.messages.filter(m => m.id === 'live-1')).toHaveLength(0)
    expect(next.messages.filter(m => m.id === 'think-1')).toHaveLength(0)
  })

  it('updateStepMessage returns state unchanged when step message not found', () => {
    const state = { ...initialExecutionState }
    // step_complete on a step that doesn't exist in stepMessageIds
    const next = sessionEventReducer(state, subAction('step_complete', { success: true }, 99))
    // Should not crash, just return state (no step messages to update)
    expect(next).toBeDefined()
  })
})
