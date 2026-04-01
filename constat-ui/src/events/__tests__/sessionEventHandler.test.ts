import { describe, it, expect, vi, beforeEach } from 'vitest'

vi.mock('@/graphql/client', () => ({
  apolloClient: {
    query: vi.fn().mockResolvedValue({ data: {} }),
    mutate: vi.fn().mockResolvedValue({ data: {} }),
  },
}))

vi.mock('@/config/auth-helpers', () => ({
  isAuthDisabled: true,
  getAuthHeaders: vi.fn().mockResolvedValue({}),
  getToken: vi.fn().mockResolvedValue(null),
}))

import { sessionEventReducer, executeSideEffects } from '../sessionEventHandler'
import { initialExecutionState } from '../types'
import type { SessionExecutionState, SessionAction } from '../types'

describe('sessionEventReducer', () => {
  let state: SessionExecutionState

  beforeEach(() => {
    state = { ...initialExecutionState }
  })

  describe('session_ready event', () => {
    it('returns state unchanged (session update handled externally)', () => {
      const action: SessionAction = {
        type: 'WS_EVENT',
        event: {
          event_type: 'session_ready',
          session_id: 'test-session',
          step_number: 0,
          timestamp: new Date().toISOString(),
          data: { active_domains: ['user', 'sales-analytics'] },
        },
      }
      const next = sessionEventReducer(state, action)
      // session_ready is handled in SessionContext (not reducer) — state is preserved
      expect(next.status).toBe(state.status)
    })
  })

  describe('welcome event', () => {
    it('populates suggestions and tagline from adjectives', () => {
      const action: SessionAction = {
        type: 'WS_EVENT',
        event: {
          event_type: 'welcome',
          session_id: 'test-session',
          step_number: 0,
          timestamp: new Date().toISOString(),
          data: {
            suggestions: ['What can you do?', 'Help me explore'],
            reliable_adjective: 'compulsively-precise',
            honest_adjective: 'constitutionally-incapable-of-lying',
            tagline: 'I help you understand data',
          },
        },
      }
      const next = sessionEventReducer(state, action)
      expect(next.suggestions).toEqual(['What can you do?', 'Help me explore'])
      expect(next.welcomeTagline).toContain('compulsively-precise')
      expect(next.welcomeTagline).toContain('constitutionally-incapable-of-lying')
    })

    it('returns empty tagline when adjectives missing', () => {
      const action: SessionAction = {
        type: 'WS_EVENT',
        event: {
          event_type: 'welcome',
          session_id: 'test-session',
          step_number: 0,
          timestamp: new Date().toISOString(),
          data: {
            suggestions: ['What can you do?'],
          },
        },
      }
      const next = sessionEventReducer(state, action)
      expect(next.suggestions).toEqual(['What can you do?'])
      expect(next.welcomeTagline).toBe('')
    })
  })

  describe('planning_start event', () => {
    it('transitions to planning phase', () => {
      const action: SessionAction = {
        type: 'WS_EVENT',
        event: {
          event_type: 'planning_start',
          session_id: 'test-session',
          step_number: 0,
          timestamp: new Date().toISOString(),
          data: {},
        },
      }
      const next = sessionEventReducer(state, action)
      expect(next.executionPhase).toBe('planning')
    })
  })

  describe('query_complete event', () => {
    it('sets status to completed', () => {
      state.status = 'executing'
      const action: SessionAction = {
        type: 'WS_EVENT',
        event: {
          event_type: 'query_complete',
          session_id: 'test-session',
          step_number: 3,
          timestamp: new Date().toISOString(),
          data: {},
        },
      }
      const next = sessionEventReducer(state, action)
      expect(next.status).toBe('completed')
      expect(next.executionPhase).toBe('idle')
    })
  })

  describe('RESET action', () => {
    it('restores messages and resets execution state', () => {
      state.status = 'executing'
      state.executionPhase = 'executing'
      const messages = [
        { id: '1', type: 'output' as const, content: 'Hello', timestamp: new Date() },
      ]
      const next = sessionEventReducer(state, { type: 'RESET', messages })
      expect(next.messages).toEqual(messages)
      expect(next.status).toBe('idle')
      expect(next.executionPhase).toBe('idle')
    })
  })

  describe('SET_STATUS action', () => {
    it('sets status', () => {
      const next = sessionEventReducer(state, { type: 'SET_STATUS', status: 'executing' })
      expect(next.status).toBe('executing')
    })
  })

  describe('ADD_MESSAGE action', () => {
    it('appends a message', () => {
      const msg = { id: '1', type: 'user' as const, content: 'test', timestamp: new Date() }
      const next = sessionEventReducer(state, { type: 'ADD_MESSAGE', message: msg })
      expect(next.messages).toHaveLength(1)
      expect(next.messages[0].content).toBe('test')
    })
  })
})

describe('executeSideEffects', () => {
  const artifactState = {
    fetchEntities: vi.fn(),
    fetchDataSources: vi.fn(),
    fetchAllSkills: vi.fn(),
    fetchAllAgents: vi.fn(),
    fetchArtifacts: vi.fn(),
    fetchFacts: vi.fn(),
    fetchTables: vi.fn(),
    fetchLearnings: vi.fn(),
    fetchScratchpad: vi.fn(),
    addStepCode: vi.fn(),
    addTable: vi.fn(),
    addArtifact: vi.fn(),
    clearInferenceCodes: vi.fn(),
    addInferenceCode: vi.fn(),
    truncateFromStep: vi.fn(),
    markStepsSuperseded: vi.fn(),
  }
  const proofState = {
    handleFactEvent: vi.fn(),
  }
  const glossaryState = {
    fetchTerms: vi.fn(),
  }
  const stores = {
    artifactStore: { getState: () => artifactState } as any,
    proofStore: { getState: () => proofState } as any,
    glossaryStore: { getState: () => glossaryState } as any,
  }
  const lastHeartbeatRef = { current: null as string | null }

  beforeEach(() => {
    vi.clearAllMocks()
    lastHeartbeatRef.current = null
  })

  it('fetches data sources on session_ready', () => {
    const event = {
      event_type: 'session_ready' as const,
      session_id: 'test-session',
      step_number: 0,
      timestamp: new Date().toISOString(),
      data: { active_domains: ['user'] },
    }
    executeSideEffects(event, 'test-session', stores, lastHeartbeatRef)
    expect(artifactState.fetchDataSources).toHaveBeenCalledWith('test-session')
    expect(artifactState.fetchAllSkills).toHaveBeenCalled()
    expect(artifactState.fetchEntities).toHaveBeenCalledWith('test-session')
    expect(glossaryState.fetchTerms).toHaveBeenCalledWith('test-session')
  })

  it('updates heartbeat on heartbeat_ack', () => {
    const event = {
      event_type: 'heartbeat_ack' as const,
      session_id: 'test-session',
      step_number: 0,
      timestamp: new Date().toISOString(),
      data: { server_time: '2026-03-31T00:00:00Z' },
    }
    executeSideEffects(event, 'test-session', stores, lastHeartbeatRef)
    expect(lastHeartbeatRef.current).toBe('2026-03-31T00:00:00Z')
  })
})
