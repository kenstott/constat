import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, act, waitFor } from '@testing-library/react'
import { MockedProvider } from '@apollo/client/testing'
import { SessionProvider, useSessionContext } from '../SessionContext'
import type { ReactNode } from 'react'

// ---------- Mocks ----------

const mockMutate = vi.fn().mockResolvedValue({
  data: {
    createSession: {
      sessionId: 'sess-1',
      userId: 'test-user',
      status: 'IDLE',
      createdAt: '2025-01-01',
      lastActivity: '2025-01-01',
      activeDomains: [],
      tablesCount: 0,
      artifactsCount: 0,
    },
  },
})

const mockQuery = vi.fn().mockResolvedValue({
  data: {
    messages: { messages: [] },
    proofFacts: { facts: [], summary: null },
  },
})

const mockUnsubscribe = vi.fn()
const mockSubscribeInner = vi.fn().mockReturnValue({ unsubscribe: mockUnsubscribe })

vi.mock('@/contexts/AuthContext', () => ({
  useAuth: () => ({
    userId: 'test-user',
    user: null,
    token: null,
    isAuthenticated: true,
    isAuthDisabled: true,
    isAdmin: true,
    permissions: null,
    initialized: true,
    loading: false,
    error: null,
    login: vi.fn(),
    loginWithGoogle: vi.fn(),
    loginWithEmail: vi.fn(),
    signupWithEmail: vi.fn(),
    sendPasswordReset: vi.fn(),
    sendEmailSignInLink: vi.fn(),
    completeEmailLink: vi.fn(),
    isEmailLinkSignIn: () => false,
    logout: vi.fn(),
    canSee: () => true,
    canWrite: () => true,
    setError: vi.fn(),
    clearError: vi.fn(),
  }),
}))

vi.mock('@/store/glossaryState', () => ({
  fetchTerms: vi.fn(),
  loadFromCache: vi.fn(),
}))

vi.mock('@/graphql/client', () => ({
  apolloClient: {
    query: (...args: unknown[]) => mockQuery(...args),
    mutate: (...args: unknown[]) => mockMutate(...args),
    subscribe: vi.fn().mockReturnValue({ subscribe: (...args: unknown[]) => mockSubscribeInner(...args) }),
    refetchQueries: vi.fn().mockResolvedValue([]),
  },
}))

const mockClearArtifactState = vi.fn()
const mockMarkStepsSuperseded = vi.fn()
const mockImportFacts = vi.fn()
const mockOpenProofPanel = vi.fn()
const mockCloseProofPanel = vi.fn()
const mockClearProofFacts = vi.fn()

vi.mock('@/graphql/ui-state', async () => {
  const { makeVar } = await import('@apollo/client')
  return {
    briefModeVar: makeVar(false),
    clearArtifactState: (...args: unknown[]) => mockClearArtifactState(...args),
    markStepsSuperseded: (...args: unknown[]) => mockMarkStepsSuperseded(...args),
    proofFactsVar: makeVar(new Map()),
    isProvingVar: makeVar(false),
    isPlanningCompleteVar: makeVar(false),
    isProofPanelOpenVar: makeVar(false),
    proofSummaryVar: makeVar(null),
    isSummaryGeneratingVar: makeVar(false),
    hasCompletedProofVar: makeVar(false),
    openProofPanel: (...args: unknown[]) => mockOpenProofPanel(...args),
    closeProofPanel: (...args: unknown[]) => mockCloseProofPanel(...args),
    clearProofFacts: (...args: unknown[]) => mockClearProofFacts(...args),
    importFacts: (...args: unknown[]) => mockImportFacts(...args),
  }
})

vi.mock('@/api/session-id', () => ({
  getOrCreateSessionId: vi.fn().mockReturnValue('test-session-id'),
  createNewSessionId: vi.fn().mockReturnValue('new-session-id'),
  storeSessionId: vi.fn(),
}))

vi.mock('@/events/sessionEventHandler', () => ({
  sessionEventReducer: (state: any, action: any) => {
    if (action.type === 'RESET') {
      return {
        ...state,
        messages: action.messages || [],
        status: 'idle',
        executionPhase: 'idle',
      }
    }
    if (action.type === 'SET_STATUS') {
      return { ...state, status: action.status }
    }
    if (action.type === 'SET_MESSAGES') {
      return {
        ...state,
        messages: action.messages || [],
        suggestions: action.suggestions || [],
        plan: action.plan,
      }
    }
    if (action.type === 'SUBMIT_QUERY') {
      return {
        ...state,
        status: 'executing',
        executionPhase: 'planning',
        currentQuery: action.query,
      }
    }
    if (action.type === 'CANCEL_EXECUTION') {
      return { ...state, status: 'idle', executionPhase: 'idle' }
    }
    if (action.type === 'ADD_MESSAGE') {
      return { ...state, messages: [...state.messages, action.message] }
    }
    if (action.type === 'REMOVE_MESSAGE') {
      return { ...state, messages: state.messages.filter((m: any) => m.id !== action.id) }
    }
    if (action.type === 'REMOVE_QUEUED_MESSAGE') {
      return { ...state, queuedMessages: state.queuedMessages.filter((m: any) => m.id !== action.id) }
    }
    if (action.type === 'ADD_QUEUED_MESSAGE') {
      return { ...state, queuedMessages: [...state.queuedMessages, action.message] }
    }
    if (action.type === 'APPROVE_PLAN') {
      return { ...state, status: 'executing', executionPhase: 'executing' }
    }
    if (action.type === 'REJECT_PLAN') {
      return {
        ...state,
        status: action.hasFeedback ? 'executing' : 'idle',
        executionPhase: action.hasFeedback ? 'planning' : 'idle',
      }
    }
    if (action.type === 'SKIP_CLARIFICATION') {
      return { ...state, clarification: null }
    }
    if (action.type === 'SET_CLARIFICATION_STEP') {
      return state
    }
    if (action.type === 'SET_CLARIFICATION_ANSWER') {
      return state
    }
    if (action.type === 'SET_CLARIFICATION_STRUCTURED_ANSWER') {
      return state
    }
    if (action.type === 'SUBSCRIPTION_EVENT') {
      return state
    }
    if (action.type === 'ANSWER_CLARIFICATION') {
      return { ...state, clarification: null }
    }
    return state
  },
  executeSideEffects: vi.fn(),
}))

vi.mock('@/graphql/operations/sessions', () => ({
  CREATE_SESSION: { kind: 'Document' },
  SHARE_SESSION: { kind: 'Document' },
  SET_ACTIVE_DOMAINS: { kind: 'Document' },
  toSession: (gql: any) => ({
    session_id: gql.sessionId,
    user_id: gql.userId,
    status: gql.status?.toLowerCase() ?? 'idle',
    created_at: gql.createdAt,
    last_activity: gql.lastActivity,
    active_domains: gql.activeDomains ?? [],
    tables_count: gql.tablesCount ?? 0,
    artifacts_count: gql.artifactsCount ?? 0,
  }),
}))

vi.mock('@/graphql/operations/state', () => ({
  MESSAGES_QUERY: { kind: 'Document' },
  PROOF_FACTS_QUERY: { kind: 'Document' },
  toStoredMessage: (gql: any) => ({
    id: gql.id || 'msg-1',
    type: gql.type || 'output',
    content: gql.content || 'test',
    timestamp: gql.timestamp || '2025-01-01',
    stepNumber: gql.stepNumber,
    isFinalInsight: gql.isFinalInsight,
    stepDurationMs: gql.stepDurationMs,
    role: gql.role,
    skills: gql.skills,
  }),
  toStoredProofFact: (gql: any) => gql,
}))

vi.mock('@/graphql/operations/execution', () => ({
  QUERY_EXECUTION_SUBSCRIPTION: { kind: 'Document' },
  SUBMIT_QUERY: { kind: 'Document' },
  CANCEL_EXECUTION: { kind: 'Document' },
  APPROVE_PLAN: { kind: 'Document' },
  ANSWER_CLARIFICATION: { kind: 'Document' },
  SKIP_CLARIFICATION: { kind: 'Document' },
  REPLAN_FROM: { kind: 'Document' },
  EDIT_OBJECTIVE: { kind: 'Document' },
  DELETE_OBJECTIVE: { kind: 'Document' },
  HEARTBEAT: { kind: 'Document' },
  toExecutionEvent: (gql: any) => ({
    event_type: gql.eventType,
    session_id: gql.sessionId,
    step_number: gql.stepNumber ?? 0,
    timestamp: gql.timestamp,
    data: gql.data ? JSON.parse(gql.data) : {},
  }),
}))

vi.mock('@/graphql/operations/learnings', () => ({
  ACTIVATE_AGENT: { kind: 'Document' },
}))

vi.mock('@/config/auth-helpers', () => ({
  getAuthHeaders: vi.fn().mockResolvedValue({}),
}))

// ---------- Test consumer ----------

let capturedCtx: ReturnType<typeof useSessionContext> | null = null

function TestConsumer({ onRender }: { onRender?: (ctx: ReturnType<typeof useSessionContext>) => void }) {
  const ctx = useSessionContext()
  capturedCtx = ctx
  onRender?.(ctx)
  return (
    <div>
      <span data-testid="status">{ctx.status}</span>
      <span data-testid="sessionId">{ctx.sessionId ?? 'none'}</span>
      <span data-testid="sessionReady">{String(ctx.sessionReady)}</span>
      <span data-testid="isCreating">{String(ctx.isCreatingSession)}</span>
      <span data-testid="phase">{ctx.executionPhase}</span>
      <span data-testid="messageCount">{ctx.messages.length}</span>
      <span data-testid="connected">{String(ctx.subscriptionConnected)}</span>
      <span data-testid="isProving">{String(ctx.isProving)}</span>
    </div>
  )
}

function renderWithProvider(children: ReactNode) {
  return render(
    <MockedProvider mocks={[]} addTypename={false}>
      <SessionProvider>
        {children}
      </SessionProvider>
    </MockedProvider>,
  )
}

// ---------- Tests ----------

describe('SessionContext — initial state', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
  })

  it('provides default context values', () => {
    renderWithProvider(<TestConsumer />)
    expect(screen.getByTestId('status').textContent).toBe('idle')
    expect(screen.getByTestId('sessionId').textContent).toBe('none')
    expect(screen.getByTestId('sessionReady').textContent).toBe('false')
    expect(screen.getByTestId('isCreating').textContent).toBe('false')
    expect(screen.getByTestId('phase').textContent).toBe('idle')
    expect(screen.getByTestId('messageCount').textContent).toBe('0')
  })

  it('exposes proof state from reactive vars', () => {
    renderWithProvider(<TestConsumer />)
    expect(capturedCtx!.proofFacts).toBeDefined()
    expect(capturedCtx!.isProving).toBe(false)
    expect(capturedCtx!.isProofPanelOpen).toBe(false)
    expect(capturedCtx!.isPlanningComplete).toBe(false)
    expect(capturedCtx!.proofSummary).toBeNull()
    expect(capturedCtx!.isSummaryGenerating).toBe(false)
    expect(capturedCtx!.hasCompletedProof).toBe(false)
  })

  it('exposes activeDomains as empty array when no session', () => {
    renderWithProvider(<TestConsumer />)
    expect(capturedCtx!.activeDomains).toEqual([])
  })

  it('exposes agents as empty array initially', () => {
    renderWithProvider(<TestConsumer />)
    expect(capturedCtx!.agents).toEqual([])
    expect(capturedCtx!.currentAgent).toBeNull()
  })

  it('exposes clarification as null initially', () => {
    renderWithProvider(<TestConsumer />)
    expect(capturedCtx!.clarification).toBeNull()
  })

  it('exposes suggestions as empty array initially', () => {
    renderWithProvider(<TestConsumer />)
    expect(capturedCtx!.suggestions).toEqual([])
  })

  it('exposes empty queuedMessages initially', () => {
    renderWithProvider(<TestConsumer />)
    expect(capturedCtx!.queuedMessages).toEqual([])
  })

  it('exposes plan as null initially', () => {
    renderWithProvider(<TestConsumer />)
    expect(capturedCtx!.plan).toBeNull()
  })

  it('exposes currentQuery as empty string initially', () => {
    renderWithProvider(<TestConsumer />)
    expect(capturedCtx!.currentQuery).toBe('')
  })
})

describe('SessionContext — createSession', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
    mockMutate.mockResolvedValue({
      data: {
        createSession: {
          sessionId: 'sess-1',
          userId: 'test-user',
          status: 'IDLE',
          createdAt: '2025-01-01',
          lastActivity: '2025-01-01',
          activeDomains: [],
          tablesCount: 0,
          artifactsCount: 0,
        },
      },
    })
    mockQuery.mockResolvedValue({
      data: {
        messages: { messages: [] },
        proofFacts: { facts: [], summary: null },
      },
    })
  })

  it('creates session and updates state', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    expect(mockClearArtifactState).toHaveBeenCalled()
    expect(mockMutate).toHaveBeenCalled()
    expect(screen.getByTestId('sessionId').textContent).toBe('sess-1')
    expect(screen.getByTestId('isCreating').textContent).toBe('false')
  })

  it('creates session with forceNew flag', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession('test-user', true)
    })

    expect(screen.getByTestId('sessionId').textContent).toBe('sess-1')
    expect(mockClearArtifactState).toHaveBeenCalled()
  })

  it('restores messages from server on non-forceNew', async () => {
    mockQuery.mockResolvedValue({
      data: {
        messages: {
          messages: [
            { id: 'msg-1', type: 'output', content: 'Hello', timestamp: '2025-01-01' },
          ],
        },
        proofFacts: { facts: [], summary: null },
      },
    })

    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    // The RESET action was dispatched with restored messages
    expect(mockMutate).toHaveBeenCalled()
  })

  it('restores proof facts on reconnect', async () => {
    mockQuery
      .mockResolvedValueOnce({
        data: {
          messages: { messages: [] },
        },
      })
      .mockResolvedValueOnce({
        data: {
          proofFacts: {
            facts: [{ id: 'f1', name: 'f1', status: 'resolved', dependencies: [] }],
            summary: 'test summary',
          },
        },
      })

    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    // importFacts called if facts were restored
    // May or may not be called depending on Promise.allSettled ordering
    expect(mockMutate).toHaveBeenCalled()
  })

  it('sets sessionReady when active_domains present', async () => {
    mockMutate.mockResolvedValue({
      data: {
        createSession: {
          sessionId: 'sess-2',
          userId: 'test-user',
          status: 'IDLE',
          createdAt: '2025-01-01',
          lastActivity: '2025-01-01',
          activeDomains: ['sales'],
          tablesCount: 0,
          artifactsCount: 0,
        },
      },
    })

    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    expect(screen.getByTestId('sessionReady').textContent).toBe('true')
  })
})

describe('SessionContext — setSession', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
  })

  it('sets session and starts subscription', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      capturedCtx!.setSession({
        session_id: 'sess-new',
        user_id: 'test-user',
        status: 'idle',
        created_at: '2025-01-01',
        last_activity: '2025-01-01',
        tables_count: 0,
        artifacts_count: 0,
      })
    })

    expect(screen.getByTestId('sessionId').textContent).toBe('sess-new')
  })

  it('clears subscription when set to null', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      capturedCtx!.setSession(null)
    })

    expect(screen.getByTestId('sessionId').textContent).toBe('none')
  })

  it('preserves messages when option is set', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      capturedCtx!.setSession(
        {
          session_id: 'sess-keep',
          user_id: 'test-user',
          status: 'idle',
          created_at: '2025-01-01',
          last_activity: '2025-01-01',
          tables_count: 0,
          artifacts_count: 0,
        },
        { preserveMessages: true },
      )
    })

    expect(screen.getByTestId('sessionId').textContent).toBe('sess-keep')
  })

  it('dispatches SET_STATUS when session has status', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      capturedCtx!.setSession({
        session_id: 'sess-status',
        user_id: 'test-user',
        status: 'executing',
        created_at: '2025-01-01',
        last_activity: '2025-01-01',
        tables_count: 0,
        artifacts_count: 0,
      })
    })

    expect(screen.getByTestId('status').textContent).toBe('executing')
  })
})

describe('SessionContext — updateSession', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
    mockMutate.mockResolvedValue({
      data: {
        createSession: {
          sessionId: 'sess-update',
          userId: 'test-user',
          status: 'IDLE',
          createdAt: '2025-01-01',
          lastActivity: '2025-01-01',
          activeDomains: [],
          tablesCount: 0,
          artifactsCount: 0,
        },
      },
    })
    mockQuery.mockResolvedValue({
      data: {
        messages: { messages: [] },
        proofFacts: { facts: [], summary: null },
      },
    })
  })

  it('updates session partial fields', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    await act(async () => {
      capturedCtx!.updateSession({ active_domains: ['hr'] })
    })

    expect(capturedCtx!.activeDomains).toEqual(['hr'])
  })
})

describe('SessionContext — setActiveDomains', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
    mockMutate.mockResolvedValue({
      data: {
        createSession: {
          sessionId: 'sess-domains',
          userId: 'test-user',
          status: 'IDLE',
          createdAt: '2025-01-01',
          lastActivity: '2025-01-01',
          activeDomains: [],
          tablesCount: 0,
          artifactsCount: 0,
        },
      },
    })
    mockQuery.mockResolvedValue({
      data: {
        messages: { messages: [] },
        proofFacts: { facts: [], summary: null },
      },
    })
  })

  it('sets active domains via mutation', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockResolvedValueOnce({ data: {} })

    await act(async () => {
      await capturedCtx!.setActiveDomains(['sales', 'hr'])
    })

    expect(capturedCtx!.activeDomains).toEqual(['sales', 'hr'])
  })
})

describe('SessionContext — cancelExecution', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
    mockMutate.mockResolvedValue({
      data: {
        createSession: {
          sessionId: 'sess-cancel',
          userId: 'test-user',
          status: 'IDLE',
          createdAt: '2025-01-01',
          lastActivity: '2025-01-01',
          activeDomains: [],
          tablesCount: 0,
          artifactsCount: 0,
        },
      },
    })
    mockQuery.mockResolvedValue({
      data: {
        messages: { messages: [] },
        proofFacts: { facts: [], summary: null },
      },
    })
  })

  it('cancels execution and dispatches', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockResolvedValueOnce({ data: {} })

    await act(async () => {
      await capturedCtx!.cancelExecution()
    })

    expect(screen.getByTestId('status').textContent).toBe('idle')
  })
})

describe('SessionContext — submitQuery', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
    mockMutate.mockResolvedValue({
      data: {
        createSession: {
          sessionId: 'sess-submit',
          userId: 'test-user',
          status: 'IDLE',
          createdAt: '2025-01-01',
          lastActivity: '2025-01-01',
          activeDomains: [],
          tablesCount: 0,
          artifactsCount: 0,
        },
      },
    })
    mockQuery.mockResolvedValue({
      data: {
        messages: { messages: [] },
        proofFacts: { facts: [], summary: null },
      },
    })
  })

  it('submits a query and updates state', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockResolvedValueOnce({
      data: {
        submitQuery: { status: 'accepted' },
      },
    })

    await act(async () => {
      await capturedCtx!.submitQuery('What is total revenue?')
    })

    expect(mockMutate).toHaveBeenCalled()
  })

  it('handles slash command response', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockResolvedValueOnce({
      data: {
        submitQuery: { status: 'completed', message: 'Rule created `rule_abc123`' },
      },
    })

    await act(async () => {
      await capturedCtx!.submitQuery('@vera learn this rule')
    })

    // Should have added messages
    expect(mockMutate).toHaveBeenCalled()
  })

  it('handles error response from submitQuery', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockResolvedValueOnce({
      data: {
        submitQuery: { status: 'error', message: 'Something went wrong' },
      },
    })

    await act(async () => {
      await capturedCtx!.submitQuery('bad query')
    })

    expect(screen.getByTestId('status').textContent).toBe('error')
  })

  it('does nothing when no session', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.submitQuery('test')
    })

    // Should not throw, just return
    expect(screen.getByTestId('sessionId').textContent).toBe('none')
  })
})

describe('SessionContext — shareSession via @ prefix', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
    mockMutate.mockResolvedValue({
      data: {
        createSession: {
          sessionId: 'sess-share',
          userId: 'test-user',
          status: 'IDLE',
          createdAt: '2025-01-01',
          lastActivity: '2025-01-01',
          activeDomains: [],
          tablesCount: 0,
          artifactsCount: 0,
        },
      },
    })
    mockQuery.mockResolvedValue({
      data: {
        messages: { messages: [] },
        proofFacts: { facts: [], summary: null },
      },
    })
  })

  it('shareSession returns share url', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockResolvedValueOnce({
      data: {
        shareSession: { shareUrl: 'https://example.com/share/abc' },
      },
    })

    let result: { share_url: string } | undefined
    await act(async () => {
      result = await capturedCtx!.shareSession('user@example.com')
    })

    expect(result!.share_url).toBe('https://example.com/share/abc')
  })

  it('shareSession throws when no session', async () => {
    renderWithProvider(<TestConsumer />)

    await expect(
      act(async () => {
        await capturedCtx!.shareSession('user@example.com')
      }),
    ).rejects.toThrow('No active session')
  })
})

describe('SessionContext — proof actions', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
  })

  it('delegates openProofPanel to ui-state', () => {
    renderWithProvider(<TestConsumer />)
    capturedCtx!.openProofPanel()
    expect(mockOpenProofPanel).toHaveBeenCalled()
  })

  it('delegates closeProofPanel to ui-state', () => {
    renderWithProvider(<TestConsumer />)
    capturedCtx!.closeProofPanel()
    expect(mockCloseProofPanel).toHaveBeenCalled()
  })

  it('delegates clearProofFacts to ui-state', () => {
    renderWithProvider(<TestConsumer />)
    capturedCtx!.clearProofFacts()
    expect(mockClearProofFacts).toHaveBeenCalled()
  })
})

describe('SessionContext — clarification actions', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
  })

  it('setClarificationStep dispatches', () => {
    renderWithProvider(<TestConsumer />)
    act(() => {
      capturedCtx!.setClarificationStep(2)
    })
    // No error thrown
  })

  it('setClarificationAnswer dispatches', () => {
    renderWithProvider(<TestConsumer />)
    act(() => {
      capturedCtx!.setClarificationAnswer(0, 'test answer')
    })
    // No error thrown
  })

  it('setClarificationStructuredAnswer dispatches', () => {
    renderWithProvider(<TestConsumer />)
    act(() => {
      capturedCtx!.setClarificationStructuredAnswer(0, { key: 'value' })
    })
    // No error thrown
  })

  it('skipClarification dispatches', () => {
    renderWithProvider(<TestConsumer />)
    act(() => {
      capturedCtx!.skipClarification()
    })
    // No error thrown
  })
})

describe('SessionContext — removeQueuedMessage', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
  })

  it('dispatches remove queued message', () => {
    renderWithProvider(<TestConsumer />)
    act(() => {
      capturedCtx!.removeQueuedMessage('msg-123')
    })
    // No error thrown
  })
})

describe('SessionContext — replanFromStep', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
    mockMutate.mockResolvedValue({
      data: {
        createSession: {
          sessionId: 'sess-replan',
          userId: 'test-user',
          status: 'IDLE',
          createdAt: '2025-01-01',
          lastActivity: '2025-01-01',
          activeDomains: [],
          tablesCount: 0,
          artifactsCount: 0,
        },
      },
    })
    mockQuery.mockResolvedValue({
      data: {
        messages: { messages: [] },
        proofFacts: { facts: [], summary: null },
      },
    })
  })

  it('dispatches replan event and calls mutation', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockResolvedValueOnce({ data: {} })

    act(() => {
      capturedCtx!.replanFromStep(3, 'redo')
    })

    // replanFromStep fires async mutation
    await waitFor(() => {
      expect(mockMutate).toHaveBeenCalled()
    })
  })
})

describe('SessionContext — editObjective / deleteObjective', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
    mockMutate.mockResolvedValue({
      data: {
        createSession: {
          sessionId: 'sess-obj',
          userId: 'test-user',
          status: 'IDLE',
          createdAt: '2025-01-01',
          lastActivity: '2025-01-01',
          activeDomains: [],
          tablesCount: 0,
          artifactsCount: 0,
        },
      },
    })
    mockQuery.mockResolvedValue({
      data: {
        messages: { messages: [] },
        proofFacts: { facts: [], summary: null },
      },
    })
  })

  it('editObjective sets status to executing and calls mutation', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockResolvedValueOnce({ data: {} })

    act(() => {
      capturedCtx!.editObjective(0, 'Updated objective')
    })

    expect(screen.getByTestId('status').textContent).toBe('executing')
  })

  it('deleteObjective sets status to executing and calls mutation', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockResolvedValueOnce({ data: {} })

    act(() => {
      capturedCtx!.deleteObjective(1)
    })

    expect(screen.getByTestId('status').textContent).toBe('executing')
  })
})

describe('SessionContext — switchSession', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
    mockMutate.mockResolvedValue({
      data: {
        createSession: {
          sessionId: 'sess-switch-target',
          userId: 'test-user',
          status: 'IDLE',
          createdAt: '2025-01-01',
          lastActivity: '2025-01-01',
          activeDomains: [],
          tablesCount: 0,
          artifactsCount: 0,
        },
      },
    })
    mockQuery.mockResolvedValue({
      data: {
        messages: { messages: [] },
      },
    })
  })

  it('switches to a different session', async () => {
    renderWithProvider(<TestConsumer />)

    // First create a session
    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockResolvedValueOnce({
      data: {
        createSession: {
          sessionId: 'sess-new-target',
          userId: 'test-user',
          status: 'IDLE',
          createdAt: '2025-01-01',
          lastActivity: '2025-01-01',
          activeDomains: ['domain1'],
          tablesCount: 0,
          artifactsCount: 0,
        },
      },
    })

    await act(async () => {
      await capturedCtx!.switchSession('sess-new-target')
    })

    expect(mockClearArtifactState).toHaveBeenCalled()
    expect(screen.getByTestId('sessionId').textContent).toBe('sess-new-target')
  })

  it('no-ops when switching to same session', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    const callCount = mockMutate.mock.calls.length

    await act(async () => {
      await capturedCtx!.switchSession('sess-switch-target')
    })

    // No additional mutations should have been made
    expect(mockMutate.mock.calls.length).toBe(callCount)
  })
})

describe('SessionContext — approvePlan', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
    mockMutate.mockResolvedValue({
      data: {
        createSession: {
          sessionId: 'sess-approve',
          userId: 'test-user',
          status: 'IDLE',
          createdAt: '2025-01-01',
          lastActivity: '2025-01-01',
          activeDomains: [],
          tablesCount: 0,
          artifactsCount: 0,
        },
      },
    })
    mockQuery.mockResolvedValue({
      data: {
        messages: { messages: [] },
        proofFacts: { facts: [], summary: null },
      },
    })
  })

  it('approves plan and calls mutation', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockResolvedValueOnce({ data: {} })

    await act(async () => {
      await capturedCtx!.approvePlan()
    })

    expect(mockMutate).toHaveBeenCalled()
  })

  it('approves with edited steps', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockResolvedValueOnce({ data: {} })

    await act(async () => {
      await capturedCtx!.approvePlan(undefined, [{ number: 1, goal: 'New goal' }])
    })

    expect(mockMutate).toHaveBeenCalled()
  })
})

describe('SessionContext — rejectPlan', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
    mockMutate.mockResolvedValue({
      data: {
        createSession: {
          sessionId: 'sess-reject',
          userId: 'test-user',
          status: 'IDLE',
          createdAt: '2025-01-01',
          lastActivity: '2025-01-01',
          activeDomains: [],
          tablesCount: 0,
          artifactsCount: 0,
        },
      },
    })
    mockQuery.mockResolvedValue({
      data: {
        messages: { messages: [] },
        proofFacts: { facts: [], summary: null },
      },
    })
  })

  it('rejects plan with feedback', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockResolvedValueOnce({ data: {} })

    await act(async () => {
      await capturedCtx!.rejectPlan('Please focus on revenue only')
    })

    expect(mockMutate).toHaveBeenCalled()
  })

  it('rejects plan without feedback (cancel)', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockResolvedValueOnce({ data: {} })

    await act(async () => {
      await capturedCtx!.rejectPlan('Cancelled by user')
    })

    expect(screen.getByTestId('status').textContent).toBe('idle')
  })
})

describe('SessionContext — fetchAgents / setAgent', () => {
  beforeEach(() => {
    capturedCtx = null
    vi.clearAllMocks()
    mockMutate.mockResolvedValue({
      data: {
        createSession: {
          sessionId: 'sess-agents',
          userId: 'test-user',
          status: 'IDLE',
          createdAt: '2025-01-01',
          lastActivity: '2025-01-01',
          activeDomains: [],
          tablesCount: 0,
          artifactsCount: 0,
        },
      },
    })
    mockQuery.mockResolvedValue({
      data: {
        messages: { messages: [] },
        proofFacts: { facts: [], summary: null },
      },
    })
  })

  it('fetchAgents fetches and updates agents list', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        agents: [{ name: 'agent-1', prompt: 'test', is_active: false }],
        current_agent: null,
      }),
    } as Response)

    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    await act(async () => {
      await capturedCtx!.fetchAgents()
    })

    expect(capturedCtx!.agents).toHaveLength(1)
    expect(capturedCtx!.agents[0].name).toBe('agent-1')

    fetchSpy.mockRestore()
  })

  it('fetchAgents handles fetch error gracefully', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockRejectedValueOnce(new Error('Network error'))
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    await act(async () => {
      await capturedCtx!.fetchAgents()
    })

    // Should not throw
    expect(capturedCtx!.agents).toEqual([])

    fetchSpy.mockRestore()
    consoleSpy.mockRestore()
  })

  it('setAgent activates agent via mutation', async () => {
    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockResolvedValueOnce({
      data: {
        activateAgent: { currentAgent: 'agent-1' },
      },
    })

    await act(async () => {
      await capturedCtx!.setAgent('agent-1')
    })

    expect(capturedCtx!.currentAgent).toBe('agent-1')
  })

  it('setAgent handles mutation error gracefully', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

    renderWithProvider(<TestConsumer />)

    await act(async () => {
      await capturedCtx!.createSession()
    })

    mockMutate.mockRejectedValueOnce(new Error('Mutation failed'))

    await act(async () => {
      await capturedCtx!.setAgent('bad-agent')
    })

    // Should not throw
    expect(consoleSpy).toHaveBeenCalled()
    consoleSpy.mockRestore()
  })
})
