// Copyright (c) 2025 Kenneth Stott
// Canary: bda9eaa7-7e8e-43a4-8682-3517ff399b20
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { describe, it, expect, vi } from 'vitest'
import { MockedProvider, MockedResponse } from '@apollo/client/testing'


import { render, screen, waitFor } from '@testing-library/react'
import {
  CREATE_SESSION,
  SESSIONS_QUERY,
  DELETE_SESSION,
  SET_ACTIVE_DOMAINS,
  RESET_CONTEXT,
  toSession,
} from '@/graphql/operations/sessions'

// vi.mock is hoisted — factory must be self-contained
vi.mock('idb', () => ({
  openDB: vi.fn(() => Promise.resolve({
    get: vi.fn(() => undefined),
    put: vi.fn(),
    delete: vi.fn(),
  })),
}))

function TestComponent({ query, variables, field }: {
  query: any
  variables?: any
  field: string
}) {
  const { useQuery } = require('@apollo/client')
  const { loading, data, error } = useQuery(query, { variables })
  if (loading) return <div>Loading...</div>
  if (error) return <div>Error: {error.message}</div>
  return <div data-testid="result">{JSON.stringify(data?.[field])}</div>
}

function MutationComponent({ mutation, variables, field }: {
  mutation: any
  variables?: any
  field: string
}) {
  const { useMutation } = require('@apollo/client')
  const [mutate, { data, loading, error }] = useMutation(mutation)

  return (
    <div>
      <button onClick={() => mutate({ variables })} data-testid="trigger">
        Trigger
      </button>
      {loading && <div>Loading...</div>}
      {error && <div data-testid="error">Error: {error.message}</div>}
      {data && <div data-testid="result">{JSON.stringify(data?.[field])}</div>}
    </div>
  )
}

const mockSessionGql = {
  __typename: 'SessionType',
  sessionId: 'sess-1',
  userId: 'user-1',
  status: 'IDLE',
  createdAt: '2025-01-01T00:00:00Z',
  lastActivity: '2025-01-01T00:00:00Z',
  currentQuery: null,
  summary: null,
  activeDomains: [],
  tablesCount: 0,
  artifactsCount: 0,
  sharedWith: [],
  isPublic: false,
}

describe('toSession mapper', () => {
  it('maps camelCase GQL to snake_case Session', () => {
    const session = toSession(mockSessionGql)
    expect(session.session_id).toBe('sess-1')
    expect(session.user_id).toBe('user-1')
    expect(session.status).toBe('idle')
    expect(session.created_at).toBe('2025-01-01T00:00:00Z')
    expect(session.last_activity).toBe('2025-01-01T00:00:00Z')
    expect(session.tables_count).toBe(0)
    expect(session.artifacts_count).toBe(0)
    expect(session.active_domains).toEqual([])
    expect(session.shared_with).toEqual([])
    expect(session.is_public).toBe(false)
  })
})

describe('CREATE_SESSION mutation', () => {
  it('returns a session on success', async () => {
    const mocks: MockedResponse[] = [
      {
        request: { query: CREATE_SESSION },
        variableMatcher: (vars) => vars.userId === 'user-1' && vars.sessionId === 'sess-new',
        result: {
          data: {
            createSession: { ...mockSessionGql, sessionId: 'sess-new' },
          },
        },
      },
    ]

    const { getByTestId } = render(
      <MockedProvider mocks={mocks}>
        <MutationComponent
          mutation={CREATE_SESSION}
          variables={{ userId: 'user-1', sessionId: 'sess-new' }}
          field="createSession"
        />
      </MockedProvider>
    )

    getByTestId('trigger').click()

    await waitFor(() => {
      const result = JSON.parse(getByTestId('result').textContent!)
      expect(result.sessionId).toBe('sess-new')
      expect(result.userId).toBe('user-1')
    })
  })
})

describe('SESSIONS_QUERY', () => {
  it('returns sessions list', async () => {
    const mocks: MockedResponse[] = [
      {
        request: { query: SESSIONS_QUERY },
        result: {
          data: {
            sessions: {
              __typename: 'SessionListType',
              sessions: [mockSessionGql],
              total: 1,
            },
          },
        },
      },
    ]

    render(
      <MockedProvider mocks={mocks}>
        <TestComponent
          query={SESSIONS_QUERY}
          field="sessions"
        />
      </MockedProvider>
    )

    await waitFor(() => {
      const el = screen.getByTestId('result')
      const result = JSON.parse(el.textContent!)
      expect(result.sessions).toHaveLength(1)
      expect(result.sessions[0].sessionId).toBe('sess-1')
      expect(result.total).toBe(1)
    })
  })
})

describe('DELETE_SESSION mutation', () => {
  it('returns true on success', async () => {
    const mocks: MockedResponse[] = [
      {
        request: { query: DELETE_SESSION },
        variableMatcher: (vars) => vars.sessionId === 'sess-1',
        result: {
          data: { deleteSession: true },
        },
      },
    ]

    const { getByTestId } = render(
      <MockedProvider mocks={mocks}>
        <MutationComponent
          mutation={DELETE_SESSION}
          variables={{ sessionId: 'sess-1' }}
          field="deleteSession"
        />
      </MockedProvider>
    )

    getByTestId('trigger').click()

    await waitFor(() => {
      expect(getByTestId('result').textContent).toBe('true')
    })
  })
})

describe('SET_ACTIVE_DOMAINS mutation', () => {
  it('returns domain list on success', async () => {
    const mocks: MockedResponse[] = [
      {
        request: { query: SET_ACTIVE_DOMAINS },
        variableMatcher: (vars) =>
          vars.sessionId === 'sess-1' &&
          JSON.stringify(vars.domains) === JSON.stringify(['sales', 'hr']),
        result: {
          data: { setActiveDomains: ['sales', 'hr'] },
        },
      },
    ]

    const { getByTestId } = render(
      <MockedProvider mocks={mocks}>
        <MutationComponent
          mutation={SET_ACTIVE_DOMAINS}
          variables={{ sessionId: 'sess-1', domains: ['sales', 'hr'] }}
          field="setActiveDomains"
        />
      </MockedProvider>
    )

    getByTestId('trigger').click()

    await waitFor(() => {
      const result = JSON.parse(getByTestId('result').textContent!)
      expect(result).toEqual(['sales', 'hr'])
    })
  })
})

describe('RESET_CONTEXT mutation', () => {
  it('returns true on success', async () => {
    const mocks: MockedResponse[] = [
      {
        request: { query: RESET_CONTEXT },
        variableMatcher: (vars) => vars.sessionId === 'sess-1',
        result: {
          data: { resetContext: true },
        },
      },
    ]

    const { getByTestId } = render(
      <MockedProvider mocks={mocks}>
        <MutationComponent
          mutation={RESET_CONTEXT}
          variables={{ sessionId: 'sess-1' }}
          field="resetContext"
        />
      </MockedProvider>
    )

    getByTestId('trigger').click()

    await waitFor(() => {
      expect(getByTestId('result').textContent).toBe('true')
    })
  })
})
