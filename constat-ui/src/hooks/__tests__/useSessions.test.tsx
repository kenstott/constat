// Copyright (c) 2025 Kenneth Stott
// Canary: cdeaf47e-da21-44c1-a48a-61cdeb6d1c18
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { describe, it, expect, vi } from 'vitest'
import { renderHook, waitFor, act } from '@testing-library/react'
import { MockedProvider, type MockedResponse } from '@apollo/client/testing'
import type { ReactNode } from 'react'
import { useSessions } from '../useSessions'
import { SESSIONS_QUERY, CREATE_SESSION, DELETE_SESSION } from '@/graphql/operations/sessions'

vi.mock('idb', () => ({
  openDB: vi.fn(() => Promise.resolve({
    get: vi.fn(() => undefined),
    put: vi.fn(),
    delete: vi.fn(),
  })),
}))

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

describe('useSessions', () => {
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

    const wrapper = ({ children }: { children: ReactNode }) => (
      <MockedProvider mocks={mocks} addTypename={false}>
        {children}
      </MockedProvider>
    )

    const { result } = renderHook(() => useSessions(), { wrapper })

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })

    expect(result.current.sessions).toHaveLength(1)
    expect(result.current.sessions[0].session_id).toBe('sess-1')
    expect(result.current.total).toBe(1)
  })

  it('createSession calls mutation and returns session', async () => {
    const mocks: MockedResponse[] = [
      {
        request: { query: SESSIONS_QUERY },
        result: {
          data: {
            sessions: { __typename: 'SessionListType', sessions: [], total: 0 },
          },
        },
      },
      {
        request: { query: CREATE_SESSION },
        variableMatcher: (vars) => vars.sessionId === 'new-sess' && vars.userId === 'user-1',
        result: {
          data: {
            createSession: { ...mockSessionGql, sessionId: 'new-sess' },
          },
        },
      },
      {
        request: { query: SESSIONS_QUERY },
        result: {
          data: {
            sessions: {
              __typename: 'SessionListType',
              sessions: [{ ...mockSessionGql, sessionId: 'new-sess' }],
              total: 1,
            },
          },
        },
      },
    ]

    const wrapper = ({ children }: { children: ReactNode }) => (
      <MockedProvider mocks={mocks} addTypename={false}>
        {children}
      </MockedProvider>
    )

    const { result } = renderHook(() => useSessions(), { wrapper })

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })

    let created: Awaited<ReturnType<typeof result.current.createSession>> | undefined
    await act(async () => {
      created = await result.current.createSession('new-sess', 'user-1')
    })

    expect(created!.session_id).toBe('new-sess')
  })

  it('deleteSession calls mutation and returns true', async () => {
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
      {
        request: { query: DELETE_SESSION },
        variableMatcher: (vars) => vars.sessionId === 'sess-1',
        result: { data: { deleteSession: true } },
      },
      {
        request: { query: SESSIONS_QUERY },
        result: {
          data: {
            sessions: { __typename: 'SessionListType', sessions: [], total: 0 },
          },
        },
      },
    ]

    const wrapper = ({ children }: { children: ReactNode }) => (
      <MockedProvider mocks={mocks} addTypename={false}>
        {children}
      </MockedProvider>
    )

    const { result } = renderHook(() => useSessions(), { wrapper })

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })

    let deleted: boolean | undefined
    await act(async () => {
      deleted = await result.current.deleteSession('sess-1')
    })

    expect(deleted).toBe(true)
  })
})
