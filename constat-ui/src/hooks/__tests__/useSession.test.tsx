// Copyright (c) 2025 Kenneth Stott
// Canary: 25e07178-5ee1-479e-bd20-f94cffde1cf2
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { describe, it, expect, vi } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { MockedProvider, type MockedResponse } from '@apollo/client/testing'
import type { ReactNode } from 'react'
import { useSession } from '../useSession'
import { SESSION_QUERY } from '@/graphql/operations/sessions'

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
  summary: 'Test session',
  activeDomains: ['sales'],
  tablesCount: 3,
  artifactsCount: 1,
  sharedWith: [],
  isPublic: false,
}

describe('useSession', () => {
  it('returns session data for a valid sessionId', async () => {
    const mocks: MockedResponse[] = [
      {
        request: { query: SESSION_QUERY, variables: { sessionId: 'sess-1' } },
        result: { data: { session: mockSessionGql } },
      },
    ]

    const wrapper = ({ children }: { children: ReactNode }) => (
      <MockedProvider mocks={mocks} addTypename={false}>
        {children}
      </MockedProvider>
    )

    const { result } = renderHook(() => useSession('sess-1'), { wrapper })

    await waitFor(() => {
      expect(result.current.loading).toBe(false)
    })

    expect(result.current.session).not.toBeNull()
    expect(result.current.session!.session_id).toBe('sess-1')
    expect(result.current.session!.summary).toBe('Test session')
    expect(result.current.session!.active_domains).toEqual(['sales'])
    expect(result.current.session!.tables_count).toBe(3)
  })

  it('skips query when sessionId is null', () => {
    const wrapper = ({ children }: { children: ReactNode }) => (
      <MockedProvider mocks={[]} addTypename={false}>
        {children}
      </MockedProvider>
    )

    const { result } = renderHook(() => useSession(null), { wrapper })

    expect(result.current.loading).toBe(false)
    expect(result.current.session).toBeNull()
  })
})
