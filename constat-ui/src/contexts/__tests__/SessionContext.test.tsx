// Copyright (c) 2025 Kenneth Stott
// Canary: e5ca3891-66f4-479f-82f9-0916b422e162
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MockedProvider } from '@apollo/client/testing'
import { SessionProvider, useSessionContext } from '../SessionContext'

// Mock auth context
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

// Mock stores used internally
vi.mock('@/store/proofStore', () => ({
  useProofStore: (selector: (s: unknown) => unknown) => {
    const state = {
      facts: new Map(),
      isPanelOpen: false,
      isPlanningComplete: false,
      proofSummary: null,
      isSummaryGenerating: false,
      isProving: false,
      hasCompletedProof: false,
      openPanel: vi.fn(),
      closePanel: vi.fn(),
      clearFacts: vi.fn(),
      importFacts: vi.fn(),
    }
    return selector ? selector(state) : state
  },
}))

vi.mock('@/store/artifactStore', () => ({
  useArtifactStore: Object.assign(
    () => ({}),
    { getState: () => ({ clear: vi.fn(), fetchTables: vi.fn(), fetchArtifacts: vi.fn() }) },
  ),
}))

vi.mock('@/store/glossaryStore', () => ({
  useGlossaryStore: Object.assign(
    () => ({}),
    { getState: () => ({ fetchTerms: vi.fn(), loadFromCache: vi.fn() }) },
  ),
}))

vi.mock('@/graphql/client', () => ({
  apolloClient: {
    query: vi.fn().mockResolvedValue({ data: {} }),
    mutate: vi.fn().mockResolvedValue({ data: {} }),
    subscribe: vi.fn().mockReturnValue({ subscribe: vi.fn().mockReturnValue({ unsubscribe: vi.fn() }) }),
  },
}))

vi.mock('@/graphql/ui-state', () => ({
  briefModeVar: () => false,
}))

vi.mock('@/api/session-id', () => ({
  getOrCreateSessionId: vi.fn().mockReturnValue('test-session-id'),
  createNewSessionId: vi.fn().mockReturnValue('new-session-id'),
}))

function TestConsumer() {
  const ctx = useSessionContext()
  return (
    <div>
      <span data-testid="hasContext">{String(!!ctx)}</span>
      <span data-testid="status">{ctx.status}</span>
    </div>
  )
}

describe('SessionContext', () => {
  it('provides initial state', () => {
    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <SessionProvider>
          <TestConsumer />
        </SessionProvider>
      </MockedProvider>,
    )
    expect(screen.getByTestId('hasContext').textContent).toBe('true')
    expect(screen.getByTestId('status').textContent).toBe('idle')
  })

  it('throws when useSessionContext used outside provider', () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(() => render(<TestConsumer />)).toThrow('useSessionContext must be used within SessionProvider')
    spy.mockRestore()
  })
})
