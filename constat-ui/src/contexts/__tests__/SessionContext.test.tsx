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

vi.mock('@/store/glossaryState', () => ({
  fetchTerms: vi.fn(),
  loadFromCache: vi.fn(),
}))

vi.mock('@/graphql/client', () => ({
  apolloClient: {
    query: vi.fn().mockResolvedValue({ data: {} }),
    mutate: vi.fn().mockResolvedValue({ data: {} }),
    subscribe: vi.fn().mockReturnValue({ subscribe: vi.fn().mockReturnValue({ unsubscribe: vi.fn() }) }),
    refetchQueries: vi.fn().mockResolvedValue([]),
  },
}))

vi.mock('@/graphql/ui-state', async () => {
  const { makeVar } = await import('@apollo/client')
  return {
    briefModeVar: makeVar(false),
    clearArtifactState: vi.fn(),
    markStepsSuperseded: vi.fn(),
    proofFactsVar: makeVar(new Map()),
    isProvingVar: makeVar(false),
    isPlanningCompleteVar: makeVar(false),
    isProofPanelOpenVar: makeVar(false),
    proofSummaryVar: makeVar(null),
    isSummaryGeneratingVar: makeVar(false),
    hasCompletedProofVar: makeVar(false),
    openProofPanel: vi.fn(),
    closeProofPanel: vi.fn(),
    clearProofFacts: vi.fn(),
    importFacts: vi.fn(),
  }
})

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
