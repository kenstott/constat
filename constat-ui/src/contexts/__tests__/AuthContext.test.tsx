// Copyright (c) 2025 Kenneth Stott
// Canary: 3441eabb-cfce-4c1a-97d9-24d5dfcb0b30
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, act, waitFor } from '@testing-library/react'
import { MockedProvider } from '@apollo/client/testing'
import { AuthProvider, useAuth, isAuthDisabledVar } from '../AuthContext'
import { LOGIN_MUTATION, MY_PERMISSIONS_QUERY } from '@/graphql/operations/auth'

// Mock firebase module
const mockSignInWithGoogle = vi.fn()
const mockSignInWithEmail = vi.fn()
const mockSignUpWithEmail = vi.fn()
const mockResetPassword = vi.fn()
const mockSendEmailLink = vi.fn()
const mockCheckEmailLink = vi.fn(() => false)
const mockCompleteEmailLinkSignIn = vi.fn()
const mockLogOut = vi.fn()

vi.mock('@/config/firebase', () => ({
  isAuthDisabled: true,
  subscribeToAuthChanges: vi.fn(() => vi.fn()),
  signInWithGoogle: (...args: unknown[]) => mockSignInWithGoogle(...args),
  signInWithEmail: (...args: unknown[]) => mockSignInWithEmail(...args),
  signUpWithEmail: (...args: unknown[]) => mockSignUpWithEmail(...args),
  resetPassword: (...args: unknown[]) => mockResetPassword(...args),
  sendEmailLink: (...args: unknown[]) => mockSendEmailLink(...args),
  checkEmailLink: () => mockCheckEmailLink(),
  completeEmailLinkSignIn: (...args: unknown[]) => mockCompleteEmailLinkSignIn(...args),
  logOut: (...args: unknown[]) => mockLogOut(...args),
  getIdToken: vi.fn(),
}))

function TestConsumer() {
  const auth = useAuth()
  return (
    <div>
      <span data-testid="authenticated">{String(auth.isAuthenticated)}</span>
      <span data-testid="userId">{auth.userId}</span>
      <span data-testid="isAdmin">{String(auth.isAdmin)}</span>
      <span data-testid="persona">{auth.permissions?.persona ?? 'none'}</span>
      <span data-testid="canSee-glossary">{String(auth.canSee('glossary'))}</span>
      <span data-testid="error">{auth.error ?? ''}</span>
      <span data-testid="initialized">{String(auth.initialized)}</span>
      <button data-testid="login-btn" onClick={() => auth.login('test@example.com', 'pass')}>Login</button>
      <button data-testid="google-btn" onClick={() => auth.loginWithGoogle()}>Google</button>
      <button data-testid="email-btn" onClick={() => auth.loginWithEmail('a@b.com', 'pw')}>Email</button>
      <button data-testid="signup-btn" onClick={() => auth.signupWithEmail('a@b.com', 'pw')}>Signup</button>
      <button data-testid="reset-btn" onClick={() => auth.sendPasswordReset('a@b.com')}>Reset</button>
      <button data-testid="link-btn" onClick={() => auth.sendEmailSignInLink('a@b.com')}>SendLink</button>
      <button data-testid="set-error-btn" onClick={() => auth.setError('custom error')}>SetError</button>
    </div>
  )
}

describe('AuthContext', () => {
  beforeEach(() => {
    isAuthDisabledVar(true)
  })

  it('provides auth values when auth disabled', () => {
    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      </MockedProvider>
    )
    expect(screen.getByTestId('authenticated').textContent).toBe('true')
    expect(screen.getByTestId('userId').textContent).toBe('default')
    expect(screen.getByTestId('isAdmin').textContent).toBe('true')
  })

  it('throws when useAuth used outside provider', () => {
    // Suppress React error boundary console output
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    expect(() => render(<TestConsumer />)).toThrow('useAuth must be used within AuthProvider')
    spy.mockRestore()
  })

  it('login calls GraphQL mutation', async () => {
    const loginMock = {
      request: {
        query: LOGIN_MUTATION,
        variables: { email: 'test@example.com', password: 'pass' },
      },
      result: {
        data: {
          login: { token: 'tok-123', userId: 'u1', email: 'test@example.com' },
        },
      },
    }

    render(
      <MockedProvider mocks={[loginMock]} addTypename={false}>
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      screen.getByTestId('login-btn').click()
    })

    await waitFor(() => {
      expect(screen.getByTestId('authenticated').textContent).toBe('true')
    })
  })

  it('initialized starts true when auth disabled', () => {
    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      </MockedProvider>
    )
    expect(screen.getByTestId('initialized').textContent).toBe('true')
  })

  it('setError updates context error', async () => {
    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      </MockedProvider>
    )
    expect(screen.getByTestId('error').textContent).toBe('')

    await act(async () => {
      screen.getByTestId('set-error-btn').click()
    })

    expect(screen.getByTestId('error').textContent).toBe('custom error')
  })

  it('loginWithGoogle calls firebase signInWithGoogle', async () => {
    isAuthDisabledVar(false)
    mockSignInWithGoogle.mockResolvedValue(null)
    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      screen.getByTestId('google-btn').click()
    })

    expect(mockSignInWithGoogle).toHaveBeenCalled()
    isAuthDisabledVar(true)
  })

  it('signupWithEmail calls firebase signUpWithEmail', async () => {
    isAuthDisabledVar(false)
    mockSignUpWithEmail.mockResolvedValue({ emailVerified: false })
    mockLogOut.mockResolvedValue(undefined)
    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      screen.getByTestId('signup-btn').click()
    })

    expect(mockSignUpWithEmail).toHaveBeenCalledWith('a@b.com', 'pw')
    isAuthDisabledVar(true)
  })

  it('sendPasswordReset calls firebase resetPassword', async () => {
    isAuthDisabledVar(false)
    mockResetPassword.mockResolvedValue(undefined)
    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      screen.getByTestId('reset-btn').click()
    })

    expect(mockResetPassword).toHaveBeenCalledWith('a@b.com')
    isAuthDisabledVar(true)
  })

  it('sendEmailSignInLink calls firebase sendEmailLink', async () => {
    isAuthDisabledVar(false)
    mockSendEmailLink.mockResolvedValue(undefined)
    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      screen.getByTestId('link-btn').click()
    })

    expect(mockSendEmailLink).toHaveBeenCalledWith('a@b.com')
    isAuthDisabledVar(true)
  })

  it('permissions loaded on mount (auth disabled)', async () => {
    const permsMock = {
      request: {
        query: MY_PERMISSIONS_QUERY,
      },
      result: {
        data: {
          myPermissions: {
            userId: 'default',
            email: null,
            admin: true,
            persona: 'platform_admin',
            domains: ['all'],
            databases: ['db1'],
            documents: [],
            apis: [],
            visibility: { glossary: true },
            writes: { glossary: true },
            feedback: { flag_answers: true },
          },
        },
      },
    }

    render(
      <MockedProvider mocks={[permsMock]} addTypename={false}>
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      </MockedProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('persona').textContent).toBe('platform_admin')
    })
  })
})
