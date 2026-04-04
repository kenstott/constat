// Copyright (c) 2025 Kenneth Stott
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MockedProvider } from '@apollo/client/testing'
import { LoginPage } from '../LoginPage'

// Mock firebase module
vi.mock('@/config/firebase', () => ({
  isAuthDisabled: false,
  subscribeToAuthChanges: vi.fn((cb: (user: null) => void) => {
    cb(null)
    return vi.fn()
  }),
  signInWithGoogle: vi.fn(),
  signInWithEmail: vi.fn(),
  signUpWithEmail: vi.fn(),
  resetPassword: vi.fn(),
  sendEmailLink: vi.fn(),
  checkEmailLink: vi.fn(() => false),
  completeEmailLinkSignIn: vi.fn(),
  logOut: vi.fn(),
  getIdToken: vi.fn(),
}))

// Mock graphql client
vi.mock('@/graphql/client', () => ({
  apolloClient: {
    mutate: vi.fn(),
  },
}))

// Mock AuthContext
const mockAuth = {
  login: vi.fn(),
  register: vi.fn(),
  loginWithGoogle: vi.fn(),
  loginWithMicrosoft: vi.fn(),
  loginWithEmail: vi.fn(),
  signupWithEmail: vi.fn(),
  sendPasswordReset: vi.fn(),
  sendEmailSignInLink: vi.fn(),
  completeEmailLink: vi.fn(),
  isEmailLinkSignIn: vi.fn(() => false),
  loading: false,
  error: null,
  setError: vi.fn(),
  clearError: vi.fn(),
  user: null,
  token: null,
  isAuthenticated: false,
  isAuthDisabled: false,
  userId: '',
  isAdmin: false,
  permissions: null,
  initialized: true,
  logout: vi.fn(),
  canSee: vi.fn(() => true),
  canWrite: vi.fn(() => true),
}

vi.mock('@/contexts/AuthContext', () => ({
  useAuth: () => mockAuth,
}))

let originalPublicKeyCredential: typeof window.PublicKeyCredential | undefined

beforeEach(() => {
  originalPublicKeyCredential = window.PublicKeyCredential
  // Mock fetch for /health endpoint
  global.fetch = vi.fn().mockResolvedValue({
    json: () => Promise.resolve({ auth: { auth_methods: ['local'] } }),
  })
})

afterEach(() => {
  if (originalPublicKeyCredential !== undefined) {
    Object.defineProperty(window, 'PublicKeyCredential', {
      value: originalPublicKeyCredential,
      writable: true,
      configurable: true,
    })
  } else {
    // Remove the property if it didn't exist before
    delete (window as unknown as Record<string, unknown>).PublicKeyCredential
  }
  vi.restoreAllMocks()
})

describe('LoginPage passkey button', () => {
  it('shows passkey button when PublicKeyCredential is available', async () => {
    Object.defineProperty(window, 'PublicKeyCredential', {
      value: class MockPKC {},
      writable: true,
      configurable: true,
    })

    render(
      <MockedProvider mocks={[]}>
        <LoginPage />
      </MockedProvider>,
    )

    await waitFor(() => {
      expect(screen.getByText('Sign in with Passkey')).toBeDefined()
    })
  })

  it('hides passkey button when PublicKeyCredential is not available', async () => {
    delete (window as unknown as Record<string, unknown>).PublicKeyCredential

    render(
      <MockedProvider mocks={[]}>
        <LoginPage />
      </MockedProvider>,
    )

    // Wait for health check to resolve
    await waitFor(() => {
      expect(screen.getByText('Sign in to your account')).toBeDefined()
    })

    expect(screen.queryByText('Sign in with Passkey')).toBeNull()
  })
})
