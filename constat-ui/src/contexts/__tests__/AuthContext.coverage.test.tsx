import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, act, waitFor } from '@testing-library/react'
import { MockedProvider } from '@apollo/client/testing'
import { AuthProvider, useAuth, isAuthDisabledVar } from '../AuthContext'
import { LOGIN_MUTATION, LOGOUT_MUTATION, MY_PERMISSIONS_QUERY } from '@/graphql/operations/auth'

// Mock firebase module
const mockSignInWithGoogle = vi.fn()
const mockSignInWithEmail = vi.fn()
const mockSignUpWithEmail = vi.fn()
const mockResetPassword = vi.fn()
const mockSendEmailLink = vi.fn()
const mockCheckEmailLink = vi.fn(() => false)
const mockCompleteEmailLinkSignIn = vi.fn()
const mockLogOut = vi.fn()
const mockGetIdToken = vi.fn()
const mockSubscribeToAuthChanges = vi.fn((_cb: any) => vi.fn())

vi.mock('@/config/firebase', () => ({
  isAuthDisabled: true,
  subscribeToAuthChanges: (cb: any) => mockSubscribeToAuthChanges(cb),
  signInWithGoogle: (...args: any[]) => mockSignInWithGoogle(...args),
  signInWithEmail: (...args: any[]) => mockSignInWithEmail(...args),
  signUpWithEmail: (...args: any[]) => mockSignUpWithEmail(...args),
  resetPassword: (...args: any[]) => mockResetPassword(...args),
  sendEmailLink: (...args: any[]) => mockSendEmailLink(...args),
  checkEmailLink: () => mockCheckEmailLink(),
  completeEmailLinkSignIn: (...args: any[]) => mockCompleteEmailLinkSignIn(...args),
  logOut: (...args: any[]) => mockLogOut(...args),
  getIdToken: (...args: any[]) => mockGetIdToken(...args),
}))

// Mock apollo client for register / microsoft
vi.mock('@/graphql/client', () => ({
  apolloClient: {
    mutate: vi.fn(),
  },
}))

function TestConsumer({ onAuth }: { onAuth?: (auth: ReturnType<typeof useAuth>) => void }) {
  const auth = useAuth()
  if (onAuth) onAuth(auth)
  return (
    <div>
      <span data-testid="authenticated">{String(auth.isAuthenticated)}</span>
      <span data-testid="userId">{auth.userId}</span>
      <span data-testid="isAdmin">{String(auth.isAdmin)}</span>
      <span data-testid="loading">{String(auth.loading)}</span>
      <span data-testid="error">{auth.error ?? ''}</span>
      <span data-testid="initialized">{String(auth.initialized)}</span>
      <span data-testid="canSee">{String(auth.canSee('dashboard'))}</span>
      <span data-testid="canWrite">{String(auth.canWrite('glossary'))}</span>
    </div>
  )
}

describe('AuthContext coverage - login error handling', () => {
  beforeEach(() => {
    isAuthDisabledVar(true)
    vi.clearAllMocks()
  })

  it('login sets error on mutation failure', async () => {
    const loginMock = {
      request: {
        query: LOGIN_MUTATION,
        variables: { email: 'bad@test.com', password: 'wrong' },
      },
      error: new Error('Invalid credentials'),
    }

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[loginMock]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      try {
        await authRef!.login('bad@test.com', 'wrong')
      } catch {
        // expected
      }
    })

    await waitFor(() => {
      expect(screen.getByTestId('error').textContent).toBeTruthy()
    })
  })

  it('register calls apollo client mutate', async () => {
    const { apolloClient } = await import('@/graphql/client')
    ;(apolloClient.mutate as any).mockResolvedValue({
      data: { register: { token: 'reg-tok', userId: 'u2', email: 'new@test.com' } },
    })

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      await authRef!.register('newuser', 'password123', 'new@test.com')
    })

    expect(apolloClient.mutate).toHaveBeenCalled()
  })

  it('register sets error on failure', async () => {
    const { apolloClient } = await import('@/graphql/client')
    ;(apolloClient.mutate as any).mockRejectedValue(new Error('Username taken'))

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      try {
        await authRef!.register('taken', 'pw')
      } catch {
        // expected
      }
    })

    await waitFor(() => {
      expect(screen.getByTestId('error').textContent).toBe('Username taken')
    })
  })
})

describe('AuthContext coverage - loginWithGoogle error', () => {
  beforeEach(() => {
    isAuthDisabledVar(false)
    vi.clearAllMocks()
  })

  afterEach(() => {
    isAuthDisabledVar(true)
  })

  it('loginWithGoogle sets error on failure', async () => {
    mockSignInWithGoogle.mockRejectedValue(new Error('Popup blocked'))

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      try {
        await authRef!.loginWithGoogle()
      } catch {
        // expected
      }
    })

    await waitFor(() => {
      expect(screen.getByTestId('error').textContent).toBe('Popup blocked')
    })
  })

  it('loginWithGoogle is noop when auth disabled', async () => {
    isAuthDisabledVar(true)

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      await authRef!.loginWithGoogle()
    })

    expect(mockSignInWithGoogle).not.toHaveBeenCalled()
  })
})

describe('AuthContext coverage - loginWithEmail', () => {
  beforeEach(() => {
    isAuthDisabledVar(false)
    vi.clearAllMocks()
  })

  afterEach(() => {
    isAuthDisabledVar(true)
  })

  it('loginWithEmail sets error if email not verified', async () => {
    mockSignInWithEmail.mockResolvedValue({ emailVerified: false })
    mockLogOut.mockResolvedValue(undefined)

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      await authRef!.loginWithEmail('unverified@test.com', 'pw')
    })

    await waitFor(() => {
      expect(screen.getByTestId('error').textContent).toContain('verify your email')
    })
    expect(mockLogOut).toHaveBeenCalled()
  })

  it('loginWithEmail sets error on failure', async () => {
    mockSignInWithEmail.mockRejectedValue(new Error('Firebase: Wrong password (auth/wrong-password).'))

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      try {
        await authRef!.loginWithEmail('a@b.com', 'bad')
      } catch {
        // expected
      }
    })

    await waitFor(() => {
      expect(screen.getByTestId('error').textContent).toBe('Wrong password')
    })
  })

  it('loginWithEmail is noop when auth disabled', async () => {
    isAuthDisabledVar(true)

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      await authRef!.loginWithEmail('a@b.com', 'pw')
    })

    expect(mockSignInWithEmail).not.toHaveBeenCalled()
  })
})

describe('AuthContext coverage - signupWithEmail error', () => {
  beforeEach(() => {
    isAuthDisabledVar(false)
    vi.clearAllMocks()
  })

  afterEach(() => {
    isAuthDisabledVar(true)
  })

  it('signupWithEmail sets error on failure', async () => {
    mockSignUpWithEmail.mockRejectedValue(new Error('Firebase: Email already in use (auth/email-already-in-use).'))

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      try {
        await authRef!.signupWithEmail('taken@test.com', 'pw')
      } catch {
        // expected
      }
    })

    await waitFor(() => {
      expect(screen.getByTestId('error').textContent).toBe('Email already in use')
    })
  })

  it('signupWithEmail is noop when auth disabled', async () => {
    isAuthDisabledVar(true)

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      await authRef!.signupWithEmail('a@b.com', 'pw')
    })

    expect(mockSignUpWithEmail).not.toHaveBeenCalled()
  })
})

describe('AuthContext coverage - sendPasswordReset error', () => {
  beforeEach(() => {
    isAuthDisabledVar(false)
    vi.clearAllMocks()
  })

  afterEach(() => {
    isAuthDisabledVar(true)
  })

  it('sendPasswordReset sets error on failure', async () => {
    mockResetPassword.mockRejectedValue(new Error('Firebase: User not found (auth/user-not-found).'))

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      try {
        await authRef!.sendPasswordReset('missing@test.com')
      } catch {
        // expected
      }
    })

    await waitFor(() => {
      expect(screen.getByTestId('error').textContent).toBe('User not found')
    })
  })

  it('sendPasswordReset is noop when auth disabled', async () => {
    isAuthDisabledVar(true)

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      await authRef!.sendPasswordReset('a@b.com')
    })

    expect(mockResetPassword).not.toHaveBeenCalled()
  })
})

describe('AuthContext coverage - sendEmailSignInLink error', () => {
  beforeEach(() => {
    isAuthDisabledVar(false)
    vi.clearAllMocks()
  })

  afterEach(() => {
    isAuthDisabledVar(true)
  })

  it('sendEmailSignInLink sets error on failure', async () => {
    mockSendEmailLink.mockRejectedValue(new Error('Firebase: Too many requests (auth/too-many-requests).'))

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      try {
        await authRef!.sendEmailSignInLink('a@b.com')
      } catch {
        // expected
      }
    })

    await waitFor(() => {
      expect(screen.getByTestId('error').textContent).toBe('Too many requests')
    })
  })

  it('sendEmailSignInLink is noop when auth disabled', async () => {
    isAuthDisabledVar(true)

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      await authRef!.sendEmailSignInLink('a@b.com')
    })

    expect(mockSendEmailLink).not.toHaveBeenCalled()
  })
})

describe('AuthContext coverage - completeEmailLink', () => {
  beforeEach(() => {
    isAuthDisabledVar(false)
    vi.clearAllMocks()
  })

  afterEach(() => {
    isAuthDisabledVar(true)
  })

  it('completeEmailLink sets error on failure', async () => {
    mockCompleteEmailLinkSignIn.mockRejectedValue(new Error('Firebase: Invalid link (auth/invalid-action-code).'))

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      try {
        await authRef!.completeEmailLink('a@b.com')
      } catch {
        // expected
      }
    })

    await waitFor(() => {
      expect(screen.getByTestId('error').textContent).toBe('Invalid link')
    })
  })

  it('completeEmailLink is noop when auth disabled', async () => {
    isAuthDisabledVar(true)

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      await authRef!.completeEmailLink('a@b.com')
    })

    expect(mockCompleteEmailLinkSignIn).not.toHaveBeenCalled()
  })
})

describe('AuthContext coverage - isEmailLinkSignIn', () => {
  it('returns false when auth disabled', () => {
    isAuthDisabledVar(true)

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    expect(authRef!.isEmailLinkSignIn()).toBe(false)
  })

  it('delegates to checkEmailLink when auth enabled', () => {
    isAuthDisabledVar(false)
    mockCheckEmailLink.mockReturnValue(true)

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    expect(authRef!.isEmailLinkSignIn()).toBe(true)
    isAuthDisabledVar(true)
  })
})

describe('AuthContext coverage - logout', () => {
  beforeEach(() => {
    isAuthDisabledVar(true)
    vi.clearAllMocks()
  })

  it('logout calls logoutMutation and clears state', async () => {
    const logoutMock = {
      request: { query: LOGOUT_MUTATION },
      result: { data: { logout: true } },
    }

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[logoutMock]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      await authRef!.logout()
    })

    await waitFor(() => {
      expect(screen.getByTestId('loading').textContent).toBe('false')
    })
  })

  it('logout calls firebaseLogOut when auth enabled', async () => {
    isAuthDisabledVar(false)
    mockLogOut.mockResolvedValue(undefined)

    const logoutMock = {
      request: { query: LOGOUT_MUTATION },
      result: { data: { logout: true } },
    }

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[logoutMock]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      await authRef!.logout()
    })

    expect(mockLogOut).toHaveBeenCalled()
    isAuthDisabledVar(true)
  })

  it('logout sets error on mutation failure', async () => {
    const logoutMock = {
      request: { query: LOGOUT_MUTATION },
      error: new Error('Logout failed'),
    }

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[logoutMock]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      await authRef!.logout()
    })

    await waitFor(() => {
      expect(screen.getByTestId('error').textContent).toBeTruthy()
    })
  })
})

describe('AuthContext coverage - canSee / canWrite', () => {
  it('canSee returns true when auth disabled', () => {
    isAuthDisabledVar(true)

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    expect(authRef!.canSee('anything')).toBe(true)
  })

  it('canWrite returns true when auth disabled', () => {
    isAuthDisabledVar(true)

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    expect(authRef!.canWrite('anything')).toBe(true)
  })

  it('canSee and canWrite check permissions visibility map', async () => {
    isAuthDisabledVar(true) // query runs when authDisabled

    const permsMock = {
      request: { query: MY_PERMISSIONS_QUERY },
      result: {
        data: {
          myPermissions: {
            userId: 'u1',
            email: 'a@b.com',
            admin: false,
            persona: 'analyst',
            domains: [],
            databases: [],
            documents: [],
            apis: [],
            visibility: { dashboard: true, admin: false },
            writes: { glossary: true },
            feedback: {},
          },
        },
      },
    }

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[permsMock]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    // Wait for perms to load
    await waitFor(() => {
      expect(authRef!.permissions).not.toBeNull()
    })

    // With authDisabled=true, canSee/canWrite always return true
    expect(authRef!.canSee('dashboard')).toBe(true)
    expect(authRef!.canSee('nonexistent')).toBe(true)
    expect(authRef!.canWrite('glossary')).toBe(true)

    // Verify permissions object itself is correct
    expect(authRef!.permissions!.visibility).toEqual({ dashboard: true, admin: false })
    expect(authRef!.permissions!.writes).toEqual({ glossary: true })
  })
})

describe('AuthContext coverage - clearError', () => {
  it('clearError resets error to null', async () => {
    isAuthDisabledVar(true)

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      authRef!.setError('some error')
    })

    expect(screen.getByTestId('error').textContent).toBe('some error')

    await act(async () => {
      authRef!.clearError()
    })

    expect(screen.getByTestId('error').textContent).toBe('')
  })
})

describe('AuthContext coverage - loginWithMicrosoft', () => {
  beforeEach(() => {
    isAuthDisabledVar(true)
    vi.clearAllMocks()
  })

  it('loginWithMicrosoft sets error when health has no client id', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ auth: {} }),
    }))

    let authRef: ReturnType<typeof useAuth> | null = null

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer onAuth={(a) => { authRef = a }} />
        </AuthProvider>
      </MockedProvider>
    )

    await act(async () => {
      try {
        await authRef!.loginWithMicrosoft()
      } catch {
        // expected
      }
    })

    await waitFor(() => {
      expect(screen.getByTestId('error').textContent).toBe('Microsoft SSO not configured')
    })

    vi.unstubAllGlobals()
  })
})

describe('AuthContext coverage - firebase auth state subscription', () => {
  it('subscribes to auth changes when auth enabled', async () => {
    isAuthDisabledVar(false)
    const unsubFn = vi.fn()
    mockSubscribeToAuthChanges.mockImplementation(((cb: (user: any) => void) => {
      // Simulate a user being signed in
      setTimeout(() => cb({ uid: 'fb-user', emailVerified: true }), 0)
      return unsubFn
    }) as any)
    mockGetIdToken.mockResolvedValue('firebase-token')

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      </MockedProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('initialized').textContent).toBe('true')
    })

    expect(mockSubscribeToAuthChanges).toHaveBeenCalled()
    isAuthDisabledVar(true)
  })

  it('sets token to null when firebase user is null', async () => {
    isAuthDisabledVar(false)
    mockSubscribeToAuthChanges.mockImplementation(((cb: (user: any) => void) => {
      setTimeout(() => cb(null), 0)
      return vi.fn()
    }) as any)

    render(
      <MockedProvider mocks={[]} addTypename={false}>
        <AuthProvider>
          <TestConsumer />
        </AuthProvider>
      </MockedProvider>
    )

    await waitFor(() => {
      expect(screen.getByTestId('initialized').textContent).toBe('true')
    })

    isAuthDisabledVar(true)
  })
})
