import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock firebase/app and firebase/auth before any imports
const mockInitializeApp = vi.fn(() => ({ name: 'test-app' }))
const mockGetAuth = vi.fn(() => ({ currentUser: null }))
const mockGoogleAuthProvider = vi.fn()
const mockSignInWithPopup = vi.fn()
const mockSignInWithEmailAndPassword = vi.fn()
const mockCreateUserWithEmailAndPassword = vi.fn()
const mockSendEmailVerification = vi.fn()
const mockSendPasswordResetEmail = vi.fn()
const mockSendSignInLinkToEmail = vi.fn()
const mockIsSignInWithEmailLink = vi.fn()
const mockSignInWithEmailLink = vi.fn()
const mockSignOut = vi.fn()
const mockOnAuthStateChanged = vi.fn()

vi.mock('firebase/app', () => ({
  initializeApp: (...args: any[]) => (mockInitializeApp as (...a: any[]) => any)(...args),
}))

vi.mock('firebase/auth', () => ({
  getAuth: (...args: any[]) => (mockGetAuth as (...a: any[]) => any)(...args),
  GoogleAuthProvider: mockGoogleAuthProvider,
  signInWithPopup: (...args: any[]) => (mockSignInWithPopup as (...a: any[]) => any)(...args),
  signInWithEmailAndPassword: (...args: any[]) => (mockSignInWithEmailAndPassword as (...a: any[]) => any)(...args),
  createUserWithEmailAndPassword: (...args: any[]) => (mockCreateUserWithEmailAndPassword as (...a: any[]) => any)(...args),
  sendEmailVerification: (...args: any[]) => (mockSendEmailVerification as (...a: any[]) => any)(...args),
  sendPasswordResetEmail: (...args: any[]) => (mockSendPasswordResetEmail as (...a: any[]) => any)(...args),
  sendSignInLinkToEmail: (...args: any[]) => (mockSendSignInLinkToEmail as (...a: any[]) => any)(...args),
  isSignInWithEmailLink: (...args: any[]) => (mockIsSignInWithEmailLink as (...a: any[]) => any)(...args),
  signInWithEmailLink: (...args: any[]) => (mockSignInWithEmailLink as (...a: any[]) => any)(...args),
  signOut: (...args: any[]) => (mockSignOut as (...a: any[]) => any)(...args),
  onAuthStateChanged: (...args: any[]) => (mockOnAuthStateChanged as (...a: any[]) => any)(...args),
}))

describe('firebase (auth disabled)', () => {
  beforeEach(() => {
    vi.resetModules()
    vi.stubEnv('VITE_AUTH_DISABLED', 'true')
  })

  it('isAuthDisabled is true when VITE_AUTH_DISABLED=true', async () => {
    const { isAuthDisabled } = await import('../firebase')
    expect(isAuthDisabled).toBe(true)
  })

  it('signInWithGoogle returns null when auth disabled', async () => {
    const { signInWithGoogle } = await import('../firebase')
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const result = await signInWithGoogle()
    expect(result).toBeNull()
    warnSpy.mockRestore()
  })

  it('signInWithEmail returns null when auth disabled', async () => {
    const { signInWithEmail } = await import('../firebase')
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const result = await signInWithEmail('test@test.com', 'password')
    expect(result).toBeNull()
    warnSpy.mockRestore()
  })

  it('signUpWithEmail returns null when auth disabled', async () => {
    const { signUpWithEmail } = await import('../firebase')
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const result = await signUpWithEmail('test@test.com', 'password')
    expect(result).toBeNull()
    warnSpy.mockRestore()
  })

  it('resetPassword returns early when auth disabled', async () => {
    const { resetPassword } = await import('../firebase')
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    await resetPassword('test@test.com')
    expect(mockSendPasswordResetEmail).not.toHaveBeenCalled()
    warnSpy.mockRestore()
  })

  it('sendEmailLink returns early when auth disabled', async () => {
    const { sendEmailLink } = await import('../firebase')
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    await sendEmailLink('test@test.com')
    expect(mockSendSignInLinkToEmail).not.toHaveBeenCalled()
    warnSpy.mockRestore()
  })

  it('checkEmailLink returns false when auth disabled', async () => {
    const { checkEmailLink } = await import('../firebase')
    expect(checkEmailLink()).toBe(false)
  })

  it('completeEmailLinkSignIn returns null when auth disabled', async () => {
    const { completeEmailLinkSignIn } = await import('../firebase')
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const result = await completeEmailLinkSignIn()
    expect(result).toBeNull()
    warnSpy.mockRestore()
  })

  it('logOut returns early when auth disabled', async () => {
    const { logOut } = await import('../firebase')
    await logOut()
    expect(mockSignOut).not.toHaveBeenCalled()
  })

  it('getIdToken returns null when auth disabled', async () => {
    const { getIdToken } = await import('../firebase')
    const result = await getIdToken()
    expect(result).toBeNull()
  })

  it('subscribeToAuthChanges calls callback with null and returns noop', async () => {
    const { subscribeToAuthChanges } = await import('../firebase')
    const cb = vi.fn()
    const unsub = subscribeToAuthChanges(cb)
    expect(cb).toHaveBeenCalledWith(null)
    expect(typeof unsub).toBe('function')
    unsub() // should be a no-op
  })

  it('auth and googleProvider are null when auth disabled', async () => {
    const { auth, googleProvider } = await import('../firebase')
    expect(auth).toBeNull()
    expect(googleProvider).toBeNull()
  })
})

describe('firebase (auth enabled)', () => {
  const mockUser = { uid: 'u1', getIdToken: vi.fn(() => Promise.resolve('tok-123')), emailVerified: true }
  const mockAuth = { currentUser: mockUser }

  beforeEach(() => {
    vi.resetModules()
    vi.stubEnv('VITE_AUTH_DISABLED', 'false')
    vi.stubEnv('VITE_FIREBASE_API_KEY', 'test-key')
    vi.stubEnv('VITE_FIREBASE_AUTH_DOMAIN', 'test.firebaseapp.com')
    vi.stubEnv('VITE_FIREBASE_PROJECT_ID', 'test-project')
    mockGetAuth.mockReturnValue(mockAuth as any)
    mockInitializeApp.mockReturnValue({ name: 'test-app' })
    mockSignInWithPopup.mockReset()
    mockSignInWithEmailAndPassword.mockReset()
    mockCreateUserWithEmailAndPassword.mockReset()
    mockSendEmailVerification.mockReset()
    mockSendPasswordResetEmail.mockReset()
    mockSendSignInLinkToEmail.mockReset()
    mockIsSignInWithEmailLink.mockReset()
    mockSignInWithEmailLink.mockReset()
    mockSignOut.mockReset()
    mockOnAuthStateChanged.mockReset()
  })

  it('signInWithGoogle calls signInWithPopup', async () => {
    mockSignInWithPopup.mockResolvedValue({ user: mockUser })
    const { signInWithGoogle } = await import('../firebase')
    const result = await signInWithGoogle()
    expect(result).toBe(mockUser)
    expect(mockSignInWithPopup).toHaveBeenCalled()
  })

  it('signInWithEmail calls signInWithEmailAndPassword', async () => {
    mockSignInWithEmailAndPassword.mockResolvedValue({ user: mockUser })
    const { signInWithEmail } = await import('../firebase')
    const result = await signInWithEmail('a@b.com', 'pw')
    expect(result).toBe(mockUser)
    expect(mockSignInWithEmailAndPassword).toHaveBeenCalled()
  })

  it('signUpWithEmail creates user and sends verification', async () => {
    mockCreateUserWithEmailAndPassword.mockResolvedValue({ user: mockUser })
    mockSendEmailVerification.mockResolvedValue(undefined)
    const { signUpWithEmail } = await import('../firebase')
    const result = await signUpWithEmail('a@b.com', 'pw')
    expect(result).toBe(mockUser)
    expect(mockSendEmailVerification).toHaveBeenCalledWith(mockUser)
  })

  it('resetPassword calls sendPasswordResetEmail', async () => {
    mockSendPasswordResetEmail.mockResolvedValue(undefined)
    const { resetPassword } = await import('../firebase')
    await resetPassword('a@b.com')
    expect(mockSendPasswordResetEmail).toHaveBeenCalled()
  })

  it('sendEmailLink calls sendSignInLinkToEmail and saves email', async () => {
    mockSendSignInLinkToEmail.mockResolvedValue(undefined)
    const { sendEmailLink } = await import('../firebase')
    await sendEmailLink('a@b.com')
    expect(mockSendSignInLinkToEmail).toHaveBeenCalled()
    expect(localStorage.getItem('emailForSignIn')).toBe('a@b.com')
  })

  it('checkEmailLink delegates to isSignInWithEmailLink', async () => {
    mockIsSignInWithEmailLink.mockReturnValue(true)
    const { checkEmailLink } = await import('../firebase')
    expect(checkEmailLink()).toBe(true)
  })

  it('completeEmailLinkSignIn returns null if not email link', async () => {
    mockIsSignInWithEmailLink.mockReturnValue(false)
    const { completeEmailLinkSignIn } = await import('../firebase')
    const result = await completeEmailLinkSignIn('a@b.com')
    expect(result).toBeNull()
  })

  it('completeEmailLinkSignIn throws if no email available', async () => {
    mockIsSignInWithEmailLink.mockReturnValue(true)
    localStorage.removeItem('emailForSignIn')
    const { completeEmailLinkSignIn } = await import('../firebase')
    await expect(completeEmailLinkSignIn()).rejects.toThrow('Please provide your email')
  })

  it('completeEmailLinkSignIn succeeds with email parameter', async () => {
    mockIsSignInWithEmailLink.mockReturnValue(true)
    mockSignInWithEmailLink.mockResolvedValue({ user: mockUser })
    const { completeEmailLinkSignIn } = await import('../firebase')
    const result = await completeEmailLinkSignIn('a@b.com')
    expect(result).toBe(mockUser)
    expect(mockSignInWithEmailLink).toHaveBeenCalled()
  })

  it('completeEmailLinkSignIn uses localStorage email', async () => {
    mockIsSignInWithEmailLink.mockReturnValue(true)
    mockSignInWithEmailLink.mockResolvedValue({ user: mockUser })
    localStorage.setItem('emailForSignIn', 'saved@b.com')
    const { completeEmailLinkSignIn } = await import('../firebase')
    const result = await completeEmailLinkSignIn()
    expect(result).toBe(mockUser)
    expect(localStorage.getItem('emailForSignIn')).toBeNull()
  })

  it('logOut calls signOut', async () => {
    mockSignOut.mockResolvedValue(undefined)
    const { logOut } = await import('../firebase')
    await logOut()
    expect(mockSignOut).toHaveBeenCalled()
  })

  it('getIdToken returns token from current user', async () => {
    const { getIdToken } = await import('../firebase')
    const token = await getIdToken()
    expect(token).toBe('tok-123')
  })

  it('getIdToken returns null when no current user', async () => {
    mockGetAuth.mockReturnValue({ currentUser: null })
    const { getIdToken } = await import('../firebase')
    const token = await getIdToken()
    expect(token).toBeNull()
  })

  it('subscribeToAuthChanges delegates to onAuthStateChanged', async () => {
    const unsubFn = vi.fn()
    mockOnAuthStateChanged.mockReturnValue(unsubFn)
    const { subscribeToAuthChanges } = await import('../firebase')
    const cb = vi.fn()
    const unsub = subscribeToAuthChanges(cb)
    expect(mockOnAuthStateChanged).toHaveBeenCalled()
    expect(unsub).toBe(unsubFn)
  })
})
