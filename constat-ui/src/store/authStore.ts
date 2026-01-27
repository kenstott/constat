// Authentication state store

import { create } from 'zustand'
import {
  isAuthDisabled,
  signInWithGoogle,
  signInWithEmail,
  signUpWithEmail,
  resetPassword,
  sendEmailLink,
  checkEmailLink,
  completeEmailLinkSignIn,
  logOut,
  getIdToken,
  subscribeToAuthChanges,
  User,
} from '@/config/firebase'

interface AuthState {
  // State
  user: User | null
  loading: boolean
  error: string | null
  initialized: boolean

  // Computed
  isAuthenticated: boolean
  userId: string

  // Actions
  initialize: () => void
  loginWithGoogle: () => Promise<void>
  loginWithEmail: (email: string, password: string) => Promise<void>
  signupWithEmail: (email: string, password: string) => Promise<void>
  sendPasswordReset: (email: string) => Promise<void>
  sendEmailSignInLink: (email: string) => Promise<void>
  completeEmailLink: (email?: string) => Promise<void>
  isEmailLinkSignIn: () => boolean
  logout: () => Promise<void>
  getToken: () => Promise<string | null>
  clearError: () => void
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  loading: true,
  error: null,
  initialized: false,

  // If auth is disabled, always authenticated with "default" user
  get isAuthenticated() {
    if (isAuthDisabled) return true
    return get().user !== null
  },

  // Return "default" when auth disabled, otherwise Firebase UID
  get userId() {
    if (isAuthDisabled) return 'default'
    return get().user?.uid || 'default'
  },

  initialize: () => {
    if (get().initialized) return

    if (isAuthDisabled) {
      // Auth disabled - immediately ready with no user
      set({ loading: false, initialized: true })
      return
    }

    // Subscribe to Firebase auth state changes
    subscribeToAuthChanges((user) => {
      set({ user, loading: false, initialized: true })
    })
  },

  loginWithGoogle: async () => {
    if (isAuthDisabled) return

    set({ loading: true, error: null })
    try {
      await signInWithGoogle()
      // Auth state will update via subscription
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to sign in with Google'
      set({ error: message, loading: false })
      throw err
    }
  },

  loginWithEmail: async (email: string, password: string) => {
    if (isAuthDisabled) return

    set({ loading: true, error: null })
    try {
      const user = await signInWithEmail(email, password)
      if (user && !user.emailVerified) {
        set({ error: 'Please verify your email before signing in. Check your inbox.', loading: false })
        await logOut()
        return
      }
      // Auth state will update via subscription
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to sign in'
      // Clean up Firebase error messages
      const cleanMessage = message
        .replace('Firebase: ', '')
        .replace(/\(auth\/[^)]+\)\.?/, '')
        .trim()
      set({ error: cleanMessage || 'Failed to sign in', loading: false })
      throw err
    }
  },

  signupWithEmail: async (email: string, password: string) => {
    if (isAuthDisabled) return

    set({ loading: true, error: null })
    try {
      await signUpWithEmail(email, password)
      // Sign out after signup - user needs to verify email first
      await logOut()
      set({
        loading: false,
        error: null,
      })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create account'
      const cleanMessage = message
        .replace('Firebase: ', '')
        .replace(/\(auth\/[^)]+\)\.?/, '')
        .trim()
      set({ error: cleanMessage || 'Failed to create account', loading: false })
      throw err
    }
  },

  sendPasswordReset: async (email: string) => {
    if (isAuthDisabled) return

    set({ loading: true, error: null })
    try {
      await resetPassword(email)
      set({ loading: false })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to send reset email'
      const cleanMessage = message
        .replace('Firebase: ', '')
        .replace(/\(auth\/[^)]+\)\.?/, '')
        .trim()
      set({ error: cleanMessage || 'Failed to send reset email', loading: false })
      throw err
    }
  },

  sendEmailSignInLink: async (email: string) => {
    if (isAuthDisabled) return

    set({ loading: true, error: null })
    try {
      await sendEmailLink(email)
      set({ loading: false })
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to send sign-in link'
      const cleanMessage = message
        .replace('Firebase: ', '')
        .replace(/\(auth\/[^)]+\)\.?/, '')
        .trim()
      set({ error: cleanMessage || 'Failed to send sign-in link', loading: false })
      throw err
    }
  },

  completeEmailLink: async (email?: string) => {
    if (isAuthDisabled) return

    set({ loading: true, error: null })
    try {
      await completeEmailLinkSignIn(email)
      // Auth state will update via subscription
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to complete sign-in'
      const cleanMessage = message
        .replace('Firebase: ', '')
        .replace(/\(auth\/[^)]+\)\.?/, '')
        .trim()
      set({ error: cleanMessage || 'Failed to complete sign-in', loading: false })
      throw err
    }
  },

  isEmailLinkSignIn: () => {
    if (isAuthDisabled) return false
    return checkEmailLink()
  },

  logout: async () => {
    if (isAuthDisabled) return

    set({ loading: true, error: null })
    try {
      await logOut()
      // Auth state will update via subscription
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to sign out'
      set({ error: message, loading: false })
    }
  },

  getToken: async () => {
    if (isAuthDisabled) return null
    return getIdToken()
  },

  clearError: () => set({ error: null }),
}))

// Export auth disabled flag for components to check
export { isAuthDisabled }