// Copyright (c) 2025 Kenneth Stott
// Canary: 56a89e72-d902-4510-bc55-94facea1296f
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react'
import { useMutation, useQuery } from '@apollo/client'
import { makeVar, useReactiveVar } from '@apollo/client'
import { LOGIN_MUTATION, LOGOUT_MUTATION, MY_PERMISSIONS_QUERY } from '@/graphql/operations/auth'
import {
  isAuthDisabled,
  subscribeToAuthChanges,
  signInWithGoogle,
  signInWithEmail,
  signUpWithEmail,
  resetPassword,
  sendEmailLink,
  checkEmailLink,
  completeEmailLinkSignIn,
  logOut as firebaseLogOut,
  getIdToken,
  User,
} from '@/config/firebase'

export const isAuthDisabledVar = makeVar<boolean>(isAuthDisabled)

interface UserPermissions {
  userId: string
  email: string | null
  admin: boolean
  persona: string
  domains: string[]
  databases: string[]
  documents: string[]
  apis: string[]
  visibility: Record<string, boolean>
  writes: Record<string, boolean>
  feedback: Record<string, boolean>
}

interface AuthContextValue {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  isAuthDisabled: boolean
  userId: string
  isAdmin: boolean
  permissions: UserPermissions | null
  initialized: boolean
  login: (email: string, password: string) => Promise<void>
  loginWithGoogle: () => Promise<void>
  loginWithEmail: (email: string, password: string) => Promise<void>
  signupWithEmail: (email: string, password: string) => Promise<void>
  sendPasswordReset: (email: string) => Promise<void>
  sendEmailSignInLink: (email: string) => Promise<void>
  completeEmailLink: (email?: string) => Promise<void>
  isEmailLinkSignIn: () => boolean
  logout: () => Promise<void>
  canSee: (section: string) => boolean
  canWrite: (resource: string) => boolean
  loading: boolean
  error: string | null
  setError: (msg: string) => void
  clearError: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [loading, setLoading] = useState(!isAuthDisabled)
  const [initialized, setInitialized] = useState(isAuthDisabled)
  const [error, setErrorState] = useState<string | null>(null)
  const authDisabled = useReactiveVar(isAuthDisabledVar)

  const [loginMutation] = useMutation(LOGIN_MUTATION)
  const [logoutMutation] = useMutation(LOGOUT_MUTATION)

  // Query permissions when authenticated
  const { data: permData } = useQuery(MY_PERMISSIONS_QUERY, {
    skip: !authDisabled && !user,
  })

  const permissions: UserPermissions | null = permData?.myPermissions ?? null

  // Subscribe to Firebase auth state
  useEffect(() => {
    if (authDisabled) return
    const unsubscribe = subscribeToAuthChanges(async (fbUser) => {
      setUser(fbUser)
      if (fbUser) {
        const idToken = await getIdToken()
        setToken(idToken)
      } else {
        setToken(null)
      }
      setLoading(false)
      setInitialized(true)
    })
    return unsubscribe
  }, [authDisabled])

  const cleanFirebaseError = (message: string) =>
    message.replace('Firebase: ', '').replace(/\(auth\/[^)]+\)\.?/, '').trim()

  const login = useCallback(async (email: string, password: string) => {
    setLoading(true)
    setErrorState(null)
    try {
      const { data } = await loginMutation({ variables: { email, password } })
      setToken(data.login.token)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Login failed'
      setErrorState(message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [loginMutation])

  const loginWithGoogle = useCallback(async () => {
    if (authDisabled) return
    setLoading(true)
    setErrorState(null)
    try {
      await signInWithGoogle()
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to sign in with Google'
      setErrorState(message)
      throw err
    }
  }, [authDisabled])

  const loginWithEmail = useCallback(async (email: string, password: string) => {
    if (authDisabled) return
    setLoading(true)
    setErrorState(null)
    try {
      const fbUser = await signInWithEmail(email, password)
      if (fbUser && !fbUser.emailVerified) {
        setErrorState('Please verify your email before signing in. Check your inbox.')
        setLoading(false)
        await firebaseLogOut()
        return
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to sign in'
      setErrorState(cleanFirebaseError(message) || 'Failed to sign in')
      setLoading(false)
      throw err
    }
  }, [authDisabled])

  const signupWithEmailCb = useCallback(async (email: string, password: string) => {
    if (authDisabled) return
    setLoading(true)
    setErrorState(null)
    try {
      await signUpWithEmail(email, password)
      await firebaseLogOut()
      setLoading(false)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create account'
      setErrorState(cleanFirebaseError(message) || 'Failed to create account')
      setLoading(false)
      throw err
    }
  }, [authDisabled])

  const sendPasswordResetCb = useCallback(async (email: string) => {
    if (authDisabled) return
    setLoading(true)
    setErrorState(null)
    try {
      await resetPassword(email)
      setLoading(false)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to send reset email'
      setErrorState(cleanFirebaseError(message) || 'Failed to send reset email')
      setLoading(false)
      throw err
    }
  }, [authDisabled])

  const sendEmailSignInLinkCb = useCallback(async (email: string) => {
    if (authDisabled) return
    setLoading(true)
    setErrorState(null)
    try {
      await sendEmailLink(email)
      setLoading(false)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to send sign-in link'
      setErrorState(cleanFirebaseError(message) || 'Failed to send sign-in link')
      setLoading(false)
      throw err
    }
  }, [authDisabled])

  const completeEmailLinkCb = useCallback(async (email?: string) => {
    if (authDisabled) return
    setLoading(true)
    setErrorState(null)
    try {
      await completeEmailLinkSignIn(email)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to complete sign-in'
      setErrorState(cleanFirebaseError(message) || 'Failed to complete sign-in')
      setLoading(false)
      throw err
    }
  }, [authDisabled])

  const isEmailLinkSignIn = useCallback(() => {
    if (authDisabled) return false
    return checkEmailLink()
  }, [authDisabled])

  const logout = useCallback(async () => {
    setLoading(true)
    setErrorState(null)
    try {
      await logoutMutation()
      if (!authDisabled) {
        await firebaseLogOut()
      }
      setUser(null)
      setToken(null)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Logout failed'
      setErrorState(message)
    } finally {
      setLoading(false)
    }
  }, [logoutMutation, authDisabled])

  const canSee = useCallback((section: string) => {
    if (authDisabled) return true
    return permissions?.visibility?.[section] ?? false
  }, [authDisabled, permissions])

  const canWrite = useCallback((resource: string) => {
    if (authDisabled) return true
    return permissions?.writes?.[resource] ?? false
  }, [authDisabled, permissions])

  const setError = useCallback((msg: string) => setErrorState(msg), [])
  const clearError = useCallback(() => setErrorState(null), [])

  const value: AuthContextValue = {
    user,
    token,
    isAuthenticated: authDisabled || !!user || !!token,
    isAuthDisabled: authDisabled,
    userId: authDisabled ? 'default' : (user?.uid ?? 'default'),
    isAdmin: authDisabled || (permissions?.persona === 'platform_admin'),
    permissions,
    initialized,
    login,
    loginWithGoogle,
    loginWithEmail,
    signupWithEmail: signupWithEmailCb,
    sendPasswordReset: sendPasswordResetCb,
    sendEmailSignInLink: sendEmailSignInLinkCb,
    completeEmailLink: completeEmailLinkCb,
    isEmailLinkSignIn,
    logout,
    canSee,
    canWrite,
    loading,
    error,
    setError,
    clearError,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used within AuthProvider')
  return ctx
}
