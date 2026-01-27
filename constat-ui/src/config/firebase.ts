// Firebase configuration and initialization

import { initializeApp, FirebaseApp } from 'firebase/app'
import {
  getAuth,
  Auth,
  GoogleAuthProvider,
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  sendEmailVerification,
  sendPasswordResetEmail,
  signOut,
  onAuthStateChanged,
  User,
} from 'firebase/auth'

// Firebase config from environment variables
const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
}

// Check if auth is disabled (local dev mode)
export const isAuthDisabled = import.meta.env.VITE_AUTH_DISABLED === 'true'

// Initialize Firebase only if auth is enabled and config is present
let app: FirebaseApp | null = null
let auth: Auth | null = null
let googleProvider: GoogleAuthProvider | null = null

if (!isAuthDisabled && firebaseConfig.apiKey) {
  app = initializeApp(firebaseConfig)
  auth = getAuth(app)
  googleProvider = new GoogleAuthProvider()
}

// Auth functions that handle disabled state
export async function signInWithGoogle(): Promise<User | null> {
  if (isAuthDisabled || !auth || !googleProvider) {
    console.warn('Auth is disabled or not configured')
    return null
  }
  const result = await signInWithPopup(auth, googleProvider)
  return result.user
}

export async function signInWithEmail(email: string, password: string): Promise<User | null> {
  if (isAuthDisabled || !auth) {
    console.warn('Auth is disabled or not configured')
    return null
  }
  const result = await signInWithEmailAndPassword(auth, email, password)
  return result.user
}

export async function signUpWithEmail(email: string, password: string): Promise<User | null> {
  if (isAuthDisabled || !auth) {
    console.warn('Auth is disabled or not configured')
    return null
  }
  const result = await createUserWithEmailAndPassword(auth, email, password)
  // Send verification email
  if (result.user) {
    await sendEmailVerification(result.user)
  }
  return result.user
}

export async function resetPassword(email: string): Promise<void> {
  if (isAuthDisabled || !auth) {
    console.warn('Auth is disabled or not configured')
    return
  }
  await sendPasswordResetEmail(auth, email)
}

export async function logOut(): Promise<void> {
  if (isAuthDisabled || !auth) {
    return
  }
  await signOut(auth)
}

export async function getIdToken(): Promise<string | null> {
  if (isAuthDisabled) {
    return null
  }
  const user = auth?.currentUser
  if (!user) {
    return null
  }
  return user.getIdToken()
}

export function subscribeToAuthChanges(callback: (user: User | null) => void): () => void {
  if (isAuthDisabled || !auth) {
    // In disabled mode, immediately call with null and return no-op unsubscribe
    callback(null)
    return () => {}
  }
  return onAuthStateChanged(auth, callback)
}

export { auth, googleProvider }
export type { User }