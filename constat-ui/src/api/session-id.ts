// Copyright (c) 2025 Kenneth Stott
// Canary: 83d8b7b5-58f3-4cc3-923f-2770229d0d96
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// Session ID persistence (localStorage management)

const SESSION_ID_BASE_KEY = 'constat-session-id'

function getSessionKey(userId?: string): string {
  return userId && userId !== 'default'
    ? `${SESSION_ID_BASE_KEY}-${userId}`
    : SESSION_ID_BASE_KEY
}

export function getStoredSessionId(userId?: string): string | null {
  return localStorage.getItem(getSessionKey(userId))
}

export function storeSessionId(sessionId: string, userId?: string): void {
  localStorage.setItem(getSessionKey(userId), sessionId)
}

export function clearStoredSessionId(userId?: string): void {
  localStorage.removeItem(getSessionKey(userId))
}

export function createNewSessionId(userId?: string): string {
  const sessionId = crypto.randomUUID()
  storeSessionId(sessionId, userId)
  return sessionId
}

export function getOrCreateSessionId(userId?: string): string {
  const existing = getStoredSessionId(userId)
  if (existing) {
    return existing
  }
  return createNewSessionId(userId)
}
