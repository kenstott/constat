// Copyright (c) 2025 Kenneth Stott
// Canary: 5b4f33dc-d15d-43b9-9ba9-bd2ec53abeba
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// Standalone auth helpers for non-React code (API clients, async handlers)
// React components should use useAuth() from AuthContext instead.

import { isAuthDisabled, getIdToken } from '@/config/firebase'

/** Get auth token (null when auth is disabled) */
export async function getToken(): Promise<string | null> {
  if (isAuthDisabled) return null
  return getIdToken()
}

/** Build auth headers for fetch calls */
export async function getAuthHeaders(): Promise<Record<string, string>> {
  const headers: Record<string, string> = {}
  if (!isAuthDisabled) {
    const token = await getToken()
    if (token) headers['Authorization'] = `Bearer ${token}`
  }
  return headers
}

export { isAuthDisabled }
