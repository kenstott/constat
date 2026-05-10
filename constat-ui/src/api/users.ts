// Copyright (c) 2025 Kenneth Stott
// Canary: 85b06b56-8fe3-4273-ac6f-3076ff180067
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// User API calls

import { get } from './client'

export interface UserPermissions {
  user_id: string
  email: string | null
  persona: string
  domains: string[]
  databases: string[]
  documents: string[]
  apis: string[]
  visibility: Record<string, boolean>
  writes: Record<string, boolean>
  feedback: Record<string, boolean>
}

/**
 * Get permissions for the current authenticated user.
 */
export async function getMyPermissions(): Promise<UserPermissions> {
  return get<UserPermissions>('/users/me/permissions')
}

/**
 * List all users with explicit permissions.
 * Requires admin access.
 */
export async function listAllPermissions(): Promise<UserPermissions[]> {
  return get<UserPermissions[]>('/users/permissions')
}