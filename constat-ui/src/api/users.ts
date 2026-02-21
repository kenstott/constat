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