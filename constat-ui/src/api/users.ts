// User API calls

import { get, put } from './client'

export interface UserPermissions {
  user_id: string
  email: string | null
  admin: boolean
  projects: string[]
}

export interface UpdatePermissionsRequest {
  email: string
  admin?: boolean
  projects?: string[]
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

/**
 * Update permissions for a user.
 * Requires admin access.
 */
export async function updateUserPermissions(
  request: UpdatePermissionsRequest
): Promise<UserPermissions> {
  return put<UserPermissions>('/users/permissions', request)
}