// Roles API calls (session-based)

import { post } from './client'

export interface DraftRoleResponse {
  name: string
  prompt: string
  description: string
}

// Draft a role using AI
export async function draftRole(
  sessionId: string,
  name: string,
  userDescription: string
): Promise<DraftRoleResponse> {
  return post<DraftRoleResponse>(`/roles/draft?session_id=${encodeURIComponent(sessionId)}`, {
    name,
    user_description: userDescription,
  })
}
