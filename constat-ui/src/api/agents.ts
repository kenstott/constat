// Agents API calls (session-based)

import { post } from './client'

export interface DraftAgentResponse {
  name: string
  prompt: string
  description: string
  skills: string[]
}

// Draft an agent using AI
export async function draftAgent(
  sessionId: string,
  name: string,
  userDescription: string
): Promise<DraftAgentResponse> {
  return post<DraftAgentResponse>(`/agents/draft?session_id=${encodeURIComponent(sessionId)}`, {
    name,
    user_description: userDescription,
  })
}
