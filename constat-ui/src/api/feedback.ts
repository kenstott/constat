// Feedback API — answer flagging and glossary suggestions

import * as api from '@/api/client'
import type { FlagRequest, FlagResponse, GlossarySuggestion } from '@/types/api'

export async function flagAnswer(
  sessionId: string,
  body: FlagRequest,
): Promise<FlagResponse> {
  return api.post<FlagResponse>(`/sessions/${sessionId}/feedback/flag`, body)
}

export async function getGlossarySuggestions(
  sessionId: string,
): Promise<GlossarySuggestion[]> {
  const resp = await api.get<{ suggestions: GlossarySuggestion[] }>(
    `/sessions/${sessionId}/feedback/glossary-suggestions`,
  )
  return resp.suggestions
}

export async function approveGlossarySuggestion(
  sessionId: string,
  learningId: string,
): Promise<void> {
  await api.post(`/sessions/${sessionId}/feedback/glossary-suggestions/${learningId}/approve`)
}

export async function rejectGlossarySuggestion(
  sessionId: string,
  learningId: string,
): Promise<void> {
  await api.post(`/sessions/${sessionId}/feedback/glossary-suggestions/${learningId}/reject`)
}
