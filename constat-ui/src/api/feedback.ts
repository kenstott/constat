// Copyright (c) 2025 Kenneth Stott
// Canary: d22de646-129a-413a-80e2-17e31e9a24f2
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

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
