// Copyright (c) 2025 Kenneth Stott
// Canary: 06895ad7-4b72-4a6b-86b1-714d93f5bb72
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

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
