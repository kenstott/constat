// Copyright (c) 2025 Kenneth Stott
// Canary: a0d47229-032d-4253-8936-c299e2f79f12
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// Query API calls

import { get, post } from './client'
import type { Plan, QueryResponse } from '@/types/api'

export async function submitQuery(
  sessionId: string,
  problem: string,
  isFollowup = false
): Promise<QueryResponse> {
  return post<QueryResponse>(`/sessions/${sessionId}/query`, {
    problem,
    is_followup: isFollowup,
  })
}

export async function cancelExecution(
  sessionId: string
): Promise<{ status: string; message: string }> {
  return post<{ status: string; message: string }>(`/sessions/${sessionId}/cancel`)
}

export async function getPlan(sessionId: string): Promise<Plan> {
  return get<Plan>(`/sessions/${sessionId}/plan`)
}

export async function approvePlan(
  sessionId: string,
  approved: boolean,
  feedback?: string,
  deletedSteps?: number[],
  editedSteps?: Array<{ number: number; goal: string }>
): Promise<{ status: string; message: string }> {
  return post<{ status: string; message: string }>(`/sessions/${sessionId}/plan/approve`, {
    approved,
    feedback,
    deleted_steps: deletedSteps,
    edited_steps: editedSteps,
  })
}