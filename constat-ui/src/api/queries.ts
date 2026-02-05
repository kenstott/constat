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