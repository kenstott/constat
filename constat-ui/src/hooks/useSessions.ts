// Copyright (c) 2025 Kenneth Stott
// Canary: 19d8ff4d-90a2-48e7-b586-d70e1300a73b
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useQuery, useMutation } from '@apollo/client'
import {
  SESSIONS_QUERY,
  CREATE_SESSION,
  DELETE_SESSION,
  toSession,
} from '@/graphql/operations/sessions'
import type { Session } from '@/types/api'

interface UseSessionsResult {
  sessions: Session[]
  total: number
  loading: boolean
  error: Error | undefined
  refetch: () => void
  createSession: (sessionId: string, userId?: string) => Promise<Session>
  deleteSession: (sessionId: string) => Promise<boolean>
}

export function useSessions(): UseSessionsResult {
  const { data, loading, error, refetch } = useQuery(SESSIONS_QUERY, {
    fetchPolicy: 'cache-and-network',
  })

  const [createMutation] = useMutation(CREATE_SESSION)
  const [deleteMutation] = useMutation(DELETE_SESSION)

  const sessions: Session[] = (data?.sessions?.sessions ?? []).map(toSession)
  const total: number = data?.sessions?.total ?? 0

  const createSession = async (sessionId: string, userId?: string): Promise<Session> => {
    const { data: result } = await createMutation({
      variables: { sessionId, userId },
      refetchQueries: [{ query: SESSIONS_QUERY }],
    })
    return toSession(result.createSession)
  }

  const deleteSession = async (sessionId: string): Promise<boolean> => {
    const { data: result } = await deleteMutation({
      variables: { sessionId },
      refetchQueries: [{ query: SESSIONS_QUERY }],
    })
    return result.deleteSession
  }

  return { sessions, total, loading, error, refetch, createSession, deleteSession }
}
