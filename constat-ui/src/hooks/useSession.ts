// Copyright (c) 2025 Kenneth Stott
// Canary: 2da908c1-8d7b-444c-9305-de9cae3b9396
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useQuery } from '@apollo/client'
import { SESSION_QUERY, toSession } from '@/graphql/operations/sessions'
import type { Session } from '@/types/api'

interface UseSessionResult {
  session: Session | null
  loading: boolean
  error: Error | undefined
  refetch: () => void
}

export function useSession(sessionId: string | null): UseSessionResult {
  const { data, loading, error, refetch } = useQuery(SESSION_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
    fetchPolicy: 'cache-and-network',
  })

  const session = data?.session ? toSession(data.session) : null

  return { session, loading, error, refetch }
}
