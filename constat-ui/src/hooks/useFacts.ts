// Copyright (c) 2025 Kenneth Stott
// Canary: 2f9b372a-24a2-42d8-bb26-11e111e769dc
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useQuery, useMutation } from '@apollo/client'
import { useSessionContext } from '@/contexts/SessionContext'
import {
  FACTS_QUERY,
  ADD_FACT,
  EDIT_FACT,
  PERSIST_FACT,
  FORGET_FACT,
  MOVE_FACT,
  toFact,
} from '@/graphql/operations/data'

export function useFacts() {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(FACTS_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  const facts = (data?.facts?.facts ?? []).map(toFact)
  return { facts, total: data?.facts?.total ?? 0, loading, error }
}

export function useFactMutations() {
  const { sessionId } = useSessionContext()
  const refetch = { refetchQueries: ['Facts'] }
  const [addFactMut] = useMutation(ADD_FACT, refetch)
  const [editFactMut] = useMutation(EDIT_FACT, refetch)
  const [persistFactMut] = useMutation(PERSIST_FACT, refetch)
  const [forgetFactMut] = useMutation(FORGET_FACT, refetch)
  const [moveFactMut] = useMutation(MOVE_FACT, refetch)

  return {
    addFact: (name: string, value: unknown, persist?: boolean) =>
      addFactMut({ variables: { sessionId: sessionId!, name, value, persist } }),
    editFact: (factName: string, value: unknown) =>
      editFactMut({ variables: { sessionId: sessionId!, factName, value } }),
    persistFact: (factName: string) =>
      persistFactMut({ variables: { sessionId: sessionId!, factName } }),
    forgetFact: (factName: string) =>
      forgetFactMut({ variables: { sessionId: sessionId!, factName } }),
    moveFact: (factName: string, toDomain: string) =>
      moveFactMut({ variables: { sessionId: sessionId!, factName, toDomain } }),
  }
}
