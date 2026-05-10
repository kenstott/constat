// Copyright (c) 2025 Kenneth Stott
// Canary: 272c4743-8b45-47ea-840e-39f73516ce9d
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
  ENTITIES_QUERY,
  ADD_ENTITY_TO_GLOSSARY,
  toEntity,
} from '@/graphql/operations/data'

export function useEntities(entityType?: string) {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(ENTITIES_QUERY, {
    variables: { sessionId: sessionId!, entityType },
    skip: !sessionId,
  })
  const entities = (data?.entities?.entities ?? []).map(toEntity)
  return { entities, total: data?.entities?.total ?? 0, loading, error }
}

export function useEntityMutations() {
  const { sessionId } = useSessionContext()
  const [addToGlossaryMut] = useMutation(ADD_ENTITY_TO_GLOSSARY)

  return {
    addToGlossary: (entityId: string) =>
      addToGlossaryMut({ variables: { sessionId: sessionId!, entityId } }),
  }
}
