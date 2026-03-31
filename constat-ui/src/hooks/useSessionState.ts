// Copyright (c) 2025 Kenneth Stott
// Canary: abc4e74e-ce58-4bbe-8df2-c657816cb48d
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
  STEPS_QUERY,
  INFERENCE_CODES_QUERY,
  SCRATCHPAD_QUERY,
  SESSION_DDL_QUERY,
  EXECUTION_OUTPUT_QUERY,
  SESSION_ROUTING_QUERY,
  PROOF_TREE_QUERY,
  PROOF_FACTS_QUERY,
  MESSAGES_QUERY,
  OBJECTIVES_QUERY,
  PROMPT_CONTEXT_QUERY,
  DATABASE_SCHEMA_QUERY,
  API_SCHEMA_QUERY,
  SAVE_PROOF_FACTS,
  SAVE_MESSAGES,
  UPDATE_SYSTEM_PROMPT,
  toStepCode,
  toInferenceCode,
  toScratchpadEntry,
  toStoredMessage,
  toStoredProofFact,
  toObjectivesEntry,
  toPromptContext,
  toDatabaseTable,
  toApiEndpoint,
} from '@/graphql/operations/state'

export function useSteps() {
  const { sessionId } = useSessionContext()
  const { data, loading, error, refetch } = useQuery(STEPS_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  const steps = (data?.steps?.steps ?? []).map(toStepCode)
  return { steps, total: data?.steps?.total ?? 0, loading, error, refetch }
}

export function useInferenceCodes() {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(INFERENCE_CODES_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  const inferences = (data?.inferenceCodes?.inferences ?? []).map(toInferenceCode)
  return { inferences, total: data?.inferenceCodes?.total ?? 0, loading, error }
}

export function useScratchpad() {
  const { sessionId } = useSessionContext()
  const { data, loading, error, refetch } = useQuery(SCRATCHPAD_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  const entries = (data?.scratchpad?.entries ?? []).map(toScratchpadEntry)
  return { entries, total: data?.scratchpad?.total ?? 0, loading, error, refetch }
}

export function useSessionDDL() {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(SESSION_DDL_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  return { ddl: data?.sessionDdl ?? '', loading, error }
}

export function useExecutionOutput() {
  const { sessionId } = useSessionContext()
  const { data, loading, error, refetch } = useQuery(EXECUTION_OUTPUT_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  return {
    output: data?.executionOutput?.output ?? '',
    suggestions: data?.executionOutput?.suggestions ?? [],
    currentQuery: data?.executionOutput?.currentQuery ?? '',
    loading,
    error,
    refetch,
  }
}

export function useSessionRouting() {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(SESSION_ROUTING_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  return { routing: data?.sessionRouting ?? null, loading, error }
}

export function useProofTree() {
  const { sessionId } = useSessionContext()
  const { data, loading, error, refetch } = useQuery(PROOF_TREE_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  return {
    facts: data?.proofTree?.facts ?? [],
    executionTrace: data?.proofTree?.executionTrace ?? null,
    loading,
    error,
    refetch,
  }
}

export function useProofFacts() {
  const { sessionId } = useSessionContext()
  const { data, loading, error, refetch } = useQuery(PROOF_FACTS_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  const facts = (data?.proofFacts?.facts ?? []).map(toStoredProofFact)
  return { facts, summary: data?.proofFacts?.summary ?? '', loading, error, refetch }
}

export function useMessages() {
  const { sessionId } = useSessionContext()
  const { data, loading, error, refetch } = useQuery(MESSAGES_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  const messages = (data?.messages?.messages ?? []).map(toStoredMessage)
  return { messages, loading, error, refetch }
}

export function useObjectives() {
  const { sessionId } = useSessionContext()
  const { data, loading, error, refetch } = useQuery(OBJECTIVES_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  const objectives = (data?.objectives ?? []).map(toObjectivesEntry)
  return { objectives, loading, error, refetch }
}

export function usePromptContext() {
  const { sessionId } = useSessionContext()
  const { data, loading, error, refetch } = useQuery(PROMPT_CONTEXT_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  return {
    context: data?.promptContext ? toPromptContext(data.promptContext) : null,
    loading,
    error,
    refetch,
  }
}

export function useDatabaseSchema(dbName: string) {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(DATABASE_SCHEMA_QUERY, {
    variables: { sessionId: sessionId!, dbName },
    skip: !sessionId || !dbName,
  })
  return {
    database: data?.databaseSchema?.database ?? dbName,
    tables: (data?.databaseSchema?.tables ?? []).map(toDatabaseTable),
    loading,
    error,
  }
}

export function useApiSchema(apiName: string) {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(API_SCHEMA_QUERY, {
    variables: { sessionId: sessionId!, apiName },
    skip: !sessionId || !apiName,
  })
  return {
    schema: data?.apiSchema ? {
      name: data.apiSchema.name,
      type: data.apiSchema.type,
      description: data.apiSchema.description,
      endpoints: (data.apiSchema.endpoints ?? []).map(toApiEndpoint),
    } : null,
    loading,
    error,
  }
}

export function useSessionStateMutations() {
  const { sessionId } = useSessionContext()
  const [saveProofFactsMut] = useMutation(SAVE_PROOF_FACTS)
  const [saveMessagesMut] = useMutation(SAVE_MESSAGES)
  const [updateSystemPromptMut] = useMutation(UPDATE_SYSTEM_PROMPT)

  return {
    saveProofFacts: (facts: unknown[], summary?: string) =>
      saveProofFactsMut({ variables: { sessionId: sessionId!, facts, summary } }),
    saveMessages: (messages: unknown[]) =>
      saveMessagesMut({ variables: { sessionId: sessionId!, messages } }),
    updateSystemPrompt: (systemPrompt: string) =>
      updateSystemPromptMut({ variables: { sessionId: sessionId!, systemPrompt } }),
  }
}
