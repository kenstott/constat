// Copyright (c) 2025 Kenneth Stott
// Canary: 9b18b8d3-a177-4c2f-91de-4d8fa28d45d0
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
  GLOSSARY_SUGGESTIONS_QUERY,
  FLAG_ANSWER,
  APPROVE_GLOSSARY_SUGGESTION,
  REJECT_GLOSSARY_SUGGESTION,
  toGlossarySuggestion,
} from '@/graphql/operations/feedback'

export function useGlossarySuggestions() {
  const { sessionId } = useSessionContext()
  const { data, loading, error, refetch } = useQuery(GLOSSARY_SUGGESTIONS_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  const suggestions = (data?.glossarySuggestions ?? []).map(toGlossarySuggestion)
  return { suggestions, loading, error, refetch }
}

export function useFeedbackMutations() {
  const { sessionId } = useSessionContext()
  const [flagAnswerMut] = useMutation(FLAG_ANSWER)
  const [approveSuggestionMut] = useMutation(APPROVE_GLOSSARY_SUGGESTION, {
    refetchQueries: ['GlossarySuggestions'],
  })
  const [rejectSuggestionMut] = useMutation(REJECT_GLOSSARY_SUGGESTION, {
    refetchQueries: ['GlossarySuggestions'],
  })

  return {
    flagAnswer: (input: { sessionId: string; queryText: string; answerText: string; feedbackText: string; feedbackType: string }) =>
      flagAnswerMut({ variables: { input } }),
    approveSuggestion: (learningId: string) =>
      approveSuggestionMut({ variables: { sessionId: sessionId!, learningId } }),
    rejectSuggestion: (learningId: string) =>
      rejectSuggestionMut({ variables: { sessionId: sessionId!, learningId } }),
  }
}
