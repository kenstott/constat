// Copyright (c) 2025 Kenneth Stott
// Canary: 25592c3c-49da-4272-9709-b374451a0d81
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
  TESTABLE_DOMAINS_QUERY,
  GOLDEN_QUESTIONS_QUERY,
  EXTRACT_EXPECTATIONS,
  CREATE_GOLDEN_QUESTION,
  UPDATE_GOLDEN_QUESTION,
  DELETE_GOLDEN_QUESTION,
  MOVE_GOLDEN_QUESTION,
  toTestableDomain,
  toGoldenQuestion,
  toExpectations,
} from '@/graphql/operations/testing'

export type TestableDomainResult = ReturnType<typeof toTestableDomain>

export function useTestableDomains() {
  const { sessionId } = useSessionContext()
  const { data, loading, error } = useQuery(TESTABLE_DOMAINS_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })
  const domains: TestableDomainResult[] = (data?.testableDomains ?? []).map(toTestableDomain)
  return { domains, loading, error }
}

export function useGoldenQuestions(domain: string) {
  const { sessionId } = useSessionContext()
  const { data, loading, error, refetch } = useQuery(GOLDEN_QUESTIONS_QUERY, {
    variables: { sessionId: sessionId!, domain },
    skip: !sessionId || !domain,
  })
  const questions = (data?.goldenQuestions ?? []).map(toGoldenQuestion)
  return { questions, loading, error, refetch }
}

export function useTestingMutations() {
  const { sessionId } = useSessionContext()

  const [extractExpectationsMut] = useMutation(EXTRACT_EXPECTATIONS)
  const [createGoldenQuestionMut] = useMutation(CREATE_GOLDEN_QUESTION, {
    refetchQueries: ['GoldenQuestions', 'TestableDomains'],
  })
  const [updateGoldenQuestionMut] = useMutation(UPDATE_GOLDEN_QUESTION, {
    refetchQueries: ['GoldenQuestions'],
  })
  const [deleteGoldenQuestionMut] = useMutation(DELETE_GOLDEN_QUESTION, {
    refetchQueries: ['GoldenQuestions', 'TestableDomains'],
  })
  const [moveGoldenQuestionMut] = useMutation(MOVE_GOLDEN_QUESTION, {
    refetchQueries: ['GoldenQuestions', 'TestableDomains'],
  })

  return {
    extractExpectations: (input: { question: string; domain: string }) =>
      extractExpectationsMut({
        variables: { sessionId: sessionId!, input },
      }).then((res) => (res.data?.extractExpectations ? toExpectations(res.data.extractExpectations) : null)),

    createGoldenQuestion: (domain: string, input: Record<string, unknown>) =>
      createGoldenQuestionMut({
        variables: { sessionId: sessionId!, domain, input },
      }).then((res) => (res.data?.createGoldenQuestion ? toGoldenQuestion(res.data.createGoldenQuestion) : null)),

    updateGoldenQuestion: (domain: string, index: number, input: Record<string, unknown>) =>
      updateGoldenQuestionMut({
        variables: { sessionId: sessionId!, domain, index, input },
      }).then((res) => (res.data?.updateGoldenQuestion ? toGoldenQuestion(res.data.updateGoldenQuestion) : null)),

    deleteGoldenQuestion: (domain: string, index: number) =>
      deleteGoldenQuestionMut({
        variables: { sessionId: sessionId!, domain, index },
      }),

    moveGoldenQuestion: (domain: string, index: number, input: Record<string, unknown>) =>
      moveGoldenQuestionMut({
        variables: { sessionId: sessionId!, domain, index, input },
      }).then((res) => (res.data?.moveGoldenQuestion ? toGoldenQuestion(res.data.moveGoldenQuestion) : null)),
  }
}
