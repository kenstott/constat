// Copyright (c) 2025 Kenneth Stott
// Canary: d535c2fb-d077-4493-8abc-7c7d1e917c5c
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useMutation, useSubscription, useQuery } from '@apollo/client'
import { useSessionContext } from '@/contexts/SessionContext'
import {
  QUERY_EXECUTION_SUBSCRIPTION,
  EXECUTION_PLAN_QUERY,
  SUBMIT_QUERY,
  CANCEL_EXECUTION,
  APPROVE_PLAN,
  ANSWER_CLARIFICATION,
  SKIP_CLARIFICATION,
  REPLAN_FROM,
  EDIT_OBJECTIVE,
  DELETE_OBJECTIVE,
  REQUEST_AUTOCOMPLETE,
  HEARTBEAT,
  toExecutionEvent,
} from '@/graphql/operations/execution'
import type { WSEvent } from '@/types/api'

interface UseQueryExecutionOptions {
  onEvent?: (event: WSEvent) => void
}

export function useQueryExecution(options?: UseQueryExecutionOptions) {
  const { sessionId } = useSessionContext()

  const { data: subData } = useSubscription(QUERY_EXECUTION_SUBSCRIPTION, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
    onData: ({ data: { data } }) => {
      if (data?.queryExecution && options?.onEvent) {
        options.onEvent(toExecutionEvent(data.queryExecution))
      }
    },
  })

  const { data: planData, loading: planLoading, refetch: refetchPlan } = useQuery(EXECUTION_PLAN_QUERY, {
    variables: { sessionId: sessionId! },
    skip: !sessionId,
  })

  const [submitQueryMut] = useMutation(SUBMIT_QUERY)
  const [cancelExecutionMut] = useMutation(CANCEL_EXECUTION)
  const [approvePlanMut] = useMutation(APPROVE_PLAN)
  const [answerClarificationMut] = useMutation(ANSWER_CLARIFICATION)
  const [skipClarificationMut] = useMutation(SKIP_CLARIFICATION)
  const [replanFromMut] = useMutation(REPLAN_FROM)
  const [editObjectiveMut] = useMutation(EDIT_OBJECTIVE)
  const [deleteObjectiveMut] = useMutation(DELETE_OBJECTIVE)
  const [requestAutocompleteMut] = useMutation(REQUEST_AUTOCOMPLETE)
  const [heartbeatMut] = useMutation(HEARTBEAT)

  return {
    lastEvent: subData?.queryExecution ? toExecutionEvent(subData.queryExecution) : null,
    plan: planData?.executionPlan ?? null,
    planLoading,
    refetchPlan,

    submit: (problem: string, isFollowup: boolean, mode?: string, guidanceNotes?: string) =>
      submitQueryMut({
        variables: {
          sessionId: sessionId!,
          input: { problem, isFollowup, mode, guidanceNotes },
        },
      }),

    cancel: () =>
      cancelExecutionMut({ variables: { sessionId: sessionId! } }),

    approvePlan: (approved: boolean, feedback?: string, deletedSteps?: number[], editedSteps?: Array<{ number: number; goal: string }>) =>
      approvePlanMut({
        variables: {
          sessionId: sessionId!,
          input: { approved, feedback, deletedSteps, editedSteps },
        },
      }),

    answerClarification: (answers: unknown, structuredAnswers: unknown) =>
      answerClarificationMut({
        variables: { sessionId: sessionId!, answers, structuredAnswers },
      }),

    skipClarification: () =>
      skipClarificationMut({ variables: { sessionId: sessionId! } }),

    replanFrom: (stepNumber: number, mode: string, editedGoal?: string) =>
      replanFromMut({
        variables: { sessionId: sessionId!, stepNumber, mode, editedGoal },
      }),

    editObjective: (objectiveIndex: number, newText: string) =>
      editObjectiveMut({
        variables: { sessionId: sessionId!, objectiveIndex, newText },
      }),

    deleteObjective: (objectiveIndex: number) =>
      deleteObjectiveMut({
        variables: { sessionId: sessionId!, objectiveIndex },
      }),

    requestAutocomplete: (context: string, prefix: string, parent?: string) =>
      requestAutocompleteMut({
        variables: { sessionId: sessionId!, context, prefix, parent },
      }),

    heartbeat: (since?: string) =>
      heartbeatMut({ variables: { sessionId: sessionId!, since } }),
  }
}
