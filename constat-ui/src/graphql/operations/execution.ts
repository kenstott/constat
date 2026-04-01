// Copyright (c) 2025 Kenneth Stott
// Canary: d49c6a74-aadd-43f2-b23c-e44338575a43
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { gql } from '@apollo/client'
import type { SubscriptionEvent } from '@/types/api'

// -- Subscription ------------------------------------------------------------

export const QUERY_EXECUTION_SUBSCRIPTION = gql`
  subscription QueryExecution($sessionId: String!) {
    queryExecution(sessionId: $sessionId) {
      eventType
      sessionId
      stepNumber
      timestamp
      data
    }
  }
`

// -- Queries -----------------------------------------------------------------

export const EXECUTION_PLAN_QUERY = gql`
  query ExecutionPlan($sessionId: String!) {
    executionPlan(sessionId: $sessionId) {
      problem
      steps {
        number
        goal
        status
        expectedInputs
        expectedOutputs
        dependsOn
        code
        domain
        result
      }
      currentStep
      completedSteps
      failedSteps
      isComplete
    }
  }
`

// -- Mutations ---------------------------------------------------------------

export const SUBMIT_QUERY = gql`
  mutation SubmitQuery($sessionId: String!, $input: SubmitQueryInput!) {
    submitQuery(sessionId: $sessionId, input: $input) {
      executionId
      status
      message
    }
  }
`

export const CANCEL_EXECUTION = gql`
  mutation CancelExecution($sessionId: String!) {
    cancelExecution(sessionId: $sessionId) {
      status
      message
    }
  }
`

export const APPROVE_PLAN = gql`
  mutation ApprovePlan($sessionId: String!, $input: ApprovePlanInput!) {
    approvePlan(sessionId: $sessionId, input: $input) {
      status
      message
    }
  }
`

export const ANSWER_CLARIFICATION = gql`
  mutation AnswerClarification($sessionId: String!, $answers: JSON!, $structuredAnswers: JSON!) {
    answerClarification(sessionId: $sessionId, answers: $answers, structuredAnswers: $structuredAnswers) {
      status
      message
    }
  }
`

export const SKIP_CLARIFICATION = gql`
  mutation SkipClarification($sessionId: String!) {
    skipClarification(sessionId: $sessionId) {
      status
      message
    }
  }
`

export const REPLAN_FROM = gql`
  mutation ReplanFrom($sessionId: String!, $stepNumber: Int!, $mode: String!, $editedGoal: String) {
    replanFrom(sessionId: $sessionId, stepNumber: $stepNumber, mode: $mode, editedGoal: $editedGoal) {
      status
      message
    }
  }
`

export const EDIT_OBJECTIVE = gql`
  mutation EditObjective($sessionId: String!, $objectiveIndex: Int!, $newText: String!) {
    editObjective(sessionId: $sessionId, objectiveIndex: $objectiveIndex, newText: $newText) {
      status
      message
    }
  }
`

export const DELETE_OBJECTIVE = gql`
  mutation DeleteObjective($sessionId: String!, $objectiveIndex: Int!) {
    deleteObjective(sessionId: $sessionId, objectiveIndex: $objectiveIndex) {
      status
      message
    }
  }
`

export const REQUEST_AUTOCOMPLETE = gql`
  mutation RequestAutocomplete($sessionId: String!, $context: String!, $prefix: String!, $parent: String) {
    requestAutocomplete(sessionId: $sessionId, context: $context, prefix: $prefix, parent: $parent) {
      requestId
      items {
        label
        value
        description
      }
    }
  }
`

export const HEARTBEAT = gql`
  mutation Heartbeat($sessionId: String!, $since: String) {
    heartbeat(sessionId: $sessionId, since: $since) {
      status
      message
    }
  }
`

// -- Mapper ------------------------------------------------------------------

// Converts camelCase GQL subscription event to snake_case SubscriptionEvent
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toExecutionEvent(gqlEvent: any): SubscriptionEvent {
  return {
    event_type: gqlEvent.eventType,
    session_id: gqlEvent.sessionId,
    step_number: gqlEvent.stepNumber ?? 0,
    timestamp: gqlEvent.timestamp,
    data: gqlEvent.data ?? {},
  }
}
