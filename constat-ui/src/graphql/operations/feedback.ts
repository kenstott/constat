// Copyright (c) 2025 Kenneth Stott
// Canary: c1f81a9b-b04b-43d3-9420-df3958b7881e
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { gql } from '@apollo/client'

// -- Queries -----------------------------------------------------------------

export const GLOSSARY_SUGGESTIONS_QUERY = gql`
  query GlossarySuggestions($sessionId: String!) {
    glossarySuggestions(sessionId: $sessionId) {
      learningId
      term
      suggestedDefinition
      message
      created
      userId
    }
  }
`

// -- Mutations ---------------------------------------------------------------

export const FLAG_ANSWER = gql`
  mutation FlagAnswer($input: FlagAnswerInput!) {
    flagAnswer(input: $input) {
      learningId
      glossarySuggestionId
    }
  }
`

export const APPROVE_GLOSSARY_SUGGESTION = gql`
  mutation ApproveGlossarySuggestion($sessionId: String!, $learningId: String!) {
    approveGlossarySuggestion(sessionId: $sessionId, learningId: $learningId) {
      status
      learningId
    }
  }
`

export const REJECT_GLOSSARY_SUGGESTION = gql`
  mutation RejectGlossarySuggestion($sessionId: String!, $learningId: String!) {
    rejectGlossarySuggestion(sessionId: $sessionId, learningId: $learningId) {
      status
      learningId
    }
  }
`

// -- Mappers -----------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toGlossarySuggestion(gql: any) {
  return {
    learning_id: gql.learningId,
    term: gql.term,
    suggested_definition: gql.suggestedDefinition,
    message: gql.message,
    created: gql.created,
    user_id: gql.userId ?? null,
  }
}
