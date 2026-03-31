// Copyright (c) 2025 Kenneth Stott
// Canary: 44632503-26c4-44a3-a4fc-dead051f849f
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { gql } from '@apollo/client'

// -- Queries -----------------------------------------------------------------

export const TESTABLE_DOMAINS_QUERY = gql`
  query TestableDomains($sessionId: String!) {
    testableDomains(sessionId: $sessionId) {
      filename
      name
      questionCount
      tags
    }
  }
`

export const GOLDEN_QUESTIONS_QUERY = gql`
  query GoldenQuestions($sessionId: String!, $domain: String!) {
    goldenQuestions(sessionId: $sessionId, domain: $domain) {
      question
      tags
      expect
      objectives
      systemPrompt
      index
    }
  }
`

// -- Mutations ---------------------------------------------------------------

export const EXTRACT_EXPECTATIONS = gql`
  mutation ExtractExpectations($sessionId: String!, $input: GoldenQuestionExpectInput!) {
    extractExpectations(sessionId: $sessionId, input: $input) {
      terms
      grounding
      relationships
      expectedOutputs
      endToEnd
      suggestedQuestion
      objectives
      stepHints
      systemPrompt
    }
  }
`

export const CREATE_GOLDEN_QUESTION = gql`
  mutation CreateGoldenQuestion($sessionId: String!, $domain: String!, $input: CreateGoldenQuestionInput!) {
    createGoldenQuestion(sessionId: $sessionId, domain: $domain, input: $input) {
      question
      tags
      expect
      objectives
      systemPrompt
      index
    }
  }
`

export const UPDATE_GOLDEN_QUESTION = gql`
  mutation UpdateGoldenQuestion($sessionId: String!, $domain: String!, $index: Int!, $input: UpdateGoldenQuestionInput!) {
    updateGoldenQuestion(sessionId: $sessionId, domain: $domain, index: $index, input: $input) {
      question
      tags
      expect
      objectives
      systemPrompt
      index
    }
  }
`

export const DELETE_GOLDEN_QUESTION = gql`
  mutation DeleteGoldenQuestion($sessionId: String!, $domain: String!, $index: Int!) {
    deleteGoldenQuestion(sessionId: $sessionId, domain: $domain, index: $index) {
      status
    }
  }
`

export const MOVE_GOLDEN_QUESTION = gql`
  mutation MoveGoldenQuestion($sessionId: String!, $domain: String!, $index: Int!, $input: MoveGoldenQuestionInput!) {
    moveGoldenQuestion(sessionId: $sessionId, domain: $domain, index: $index, input: $input) {
      question
      tags
      expect
      objectives
      systemPrompt
      index
      warnings
      domain
    }
  }
`

// -- Mappers -----------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toTestableDomain(gql: any): { filename: string; name: string; question_count: number; tags: string[] } {
  return {
    filename: gql.filename,
    name: gql.name,
    question_count: gql.questionCount,
    tags: gql.tags ?? [],
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toGoldenQuestion(gql: any) {
  return {
    question: gql.question,
    tags: gql.tags ?? [],
    expect: gql.expect,
    objectives: gql.objectives ?? [],
    system_prompt: gql.systemPrompt ?? null,
    index: gql.index,
    warnings: gql.warnings ?? undefined,
    domain: gql.domain ?? undefined,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toExpectations(gql: any) {
  return {
    terms: gql.terms ?? [],
    grounding: gql.grounding ?? [],
    relationships: gql.relationships ?? [],
    expected_outputs: gql.expectedOutputs ?? [],
    end_to_end: gql.endToEnd ?? null,
    suggested_question: gql.suggestedQuestion ?? null,
    objectives: gql.objectives ?? [],
    step_hints: gql.stepHints ?? [],
    system_prompt: gql.systemPrompt ?? null,
  }
}
