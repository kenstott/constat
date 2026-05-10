// Copyright (c) 2025 Kenneth Stott
// Canary: 69a7fd02-980b-4847-ab9e-1db943c83f0d
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { gql } from '@apollo/client'

// -- Queries -----------------------------------------------------------------

export const STEPS_QUERY = gql`
  query Steps($sessionId: String!) {
    steps(sessionId: $sessionId) {
      steps {
        stepNumber
        goal
        code
        prompt
        model
      }
      total
    }
  }
`

export const INFERENCE_CODES_QUERY = gql`
  query InferenceCodes($sessionId: String!) {
    inferenceCodes(sessionId: $sessionId) {
      inferences {
        inferenceId
        name
        operation
        code
        attempt
        prompt
        model
      }
      total
    }
  }
`

export const SCRATCHPAD_QUERY = gql`
  query Scratchpad($sessionId: String!) {
    scratchpad(sessionId: $sessionId) {
      entries {
        stepNumber
        goal
        narrative
        tablesCreated
        code
        userQuery
        objectiveIndex
      }
      total
    }
  }
`

export const SESSION_DDL_QUERY = gql`
  query SessionDDL($sessionId: String!) {
    sessionDdl(sessionId: $sessionId)
  }
`

export const EXECUTION_OUTPUT_QUERY = gql`
  query ExecutionOutput($sessionId: String!) {
    executionOutput(sessionId: $sessionId) {
      output
      suggestions
      currentQuery
    }
  }
`

export const SESSION_ROUTING_QUERY = gql`
  query SessionRouting($sessionId: String!) {
    sessionRouting(sessionId: $sessionId)
  }
`

export const PROOF_TREE_QUERY = gql`
  query ProofTree($sessionId: String!) {
    proofTree(sessionId: $sessionId) {
      facts {
        name
        value
        source
        reasoning
        dependencies
      }
      executionTrace
    }
  }
`

export const PROOF_FACTS_QUERY = gql`
  query ProofFacts($sessionId: String!) {
    proofFacts(sessionId: $sessionId) {
      facts {
        id
        name
        description
        status
        value
        source
        confidence
        tier
        strategy
        formula
        reason
        dependencies
        elapsedMs
      }
      summary
    }
  }
`

export const MESSAGES_QUERY = gql`
  query Messages($sessionId: String!) {
    messages(sessionId: $sessionId) {
      messages {
        id
        type
        content
        timestamp
        stepNumber
        isFinalInsight
        stepDurationMs
        role
        skills
      }
    }
  }
`

export const OBJECTIVES_QUERY = gql`
  query Objectives($sessionId: String!) {
    objectives(sessionId: $sessionId) {
      type
      text
      question
      answer
      mode
      guidance
      ts
    }
  }
`

export const PROMPT_CONTEXT_QUERY = gql`
  query PromptContext($sessionId: String!) {
    promptContext(sessionId: $sessionId) {
      systemPrompt
      activeAgent {
        name
        prompt
      }
      activeSkills {
        name
        prompt
        description
      }
    }
  }
`

export const DATABASE_SCHEMA_QUERY = gql`
  query DatabaseSchema($sessionId: String!, $dbName: String!) {
    databaseSchema(sessionId: $sessionId, dbName: $dbName) {
      database
      tables {
        name
        rowCount
        columnCount
      }
    }
  }
`

export const API_SCHEMA_QUERY = gql`
  query ApiSchema($sessionId: String!, $apiName: String!) {
    apiSchema(sessionId: $sessionId, apiName: $apiName) {
      name
      type
      description
      endpoints {
        name
        kind
        returnType
        description
        httpMethod
        httpPath
        fields {
          name
          type
          description
          isRequired
        }
      }
    }
  }
`

// -- Mutations ---------------------------------------------------------------

export const SAVE_PROOF_FACTS = gql`
  mutation SaveProofFacts($sessionId: String!, $facts: JSON!, $summary: String) {
    saveProofFacts(sessionId: $sessionId, facts: $facts, summary: $summary) {
      status
      count
    }
  }
`

export const SAVE_MESSAGES = gql`
  mutation SaveMessages($sessionId: String!, $messages: JSON!) {
    saveMessages(sessionId: $sessionId, messages: $messages) {
      status
      count
    }
  }
`

export const UPDATE_SYSTEM_PROMPT = gql`
  mutation UpdateSystemPrompt($sessionId: String!, $systemPrompt: String!) {
    updateSystemPrompt(sessionId: $sessionId, systemPrompt: $systemPrompt) {
      status
      systemPrompt
    }
  }
`

// -- Mappers -----------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toStepCode(gql: any) {
  return {
    step_number: gql.stepNumber,
    goal: gql.goal,
    code: gql.code,
    prompt: gql.prompt ?? undefined,
    model: gql.model ?? undefined,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toInferenceCode(gql: any) {
  return {
    inference_id: gql.inferenceId,
    name: gql.name,
    operation: gql.operation,
    code: gql.code,
    attempt: gql.attempt,
    prompt: gql.prompt ?? undefined,
    model: gql.model ?? undefined,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toScratchpadEntry(gql: any) {
  return {
    step_number: gql.stepNumber,
    goal: gql.goal,
    narrative: gql.narrative,
    tables_created: gql.tablesCreated,
    code: gql.code,
    user_query: gql.userQuery,
    objective_index: gql.objectiveIndex ?? null,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toStoredMessage(gql: any) {
  return {
    id: gql.id,
    type: gql.type,
    content: gql.content,
    timestamp: gql.timestamp,
    stepNumber: gql.stepNumber ?? undefined,
    isFinalInsight: gql.isFinalInsight ?? undefined,
    stepDurationMs: gql.stepDurationMs ?? undefined,
    role: gql.role ?? undefined,
    skills: gql.skills ?? undefined,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toStoredProofFact(gql: any) {
  return {
    id: gql.id,
    name: gql.name,
    description: gql.description ?? undefined,
    status: gql.status,
    value: gql.value ?? undefined,
    source: gql.source ?? undefined,
    confidence: gql.confidence ?? undefined,
    tier: gql.tier ?? undefined,
    strategy: gql.strategy ?? undefined,
    formula: gql.formula ?? undefined,
    reason: gql.reason ?? undefined,
    dependencies: gql.dependencies ?? [],
    elapsed_ms: gql.elapsedMs ?? undefined,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toPromptContext(gql: any) {
  return {
    system_prompt: gql.systemPrompt,
    active_agent: gql.activeAgent ? { name: gql.activeAgent.name, prompt: gql.activeAgent.prompt } : null,
    active_skills: (gql.activeSkills ?? []).map((s: any) => ({
      name: s.name,
      prompt: s.prompt,
      description: s.description ?? '',
    })),
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toObjectivesEntry(gql: any) {
  return {
    type: gql.type,
    text: gql.text ?? undefined,
    question: gql.question ?? undefined,
    answer: gql.answer ?? undefined,
    mode: gql.mode ?? undefined,
    guidance: gql.guidance ?? undefined,
    ts: gql.ts ?? undefined,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toDatabaseTable(gql: any) {
  return {
    name: gql.name,
    row_count: gql.rowCount ?? null,
    column_count: gql.columnCount,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toApiEndpoint(gql: any) {
  return {
    name: gql.name,
    kind: gql.kind,
    return_type: gql.returnType ?? undefined,
    description: gql.description ?? undefined,
    http_method: gql.httpMethod ?? undefined,
    http_path: gql.httpPath ?? undefined,
    fields: (gql.fields ?? []).map((f: any) => ({
      name: f.name,
      type: f.type,
      description: f.description ?? undefined,
      is_required: f.isRequired,
    })),
  }
}
