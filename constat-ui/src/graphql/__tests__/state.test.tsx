// Copyright (c) 2025 Kenneth Stott
// Canary: eb1a2189-8194-4cfc-8f7b-3250683775c6
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { describe, it, expect, vi } from 'vitest'
import { getOperationAST } from 'graphql'

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
  toPromptContext,
  toObjectivesEntry,
  toDatabaseTable,
  toApiEndpoint,
} from '@/graphql/operations/state'

// vi.mock is hoisted — factory must be self-contained
vi.mock('idb', () => ({
  openDB: vi.fn(() => Promise.resolve({
    get: vi.fn(() => undefined),
    put: vi.fn(),
    delete: vi.fn(),
  })),
}))

function opName(doc: any): string {
  const op = getOperationAST(doc)
  return op!.name!.value
}

function opType(doc: any): string {
  const op = getOperationAST(doc)
  return op!.operation
}

describe('GQL documents — queries', () => {
  const queries: [string, any][] = [
    ['Steps', STEPS_QUERY],
    ['InferenceCodes', INFERENCE_CODES_QUERY],
    ['Scratchpad', SCRATCHPAD_QUERY],
    ['SessionDDL', SESSION_DDL_QUERY],
    ['ExecutionOutput', EXECUTION_OUTPUT_QUERY],
    ['SessionRouting', SESSION_ROUTING_QUERY],
    ['ProofTree', PROOF_TREE_QUERY],
    ['ProofFacts', PROOF_FACTS_QUERY],
    ['Messages', MESSAGES_QUERY],
    ['Objectives', OBJECTIVES_QUERY],
    ['PromptContext', PROMPT_CONTEXT_QUERY],
    ['DatabaseSchema', DATABASE_SCHEMA_QUERY],
    ['ApiSchema', API_SCHEMA_QUERY],
  ]

  it.each(queries)('%s parses with correct name and type', (name, doc) => {
    expect(opName(doc)).toBe(name)
    expect(opType(doc)).toBe('query')
  })
})

describe('GQL documents — mutations', () => {
  const mutations: [string, any][] = [
    ['SaveProofFacts', SAVE_PROOF_FACTS],
    ['SaveMessages', SAVE_MESSAGES],
    ['UpdateSystemPrompt', UPDATE_SYSTEM_PROMPT],
  ]

  it.each(mutations)('%s parses with correct name and type', (name, doc) => {
    expect(opName(doc)).toBe(name)
    expect(opType(doc)).toBe('mutation')
  })
})

describe('toStepCode mapper', () => {
  it('maps camelCase to snake_case', () => {
    const result = toStepCode({
      stepNumber: 3,
      goal: 'compute total',
      code: 'SELECT 1',
      prompt: 'do it',
      model: 'gpt-4',
    })
    expect(result).toEqual({
      step_number: 3,
      goal: 'compute total',
      code: 'SELECT 1',
      prompt: 'do it',
      model: 'gpt-4',
    })
  })

  it('converts null optional fields to undefined', () => {
    const result = toStepCode({
      stepNumber: 1,
      goal: 'g',
      code: 'c',
      prompt: null,
      model: null,
    })
    expect(result.prompt).toBeUndefined()
    expect(result.model).toBeUndefined()
  })
})

describe('toInferenceCode mapper', () => {
  it('maps all fields correctly', () => {
    const result = toInferenceCode({
      inferenceId: 'inf-1',
      name: 'total_revenue',
      operation: 'sum',
      code: 'SELECT SUM(x)',
      attempt: 2,
      prompt: 'prompt text',
      model: 'claude-3',
    })
    expect(result).toEqual({
      inference_id: 'inf-1',
      name: 'total_revenue',
      operation: 'sum',
      code: 'SELECT SUM(x)',
      attempt: 2,
      prompt: 'prompt text',
      model: 'claude-3',
    })
  })
})

describe('toScratchpadEntry mapper', () => {
  it('maps fields including null objectiveIndex', () => {
    const result = toScratchpadEntry({
      stepNumber: 1,
      goal: 'load data',
      narrative: 'loaded 100 rows',
      tablesCreated: ['t1'],
      code: 'SELECT *',
      userQuery: 'show me data',
      objectiveIndex: null,
    })
    expect(result).toEqual({
      step_number: 1,
      goal: 'load data',
      narrative: 'loaded 100 rows',
      tables_created: ['t1'],
      code: 'SELECT *',
      user_query: 'show me data',
      objective_index: null,
    })
  })
})

describe('toStoredMessage mapper', () => {
  it('maps required and optional fields', () => {
    const result = toStoredMessage({
      id: 'msg-1',
      type: 'answer',
      content: 'hello',
      timestamp: '2025-01-01T00:00:00Z',
      stepNumber: 2,
      isFinalInsight: true,
      stepDurationMs: 500,
      role: 'assistant',
      skills: ['sql'],
    })
    expect(result).toEqual({
      id: 'msg-1',
      type: 'answer',
      content: 'hello',
      timestamp: '2025-01-01T00:00:00Z',
      stepNumber: 2,
      isFinalInsight: true,
      stepDurationMs: 500,
      role: 'assistant',
      skills: ['sql'],
    })
  })

  it('converts null optionals to undefined', () => {
    const result = toStoredMessage({
      id: 'msg-2',
      type: 'question',
      content: 'hi',
      timestamp: '2025-01-01T00:00:00Z',
      stepNumber: null,
      isFinalInsight: null,
      stepDurationMs: null,
      role: null,
      skills: null,
    })
    expect(result.stepNumber).toBeUndefined()
    expect(result.isFinalInsight).toBeUndefined()
    expect(result.stepDurationMs).toBeUndefined()
    expect(result.role).toBeUndefined()
    expect(result.skills).toBeUndefined()
  })
})

describe('toStoredProofFact mapper', () => {
  it('maps all fields with snake_case elapsed_ms', () => {
    const result = toStoredProofFact({
      id: 'pf-1',
      name: 'total',
      description: 'the total',
      status: 'proven',
      value: '42',
      source: 'step-1',
      confidence: 0.95,
      tier: 'gold',
      strategy: 'direct',
      formula: 'SUM(x)',
      reason: 'computed',
      dependencies: ['dep-1'],
      elapsedMs: 123,
    })
    expect(result).toEqual({
      id: 'pf-1',
      name: 'total',
      description: 'the total',
      status: 'proven',
      value: '42',
      source: 'step-1',
      confidence: 0.95,
      tier: 'gold',
      strategy: 'direct',
      formula: 'SUM(x)',
      reason: 'computed',
      dependencies: ['dep-1'],
      elapsed_ms: 123,
    })
  })

  it('defaults dependencies to empty array when null', () => {
    const result = toStoredProofFact({
      id: 'pf-2',
      name: 'n',
      status: 'pending',
      dependencies: null,
    })
    expect(result.dependencies).toEqual([])
  })
})

describe('toPromptContext mapper', () => {
  it('maps system prompt, agent, and skills', () => {
    const result = toPromptContext({
      systemPrompt: 'You are helpful',
      activeAgent: { name: 'analyst', prompt: 'analyze' },
      activeSkills: [
        { name: 'sql', prompt: 'write sql', description: 'SQL skill' },
      ],
    })
    expect(result).toEqual({
      system_prompt: 'You are helpful',
      active_agent: { name: 'analyst', prompt: 'analyze' },
      active_skills: [{ name: 'sql', prompt: 'write sql', description: 'SQL skill' }],
    })
  })

  it('maps null agent to null and empty skills', () => {
    const result = toPromptContext({
      systemPrompt: 'sp',
      activeAgent: null,
      activeSkills: null,
    })
    expect(result.active_agent).toBeNull()
    expect(result.active_skills).toEqual([])
  })
})

describe('toObjectivesEntry mapper', () => {
  it('maps all fields', () => {
    const result = toObjectivesEntry({
      type: 'question',
      text: 'objective text',
      question: 'what is revenue?',
      answer: '42',
      mode: 'explore',
      guidance: 'look at sales',
      ts: '2025-01-01T00:00:00Z',
    })
    expect(result).toEqual({
      type: 'question',
      text: 'objective text',
      question: 'what is revenue?',
      answer: '42',
      mode: 'explore',
      guidance: 'look at sales',
      ts: '2025-01-01T00:00:00Z',
    })
  })
})

describe('toDatabaseTable mapper', () => {
  it('maps rowCount to row_count and columnCount to column_count', () => {
    const result = toDatabaseTable({
      name: 'orders',
      rowCount: 1000,
      columnCount: 5,
    })
    expect(result).toEqual({
      name: 'orders',
      row_count: 1000,
      column_count: 5,
    })
  })

  it('defaults null rowCount to null', () => {
    const result = toDatabaseTable({
      name: 't',
      rowCount: null,
      columnCount: 3,
    })
    expect(result.row_count).toBeNull()
  })
})

describe('toApiEndpoint mapper', () => {
  it('maps nested fields with snake_case', () => {
    const result = toApiEndpoint({
      name: 'getUser',
      kind: 'query',
      returnType: 'User',
      description: 'fetch user',
      httpMethod: 'GET',
      httpPath: '/users/{id}',
      fields: [
        { name: 'id', type: 'Int', description: 'user id', isRequired: true },
        { name: 'name', type: 'String', description: null, isRequired: false },
      ],
    })
    expect(result).toEqual({
      name: 'getUser',
      kind: 'query',
      return_type: 'User',
      description: 'fetch user',
      http_method: 'GET',
      http_path: '/users/{id}',
      fields: [
        { name: 'id', type: 'Int', description: 'user id', is_required: true },
        { name: 'name', type: 'String', description: undefined, is_required: false },
      ],
    })
  })

  it('defaults null fields to empty array', () => {
    const result = toApiEndpoint({
      name: 'ep',
      kind: 'mutation',
      fields: null,
    })
    expect(result.fields).toEqual([])
  })
})
