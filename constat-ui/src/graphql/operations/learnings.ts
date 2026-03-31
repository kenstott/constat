// Copyright (c) 2025 Kenneth Stott
// Canary: f87c5764-7cb8-4ce0-8fb8-2fbf3be02d91
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { gql } from '@apollo/client'

// -- Queries -----------------------------------------------------------------

export const LEARNINGS_QUERY = gql`
  query Learnings($category: String) {
    learnings(category: $category) {
      learnings {
        id
        content
        category
        source
        context
        appliedCount
        createdAt
        scope
      }
      rules {
        id
        summary
        category
        confidence
        sourceCount
        tags
        domain
        source
        scope
      }
    }
  }
`

export const SKILLS_QUERY = gql`
  query Skills {
    skills {
      skills {
        name
        description
        prompt
        filename
        isActive
        domain
        source
      }
      activeSkills
      skillsDir
    }
  }
`

export const SKILL_CONTENT_QUERY = gql`
  query SkillContent($name: String!) {
    skill(name: $name) {
      name
      content
      path
    }
  }
`

export const AGENTS_QUERY = gql`
  query Agents($sessionId: String!) {
    agents(sessionId: $sessionId) {
      name
      description
      domain
      source
      isActive
    }
  }
`

// -- Mutations ---------------------------------------------------------------

export const COMPACT_LEARNINGS = gql`
  mutation CompactLearnings {
    compactLearnings {
      status
      message
      rulesCreated
      rulesStrengthened
      rulesMerged
      learningsArchived
      groupsFound
      skippedLowConfidence
      errors
    }
  }
`

export const DELETE_LEARNING = gql`
  mutation DeleteLearning($learningId: String!) {
    deleteLearning(learningId: $learningId) {
      status
      id
    }
  }
`

export const CREATE_RULE = gql`
  mutation CreateRule($input: CreateRuleInput!) {
    createRule(input: $input) {
      id
      summary
      category
      confidence
      sourceCount
      tags
      domain
      source
    }
  }
`

export const UPDATE_RULE = gql`
  mutation UpdateRule($ruleId: String!, $input: UpdateRuleInput!) {
    updateRule(ruleId: $ruleId, input: $input) {
      id
      summary
      category
      confidence
      sourceCount
      tags
      domain
      source
    }
  }
`

export const DELETE_RULE = gql`
  mutation DeleteRule($ruleId: String!, $sessionId: String) {
    deleteRule(ruleId: $ruleId, sessionId: $sessionId) {
      status
      id
    }
  }
`

export const ACTIVATE_AGENT = gql`
  mutation ActivateAgent($sessionId: String!, $agentName: String) {
    activateAgent(sessionId: $sessionId, agentName: $agentName) {
      success
      currentAgent
      message
    }
  }
`

export const SET_ACTIVE_SKILLS = gql`
  mutation SetActiveSkills($skillNames: [String!]!) {
    setActiveSkills(skillNames: $skillNames) {
      status
      activeSkills
    }
  }
`

// -- Mappers -----------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toLearningInfo(gql: any) {
  return {
    id: gql.id,
    content: gql.content,
    category: gql.category,
    source: gql.source ?? undefined,
    context: gql.context ?? undefined,
    applied_count: gql.appliedCount ?? 0,
    created_at: gql.createdAt ?? undefined,
    scope: gql.scope ?? undefined,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toRuleInfo(gql: any) {
  return {
    id: gql.id,
    summary: gql.summary,
    category: gql.category,
    confidence: gql.confidence,
    source_count: gql.sourceCount,
    tags: gql.tags ?? [],
    domain: gql.domain ?? undefined,
    source: gql.source ?? undefined,
    scope: gql.scope ?? undefined,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toSkillInfo(gql: any) {
  return {
    name: gql.name,
    description: gql.description ?? undefined,
    prompt: gql.prompt ?? undefined,
    filename: gql.filename ?? undefined,
    is_active: gql.isActive,
    domain: gql.domain ?? undefined,
    source: gql.source ?? undefined,
  }
}
