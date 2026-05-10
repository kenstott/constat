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

export const CREATE_SKILL = gql`
  mutation CreateSkill($input: CreateSkillInput!) {
    createSkill(input: $input) {
      name
      description
      prompt
      filename
      isActive
    }
  }
`

export const UPDATE_SKILL = gql`
  mutation UpdateSkill($name: String!, $input: UpdateSkillInput!) {
    updateSkill(name: $name, input: $input) {
      status
      name
    }
  }
`

export const DELETE_SKILL = gql`
  mutation DeleteSkill($name: String!, $domain: String) {
    deleteSkill(name: $name, domain: $domain) {
      status
      name
    }
  }
`

export const DRAFT_SKILL = gql`
  mutation DraftSkill($sessionId: String!, $input: DraftSkillInput!) {
    draftSkill(sessionId: $sessionId, input: $input) {
      name
      content
      description
    }
  }
`

export const CREATE_SKILL_FROM_PROOF = gql`
  mutation CreateSkillFromProof($sessionId: String!, $input: CreateSkillFromProofInput!) {
    createSkillFromProof(sessionId: $sessionId, input: $input) {
      name
      content
      description
      hasScript
    }
  }
`

export const CREATE_AGENT = gql`
  mutation CreateAgent($sessionId: String!, $input: CreateAgentInput!) {
    createAgent(sessionId: $sessionId, input: $input) {
      name
      description
      isActive
    }
  }
`

export const UPDATE_AGENT = gql`
  mutation UpdateAgent($sessionId: String!, $name: String!, $input: UpdateAgentInput!) {
    updateAgent(sessionId: $sessionId, name: $name, input: $input) {
      status
      name
    }
  }
`

export const DELETE_AGENT = gql`
  mutation DeleteAgent($sessionId: String!, $name: String!) {
    deleteAgent(sessionId: $sessionId, name: $name) {
      status
      name
    }
  }
`

export const DRAFT_AGENT = gql`
  mutation DraftAgent($sessionId: String!, $input: DraftAgentInput!) {
    draftAgent(sessionId: $sessionId, input: $input) {
      name
      prompt
      description
      skills
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
