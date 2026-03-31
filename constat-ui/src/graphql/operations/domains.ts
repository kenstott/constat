// Copyright (c) 2025 Kenneth Stott
// Canary: 346def07-ad89-4f94-8305-642f51853d57
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { gql } from '@apollo/client'
import type { DomainTreeNode } from '@/types/api'

// -- Fragments (inline, 5-level nesting for recursive DomainTreeNode) --------

const DOMAIN_NODE_FIELDS = `
  filename
  name
  path
  description
  tier
  active
  owner
  steward
  databases
  apis
  documents
  skills
  agents
  rules
  facts
  systemPrompt
  domains
`

// 5-level inline nesting — GraphQL has no recursive fragments
const DOMAIN_TREE_CHILDREN = `
  children {
    ${DOMAIN_NODE_FIELDS}
    children {
      ${DOMAIN_NODE_FIELDS}
      children {
        ${DOMAIN_NODE_FIELDS}
        children {
          ${DOMAIN_NODE_FIELDS}
          children {
            ${DOMAIN_NODE_FIELDS}
          }
        }
      }
    }
  }
`

// -- Queries -----------------------------------------------------------------

export const DOMAINS_QUERY = gql`
  query Domains {
    domains {
      filename
      name
      description
      tier
      active
    }
  }
`

export const DOMAIN_TREE_QUERY = gql`
  query DomainTree {
    domainTree {
      ${DOMAIN_NODE_FIELDS}
      ${DOMAIN_TREE_CHILDREN}
    }
  }
`

export const DOMAIN_QUERY = gql`
  query Domain($filename: String!) {
    domain(filename: $filename) {
      ${DOMAIN_NODE_FIELDS}
      ${DOMAIN_TREE_CHILDREN}
    }
  }
`

export const DOMAIN_CONTENT_QUERY = gql`
  query DomainContent($filename: String!) {
    domainContent(filename: $filename) {
      content
      path
      filename
    }
  }
`

export const DOMAIN_SKILLS_QUERY = gql`
  query DomainSkills($filename: String!) {
    domainSkills(filename: $filename) {
      name
      description
      domain
    }
  }
`

export const DOMAIN_AGENTS_QUERY = gql`
  query DomainAgents($filename: String!) {
    domainAgents(filename: $filename) {
      name
      description
      domain
    }
  }
`

export const DOMAIN_RULES_QUERY = gql`
  query DomainRules($filename: String!) {
    domainRules(filename: $filename) {
      id
      text
      domain
    }
  }
`

export const DOMAIN_FACTS_QUERY = gql`
  query DomainFacts($filename: String!) {
    domainFacts(filename: $filename) {
      name
      value
      domain
    }
  }
`

// -- Mutations ---------------------------------------------------------------

export const CREATE_DOMAIN = gql`
  mutation CreateDomain($name: String!, $description: String, $parentDomain: String, $initialDomains: [String!], $systemPrompt: String) {
    createDomain(name: $name, description: $description, parentDomain: $parentDomain, initialDomains: $initialDomains, systemPrompt: $systemPrompt) {
      status
      filename
      name
      description
    }
  }
`

export const UPDATE_DOMAIN = gql`
  mutation UpdateDomain($filename: String!, $name: String, $description: String, $order: Int, $active: Boolean) {
    updateDomain(filename: $filename, name: $name, description: $description, order: $order, active: $active) {
      status
      filename
    }
  }
`

export const DELETE_DOMAIN = gql`
  mutation DeleteDomain($filename: String!) {
    deleteDomain(filename: $filename) {
      status
      filename
    }
  }
`

export const UPDATE_DOMAIN_CONTENT = gql`
  mutation UpdateDomainContent($filename: String!, $content: String!) {
    updateDomainContent(filename: $filename, content: $content) {
      status
      filename
      path
    }
  }
`

export const PROMOTE_DOMAIN = gql`
  mutation PromoteDomain($filename: String!, $targetName: String) {
    promoteDomain(filename: $filename, targetName: $targetName) {
      status
      filename
      newTier
    }
  }
`

export const MOVE_DOMAIN_SOURCE = gql`
  mutation MoveDomainSource($sourceType: String!, $sourceName: String!, $fromDomain: String!, $toDomain: String!, $sessionId: String) {
    moveDomainSource(sourceType: $sourceType, sourceName: $sourceName, fromDomain: $fromDomain, toDomain: $toDomain, sessionId: $sessionId) {
      status
    }
  }
`

export const MOVE_DOMAIN_SKILL = gql`
  mutation MoveDomainSkill($skillName: String!, $fromDomain: String!, $toDomain: String!, $validateOnly: Boolean) {
    moveDomainSkill(skillName: $skillName, fromDomain: $fromDomain, toDomain: $toDomain, validateOnly: $validateOnly) {
      status
      warnings
    }
  }
`

export const MOVE_DOMAIN_AGENT = gql`
  mutation MoveDomainAgent($agentName: String!, $fromDomain: String!, $toDomain: String!) {
    moveDomainAgent(agentName: $agentName, fromDomain: $fromDomain, toDomain: $toDomain) {
      status
    }
  }
`

export const MOVE_DOMAIN_RULE = gql`
  mutation MoveDomainRule($ruleId: String!, $toDomain: String!) {
    moveDomainRule(ruleId: $ruleId, toDomain: $toDomain) {
      status
    }
  }
`

// -- Mappers -----------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toDomainTreeNode(gql: any): DomainTreeNode {
  return {
    filename: gql.filename,
    name: gql.name,
    path: gql.path,
    description: gql.description,
    tier: gql.tier,
    active: gql.active,
    owner: gql.owner,
    steward: gql.steward,
    databases: gql.databases ?? [],
    apis: gql.apis ?? [],
    documents: gql.documents ?? [],
    skills: gql.skills ?? [],
    agents: gql.agents ?? [],
    rules: gql.rules ?? [],
    facts: gql.facts ?? [],
    system_prompt: gql.systemPrompt,
    domains: gql.domains ?? [],
    children: (gql.children ?? []).map(toDomainTreeNode),
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toDomainInfo(gql: any) {
  return {
    filename: gql.filename,
    name: gql.name,
    description: gql.description,
    tier: gql.tier,
    active: gql.active,
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function toDomainContent(gql: any) {
  return {
    content: gql.content,
    path: gql.path,
    filename: gql.filename,
  }
}
