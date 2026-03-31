// Copyright (c) 2025 Kenneth Stott
// Canary: 0ec9c917-9379-4803-9077-8d30ec33796f
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
  DOMAINS_QUERY,
  DOMAIN_TREE_QUERY,
  DOMAIN_QUERY,
  DOMAIN_CONTENT_QUERY,
  DOMAIN_SKILLS_QUERY,
  DOMAIN_AGENTS_QUERY,
  DOMAIN_RULES_QUERY,
  CREATE_DOMAIN,
  UPDATE_DOMAIN,
  UPDATE_DOMAIN_CONTENT,
  DELETE_DOMAIN,
  PROMOTE_DOMAIN,
  MOVE_DOMAIN_SOURCE,
  MOVE_DOMAIN_SKILL,
  MOVE_DOMAIN_AGENT,
  MOVE_DOMAIN_RULE,
  toDomainInfo,
  toDomainTreeNode,
  toDomainContent,
} from '@/graphql/operations/domains'

export function useDomains() {
  const { data, loading, error } = useQuery(DOMAINS_QUERY)
  const domains = (data?.domains ?? []).map(toDomainInfo)
  return { domains, loading, error }
}

export function useDomainTree() {
  const { data, loading, error } = useQuery(DOMAIN_TREE_QUERY)
  const tree = (data?.domainTree ?? []).map(toDomainTreeNode)
  return { tree, loading, error }
}

export function useDomain(filename: string) {
  const { data, loading, error } = useQuery(DOMAIN_QUERY, {
    variables: { filename },
    skip: !filename,
  })
  const domain = data?.domain ? toDomainTreeNode(data.domain) : null
  return { domain, loading, error }
}

export function useDomainContent(filename: string) {
  const { data, loading, error } = useQuery(DOMAIN_CONTENT_QUERY, {
    variables: { filename },
    skip: !filename,
  })
  const content = data?.domainContent ? toDomainContent(data.domainContent) : null
  return { content, loading, error }
}

export function useDomainSkills(filename: string) {
  const { data, loading, error } = useQuery(DOMAIN_SKILLS_QUERY, {
    variables: { filename },
    skip: !filename,
  })
  return { skills: data?.domainSkills ?? [], loading, error }
}

export function useDomainAgents(filename: string) {
  const { data, loading, error } = useQuery(DOMAIN_AGENTS_QUERY, {
    variables: { filename },
    skip: !filename,
  })
  return { agents: data?.domainAgents ?? [], loading, error }
}

export function useDomainRules(filename: string) {
  const { data, loading, error } = useQuery(DOMAIN_RULES_QUERY, {
    variables: { filename },
    skip: !filename,
  })
  return { rules: data?.domainRules ?? [], loading, error }
}

export function useDomainMutations() {
  const { sessionId } = useSessionContext()
  const refetchLists = { refetchQueries: ['Domains', 'DomainTree'] }

  const [createDomainMut] = useMutation(CREATE_DOMAIN, refetchLists)
  const [updateDomainMut] = useMutation(UPDATE_DOMAIN, refetchLists)
  const [updateDomainContentMut] = useMutation(UPDATE_DOMAIN_CONTENT)
  const [deleteDomainMut] = useMutation(DELETE_DOMAIN, refetchLists)
  const [promoteDomainMut] = useMutation(PROMOTE_DOMAIN, refetchLists)
  const [moveDomainSourceMut] = useMutation(MOVE_DOMAIN_SOURCE, refetchLists)
  const [moveDomainSkillMut] = useMutation(MOVE_DOMAIN_SKILL, refetchLists)
  const [moveDomainAgentMut] = useMutation(MOVE_DOMAIN_AGENT, refetchLists)
  const [moveDomainRuleMut] = useMutation(MOVE_DOMAIN_RULE, refetchLists)

  return {
    createDomain: (name: string, description?: string, parentDomain?: string, initialDomains?: string[], systemPrompt?: string) =>
      createDomainMut({ variables: { name, description, parentDomain, initialDomains, systemPrompt } }),
    updateDomain: (filename: string, updates: { name?: string; description?: string; order?: number; active?: boolean }) =>
      updateDomainMut({ variables: { filename, ...updates } }),
    updateDomainContent: (filename: string, content: string) =>
      updateDomainContentMut({ variables: { filename, content } }),
    deleteDomain: (filename: string) =>
      deleteDomainMut({ variables: { filename } }),
    promoteDomain: (filename: string, targetName?: string) =>
      promoteDomainMut({ variables: { filename, targetName } }),
    moveDomainSource: (sourceType: string, sourceName: string, fromDomain: string, toDomain: string) =>
      moveDomainSourceMut({ variables: { sourceType, sourceName, fromDomain, toDomain, sessionId } }),
    moveDomainSkill: (skillName: string, fromDomain: string, toDomain: string, validateOnly?: boolean) =>
      moveDomainSkillMut({ variables: { skillName, fromDomain, toDomain, validateOnly } }),
    moveDomainAgent: (agentName: string, fromDomain: string, toDomain: string) =>
      moveDomainAgentMut({ variables: { agentName, fromDomain, toDomain } }),
    moveDomainRule: (ruleId: string, toDomain: string) =>
      moveDomainRuleMut({ variables: { ruleId, toDomain } }),
  }
}
