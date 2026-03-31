// Copyright (c) 2025 Kenneth Stott
// Canary: 7c8fb749-1fc4-4073-a63c-7da539b71f1e
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
  LEARNINGS_QUERY,
  SKILLS_QUERY,
  SKILL_CONTENT_QUERY,
  COMPACT_LEARNINGS,
  DELETE_LEARNING,
  CREATE_RULE,
  UPDATE_RULE,
  DELETE_RULE,
  ACTIVATE_AGENT,
  SET_ACTIVE_SKILLS,
  toLearningInfo,
  toRuleInfo,
  toSkillInfo,
} from '@/graphql/operations/learnings'

export function useLearnings(category?: string) {
  const { data, loading, error, refetch } = useQuery(LEARNINGS_QUERY, {
    variables: category ? { category } : {},
  })
  const learnings = (data?.learnings?.learnings ?? []).map(toLearningInfo)
  const rules = (data?.learnings?.rules ?? []).map(toRuleInfo)
  return { learnings, rules, loading, error, refetch }
}

export function useSkills() {
  const { data, loading, error, refetch } = useQuery(SKILLS_QUERY)
  const skills = (data?.skills?.skills ?? []).map(toSkillInfo)
  const activeSkills: string[] = data?.skills?.activeSkills ?? []
  const skillsDir: string | undefined = data?.skills?.skillsDir ?? undefined
  return { skills, activeSkills, skillsDir, loading, error, refetch }
}

export function useSkill(name: string) {
  const { data, loading, error } = useQuery(SKILL_CONTENT_QUERY, {
    variables: { name },
    skip: !name,
  })
  return {
    skill: data?.skill ?? null,
    loading,
    error,
  }
}

export function useLearningMutations() {
  const { sessionId } = useSessionContext()

  const [compactMut] = useMutation(COMPACT_LEARNINGS, {
    refetchQueries: ['Learnings'],
  })
  const [deleteLearningMut] = useMutation(DELETE_LEARNING, {
    refetchQueries: ['Learnings'],
  })
  const [createRuleMut] = useMutation(CREATE_RULE, {
    refetchQueries: ['Learnings'],
  })
  const [updateRuleMut] = useMutation(UPDATE_RULE, {
    refetchQueries: ['Learnings'],
  })
  const [deleteRuleMut] = useMutation(DELETE_RULE, {
    refetchQueries: ['Learnings'],
  })
  const [activateAgentMut] = useMutation(ACTIVATE_AGENT, {
    refetchQueries: ['Agents'],
  })
  const [setActiveSkillsMut] = useMutation(SET_ACTIVE_SKILLS, {
    refetchQueries: ['Skills'],
  })

  return {
    compactLearnings: () => compactMut(),
    deleteLearning: (learningId: string) =>
      deleteLearningMut({ variables: { learningId } }),
    createRule: (input: Record<string, unknown>) =>
      createRuleMut({ variables: { input } }),
    updateRule: (ruleId: string, input: Record<string, unknown>) =>
      updateRuleMut({ variables: { ruleId, input } }),
    deleteRule: (ruleId: string) =>
      deleteRuleMut({ variables: { ruleId, sessionId } }),
    activateAgent: (agentName?: string) =>
      activateAgentMut({ variables: { sessionId: sessionId!, agentName } }),
    setActiveSkills: (skillNames: string[]) =>
      setActiveSkillsMut({ variables: { skillNames } }),
  }
}
