// Copyright (c) 2025 Kenneth Stott
// Canary: 4ebab5aa-255e-47a4-841e-c532f1f54048
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import React, { useState, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import {
  PlusIcon,
  PencilIcon,
  TrashIcon,
  CheckIcon,
  XMarkIcon,
  UserCircleIcon,
  SparklesIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  CpuChipIcon,
  ArrowDownTrayIcon,
  ArrowsRightLeftIcon,
} from '@heroicons/react/24/outline'
import { useSkills } from '@/hooks/useLearnings'
import { AccordionSection } from '../ArtifactAccordion'
import { SkeletonLoader } from '../../common/SkeletonLoader'
import { DomainBadge } from '../../common/DomainBadge'
import { apolloClient } from '@/graphql/client'
import { MOVE_DOMAIN_SKILL, MOVE_DOMAIN_AGENT } from '@/graphql/operations/domains'
import { getAuthHeaders } from '@/config/auth-helpers'
import * as agentsApi from '@/api/agents'
import { expandSection } from '@/graphql/ui-state'

// Helper to parse YAML front-matter from markdown content
function parseFrontMatter(content: string): { frontMatter: Record<string, unknown> | null; body: string } {
  // Handle edge cases
  if (!content) {
    return { frontMatter: null, body: '' }
  }

  const match = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/)
  if (!match) {
    return { frontMatter: null, body: content }
  }

  // Simple YAML parsing for common fields
  const yamlStr = match[1]
  let body = match[2]

  // Handle case where body starts with another frontmatter block (malformed file)
  // Strip any additional frontmatter blocks from the body
  while (body.startsWith('---\n')) {
    const innerMatch = body.match(/^---\n[\s\S]*?\n---\n([\s\S]*)$/)
    if (innerMatch) {
      body = innerMatch[1]
    } else {
      break
    }
  }

  const frontMatter: Record<string, unknown> = {}

  let currentKey = ''
  let inArray = false
  let arrayValues: string[] = []

  try {
    for (const line of yamlStr.split('\n')) {
      if (line.startsWith('  - ') && inArray) {
        arrayValues.push(line.slice(4).trim())
      } else if (line.includes(':')) {
        if (inArray && currentKey) {
          frontMatter[currentKey] = arrayValues
          arrayValues = []
          inArray = false
        }
        const [key, ...valueParts] = line.split(':')
        const value = valueParts.join(':').trim()
        currentKey = key.trim()
        if (value === '') {
          inArray = true
        } else if (value.startsWith('[') && value.endsWith(']')) {
          // Inline YAML array: [item1, item2, item3]
          frontMatter[currentKey] = value.slice(1, -1).split(',').map(s => s.trim()).filter(Boolean)
        } else {
          frontMatter[currentKey] = value
        }
      }
    }
    if (inArray && currentKey) {
      frontMatter[currentKey] = arrayValues
    }
  } catch (e) {
    console.error('Failed to parse frontmatter:', e)
    return { frontMatter: null, body: content }
  }

  return { frontMatter, body }
}

// Build SKILL.md content from structured fields
function buildSkillContent(skill: { name: string; description: string; allowedTools: string[]; body: string }) {
  const toolsYaml = skill.allowedTools.length > 0
    ? `allowed-tools:\n${skill.allowedTools.map(t => `  - ${t}`).join('\n')}`
    : 'allowed-tools: []'
  return `---
name: ${skill.name}
description: ${skill.description}
${toolsYaml}
---

${skill.body}`
}

// Parse SKILL.md content into structured fields
function parseSkillContent(content: string, skillName: string) {
  const { frontMatter, body } = parseFrontMatter(content)
  return {
    name: (frontMatter?.name as string) || skillName,
    description: (frontMatter?.description as string) || '',
    allowedTools: (frontMatter?.['allowed-tools'] as string[]) || [],
    body: body.trim(),
  }
}

interface ReasoningSectionProps {
  sessionId: string
  reasoningVisible: boolean
  configVisible: boolean
  canSeeSection: (section: string) => boolean
  canWrite: (section: string) => boolean
  configLoading: boolean
  promptContext: { systemPrompt: string; activeSkills: Array<{ name: string; prompt: string; description: string }> } | null
  taskRouting: Record<string, Record<string, Array<{ model: string; provider: string }>>> | null
  allAgents: Array<{ name: string; prompt: string; domain: string; is_active: boolean; description?: string; source?: string }>
  domainList: { filename: string; name: string }[]
  // Context callbacks
  createSkill: (name: string, content: string, description: string) => Promise<void>
  updateSkill: (name: string, content: string) => Promise<void>
  deleteSkill: (name: string) => Promise<void>
  draftSkill: (sessionId: string, name: string, description: string) => Promise<{ content: string }>
  updateSystemPrompt: (sessionId: string, draft: string) => Promise<void>
  fetchAllAgents: (sessionId: string) => Promise<void>
}

export function ReasoningSection({
  sessionId,
  reasoningVisible,
  configVisible,
  canSeeSection,
  canWrite,
  configLoading,
  promptContext,
  taskRouting,
  allAgents,
  domainList,
  createSkill,
  updateSkill,
  deleteSkill,
  draftSkill,
  updateSystemPrompt,
  fetchAllAgents,
}: ReasoningSectionProps) {
  const expandArtifactSection = expandSection
  const { skills: allSkills } = useSkills()
  const hasRouting = taskRouting && Object.keys(taskRouting).length > 0

  // Collapsible state
  const [reasoningCollapsed, setReasoningCollapsed] = useState(() =>
    localStorage.getItem('constat-reasoning-collapsed') === 'true'
  )
  const [configCollapsed, setConfigCollapsed] = useState(() =>
    localStorage.getItem('constat-config-collapsed') === 'true'
  )

  // System prompt editing state
  const [editingSystemPrompt, setEditingSystemPrompt] = useState(false)
  const [systemPromptDraft, setSystemPromptDraft] = useState('')

  // Skill editing state
  const [editingSkill, setEditingSkill] = useState<{
    name: string
    description: string
    allowedTools: string[]
    body: string
  } | null>(null)
  const [expandedSkills, setExpandedSkills] = useState<Set<string>>(new Set())
  const [skillContents, setSkillContents] = useState<Record<string, string>>({})
  const [creatingSkill, setCreatingSkill] = useState(false)
  const [newSkill, setNewSkill] = useState({
    name: '',
    description: '',
    allowedTools: [] as string[],
    body: '',
  })
  const [draftingSkill, setDraftingSkill] = useState(false)
  const [newToolInput, setNewToolInput] = useState('')
  const [deletingSkill, setDeletingSkill] = useState<string | null>(null)
  const [movingSkill, setMovingSkill] = useState<string | null>(null)

  // Agent editing state
  const [draftingAgent, setDraftingAgent] = useState(false)
  const [editingAgent, setEditingAgent] = useState<{ name: string; prompt: string; description: string; skills: string[] } | null>(null)
  const [expandedAgents, setExpandedAgents] = useState<Set<string>>(new Set())
  const [agentContents, setAgentContents] = useState<Record<string, { prompt: string; description: string; skills: string[] }>>({})
  const [creatingAgent, setCreatingAgent] = useState(false)
  const [newAgent, setNewAgent] = useState({ name: '', prompt: '', description: '', skills: [] as string[] })
  const [movingAgent, setMovingAgent] = useState<string | null>(null)

  // --- Handlers ---

  const handleEditSystemPrompt = useCallback(() => {
    setSystemPromptDraft(promptContext?.systemPrompt || '')
    setEditingSystemPrompt(true)
  }, [promptContext])

  const handleSaveSystemPrompt = useCallback(async () => {
    try {
      await updateSystemPrompt(sessionId, systemPromptDraft)
      setEditingSystemPrompt(false)
    } catch (err) {
      console.error('Failed to update system prompt:', err)
    }
  }, [sessionId, systemPromptDraft, updateSystemPrompt])

  const handleMoveSkill = useCallback(async (skillName: string, fromDomain: string, toDomain: string) => {
    // Validate first
    const { data: valData } = await apolloClient.mutate({ mutation: MOVE_DOMAIN_SKILL, variables: { skillName, fromDomain, toDomain, validateOnly: true } })
    const warnings = valData?.moveDomainSkill?.warnings
    if (warnings && warnings.length > 0) {
      const ok = window.confirm(`Warning:\n${warnings.join('\n')}\n\nMove anyway?`)
      if (!ok) return
    }
    await apolloClient.mutate({ mutation: MOVE_DOMAIN_SKILL, variables: { skillName, fromDomain, toDomain } })
    setMovingSkill(null)
    apolloClient.refetchQueries({ include: ['Skills'] })
  }, [])

  const handleMoveAgent = useCallback(async (agentName: string, fromDomain: string, toDomain: string) => {
    await apolloClient.mutate({ mutation: MOVE_DOMAIN_AGENT, variables: { agentName, fromDomain, toDomain } })
    setMovingAgent(null)
    fetchAllAgents(sessionId)
  }, [sessionId, fetchAllAgents])

  const handleCreateSkill = useCallback(async () => {
    if (!newSkill.name.trim() || !newSkill.body.trim()) return
    try {
      const content = buildSkillContent({
        name: newSkill.name.trim(),
        description: newSkill.description.trim(),
        allowedTools: newSkill.allowedTools,
        body: newSkill.body.trim(),
      })
      // Create with placeholder, then update with full content
      await createSkill(newSkill.name.trim(), 'placeholder', newSkill.description.trim())
      await updateSkill(newSkill.name.trim(), content)
      setNewSkill({ name: '', description: '', allowedTools: [], body: '' })
      setNewToolInput('')
      setCreatingSkill(false)
    } catch (err) {
      console.error('Failed to create skill:', err)
    }
  }, [newSkill, createSkill, updateSkill])

  const handleDraftSkill = useCallback(async () => {
    if (!newSkill.name.trim() || !newSkill.description.trim()) return
    setDraftingSkill(true)
    try {
      const result = await draftSkill(sessionId, newSkill.name.trim(), newSkill.description.trim())
      // Parse the drafted content into structured fields
      const parsed = parseSkillContent(result.content, newSkill.name.trim())
      setNewSkill(prev => ({
        ...prev,
        description: parsed.description || prev.description,
        allowedTools: parsed.allowedTools.length > 0 ? parsed.allowedTools : prev.allowedTools,
        body: parsed.body || prev.body,
      }))
    } catch (err) {
      console.error('Failed to draft skill:', err)
    } finally {
      setDraftingSkill(false)
    }
  }, [sessionId, newSkill, draftSkill])

  const handleUpdateSkill = useCallback(async () => {
    if (!editingSkill) return
    try {
      const content = buildSkillContent(editingSkill)
      await updateSkill(editingSkill.name, content)
      setEditingSkill(null)
    } catch (err) {
      console.error('Failed to update skill:', err)
    }
  }, [editingSkill, updateSkill])

  const handleDeleteSkill = useCallback(async (skillName: string) => {
    if (!confirm(`Delete skill "${skillName}"?`)) return
    setDeletingSkill(skillName)
    try {
      await deleteSkill(skillName)
    } catch (err) {
      console.error('Failed to delete skill:', err)
    } finally {
      setDeletingSkill(null)
    }
  }, [deleteSkill])

  const handleEditSkill = useCallback(async (skillName: string) => {
    try {
      const headers = await getAuthHeaders()
      const response = await fetch(`/api/skills/${encodeURIComponent(skillName)}`, {
        headers,
        credentials: 'include',
      })
      if (response.ok) {
        const data = await response.json()
        const parsed = parseSkillContent(data.content, skillName)
        setEditingSkill(parsed)
      }
    } catch (err) {
      console.error('Failed to fetch skill content:', err)
    }
  }, [])

  const handleToggleSkillExpand = useCallback(async (skillName: string) => {
    const newExpanded = new Set(expandedSkills)
    if (newExpanded.has(skillName)) {
      newExpanded.delete(skillName)
      setExpandedSkills(newExpanded)
      return
    }

    // Expand and load content if not already loaded
    newExpanded.add(skillName)
    setExpandedSkills(newExpanded)

    if (skillContents[skillName]) return // Already loaded

    try {
      const headers = await getAuthHeaders()
      const response = await fetch(`/api/skills/${encodeURIComponent(skillName)}`, {
        headers,
        credentials: 'include',
      })
      if (response.ok) {
        const data = await response.json()
        setSkillContents(prev => ({ ...prev, [skillName]: data.content }))
      }
    } catch (err) {
      console.error('Failed to fetch skill content:', err)
    }
  }, [expandedSkills, skillContents])

  const handleDraftAgent = useCallback(async () => {
    if (!newAgent.name.trim() || !newAgent.description.trim()) return
    setDraftingAgent(true)
    try {
      const result = await agentsApi.draftAgent(sessionId, newAgent.name.trim(), newAgent.description.trim())
      setNewAgent(prev => ({ ...prev, prompt: result.prompt || '', description: result.description || prev.description, skills: result.skills || [] }))
    } catch (err) {
      console.error('Failed to draft agent:', err)
    } finally {
      setDraftingAgent(false)
    }
  }, [sessionId, newAgent])

  const handleToggleAgentExpand = useCallback(async (agentName: string) => {
    const newExpanded = new Set(expandedAgents)
    if (newExpanded.has(agentName)) {
      newExpanded.delete(agentName)
      setExpandedAgents(newExpanded)
      return
    }

    // Expand and load content if not already loaded
    newExpanded.add(agentName)
    setExpandedAgents(newExpanded)

    if (agentContents[agentName]) return // Already loaded

    try {
      const headers = await getAuthHeaders()
      const response = await fetch(
        `/api/sessions/agents/${encodeURIComponent(agentName)}?session_id=${sessionId}`,
        { headers, credentials: 'include' }
      )
      if (response.ok) {
        const data = await response.json()
        setAgentContents(prev => ({ ...prev, [agentName]: { prompt: data.prompt, description: data.description, skills: data.skills || [] } }))
      }
    } catch (err) {
      console.error('Failed to fetch agent content:', err)
    }
  }, [sessionId, expandedAgents, agentContents])

  const handleEditAgent = useCallback(async (agentName: string) => {
    try {
      const headers = await getAuthHeaders()
      const response = await fetch(
        `/api/sessions/agents/${encodeURIComponent(agentName)}?session_id=${sessionId}`,
        { headers, credentials: 'include' }
      )
      if (response.ok) {
        const data = await response.json()
        setEditingAgent({ name: data.name, prompt: data.prompt || '', description: data.description || '', skills: data.skills || [] })
      }
    } catch (err) {
      console.error('Failed to fetch agent content:', err)
    }
  }, [sessionId])

  const handleCreateAgent = useCallback(async () => {
    if (!newAgent.name.trim() || !newAgent.prompt.trim()) return
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json', ...await getAuthHeaders() }
      const response = await fetch(
        `/api/sessions/agents?session_id=${sessionId}`,
        {
          method: 'POST',
          headers,
          credentials: 'include',
          body: JSON.stringify(newAgent),
        }
      )
      if (response.ok) {
        setNewAgent({ name: '', prompt: '', description: '', skills: [] })
        setCreatingAgent(false)
        fetchAllAgents(sessionId)
      }
    } catch (err) {
      console.error('Failed to create agent:', err)
    }
  }, [sessionId, newAgent, fetchAllAgents])

  const handleUpdateAgent = useCallback(async () => {
    if (!editingAgent) return
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json', ...await getAuthHeaders() }
      const response = await fetch(
        `/api/sessions/agents/${encodeURIComponent(editingAgent.name)}?session_id=${sessionId}`,
        {
          method: 'PUT',
          headers,
          credentials: 'include',
          body: JSON.stringify({ prompt: editingAgent.prompt, description: editingAgent.description, skills: editingAgent.skills }),
        }
      )
      if (response.ok) {
        setEditingAgent(null)
        fetchAllAgents(sessionId)
      }
    } catch (err) {
      console.error('Failed to update agent:', err)
    }
  }, [sessionId, editingAgent, fetchAllAgents])

  const handleDeleteAgent = useCallback(async (agentName: string) => {
    if (!confirm(`Delete agent "${agentName}"?`)) return
    try {
      const headers = await getAuthHeaders()
      const response = await fetch(
        `/api/sessions/agents/${encodeURIComponent(agentName)}?session_id=${sessionId}`,
        {
          method: 'DELETE',
          headers,
          credentials: 'include',
        }
      )
      if (response.ok) {
        fetchAllAgents(sessionId)
      }
    } catch (err) {
      console.error('Failed to delete agent:', err)
    }
  }, [sessionId, fetchAllAgents])

  if (!reasoningVisible) return null

  return (
    <>
      {/* ═══════════════ REASONING ═══════════════ */}
      <button
        onClick={() => {
          const newVal = !reasoningCollapsed
          setReasoningCollapsed(newVal)
          localStorage.setItem('constat-reasoning-collapsed', String(newVal))
        }}
        className="w-full px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between hover:bg-gray-150 dark:hover:bg-gray-750 transition-colors"
      >
        <span className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
          Reasoning
          {configLoading && <span className="inline-block w-2.5 h-2.5 border-[1.5px] border-gray-400 border-t-transparent rounded-full animate-spin" />}
        </span>
        <ChevronRightIcon className={`w-3 h-3 text-gray-400 transition-transform ${reasoningCollapsed ? '' : 'rotate-90'}`} />
      </button>

      {!reasoningCollapsed && (
      <>

      {/* --- Configuration sub-group --- */}
      {configVisible && (
      <button
        onClick={() => {
          const newVal = !configCollapsed
          setConfigCollapsed(newVal)
          localStorage.setItem('constat-config-collapsed', String(newVal))
        }}
        className="w-full px-4 py-1.5 bg-gray-50 dark:bg-gray-800/50 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between hover:bg-gray-100 dark:hover:bg-gray-750 transition-colors"
      >
        <span className="text-[9px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider pl-2">
          Configuration
        </span>
        <ChevronRightIcon className={`w-3 h-3 text-gray-400 transition-transform ${configCollapsed ? '' : 'rotate-90'}`} />
      </button>
      )}

      {configVisible && !configCollapsed && (
      <>

      {/* Session Prompt */}
      {canSeeSection('system_prompt') && (
      <AccordionSection
        id="session-prompt"
        title="Session Prompt"
        icon={<PencilIcon className="w-4 h-4" />}
        action={
          canWrite('system_prompt') ? (
            <button
              onClick={handleEditSystemPrompt}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Edit session prompt"
            >
              <PencilIcon className="w-4 h-4" />
            </button>
          ) : undefined
        }
      >
        {editingSystemPrompt ? (
          <div className="space-y-2">
            <textarea
              value={systemPromptDraft || ''}
              onChange={(e) => setSystemPromptDraft(e.target.value)}
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 resize-none min-h-[150px]"
              placeholder="Enter session prompt..."
            />
            <div className="flex gap-1">
              <button onClick={handleSaveSystemPrompt} className="p-1 text-green-600 hover:bg-green-100 dark:hover:bg-green-900/30 rounded" title="Save">
                <CheckIcon className="w-4 h-4" />
              </button>
              <button onClick={() => setEditingSystemPrompt(false)} className="p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded" title="Cancel">
                <XMarkIcon className="w-4 h-4" />
              </button>
            </div>
          </div>
        ) : promptContext?.systemPrompt ? (
          <div className="text-sm text-gray-600 dark:text-gray-400 whitespace-pre-wrap max-h-48 overflow-y-auto">
            {promptContext.systemPrompt}
          </div>
        ) : promptContext === null ? (
          <SkeletonLoader lines={3} />
        ) : (
          <p className="text-sm text-gray-500 dark:text-gray-400 italic">No session prompt configured</p>
        )}
      </AccordionSection>
      )}

      {/* Models (task routing) */}
      {hasRouting && (
      <AccordionSection
        id="models"
        title="Models"
        count={Object.values(taskRouting!).reduce((n, routes) => n + Object.keys(routes).length, 0)}
        icon={<CpuChipIcon className="w-4 h-4" />}
      >
        <div className="space-y-3">
          {Object.entries(taskRouting!).map(([layerName, routes]) => {
            const isSystem = layerName === 'system'
            const isUser = layerName === 'user'
            const label = isSystem ? 'System Defaults' : isUser ? 'User Overrides' : layerName
            return (
            <div key={layerName}>
              <div className="flex items-center gap-1.5 mb-1.5 pb-1 border-b border-gray-200 dark:border-gray-700">
                <span className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
                  isSystem ? 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400' :
                  isUser ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300' :
                  'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                }`}>
                  {label}
                </span>
                <span className="text-[9px] text-gray-400">{Object.keys(routes).length} tasks</span>
              </div>
              <div className="space-y-1.5 pl-1">
                {Object.entries(routes).map(([taskType, models]) => (
                  <div key={taskType}>
                    <div className="text-[10px] font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-0.5">
                      {taskType.replace(/_/g, ' ')}
                    </div>
                    <div className="space-y-0.5">
                      {models.map((m, i) => (
                        <div key={i} className="flex items-center gap-1.5 text-xs text-gray-600 dark:text-gray-400">
                          <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${i === 0 ? 'bg-green-500' : 'bg-gray-300 dark:bg-gray-600'}`} />
                          <span className="font-mono text-[11px] truncate">{m.provider}/{m.model}</span>
                          {i === 0 && <span className="text-[9px] text-green-600 dark:text-green-400 flex-shrink-0">primary</span>}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            )
          })}
        </div>
      </AccordionSection>
      )}

      {/* Agents */}
      {canSeeSection('agents') && (
      <AccordionSection
        id="agents"
        title="Agents"
        count={allAgents.length}
        icon={<UserCircleIcon className="w-4 h-4" />}
        command="/agent"
        action={
          canWrite('agents') ? (
            <button
              onClick={() => {
                expandArtifactSection('agents')
                setCreatingAgent(true)
              }}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Create agent"
            >
              <PlusIcon className="w-4 h-4" />
            </button>
          ) : <div className="w-6 h-6" />
        }
      >
        {/* Create agent form */}
        {creatingAgent && (
          <div className="mb-3 p-2 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
            <input
              type="text"
              placeholder="Agent name"
              value={newAgent.name || ''}
              onChange={(e) => setNewAgent({ ...newAgent, name: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded mb-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
            <input
              type="text"
              placeholder="Description (for AI drafting or display)"
              value={newAgent.description || ''}
              onChange={(e) => setNewAgent({ ...newAgent, description: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded mb-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
            <textarea
              placeholder="Agent prompt (persona definition)..."
              value={newAgent.prompt || ''}
              onChange={(e) => setNewAgent({ ...newAgent, prompt: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded mb-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 min-h-[100px] resize-none"
            />
            {allSkills.length > 0 && (
              <div className="mb-2">
                <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Skills</label>
                <div className="flex flex-wrap gap-1">
                  {allSkills.map((skill: { name: string; description?: string; prompt?: string; filename?: string; is_active: boolean; domain?: string; source?: string }) => (
                    <button
                      key={skill.name}
                      onClick={() => {
                        const has = newAgent.skills.includes(skill.name)
                        setNewAgent({ ...newAgent, skills: has ? newAgent.skills.filter(s => s !== skill.name) : [...newAgent.skills, skill.name] })
                      }}
                      className={`px-2 py-0.5 text-xs rounded-full border ${
                        newAgent.skills.includes(skill.name)
                          ? 'bg-blue-100 dark:bg-blue-900/40 border-blue-300 dark:border-blue-600 text-blue-700 dark:text-blue-300'
                          : 'bg-gray-50 dark:bg-gray-700 border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400'
                      }`}
                      title={skill.description}
                    >
                      {skill.name}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="flex gap-1 items-center">
              <button
                onClick={handleDraftAgent}
                disabled={draftingAgent || !newAgent.name.trim() || !newAgent.description.trim()}
                className="flex items-center gap-1 px-2 py-1 text-xs text-purple-600 dark:text-purple-400 hover:bg-purple-100 dark:hover:bg-purple-900/30 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Draft with AI (requires name and description)"
              >
                <SparklesIcon className="w-3 h-3" />
                {draftingAgent ? 'Drafting...' : 'Draft with AI'}
              </button>
              <div className="flex-1" />
              <button
                onClick={handleCreateAgent}
                disabled={!newAgent.name.trim() || !newAgent.prompt.trim()}
                className="p-1 text-green-600 hover:bg-green-100 dark:hover:bg-green-900/30 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Create"
              >
                <CheckIcon className="w-4 h-4" />
              </button>
              <button
                onClick={() => {
                  setCreatingAgent(false)
                  setNewAgent({ name: '', prompt: '', description: '', skills: [] })
                }}
                className="p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                title="Cancel"
              >
                <XMarkIcon className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* Edit agent modal */}
        {editingAgent && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 w-[500px] max-h-[80vh] shadow-xl flex flex-col">
              <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
                Edit Agent: {editingAgent.name}
              </h3>
              <input
                type="text"
                placeholder="Description (optional)"
                value={editingAgent.description || ''}
                onChange={(e) => setEditingAgent({ ...editingAgent, description: e.target.value })}
                className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded mb-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
              <textarea
                value={editingAgent.prompt || ''}
                onChange={(e) => setEditingAgent({ ...editingAgent, prompt: e.target.value })}
                className="flex-1 min-h-[300px] px-3 py-2 text-sm font-mono border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 resize-none"
              />
              {allSkills.length > 0 && (
                <div className="mt-2">
                  <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Skills</label>
                  <div className="flex flex-wrap gap-1">
                    {allSkills.map((skill: { name: string; description?: string; prompt?: string; filename?: string; is_active: boolean; domain?: string; source?: string }) => (
                      <button
                        key={skill.name}
                        onClick={() => {
                          const has = editingAgent.skills.includes(skill.name)
                          setEditingAgent({ ...editingAgent, skills: has ? editingAgent.skills.filter(s => s !== skill.name) : [...editingAgent.skills, skill.name] })
                        }}
                        className={`px-2 py-0.5 text-xs rounded-full border ${
                          editingAgent.skills.includes(skill.name)
                            ? 'bg-blue-100 dark:bg-blue-900/40 border-blue-300 dark:border-blue-600 text-blue-700 dark:text-blue-300'
                            : 'bg-gray-50 dark:bg-gray-700 border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400'
                        }`}
                        title={skill.description}
                      >
                        {skill.name}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <div className="flex justify-end gap-2 mt-4">
                <button
                  onClick={() => setEditingAgent(null)}
                  className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
                >
                  Cancel
                </button>
                <button
                  onClick={handleUpdateAgent}
                  className="px-3 py-1.5 text-sm bg-primary-600 text-white rounded-md hover:bg-primary-700"
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        )}

        {allAgents.length === 0 && !creatingAgent ? (
          configLoading ? <SkeletonLoader lines={2} /> :
          <p className="text-sm text-gray-500 dark:text-gray-400">No agents defined</p>
        ) : (
          <div className="-mx-4">
            {allAgents.map((agent) => {
              const isExpanded = expandedAgents.has(agent.name)
              const content = agentContents[agent.name]

              return (
                <div key={agent.name} id={`agent-${agent.name}`} className="border-b border-gray-200 dark:border-gray-700 last:border-b-0">
                  {/* Sub-accordion header */}
                  <div className="flex items-center group">
                    <button
                      onClick={() => handleToggleAgentExpand(agent.name)}
                      className="flex-1 flex items-center gap-2 px-4 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                    >
                      <span className="flex-1 text-sm font-medium text-gray-700 dark:text-gray-300">
                        {agent.name}
                      </span>
                      <DomainBadge domain={agent.domain} />
                      <ChevronDownIcon
                        className={`w-4 h-4 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                      />
                    </button>
                    <div className="flex gap-1 pr-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => handleEditAgent(agent.name)}
                        className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 rounded"
                        title="Edit agent"
                      >
                        <PencilIcon className="w-3 h-3" />
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); setMovingAgent(movingAgent === agent.name ? null : agent.name) }}
                        className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 rounded"
                        title="Move to domain"
                      >
                        <ArrowsRightLeftIcon className="w-3 h-3" />
                      </button>
                      <button
                        onClick={() => handleDeleteAgent(agent.name)}
                        className="p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 rounded"
                        title="Delete agent"
                      >
                        <TrashIcon className="w-3 h-3" />
                      </button>
                    </div>
                  </div>

                  {/* Move-to-domain picker */}
                  {movingAgent === agent.name && (
                    <div className="flex items-center gap-2 px-4 py-1.5 bg-blue-50 dark:bg-blue-900/20 border-b border-gray-200 dark:border-gray-700">
                      <span className="text-[11px] text-gray-600 dark:text-gray-400">Move to:</span>
                      <select
                        autoFocus
                        className="text-[11px] bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded px-1.5 py-0.5"
                        defaultValue=""
                        onChange={(e) => { if (e.target.value) handleMoveAgent(agent.name, agent.domain || 'user', e.target.value) }}
                      >
                        <option value="" disabled>Select domain...</option>
                        {domainList.filter((d) => d.filename !== (agent.domain || 'user')).map((d) => (
                          <option key={d.filename} value={d.filename}>{d.name}</option>
                        ))}
                      </select>
                      <button onClick={() => setMovingAgent(null)} className="text-[11px] text-gray-400 hover:text-gray-600">Cancel</button>
                    </div>
                  )}

                  {/* Expanded content */}
                  {isExpanded && (
                    <div className="px-4 py-3 bg-gray-50 dark:bg-gray-800/50">
                      {/* Loading state */}
                      {!content && (
                        <p className="text-sm text-gray-500 dark:text-gray-400">Loading...</p>
                      )}

                      {/* Description */}
                      {content?.description && (
                        <p className="text-xs text-gray-600 dark:text-gray-400 italic mb-3">
                          {content.description}
                        </p>
                      )}

                      {/* Skills pills */}
                      {content?.skills && content.skills.length > 0 && (
                        <div className="flex flex-wrap gap-1 mb-3">
                          {content.skills.map((skillName: string) => (
                            <button
                              key={skillName}
                              onClick={() => {
                                // Scroll to skills section if visible
                                const el = document.getElementById(`skill-${skillName}`)
                                if (el) el.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
                              }}
                              className="px-2 py-0.5 text-[11px] rounded-full bg-blue-100 dark:bg-blue-900/40 border border-blue-300 dark:border-blue-600 text-blue-700 dark:text-blue-300 hover:bg-blue-200 dark:hover:bg-blue-800/60 cursor-pointer transition-colors"
                            >
                              {skillName}
                            </button>
                          ))}
                        </div>
                      )}

                      {/* Markdown-formatted prompt */}
                      {content && (
                        <div className="max-h-[400px] overflow-auto">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              p: ({ children }) => <p className="text-sm text-gray-700 dark:text-gray-300 mb-3 last:mb-0">{children}</p>,
                              h1: ({ children }) => <h1 className="text-base font-bold text-gray-900 dark:text-gray-100 mt-4 mb-2 first:mt-0">{children}</h1>,
                              h2: ({ children }) => <h2 className="text-sm font-bold text-gray-900 dark:text-gray-100 mt-3 mb-2">{children}</h2>,
                              h3: ({ children }) => <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200 mt-2 mb-1">{children}</h3>,
                              ul: ({ children }) => <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 mb-2 ml-2">{children}</ul>,
                              ol: ({ children }) => <ol className="list-decimal list-inside text-sm text-gray-700 dark:text-gray-300 mb-2 ml-2">{children}</ol>,
                              li: ({ children }) => <li className="mb-1">{children}</li>,
                              strong: ({ children }) => <strong className="font-semibold text-gray-900 dark:text-gray-100">{children}</strong>,
                              code: ({ className, children }) => {
                                const match = /language-(\w+)/.exec(className || '')
                                const isInline = !match
                                return isInline ? (
                                  <code className="bg-gray-200 dark:bg-gray-700 px-1 py-0.5 rounded text-xs font-mono">{children}</code>
                                ) : (
                                  <SyntaxHighlighter
                                    style={oneDark as Record<string, React.CSSProperties>}
                                    language={match[1]}
                                    PreTag="div"
                                    customStyle={{
                                      margin: '0.5rem 0',
                                      padding: '0.75rem',
                                      borderRadius: '0.375rem',
                                      fontSize: '0.75rem',
                                    }}
                                  >
                                    {String(children).replace(/\n$/, '')}
                                  </SyntaxHighlighter>
                                )
                              },
                            }}
                          >
                            {content.prompt}
                          </ReactMarkdown>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </AccordionSection>
      )}

      {/* Skills */}
      {canSeeSection('skills') && (
      <AccordionSection
        id="skills"
        title="Skills"
        count={allSkills.length}
        icon={<SparklesIcon className="w-4 h-4" />}
        command="/skills"
        action={
          canWrite('skills') ? (
            <button
              onClick={() => {
                expandArtifactSection('skills')
                setCreatingSkill(true)
              }}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Create skill"
            >
              <PlusIcon className="w-4 h-4" />
            </button>
          ) : <div className="w-6 h-6" />
        }
      >
        {/* Create skill form */}
        {creatingSkill && (
          <div className="mb-3 p-3 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg">
            <input
              type="text"
              placeholder="Skill name"
              value={newSkill.name || ''}
              onChange={(e) => setNewSkill({ ...newSkill, name: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 mb-2"
            />
            <input
              type="text"
              placeholder="Description (for AI drafting or display)"
              value={newSkill.description || ''}
              onChange={(e) => setNewSkill({ ...newSkill, description: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 mb-2"
            />
            {/* Allowed tools */}
            <div className="mb-2">
              <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">Allowed Tools</label>
              <div className="flex flex-wrap gap-1 mb-1">
                {newSkill.allowedTools.map((tool, idx) => (
                  <span
                    key={idx}
                    className="inline-flex items-center gap-1 px-2 py-0.5 bg-gray-200 dark:bg-gray-700 rounded text-xs text-gray-700 dark:text-gray-300"
                  >
                    {tool}
                    <button
                      onClick={() => setNewSkill({
                        ...newSkill,
                        allowedTools: newSkill.allowedTools.filter((_, i) => i !== idx)
                      })}
                      className="text-gray-400 hover:text-red-500"
                    >
                      <XMarkIcon className="w-3 h-3" />
                    </button>
                  </span>
                ))}
              </div>
              <div className="flex gap-1">
                <input
                  type="text"
                  placeholder="Add tool (e.g., run_sql)"
                  value={newToolInput || ''}
                  onChange={(e) => setNewToolInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && newToolInput.trim()) {
                      e.preventDefault()
                      if (!newSkill.allowedTools.includes(newToolInput.trim())) {
                        setNewSkill({
                          ...newSkill,
                          allowedTools: [...newSkill.allowedTools, newToolInput.trim()]
                        })
                      }
                      setNewToolInput('')
                    }
                  }}
                  className="flex-1 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                />
                <button
                  type="button"
                  onClick={() => {
                    if (newToolInput.trim() && !newSkill.allowedTools.includes(newToolInput.trim())) {
                      setNewSkill({
                        ...newSkill,
                        allowedTools: [...newSkill.allowedTools, newToolInput.trim()]
                      })
                      setNewToolInput('')
                    }
                  }}
                  className="px-2 py-1 text-xs bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
                >
                  Add
                </button>
              </div>
            </div>
            <textarea
              placeholder="Skill body (markdown with SQL patterns, metrics, domain knowledge)..."
              value={newSkill.body || ''}
              onChange={(e) => setNewSkill({ ...newSkill, body: e.target.value })}
              className="w-full px-2 py-1 text-sm font-mono border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 resize-none mb-2"
              rows={8}
            />
            <div className="flex gap-1 items-center">
              <button
                onClick={handleDraftSkill}
                disabled={draftingSkill || !newSkill.name.trim() || !newSkill.description.trim()}
                className="flex items-center gap-1 px-2 py-1 text-xs text-purple-600 dark:text-purple-400 hover:bg-purple-100 dark:hover:bg-purple-900/30 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Draft with AI (requires name and description)"
              >
                <SparklesIcon className="w-3 h-3" />
                {draftingSkill ? 'Drafting...' : 'Draft with AI'}
              </button>
              <div className="flex-1" />
              <button
                onClick={handleCreateSkill}
                disabled={!newSkill.name.trim() || !newSkill.body.trim()}
                className="p-1 text-green-600 hover:bg-green-100 dark:hover:bg-green-900/30 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Create"
              >
                <CheckIcon className="w-4 h-4" />
              </button>
              <button
                onClick={() => {
                  setCreatingSkill(false)
                  setNewSkill({ name: '', description: '', allowedTools: [], body: '' })
                  setNewToolInput('')
                }}
                className="p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                title="Cancel"
              >
                <XMarkIcon className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* Edit skill modal */}
        {editingSkill && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 w-[500px] max-h-[80vh] shadow-xl flex flex-col">
              <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
                Edit Skill: {editingSkill.name}
              </h3>
              <div className="space-y-3 flex-1 overflow-y-auto">
                <div>
                  <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">Name</label>
                  <input
                    type="text"
                    value={editingSkill.name || ''}
                    onChange={(e) => setEditingSkill({ ...editingSkill, name: e.target.value })}
                    className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">Description</label>
                  <input
                    type="text"
                    value={editingSkill.description || ''}
                    onChange={(e) => setEditingSkill({ ...editingSkill, description: e.target.value })}
                    className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">Allowed Tools</label>
                  <div className="flex flex-wrap gap-1 mb-1">
                    {editingSkill.allowedTools.map((tool, idx) => (
                      <span
                        key={idx}
                        className="inline-flex items-center gap-1 px-2 py-0.5 bg-gray-200 dark:bg-gray-700 rounded text-xs text-gray-700 dark:text-gray-300"
                      >
                        {tool}
                        <button
                          onClick={() => setEditingSkill({
                            ...editingSkill,
                            allowedTools: editingSkill.allowedTools.filter((_, i) => i !== idx)
                          })}
                          className="text-gray-400 hover:text-red-500"
                        >
                          <XMarkIcon className="w-3 h-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                  <div className="flex gap-1">
                    <input
                      type="text"
                      placeholder="Add tool (e.g., run_sql)"
                      value={newToolInput || ''}
                      onChange={(e) => setNewToolInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && newToolInput.trim()) {
                          e.preventDefault()
                          if (!editingSkill.allowedTools.includes(newToolInput.trim())) {
                            setEditingSkill({
                              ...editingSkill,
                              allowedTools: [...editingSkill.allowedTools, newToolInput.trim()]
                            })
                          }
                          setNewToolInput('')
                        }
                      }}
                      className="flex-1 px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                    <button
                      type="button"
                      onClick={() => {
                        if (newToolInput.trim() && !editingSkill.allowedTools.includes(newToolInput.trim())) {
                          setEditingSkill({
                            ...editingSkill,
                            allowedTools: [...editingSkill.allowedTools, newToolInput.trim()]
                          })
                          setNewToolInput('')
                        }
                      }}
                      className="px-2 py-1 text-xs bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded hover:bg-gray-300 dark:hover:bg-gray-600"
                    >
                      Add
                    </button>
                  </div>
                </div>
                <div className="flex-1">
                  <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">Body (Markdown)</label>
                  <textarea
                    value={editingSkill.body || ''}
                    onChange={(e) => setEditingSkill({ ...editingSkill, body: e.target.value })}
                    className="w-full min-h-[250px] px-3 py-2 text-sm font-mono border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 resize-none"
                  />
                </div>
              </div>
              <div className="flex justify-end gap-2 mt-4">
                <button
                  onClick={() => {
                    setEditingSkill(null)
                    setNewToolInput('')
                  }}
                  className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
                >
                  Cancel
                </button>
                <button
                  onClick={handleUpdateSkill}
                  className="px-3 py-1.5 text-sm bg-primary-600 text-white rounded-md hover:bg-primary-700"
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        )}

        {allSkills.length === 0 && !creatingSkill ? (
          configLoading ? <SkeletonLoader lines={2} /> :
          <p className="text-sm text-gray-500 dark:text-gray-400">No skills defined</p>
        ) : (
          <div className="-mx-4">
            {allSkills.map((skill: { name: string; description?: string; prompt?: string; filename?: string; is_active: boolean; domain?: string; source?: string }) => {
              const isExpanded = expandedSkills.has(skill.name)
              const content = skillContents[skill.name]
              const { frontMatter, body } = content ? parseFrontMatter(content) : { frontMatter: null, body: '' }
              const rawTools = frontMatter?.['allowed-tools']
              const allowedTools = Array.isArray(rawTools) ? rawTools as string[] : typeof rawTools === 'string' ? rawTools.split(',').map(s => s.trim()).filter(Boolean) : undefined

              return (
                <div key={skill.name} className={`border-b border-gray-200 dark:border-gray-700 last:border-b-0${deletingSkill === skill.name ? ' opacity-40 animate-pulse pointer-events-none' : ''}`}>
                  {/* Sub-accordion header */}
                  <div className="flex items-center group">
                    <button
                      onClick={() => handleToggleSkillExpand(skill.name)}
                      className="flex-1 flex items-center gap-2 px-4 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                    >
                      <span className="flex-1 text-sm font-medium text-gray-700 dark:text-gray-300">
                        {skill.name}
                      </span>
                      <DomainBadge domain={skill.domain} />
                      <ChevronDownIcon
                        className={`w-4 h-4 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                      />
                    </button>
                    <div className="flex gap-1 pr-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => handleEditSkill(skill.name)}
                        className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 rounded"
                        title="Edit skill"
                      >
                        <PencilIcon className="w-3 h-3" />
                      </button>
                      <button
                        onClick={async () => {
                          try {
                            const headers = await getAuthHeaders()
                            const response = await fetch(
                              `/api/skills/${encodeURIComponent(skill.name)}/download`,
                              { headers, credentials: 'include' }
                            )
                            if (!response.ok) {
                              const errorData = await response.json().catch(() => ({}))
                              alert(errorData.detail || 'Failed to download skill')
                              return
                            }
                            const blob = await response.blob()
                            const url = URL.createObjectURL(blob)
                            const a = document.createElement('a')
                            a.href = url
                            a.download = `${skill.name}.zip`
                            document.body.appendChild(a)
                            a.click()
                            document.body.removeChild(a)
                            URL.revokeObjectURL(url)
                          } catch (err) {
                            console.error('Skill download failed:', err)
                            alert('Failed to download skill.')
                          }
                        }}
                        className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 rounded"
                        title="Download skill as zip"
                      >
                        <ArrowDownTrayIcon className="w-3 h-3" />
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); setMovingSkill(movingSkill === skill.name ? null : skill.name) }}
                        className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 rounded"
                        title="Move to domain"
                      >
                        <ArrowsRightLeftIcon className="w-3 h-3" />
                      </button>
                      <button
                        onClick={() => handleDeleteSkill(skill.name)}
                        className="p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 rounded"
                        title="Delete skill"
                      >
                        <TrashIcon className="w-3 h-3" />
                      </button>
                    </div>
                  </div>

                  {/* Move-to-domain picker */}
                  {movingSkill === skill.name && (
                    <div className="flex items-center gap-2 px-4 py-1.5 bg-blue-50 dark:bg-blue-900/20 border-b border-gray-200 dark:border-gray-700">
                      <span className="text-[11px] text-gray-600 dark:text-gray-400">Move to:</span>
                      <select
                        autoFocus
                        className="text-[11px] bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded px-1.5 py-0.5"
                        defaultValue=""
                        onChange={(e) => { if (e.target.value) handleMoveSkill(skill.name, skill.domain || 'user', e.target.value) }}
                      >
                        <option value="" disabled>Select domain...</option>
                        {domainList.filter((d) => d.filename !== (skill.domain || 'user')).map((d) => (
                          <option key={d.filename} value={d.filename}>{d.name}</option>
                        ))}
                      </select>
                      <button onClick={() => setMovingSkill(null)} className="text-[11px] text-gray-400 hover:text-gray-600">Cancel</button>
                    </div>
                  )}

                  {/* Expanded content */}
                  {isExpanded && (
                    <div className="px-4 py-3 bg-gray-50 dark:bg-gray-800/50">
                      {/* Loading state */}
                      {!content && (
                        <p className="text-sm text-gray-500 dark:text-gray-400">Loading...</p>
                      )}

                      {/* Front-matter metadata */}
                      {content && frontMatter && (
                        <div className="mb-3 text-xs space-y-1">
                          {typeof frontMatter.description === 'string' && frontMatter.description && (
                            <p className="text-gray-600 dark:text-gray-400 italic">
                              {frontMatter.description}
                            </p>
                          )}
                          {allowedTools && allowedTools.length > 0 && (
                            <div className="flex flex-wrap gap-1 items-center">
                              <span className="text-gray-500 dark:text-gray-500">Tools:</span>
                              {allowedTools.map((tool) => (
                                <span
                                  key={tool}
                                  className="px-1.5 py-0.5 bg-gray-200 dark:bg-gray-700 rounded text-gray-600 dark:text-gray-400"
                                >
                                  {tool}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Markdown body */}
                      {content && (
                        <div className="max-h-[400px] overflow-auto">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              p: ({ children }) => <p className="text-sm text-gray-700 dark:text-gray-300 mb-3 last:mb-0">{children}</p>,
                              h1: ({ children }) => <h1 className="text-base font-bold text-gray-900 dark:text-gray-100 mt-4 mb-2 first:mt-0">{children}</h1>,
                              h2: ({ children }) => <h2 className="text-sm font-bold text-gray-900 dark:text-gray-100 mt-3 mb-2">{children}</h2>,
                              h3: ({ children }) => <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200 mt-2 mb-1">{children}</h3>,
                              ul: ({ children }) => <ul className="list-disc list-inside text-sm text-gray-700 dark:text-gray-300 mb-2 ml-2">{children}</ul>,
                              ol: ({ children }) => <ol className="list-decimal list-inside text-sm text-gray-700 dark:text-gray-300 mb-2 ml-2">{children}</ol>,
                              li: ({ children }) => <li className="mb-1">{children}</li>,
                              strong: ({ children }) => <strong className="font-semibold text-gray-900 dark:text-gray-100">{children}</strong>,
                              code: ({ className, children }) => {
                                const match = /language-(\w+)/.exec(className || '')
                                const isInline = !match
                                return isInline ? (
                                  <code className="bg-gray-200 dark:bg-gray-700 px-1 py-0.5 rounded text-xs font-mono">{children}</code>
                                ) : (
                                  <SyntaxHighlighter
                                    style={oneDark as Record<string, React.CSSProperties>}
                                    language={match[1]}
                                    PreTag="div"
                                    customStyle={{
                                      margin: '0.5rem 0',
                                      padding: '0.75rem',
                                      borderRadius: '0.375rem',
                                      fontSize: '0.75rem',
                                    }}
                                  >
                                    {String(children).replace(/\n$/, '')}
                                  </SyntaxHighlighter>
                                )
                              },
                            }}
                          >
                            {body}
                          </ReactMarkdown>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}</AccordionSection>
      )}

      </>
      )}

      </>
      )}
    </>
  )
}
