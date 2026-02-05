// Artifact Panel container

import { useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import {
  CodeBracketIcon,
  LightBulbIcon,
  TagIcon,
  StarIcon,
  CircleStackIcon,
  GlobeAltIcon,
  DocumentTextIcon,
  PlusIcon,
  MinusIcon,
  AcademicCapIcon,
  ArrowPathIcon,
  ArrowDownTrayIcon,
  ArrowUpTrayIcon,
  PencilIcon,
  TrashIcon,
  CheckIcon,
  XMarkIcon,
  Cog6ToothIcon,
  UserCircleIcon,
  SparklesIcon,
  ChevronDownIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline'
import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import { useUIStore } from '@/store/uiStore'
import { AccordionSection } from './ArtifactAccordion'
import { TableAccordion } from './TableAccordion'
import { ArtifactItemAccordion } from './ArtifactItemAccordion'
import { CodeViewer } from './CodeViewer'
import { EntityAccordion } from './EntityAccordion'
import * as sessionsApi from '@/api/sessions'
import * as rolesApi from '@/api/roles'

type ModalType = 'database' | 'api' | 'document' | 'fact' | 'rule' | null

// Helper to parse YAML front-matter from markdown content
function parseFrontMatter(content: string): { frontMatter: Record<string, unknown> | null; body: string } {
  // Handle edge cases
  if (!content || typeof content !== 'string') {
    return { frontMatter: null, body: content || '' }
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

export function ArtifactPanel() {
  const { session } = useSessionStore()
  const { expandedArtifactSections, expandArtifactSection } = useUIStore()
  const {
    artifacts,
    tables,
    facts,
    entities,
    learnings,
    rules,
    databases,
    apis,
    documents,
    stepCodes,
    promptContext,
    allSkills,
    allRoles,
    fetchArtifacts,
    fetchTables,
    fetchFacts,
    fetchEntities,
    fetchLearnings,
    fetchDataSources,
    fetchPromptContext,
    fetchAllSkills,
    fetchAllRoles,
    createSkill,
    updateSkill,
    deleteSkill,
    draftSkill,
    userPermissions,
    fetchPermissions,
    updateSystemPrompt,
  } = useArtifactStore()

  const [showModal, setShowModal] = useState<ModalType>(null)
  const [modalInput, setModalInput] = useState({ name: '', value: '', uri: '', type: '', persist: false })
  const [compacting, setCompacting] = useState(false)
  const [editingRule, setEditingRule] = useState<{ id: string; summary: string } | null>(null)
  // Skill editing state
  // Structured skill editor state
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
  // Role editing state
  const [draftingRole, setDraftingRole] = useState(false)
  const [editingRole, setEditingRole] = useState<{ name: string; prompt: string; description: string } | null>(null)
  const [expandedRoles, setExpandedRoles] = useState<Set<string>>(new Set())
  const [roleContents, setRoleContents] = useState<Record<string, { prompt: string; description: string }>>({})
  const [creatingRole, setCreatingRole] = useState(false)
  const [newRole, setNewRole] = useState({ name: '', prompt: '', description: '' })
  // System prompt editing state (admin only)
  const [editingSystemPrompt, setEditingSystemPrompt] = useState(false)
  const [systemPromptDraft, setSystemPromptDraft] = useState('')
  // Document modal state
  const [docSourceType, setDocSourceType] = useState<'uri' | 'files'>('uri')
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  // Document viewer state
  const [viewingDocument, setViewingDocument] = useState<{ name: string; content: string; format?: string } | null>(null)
  const [loadingDocument, setLoadingDocument] = useState(false)
  // Results filter - persisted in localStorage
  const [showPublishedOnly, setShowPublishedOnly] = useState(() => {
    const stored = localStorage.getItem('constat-results-filter')
    return stored !== 'all' // Default to published only
  })
  // Collapsible section states - persisted in localStorage
  const [sourcesCollapsed, setSourcesCollapsed] = useState(() => {
    return localStorage.getItem('constat-sources-collapsed') === 'true'
  })
  const [reasoningCollapsed, setReasoningCollapsed] = useState(() => {
    return localStorage.getItem('constat-reasoning-collapsed') === 'true'
  })

  // Persist filter preference
  const toggleResultsFilter = () => {
    const newValue = !showPublishedOnly
    setShowPublishedOnly(newValue)
    localStorage.setItem('constat-results-filter', newValue ? 'published' : 'all')
  }

  // Fetch permissions on mount
  useEffect(() => {
    fetchPermissions()
  }, [fetchPermissions])

  // Fetch data when session changes
  useEffect(() => {
    if (session) {
      fetchArtifacts(session.session_id)
      fetchTables(session.session_id)
      fetchFacts(session.session_id)
      fetchEntities(session.session_id)
      fetchLearnings()
      fetchDataSources(session.session_id)
      fetchPromptContext(session.session_id)
      fetchAllSkills()
      fetchAllRoles(session.session_id)
    }
  }, [session, fetchArtifacts, fetchTables, fetchFacts, fetchEntities, fetchLearnings, fetchDataSources, fetchPromptContext, fetchAllSkills, fetchAllRoles])

  // Handlers
  const handleForgetFact = async (factName: string) => {
    if (!session) return
    await sessionsApi.forgetFact(session.session_id, factName)
    fetchFacts(session.session_id)
  }

  const handlePersistFact = async (factName: string) => {
    if (!session) return
    await sessionsApi.persistFact(session.session_id, factName)
    fetchFacts(session.session_id)
  }

  const handleAddFact = async () => {
    if (!session || !modalInput.name || !modalInput.value) return
    await sessionsApi.addFact(session.session_id, modalInput.name, modalInput.value, modalInput.persist)
    fetchFacts(session.session_id)
    setShowModal(null)
    setModalInput({ name: '', value: '', uri: '', type: '', persist: false })
  }

  const handleAddDatabase = async () => {
    if (!session || !modalInput.name || !modalInput.uri) return
    await sessionsApi.addDatabase(session.session_id, {
      name: modalInput.name,
      uri: modalInput.uri,
      type: modalInput.type || 'duckdb',
    })
    fetchDataSources(session.session_id)
    setShowModal(null)
    setModalInput({ name: '', value: '', uri: '', type: '', persist: false })
  }

  const handleAddDocument = async () => {
    if (!session) return

    if (docSourceType === 'files') {
      // Upload files
      if (selectedFiles.length === 0) return
      setUploading(true)
      try {
        await sessionsApi.uploadDocuments(session.session_id, selectedFiles)
        fetchDataSources(session.session_id)
        fetchEntities(session.session_id)  // Refresh entities after indexing
        setShowModal(null)
        setSelectedFiles([])
        setDocSourceType('uri')
      } finally {
        setUploading(false)
      }
    } else {
      // Add from URI
      if (!modalInput.name || !modalInput.uri) return
      await sessionsApi.addFileRef(session.session_id, {
        name: modalInput.name,
        uri: modalInput.uri,
      })
      fetchDataSources(session.session_id)
      fetchEntities(session.session_id)  // Refresh entities after indexing
      setShowModal(null)
    }
    setModalInput({ name: '', value: '', uri: '', type: '', persist: false })
  }

  const handleDeleteDocument = async (docName: string) => {
    if (!session) return
    if (!confirm(`Delete document "${docName}" and its extracted entities?`)) return

    try {
      await sessionsApi.deleteFileRef(session.session_id, docName)
      fetchDataSources(session.session_id)
      fetchEntities(session.session_id)  // Refresh entities after deletion
    } catch (err) {
      console.error('Failed to delete document:', err)
      alert('Failed to delete document. Please try again.')
    }
  }

  const handleViewDocument = async (documentName: string) => {
    if (!session) return
    setLoadingDocument(true)
    try {
      const doc = await sessionsApi.getDocument(session.session_id, documentName)

      // For file types (PDF, Office docs), open via file serving endpoint
      if (doc.type === 'file' && doc.path) {
        // Open file in new tab via file serving endpoint
        const fileUrl = `/api/sessions/${session.session_id}/file?path=${encodeURIComponent(doc.path)}`
        window.open(fileUrl, '_blank')
        return
      }

      // For content types (markdown, text), show in modal
      setViewingDocument({
        name: doc.name || documentName,
        content: doc.content || '',
        format: doc.format
      })
    } catch (err) {
      console.error('Failed to load document:', err)
      alert('Failed to load document. Please try again.')
    } finally {
      setLoadingDocument(false)
    }
  }

  const openModal = (type: ModalType) => {
    setModalInput({ name: '', value: '', uri: '', type: '', persist: false })
    setDocSourceType('uri')
    setSelectedFiles([])
    setShowModal(type)
  }

  const handleAddRule = async () => {
    if (!modalInput.value.trim()) return
    await useArtifactStore.getState().addRule(modalInput.value.trim())
    setShowModal(null)
    setModalInput({ name: '', value: '', uri: '', type: '', persist: false })
  }

  const handleUpdateRule = async () => {
    if (!editingRule || !editingRule.summary.trim()) return
    await useArtifactStore.getState().updateRule(editingRule.id, editingRule.summary.trim())
    setEditingRule(null)
  }

  const handleDeleteRule = async (ruleId: string) => {
    await useArtifactStore.getState().deleteRule(ruleId)
  }

  const handleDeleteLearning = async (learningId: string) => {
    await useArtifactStore.getState().deleteLearning(learningId)
  }

  // Build SKILL.md content from structured fields
  const buildSkillContent = (skill: { name: string; description: string; allowedTools: string[]; body: string }) => {
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
  const parseSkillContent = (content: string, skillName: string) => {
    const { frontMatter, body } = parseFrontMatter(content)
    return {
      name: (frontMatter?.name as string) || skillName,
      description: (frontMatter?.description as string) || '',
      allowedTools: (frontMatter?.['allowed-tools'] as string[]) || [],
      body: body.trim(),
    }
  }

  const handleCreateSkill = async () => {
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
  }

  const handleDraftSkill = async () => {
    if (!session || !newSkill.name.trim() || !newSkill.description.trim()) return
    setDraftingSkill(true)
    try {
      const result = await draftSkill(session.session_id, newSkill.name.trim(), newSkill.description.trim())
      // Parse the drafted content into structured fields
      const parsed = parseSkillContent(result.content, newSkill.name.trim())
      setNewSkill(prev => ({
        ...prev,
        description: parsed.description || prev.description,
        allowedTools: parsed.allowedTools.length > 0 ? parsed.allowedTools : prev.allowedTools,
        body: parsed.body,
      }))
    } catch (err) {
      console.error('Failed to draft skill:', err)
    } finally {
      setDraftingSkill(false)
    }
  }

  const handleDraftRole = async () => {
    if (!session || !newRole.name.trim() || !newRole.description.trim()) return
    setDraftingRole(true)
    try {
      const result = await rolesApi.draftRole(session.session_id, newRole.name.trim(), newRole.description.trim())
      setNewRole(prev => ({ ...prev, prompt: result.prompt, description: result.description || prev.description }))
    } catch (err) {
      console.error('Failed to draft role:', err)
    } finally {
      setDraftingRole(false)
    }
  }

  const handleUpdateSkill = async () => {
    if (!editingSkill) return
    try {
      const content = buildSkillContent(editingSkill)
      await updateSkill(editingSkill.name, content)
      setEditingSkill(null)
    } catch (err) {
      console.error('Failed to update skill:', err)
    }
  }

  const handleDeleteSkill = async (skillName: string) => {
    if (!confirm(`Delete skill "${skillName}"?`)) return
    try {
      await deleteSkill(skillName)
    } catch (err) {
      console.error('Failed to delete skill:', err)
    }
  }

  const handleToggleRoleExpand = async (roleName: string) => {
    if (!session) return

    const newExpanded = new Set(expandedRoles)
    if (newExpanded.has(roleName)) {
      newExpanded.delete(roleName)
      setExpandedRoles(newExpanded)
      return
    }

    // Expand and load content if not already loaded
    newExpanded.add(roleName)
    setExpandedRoles(newExpanded)

    if (roleContents[roleName]) return // Already loaded

    try {
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = {}
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) headers['Authorization'] = `Bearer ${token}`
      }
      const response = await fetch(
        `/api/sessions/roles/${encodeURIComponent(roleName)}?session_id=${session.session_id}`,
        { headers, credentials: 'include' }
      )
      if (response.ok) {
        const data = await response.json()
        setRoleContents(prev => ({ ...prev, [roleName]: { prompt: data.prompt, description: data.description } }))
      }
    } catch (err) {
      console.error('Failed to fetch role content:', err)
    }
  }

  const handleEditRole = async (roleName: string) => {
    if (!session) return
    try {
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = {}
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) headers['Authorization'] = `Bearer ${token}`
      }
      const response = await fetch(
        `/api/sessions/roles/${encodeURIComponent(roleName)}?session_id=${session.session_id}`,
        { headers, credentials: 'include' }
      )
      if (response.ok) {
        const data = await response.json()
        setEditingRole({ name: data.name, prompt: data.prompt, description: data.description })
      }
    } catch (err) {
      console.error('Failed to fetch role content:', err)
    }
  }

  const handleCreateRole = async () => {
    if (!session || !newRole.name.trim() || !newRole.prompt.trim()) return
    try {
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) headers['Authorization'] = `Bearer ${token}`
      }
      const response = await fetch(
        `/api/sessions/roles?session_id=${session.session_id}`,
        {
          method: 'POST',
          headers,
          credentials: 'include',
          body: JSON.stringify(newRole),
        }
      )
      if (response.ok) {
        setNewRole({ name: '', prompt: '', description: '' })
        setCreatingRole(false)
        fetchAllRoles(session.session_id)
      }
    } catch (err) {
      console.error('Failed to create role:', err)
    }
  }

  const handleUpdateRole = async () => {
    if (!session || !editingRole) return
    try {
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) headers['Authorization'] = `Bearer ${token}`
      }
      const response = await fetch(
        `/api/sessions/roles/${encodeURIComponent(editingRole.name)}?session_id=${session.session_id}`,
        {
          method: 'PUT',
          headers,
          credentials: 'include',
          body: JSON.stringify({ prompt: editingRole.prompt, description: editingRole.description }),
        }
      )
      if (response.ok) {
        setEditingRole(null)
        fetchAllRoles(session.session_id)
      }
    } catch (err) {
      console.error('Failed to update role:', err)
    }
  }

  const handleDeleteRole = async (roleName: string) => {
    if (!session || !confirm(`Delete role "${roleName}"?`)) return
    try {
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = {}
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) headers['Authorization'] = `Bearer ${token}`
      }
      const response = await fetch(
        `/api/sessions/roles/${encodeURIComponent(roleName)}?session_id=${session.session_id}`,
        {
          method: 'DELETE',
          headers,
          credentials: 'include',
        }
      )
      if (response.ok) {
        fetchAllRoles(session.session_id)
      }
    } catch (err) {
      console.error('Failed to delete role:', err)
    }
  }

  const handleEditSkill = async (skillName: string) => {
    try {
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = {}
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) headers['Authorization'] = `Bearer ${token}`
      }
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
  }

  const handleToggleSkillExpand = async (skillName: string) => {
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
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = {}
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) headers['Authorization'] = `Bearer ${token}`
      }
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
  }

  const handleEditSystemPrompt = () => {
    setSystemPromptDraft(promptContext?.systemPrompt || '')
    setEditingSystemPrompt(true)
  }

  const handleSaveSystemPrompt = async () => {
    if (!session) return
    try {
      await updateSystemPrompt(session.session_id, systemPromptDraft)
      setEditingSystemPrompt(false)
    } catch (err) {
      console.error('Failed to update system prompt:', err)
    }
  }

  // Unified Results: combine tables and artifacts into a flat list
  type ResultItem =
    | { type: 'table'; data: typeof tables[0]; created_at: string; is_published: boolean }
    | { type: 'artifact'; data: typeof artifacts[0]; created_at: string; is_published: boolean }

  // Types to exclude when showing all (non-result artifacts)
  // Note: 'table' is excluded because tables are already shown via the tables array
  const excludedArtifactTypes = new Set(['code', 'error', 'output', 'table'])

  // Build unified results list (filter out code, error, output artifacts)
  const allResults: ResultItem[] = [
    ...tables.map((t) => ({
      type: 'table' as const,
      data: t,
      created_at: '', // Tables don't have created_at
      is_published: t.is_starred || false,
    })),
    ...artifacts
      .filter((a) => !excludedArtifactTypes.has(a.artifact_type))
      .map((a) => ({
        type: 'artifact' as const,
        data: a,
        created_at: a.created_at || '',
        is_published: a.is_starred || a.is_key_result || false,
      })),
  ]

  // Sort by step_number descending (most recent steps first), then by name
  allResults.sort((a, b) => {
    const stepDiff = (b.data.step_number || 0) - (a.data.step_number || 0)
    if (stepDiff !== 0) return stepDiff
    return a.data.name.localeCompare(b.data.name)
  })

  // Filter based on toggle
  const displayedResults = showPublishedOnly
    ? allResults.filter((r) => r.is_published)
    : allResults

  const totalCount = allResults.length

  // Auto-expand: find best item to expand
  const isResultsSectionExpanded = expandedArtifactSections.includes('results')
  let bestResultId: string | null = null

  if (isResultsSectionExpanded && displayedResults.length > 0) {
    // Prefer published items with priority keywords
    const hasPriorityKeyword = (name?: string, title?: string): boolean => {
      const text = `${name || ''} ${title || ''}`.toLowerCase()
      return ['final', 'recommended', 'answer', 'result', 'conclusion'].some(kw => text.includes(kw))
    }

    const withKeyword = displayedResults.find((r) => {
      const title = r.type === 'artifact' ? r.data.title : undefined
      return hasPriorityKeyword(r.data.name, title)
    })
    const best = withKeyword || displayedResults[0]
    bestResultId = best.type === 'table' ? `table-${best.data.name}` : `artifact-${best.data.id}`
  }

  if (!session) {
    return (
      <div className="flex-1 flex items-center justify-center p-4">
        <p className="text-sm text-gray-500 dark:text-gray-400">
          No session active
        </p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto">
      {/* Add Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 w-80 shadow-xl">
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
              Add {showModal === 'fact' ? 'Fact' : showModal === 'database' ? 'Database' : showModal === 'api' ? 'API' : showModal === 'rule' ? 'Rule' : 'Document'}
            </h3>
            <div className="space-y-3">
              {showModal === 'rule' ? (
                <textarea
                  placeholder="Enter the rule text..."
                  value={modalInput.value}
                  onChange={(e) => setModalInput({ ...modalInput, value: e.target.value })}
                  className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 resize-none"
                  rows={3}
                />
              ) : showModal === 'document' ? (
                <>
                  {/* Document source type selector */}
                  <div className="flex gap-4">
                    <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 cursor-pointer">
                      <input
                        type="radio"
                        name="docSourceType"
                        checked={docSourceType === 'uri'}
                        onChange={() => setDocSourceType('uri')}
                        className="text-primary-600"
                      />
                      From URI
                    </label>
                    <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 cursor-pointer">
                      <input
                        type="radio"
                        name="docSourceType"
                        checked={docSourceType === 'files'}
                        onChange={() => setDocSourceType('files')}
                        className="text-primary-600"
                      />
                      From Files
                    </label>
                  </div>

                  {docSourceType === 'uri' ? (
                    <>
                      <input
                        type="text"
                        placeholder="Name"
                        value={modalInput.name}
                        onChange={(e) => setModalInput({ ...modalInput, name: e.target.value })}
                        className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                      <input
                        type="text"
                        placeholder="URI (file:// or http://)"
                        value={modalInput.uri}
                        onChange={(e) => setModalInput({ ...modalInput, uri: e.target.value })}
                        className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                    </>
                  ) : (
                    <>
                      <input
                        type="file"
                        multiple
                        accept=".md,.txt,.pdf,.docx,.html,.htm,.pptx,.xlsx,.csv,.tsv,.parquet,.json"
                        onChange={(e) => setSelectedFiles(Array.from(e.target.files || []))}
                        className="w-full text-sm text-gray-600 dark:text-gray-400 file:mr-3 file:py-1.5 file:px-3 file:rounded-md file:border-0 file:text-sm file:bg-primary-50 file:text-primary-700 dark:file:bg-primary-900/30 dark:file:text-primary-400 hover:file:bg-primary-100 dark:hover:file:bg-primary-900/50 cursor-pointer"
                      />
                      {selectedFiles.length > 0 && (
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {selectedFiles.length} file{selectedFiles.length !== 1 ? 's' : ''} selected:
                          <ul className="mt-1 space-y-0.5">
                            {selectedFiles.map((f, i) => (
                              <li key={i} className="truncate">{f.name}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </>
                  )}
                </>
              ) : (
                <>
                  <input
                    type="text"
                    placeholder="Name"
                    value={modalInput.name}
                    onChange={(e) => setModalInput({ ...modalInput, name: e.target.value })}
                    className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                  {showModal === 'fact' ? (
                    <>
                      <input
                        type="text"
                        placeholder="Value"
                        value={modalInput.value}
                        onChange={(e) => setModalInput({ ...modalInput, value: e.target.value })}
                        className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                        <input
                          type="checkbox"
                          checked={modalInput.persist}
                          onChange={(e) => setModalInput({ ...modalInput, persist: e.target.checked })}
                          className="rounded border-gray-300 dark:border-gray-600"
                        />
                        Save for future sessions
                      </label>
                    </>
                  ) : (
                    <input
                      type="text"
                      placeholder="URI / Path"
                      value={modalInput.uri}
                      onChange={(e) => setModalInput({ ...modalInput, uri: e.target.value })}
                      className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                  )}
                </>
              )}
              {showModal === 'database' && (
                <select
                  value={modalInput.type}
                  onChange={(e) => setModalInput({ ...modalInput, type: e.target.value })}
                  className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                >
                  <option value="">Type (optional)</option>
                  <option value="duckdb">DuckDB</option>
                  <option value="sqlite">SQLite</option>
                  <option value="postgresql">PostgreSQL</option>
                  <option value="mysql">MySQL</option>
                </select>
              )}
            </div>
            <div className="flex justify-end gap-2 mt-4">
              <button
                onClick={() => setShowModal(null)}
                className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  if (showModal === 'fact') handleAddFact()
                  else if (showModal === 'database') handleAddDatabase()
                  else if (showModal === 'document') handleAddDocument()
                  else if (showModal === 'rule') handleAddRule()
                  else setShowModal(null) // API - not implemented yet
                }}
                disabled={uploading || (showModal === 'document' && docSourceType === 'files' && selectedFiles.length === 0)}
                className="px-3 py-1.5 text-sm bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {uploading && (
                  <div className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
                )}
                {uploading ? 'Uploading...' : 'Add'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Document Viewer Modal */}
      {(viewingDocument || loadingDocument) && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-3xl max-h-[80vh] flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  {loadingDocument ? 'Loading...' : viewingDocument?.name}
                </h3>
                {viewingDocument?.format && (
                  <span className="text-[10px] px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 rounded">
                    {viewingDocument.format}
                  </span>
                )}
              </div>
              <button
                onClick={() => setViewingDocument(null)}
                className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded transition-colors"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              {loadingDocument ? (
                <div className="flex items-center justify-center py-8">
                  <div className="w-6 h-6 border-2 border-primary-500 border-t-transparent rounded-full animate-spin" />
                </div>
              ) : viewingDocument?.content ? (
                viewingDocument.format === 'markdown' ? (
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {viewingDocument.content}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <pre className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap font-mono">
                    {viewingDocument.content}
                  </pre>
                )
              ) : (
                <p className="text-sm text-gray-500 dark:text-gray-400">No content available</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ═══════════════ RESULTS ═══════════════ */}
      {totalCount > 0 && (
        <>
          <AccordionSection
            id="results"
            title="Results"
            count={displayedResults.length}
            icon={<StarIcon className="w-4 h-4" />}
            command="/results"
            action={
              <button
                onClick={(e) => { e.stopPropagation(); toggleResultsFilter(); }}
                className={`text-[10px] px-2 py-0.5 rounded-full transition-colors ${
                  showPublishedOnly
                    ? 'bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-400'
                    : 'bg-gray-200 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                }`}
                title={showPublishedOnly ? 'Showing published only. Click to show all.' : 'Showing all. Click to show published only.'}
              >
                {showPublishedOnly ? 'published' : 'all'}
              </button>
            }
          >
            {displayedResults.length === 0 ? (
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {showPublishedOnly && totalCount > 0
                  ? 'No published results yet. Click toggle to show all.'
                  : 'No results yet'}
              </p>
            ) : (
              <div className="space-y-2">
                {displayedResults.map((result) =>
                  result.type === 'table' ? (
                    <TableAccordion
                      key={`table-${result.data.name}`}
                      table={result.data}
                      initiallyOpen={bestResultId === `table-${result.data.name}`}
                    />
                  ) : (
                    <ArtifactItemAccordion
                      key={`artifact-${result.data.id}`}
                      artifact={result.data}
                      initiallyOpen={bestResultId === `artifact-${result.data.id}`}
                    />
                  )
                )}
              </div>
            )}
          </AccordionSection>
        </>
      )}

      {/* ═══════════════ SOURCES ═══════════════ */}
      <button
        onClick={() => {
          const newVal = !sourcesCollapsed
          setSourcesCollapsed(newVal)
          localStorage.setItem('constat-sources-collapsed', String(newVal))
        }}
        className="w-full px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between hover:bg-gray-150 dark:hover:bg-gray-750 transition-colors"
      >
        <span className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          Sources
        </span>
        <ChevronRightIcon className={`w-3 h-3 text-gray-400 transition-transform ${sourcesCollapsed ? '' : 'rotate-90'}`} />
      </button>

      {/* Databases */}
      {!sourcesCollapsed && (
      <>
      <AccordionSection
        id="databases"
        title="Databases"
        count={databases.length}
        icon={<CircleStackIcon className="w-4 h-4" />}
        command="/databases"
        action={
          <button
            onClick={() => openModal('database')}
            className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            title="Add database"
          >
            <PlusIcon className="w-4 h-4" />
          </button>
        }
      >
        {databases.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">No databases configured</p>
        ) : (
          <div className="space-y-2">
            {databases.map((db) => (
              <div
                key={db.name}
                className="group p-2 bg-gray-50 dark:bg-gray-800/50 rounded-md"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      {db.name}
                    </span>
                    {db.source && db.source !== 'config' && (
                      <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                        db.source === 'session'
                          ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                          : 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400'
                      }`}>
                        {db.source === 'session' ? 'session' : db.source.replace('.yaml', '')}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs px-1.5 py-0.5 rounded bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400">
                      {db.type}
                    </span>
                    {/* Only show delete for session-added databases (is_dynamic) */}
                    {db.is_dynamic && (
                      <button
                        onClick={async () => {
                          if (!session) return
                          if (!confirm(`Remove database "${db.name}" from this session?`)) return
                          try {
                            await sessionsApi.removeDatabase(session.session_id, db.name)
                            fetchDataSources(session.session_id)
                          } catch (err) {
                            console.error('Failed to remove database:', err)
                            alert('Failed to remove database. Please try again.')
                          }
                        }}
                        className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 transition-all"
                        title="Remove database"
                      >
                        <TrashIcon className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>
                {db.table_count !== undefined && db.table_count > 0 && (
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {db.table_count} tables
                  </p>
                )}
                {db.description && (
                  <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                    {db.description}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </AccordionSection>

      {/* APIs */}
      <AccordionSection
        id="apis"
        title="APIs"
        count={apis.length}
        icon={<GlobeAltIcon className="w-4 h-4" />}
        command="/apis"
        action={
          <button
            onClick={() => openModal('api')}
            className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            title="Add API"
          >
            <PlusIcon className="w-4 h-4" />
          </button>
        }
      >
        {apis.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">No APIs configured</p>
        ) : (
          <div className="space-y-2">
            {apis.map((api) => (
              <div
                key={api.name}
                className="p-2 bg-gray-50 dark:bg-gray-800/50 rounded-md"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      {api.name}
                    </span>
                    {api.source && api.source !== 'config' && (
                      <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                        api.source === 'session'
                          ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                          : 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400'
                      }`}>
                        {api.source === 'session' ? 'session' : api.source.replace('.yaml', '')}
                      </span>
                    )}
                  </div>
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded ${
                      api.connected
                        ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                        : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                    }`}
                  >
                    {api.connected ? 'Available' : 'Pending'}
                  </span>
                </div>
                {api.type && (
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {api.type}
                  </p>
                )}
                {api.description && (
                  <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                    {api.description}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </AccordionSection>

      {/* Documents */}
      <AccordionSection
        id="documents"
        title="Documents"
        count={documents.length}
        icon={<DocumentTextIcon className="w-4 h-4" />}
        command="/docs"
        action={
          <button
            onClick={() => openModal('document')}
            className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            title="Add document"
          >
            <PlusIcon className="w-4 h-4" />
          </button>
        }
      >
        {documents.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">No documents indexed</p>
        ) : (
          <div className="space-y-2">
            {documents.map((doc) => (
              <div
                key={doc.name}
                className="group p-2 bg-gray-50 dark:bg-gray-800/50 rounded-md"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleViewDocument(doc.name)}
                      className="text-sm font-medium text-blue-600 dark:text-blue-400 hover:underline cursor-pointer"
                    >
                      {doc.name}
                    </button>
                    {doc.source && doc.source !== 'config' && (
                      <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                        doc.source === 'session'
                          ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                          : 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400'
                      }`}>
                        {doc.source === 'session' ? 'session' : doc.source.replace('.yaml', '')}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-xs px-1.5 py-0.5 rounded ${
                        doc.indexed
                          ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                          : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                      }`}
                    >
                      {doc.indexed ? 'Indexed' : 'Pending'}
                    </span>
                    {/* Only show delete for session-added documents (not from_config) */}
                    {!doc.from_config && (
                      <button
                        onClick={() => handleDeleteDocument(doc.name)}
                        className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 transition-all"
                        title="Remove document"
                      >
                        <TrashIcon className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>
                {doc.type && (
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {doc.type}
                  </p>
                )}
                {doc.description && (
                  <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                    {doc.description}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </AccordionSection>
      </>
      )}

      {/* ═══════════════ REASONING ═══════════════ */}
      <button
        onClick={() => {
          const newVal = !reasoningCollapsed
          setReasoningCollapsed(newVal)
          localStorage.setItem('constat-reasoning-collapsed', String(newVal))
        }}
        className="w-full px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between hover:bg-gray-150 dark:hover:bg-gray-750 transition-colors"
      >
        <span className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          Reasoning
        </span>
        <ChevronRightIcon className={`w-3 h-3 text-gray-400 transition-transform ${reasoningCollapsed ? '' : 'rotate-90'}`} />
      </button>

      {!reasoningCollapsed && (
      <>

      {/* System Prompt - show when there's content or user is admin */}
      {(promptContext?.systemPrompt || userPermissions.isAdmin) && (
        <AccordionSection
          id="system-prompt"
          title="System Prompt"
          icon={<Cog6ToothIcon className="w-4 h-4" />}
          command="/system"
          action={
            userPermissions.isAdmin ? (
              <button
                onClick={handleEditSystemPrompt}
                className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                title="Edit system prompt"
              >
                <PencilIcon className="w-4 h-4" />
              </button>
            ) : (
              <div className="w-6 h-6" />
            )
          }
        >
          {editingSystemPrompt ? (
            <div className="space-y-2">
              <textarea
                value={systemPromptDraft}
                onChange={(e) => setSystemPromptDraft(e.target.value)}
                className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 resize-none min-h-[150px]"
                placeholder="Enter system prompt..."
              />
              <div className="flex gap-1">
                <button
                  onClick={handleSaveSystemPrompt}
                  className="p-1 text-green-600 hover:bg-green-100 dark:hover:bg-green-900/30 rounded"
                  title="Save"
                >
                  <CheckIcon className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setEditingSystemPrompt(false)}
                  className="p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                  title="Cancel"
                >
                  <XMarkIcon className="w-4 h-4" />
                </button>
              </div>
            </div>
          ) : promptContext?.systemPrompt ? (
            <div className="text-sm text-gray-600 dark:text-gray-400 whitespace-pre-wrap max-h-48 overflow-y-auto">
              {promptContext.systemPrompt}
            </div>
          ) : (
            <p className="text-sm text-gray-500 dark:text-gray-400 italic">No system prompt configured</p>
          )}
        </AccordionSection>
      )}

      {/* Roles - always show with create/edit/delete */}
      <AccordionSection
        id="roles"
        title="Roles"
        count={allRoles.length}
        icon={<UserCircleIcon className="w-4 h-4" />}
        command="/role"
        action={
          <button
            onClick={() => {
              expandArtifactSection('roles')
              setCreatingRole(true)
            }}
            className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            title="Create role"
          >
            <PlusIcon className="w-4 h-4" />
          </button>
        }
      >
        {/* Create role form */}
        {creatingRole && (
          <div className="mb-3 p-2 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
            <input
              type="text"
              placeholder="Role name"
              value={newRole.name}
              onChange={(e) => setNewRole({ ...newRole, name: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded mb-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
            <input
              type="text"
              placeholder="Description (for AI drafting or display)"
              value={newRole.description}
              onChange={(e) => setNewRole({ ...newRole, description: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded mb-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
            <textarea
              placeholder="Role prompt (persona definition)..."
              value={newRole.prompt}
              onChange={(e) => setNewRole({ ...newRole, prompt: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded mb-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 min-h-[100px] resize-none"
            />
            <div className="flex gap-1 items-center">
              <button
                onClick={handleDraftRole}
                disabled={draftingRole || !newRole.name.trim() || !newRole.description.trim()}
                className="flex items-center gap-1 px-2 py-1 text-xs text-purple-600 dark:text-purple-400 hover:bg-purple-100 dark:hover:bg-purple-900/30 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Draft with AI (requires name and description)"
              >
                <SparklesIcon className="w-3 h-3" />
                {draftingRole ? 'Drafting...' : 'Draft with AI'}
              </button>
              <div className="flex-1" />
              <button
                onClick={handleCreateRole}
                disabled={!newRole.name.trim() || !newRole.prompt.trim()}
                className="p-1 text-green-600 hover:bg-green-100 dark:hover:bg-green-900/30 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Create"
              >
                <CheckIcon className="w-4 h-4" />
              </button>
              <button
                onClick={() => {
                  setCreatingRole(false)
                  setNewRole({ name: '', prompt: '', description: '' })
                }}
                className="p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                title="Cancel"
              >
                <XMarkIcon className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* Edit role modal */}
        {editingRole && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white dark:bg-gray-800 rounded-lg p-4 w-[500px] max-h-[80vh] shadow-xl flex flex-col">
              <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">
                Edit Role: {editingRole.name}
              </h3>
              <input
                type="text"
                placeholder="Description (optional)"
                value={editingRole.description}
                onChange={(e) => setEditingRole({ ...editingRole, description: e.target.value })}
                className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded mb-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
              <textarea
                value={editingRole.prompt}
                onChange={(e) => setEditingRole({ ...editingRole, prompt: e.target.value })}
                className="flex-1 min-h-[300px] px-3 py-2 text-sm font-mono border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 resize-none"
              />
              <div className="flex justify-end gap-2 mt-4">
                <button
                  onClick={() => setEditingRole(null)}
                  className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md"
                >
                  Cancel
                </button>
                <button
                  onClick={handleUpdateRole}
                  className="px-3 py-1.5 text-sm bg-primary-600 text-white rounded-md hover:bg-primary-700"
                >
                  Save
                </button>
              </div>
            </div>
          </div>
        )}

        {allRoles.length === 0 && !creatingRole ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">No roles defined</p>
        ) : (
          <div className="-mx-4">
            {allRoles.map((role) => {
              const isExpanded = expandedRoles.has(role.name)
              const content = roleContents[role.name]

              return (
                <div key={role.name} className="border-b border-gray-200 dark:border-gray-700 last:border-b-0">
                  {/* Sub-accordion header */}
                  <div className="flex items-center group">
                    <button
                      onClick={() => handleToggleRoleExpand(role.name)}
                      className="flex-1 flex items-center gap-2 px-4 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                    >
                      <span className="flex-1 text-sm font-medium text-gray-700 dark:text-gray-300">
                        {role.name}
                      </span>
                      <ChevronDownIcon
                        className={`w-4 h-4 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                      />
                    </button>
                    <div className="flex gap-1 pr-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => handleEditRole(role.name)}
                        className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 rounded"
                        title="Edit role"
                      >
                        <PencilIcon className="w-3 h-3" />
                      </button>
                      <button
                        onClick={() => handleDeleteRole(role.name)}
                        className="p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 rounded"
                        title="Delete role"
                      >
                        <TrashIcon className="w-3 h-3" />
                      </button>
                    </div>
                  </div>

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

      {/* Skills - always show with create/edit/delete */}
      <AccordionSection
        id="skills"
        title="Skills"
        count={allSkills.length}
        icon={<SparklesIcon className="w-4 h-4" />}
        command="/skills"
        action={
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
        }
      >
        {/* Create skill form */}
        {creatingSkill && (
          <div className="mb-3 p-3 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg">
            <input
              type="text"
              placeholder="Skill name"
              value={newSkill.name}
              onChange={(e) => setNewSkill({ ...newSkill, name: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 mb-2"
            />
            <input
              type="text"
              placeholder="Description (for AI drafting or display)"
              value={newSkill.description}
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
                  value={newToolInput}
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
              value={newSkill.body}
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
                    value={editingSkill.name}
                    onChange={(e) => setEditingSkill({ ...editingSkill, name: e.target.value })}
                    className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">Description</label>
                  <input
                    type="text"
                    value={editingSkill.description}
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
                      value={newToolInput}
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
                    value={editingSkill.body}
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
          <p className="text-sm text-gray-500 dark:text-gray-400">No skills defined</p>
        ) : (
          <div className="-mx-4">
            {allSkills.map((skill) => {
              const isExpanded = expandedSkills.has(skill.name)
              const content = skillContents[skill.name]
              const { frontMatter, body } = content ? parseFrontMatter(content) : { frontMatter: null, body: '' }
              const allowedTools = frontMatter?.['allowed-tools'] as string[] | undefined

              return (
                <div key={skill.name} className="border-b border-gray-200 dark:border-gray-700 last:border-b-0">
                  {/* Sub-accordion header */}
                  <div className="flex items-center group">
                    <button
                      onClick={() => handleToggleSkillExpand(skill.name)}
                      className="flex-1 flex items-center gap-2 px-4 py-2 text-left hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                    >
                      <span className="flex-1 text-sm font-medium text-gray-700 dark:text-gray-300">
                        {skill.name}
                      </span>
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
                        onClick={() => handleDeleteSkill(skill.name)}
                        className="p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 rounded"
                        title="Delete skill"
                      >
                        <TrashIcon className="w-3 h-3" />
                      </button>
                    </div>
                  </div>

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

      {/* Learnings - always show (has add action) */}
      <AccordionSection
        id="learnings"
        title="Learnings"
        count={learnings.length + rules.length}
        icon={<AcademicCapIcon className="w-4 h-4" />}
        command="/learnings"
        action={
          <div className="flex items-center gap-1">
            {learnings.length >= 2 && (
              <button
                onClick={async () => {
                  setCompacting(true)
                  try {
                    const result = await sessionsApi.compactLearnings()
                    if (result.status === 'success') {
                      fetchLearnings()
                    }
                  } finally {
                    setCompacting(false)
                  }
                }}
                disabled={compacting}
                className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors disabled:opacity-50"
                title="Compact learnings into rules"
              >
                <ArrowPathIcon className={`w-4 h-4 ${compacting ? 'animate-spin' : ''}`} />
              </button>
            )}
            <button
              onClick={() => openModal('rule')}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Add rule"
            >
              <PlusIcon className="w-4 h-4" />
            </button>
          </div>
        }
      >
        {learnings.length === 0 && rules.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">No learnings yet</p>
        ) : (
          <div className="space-y-3">
            {/* Rules section */}
            {rules.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                  Rules ({rules.length})
                </p>
                {rules.map((rule) => (
                  <div
                    key={rule.id}
                    className="p-2 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg group"
                  >
                    {editingRule?.id === rule.id ? (
                      <div className="space-y-2">
                        <textarea
                          value={editingRule.summary}
                          onChange={(e) => setEditingRule({ ...editingRule, summary: e.target.value })}
                          className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 resize-none"
                          rows={2}
                          autoFocus
                        />
                        <div className="flex gap-1">
                          <button
                            onClick={handleUpdateRule}
                            className="p-1 text-green-600 hover:bg-green-100 dark:hover:bg-green-900/30 rounded"
                            title="Save"
                          >
                            <CheckIcon className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => setEditingRule(null)}
                            className="p-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
                            title="Cancel"
                          >
                            <XMarkIcon className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <div className="flex items-start justify-between gap-2">
                          <p className="text-sm text-gray-700 dark:text-gray-300 flex-1">
                            {rule.summary}
                          </p>
                          <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            <button
                              onClick={() => setEditingRule({ id: rule.id, summary: rule.summary })}
                              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 rounded"
                              title="Edit rule"
                            >
                              <PencilIcon className="w-3 h-3" />
                            </button>
                            <button
                              onClick={() => handleDeleteRule(rule.id)}
                              className="p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 rounded"
                              title="Delete rule"
                            >
                              <TrashIcon className="w-3 h-3" />
                            </button>
                          </div>
                        </div>
                        <div className="mt-1 flex items-center gap-2 text-xs text-gray-400 dark:text-gray-500">
                          <span className="px-1.5 py-0.5 bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200 rounded">
                            {Math.round(rule.confidence * 100)}% confidence
                          </span>
                          <span>{rule.source_count} sources</span>
                          {rule.tags.length > 0 && (
                            <span className="text-gray-300 dark:text-gray-600">
                              {rule.tags.join(', ')}
                            </span>
                          )}
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            )}
            {/* Raw learnings section */}
            {learnings.length > 0 && (
              <div className="space-y-2">
                {rules.length > 0 && (
                  <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                    Pending ({learnings.length})
                  </p>
                )}
                {learnings.map((learning) => (
                  <div
                    key={learning.id}
                    className="p-2 bg-gray-50 dark:bg-gray-800/50 rounded-lg group"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <p className="text-sm text-gray-700 dark:text-gray-300 flex-1">
                        {learning.content}
                      </p>
                      <button
                        onClick={() => handleDeleteLearning(learning.id)}
                        className="p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 rounded opacity-0 group-hover:opacity-100 transition-opacity"
                        title="Delete learning"
                      >
                        <TrashIcon className="w-3 h-3" />
                      </button>
                    </div>
                    <div className="mt-1 flex items-center gap-2 text-xs text-gray-400 dark:text-gray-500">
                      <span className="px-1.5 py-0.5 bg-gray-200 dark:bg-gray-700 rounded">
                        {learning.category}
                      </span>
                      {learning.applied_count > 0 && (
                        <span>Applied {learning.applied_count}x</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </AccordionSection>

      {/* Code - only show when there's code */}
      {stepCodes.length > 0 && (
        <AccordionSection
          id="code"
          title="Code"
          count={stepCodes.length}
          icon={<CodeBracketIcon className="w-4 h-4" />}
          command="/code"
          action={
            <button
              onClick={async () => {
                if (!session) return
                try {
                  const response = await fetch(
                    `/api/sessions/${session.session_id}/download-code`,
                    { credentials: 'include' }
                  )
                  if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}))
                    const message = errorData.detail || 'Failed to download code'
                    alert(message)
                    return
                  }
                  const blob = await response.blob()
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `session_${session.session_id.slice(0, 8)}_code.py`
                  document.body.appendChild(a)
                  a.click()
                  document.body.removeChild(a)
                  URL.revokeObjectURL(url)
                } catch (err) {
                  console.error('Download failed:', err)
                  alert('Failed to download code. Please try again.')
                }
              }}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Download as Python script"
            >
              <ArrowDownTrayIcon className="w-4 h-4" />
            </button>
          }
        >
          <div className="space-y-3">
            {stepCodes.map((step) => (
              <div key={step.step_number}>
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                  Step {step.step_number}: {step.goal}
                </p>
                <CodeViewer
                  code={step.code}
                  language="python"
                />
              </div>
            ))}
          </div>
        </AccordionSection>
      )}

      {/* Facts - always show (has add action) */}
      <AccordionSection
        id="facts"
        title="Facts"
        count={facts.length}
        icon={<LightBulbIcon className="w-4 h-4" />}
        command="/facts"
        action={
          <button
            onClick={() => openModal('fact')}
            className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            title="Add fact"
          >
            <PlusIcon className="w-4 h-4" />
          </button>
        }
      >
        {facts.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">No facts yet</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-2 px-1 font-medium text-gray-600 dark:text-gray-400">
                    Name
                  </th>
                  <th className="text-left py-2 px-1 font-medium text-gray-600 dark:text-gray-400">
                    Value
                  </th>
                  <th className="text-left py-2 px-1 font-medium text-gray-600 dark:text-gray-400">
                    Source
                  </th>
                  <th className="w-8"></th>
                </tr>
              </thead>
              <tbody>
                {facts.map((fact) => (
                  <tr
                    key={fact.name}
                    className={`border-b border-gray-100 dark:border-gray-800 last:border-b-0 group ${
                      fact.is_persisted ? 'bg-amber-50/50 dark:bg-amber-900/10' : ''
                    }`}
                  >
                    <td className="py-2 px-1 font-medium text-gray-700 dark:text-gray-300">
                      <span className="flex items-center gap-1 flex-wrap">
                        {fact.name}
                        {fact.is_persisted && (
                          <span className="px-1 py-0.5 text-[10px] bg-amber-200 dark:bg-amber-800 text-amber-800 dark:text-amber-200 rounded">
                            saved
                          </span>
                        )}
                        {fact.role_id && (
                          <span className="px-1 py-0.5 text-[10px] bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded">
                            {fact.role_id}
                          </span>
                        )}
                      </span>
                    </td>
                    <td className="py-2 px-1 text-gray-600 dark:text-gray-400">
                      {String(fact.value)}
                    </td>
                    <td className="py-2 px-1 text-xs text-gray-400 dark:text-gray-500">
                      {fact.source}
                    </td>
                    <td className="py-2 px-1 flex items-center gap-1">
                      {!fact.is_persisted && (
                        <button
                          onClick={() => handlePersistFact(fact.name)}
                          className="p-1 text-gray-300 dark:text-gray-600 hover:text-amber-500 dark:hover:text-amber-400 opacity-0 group-hover:opacity-100 transition-opacity"
                          title="Save permanently"
                        >
                          <ArrowUpTrayIcon className="w-3 h-3" />
                        </button>
                      )}
                      <button
                        onClick={() => handleForgetFact(fact.name)}
                        className="p-1 text-gray-300 dark:text-gray-600 hover:text-red-500 dark:hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                        title="Forget fact"
                      >
                        <MinusIcon className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </AccordionSection>

      {/* Entities - only show when there are entities */}
      {entities.length > 0 && (
        <AccordionSection
          id="entities"
          title="Entities"
          count={entities.length}
          icon={<TagIcon className="w-4 h-4" />}
          command="/entities"
          action={<div className="w-6 h-6" />}
        >
          <EntityAccordion entities={entities} />
        </AccordionSection>
      )}
      </>
      )}
    </div>
  )
}