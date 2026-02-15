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
    inferenceCodes,
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

  const [expandedDb, setExpandedDb] = useState<string | null>(null)
  const [dbTables, setDbTables] = useState<Record<string, sessionsApi.DatabaseTableInfo[]>>({})
  const [dbTablesLoading, setDbTablesLoading] = useState<string | null>(null)
  const [previewDb, setPreviewDb] = useState<string | null>(null)
  const [previewTable, setPreviewTable] = useState<string | null>(null)
  const [previewData, setPreviewData] = useState<sessionsApi.DatabaseTablePreview | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewPage, setPreviewPage] = useState(1)
  const [expandedApi, setExpandedApi] = useState<string | null>(null)
  const [apiEndpoints, setApiEndpoints] = useState<Record<string, sessionsApi.ApiEndpointInfo[]>>({})
  const [apiEndpointsLoading, setApiEndpointsLoading] = useState<string | null>(null)
  const [expandedEndpoint, setExpandedEndpoint] = useState<string | null>(null)
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
  const [editingRole, setEditingRole] = useState<{ name: string; prompt: string; description: string; skills: string[] } | null>(null)
  const [expandedRoles, setExpandedRoles] = useState<Set<string>>(new Set())
  const [roleContents, setRoleContents] = useState<Record<string, { prompt: string; description: string }>>({})
  const [creatingRole, setCreatingRole] = useState(false)
  const [newRole, setNewRole] = useState({ name: '', prompt: '', description: '', skills: [] as string[] })
  // System prompt editing state (admin only)
  const [editingSystemPrompt, setEditingSystemPrompt] = useState(false)
  const [systemPromptDraft, setSystemPromptDraft] = useState('')
  // Document modal state
  const [docSourceType, setDocSourceType] = useState<'uri' | 'files'>('files')
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

  const handleAddApi = async () => {
    if (!session || !modalInput.name || !modalInput.uri) return
    await sessionsApi.addApi(session.session_id, {
      name: modalInput.name,
      base_url: modalInput.uri,
      type: modalInput.type || 'rest',
    })
    fetchDataSources(session.session_id)
    // Entities refresh via entity_rebuild_complete WS event
    setShowModal(null)
    setModalInput({ name: '', value: '', uri: '', type: '', persist: false })
  }

  const handleDeleteApi = async (apiName: string) => {
    if (!session) return
    if (!confirm(`Remove API "${apiName}" from this session?`)) return

    try {
      await sessionsApi.removeApi(session.session_id, apiName)
      fetchDataSources(session.session_id)
      // Entities refresh via entity_rebuild_complete WS event
    } catch (err) {
      console.error('Failed to remove API:', err)
      alert('Failed to remove API. Please try again.')
    }
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
        // Entities refresh via entity_rebuild_complete WS event
        setShowModal(null)
        setSelectedFiles([])
        setDocSourceType('files')
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
      // Entities refresh via entity_rebuild_complete WS event
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
      // Entities refresh via entity_rebuild_complete WS event
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

  const FILE_DB_TYPES = new Set(['csv', 'json', 'jsonl', 'parquet', 'arrow', 'feather', 'tsv'])

  const toggleDbExpand = async (dbName: string, dbType?: string) => {
    if (expandedDb === dbName) {
      setExpandedDb(null)
      setPreviewDb(null)
      setPreviewTable(null)
      setPreviewData(null)
      return
    }
    setExpandedDb(dbName)
    setPreviewDb(null)
    setPreviewTable(null)
    setPreviewData(null)
    if (!dbTables[dbName] && session) {
      setDbTablesLoading(dbName)
      try {
        const res = await sessionsApi.listDatabaseTables(session.session_id, dbName)
        setDbTables((prev) => ({ ...prev, [dbName]: res.tables }))
        // File-based DBs are single-table — jump straight to preview
        if (dbType && FILE_DB_TYPES.has(dbType) && res.tables.length === 1) {
          openTablePreview(dbName, res.tables[0].name)
        }
      } catch (err) {
        console.error('Failed to list tables:', err)
        setDbTables((prev) => ({ ...prev, [dbName]: [] }))
      } finally {
        setDbTablesLoading(null)
      }
    } else if (dbTables[dbName] && dbType && FILE_DB_TYPES.has(dbType) && dbTables[dbName].length === 1) {
      // Already cached — jump to preview
      openTablePreview(dbName, dbTables[dbName][0].name)
    }
  }

  const openTablePreview = async (dbName: string, tableName: string, page = 1) => {
    if (!session) return
    setPreviewDb(dbName)
    setPreviewTable(tableName)
    setPreviewPage(page)
    setPreviewLoading(true)
    try {
      const data = await sessionsApi.getDatabaseTablePreview(
        session.session_id, dbName, tableName, page
      )
      setPreviewData(data)
    } catch (err) {
      console.error('Failed to preview table:', err)
      setPreviewData(null)
    } finally {
      setPreviewLoading(false)
    }
  }

  const toggleApiExpand = async (apiName: string) => {
    if (expandedApi === apiName) {
      setExpandedApi(null)
      setExpandedEndpoint(null)
      return
    }
    setExpandedApi(apiName)
    setExpandedEndpoint(null)
    if (!apiEndpoints[apiName] && session) {
      setApiEndpointsLoading(apiName)
      try {
        const res = await sessionsApi.getApiSchema(session.session_id, apiName)
        setApiEndpoints((prev) => ({ ...prev, [apiName]: res.endpoints }))
      } catch (err) {
        console.error('Failed to load API schema:', err)
        setApiEndpoints((prev) => ({ ...prev, [apiName]: [] }))
      } finally {
        setApiEndpointsLoading(null)
      }
    }
  }

  const openModal = (type: ModalType) => {
    setModalInput({ name: '', value: '', uri: '', type: '', persist: false })
    setDocSourceType('files')
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
        body: parsed.body || prev.body,
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
      setNewRole(prev => ({ ...prev, prompt: result.prompt || '', description: result.description || prev.description, skills: result.skills || [] }))
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
        setEditingRole({ name: data.name, prompt: data.prompt || '', description: data.description || '', skills: data.skills || [] })
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
        setNewRole({ name: '', prompt: '', description: '', skills: [] })
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
          body: JSON.stringify({ prompt: editingRole.prompt, description: editingRole.description, skills: editingRole.skills }),
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
                  value={modalInput.value || ''}
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
                        onChange={() => setDocSourceType('files')}
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
                        value={modalInput.name || ''}
                        onChange={(e) => setModalInput({ ...modalInput, name: e.target.value })}
                        className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      />
                      <input
                        type="text"
                        placeholder="URI (file:// or http://)"
                        value={modalInput.uri || ''}
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
                    value={modalInput.name || ''}
                    onChange={(e) => setModalInput({ ...modalInput, name: e.target.value })}
                    className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  />
                  {showModal === 'fact' ? (
                    <>
                      <input
                        type="text"
                        placeholder="Value"
                        value={modalInput.value || ''}
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
                      value={modalInput.uri || ''}
                      onChange={(e) => setModalInput({ ...modalInput, uri: e.target.value })}
                      className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />
                  )}
                </>
              )}
              {showModal === 'database' && (
                <select
                  value={modalInput.type || ''}
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
              {showModal === 'api' && (
                <select
                  value={modalInput.type || ''}
                  onChange={(e) => setModalInput({ ...modalInput, type: e.target.value })}
                  className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                >
                  <option value="">Type (optional)</option>
                  <option value="rest">REST</option>
                  <option value="graphql">GraphQL</option>
                  <option value="openapi">OpenAPI</option>
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
                  else if (showModal === 'api') handleAddApi()
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
                    <button
                      onClick={() => toggleDbExpand(db.name, db.type)}
                      className="flex items-center gap-1 text-sm font-medium text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 hover:underline"
                    >
                      {expandedDb === db.name ? (
                        <ChevronDownIcon className="w-3 h-3 flex-shrink-0" />
                      ) : (
                        <ChevronRightIcon className="w-3 h-3 flex-shrink-0" />
                      )}
                      {db.name}
                    </button>
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
                    {db.is_dynamic && (
                      <button
                        onClick={async () => {
                          if (!session) return
                          if (!confirm(`Remove database "${db.name}" from this session?`)) return
                          try {
                            await sessionsApi.removeDatabase(session.session_id, db.name)
                            await fetchDataSources(session.session_id)
                            // Entities refresh via entity_rebuild_complete WS event
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
                {expandedDb !== db.name && db.table_count !== undefined && db.table_count > 0 && (
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {db.table_count} tables
                  </p>
                )}
                {db.description && (
                  <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                    {db.description}
                  </p>
                )}

                {/* Expanded: table list */}
                {expandedDb === db.name && (
                  <div className="mt-2 ml-4 space-y-1">
                    {dbTablesLoading === db.name ? (
                      <div className="flex items-center gap-2 py-1">
                        <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-primary-500" />
                        <span className="text-xs text-gray-500">Loading tables...</span>
                      </div>
                    ) : (dbTables[db.name] || []).length === 0 ? (
                      <p className="text-xs text-gray-500">No tables found</p>
                    ) : (
                      (dbTables[db.name] || []).map((t) => (
                        <div key={t.name}>
                          <button
                            onClick={() => openTablePreview(db.name, t.name)}
                            className={`w-full text-left text-xs px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors ${
                              previewDb === db.name && previewTable === t.name
                                ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                                : 'text-gray-600 dark:text-gray-400'
                            }`}
                          >
                            <span className="font-medium">{t.name}</span>
                            {t.row_count != null && (
                              <span className="ml-1 text-gray-400">({t.row_count.toLocaleString()} rows)</span>
                            )}
                          </button>

                          {/* Inline preview */}
                          {previewDb === db.name && previewTable === t.name && (
                            <div className="mt-1 mb-2 border border-gray-200 dark:border-gray-700 rounded overflow-hidden">
                              {previewLoading ? (
                                <div className="flex items-center justify-center py-4">
                                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-500" />
                                </div>
                              ) : previewData ? (
                                <div>
                                  <div className="overflow-auto max-h-[200px]">
                                    <table className="min-w-full text-xs">
                                      <thead className="sticky top-0 bg-gray-50 dark:bg-gray-800">
                                        <tr className="border-b border-gray-200 dark:border-gray-700">
                                          {previewData.columns.map((col) => (
                                            <th key={col} className="px-2 py-1 text-left font-medium text-gray-500 dark:text-gray-400 whitespace-nowrap">
                                              {col}
                                            </th>
                                          ))}
                                        </tr>
                                      </thead>
                                      <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                                        {previewData.data.map((row, i) => (
                                          <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                                            {previewData.columns.map((col) => (
                                              <td key={col} className="px-2 py-1 text-gray-700 dark:text-gray-300 whitespace-nowrap">
                                                {row[col] != null && typeof row[col] === 'object'
                                                  ? JSON.stringify(row[col])
                                                  : String(row[col] ?? '')}
                                              </td>
                                            ))}
                                          </tr>
                                        ))}
                                      </tbody>
                                    </table>
                                  </div>
                                  {(previewData.has_more || previewPage > 1) && (
                                    <div className="flex items-center justify-between text-xs px-2 py-1 bg-gray-50 dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
                                      <span className="text-gray-500">
                                        {previewData.total_rows.toLocaleString()} rows
                                      </span>
                                      <div className="flex gap-2">
                                        <button
                                          onClick={() => openTablePreview(db.name, t.name, previewPage - 1)}
                                          disabled={previewPage === 1}
                                          className="text-primary-600 dark:text-primary-400 disabled:opacity-50"
                                        >
                                          Prev
                                        </button>
                                        <button
                                          onClick={() => openTablePreview(db.name, t.name, previewPage + 1)}
                                          disabled={!previewData.has_more}
                                          className="text-primary-600 dark:text-primary-400 disabled:opacity-50"
                                        >
                                          Next
                                        </button>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              ) : (
                                <p className="text-xs text-gray-500 p-2">Failed to load preview</p>
                              )}
                            </div>
                          )}
                        </div>
                      ))
                    )}
                  </div>
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
                className="p-2 bg-gray-50 dark:bg-gray-800/50 rounded-md group"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => toggleApiExpand(api.name)}
                      className="flex items-center gap-1 text-sm font-medium text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 hover:underline"
                    >
                      {expandedApi === api.name ? (
                        <ChevronDownIcon className="w-3 h-3 flex-shrink-0" />
                      ) : (
                        <ChevronRightIcon className="w-3 h-3 flex-shrink-0" />
                      )}
                      {api.name}
                    </button>
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
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-xs px-1.5 py-0.5 rounded ${
                        api.connected
                          ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                          : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                      }`}
                    >
                      {api.connected ? 'Available' : 'Pending'}
                    </span>
                    {api.source === 'session' && (
                      <button
                        onClick={() => handleDeleteApi(api.name)}
                        className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 transition-all"
                        title="Remove API"
                      >
                        <TrashIcon className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>
                {expandedApi !== api.name && api.type && (
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    {api.type}
                  </p>
                )}
                {api.description && (
                  <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                    {api.description}
                  </p>
                )}

                {/* Expanded: endpoint/query list */}
                {expandedApi === api.name && (
                  <div className="mt-2 ml-4 space-y-1">
                    {apiEndpointsLoading === api.name ? (
                      <div className="flex items-center gap-2 py-1">
                        <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-primary-500" />
                        <span className="text-xs text-gray-500">Loading schema...</span>
                      </div>
                    ) : (apiEndpoints[api.name] || []).length === 0 ? (
                      <p className="text-xs text-gray-500">No endpoints found</p>
                    ) : (() => {
                      const allEps = apiEndpoints[api.name] || []

                      const renderEndpoint = (ep: sessionsApi.ApiEndpointInfo) => (
                        <div key={ep.name}>
                          <button
                            onClick={() => setExpandedEndpoint(expandedEndpoint === ep.name ? null : ep.name)}
                            className={`w-full text-left text-xs px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors ${
                              expandedEndpoint === ep.name
                                ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                                : 'text-gray-600 dark:text-gray-400'
                            }`}
                          >
                            <div className="flex items-center gap-1.5">
                              {expandedEndpoint === ep.name ? (
                                <ChevronDownIcon className="w-2.5 h-2.5 flex-shrink-0" />
                              ) : (
                                <ChevronRightIcon className="w-2.5 h-2.5 flex-shrink-0" />
                              )}
                              <span className="font-medium">{ep.name}</span>
                              {ep.return_type && (
                                <span className="font-mono text-[10px] text-purple-600 dark:text-purple-400">{ep.return_type}</span>
                              )}
                            </div>
                            {ep.description && (
                              <p className="text-[10px] text-gray-400 mt-0.5 ml-4">{ep.description}</p>
                            )}
                          </button>

                          {expandedEndpoint === ep.name && ep.fields.length > 0 && (
                            <div className="ml-6 mt-1 mb-2 border-l-2 border-gray-200 dark:border-gray-700 pl-2 space-y-0.5">
                              <p className="text-[10px] font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Fields</p>
                              {ep.fields.map((f) => (
                                <div key={f.name} className="text-xs flex items-baseline gap-1.5">
                                  <span className="font-medium text-gray-700 dark:text-gray-300">{f.name}</span>
                                  <span className="font-mono text-[10px] text-purple-600 dark:text-purple-400">{f.type}</span>
                                  {f.is_required && (
                                    <span className="text-[9px] text-red-500">required</span>
                                  )}
                                  {f.description && (
                                    <span className="text-gray-400 truncate">{f.description}</span>
                                  )}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )

                      const renderSection = (label: string, items: sessionsApi.ApiEndpointInfo[]) => (
                        items.length > 0 ? (
                          <div key={label}>
                            <p className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
                              {label} <span className="font-normal">({items.length})</span>
                            </p>
                            <div className="space-y-0.5">
                              {items.map(renderEndpoint)}
                            </div>
                          </div>
                        ) : null
                      )

                      // GraphQL: group by operation type + types
                      const gqlKinds: Record<string, string> = {
                        graphql_query: 'Queries',
                        graphql_mutation: 'Mutations',
                        graphql_subscription: 'Subscriptions',
                        graphql_type: 'Types',
                      }
                      const gqlGroups = Object.entries(gqlKinds)
                        .map(([kind, label]) => ({ label, items: allEps.filter((ep) => ep.kind === kind) }))
                        .filter((g) => g.items.length > 0)

                      // REST: operations grouped by HTTP method, then schema types
                      const restOps = allEps.filter((ep) => ep.kind === 'rest' || (!ep.kind?.startsWith('graphql_') && !ep.kind?.includes('/') && ep.http_method))
                      const restTypes = allEps.filter((ep) => ep.kind === 'rest/schema')
                      const restOther = allEps.filter((ep) => !ep.kind?.startsWith('graphql_') && ep.kind !== 'rest' && ep.kind !== 'rest/schema' && !ep.http_method)
                      const methodOrder = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE']
                      const restMethods = [...new Set(restOps.map((ep) => ep.http_method || 'OTHER'))]
                        .sort((a, b) => (methodOrder.indexOf(a) === -1 ? 99 : methodOrder.indexOf(a)) - (methodOrder.indexOf(b) === -1 ? 99 : methodOrder.indexOf(b)))
                      const restGroups = [
                        ...restMethods.map((method) => ({ label: method, items: restOps.filter((ep) => (ep.http_method || 'OTHER') === method) })),
                        ...(restTypes.length > 0 ? [{ label: 'Types', items: restTypes }] : []),
                        ...(restOther.length > 0 ? [{ label: 'Other', items: restOther }] : []),
                      ].filter((g) => g.items.length > 0)

                      return (
                        <div className="space-y-2">
                          {gqlGroups.map((g) => renderSection(g.label, g.items))}
                          {restGroups.map((g) => renderSection(g.label, g.items))}
                        </div>
                      )
                    })()}
                  </div>
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
                value={systemPromptDraft || ''}
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
              value={newRole.name || ''}
              onChange={(e) => setNewRole({ ...newRole, name: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded mb-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
            <input
              type="text"
              placeholder="Description (for AI drafting or display)"
              value={newRole.description || ''}
              onChange={(e) => setNewRole({ ...newRole, description: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded mb-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
            <textarea
              placeholder="Role prompt (persona definition)..."
              value={newRole.prompt || ''}
              onChange={(e) => setNewRole({ ...newRole, prompt: e.target.value })}
              className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded mb-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 min-h-[100px] resize-none"
            />
            {allSkills.length > 0 && (
              <div className="mb-2">
                <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Skills</label>
                <div className="flex flex-wrap gap-1">
                  {allSkills.map((skill) => (
                    <button
                      key={skill.name}
                      onClick={() => {
                        const has = newRole.skills.includes(skill.name)
                        setNewRole({ ...newRole, skills: has ? newRole.skills.filter(s => s !== skill.name) : [...newRole.skills, skill.name] })
                      }}
                      className={`px-2 py-0.5 text-xs rounded-full border ${
                        newRole.skills.includes(skill.name)
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
                  setNewRole({ name: '', prompt: '', description: '', skills: [] })
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
                value={editingRole.description || ''}
                onChange={(e) => setEditingRole({ ...editingRole, description: e.target.value })}
                className="w-full px-2 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded mb-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              />
              <textarea
                value={editingRole.prompt || ''}
                onChange={(e) => setEditingRole({ ...editingRole, prompt: e.target.value })}
                className="flex-1 min-h-[300px] px-3 py-2 text-sm font-mono border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 resize-none"
              />
              {allSkills.length > 0 && (
                <div className="mt-2">
                  <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Skills</label>
                  <div className="flex flex-wrap gap-1">
                    {allSkills.map((skill) => (
                      <button
                        key={skill.name}
                        onClick={() => {
                          const has = editingRole.skills.includes(skill.name)
                          setEditingRole({ ...editingRole, skills: has ? editingRole.skills.filter(s => s !== skill.name) : [...editingRole.skills, skill.name] })
                        }}
                        className={`px-2 py-0.5 text-xs rounded-full border ${
                          editingRole.skills.includes(skill.name)
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
          <p className="text-sm text-gray-500 dark:text-gray-400">No skills defined</p>
        ) : (
          <div className="-mx-4">
            {allSkills.map((skill) => {
              const isExpanded = expandedSkills.has(skill.name)
              const content = skillContents[skill.name]
              const { frontMatter, body } = content ? parseFrontMatter(content) : { frontMatter: null, body: '' }
              const rawTools = frontMatter?.['allowed-tools']
              const allowedTools = Array.isArray(rawTools) ? rawTools as string[] : typeof rawTools === 'string' ? rawTools.split(',').map(s => s.trim()).filter(Boolean) : undefined

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
                        onClick={async () => {
                          try {
                            const headers: Record<string, string> = {}
                            const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
                            if (!isAuthDisabled) {
                              const token = await useAuthStore.getState().getToken()
                              if (token) {
                                headers['Authorization'] = `Bearer ${token}`
                              }
                            }
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
                          value={editingRule.summary || ''}
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
                              onClick={() => setEditingRule({ id: rule.id, summary: rule.summary || '' })}
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
          title="Exploratory Code"
          count={stepCodes.length}
          icon={<CodeBracketIcon className="w-4 h-4" />}
          command="/code"
          action={
            <button
              onClick={async () => {
                if (!session) return
                try {
                  // Build headers with auth token
                  const headers: Record<string, string> = {}
                  const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
                  if (!isAuthDisabled) {
                    const token = await useAuthStore.getState().getToken()
                    if (token) {
                      headers['Authorization'] = `Bearer ${token}`
                    }
                  }

                  const response = await fetch(
                    `/api/sessions/${session.session_id}/download-code`,
                    { headers, credentials: 'include' }
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

      {/* Inference Code - auditable mode (separate from step code) */}
      {inferenceCodes.length > 0 && (
        <AccordionSection
          id="inference-code"
          title="Inference Code"
          count={inferenceCodes.length}
          icon={<CodeBracketIcon className="w-4 h-4" />}
          action={
            <button
              onClick={async () => {
                if (!session) return
                try {
                  const headers: Record<string, string> = {}
                  const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
                  if (!isAuthDisabled) {
                    const token = await useAuthStore.getState().getToken()
                    if (token) {
                      headers['Authorization'] = `Bearer ${token}`
                    }
                  }
                  const response = await fetch(
                    `/api/sessions/${session.session_id}/download-inference-code`,
                    { headers, credentials: 'include' }
                  )
                  if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}))
                    alert(errorData.detail || 'Failed to download inference code')
                    return
                  }
                  const blob = await response.blob()
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `session_${session.session_id.slice(0, 8)}_inference.py`
                  document.body.appendChild(a)
                  a.click()
                  document.body.removeChild(a)
                  URL.revokeObjectURL(url)
                } catch (err) {
                  console.error('Download failed:', err)
                  alert('Failed to download inference code. Please try again.')
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
            {inferenceCodes.map((inf) => (
              <div key={inf.inference_id}>
                <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                  {inf.inference_id}: {inf.name} = {inf.operation}
                </p>
                <CodeViewer
                  code={inf.code}
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
          <div className="flex items-center gap-1">
            {facts.length > 0 && (
              <button
                onClick={() => {
                  // Generate CSV content with name, value columns
                  const csvRows = [
                    ['name', 'value'],
                    ...facts.map(f => [
                      // Escape quotes and wrap in quotes if contains comma
                      `"${String(f.name).replace(/"/g, '""')}"`,
                      `"${String(f.value).replace(/"/g, '""')}"`
                    ])
                  ]
                  const csvContent = csvRows.map(row => row.join(',')).join('\n')
                  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
                  const url = URL.createObjectURL(blob)
                  const link = document.createElement('a')
                  link.href = url
                  link.download = 'facts.csv'
                  link.click()
                  URL.revokeObjectURL(url)
                }}
                className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                title="Download facts as CSV"
              >
                <ArrowDownTrayIcon className="w-4 h-4" />
              </button>
            )}
            <button
              onClick={() => openModal('fact')}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Add fact"
            >
              <PlusIcon className="w-4 h-4" />
            </button>
          </div>
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
                      fact.source === 'config'
                        ? 'bg-purple-50/50 dark:bg-purple-900/10'
                        : fact.is_persisted
                        ? 'bg-amber-50/50 dark:bg-amber-900/10'
                        : ''
                    }`}
                  >
                    <td className="py-2 px-1 font-medium text-gray-700 dark:text-gray-300">
                      <span className="flex items-center gap-1 flex-wrap">
                        {fact.name}
                        {fact.source === 'config' && (
                          <span className="px-1 py-0.5 text-[10px] bg-purple-200 dark:bg-purple-800 text-purple-800 dark:text-purple-200 rounded">
                            core
                          </span>
                        )}
                        {fact.is_persisted && fact.source !== 'config' && (
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
                      {fact.source === 'config' ? 'core' : fact.source}
                    </td>
                    <td className="py-2 px-1 flex items-center gap-1">
                      {/* Don't show persist/forget buttons for config facts */}
                      {fact.source !== 'config' && (
                        <>
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
                        </>
                      )}
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