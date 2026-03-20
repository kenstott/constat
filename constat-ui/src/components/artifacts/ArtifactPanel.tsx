// Artifact Panel container

import React, { useEffect, useState, useCallback } from 'react'
import { DomainBadge } from '../common/DomainBadge'
import { ScopeBadge } from '../common/ScopeBadge'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import {
  BeakerIcon,
  CodeBracketIcon,
  LightBulbIcon,
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
  ArrowsRightLeftIcon,
  PencilIcon,
  TrashIcon,
  CheckIcon,
  XMarkIcon,
  UserCircleIcon,
  SparklesIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  CpuChipIcon,
  QuestionMarkCircleIcon,
} from '@heroicons/react/24/outline'
import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import { useAuthStore, isAuthDisabled } from '@/store/authStore'
import { useUIStore } from '@/store/uiStore'
import { useGlossaryStore } from '@/store/glossaryStore'
import { AccordionSection } from './ArtifactAccordion'
import { TableAccordion } from './TableAccordion'
import type { FineTuneJob, FineTuneProvider } from '@/types/api'
import { ArtifactItemAccordion } from './ArtifactItemAccordion'
import { CodeViewer } from './CodeViewer'
import GlossaryPanel from './GlossaryPanel'
import RegressionPanel from './RegressionPanel'
import * as sessionsApi from '@/api/sessions'
import * as agentsApi from '@/api/agents'

type ModalType = 'database' | 'api' | 'document' | 'fact' | 'rule' | null

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

function formatMs(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  const seconds = ms / 1000
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const minutes = Math.floor(seconds / 60)
  const remainSec = Math.round(seconds % 60)
  return `${minutes}m ${remainSec}s`
}

/** Collapsible step code list for the Exploratory Code section */
function StepCodeList({ stepCodes, supersededStepNumbers, tables, artifacts }: {
  stepCodes: Array<{ step_number: number; goal: string; code: string; prompt?: string; model?: string }>
  supersededStepNumbers: Set<number>
  tables: Array<{ name: string; step_number: number }>
  artifacts: Array<{ id: number; name: string; step_number: number; artifact_type: string }>
}) {
  const [collapsedSteps, setCollapsedSteps] = useState<Set<number>>(() => new Set(stepCodes.map((s) => s.step_number)))
  const [showPrompt, setShowPrompt] = useState<Set<number>>(new Set())

  // Build step → outputs lookup
  const stepOutputs = new Map<number, Array<{ type: 'table' | 'artifact'; name: string; id: string }>>()
  const excludedTypes = new Set(['code', 'error', 'output', 'table'])
  for (const t of tables) {
    if (!stepOutputs.has(t.step_number)) stepOutputs.set(t.step_number, [])
    stepOutputs.get(t.step_number)!.push({ type: 'table', name: t.name, id: `table-${t.name}` })
  }
  for (const a of artifacts) {
    if (excludedTypes.has(a.artifact_type)) continue
    if (!stepOutputs.has(a.step_number)) stepOutputs.set(a.step_number, [])
    stepOutputs.get(a.step_number)!.push({ type: 'artifact', name: a.name, id: `artifact-${a.id}` })
  }

  // Collapse newly arriving step codes by default
  useEffect(() => {
    setCollapsedSteps((prev) => {
      const next = new Set(prev)
      for (const s of stepCodes) {
        next.add(s.step_number)
      }
      return next.size === prev.size ? prev : next
    })
  }, [stepCodes])

  const toggleStep = useCallback((stepNumber: number) => {
    setCollapsedSteps((prev) => {
      const next = new Set(prev)
      if (next.has(stepNumber)) {
        next.delete(stepNumber)
      } else {
        next.add(stepNumber)
      }
      return next
    })
  }, [])

  const togglePrompt = useCallback((stepNumber: number) => {
    setShowPrompt((prev) => {
      const next = new Set(prev)
      if (next.has(stepNumber)) {
        next.delete(stepNumber)
      } else {
        next.add(stepNumber)
      }
      return next
    })
  }, [])

  return (
    <div className="space-y-1">
      {stepCodes.map((step) => {
        const isCollapsed = collapsedSteps.has(step.step_number)
        return (
          <div key={step.step_number} className={supersededStepNumbers.has(step.step_number) ? 'opacity-40' : ''}>
            <div className="flex items-center gap-1">
              <button
                onClick={() => toggleStep(step.step_number)}
                className="flex items-center gap-1 flex-1 text-left text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 py-1 transition-colors"
              >
                {isCollapsed ? (
                  <ChevronRightIcon className="w-3 h-3 flex-shrink-0" />
                ) : (
                  <ChevronDownIcon className="w-3 h-3 flex-shrink-0" />
                )}
                <span>Step {step.step_number}</span>
                {step.model && (
                  <span className="text-[10px] text-gray-400 dark:text-gray-500 font-mono ml-1">{step.model}</span>
                )}
              </button>
              {step.prompt && (
                <button
                  onClick={(e) => { e.stopPropagation(); togglePrompt(step.step_number) }}
                  className="ml-auto p-0.5 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                  title="View generation prompt"
                >
                  <QuestionMarkCircleIcon className={`w-4 h-4 ${showPrompt.has(step.step_number) ? 'text-primary-500' : 'text-gray-400 hover:text-primary-500'}`} />
                </button>
              )}
            </div>
            {!isCollapsed && (
              <>
                {showPrompt.has(step.step_number) && step.prompt && (
                  <pre className="text-[10px] leading-tight text-gray-400 bg-gray-50 dark:bg-gray-800/50 p-2 rounded overflow-auto max-h-60 mb-1 whitespace-pre-wrap">
                    {step.prompt}
                  </pre>
                )}
                <CodeViewer
                  code={step.code}
                  language="python"
                />
                {stepOutputs.has(step.step_number) && (
                  <div className="flex flex-wrap gap-1 mt-1 px-1">
                    {stepOutputs.get(step.step_number)!.map((output) => (
                      <button
                        key={output.id}
                        onClick={() => {
                          const el = document.getElementById(`section-results`)
                          el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                          // After scroll, highlight the specific item
                          requestAnimationFrame(() => {
                            requestAnimationFrame(() => {
                              const item = document.getElementById(output.id)
                              if (item) {
                                item.scrollIntoView({ behavior: 'smooth', block: 'center' })
                                item.classList.add('ring-2', 'ring-primary-400')
                                setTimeout(() => item.classList.remove('ring-2', 'ring-primary-400'), 2000)
                              }
                            })
                          })
                        }}
                        className="inline-flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded bg-primary-50 text-primary-600 hover:bg-primary-100 dark:bg-primary-900/20 dark:text-primary-400 dark:hover:bg-primary-900/40 transition-colors"
                        title={`Jump to ${output.name}`}
                      >
                        {output.type === 'table' ? <CircleStackIcon className="w-3 h-3" /> : <StarIcon className="w-3 h-3" />}
                        {output.name}
                      </button>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        )
      })}
    </div>
  )
}

/** Collapsible scratchpad list for the Scratchpad section */
function ScratchpadList({ entries, supersededStepNumbers, tables }: {
  entries: Array<{ step_number: number; goal: string; narrative: string; tables_created: string[]; code: string; user_query: string; objective_index: number | null }>
  supersededStepNumbers: Set<number>
  tables: Array<{ name: string }>
}) {
  const [collapsedSteps, setCollapsedSteps] = useState<Set<number>>(() => new Set(entries.map((e) => e.step_number)))

  // Collapse newly arriving entries by default
  useEffect(() => {
    setCollapsedSteps((prev) => {
      const next = new Set(prev)
      for (const e of entries) {
        next.add(e.step_number)
      }
      return next.size === prev.size ? prev : next
    })
  }, [entries])

  const toggleStep = useCallback((stepNumber: number) => {
    setCollapsedSteps((prev) => {
      const next = new Set(prev)
      if (next.has(stepNumber)) {
        next.delete(stepNumber)
      } else {
        next.add(stepNumber)
      }
      return next
    })
  }, [])

  // Build set of existing table names for linking
  const tableNames = new Set(tables.map((t) => t.name))

  return (
    <div className="space-y-1">
      {entries.map((entry) => {
        const isCollapsed = collapsedSteps.has(entry.step_number)
        return (
          <div key={entry.step_number} className={supersededStepNumbers.has(entry.step_number) ? 'opacity-40' : ''}>
            <button
              onClick={() => toggleStep(entry.step_number)}
              className="flex items-center gap-1 w-full text-left text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 py-1 transition-colors"
            >
              {isCollapsed ? (
                <ChevronRightIcon className="w-3 h-3 flex-shrink-0" />
              ) : (
                <ChevronDownIcon className="w-3 h-3 flex-shrink-0" />
              )}
              <span className="font-medium">Step {entry.step_number}:</span>
              <span>{entry.goal}</span>
            </button>
            {!isCollapsed && (
              <div className="pl-5 pb-2 space-y-1.5">
                <p className="text-xs text-gray-600 dark:text-gray-300 whitespace-pre-wrap leading-relaxed">
                  {entry.narrative}
                </p>
                {entry.tables_created.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {entry.tables_created.map((tbl) => (
                      <button
                        key={tbl}
                        onClick={() => {
                          if (!tableNames.has(tbl)) return
                          const el = document.getElementById('section-results')
                          el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                          requestAnimationFrame(() => {
                            requestAnimationFrame(() => {
                              const item = document.getElementById(`table-${tbl}`)
                              if (item) {
                                item.scrollIntoView({ behavior: 'smooth', block: 'center' })
                                item.classList.add('ring-2', 'ring-primary-400')
                                setTimeout(() => item.classList.remove('ring-2', 'ring-primary-400'), 2000)
                              }
                            })
                          })
                        }}
                        className={`inline-flex items-center gap-0.5 text-[10px] px-1.5 py-0.5 rounded transition-colors ${
                          tableNames.has(tbl)
                            ? 'bg-primary-50 text-primary-600 hover:bg-primary-100 dark:bg-primary-900/20 dark:text-primary-400 dark:hover:bg-primary-900/40 cursor-pointer'
                            : 'bg-gray-100 text-gray-500 dark:bg-gray-800 dark:text-gray-500 cursor-default'
                        }`}
                        title={tableNames.has(tbl) ? `Jump to ${tbl}` : tbl}
                      >
                        <CircleStackIcon className="w-3 h-3" />
                        {tbl}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function ArtifactPanel() {
  const { session, messages } = useSessionStore()
  const { expandedArtifactSections, expandArtifactSection, pendingDeepLink, consumeDeepLink } = useUIStore()
  const {
    artifacts,
    tables,
    facts,
    learnings,
    rules,
    databases,
    apis,
    documents,
    stepCodes,
    inferenceCodes,
    scratchpadEntries,
    sessionDDL,
    supersededStepNumbers,
    promptContext,
    taskRouting,
    allSkills,
    allAgents,
    fetchArtifacts,
    fetchTables,
    fetchFacts,
    fetchEntities,
    fetchLearnings,
    fetchDataSources,
    fetchPromptContext,
    fetchTaskRouting,
    fetchAllSkills,
    fetchAllAgents,
    fetchScratchpad,
    fetchDDL,
    createSkill,
    updateSkill,
    deleteSkill,
    draftSkill,
    updateSystemPrompt,
  } = useArtifactStore()
  const { totalDefined, totalSelfDescribing } = useGlossaryStore()
  const authPermissions = useAuthStore(s => s.permissions)
  const canSeeSection = (key: string) => isAuthDisabled || (authPermissions?.visibility?.[key] ?? false)
  const canWrite = (key: string) => isAuthDisabled || (authPermissions?.writes?.[key] ?? false)
  // Sources group visible if any child section visible
  const sourcesVisible = canSeeSection('databases') || canSeeSection('apis') || canSeeSection('documents') || canSeeSection('facts')
  // Reasoning group visible if any child section visible
  const hasRouting = taskRouting && Object.keys(taskRouting).length > 0
  const reasoningVisible = canSeeSection('system_prompt') || hasRouting || canSeeSection('agents') || canSeeSection('skills') || canSeeSection('learnings') || canSeeSection('code') || canSeeSection('inference_code')
  // Reasoning sub-group visibility
  const configVisible = canSeeSection('system_prompt') || hasRouting || canSeeSection('agents') || canSeeSection('skills')
  const improvementVisible = (canSeeSection('learnings') && (learnings.length > 0 || rules.length > 0)) || false
  const codeLogVisible = canSeeSection('code') || canSeeSection('inference_code') || scratchpadEntries.length > 0

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
  const [learningsTab, setLearningsTab] = useState<'rules' | 'pending' | 'export' | 'fine-tune'>('rules')
  const [exportFormat, setExportFormat] = useState<'messages' | 'alpaca' | 'sharegpt'>('messages')
  const [exportInclude, setExportInclude] = useState<Set<string>>(new Set(['corrections', 'rules']))
  const [exportMinConfidence, setExportMinConfidence] = useState(0.6)
  const [exporting, setExporting] = useState(false)

  // Fine-tune state
  const [ftJobs, setFtJobs] = useState<FineTuneJob[]>([])
  const [ftProviders, setFtProviders] = useState<FineTuneProvider[]>([])
  const [ftShowForm, setFtShowForm] = useState(false)
  const [ftName, setFtName] = useState('')
  const [ftProvider, setFtProvider] = useState('')
  const [ftBaseModel, setFtBaseModel] = useState('')
  const [ftTaskTypes, setFtTaskTypes] = useState<Set<string>>(new Set(['sql_generation']))
  const [ftDomain, setFtDomain] = useState('')
  const [ftInclude, setFtInclude] = useState<Set<string>>(new Set(['corrections', 'rules']))
  const [ftMinConf, setFtMinConf] = useState(0.6)
  const [ftSubmitting, setFtSubmitting] = useState(false)
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
  // Agent editing state
  const [draftingAgent, setDraftingAgent] = useState(false)
  const [editingAgent, setEditingAgent] = useState<{ name: string; prompt: string; description: string; skills: string[] } | null>(null)
  const [expandedAgents, setExpandedAgents] = useState<Set<string>>(new Set())
  const [agentContents, setAgentContents] = useState<Record<string, { prompt: string; description: string; skills: string[] }>>({})
  const [creatingAgent, setCreatingAgent] = useState(false)
  const [newAgent, setNewAgent] = useState({ name: '', prompt: '', description: '', skills: [] as string[] })
  // System prompt editing state (admin only)
  const [editingSystemPrompt, setEditingSystemPrompt] = useState(false)
  const [systemPromptDraft, setSystemPromptDraft] = useState('')
  // Document modal state
  const [docSourceType, setDocSourceType] = useState<'uri' | 'files'>('files')
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [docFollowLinks, setDocFollowLinks] = useState(false)
  const [docMaxDepth, setDocMaxDepth] = useState(2)
  const [docMaxDocuments, setDocMaxDocuments] = useState(20)
  const [docSameDomainOnly, setDocSameDomainOnly] = useState(true)
  const [docContentType, setDocContentType] = useState('auto')
  const [docShowAdvanced, setDocShowAdvanced] = useState(false)
  // Document viewer state
  const [viewingDocument, setViewingDocument] = useState<{ name: string; content: string; format?: string; url?: string } | null>(null)
  const [iframeBlocked, setIframeBlocked] = useState(false)
  const [loadingDocument, setLoadingDocument] = useState(false)
  // Results filter - persisted in localStorage
  const [showInferencePrompt, setShowInferencePrompt] = useState<Set<string>>(new Set())
  const [collapsedInferences, setCollapsedInferences] = useState<Set<string>>(new Set<string>())
  const { collapsedResultSteps, toggleResultStep, resultsShowPublishedOnly: showPublishedOnly, setResultsShowPublishedOnly: setShowPublishedOnly } = useUIStore()
  // Collapsible section states - persisted in localStorage
  const [resultsCollapsed, setResultsCollapsed] = useState(() => {
    return localStorage.getItem('constat-results-collapsed') === 'true'
  })
  const [sourcesCollapsed, setSourcesCollapsed] = useState(() => {
    return localStorage.getItem('constat-sources-collapsed') === 'true'
  })
  const [reasoningCollapsed, setReasoningCollapsed] = useState(() => {
    return localStorage.getItem('constat-reasoning-collapsed') === 'true'
  })
  const [glossaryCollapsed, setGlossaryCollapsed] = useState(() => localStorage.getItem('constat-glossary-collapsed') === 'true')
  const [configCollapsed, setConfigCollapsed] = useState(() => localStorage.getItem('constat-config-collapsed') === 'true')
  const [improvementCollapsed, setImprovementCollapsed] = useState(() => localStorage.getItem('constat-improvement-collapsed') === 'true')
  const [codeLogCollapsed, setCodeLogCollapsed] = useState(() => localStorage.getItem('constat-codelog-collapsed') === 'true')
  useEffect(() => {
    if (expandedArtifactSections.includes('results') && resultsCollapsed) {
      setResultsCollapsed(false)
      localStorage.setItem('constat-results-collapsed', 'false')
      // Consume the signal so collapsing works again
      useUIStore.getState().toggleArtifactSection('results')
    }
  }, [expandedArtifactSections, resultsCollapsed])
  // Move-to-domain state
  const [domainList, setDomainList] = useState<{ filename: string; name: string }[]>([])
  const [movingSkill, setMovingSkill] = useState<string | null>(null)
  const [movingAgent, setMovingAgent] = useState<string | null>(null)
  const [movingFact, setMovingFact] = useState<string | null>(null)
  const [movingRule, setMovingRule] = useState<string | null>(null)

  const toggleResultsFilter = () => {
    setShowPublishedOnly(!showPublishedOnly)
  }

  // Default inference codes to collapsed when they load
  useEffect(() => {
    if (inferenceCodes.length > 0) {
      setCollapsedInferences(new Set(inferenceCodes.map((inf) => inf.inference_id)))
    }
  }, [inferenceCodes])

  // Deep link: ref stores the link for phase 2 (data loading after sections render)
  // Handle deep links: uncollapse sections, load data, scroll into view
  useEffect(() => {
    if (!pendingDeepLink || !session) return
    const link = consumeDeepLink()
    if (!link) return

    console.log('[deep-link] handling:', link.type, link)

    // Uncollapse the appropriate group section
    if (link.type === 'table' || link.type === 'document' || link.type === 'api') {
      setSourcesCollapsed(false)
      localStorage.setItem('constat-sources-collapsed', 'false')
    }
    // Glossary is now top-level — no group to uncollapse

    const scrollToSection = (sectionId: string) => {
      // Double rAF to ensure DOM has updated after state changes
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          document.getElementById(`section-${sectionId}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
        })
      })
    }

    const loadData = async () => {
      switch (link.type) {
        case 'table':
          if (link.dbName && link.tableName) {
            setExpandedDb(link.dbName)
            setPreviewDb(null)
            setPreviewTable(null)
            setPreviewData(null)
            setDbTablesLoading(link.dbName)
            try {
              const res = await sessionsApi.listDatabaseTables(session.session_id, link.dbName)
              setDbTables((prev) => ({ ...prev, [link.dbName!]: res.tables }))
            } catch (err) {
              console.error('Failed to list tables:', err)
              setDbTables((prev) => ({ ...prev, [link.dbName!]: [] }))
            } finally {
              setDbTablesLoading(null)
            }
            openTablePreview(link.dbName, link.tableName)
            scrollToSection('databases')
          }
          break
        case 'document':
          if (link.documentName) {
            handleViewDocument(link.documentName)
            scrollToSection('documents')
          }
          break
        case 'api':
          if (link.apiName) {
            setExpandedApi(link.apiName)
            setExpandedEndpoint(null)
            setApiEndpointsLoading(link.apiName)
            try {
              const res = await sessionsApi.getApiSchema(session.session_id, link.apiName)
              setApiEndpoints((prev) => ({ ...prev, [link.apiName!]: res.endpoints }))
            } catch (err) {
              console.error('Failed to load API schema:', err)
              setApiEndpoints((prev) => ({ ...prev, [link.apiName!]: [] }))
            } finally {
              setApiEndpointsLoading(null)
            }
            scrollToSection('apis')
          }
          break
        case 'glossary_term':
          // Handled by GlossaryPanel via glossaryStore.selectTerm
          // Scroll to specific term element after a short delay for render
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              const el = document.getElementById(`glossary-term-${link.termName}`)
              if (el) {
                el.scrollIntoView({ behavior: 'smooth', block: 'center' })
              } else {
                scrollToSection('glossary')
              }
            })
          })
          break
      }
    }
    loadData()
  }, [pendingDeepLink])

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
      fetchAllAgents(session.session_id)
      fetchTaskRouting(session.session_id)
      fetchScratchpad(session.session_id)
      fetchDDL(session.session_id)
      // Fetch domain list for move-to pickers
      sessionsApi.getDomainTree().then((nodes) => {
        const collect = (ns: sessionsApi.DomainTreeNode[]): { filename: string; name: string }[] =>
          ns.flatMap((n) => [{ filename: n.filename, name: n.name }, ...collect(n.children)])
        setDomainList(collect(nodes))
      }).catch(() => {})
    }
  }, [session, fetchArtifacts, fetchTables, fetchFacts, fetchEntities, fetchLearnings, fetchDataSources, fetchPromptContext, fetchAllSkills, fetchAllAgents, fetchTaskRouting, fetchScratchpad, fetchDDL])

  // Auto-refresh fine-tune jobs when any are training
  useEffect(() => {
    const hasTraining = ftJobs.some(j => j.status === 'training')
    if (!hasTraining || learningsTab !== 'fine-tune') return
    const interval = setInterval(() => {
      sessionsApi.listFineTuneJobs().then(setFtJobs).catch(() => {})
    }, 30000)
    return () => clearInterval(interval)
  }, [ftJobs, learningsTab])

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

  const handleMoveSkill = async (skillName: string, fromDomain: string, toDomain: string) => {
    // Validate first
    const validation = await sessionsApi.moveSkill({ skill_name: skillName, from_domain: fromDomain, to_domain: toDomain, validate_only: true })
    if (validation.warnings && validation.warnings.length > 0) {
      const ok = window.confirm(`Warning:\n${validation.warnings.join('\n')}\n\nMove anyway?`)
      if (!ok) return
    }
    await sessionsApi.moveSkill({ skill_name: skillName, from_domain: fromDomain, to_domain: toDomain })
    setMovingSkill(null)
    fetchAllSkills()
  }

  const handleMoveAgent = async (agentName: string, fromDomain: string, toDomain: string) => {
    if (!session) return
    await sessionsApi.moveAgent({ agent_name: agentName, from_domain: fromDomain, to_domain: toDomain })
    setMovingAgent(null)
    fetchAllAgents(session.session_id)
  }

  const handleMoveFact = async (factName: string, toDomain: string) => {
    if (!session) return
    await sessionsApi.moveFact(session.session_id, factName, toDomain)
    setMovingFact(null)
    fetchFacts(session.session_id)
  }

  const handleMoveRule = async (ruleId: string, toDomain: string) => {
    await sessionsApi.moveRule({ rule_id: ruleId, to_domain: toDomain })
    setMovingRule(null)
    fetchLearnings()
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
      setUploading(true)
      try {
        if (modalInput.uri.startsWith('http://') || modalInput.uri.startsWith('https://')) {
          await sessionsApi.addDocumentURI(session.session_id, {
            name: modalInput.name,
            url: modalInput.uri,
            follow_links: docFollowLinks,
            max_depth: docMaxDepth,
            max_documents: docMaxDocuments,
            same_domain_only: docSameDomainOnly,
            type: docContentType,
          })
        } else {
          await sessionsApi.addFileRef(session.session_id, {
            name: modalInput.name,
            uri: modalInput.uri,
          })
        }
        fetchDataSources(session.session_id)
        // Entities refresh via entity_rebuild_complete WS event
        setShowModal(null)
      } catch (err) {
        console.error('Failed to add document:', err)
        alert(`Failed to add document: ${err instanceof Error ? err.message : 'Unknown error'}`)
      } finally {
        setUploading(false)
      }
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
      setIframeBlocked(false)
      setViewingDocument({
        name: doc.name || documentName,
        content: doc.content || '',
        format: doc.format,
        url: doc.url,
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

  const handleDraftAgent = async () => {
    if (!session || !newAgent.name.trim() || !newAgent.description.trim()) return
    setDraftingAgent(true)
    try {
      const result = await agentsApi.draftAgent(session.session_id, newAgent.name.trim(), newAgent.description.trim())
      setNewAgent(prev => ({ ...prev, prompt: result.prompt || '', description: result.description || prev.description, skills: result.skills || [] }))
    } catch (err) {
      console.error('Failed to draft agent:', err)
    } finally {
      setDraftingAgent(false)
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

  const handleToggleAgentExpand = async (agentName: string) => {
    if (!session) return

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
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = {}
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) headers['Authorization'] = `Bearer ${token}`
      }
      const response = await fetch(
        `/api/sessions/agents/${encodeURIComponent(agentName)}?session_id=${session.session_id}`,
        { headers, credentials: 'include' }
      )
      if (response.ok) {
        const data = await response.json()
        setAgentContents(prev => ({ ...prev, [agentName]: { prompt: data.prompt, description: data.description, skills: data.skills || [] } }))
      }
    } catch (err) {
      console.error('Failed to fetch agent content:', err)
    }
  }

  const handleEditAgent = async (agentName: string) => {
    if (!session) return
    try {
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = {}
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) headers['Authorization'] = `Bearer ${token}`
      }
      const response = await fetch(
        `/api/sessions/agents/${encodeURIComponent(agentName)}?session_id=${session.session_id}`,
        { headers, credentials: 'include' }
      )
      if (response.ok) {
        const data = await response.json()
        setEditingAgent({ name: data.name, prompt: data.prompt || '', description: data.description || '', skills: data.skills || [] })
      }
    } catch (err) {
      console.error('Failed to fetch agent content:', err)
    }
  }

  const handleCreateAgent = async () => {
    if (!session || !newAgent.name.trim() || !newAgent.prompt.trim()) return
    try {
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) headers['Authorization'] = `Bearer ${token}`
      }
      const response = await fetch(
        `/api/sessions/agents?session_id=${session.session_id}`,
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
        fetchAllAgents(session.session_id)
      }
    } catch (err) {
      console.error('Failed to create agent:', err)
    }
  }

  const handleUpdateAgent = async () => {
    if (!session || !editingAgent) return
    try {
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) headers['Authorization'] = `Bearer ${token}`
      }
      const response = await fetch(
        `/api/sessions/agents/${encodeURIComponent(editingAgent.name)}?session_id=${session.session_id}`,
        {
          method: 'PUT',
          headers,
          credentials: 'include',
          body: JSON.stringify({ prompt: editingAgent.prompt, description: editingAgent.description, skills: editingAgent.skills }),
        }
      )
      if (response.ok) {
        setEditingAgent(null)
        fetchAllAgents(session.session_id)
      }
    } catch (err) {
      console.error('Failed to update agent:', err)
    }
  }

  const handleDeleteAgent = async (agentName: string) => {
    if (!session || !confirm(`Delete agent "${agentName}"?`)) return
    try {
      const { useAuthStore, isAuthDisabled } = await import('@/store/authStore')
      const headers: Record<string, string> = {}
      if (!isAuthDisabled) {
        const token = await useAuthStore.getState().getToken()
        if (token) headers['Authorization'] = `Bearer ${token}`
      }
      const response = await fetch(
        `/api/sessions/agents/${encodeURIComponent(agentName)}?session_id=${session.session_id}`,
        {
          method: 'DELETE',
          headers,
          credentials: 'include',
        }
      )
      if (response.ok) {
        fetchAllAgents(session.session_id)
      }
    } catch (err) {
      console.error('Failed to delete agent:', err)
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

  // Sort by step_number ascending (1, 2, 3...), then by name
  allResults.sort((a, b) => {
    const stepDiff = (a.data.step_number || 0) - (b.data.step_number || 0)
    if (stepDiff !== 0) return stepDiff
    return a.data.name.localeCompare(b.data.name)
  })

  // Filter based on toggle
  const displayedResults = showPublishedOnly
    ? allResults.filter((r) => r.is_published)
    : allResults

  // Group results by step_number
  const resultsByStep: { stepNumber: number; goal: string | undefined; items: ResultItem[] }[] = []
  const stepMap = new Map<number, ResultItem[]>()
  for (const r of displayedResults) {
    const sn = r.data.step_number || 0
    if (!stepMap.has(sn)) stepMap.set(sn, [])
    stepMap.get(sn)!.push(r)
  }
  // Build ordered groups, look up goal from stepCodes
  const stepGoalMap = new Map(stepCodes.map((s) => [s.step_number, s.goal]))
  for (const [stepNumber, items] of stepMap) {
    resultsByStep.push({ stepNumber, goal: stepGoalMap.get(stepNumber), items })
  }

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
                    <div className="space-y-2">
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

                      {/* Crawling options (only for HTTP URLs) */}
                      {(modalInput.uri || '').startsWith('http') && (
                        <>
                          <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                            <input
                              type="checkbox"
                              checked={docFollowLinks}
                              onChange={(e) => setDocFollowLinks(e.target.checked)}
                              className="rounded text-primary-600"
                            />
                            Follow links (crawl linked pages)
                          </label>

                          {docFollowLinks && (
                            <div className="pl-6 space-y-2">
                              <div className="flex items-center gap-2">
                                <label className="text-xs text-gray-500 dark:text-gray-400 w-28">Max depth</label>
                                <select value={docMaxDepth} onChange={(e) => setDocMaxDepth(Number(e.target.value))} className="text-sm px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100">
                                  {[1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{n}</option>)}
                                </select>
                              </div>
                              <div className="flex items-center gap-2">
                                <label className="text-xs text-gray-500 dark:text-gray-400 w-28">Max documents</label>
                                <select value={docMaxDocuments} onChange={(e) => setDocMaxDocuments(Number(e.target.value))} className="text-sm px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100">
                                  {[5, 10, 20, 50, 100].map(n => <option key={n} value={n}>{n}</option>)}
                                </select>
                              </div>
                              <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                                <input
                                  type="checkbox"
                                  checked={docSameDomainOnly}
                                  onChange={(e) => setDocSameDomainOnly(e.target.checked)}
                                  className="rounded text-primary-600"
                                />
                                Same domain only
                              </label>
                            </div>
                          )}

                          <button
                            type="button"
                            onClick={() => setDocShowAdvanced(!docShowAdvanced)}
                            className="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
                          >
                            {docShowAdvanced ? 'Hide' : 'Show'} advanced options
                          </button>

                          {docShowAdvanced && (
                            <div className="space-y-2">
                              <div className="flex items-center gap-2">
                                <label className="text-xs text-gray-500 dark:text-gray-400 w-28">Content type</label>
                                <select value={docContentType} onChange={(e) => setDocContentType(e.target.value)} className="text-sm px-2 py-1 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100">
                                  <option value="auto">Auto</option>
                                  <option value="html">HTML</option>
                                  <option value="pdf">PDF</option>
                                  <option value="markdown">Markdown</option>
                                  <option value="text">Text</option>
                                </select>
                              </div>
                            </div>
                          )}
                        </>
                      )}
                    </div>
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
          <div className={`bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full flex flex-col ${
            viewingDocument?.url && !iframeBlocked ? 'max-w-5xl max-h-[90vh]' : 'max-w-3xl max-h-[80vh]'
          }`}>
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
                {viewingDocument?.url && (
                  <a
                    href={viewingDocument.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[10px] px-1.5 py-0.5 bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400 rounded hover:underline"
                  >
                    Open in browser
                  </a>
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
              ) : viewingDocument?.url && !iframeBlocked ? (
                <iframe
                  src={viewingDocument.url}
                  className="w-full h-[75vh] border-0 rounded"
                  sandbox="allow-scripts allow-same-origin allow-popups"
                  onError={() => setIframeBlocked(true)}
                  onLoad={(e) => {
                    // Detect X-Frame-Options blocking: iframe loads but content is blank
                    try {
                      const iframe = e.target as HTMLIFrameElement
                      // Cross-origin iframes throw on contentDocument access
                      // If it doesn't throw, check if it's a blank error page
                      if (iframe.contentDocument?.title === '') {
                        setIframeBlocked(true)
                      }
                    } catch {
                      // Cross-origin = loaded successfully (can't inspect but content is there)
                    }
                  }}
                />
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
      {totalCount > 0 && (() => {
        const finalInsight = messages.find((m) => m.isFinalInsight && m.stepDurationMs != null)
        const totalElapsedMs = finalInsight?.stepDurationMs ?? null

        return (
        <>
          <button
            onClick={() => {
              const newVal = !resultsCollapsed
              setResultsCollapsed(newVal)
              localStorage.setItem('constat-results-collapsed', String(newVal))
            }}
            className="w-full px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between hover:bg-gray-150 dark:hover:bg-gray-750 transition-colors"
          >
            <span className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Results ({displayedResults.length})
            </span>
            <div className="flex items-center gap-1.5">
              {totalElapsedMs != null && totalElapsedMs > 0 && (
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-gray-200 text-gray-500 dark:bg-gray-700 dark:text-gray-400" title="Total elapsed time">
                  {formatMs(totalElapsedMs)}
                </span>
              )}
              <span
                role="button"
                onClick={(e) => { e.stopPropagation(); toggleResultsFilter(); }}
                className={`text-[10px] px-2 py-0.5 rounded-full transition-colors ${
                  showPublishedOnly
                    ? 'bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-400'
                    : 'bg-gray-200 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                }`}
                title={showPublishedOnly ? 'Showing published only. Click to show all.' : 'Showing all. Click to show published only.'}
              >
                {showPublishedOnly ? 'published' : 'all'}
              </span>
              <ChevronRightIcon className={`w-3 h-3 text-gray-400 transition-transform ${resultsCollapsed ? '' : 'rotate-90'}`} />
            </div>
          </button>

          {!resultsCollapsed && (
          <div id="section-results" className="border-b border-gray-200 dark:border-gray-700 px-4 py-3 bg-white dark:bg-gray-800">
            {displayedResults.length === 0 ? (
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {showPublishedOnly && totalCount > 0
                  ? 'No published results yet. Click toggle to show all.'
                  : 'No results yet'}
              </p>
            ) : (
              <div className="space-y-3">
                {resultsByStep.map(({ stepNumber, goal, items }) => {
                  const isStepCollapsed = collapsedResultSteps.has(stepNumber)
                  const tableCount = items.filter(i => i.type === 'table').length
                  const artifactCount = items.filter(i => i.type === 'artifact').length
                  const tooltipParts = [
                    goal,
                    `${tableCount} table(s), ${artifactCount} artifact(s)`,
                  ].filter(Boolean).join('\n')
                  return (
                  <div key={`step-group-${stepNumber}`}>
                    {stepNumber > 0 && (
                      <button
                        onClick={() => toggleResultStep(stepNumber)}
                        className="flex items-center gap-1 text-[10px] font-medium text-gray-400 dark:text-gray-500 uppercase tracking-wide mb-1 px-1 hover:text-gray-600 dark:hover:text-gray-300 transition-colors w-full text-left"
                        title={tooltipParts}
                      >
                        <ChevronDownIcon className={`w-3 h-3 transition-transform ${isStepCollapsed ? '-rotate-90' : ''}`} />
                        Step {stepNumber}
                        <span className="text-gray-300 dark:text-gray-600 ml-auto normal-case">{items.length}</span>
                      </button>
                    )}
                    {(!isStepCollapsed || stepNumber === 0) && (
                    <div className="space-y-2">
                      {items.map((result) =>
                        result.type === 'table' ? (
                          <div key={`table-${result.data.name}`} id={`table-${result.data.name}`}>
                            <TableAccordion
                              table={result.data}
                              initiallyOpen={bestResultId === `table-${result.data.name}`}
                            />
                          </div>
                        ) : (
                          <div key={`artifact-${result.data.id}`} id={`artifact-${result.data.id}`}>
                            <ArtifactItemAccordion
                              artifact={result.data}
                              initiallyOpen={bestResultId === `artifact-${result.data.id}`}
                            />
                          </div>
                        )
                      )}
                    </div>
                    )}
                  </div>
                  )
                })}
              </div>
            )}
          </div>
          )}
        </>
        )
      })()}

      {/* ═══════════════ SOURCES ═══════════════ */}
      {sourcesVisible && (
      <button
        onClick={() => {
          const newVal = !sourcesCollapsed
          setSourcesCollapsed(newVal)
          localStorage.setItem('constat-sources-collapsed', String(newVal))
        }}
        className="w-full px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between hover:bg-gray-150 dark:hover:bg-gray-750 transition-colors"
      >
        <span className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          Sources ({databases.length + apis.length + documents.length + facts.length})
        </span>
        <ChevronRightIcon className={`w-3 h-3 text-gray-400 transition-transform ${sourcesCollapsed ? '' : 'rotate-90'}`} />
      </button>
      )}

      {/* Databases */}
      {sourcesVisible && !sourcesCollapsed && (
      <>
      {canSeeSection('databases') && (
      <AccordionSection
        id="databases"
        title="Databases"
        count={databases.length}
        icon={<CircleStackIcon className="w-4 h-4" />}
        command="/databases"
        action={
          canWrite('sources') ? (
            <button
              onClick={() => openModal('database')}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Add database"
            >
              <PlusIcon className="w-4 h-4" />
            </button>
          ) : <div className="w-6 h-6" />
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
                    {db.source === 'session' && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">session</span>
                    )}
                    {db.source && db.source !== 'session' && db.source !== 'config' && (
                      <DomainBadge domain={db.source.replace('.yaml', '')} />
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
      )}

      {/* APIs */}
      {canSeeSection('apis') && (
      <AccordionSection
        id="apis"
        title="APIs"
        count={apis.length}
        icon={<GlobeAltIcon className="w-4 h-4" />}
        command="/apis"
        action={
          canWrite('sources') ? (
            <button
              onClick={() => openModal('api')}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Add API"
          >
            <PlusIcon className="w-4 h-4" />
          </button>
          ) : <div className="w-6 h-6" />
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
                    {api.source === 'session' && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">session</span>
                    )}
                    {api.source && api.source !== 'session' && api.source !== 'config' && (
                      <DomainBadge domain={api.source.replace('.yaml', '')} />
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
                      const restTypes = allEps.filter((ep) => ep.kind === 'openapi/model')
                      const restOther = allEps.filter((ep) => !ep.kind?.startsWith('graphql_') && ep.kind !== 'rest' && ep.kind !== 'openapi/model' && !ep.http_method)
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
      )}

      {/* Documents */}
      {canSeeSection('documents') && (
      <AccordionSection
        id="documents"
        title="Documents"
        count={documents.length}
        icon={<DocumentTextIcon className="w-4 h-4" />}
        command="/docs"
        action={
          canWrite('sources') ? (
            <button
              onClick={() => openModal('document')}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Add document"
            >
              <PlusIcon className="w-4 h-4" />
            </button>
          ) : <div className="w-6 h-6" />
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
                    {doc.source === 'session' && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400">session</span>
                    )}
                    {doc.source && doc.source !== 'session' && doc.source !== 'config' && (
                      <DomainBadge domain={doc.source.replace('.yaml', '')} />
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
      )}

      {/* Facts (in Sources group) */}
      {canSeeSection('facts') && (
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
                  const csvRows = [
                    ['name', 'value'],
                    ...facts.map(f => [
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
                  <th className="text-left py-2 px-1 font-medium text-gray-600 dark:text-gray-400">Name</th>
                  <th className="text-left py-2 px-1 font-medium text-gray-600 dark:text-gray-400">Value</th>
                  <th className="text-left py-2 px-1 font-medium text-gray-600 dark:text-gray-400">Source</th>
                  <th className="w-8"></th>
                </tr>
              </thead>
              <tbody>
                {facts.map((fact) => (
                  <React.Fragment key={fact.name}>
                  <tr className="border-b border-gray-100 dark:border-gray-800 last:border-b-0 group">
                    <td className="py-2 px-1 font-medium text-gray-700 dark:text-gray-300">
                      <span className="flex items-center gap-1 flex-wrap">
                        {fact.name}
                        <DomainBadge domain={fact.source === 'config' ? 'system' : fact.domain || (fact.is_persisted ? 'user' : null)} />
                        {fact.role_id && <DomainBadge domain={fact.role_id} />}
                      </span>
                    </td>
                    <td className="py-2 px-1 text-gray-600 dark:text-gray-400">{String(fact.value)}</td>
                    <td className="py-2 px-1 text-xs text-gray-400 dark:text-gray-500">
                      <DomainBadge domain={fact.source === 'config' ? 'system' : fact.source} />
                    </td>
                    <td className="py-2 px-1 flex items-center gap-1">
                      {fact.source !== 'config' && (
                        <>
                          {!fact.is_persisted && (
                            <button onClick={() => handlePersistFact(fact.name)} className="p-1 text-gray-300 dark:text-gray-600 hover:text-amber-500 dark:hover:text-amber-400 opacity-0 group-hover:opacity-100 transition-opacity" title="Save permanently">
                              <ArrowUpTrayIcon className="w-3 h-3" />
                            </button>
                          )}
                          {fact.is_persisted && (
                            <button onClick={() => setMovingFact(movingFact === fact.name ? null : fact.name)} className="p-1 text-gray-300 dark:text-gray-600 hover:text-primary-600 dark:hover:text-primary-400 opacity-0 group-hover:opacity-100 transition-opacity" title="Move to domain">
                              <ArrowsRightLeftIcon className="w-3 h-3" />
                            </button>
                          )}
                          <button onClick={() => handleForgetFact(fact.name)} className="p-1 text-gray-300 dark:text-gray-600 hover:text-red-500 dark:hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity" title="Forget fact">
                            <MinusIcon className="w-3 h-3" />
                          </button>
                        </>
                      )}
                    </td>
                  </tr>
                  {movingFact === fact.name && (
                    <tr>
                      <td colSpan={4} className="py-1.5 px-1">
                        <div className="flex items-center gap-2 bg-blue-50 dark:bg-blue-900/20 rounded px-2 py-1">
                          <span className="text-[11px] text-gray-600 dark:text-gray-400">Move to:</span>
                          <select autoFocus className="text-[11px] bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded px-1.5 py-0.5" defaultValue="" onChange={(e) => { if (e.target.value) handleMoveFact(fact.name, e.target.value) }}>
                            <option value="" disabled>Select domain...</option>
                            {domainList.filter((d) => d.filename !== (fact.domain || '')).map((d) => (
                              <option key={d.filename} value={d.filename}>{d.name}</option>
                            ))}
                          </select>
                          <button onClick={() => setMovingFact(null)} className="text-[11px] text-gray-400 hover:text-gray-600">Cancel</button>
                        </div>
                      </td>
                    </tr>
                  )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </AccordionSection>
      )}

      </>
      )}

      {/* ═══════════════ GLOSSARY ═══════════════ */}
      {canSeeSection('glossary') && session && (
      <button
        onClick={() => {
          const newVal = !glossaryCollapsed
          setGlossaryCollapsed(newVal)
          localStorage.setItem('constat-glossary-collapsed', String(newVal))
        }}
        className="w-full px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between hover:bg-gray-150 dark:hover:bg-gray-750 transition-colors"
      >
        <span className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          Glossary ({totalDefined + totalSelfDescribing})
        </span>
        <ChevronRightIcon className={`w-3 h-3 text-gray-400 transition-transform ${glossaryCollapsed ? '' : 'rotate-90'}`} />
      </button>
      )}

      {canSeeSection('glossary') && session && !glossaryCollapsed && (
      <div id="section-glossary" className="border-b border-gray-200 dark:border-gray-700 px-4 py-3 bg-white dark:bg-gray-800">
        <GlossaryPanel sessionId={session.session_id} />
      </div>
      )}

      {/* ═══════════════ REASONING ═══════════════ */}
      {reasoningVisible && (
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
      )}

      {reasoningVisible && !reasoningCollapsed && (
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
                  {allSkills.map((skill) => (
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
                    {allSkills.map((skill) => (
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

      {/* --- Improvement sub-group --- */}
      {improvementVisible && (
      <button
        onClick={() => {
          const newVal = !improvementCollapsed
          setImprovementCollapsed(newVal)
          localStorage.setItem('constat-improvement-collapsed', String(newVal))
        }}
        className="w-full px-4 py-1.5 bg-gray-50 dark:bg-gray-800/50 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between hover:bg-gray-100 dark:hover:bg-gray-750 transition-colors"
      >
        <span className="text-[9px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider pl-2">
          Improvement
        </span>
        <ChevronRightIcon className={`w-3 h-3 text-gray-400 transition-transform ${improvementCollapsed ? '' : 'rotate-90'}`} />
      </button>
      )}

      {improvementVisible && !improvementCollapsed && (
      <>

      {/* Learnings */}
      {canSeeSection('learnings') && (
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
            {/* Tab bar */}
            <div className="flex gap-1 border-b border-gray-200 dark:border-gray-700">
              <button
                onClick={() => setLearningsTab('rules')}
                className={`px-3 py-1.5 text-xs font-medium border-b-2 transition-colors ${
                  learningsTab === 'rules'
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                }`}
              >
                Rules ({rules.length})
              </button>
              <button
                onClick={() => setLearningsTab('pending')}
                className={`px-3 py-1.5 text-xs font-medium border-b-2 transition-colors ${
                  learningsTab === 'pending'
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                }`}
              >
                Pending ({learnings.length})
              </button>
              <button
                onClick={() => setLearningsTab('export')}
                className={`px-3 py-1.5 text-xs font-medium border-b-2 transition-colors ${
                  learningsTab === 'export'
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                }`}
              >
                Export
              </button>
              <button
                onClick={() => {
                  setLearningsTab('fine-tune')
                  sessionsApi.listFineTuneJobs().then(setFtJobs).catch(() => {})
                  sessionsApi.listFineTuneProviders().then((p) => {
                    setFtProviders(p)
                    if (p.length > 0 && !ftProvider) {
                      setFtProvider(p[0].name)
                      if (p[0].models.length > 0) setFtBaseModel(p[0].models[0])
                    }
                  }).catch(() => {})
                }}
                className={`px-3 py-1.5 text-xs font-medium border-b-2 transition-colors ${
                  learningsTab === 'fine-tune'
                    ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                    : 'border-transparent text-gray-500 hover:text-gray-700 dark:hover:text-gray-300'
                }`}
              >
                Fine-Tune
              </button>
            </div>

            {/* Rules tab */}
            {learningsTab === 'rules' && rules.length > 0 && (
              <div className="space-y-2">
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
                              onClick={() => setMovingRule(movingRule === rule.id ? null : rule.id)}
                              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 rounded"
                              title="Move to domain"
                            >
                              <ArrowsRightLeftIcon className="w-3 h-3" />
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
                        {/* Move-to-domain picker */}
                        {movingRule === rule.id && (
                          <div className="mt-1 flex items-center gap-2 bg-blue-50 dark:bg-blue-900/20 rounded px-2 py-1">
                            <span className="text-[11px] text-gray-600 dark:text-gray-400">Move to:</span>
                            <select
                              autoFocus
                              className="text-[11px] bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded px-1.5 py-0.5"
                              defaultValue=""
                              onChange={(e) => { if (e.target.value) handleMoveRule(rule.id, e.target.value) }}
                            >
                              <option value="" disabled>Select domain...</option>
                              {domainList.filter((d) => d.filename !== (rule.domain || '')).map((d) => (
                                <option key={d.filename} value={d.filename}>{d.name}</option>
                              ))}
                            </select>
                            <button onClick={() => setMovingRule(null)} className="text-[11px] text-gray-400 hover:text-gray-600">Cancel</button>
                          </div>
                        )}
                        <div className="mt-1 flex items-center gap-2 text-xs text-gray-400 dark:text-gray-500">
                          <span className="px-1.5 py-0.5 bg-green-200 dark:bg-green-800 text-green-800 dark:text-green-200 rounded">
                            {Math.round(rule.confidence * 100)}% confidence
                          </span>
                          <span>{rule.source_count} sources</span>
                          <DomainBadge domain={rule.domain || 'user'} />
                          <ScopeBadge scope={rule.scope} />
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
            {learningsTab === 'rules' && rules.length === 0 && (
              <p className="text-sm text-gray-500 dark:text-gray-400">No rules yet. Compact pending learnings to generate rules.</p>
            )}

            {/* Pending tab */}
            {learningsTab === 'pending' && learnings.length > 0 && (
              <div className="space-y-2">
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
                      <DomainBadge domain="user" />
                      <ScopeBadge scope={learning.scope} />
                      {learning.applied_count > 0 && (
                        <span>Applied {learning.applied_count}x</span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
            {learningsTab === 'pending' && learnings.length === 0 && (
              <p className="text-sm text-gray-500 dark:text-gray-400">No pending learnings.</p>
            )}

            {/* Export tab */}
            {learningsTab === 'export' && (
              <div className="space-y-3">
                <div>
                  <label className="text-xs font-medium text-gray-600 dark:text-gray-400">Format</label>
                  <select
                    value={exportFormat}
                    onChange={(e) => setExportFormat(e.target.value as typeof exportFormat)}
                    className="mt-1 w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  >
                    <option value="messages">OpenAI Messages</option>
                    <option value="alpaca">Alpaca</option>
                    <option value="sharegpt">ShareGPT</option>
                  </select>
                </div>

                <div>
                  <label className="text-xs font-medium text-gray-600 dark:text-gray-400">Include</label>
                  <div className="mt-1 space-y-1">
                    {(['corrections', 'rules', 'glossary'] as const).map((item) => (
                      <label key={item} className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                        <input
                          type="checkbox"
                          checked={exportInclude.has(item)}
                          onChange={(e) => {
                            const next = new Set(exportInclude)
                            if (e.target.checked) next.add(item); else next.delete(item)
                            setExportInclude(next)
                          }}
                          className="rounded text-primary-600"
                        />
                        {item === 'corrections' ? `Corrections (${learnings.length})` :
                         item === 'rules' ? `Rules (${rules.length})` :
                         'Glossary terms'}
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="text-xs font-medium text-gray-600 dark:text-gray-400">
                    Min rule confidence: {Math.round(exportMinConfidence * 100)}%
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    step={5}
                    value={exportMinConfidence * 100}
                    onChange={(e) => setExportMinConfidence(Number(e.target.value) / 100)}
                    className="mt-1 w-full"
                  />
                </div>

                <button
                  onClick={async () => {
                    setExporting(true)
                    try {
                      await sessionsApi.downloadSimpleExemplars({
                        format: exportFormat,
                        include: Array.from(exportInclude),
                        min_confidence: exportMinConfidence,
                      })
                    } catch (err) {
                      console.error('Export failed:', err)
                    } finally {
                      setExporting(false)
                    }
                  }}
                  disabled={exporting || exportInclude.size === 0}
                  className="w-full px-3 py-2 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded-md disabled:opacity-50 transition-colors flex items-center justify-center gap-2"
                >
                  <ArrowDownTrayIcon className="w-4 h-4" />
                  {exporting ? 'Downloading...' : 'Download JSONL'}
                </button>
              </div>
            )}

            {/* Fine-Tune tab */}
            {learningsTab === 'fine-tune' && (
              <div className="space-y-3">
                {/* Job list */}
                {ftJobs.length > 0 && (
                  <div className="space-y-2">
                    {ftJobs.map((job) => (
                      <div
                        key={job.id}
                        className="p-2 bg-gray-50 dark:bg-gray-800/50 rounded-lg group"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
                                {job.name}
                              </span>
                              <span className={`px-1.5 py-0.5 text-[10px] font-medium rounded ${
                                job.status === 'ready' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300' :
                                job.status === 'training' ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300 animate-pulse' :
                                job.status === 'failed' ? 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300' :
                                'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                              }`}>
                                {job.status}
                              </span>
                            </div>
                            <div className="mt-0.5 text-[11px] text-gray-500 dark:text-gray-400">
                              {job.provider}/{job.base_model} · {job.exemplar_count} examples
                              {job.task_types.length > 0 && ` · ${job.task_types.join(', ')}`}
                            </div>
                          </div>
                          <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            {job.status === 'training' && (
                              <button
                                onClick={async () => {
                                  await sessionsApi.cancelFineTuneJob(job.id)
                                  setFtJobs(ftJobs.map(j => j.id === job.id ? { ...j, status: 'failed' as const } : j))
                                }}
                                className="p-1 text-gray-400 hover:text-yellow-600 rounded"
                                title="Cancel"
                              >
                                <XMarkIcon className="w-3 h-3" />
                              </button>
                            )}
                            <button
                              onClick={async () => {
                                await sessionsApi.deleteFineTuneJob(job.id)
                                setFtJobs(ftJobs.filter(j => j.id !== job.id))
                              }}
                              className="p-1 text-gray-400 hover:text-red-500 rounded"
                              title="Delete"
                            >
                              <TrashIcon className="w-3 h-3" />
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* New job form */}
                {ftShowForm ? (
                  <div className="space-y-2 p-2 border border-gray-200 dark:border-gray-700 rounded-lg">
                    <input
                      type="text"
                      placeholder="Model name (e.g., sales-sql-v1)"
                      value={ftName}
                      onChange={(e) => setFtName(e.target.value)}
                      className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    />

                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className="text-[11px] font-medium text-gray-600 dark:text-gray-400">Provider</label>
                        <select
                          value={ftProvider}
                          onChange={(e) => {
                            setFtProvider(e.target.value)
                            const prov = ftProviders.find(p => p.name === e.target.value)
                            if (prov && prov.models.length > 0) setFtBaseModel(prov.models[0])
                          }}
                          className="mt-0.5 w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                        >
                          {ftProviders.length === 0 && <option value="">No providers (set API keys)</option>}
                          {ftProviders.map(p => (
                            <option key={p.name} value={p.name}>{p.name}</option>
                          ))}
                        </select>
                      </div>
                      <div>
                        <label className="text-[11px] font-medium text-gray-600 dark:text-gray-400">Base Model</label>
                        <select
                          value={ftBaseModel}
                          onChange={(e) => setFtBaseModel(e.target.value)}
                          className="mt-0.5 w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                        >
                          {(ftProviders.find(p => p.name === ftProvider)?.models || []).map((m: string) => (
                            <option key={m} value={m}>{m}</option>
                          ))}
                        </select>
                      </div>
                    </div>

                    <div>
                      <label className="text-[11px] font-medium text-gray-600 dark:text-gray-400">Task Types</label>
                      <div className="mt-0.5 flex flex-wrap gap-1">
                        {['sql_generation', 'python_analysis', 'planning', 'summarization'].map(tt => (
                          <label key={tt} className="flex items-center gap-1 text-[11px] text-gray-700 dark:text-gray-300">
                            <input
                              type="checkbox"
                              checked={ftTaskTypes.has(tt)}
                              onChange={(e) => {
                                const next = new Set(ftTaskTypes)
                                if (e.target.checked) next.add(tt); else next.delete(tt)
                                setFtTaskTypes(next)
                              }}
                              className="rounded text-primary-600"
                            />
                            {tt}
                          </label>
                        ))}
                      </div>
                    </div>

                    <div>
                      <label className="text-[11px] font-medium text-gray-600 dark:text-gray-400">Training Data</label>
                      <div className="mt-0.5 flex flex-wrap gap-2">
                        {(['corrections', 'rules', 'glossary'] as const).map(item => (
                          <label key={item} className="flex items-center gap-1 text-[11px] text-gray-700 dark:text-gray-300">
                            <input
                              type="checkbox"
                              checked={ftInclude.has(item)}
                              onChange={(e) => {
                                const next = new Set(ftInclude)
                                if (e.target.checked) next.add(item); else next.delete(item)
                                setFtInclude(next)
                              }}
                              className="rounded text-primary-600"
                            />
                            {item}
                          </label>
                        ))}
                      </div>
                    </div>

                    <div>
                      <label className="text-[11px] font-medium text-gray-600 dark:text-gray-400">Domain (optional)</label>
                      <select
                        value={ftDomain}
                        onChange={(e) => setFtDomain(e.target.value)}
                        className="mt-0.5 w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      >
                        <option value="">All domains (cross-domain)</option>
                        {domainList.map(d => (
                          <option key={d.filename} value={d.filename}>{d.name}</option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="text-[11px] font-medium text-gray-600 dark:text-gray-400">
                        Min confidence: {Math.round(ftMinConf * 100)}%
                      </label>
                      <input
                        type="range"
                        min={0} max={100} step={5}
                        value={ftMinConf * 100}
                        onChange={(e) => setFtMinConf(Number(e.target.value) / 100)}
                        className="mt-0.5 w-full"
                      />
                    </div>

                    <div className="flex gap-2">
                      <button
                        onClick={async () => {
                          if (!ftName || !ftProvider || !ftBaseModel || ftTaskTypes.size === 0) return
                          setFtSubmitting(true)
                          try {
                            const job = await sessionsApi.startFineTuneJob({
                              name: ftName,
                              provider: ftProvider,
                              base_model: ftBaseModel,
                              task_types: Array.from(ftTaskTypes),
                              domain: ftDomain || undefined,
                              include: Array.from(ftInclude),
                              min_confidence: ftMinConf,
                            })
                            setFtJobs([job, ...ftJobs])
                            setFtShowForm(false)
                            setFtName('')
                          } catch (err) {
                            console.error('Fine-tune start failed:', err)
                          } finally {
                            setFtSubmitting(false)
                          }
                        }}
                        disabled={ftSubmitting || !ftName || !ftProvider || !ftBaseModel || ftTaskTypes.size === 0}
                        className="flex-1 px-3 py-1.5 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded disabled:opacity-50 transition-colors"
                      >
                        {ftSubmitting ? 'Starting...' : 'Start Training'}
                      </button>
                      <button
                        onClick={() => setFtShowForm(false)}
                        className="px-3 py-1.5 text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <button
                    onClick={() => setFtShowForm(true)}
                    className="w-full px-3 py-2 text-sm font-medium text-primary-600 dark:text-primary-400 border border-dashed border-primary-300 dark:border-primary-700 hover:bg-primary-50 dark:hover:bg-primary-900/20 rounded-md transition-colors flex items-center justify-center gap-2"
                  >
                    <PlusIcon className="w-4 h-4" />
                    New Fine-Tune Job
                  </button>
                )}

                {ftJobs.length === 0 && !ftShowForm && (
                  <p className="text-sm text-gray-500 dark:text-gray-400">No fine-tuning jobs yet.</p>
                )}
              </div>
            )}
          </div>
        )}
      </AccordionSection>
      )}

      {/* Regression Tests (in Improvement sub-group) */}
      {session && (
        <AccordionSection
          id="regression"
          title="Regression Tests"
          icon={<BeakerIcon className="w-4 h-4" />}
        >
          <RegressionPanel sessionId={session.session_id} />
        </AccordionSection>
      )}

      </>
      )}

      {/* --- Code Log sub-group --- */}
      {codeLogVisible && (
      <button
        onClick={() => {
          const newVal = !codeLogCollapsed
          setCodeLogCollapsed(newVal)
          localStorage.setItem('constat-codelog-collapsed', String(newVal))
        }}
        className="w-full px-4 py-1.5 bg-gray-50 dark:bg-gray-800/50 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between hover:bg-gray-100 dark:hover:bg-gray-750 transition-colors"
      >
        <span className="text-[9px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider pl-2">
          Code Log
        </span>
        <ChevronRightIcon className={`w-3 h-3 text-gray-400 transition-transform ${codeLogCollapsed ? '' : 'rotate-90'}`} />
      </button>
      )}

      {codeLogVisible && !codeLogCollapsed && (
      <>

      {/* Scratchpad - execution narrative per step */}
      {scratchpadEntries.length > 0 && (
        <AccordionSection
          id="scratchpad"
          title="Scratchpad"
          count={scratchpadEntries.length}
          icon={<DocumentTextIcon className="w-4 h-4" />}
        >
          <ScratchpadList entries={scratchpadEntries} supersededStepNumbers={supersededStepNumbers} tables={tables} />
        </AccordionSection>
      )}

      {/* Code - only show when there's code and user can see it */}
      {canSeeSection('code') && stepCodes.length > 0 && (
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
          <StepCodeList stepCodes={stepCodes} supersededStepNumbers={supersededStepNumbers} tables={tables} artifacts={artifacts} />
        </AccordionSection>
      )}

      {/* Inference Code - auditable mode (separate from step code) */}
      {canSeeSection('inference_code') && inferenceCodes.length > 0 && (
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
          <div className="space-y-1">
            {inferenceCodes.map((inf) => {
              const isCollapsed = collapsedInferences.has(inf.inference_id)
              return (
                <div key={inf.inference_id}>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => setCollapsedInferences((prev) => {
                        const next = new Set(prev)
                        if (next.has(inf.inference_id)) next.delete(inf.inference_id)
                        else next.add(inf.inference_id)
                        return next
                      })}
                      className="flex items-start gap-1 flex-1 text-left text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 py-1 transition-colors"
                    >
                      {isCollapsed ? (
                        <ChevronRightIcon className="w-3 h-3 flex-shrink-0 mt-0.5" />
                      ) : (
                        <ChevronDownIcon className="w-3 h-3 flex-shrink-0 mt-0.5" />
                      )}
                      <span className="flex flex-col">
                        <span>{inf.inference_id}: {inf.name} = {inf.operation}</span>
                        {inf.model && (
                          <span className="text-[10px] text-gray-400 dark:text-gray-500 font-mono">{inf.model}</span>
                        )}
                      </span>
                    </button>
                    {inf.prompt && (
                      <button
                        onClick={() => setShowInferencePrompt((prev) => {
                          const next = new Set(prev)
                          if (next.has(inf.inference_id)) next.delete(inf.inference_id)
                          else next.add(inf.inference_id)
                          return next
                        })}
                        className="text-[10px] text-gray-400 hover:text-primary-500 ml-auto px-1"
                      >
                        {showInferencePrompt.has(inf.inference_id) ? 'Hide Prompt' : 'Prompt'}
                      </button>
                    )}
                  </div>
                  {!isCollapsed && (
                    <>
                      {showInferencePrompt.has(inf.inference_id) && inf.prompt && (
                        <pre className="text-[10px] leading-tight text-gray-400 bg-gray-50 dark:bg-gray-800/50 p-2 rounded overflow-auto max-h-60 mb-1 whitespace-pre-wrap">
                          {inf.prompt}
                        </pre>
                      )}
                      <CodeViewer
                        code={inf.code}
                        language="python"
                      />
                    </>
                  )}
                </div>
              )
            })}
          </div>
        </AccordionSection>
      )}

      {/* Session Store DDL */}
      {sessionDDL && (
        <AccordionSection
          id="session-ddl"
          title="Session Store"
          icon={<CircleStackIcon className="w-4 h-4" />}
          action={
            <button
              onClick={() => {
                if (session) fetchDDL(session.session_id)
              }}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Refresh DDL"
            >
              <ArrowPathIcon className="w-4 h-4" />
            </button>
          }
        >
          <pre className="text-[11px] leading-relaxed text-gray-600 dark:text-gray-300 bg-gray-50 dark:bg-gray-800/50 p-3 rounded overflow-auto max-h-96 whitespace-pre font-mono">
            {sessionDDL}
          </pre>
        </AccordionSection>
      )}

      </>
      )}

      </>
      )}
    </div>
  )
}