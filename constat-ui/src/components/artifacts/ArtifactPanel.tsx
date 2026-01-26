// Artifact Panel container

import { useEffect, useState } from 'react'
import {
  ChartBarIcon,
  TableCellsIcon,
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
  PencilIcon,
  TrashIcon,
  CheckIcon,
  XMarkIcon,
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

type ModalType = 'database' | 'api' | 'document' | 'fact' | 'rule' | null

export function ArtifactPanel() {
  const { session } = useSessionStore()
  const { expandedArtifactSections } = useUIStore()
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
    fetchArtifacts,
    fetchTables,
    fetchFacts,
    fetchEntities,
    fetchLearnings,
    fetchDataSources,
  } = useArtifactStore()

  const [showModal, setShowModal] = useState<ModalType>(null)
  const [modalInput, setModalInput] = useState({ name: '', value: '', uri: '', type: '', persist: false })
  const [compacting, setCompacting] = useState(false)
  const [editingRule, setEditingRule] = useState<{ id: string; summary: string } | null>(null)
  // Document modal state
  const [docSourceType, setDocSourceType] = useState<'uri' | 'files'>('uri')
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)

  // Fetch data when session changes
  useEffect(() => {
    if (session) {
      fetchArtifacts(session.session_id)
      fetchTables(session.session_id)
      fetchFacts(session.session_id)
      fetchEntities(session.session_id)
      fetchLearnings()
      fetchDataSources(session.session_id)
    }
  }, [session, fetchArtifacts, fetchTables, fetchFacts, fetchEntities, fetchLearnings, fetchDataSources])

  // Handlers
  const handleForgetFact = async (factName: string) => {
    if (!session) return
    await sessionsApi.forgetFact(session.session_id, factName)
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

  // Visualizations: charts, images, HTML reports, markdown, etc.
  const visualArtifacts = artifacts.filter((a) =>
    ['chart', 'plotly', 'svg', 'png', 'jpeg', 'html', 'image', 'markdown', 'md', 'vega'].includes(a.artifact_type?.toLowerCase())
  )
  // Key artifacts: marked as important results to keep
  const keyArtifacts = artifacts.filter((a) => a.is_key_result)

  // Helper to check if name/title contains priority keywords
  const hasPriorityKeyword = (name?: string, title?: string): boolean => {
    const text = `${name || ''} ${title || ''}`.toLowerCase()
    return ['final', 'recommended', 'answer', 'result', 'conclusion'].some(kw => text.includes(kw))
  }

  // Helper to find best item from a list (prefers items with priority keywords)
  const findBestItem = <T extends { name: string; title?: string }>(items: T[]): T | null => {
    if (items.length === 0) return null
    if (items.length === 1) return items[0]
    // Multiple items: prefer one with priority keyword
    const withKeyword = items.find(item => hasPriorityKeyword(item.name, item.title))
    return withKeyword || items[0]
  }

  // Determine the "best" artifact to auto-expand
  // Priority: 1) visualizations in key artifacts, 2) tables in key artifacts
  // Only auto-expand if the parent section is already expanded
  const keyVisualizations = keyArtifacts.filter((a) =>
    ['chart', 'plotly', 'svg', 'png', 'jpeg', 'html', 'image', 'markdown', 'md', 'vega'].includes(a.artifact_type?.toLowerCase())
  )
  const keyTables = keyArtifacts.filter((a) => a.artifact_type === 'table')

  // Check which sections are expanded
  const isArtifactsSectionExpanded = expandedArtifactSections.includes('artifacts')
  const isTablesSectionExpanded = expandedArtifactSections.includes('tables')

  let bestArtifactId: number | null = null
  let bestTableName: string | null = null

  // Only auto-expand items if their parent section is expanded
  if (isArtifactsSectionExpanded && keyVisualizations.length > 0) {
    // Rule 1 & 2: Use visualization (prefer one with priority keyword if multiple)
    const best = findBestItem(keyVisualizations)
    bestArtifactId = best?.id ?? null
  } else if (isArtifactsSectionExpanded && keyTables.length > 0) {
    // Rule 3 & 4: Use table from key artifacts (prefer one with priority keyword if multiple)
    const best = findBestItem(keyTables)
    bestArtifactId = best?.id ?? null
  } else if (isTablesSectionExpanded && tables.length > 0) {
    // Fallback: No key artifacts, expand first table in Tables section (only if section expanded)
    const best = findBestItem(tables)
    bestTableName = best?.name ?? null
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

      {/* ═══════════════ RESULTS ═══════════════ */}
      {(keyArtifacts.length > 0 || visualArtifacts.length > 0 || tables.length > 0) && (
        <>
          <div className="px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
            <span className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Results
            </span>
          </div>

          {/* Key Artifacts */}
          {keyArtifacts.length > 0 && (
            <AccordionSection
              id="artifacts"
              title="Artifacts"
              count={keyArtifacts.length}
              icon={<StarIcon className="w-4 h-4" />}
              command="/artifacts"
              action={<div className="w-6 h-6" />}
            >
              <div className="space-y-2">
                {keyArtifacts.map((artifact) => (
                  <ArtifactItemAccordion
                    key={artifact.id}
                    artifact={artifact}
                    initiallyOpen={artifact.id === bestArtifactId}
                  />
                ))}
              </div>
            </AccordionSection>
          )}

          {/* Visualizations */}
          {visualArtifacts.length > 0 && (
            <AccordionSection
              id="visualizations"
              title="Visualizations"
              count={visualArtifacts.length}
              icon={<ChartBarIcon className="w-4 h-4" />}
              action={<div className="w-6 h-6" />}
            >
              <div className="space-y-2">
                {visualArtifacts.map((artifact) => (
                  <ArtifactItemAccordion
                    key={artifact.id}
                    artifact={artifact}
                    initiallyOpen={artifact.id === bestArtifactId}
                  />
                ))}
              </div>
            </AccordionSection>
          )}

          {/* Tables */}
          {tables.length > 0 && (
            <AccordionSection
              id="tables"
              title="Tables"
              count={tables.length}
              icon={<TableCellsIcon className="w-4 h-4" />}
              command="/tables"
              action={<div className="w-6 h-6" />}
            >
              <div className="space-y-2">
                {tables.map((table) => (
                  <TableAccordion
                    key={table.name}
                    table={table}
                    initiallyOpen={table.name === bestTableName}
                  />
                ))}
              </div>
            </AccordionSection>
          )}
        </>
      )}

      {/* ═══════════════ SOURCES ═══════════════ */}
      <div className="px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <span className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          Sources
        </span>
      </div>

      {/* Databases */}
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
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    {db.name}
                  </span>
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
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    {api.name}
                  </span>
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
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    {doc.name}
                  </span>
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

      {/* ═══════════════ REASONING ═══════════════ */}
      {/* Always show header since Facts always has an action */}
      <div className="px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <span className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          Reasoning
        </span>
      </div>

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
                    `/api/sessions/${session.session_id}/download-code`
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
                      <span className="flex items-center gap-1">
                        {fact.name}
                        {fact.is_persisted && (
                          <span className="px-1 py-0.5 text-[10px] bg-amber-200 dark:bg-amber-800 text-amber-800 dark:text-amber-200 rounded">
                            saved
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
                    <td className="py-2 px-1">
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
    </div>
  )
}