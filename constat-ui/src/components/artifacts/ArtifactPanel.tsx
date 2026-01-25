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
} from '@heroicons/react/24/outline'
import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import { AccordionSection } from './ArtifactAccordion'
import { TableAccordion } from './TableAccordion'
import { CodeViewer } from './CodeViewer'
import { EntityAccordion } from './EntityAccordion'
import * as sessionsApi from '@/api/sessions'

type ModalType = 'database' | 'api' | 'document' | 'fact' | null

export function ArtifactPanel() {
  const { session } = useSessionStore()
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
    selectedArtifact,
    fetchArtifacts,
    fetchTables,
    fetchFacts,
    fetchEntities,
    fetchLearnings,
    fetchDataSources,
    selectArtifact,
  } = useArtifactStore()

  const [showModal, setShowModal] = useState<ModalType>(null)
  const [modalInput, setModalInput] = useState({ name: '', value: '', uri: '', type: '', persist: false })
  const [compacting, setCompacting] = useState(false)

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
    if (!session || !modalInput.name || !modalInput.uri) return
    await sessionsApi.addFileRef(session.session_id, {
      name: modalInput.name,
      uri: modalInput.uri,
    })
    fetchDataSources(session.session_id)
    setShowModal(null)
    setModalInput({ name: '', value: '', uri: '', type: '', persist: false })
  }

  const openModal = (type: ModalType) => {
    setModalInput({ name: '', value: '', uri: '', type: '', persist: false })
    setShowModal(type)
  }

  // Visualizations: charts, images, HTML reports, markdown, etc.
  const visualArtifacts = artifacts.filter((a) =>
    ['chart', 'plotly', 'svg', 'png', 'jpeg', 'html', 'image', 'markdown', 'vega'].includes(a.artifact_type)
  )
  // Key artifacts: marked as important results to keep
  const keyArtifacts = artifacts.filter((a) => a.is_key_result)

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
              Add {showModal === 'fact' ? 'Fact' : showModal === 'database' ? 'Database' : showModal === 'api' ? 'API' : 'Document'}
            </h3>
            <div className="space-y-3">
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
                  else setShowModal(null) // API - not implemented yet
                }}
                className="px-3 py-1.5 text-sm bg-primary-600 text-white rounded-md hover:bg-primary-700"
              >
                Add
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ═══════════════ SETUP ═══════════════ */}
      <div className="px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <span className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          Setup
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
                className="p-2 bg-gray-50 dark:bg-gray-800/50 rounded-md"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    {db.name}
                  </span>
                  <span className="text-xs px-1.5 py-0.5 rounded bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400">
                    {db.type}
                  </span>
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
                className="p-2 bg-gray-50 dark:bg-gray-800/50 rounded-md"
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    {doc.name}
                  </span>
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded ${
                      doc.indexed
                        ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400'
                        : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                    }`}
                  >
                    {doc.indexed ? 'Indexed' : 'Pending'}
                  </span>
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

      {/* ═══════════════ RESULTS ═══════════════ */}
      {(keyArtifacts.length > 0 || visualArtifacts.length > 0 || tables.length > 0 || selectedArtifact) && (
        <>
          <div className="px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
            <span className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
              Results
            </span>
          </div>

          {/* Selected Artifact Viewer */}
          {selectedArtifact && (
            <div className="border-b border-gray-200 dark:border-gray-700">
              <div className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    {selectedArtifact.name}
                  </h4>
                  <button
                    onClick={() => useArtifactStore.setState({ selectedArtifact: null })}
                    className="text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                  >
                    Close
                  </button>
                </div>
                <div className="rounded-lg overflow-hidden bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700">
                  {/* Render based on artifact type */}
                  {selectedArtifact.is_binary && selectedArtifact.mime_type.startsWith('image/') ? (
                    <img
                      src={`data:${selectedArtifact.mime_type};base64,${selectedArtifact.content}`}
                      alt={selectedArtifact.name}
                      className="max-w-full h-auto"
                    />
                  ) : selectedArtifact.mime_type === 'text/html' ? (
                    <iframe
                      srcDoc={selectedArtifact.content}
                      className="w-full h-64 border-0"
                      title={selectedArtifact.name}
                      sandbox="allow-scripts"
                    />
                  ) : selectedArtifact.artifact_type === 'plotly' ? (
                    <iframe
                      srcDoc={`
                        <!DOCTYPE html>
                        <html>
                        <head>
                          <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
                        </head>
                        <body style="margin:0">
                          <div id="plot"></div>
                          <script>
                            const data = ${selectedArtifact.content};
                            Plotly.newPlot('plot', data.data || data, data.layout || {}, {responsive: true});
                          </script>
                        </body>
                        </html>
                      `}
                      className="w-full h-64 border-0"
                      title={selectedArtifact.name}
                      sandbox="allow-scripts"
                    />
                  ) : (
                    <pre className="p-3 text-xs text-gray-600 dark:text-gray-400 overflow-auto max-h-48">
                      {selectedArtifact.content.substring(0, 2000)}
                      {selectedArtifact.content.length > 2000 && '...'}
                    </pre>
                  )}
                </div>
              </div>
            </div>
          )}

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
              <div className="space-y-3">
                {keyArtifacts.map((artifact) => (
                  <button
                    key={artifact.id}
                    onClick={() => session && selectArtifact(session.session_id, artifact.id)}
                    className={`w-full text-left p-3 rounded-lg transition-colors ${
                      selectedArtifact?.id === artifact.id
                        ? 'bg-yellow-100 dark:bg-yellow-900/40 border-2 border-yellow-400 dark:border-yellow-600'
                        : 'bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 hover:bg-yellow-100 dark:hover:bg-yellow-900/30'
                    }`}
                  >
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      {artifact.title || artifact.name}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {artifact.artifact_type} · Step {artifact.step_number}
                    </p>
                  </button>
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
              <div className="space-y-3">
                {visualArtifacts.map((artifact) => (
                  <button
                    key={artifact.id}
                    onClick={() => session && selectArtifact(session.session_id, artifact.id)}
                    className={`w-full text-left p-3 rounded-lg transition-colors ${
                      selectedArtifact?.id === artifact.id
                        ? 'bg-primary-100 dark:bg-primary-900/40 border-2 border-primary-400 dark:border-primary-600'
                        : 'bg-gray-50 dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-700/50'
                    }`}
                  >
                    <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      {artifact.title || artifact.name}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Step {artifact.step_number}
                    </p>
                  </button>
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
                  <TableAccordion key={table.name} table={table} />
                ))}
              </div>
            </AccordionSection>
          )}
        </>
      )}

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

      {/* Learnings - only show when there are learnings or rules */}
      {(learnings.length > 0 || rules.length > 0) && (
        <AccordionSection
          id="learnings"
          title="Learnings"
          count={learnings.length + rules.length}
          icon={<AcademicCapIcon className="w-4 h-4" />}
          command="/learnings"
          action={
            learnings.length >= 2 ? (
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
            ) : (
              <div className="w-6 h-6" />
            )
          }
        >
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
                    className="p-2 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg"
                  >
                    <p className="text-sm text-gray-700 dark:text-gray-300">
                      {rule.summary}
                    </p>
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
                    className="p-2 bg-gray-50 dark:bg-gray-800/50 rounded-lg"
                  >
                    <p className="text-sm text-gray-700 dark:text-gray-300">
                      {learning.content}
                    </p>
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
        </AccordionSection>
      )}
    </div>
  )
}