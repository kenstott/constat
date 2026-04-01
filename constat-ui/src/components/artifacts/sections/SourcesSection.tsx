// Copyright (c) 2025 Kenneth Stott
// Canary: c7d8aec6-3a7c-4f96-84f1-7253d1cf2ce4
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
import {
  CircleStackIcon,
  GlobeAltIcon,
  DocumentTextIcon,
  LightBulbIcon,
  PlusIcon,
  MinusIcon,
  TrashIcon,
  ArrowsRightLeftIcon,
  ArrowUpTrayIcon,
  ArrowDownTrayIcon,
  ChevronRightIcon,
  ChevronDownIcon,
  EnvelopeIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline'
import { AccordionSection } from '../ArtifactAccordion'
import { SkeletonLoader } from '../../common/SkeletonLoader'
import { DomainBadge } from '../../common/DomainBadge'
import { apolloClient } from '@/graphql/client'
import { DATABASE_SCHEMA_QUERY, API_SCHEMA_QUERY, toDatabaseTable, toApiEndpoint } from '@/graphql/operations/state'
import { FORGET_FACT, PERSIST_FACT, MOVE_FACT } from '@/graphql/operations/data'
import { REMOVE_API, REMOVE_DATABASE, DELETE_FILE_REF, DATABASE_TABLE_PREVIEW_QUERY, toDatabaseTablePreview } from '@/graphql/operations/sources'
import { useDataSources } from '@/hooks/useDataSources'
import { useFacts } from '@/hooks/useFacts'
import { useSessionContext } from '@/contexts/SessionContext'
import { getDocument } from '@/api/sessions'
import type { DatabaseTableInfo, DatabaseTablePreview, ApiEndpointInfo, SessionDatabase, ApiSourceInfo, DocumentSourceInfo } from '@/types/api'
import { toFact } from '@/graphql/operations/data'

interface SourcesSectionProps {
  sessionId: string
  sourcesVisible: boolean
  canSeeSection: (section: string) => boolean
  canWrite: (section: string) => boolean
  onOpenModal: (type: 'database' | 'api' | 'document' | 'email' | 'fact') => void
  ingestingSource: string | null
  ingestProgress: { current: number; total: number } | null
  domainList: { filename: string; name: string }[]
  movingFact: string | null
  setMovingFact: (name: string | null) => void
}

const FILE_DB_TYPES = new Set(['csv', 'json', 'jsonl', 'parquet', 'arrow', 'feather', 'tsv'])

export const SourcesSection: React.FC<SourcesSectionProps> = ({
  sourcesVisible,
  canSeeSection,
  canWrite,
  onOpenModal,
  ingestingSource,
  ingestProgress,
  domainList,
  movingFact,
  setMovingFact,
}) => {
  const { session } = useSessionContext()
  const { databases, apis, documents, loading: sourcesLoading } = useDataSources()
  const { facts, loading: factsLoading } = useFacts()

  // Collapsible state - persisted in localStorage
  const [sourcesCollapsed, setSourcesCollapsed] = useState(() => {
    return localStorage.getItem('constat-sources-collapsed') === 'true'
  })

  // Database expand/preview state
  const [expandedDb, setExpandedDb] = useState<string | null>(null)
  const [dbTables, setDbTables] = useState<Record<string, DatabaseTableInfo[]>>({})
  const [dbTablesLoading, setDbTablesLoading] = useState<string | null>(null)
  const [previewDb, setPreviewDb] = useState<string | null>(null)
  const [previewTable, setPreviewTable] = useState<string | null>(null)
  const [previewData, setPreviewData] = useState<DatabaseTablePreview | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewPage, setPreviewPage] = useState(1)

  // API expand state
  const [expandedApi, setExpandedApi] = useState<string | null>(null)
  const [apiEndpoints, setApiEndpoints] = useState<Record<string, ApiEndpointInfo[]>>({})
  const [apiEndpointsLoading, setApiEndpointsLoading] = useState<string | null>(null)
  const [expandedEndpoint, setExpandedEndpoint] = useState<string | null>(null)

  // Document viewer state
  const [viewingDocument, setViewingDocument] = useState<{ name: string; content: string; format?: string; url?: string; imageUrl?: string } | null>(null)
  const [iframeBlocked, setIframeBlocked] = useState(false)
  const [loadingDocument, setLoadingDocument] = useState(false)

  // --- Handlers ---

  const openTablePreview = useCallback(async (dbName: string, tableName: string, page = 1) => {
    if (!session) return
    setPreviewDb(dbName)
    setPreviewTable(tableName)
    setPreviewPage(page)
    setPreviewLoading(true)
    try {
      const { data: gqlData } = await apolloClient.query({ query: DATABASE_TABLE_PREVIEW_QUERY, variables: { sessionId: session.session_id, dbName, tableName, page }, fetchPolicy: 'network-only' })
      setPreviewData(toDatabaseTablePreview(gqlData.databaseTablePreview))
    } catch (err) {
      console.error('Failed to preview table:', err)
      setPreviewData(null)
    } finally {
      setPreviewLoading(false)
    }
  }, [session])

  const toggleDbExpand = useCallback(async (dbName: string, dbType?: string) => {
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
        const { data: dbData } = await apolloClient.query({ query: DATABASE_SCHEMA_QUERY, variables: { sessionId: session.session_id, dbName }, fetchPolicy: 'network-only' })
        const tables = dbData.databaseSchema.tables.map(toDatabaseTable)
        setDbTables((prev) => ({ ...prev, [dbName]: tables }))
        // File-based DBs are single-table — jump straight to preview
        if (dbType && FILE_DB_TYPES.has(dbType) && tables.length === 1) {
          openTablePreview(dbName, tables[0].name)
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
  }, [expandedDb, dbTables, session, openTablePreview])

  const toggleApiExpand = useCallback(async (apiName: string) => {
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
        const { data: apiData } = await apolloClient.query({ query: API_SCHEMA_QUERY, variables: { sessionId: session.session_id, apiName }, fetchPolicy: 'network-only' })
        setApiEndpoints((prev) => ({ ...prev, [apiName]: apiData.apiSchema.endpoints.map(toApiEndpoint) }))
      } catch (err) {
        console.error('Failed to load API schema:', err)
        setApiEndpoints((prev) => ({ ...prev, [apiName]: [] }))
      } finally {
        setApiEndpointsLoading(null)
      }
    }
  }, [expandedApi, apiEndpoints, session])

  const handleDeleteApi = useCallback(async (apiName: string) => {
    if (!session) return
    if (!confirm(`Remove API "${apiName}" from this session?`)) return

    try {
      await apolloClient.mutate({ mutation: REMOVE_API, variables: { sessionId: session.session_id, name: apiName } })
      apolloClient.refetchQueries({ include: ['DataSources'] })
      // Entities refresh via entity_rebuild_complete WS event
    } catch (err) {
      console.error('Failed to remove API:', err)
      alert('Failed to remove API. Please try again.')
    }
  }, [session])

  const handleDeleteDocument = useCallback(async (docName: string) => {
    if (!session) return
    if (!confirm(`Delete document "${docName}" and its extracted entities?`)) return

    try {
      await apolloClient.mutate({ mutation: DELETE_FILE_REF, variables: { sessionId: session.session_id, name: docName } })
      apolloClient.refetchQueries({ include: ['DataSources'] })
      // Entities refresh via entity_rebuild_complete WS event
    } catch (err) {
      console.error('Failed to delete document:', err)
      alert('Failed to delete document. Please try again.')
    }
  }, [session])

  const handleViewDocument = useCallback(async (documentName: string) => {
    if (!session) return
    setLoadingDocument(true)
    try {
      const doc = await getDocument(session.session_id, documentName)

      // For file types (PDF, Office docs), open via file serving endpoint
      if (doc.type === 'file' && doc.path) {
        // Open file in new tab via file serving endpoint
        const fileUrl = `/api/sessions/${session.session_id}/file?path=${encodeURIComponent(doc.path)}`
        window.open(fileUrl, '_blank')
        return
      }

      // For content types (markdown, text), show in modal
      setIframeBlocked(false)
      const imageUrl = doc.image_path && session
        ? `/api/sessions/${session.session_id}/file?path=${encodeURIComponent(doc.image_path)}`
        : undefined
      // Only pass HTTP/HTTPS URLs for iframe rendering — skip imap://, file://, etc.
      const httpUrl = doc.url && /^https?:\/\//i.test(doc.url) ? doc.url : undefined
      setViewingDocument({
        name: doc.name || documentName,
        content: doc.content || '',
        format: doc.format,
        url: httpUrl,
        imageUrl,
      })
    } catch (err) {
      console.error('Failed to load document:', err)
      // Show error in modal instead of alert so user sees context
      setViewingDocument({
        name: documentName,
        content: `Document not found or could not be loaded.\n\n${err instanceof Error ? err.message : String(err)}`,
        format: 'text',
      })
    } finally {
      setLoadingDocument(false)
    }
  }, [session])

  const handleForgetFact = useCallback(async (factName: string) => {
    if (!session) return
    await apolloClient.mutate({ mutation: FORGET_FACT, variables: { sessionId: session.session_id, factName } })
    apolloClient.refetchQueries({ include: ['Facts'] })
  }, [session])

  const handlePersistFact = useCallback(async (factName: string) => {
    if (!session) return
    await apolloClient.mutate({ mutation: PERSIST_FACT, variables: { sessionId: session.session_id, factName } })
    apolloClient.refetchQueries({ include: ['Facts'] })
  }, [session])

  const handleMoveFact = useCallback(async (factName: string, toDomain: string) => {
    if (!session) return
    await apolloClient.mutate({ mutation: MOVE_FACT, variables: { sessionId: session.session_id, factName, toDomain } })
    setMovingFact(null)
    apolloClient.refetchQueries({ include: ['Facts'] })
  }, [session, setMovingFact])

  return (
    <>
      {/* ═══════════════ SOURCES & TOOLS ═══════════════ */}
      {sourcesVisible && (
      <button
        onClick={() => {
          const newVal = !sourcesCollapsed
          setSourcesCollapsed(newVal)
          localStorage.setItem('constat-sources-collapsed', String(newVal))
        }}
        className="w-full px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between hover:bg-gray-150 dark:hover:bg-gray-750 transition-colors"
      >
        <span className="text-[10px] font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
          Sources & Tools ({databases.length + apis.length + documents.length + facts.length})
          {(sourcesLoading || factsLoading) && <span className="inline-block w-2.5 h-2.5 border-[1.5px] border-gray-400 border-t-transparent rounded-full animate-spin" />}
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
              onClick={() => onOpenModal('database')}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Add database"
            >
              <PlusIcon className="w-4 h-4" />
            </button>
          ) : <div className="w-6 h-6" />
        }
      >
        {databases.length === 0 ? (
          sourcesLoading ? <SkeletonLoader lines={2} /> :
          <p className="text-sm text-gray-500 dark:text-gray-400">No databases configured</p>
        ) : (
          <div className="space-y-2">
            {databases.map((db: SessionDatabase) => (
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
                            await apolloClient.mutate({ mutation: REMOVE_DATABASE, variables: { sessionId: session.session_id, name: db.name } })
                            await apolloClient.refetchQueries({ include: ['DataSources'] })
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
              onClick={() => onOpenModal('api')}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Add API"
          >
            <PlusIcon className="w-4 h-4" />
          </button>
          ) : <div className="w-6 h-6" />
        }
      >
        {apis.length === 0 ? (
          sourcesLoading ? <SkeletonLoader lines={2} /> :
          <p className="text-sm text-gray-500 dark:text-gray-400">No APIs configured</p>
        ) : (
          <div className="space-y-2">
            {apis.map((api: ApiSourceInfo) => (
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

                      const renderEndpoint = (ep: ApiEndpointInfo) => (
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

                      const renderSection = (label: string, items: ApiEndpointInfo[]) => (
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
            <div className="flex items-center gap-0.5">
              <button
                onClick={() => onOpenModal('document')}
                className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                title="Add document"
              >
                <PlusIcon className="w-4 h-4" />
              </button>
              <button
                onClick={() => onOpenModal('email')}
                className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                title="Add email source"
              >
                <EnvelopeIcon className="w-4 h-4" />
              </button>
            </div>
          ) : <div className="w-6 h-6" />
        }
      >
        {ingestingSource && !documents.some((d: DocumentSourceInfo) => d.name === ingestingSource) && (
          <div className="flex items-center gap-2 text-xs text-blue-600 dark:text-blue-400 mb-2">
            <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <span>Indexing {ingestingSource}{ingestProgress ? ` (${ingestProgress.current}/${ingestProgress.total})` : '...'}</span>
          </div>
        )}
        {documents.length === 0 ? (
          sourcesLoading ? <SkeletonLoader lines={2} /> :
          <p className="text-sm text-gray-500 dark:text-gray-400">No documents indexed</p>
        ) : (
          <div className="space-y-2">
            {documents.map((doc: DocumentSourceInfo) => (
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
                    ...facts.map((f: ReturnType<typeof toFact>) => [
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
              onClick={() => onOpenModal('fact')}
              className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
              title="Add fact"
            >
              <PlusIcon className="w-4 h-4" />
            </button>
          </div>
        }
      >
        {facts.length === 0 ? (
          factsLoading ? <SkeletonLoader lines={2} /> :
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
                {facts.map((fact: ReturnType<typeof toFact>) => (
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
                <div>
                  {viewingDocument.imageUrl && (
                    <div className="mb-4 flex justify-center">
                      <img
                        src={viewingDocument.imageUrl}
                        alt={viewingDocument.name}
                        className="max-w-full max-h-[40vh] object-contain rounded border border-gray-200 dark:border-gray-700"
                      />
                    </div>
                  )}
                  {viewingDocument.format === 'markdown' ? (
                    <div className="prose prose-sm dark:prose-invert max-w-none">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {viewingDocument.content}
                      </ReactMarkdown>
                    </div>
                  ) : (
                    <pre className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap font-mono">
                      {viewingDocument.content}
                    </pre>
                  )}
                </div>
              ) : (
                <p className="text-sm text-gray-500 dark:text-gray-400">No content available</p>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  )
}
