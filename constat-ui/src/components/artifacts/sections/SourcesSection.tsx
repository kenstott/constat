// Copyright (c) 2025 Kenneth Stott
// Canary: c7d8aec6-3a7c-4f96-84f1-7253d1cf2ce4
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import React, { useState, useCallback, useEffect } from 'react'
import { useReactiveVar } from '@apollo/client'
import { activeDeepLinkVar, consumeDeepLink, expandSection } from '@/graphql/ui-state'
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
  PencilIcon,
  PencilSquareIcon,
  UserPlusIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline'
import { AccordionSection } from '../ArtifactAccordion'
import { SkeletonLoader } from '../../common/SkeletonLoader'
import { DomainBadge } from '../../common/DomainBadge'
import { DocumentViewerModal } from '../DocumentViewerModal'
import { apolloClient } from '@/graphql/client'
import { DATABASE_SCHEMA_QUERY, API_SCHEMA_QUERY, toDatabaseTable, toApiEndpoint } from '@/graphql/operations/state'
import { FORGET_FACT, PERSIST_FACT, MOVE_FACT, EDIT_FACT } from '@/graphql/operations/data'
import { REMOVE_API, REMOVE_DATABASE, DELETE_FILE_REF, DATABASE_TABLE_PREVIEW_QUERY, toDatabaseTablePreview } from '@/graphql/operations/sources'
import { EditDatabaseModal } from '../EditDatabaseModal'
import { EditApiModal } from '../EditApiModal'
import { EditDocumentModal } from '../EditDocumentModal'
import { useDataSources } from '@/hooks/useDataSources'
import { useFacts } from '@/hooks/useFacts'
import { useSessionContext } from '@/contexts/SessionContext'
import { getDocument } from '@/api/sessions'
import type { DatabaseTableInfo, DatabaseTablePreview, ApiEndpointInfo, SessionDatabase, ApiSourceInfo, DocumentSourceInfo } from '@/types/api'
import { toFact } from '@/graphql/operations/data'
import type { ViewingDocument } from '../DocumentViewerModal'
import { ApiEndpointPanel } from '../ApiEndpointPanel'

interface SourcesSectionProps {
  sessionId: string
  sourcesVisible: boolean
  canSeeSection: (section: string) => boolean
  canWrite: (section: string) => boolean
  onOpenModal: (type: 'database' | 'api' | 'document' | 'email' | 'fact' | 'personal' | 'accounts') => void
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

  // Sources always visible when parent Context group is open

  // Database expand/preview state
  const [expandedDb, setExpandedDb] = useState<string | null>(null)
  const [dbTables, setDbTables] = useState<Record<string, DatabaseTableInfo[]>>({})
  const [dbTablesLoading, setDbTablesLoading] = useState<string | null>(null)
  const [previewDb, setPreviewDb] = useState<string | null>(null)
  const [previewTable, setPreviewTable] = useState<string | null>(null)
  const [previewData, setPreviewData] = useState<DatabaseTablePreview | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewPage, setPreviewPage] = useState(1)

  // Document source picker dropdown
  const [docPickerOpen, setDocPickerOpen] = useState(false)
  useEffect(() => {
    if (!docPickerOpen) return
    const close = () => setDocPickerOpen(false)
    document.addEventListener('click', close)
    return () => document.removeEventListener('click', close)
  }, [docPickerOpen])

  // API expand state
  const [expandedApi, setExpandedApi] = useState<string | null>(null)
  const [apiEndpoints, setApiEndpoints] = useState<Record<string, ApiEndpointInfo[]>>({})
  const [apiEndpointsLoading, setApiEndpointsLoading] = useState<string | null>(null)
  const [expandedEndpoint, setExpandedEndpoint] = useState<string | null>(null)

  // Document viewer state
  const [viewingDocument, setViewingDocument] = useState<ViewingDocument | null>(null)
  const [loadingDocument, setLoadingDocument] = useState(false)

  // Fact inline edit state
  const [editingFact, setEditingFact] = useState<string | null>(null)
  const [editingFactValue, setEditingFactValue] = useState('')

  // Edit source modal state
  const [editingDb, setEditingDb] = useState<SessionDatabase | null>(null)
  const [editingApi, setEditingApi] = useState<ApiSourceInfo | null>(null)
  const [editingDoc, setEditingDoc] = useState<DocumentSourceInfo | null>(null)

  // Deep link handling — expand the target source, scroll, consume
  const pendingDeepLink = useReactiveVar(activeDeepLinkVar)
  useEffect(() => {
    if (!pendingDeepLink) return
    const link = pendingDeepLink
    console.log('[deep-link] SourcesSection received:', link.type, link)

    if (link.type !== 'table' && link.type !== 'api' && link.type !== 'document') {
      return
    }

    consumeDeepLink()

    // Expand the accordion section
    const accordionId = link.type === 'api' ? 'apis' : link.type === 'table' ? 'databases' : 'documents'
    expandSection(accordionId)

    if (link.type === 'api' && link.apiName) {
      console.log('[deep-link] expanding api:', link.apiName, 'endpoint:', link.apiEndpoint)
      toggleApiExpand(link.apiName).then(() => {
        if (link.apiEndpoint) {
          setExpandedEndpoint(link.apiEndpoint)
        }
      })
    } else if (link.type === 'table' && link.dbName) {
      console.log('[deep-link] expanding db:', link.dbName, 'table:', link.tableName)
      toggleDbExpand(link.dbName).then(() => {
        if (link.tableName) {
          openTablePreview(link.dbName!, link.tableName)
        }
      })
    } else if (link.type === 'document' && link.documentName) {
      console.log('[deep-link] opening document viewer:', link.documentName)
      // Open document viewer modal — inline to avoid forward-ref to handleViewDocument
      if (session) {
        setLoadingDocument(true)
        getDocument(session.session_id, link.documentName)
          .then(doc => {
            const httpUrl = doc.url && /^https?:\/\//i.test(doc.url) ? doc.url : undefined
            setViewingDocument({
              name: doc.name || link.documentName!,
              content: doc.content || '',
              format: doc.format,
              url: httpUrl,
            })
          })
          .catch(err => {
            setViewingDocument({
              name: link.documentName!,
              content: `Document not found or could not be loaded.\n\n${err instanceof Error ? err.message : String(err)}`,
              format: 'text',
            })
          })
          .finally(() => setLoadingDocument(false))
      }
    }

    // Scroll after accordion + content renders
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const el = document.getElementById(`section-${accordionId}`)
        console.log('[deep-link] scroll target:', `section-${accordionId}`, el ? 'found' : 'NOT FOUND')
        el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      })
    })
  }, [pendingDeepLink, apis, databases, expandedApi, expandedDb, session])

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

  const handleEditFact = useCallback(async (factName: string, value: string) => {
    if (!session) return
    await apolloClient.mutate({ mutation: EDIT_FACT, variables: { sessionId: session.session_id, factName, value } })
    setEditingFact(null)
    setEditingFactValue('')
    apolloClient.refetchQueries({ include: ['Facts'] })
  }, [session])

  return (
    <>
      {/* Databases */}
      {sourcesVisible && (
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
                    {db.is_dynamic && canWrite('sources') && (
                      <>
                        <button
                          onClick={() => setEditingDb(db)}
                          className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 transition-all"
                          title="Edit database"
                        >
                          <PencilSquareIcon className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={async () => {
                            if (!session) return
                            if (!confirm(`Remove database "${db.name}" from this session?`)) return
                            try {
                              await apolloClient.mutate({ mutation: REMOVE_DATABASE, variables: { sessionId: session.session_id, name: db.name } })
                              await apolloClient.refetchQueries({ include: ['DataSources'] })
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
                      </>
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
                    {api.source === 'session' && canWrite('sources') && (
                      <>
                        <button
                          onClick={() => setEditingApi(api)}
                          className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 transition-all"
                          title="Edit API"
                        >
                          <PencilSquareIcon className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => handleDeleteApi(api.name)}
                          className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 transition-all"
                          title="Remove API"
                        >
                          <TrashIcon className="w-3.5 h-3.5" />
                        </button>
                      </>
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
                    ) : (
                      <ApiEndpointPanel
                        endpoints={apiEndpoints[api.name] || []}
                        expandedEndpoint={expandedEndpoint}
                        setExpandedEndpoint={setExpandedEndpoint}
                      />
                    )}
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
            <div className="relative">
              <button
                onClick={(e) => { e.stopPropagation(); setDocPickerOpen((v) => !v) }}
                className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                title="Add source"
              >
                <PlusIcon className="w-4 h-4" />
              </button>
              {docPickerOpen && (
                <div
                  className="absolute right-0 top-full mt-1 z-50 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded shadow-lg min-w-[160px] py-1"
                  onClick={(e) => e.stopPropagation()}
                >
                  {[
                    { label: 'File / URI', icon: <DocumentTextIcon className="w-4 h-4" />, type: 'document' as const },
                    { label: 'Personal resource', icon: <UserPlusIcon className="w-4 h-4" />, type: 'personal' as const },
                    { label: 'Manage accounts', icon: <Cog6ToothIcon className="w-4 h-4" />, type: 'accounts' as const },
                  ].map(({ label, icon, type }) => (
                    <button
                      key={type}
                      onClick={() => { setDocPickerOpen(false); onOpenModal(type) }}
                      className="flex items-center gap-2 w-full px-3 py-1.5 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                    >
                      {icon}{label}
                    </button>
                  ))}
                </div>
              )}
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
                    {!doc.from_config && canWrite('sources') && (
                      <>
                        <button
                          onClick={() => setEditingDoc(doc)}
                          className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 transition-all"
                          title="Edit document"
                        >
                          <PencilSquareIcon className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => handleDeleteDocument(doc.name)}
                          className="opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 transition-all"
                          title="Remove document"
                        >
                          <TrashIcon className="w-3.5 h-3.5" />
                        </button>
                      </>
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
            {canWrite('sources') && (
              <button
                onClick={() => onOpenModal('fact')}
                className="p-1 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                title="Add fact"
              >
                <PlusIcon className="w-4 h-4" />
              </button>
            )}
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
                    <td className="py-2 px-1 text-gray-600 dark:text-gray-400">
                      {editingFact === fact.name ? (
                        <span className="flex items-center gap-1">
                          <input
                            autoFocus
                            className="text-xs border border-gray-300 dark:border-gray-600 rounded px-1.5 py-0.5 bg-white dark:bg-gray-800 w-32"
                            value={editingFactValue}
                            onChange={(e) => setEditingFactValue(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') handleEditFact(fact.name, editingFactValue)
                              if (e.key === 'Escape') { setEditingFact(null); setEditingFactValue('') }
                            }}
                          />
                          <button onClick={() => handleEditFact(fact.name, editingFactValue)} className="text-[10px] text-primary-600 hover:underline">Save</button>
                          <button onClick={() => { setEditingFact(null); setEditingFactValue('') }} className="text-[10px] text-gray-400 hover:underline">Cancel</button>
                        </span>
                      ) : (
                        String(fact.value)
                      )}
                    </td>
                    <td className="py-2 px-1 text-xs text-gray-400 dark:text-gray-500">
                      <DomainBadge domain={fact.source === 'config' ? 'system' : fact.source} />
                    </td>
                    <td className="py-2 px-1 flex items-center gap-1">
                      {fact.source !== 'config' && canWrite('sources') && (
                        <>
                          <button
                            onClick={() => { setEditingFact(fact.name); setEditingFactValue(String(fact.value)) }}
                            className="p-1 text-gray-300 dark:text-gray-600 hover:text-primary-600 dark:hover:text-primary-400 opacity-0 group-hover:opacity-100 transition-opacity"
                            title="Edit value"
                          >
                            <PencilIcon className="w-3 h-3" />
                          </button>
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

      <DocumentViewerModal
        viewingDocument={viewingDocument}
        loadingDocument={loadingDocument}
        onClose={() => setViewingDocument(null)}
      />

      {editingDb && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 w-80 shadow-xl max-h-[90vh] overflow-y-auto">
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">Edit Database</h3>
            <EditDatabaseModal
              db={editingDb}
              onSuccess={() => { setEditingDb(null); apolloClient.refetchQueries({ include: ['DataSources'] }) }}
              onCancel={() => setEditingDb(null)}
            />
          </div>
        </div>
      )}

      {editingApi && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 w-80 shadow-xl max-h-[90vh] overflow-y-auto">
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">Edit API</h3>
            <EditApiModal
              api={editingApi}
              onSuccess={() => { setEditingApi(null); apolloClient.refetchQueries({ include: ['DataSources'] }) }}
              onCancel={() => setEditingApi(null)}
            />
          </div>
        </div>
      )}

      {editingDoc && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-4 w-80 shadow-xl max-h-[90vh] overflow-y-auto">
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-3">Edit Document</h3>
            <EditDocumentModal
              doc={editingDoc}
              onSuccess={() => { setEditingDoc(null); apolloClient.refetchQueries({ include: ['DataSources'] }) }}
              onCancel={() => setEditingDoc(null)}
            />
          </div>
        </div>
      )}
    </>
  )
}
