// Table Accordion - individual table with expandable details and fullscreen mode

import { useState, useEffect, useRef } from 'react'
import {
  ChevronDownIcon,
  ChevronRightIcon,
  ArrowsPointingOutIcon,
  ArrowDownTrayIcon,
  XMarkIcon,
  StarIcon as StarOutline,
} from '@heroicons/react/24/outline'
import { StarIcon as StarSolid } from '@heroicons/react/24/solid'
import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import * as sessionsApi from '@/api/sessions'
import type { TableData, TableInfo, TableVersionInfo } from '@/types/api'

interface TableAccordionProps {
  table: TableInfo
  initiallyOpen?: boolean
}

export function TableAccordion({ table, initiallyOpen = false }: TableAccordionProps) {
  const { session } = useSessionStore()
  const { toggleTableStar } = useArtifactStore()
  const [isOpen, setIsOpen] = useState(initiallyOpen)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [data, setData] = useState<TableData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(1)
  const [showVersions, setShowVersions] = useState(false)
  const [versions, setVersions] = useState<TableVersionInfo[] | null>(null)
  const [viewingVersion, setViewingVersion] = useState<number | null>(null)
  const versionDropdownRef = useRef<HTMLDivElement>(null)

  const hasVersions = (table.version_count ?? 1) > 1

  // Fetch data when opened
  useEffect(() => {
    if (!session || !isOpen) return
    // Skip if viewing a specific older version (content loaded by handleSelectVersion)
    if (viewingVersion) return

    const fetchData = async () => {
      setLoading(true)
      setError(null)
      try {
        const tableData = await sessionsApi.getTableData(
          session.session_id,
          table.name,
          page
        )
        setData(tableData)
      } catch (err) {
        setError(String(err))
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [session, table.name, page, isOpen, viewingVersion])

  // Close fullscreen on Escape
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isFullscreen) {
        setIsFullscreen(false)
      }
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [isFullscreen])

  // Close version dropdown on outside click
  useEffect(() => {
    if (!showVersions) return
    const handleClick = (e: MouseEvent) => {
      if (versionDropdownRef.current && !versionDropdownRef.current.contains(e.target as Node)) {
        setShowVersions(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [showVersions])

  const toggleOpen = () => {
    setIsOpen(!isOpen)
  }

  const openFullscreen = (e: React.MouseEvent) => {
    e.stopPropagation()
    setIsOpen(true) // Ensure data is loaded
    setIsFullscreen(true)
  }

  const handleToggleStar = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (session) {
      toggleTableStar(session.session_id, table.name)
    }
  }

  const handleVersionBadgeClick = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!session || !hasVersions) return

    if (showVersions) {
      setShowVersions(false)
      return
    }

    // Fetch versions if not loaded
    if (!versions) {
      try {
        const resp = await sessionsApi.getTableVersions(session.session_id, table.name)
        setVersions(resp.versions)
      } catch (err) {
        console.error('Failed to load table versions:', err)
        return
      }
    }
    setShowVersions(true)
  }

  const handleSelectVersion = async (version: number) => {
    if (!session) return
    setShowVersions(false)

    const currentVersion = table.version ?? 1
    if (version === currentVersion) {
      // Back to current version
      setViewingVersion(null)
      setData(null)
      return
    }

    setViewingVersion(version)
    setLoading(true)
    setError(null)
    try {
      const tableData = await sessionsApi.getTableVersionData(
        session.session_id,
        table.name,
        version,
        1 // reset to page 1
      )
      setData(tableData)
      setPage(1)
      setIsOpen(true)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (!session) return

    try {
      const response = await fetch(
        `/api/sessions/${session.session_id}/tables/${table.name}/download`
      )
      if (!response.ok) throw new Error('Failed to download')
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${table.name}.csv`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('Download failed:', err)
      alert('Failed to download. Please try again.')
    }
  }

  const renderTable = (maxHeight?: string) => {
    if (loading) {
      return (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-500" />
        </div>
      )
    }

    if (error) {
      return (
        <div className="text-sm text-red-500 dark:text-red-400 py-4">
          Error: {error}
        </div>
      )
    }

    if (!data || data.data.length === 0) {
      return (
        <div className="text-sm text-gray-500 dark:text-gray-400 py-4">
          No data available
        </div>
      )
    }

    return (
      <div className="space-y-3">
        {/* Table with scrollable container */}
        <div
          className="overflow-auto"
          style={{ maxHeight: maxHeight || '200px' }}
        >
          <table className="min-w-full text-sm">
            <thead className="sticky top-0 bg-white dark:bg-gray-900">
              <tr className="border-b border-gray-200 dark:border-gray-700">
                {data.columns.map((col) => (
                  <th
                    key={col}
                    className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider whitespace-nowrap"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
              {data.data.map((row, i) => (
                <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                  {data.columns.map((col) => (
                    <td
                      key={col}
                      className="px-3 py-2 text-gray-700 dark:text-gray-300 whitespace-nowrap"
                    >
                      {row[col] != null && typeof row[col] === 'object' ? JSON.stringify(row[col]) : String(row[col] ?? '')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        {(data.has_more || page > 1) && (
          <div className="flex items-center justify-between text-sm px-1">
            <span className="text-gray-500 dark:text-gray-400">
              Page {page} of {Math.ceil(data.total_rows / data.page_size)}
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="btn-ghost text-xs disabled:opacity-50"
              >
                Previous
              </button>
              <button
                onClick={() => setPage((p) => p + 1)}
                disabled={!data.has_more}
                className="btn-ghost text-xs disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    )
  }

  const currentVersion = table.version ?? 1

  return (
    <>
      {/* Accordion Item */}
      <div className="border border-gray-200 dark:border-gray-700 rounded-lg">
        {/* Header - overflow-visible to allow version dropdown to escape */}
        <button
          onClick={toggleOpen}
          className="relative w-full flex items-center justify-between px-3 py-2 bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors rounded-t-lg"
        >
          <div className="flex items-center gap-2 min-w-0">
            {isOpen ? (
              <ChevronDownIcon className="w-4 h-4 text-gray-500 flex-shrink-0" />
            ) : (
              <ChevronRightIcon className="w-4 h-4 text-gray-500 flex-shrink-0" />
            )}
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300 truncate">
              {table.name}
            </span>
            <span className="text-xs text-gray-400 dark:text-gray-500 flex-shrink-0">
              ({table.row_count} rows)
            </span>
            <div className="relative flex-shrink-0" ref={versionDropdownRef}>
              <button
                onClick={hasVersions ? handleVersionBadgeClick : undefined}
                className={`px-1.5 py-0.5 text-[10px] font-medium rounded transition-colors ${
                  hasVersions
                    ? 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400 hover:bg-indigo-200 dark:hover:bg-indigo-800/40 cursor-pointer'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-default'
                }`}
                title={hasVersions
                  ? `Version ${currentVersion} of ${table.version_count ?? 1} — click to browse`
                  : `Version ${currentVersion}`
                }
              >
                v{viewingVersion ?? currentVersion}
              </button>
              {showVersions && versions && (
                <div className="absolute top-full left-0 mt-1 z-50 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg py-1 min-w-[200px]">
                  {versions.map((v) => (
                    <button
                      key={v.version}
                      onClick={(e) => { e.stopPropagation(); handleSelectVersion(v.version) }}
                      className={`w-full text-left px-3 py-1.5 text-xs hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex justify-between items-center gap-2 ${
                        (viewingVersion === v.version || (!viewingVersion && v.version === currentVersion))
                          ? 'bg-indigo-50 dark:bg-indigo-900/20 text-indigo-700 dark:text-indigo-400'
                          : 'text-gray-700 dark:text-gray-300'
                      }`}
                    >
                      <span>v{v.version}</span>
                      <span className="text-gray-400 dark:text-gray-500">
                        {v.row_count} rows{v.step_number != null ? ` · step ${v.step_number}` : ''}
                      </span>
                    </button>
                  ))}
                </div>
              )}
            </div>
            {viewingVersion && (
              <span className="px-1 py-0.5 text-[10px] bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 rounded flex-shrink-0">
                viewing older version
              </span>
            )}
            {table.role_id && (
              <span className="px-1 py-0.5 text-[10px] bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded flex-shrink-0">
                {table.role_id}
              </span>
            )}
          </div>
          <div className="flex items-center gap-1 flex-shrink-0">
            <button
              onClick={handleToggleStar}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded transition-colors"
              title={table.is_starred ? "Unstar" : "Star"}
            >
              {table.is_starred ? (
                <StarSolid className="w-4 h-4 text-yellow-500" />
              ) : (
                <StarOutline className="w-4 h-4 text-gray-400 hover:text-yellow-500" />
              )}
            </button>
            <button
              onClick={handleDownload}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded transition-colors"
              title="Download as CSV"
            >
              <ArrowDownTrayIcon className="w-4 h-4 text-gray-500" />
            </button>
            <button
              onClick={openFullscreen}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded transition-colors"
              title="Expand to fullscreen"
            >
              <ArrowsPointingOutIcon className="w-4 h-4 text-gray-500" />
            </button>
          </div>
        </button>

        {/* Collapsible Content */}
        {isOpen && (
          <div className="px-3 py-2 bg-white dark:bg-gray-900 overflow-hidden rounded-b-lg">
            {renderTable('200px')}
          </div>
        )}
      </div>

      {/* Fullscreen Modal */}
      {isFullscreen && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-gray-900 rounded-lg shadow-xl w-full h-full max-w-[95vw] max-h-[95vh] flex flex-col">
            {/* Modal Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center gap-2">
                <span className="text-lg font-semibold text-gray-800 dark:text-gray-200">
                  {table.name}
                </span>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  ({table.row_count} rows)
                </span>
              </div>
              <button
                onClick={() => setIsFullscreen(false)}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
                title="Close (Esc)"
              >
                <XMarkIcon className="w-5 h-5 text-gray-500" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="flex-1 overflow-hidden p-4">
              {renderTable('calc(95vh - 120px)')}
            </div>
          </div>
        </div>
      )}
    </>
  )
}
