// Table Accordion - individual table with expandable details and fullscreen mode

import { useState, useEffect } from 'react'
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
import type { TableData, TableInfo } from '@/types/api'

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

  // Fetch data when opened
  useEffect(() => {
    if (!session || !isOpen) return

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
  }, [session, table.name, page, isOpen])

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
                      {String(row[col] ?? '')}
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

  return (
    <>
      {/* Accordion Item */}
      <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
        {/* Header */}
        <button
          onClick={toggleOpen}
          className="w-full flex items-center justify-between px-3 py-2 bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
        >
          <div className="flex items-center gap-2">
            {isOpen ? (
              <ChevronDownIcon className="w-4 h-4 text-gray-500" />
            ) : (
              <ChevronRightIcon className="w-4 h-4 text-gray-500" />
            )}
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              {table.name}
            </span>
            <span className="text-xs text-gray-400 dark:text-gray-500">
              ({table.row_count} rows)
            </span>
          </div>
          <div className="flex items-center gap-1">
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
          <div className="px-3 py-2 bg-white dark:bg-gray-900">
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