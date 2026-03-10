// Table Viewer component

import { useState, useEffect, useCallback } from 'react'
import { ClipboardDocumentIcon, ClipboardDocumentCheckIcon } from '@heroicons/react/24/outline'
import { useSessionStore } from '@/store/sessionStore'
import * as sessionsApi from '@/api/sessions'
import type { TableData } from '@/types/api'

interface TableViewerProps {
  tableName: string
}

export function TableViewer({ tableName }: TableViewerProps) {
  const { session } = useSessionStore()
  const [data, setData] = useState<TableData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [page, setPage] = useState(1)

  useEffect(() => {
    if (!session) return

    const fetchData = async () => {
      setLoading(true)
      setError(null)
      try {
        const tableData = await sessionsApi.getTableData(
          session.session_id,
          tableName,
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
  }, [session, tableName, page])

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
        Error loading table: {error}
      </div>
    )
  }

  const [copied, setCopied] = useState<'csv' | 'json' | null>(null)
  const [showCopyMenu, setShowCopyMenu] = useState(false)

  const handleCopy = useCallback((format: 'csv' | 'json') => {
    if (!data) return
    let text: string
    if (format === 'csv') {
      const header = data.columns.join(',')
      const rows = data.data.map(row =>
        data.columns.map(col => {
          const v = row[col]
          const s = v != null && typeof v === 'object' ? JSON.stringify(v) : String(v ?? '')
          return s.includes(',') || s.includes('"') || s.includes('\n')
            ? `"${s.replace(/"/g, '""')}"`
            : s
        }).join(',')
      )
      text = [header, ...rows].join('\n')
    } else {
      text = JSON.stringify(data.data, null, 2)
    }
    navigator.clipboard.writeText(text)
    setCopied(format)
    setShowCopyMenu(false)
    setTimeout(() => setCopied(null), 2000)
  }, [data])

  if (!data || data.data.length === 0) {
    return (
      <div className="text-sm text-gray-500 dark:text-gray-400 py-4">
        No data available
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* Copy button */}
      <div className="flex justify-end relative">
        <div className="relative">
          <button
            onClick={() => setShowCopyMenu(!showCopyMenu)}
            className={`flex items-center gap-1 px-2 py-1 text-xs rounded transition-all ${
              copied
                ? 'text-green-500 dark:text-green-400'
                : 'text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300'
            }`}
            title="Copy table data"
          >
            {copied ? (
              <>
                <ClipboardDocumentCheckIcon className="w-3.5 h-3.5" />
                Copied {copied.toUpperCase()}
              </>
            ) : (
              <>
                <ClipboardDocumentIcon className="w-3.5 h-3.5" />
                Copy
              </>
            )}
          </button>
          {showCopyMenu && (
            <div className="absolute right-0 top-full mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg z-10">
              <button
                onClick={() => handleCopy('csv')}
                className="block w-full px-4 py-2 text-xs text-left text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-t-md"
              >
                Copy as CSV
              </button>
              <button
                onClick={() => handleCopy('json')}
                className="block w-full px-4 py-2 text-xs text-left text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-b-md"
              >
                Copy as JSON
              </button>
            </div>
          )}
        </div>
      </div>
      {/* Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full text-sm">
          <thead>
            <tr className="border-b border-gray-200 dark:border-gray-700">
              {data.columns.map((col) => (
                <th
                  key={col}
                  className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
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
        <div className="flex items-center justify-between text-sm">
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