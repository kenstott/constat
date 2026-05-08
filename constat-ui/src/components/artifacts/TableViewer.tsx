// Table Viewer component

import { useState, useEffect } from 'react'
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

  if (!data || data.data.length === 0) {
    return (
      <div className="text-sm text-gray-500 dark:text-gray-400 py-4">
        No data available
      </div>
    )
  }

  return (
    <div className="space-y-3">
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