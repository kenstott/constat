// Fullscreen Artifact Modal - triggered from View Result button

import { useState, useEffect } from 'react'
import { XMarkIcon } from '@heroicons/react/24/outline'
import { useUIStore } from '@/store/uiStore'
import { useSessionStore } from '@/store/sessionStore'
import * as sessionsApi from '@/api/sessions'
import type { ArtifactContent, TableData } from '@/types/api'
import type { DatabaseTablePreview } from '@/api/sessions'

export function FullscreenArtifactModal() {
  const { fullscreenArtifact, closeFullscreenArtifact } = useUIStore()
  const { session } = useSessionStore()
  const [content, setContent] = useState<ArtifactContent | null>(null)
  const [tableData, setTableData] = useState<TableData | null>(null)
  const [dbTableData, setDbTableData] = useState<DatabaseTablePreview | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [tablePage, setTablePage] = useState(1)

  // Close on Escape
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        closeFullscreenArtifact()
      }
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [closeFullscreenArtifact])

  // Fetch content when modal opens
  useEffect(() => {
    if (!fullscreenArtifact || !session) return

    // proof_value type doesn't need fetching - content is already provided
    if (fullscreenArtifact.type === 'proof_value') {
      setLoading(false)
      return
    }

    const fetchContent = async () => {
      setLoading(true)
      setError(null)
      setContent(null)
      setTableData(null)
      setDbTableData(null)

      try {
        if (fullscreenArtifact.type === 'table' && fullscreenArtifact.name) {
          const data = await sessionsApi.getTableData(
            session.session_id,
            fullscreenArtifact.name,
            tablePage
          )
          setTableData(data)
        } else if (fullscreenArtifact.type === 'database_table' && fullscreenArtifact.dbName && fullscreenArtifact.tableName) {
          const data = await sessionsApi.getDatabaseTablePreview(
            session.session_id,
            fullscreenArtifact.dbName,
            fullscreenArtifact.tableName,
            tablePage
          )
          setDbTableData(data)
        } else if (fullscreenArtifact.type === 'artifact' && fullscreenArtifact.id) {
          const artifactContent = await sessionsApi.getArtifact(
            session.session_id,
            fullscreenArtifact.id
          )
          setContent(artifactContent)
        }
      } catch (err) {
        setError(String(err))
      } finally {
        setLoading(false)
      }
    }

    fetchContent()
  }, [fullscreenArtifact, session, tablePage])

  if (!fullscreenArtifact) return null

  const renderContent = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center h-full">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500" />
        </div>
      )
    }

    if (error) {
      return (
        <div className="flex items-center justify-center h-full text-red-500">
          Error: {error}
        </div>
      )
    }

    // Proof value content (markdown table from proof node)
    if (fullscreenArtifact.type === 'proof_value' && fullscreenArtifact.content) {
      const isDark = document.documentElement.classList.contains('dark')
      return (
        <iframe
          srcDoc={`
            <!DOCTYPE html>
            <html class="${isDark ? 'dark' : ''}">
            <head>
              <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
              <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 24px; margin: 0; line-height: 1.6; color: #1f2937; background: #fff; }
                html.dark body { color: #e5e7eb; background: #111827; }
                table { border-collapse: collapse; width: 100%; margin: 1em 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                html.dark th, html.dark td { border-color: #374151; }
                th { background: #f5f5f5; font-weight: 600; }
                html.dark th { background: #1f2937; }
                tr:nth-child(even) { background: #f9fafb; }
                html.dark tr:nth-child(even) { background: #1f2937; }
                code { background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }
                html.dark code { background: #1f2937; }
                pre { background: #f5f5f5; padding: 12px; border-radius: 6px; overflow-x: auto; }
                html.dark pre { background: #1f2937; }
              </style>
            </head>
            <body>
              <div id="content"></div>
              <script>
                document.getElementById('content').innerHTML = marked.parse(${JSON.stringify(fullscreenArtifact.content)});
              </script>
            </body>
            </html>
          `}
          className="w-full h-full border-0"
          title={fullscreenArtifact.name || 'Proof Value'}
          sandbox="allow-scripts"
        />
      )
    }

    // Table content (session datastore tables)
    if (fullscreenArtifact.type === 'table' && tableData) {
      return (
        <div className="h-full flex flex-col">
          <div className="flex-1 overflow-auto">
            <table className="min-w-full text-sm">
              <thead className="sticky top-0 bg-white dark:bg-gray-900">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  {tableData.columns.map((col) => (
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
                {tableData.data.map((row, i) => (
                  <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                    {tableData.columns.map((col) => (
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
          {(tableData.has_more || tablePage > 1) && (
            <div className="flex items-center justify-between text-sm px-4 py-3 border-t border-gray-200 dark:border-gray-700">
              <span className="text-gray-500 dark:text-gray-400">
                Page {tablePage} of {Math.ceil(tableData.total_rows / tableData.page_size)}
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setTablePage((p) => Math.max(1, p - 1))}
                  disabled={tablePage === 1}
                  className="px-3 py-1 text-sm border rounded hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-50"
                >
                  Previous
                </button>
                <button
                  onClick={() => setTablePage((p) => p + 1)}
                  disabled={!tableData.has_more}
                  className="px-3 py-1 text-sm border rounded hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      )
    }

    // Database source table content
    if (fullscreenArtifact.type === 'database_table' && dbTableData) {
      return (
        <div className="h-full flex flex-col">
          <div className="flex-1 overflow-auto">
            <table className="min-w-full text-sm">
              <thead className="sticky top-0 bg-white dark:bg-gray-900">
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  {dbTableData.columns.map((col) => (
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
                {dbTableData.data.map((row, i) => (
                  <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-800/50">
                    {dbTableData.columns.map((col) => (
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
          {(dbTableData.has_more || tablePage > 1) && (
            <div className="flex items-center justify-between text-sm px-4 py-3 border-t border-gray-200 dark:border-gray-700">
              <span className="text-gray-500 dark:text-gray-400">
                Page {tablePage} of {Math.ceil(dbTableData.total_rows / dbTableData.page_size)}
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setTablePage((p) => Math.max(1, p - 1))}
                  disabled={tablePage === 1}
                  className="px-3 py-1 text-sm border rounded hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-50"
                >
                  Previous
                </button>
                <button
                  onClick={() => setTablePage((p) => p + 1)}
                  disabled={!dbTableData.has_more}
                  className="px-3 py-1 text-sm border rounded hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      )
    }

    // Artifact content
    if (content) {
      // Binary images
      if (content.is_binary && content.mime_type.startsWith('image/')) {
        return (
          <div className="h-full overflow-auto flex items-center justify-center p-4">
            <img
              src={`data:${content.mime_type};base64,${content.content}`}
              alt={content.name}
              className="max-w-full max-h-full object-contain"
            />
          </div>
        )
      }

      // HTML content
      if (content.mime_type === 'text/html' || content.artifact_type === 'html') {
        return (
          <iframe
            srcDoc={content.content}
            className="w-full h-full border-0"
            title={content.name}
            sandbox="allow-scripts"
          />
        )
      }

      // Plotly charts
      if (content.artifact_type === 'plotly') {
        return (
          <iframe
            srcDoc={`
              <!DOCTYPE html>
              <html>
              <head>
                <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
              </head>
              <body style="margin:0;height:100vh">
                <div id="plot" style="height:100%"></div>
                <script>
                  const data = ${content.content};
                  Plotly.newPlot('plot', data.data || data, data.layout || {}, {responsive: true});
                </script>
              </body>
              </html>
            `}
            className="w-full h-full border-0"
            title={content.name}
            sandbox="allow-scripts"
          />
        )
      }

      // Markdown
      if (
        ['markdown', 'md'].includes(content.artifact_type?.toLowerCase()) ||
        content.mime_type === 'text/markdown'
      ) {
        const isDark = document.documentElement.classList.contains('dark')
        return (
          <iframe
            srcDoc={`
              <!DOCTYPE html>
              <html class="${isDark ? 'dark' : ''}">
              <head>
                <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
                <style>
                  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 24px; margin: 0; line-height: 1.6; color: #1f2937; background: #fff; }
                  html.dark body { color: #e5e7eb; background: #111827; }
                  table { border-collapse: collapse; width: 100%; margin: 1em 0; }
                  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                  html.dark th, html.dark td { border-color: #374151; }
                  th { background: #f5f5f5; }
                  html.dark th { background: #1f2937; }
                  code { background: #f5f5f5; padding: 2px 4px; border-radius: 3px; }
                  html.dark code { background: #1f2937; }
                  pre { background: #f5f5f5; padding: 12px; border-radius: 6px; overflow-x: auto; }
                  html.dark pre { background: #1f2937; }
                  pre code { background: none; padding: 0; }
                  h1, h2, h3 { margin-top: 1em; margin-bottom: 0.5em; }
                  ul, ol { padding-left: 1.5em; }
                  a { color: #3b82f6; }
                  html.dark a { color: #60a5fa; }
                </style>
              </head>
              <body>
                <div id="content"></div>
                <script>
                  document.getElementById('content').innerHTML = marked.parse(${JSON.stringify(content.content)});
                </script>
              </body>
              </html>
            `}
            className="w-full h-full border-0"
            title={content.name}
            sandbox="allow-scripts"
          />
        )
      }

      // SVG
      if (content.artifact_type === 'svg' || content.mime_type === 'image/svg+xml') {
        return (
          <div
            className="h-full overflow-auto p-4 bg-white dark:bg-gray-900 flex items-center justify-center"
            dangerouslySetInnerHTML={{ __html: content.content }}
          />
        )
      }

      // Vega/Vega-Lite charts
      if (content.artifact_type === 'vega') {
        return (
          <iframe
            srcDoc={`
              <!DOCTYPE html>
              <html>
              <head>
                <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
                <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
                <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
              </head>
              <body style="margin:0;padding:24px;height:calc(100vh - 48px)">
                <div id="vis" style="height:100%"></div>
                <script>
                  const spec = ${content.content};
                  vegaEmbed('#vis', spec, {actions: false});
                </script>
              </body>
              </html>
            `}
            className="w-full h-full border-0"
            title={content.name}
            sandbox="allow-scripts"
          />
        )
      }

      // Default: show as pre-formatted text
      return (
        <div className="h-full overflow-auto p-4">
          <pre className="text-sm text-gray-600 dark:text-gray-400 whitespace-pre-wrap">
            {content.content}
          </pre>
        </div>
      )
    }

    return (
      <div className="flex items-center justify-center h-full text-gray-500">
        No content available
      </div>
    )
  }

  const title = fullscreenArtifact.type === 'table'
    ? fullscreenArtifact.name
    : fullscreenArtifact.type === 'database_table'
    ? `${fullscreenArtifact.dbName}.${fullscreenArtifact.tableName}`
    : fullscreenArtifact.type === 'proof_value'
    ? fullscreenArtifact.name || 'Proof Value'
    : content?.title || content?.name || 'Artifact'

  return (
    <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-gray-900 rounded-lg shadow-xl w-full h-full max-w-[95vw] max-h-[95vh] flex flex-col">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2">
            <span className="text-lg font-semibold text-gray-800 dark:text-gray-200">
              {title}
            </span>
            {fullscreenArtifact.type === 'table' && tableData && (
              <span className="text-sm text-gray-500 dark:text-gray-400">
                ({tableData.total_rows} rows)
              </span>
            )}
            {fullscreenArtifact.type === 'database_table' && dbTableData && (
              <span className="text-sm text-gray-500 dark:text-gray-400">
                ({dbTableData.total_rows} rows)
              </span>
            )}
          </div>
          <button
            onClick={closeFullscreenArtifact}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            title="Close (Esc)"
          >
            <XMarkIcon className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="flex-1 overflow-hidden">
          {renderContent()}
        </div>
      </div>
    </div>
  )
}
