// Artifact Item Accordion - individual artifact with expandable content and fullscreen mode

import { useState, useEffect } from 'react'
import {
  ChevronDownIcon,
  ChevronRightIcon,
  ArrowsPointingOutIcon,
  XMarkIcon,
  StarIcon as StarOutline,
} from '@heroicons/react/24/outline'
import { StarIcon as StarSolid } from '@heroicons/react/24/solid'
import { useSessionStore } from '@/store/sessionStore'
import { useArtifactStore } from '@/store/artifactStore'
import * as sessionsApi from '@/api/sessions'
import type { Artifact, ArtifactContent } from '@/types/api'

interface ArtifactItemAccordionProps {
  artifact: Artifact
}

export function ArtifactItemAccordion({ artifact }: ArtifactItemAccordionProps) {
  const { session } = useSessionStore()
  const { toggleArtifactStar } = useArtifactStore()
  const [isOpen, setIsOpen] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [content, setContent] = useState<ArtifactContent | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleToggleStar = (e: React.MouseEvent) => {
    e.stopPropagation()
    toggleArtifactStar(artifact.id)
  }

  // Fetch content when opened
  useEffect(() => {
    if (!session || !isOpen || content) return

    const fetchContent = async () => {
      setLoading(true)
      setError(null)
      try {
        const artifactContent = await sessionsApi.getArtifact(
          session.session_id,
          artifact.id
        )
        setContent(artifactContent)
      } catch (err) {
        setError(String(err))
      } finally {
        setLoading(false)
      }
    }

    fetchContent()
  }, [session, artifact.id, isOpen, content])

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
    setIsOpen(true) // Ensure content is loaded
    setIsFullscreen(true)
  }

  const renderContent = (maxHeight?: string) => {
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

    if (!content) {
      return (
        <div className="text-sm text-gray-500 dark:text-gray-400 py-4">
          No content available
        </div>
      )
    }

    const containerStyle = maxHeight ? { maxHeight } : undefined

    // Binary images
    if (content.is_binary && content.mime_type.startsWith('image/')) {
      return (
        <div className="overflow-auto" style={containerStyle}>
          <img
            src={`data:${content.mime_type};base64,${content.content}`}
            alt={content.name}
            className="max-w-full h-auto"
          />
        </div>
      )
    }

    // HTML content
    if (content.mime_type === 'text/html' || content.artifact_type === 'html') {
      return (
        <iframe
          srcDoc={content.content}
          className="w-full border-0"
          style={{ height: maxHeight || '300px' }}
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
            <body style="margin:0">
              <div id="plot"></div>
              <script>
                const data = ${content.content};
                Plotly.newPlot('plot', data.data || data, data.layout || {}, {responsive: true});
              </script>
            </body>
            </html>
          `}
          className="w-full border-0"
          style={{ height: maxHeight || '300px' }}
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
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 16px; margin: 0; line-height: 1.6; color: #1f2937; background: #fff; }
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
          className="w-full border-0"
          style={{ height: maxHeight || '300px' }}
          title={content.name}
          sandbox="allow-scripts"
        />
      )
    }

    // SVG
    if (content.artifact_type === 'svg' || content.mime_type === 'image/svg+xml') {
      return (
        <div
          className="overflow-auto p-4 bg-white dark:bg-gray-900"
          style={containerStyle}
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
            <body style="margin:0;padding:16px">
              <div id="vis"></div>
              <script>
                const spec = ${content.content};
                vegaEmbed('#vis', spec, {actions: false});
              </script>
            </body>
            </html>
          `}
          className="w-full border-0"
          style={{ height: maxHeight || '300px' }}
          title={content.name}
          sandbox="allow-scripts"
        />
      )
    }

    // Default: show as pre-formatted text
    return (
      <div className="overflow-auto" style={containerStyle}>
        <pre className="p-3 text-xs text-gray-600 dark:text-gray-400 whitespace-pre-wrap">
          {content.content?.substring(0, 5000)}
          {content.content?.length > 5000 && '...'}
        </pre>
      </div>
    )
  }

  const typeLabel = artifact.artifact_type || 'artifact'

  return (
    <>
      {/* Accordion Item */}
      <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
        {/* Header */}
        <button
          onClick={toggleOpen}
          className="w-full flex items-center justify-between px-3 py-2 bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
        >
          <div className="flex items-center gap-2 min-w-0">
            {isOpen ? (
              <ChevronDownIcon className="w-4 h-4 text-gray-500 flex-shrink-0" />
            ) : (
              <ChevronRightIcon className="w-4 h-4 text-gray-500 flex-shrink-0" />
            )}
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300 truncate">
              {artifact.title || artifact.name}
            </span>
            <span className="text-xs text-gray-400 dark:text-gray-500 flex-shrink-0">
              ({typeLabel})
            </span>
          </div>
          <div className="flex items-center gap-1 flex-shrink-0">
            <button
              onClick={handleToggleStar}
              className="p-1 hover:bg-gray-200 dark:hover:bg-gray-600 rounded transition-colors"
              title={artifact.is_key_result ? "Remove from key results" : "Add to key results"}
            >
              {artifact.is_key_result ? (
                <StarSolid className="w-4 h-4 text-yellow-500" />
              ) : (
                <StarOutline className="w-4 h-4 text-gray-400 hover:text-yellow-500" />
              )}
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
          <div className="bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
            {renderContent('300px')}
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
                  {artifact.title || artifact.name}
                </span>
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  ({typeLabel})
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
            <div className="flex-1 overflow-hidden">
              {renderContent('calc(95vh - 70px)')}
            </div>
          </div>
        </div>
      )}
    </>
  )
}