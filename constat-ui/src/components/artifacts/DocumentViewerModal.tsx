// Copyright (c) 2025 Kenneth Stott
// Canary: 7f2e4a91-b3c5-4d8e-a123-9f6b7c8d0e1f
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { XMarkIcon } from '@heroicons/react/24/outline'

export interface ViewingDocument {
  name: string
  content: string
  format?: string
  url?: string
  imageUrl?: string
}

interface DocumentViewerModalProps {
  viewingDocument: ViewingDocument | null
  loadingDocument: boolean
  onClose: () => void
}

export function DocumentViewerModal({ viewingDocument, loadingDocument, onClose }: DocumentViewerModalProps) {
  const [iframeBlocked, setIframeBlocked] = useState(false)

  if (!viewingDocument && !loadingDocument) return null

  return (
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
            onClick={onClose}
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
                try {
                  const iframe = e.target as HTMLIFrameElement
                  if (iframe.contentDocument?.title === '') {
                    setIframeBlocked(true)
                  }
                } catch {
                  // Cross-origin = loaded successfully
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
  )
}
