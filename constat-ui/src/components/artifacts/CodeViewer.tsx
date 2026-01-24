// Code Viewer component with syntax highlighting

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { ClipboardIcon, CheckIcon } from '@heroicons/react/24/outline'
import { useState } from 'react'

interface CodeViewerProps {
  code: string
  language?: string
  title?: string
  showLineNumbers?: boolean
}

export function CodeViewer({
  code,
  language = 'python',
  title,
  showLineNumbers = true,
}: CodeViewerProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <span className="text-xs font-medium text-gray-600 dark:text-gray-400">
          {title || language}
        </span>
        <button
          onClick={handleCopy}
          className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          title="Copy code"
        >
          {copied ? (
            <CheckIcon className="w-4 h-4 text-green-500" />
          ) : (
            <ClipboardIcon className="w-4 h-4 text-gray-500 dark:text-gray-400" />
          )}
        </button>
      </div>

      {/* Code */}
      <SyntaxHighlighter
        language={language}
        style={oneDark}
        showLineNumbers={showLineNumbers}
        customStyle={{
          margin: 0,
          padding: '1rem',
          fontSize: '0.75rem',
          lineHeight: '1.5',
          maxHeight: '400px',
          overflow: 'auto',
        }}
        lineNumberStyle={{
          minWidth: '2.5em',
          paddingRight: '1em',
          color: '#6b7280',
          userSelect: 'none',
        }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  )
}