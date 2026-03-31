// Copyright (c) 2025 Kenneth Stott
// Canary: 1ea6112e-a2bb-45dd-a567-275dcbf59a4f
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// FlagButton — flag an assistant answer as incorrect

import { useState, useRef, useEffect } from 'react'
import { FlagIcon } from '@heroicons/react/24/outline'
import { useAuth } from '@/contexts/AuthContext'
import { useSessionContext } from '@/contexts/SessionContext'
import { apolloClient } from '@/graphql/client'
import { FLAG_ANSWER } from '@/graphql/operations/feedback'

interface FlagButtonProps {
  /** The user query that produced this answer */
  queryText: string
  /** Brief summary of the answer being flagged */
  answerSummary: string
}

export function FlagButton({ queryText, answerSummary }: FlagButtonProps) {
  const { permissions } = useAuth()
  const { sessionId } = useSessionContext()

  // Only show if user has flag_answers feedback permission
  if (!permissions?.feedback?.flag_answers) return null
  if (!sessionId) return null

  return (
    <FlagButtonInner
      queryText={queryText}
      answerSummary={answerSummary}
      sessionId={sessionId}
    />
  )
}

function FlagButtonInner({
  queryText,
  answerSummary,
  sessionId,
}: FlagButtonProps & { sessionId: string }) {
  const [open, setOpen] = useState(false)
  const [message, setMessage] = useState('')
  const [glossaryTerm, setGlossaryTerm] = useState('')
  const [suggestedDefinition, setSuggestedDefinition] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const popoverRef = useRef<HTMLDivElement>(null)

  // Close on outside click
  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (popoverRef.current && !popoverRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  const handleSubmit = async () => {
    if (!message.trim()) return
    setSubmitting(true)
    try {
      await apolloClient.mutate({
        mutation: FLAG_ANSWER,
        variables: {
          input: {
            sessionId,
            queryText,
            answerSummary,
            message: message.trim(),
            glossaryTerm: glossaryTerm.trim() || undefined,
            suggestedDefinition: suggestedDefinition.trim() || undefined,
          },
        },
      })
      setSubmitted(true)
      setTimeout(() => {
        setOpen(false)
        setSubmitted(false)
        setMessage('')
        setGlossaryTerm('')
        setSuggestedDefinition('')
      }, 1500)
    } catch (err) {
      console.error('Failed to flag answer:', err)
    } finally {
      setSubmitting(false)
    }
  }

  if (submitted && !open) {
    return (
      <span className="text-xs text-green-500 dark:text-green-400">Flagged</span>
    )
  }

  return (
    <div className="relative inline-block">
      <button
        onClick={() => setOpen(!open)}
        className="p-1 rounded text-gray-400 hover:text-orange-500 dark:text-gray-500 dark:hover:text-orange-400 opacity-0 group-hover:opacity-100 transition-opacity"
        title="Flag this answer"
      >
        <FlagIcon className="w-4 h-4" />
      </button>

      {open && (
        <div
          ref={popoverRef}
          className="absolute bottom-full right-0 mb-2 w-80 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-50 p-3"
        >
          {submitted ? (
            <p className="text-sm text-green-600 dark:text-green-400 text-center py-2">
              Feedback submitted
            </p>
          ) : (
            <>
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Flag this answer
              </p>
              <textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="What was wrong with this answer?"
                rows={3}
                className="w-full text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1.5 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-blue-500 resize-none"
                autoFocus
              />

              {/* Optional glossary correction */}
              <details className="mt-2">
                <summary className="text-xs text-gray-500 dark:text-gray-400 cursor-pointer hover:text-gray-700 dark:hover:text-gray-300">
                  Suggest glossary correction
                </summary>
                <div className="mt-1 space-y-1">
                  <input
                    value={glossaryTerm}
                    onChange={(e) => setGlossaryTerm(e.target.value)}
                    placeholder="Term name"
                    className="w-full text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  />
                  <input
                    value={suggestedDefinition}
                    onChange={(e) => setSuggestedDefinition(e.target.value)}
                    placeholder="Suggested definition"
                    className="w-full text-sm border border-gray-300 dark:border-gray-600 rounded px-2 py-1 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-blue-500"
                  />
                </div>
              </details>

              <div className="flex justify-end mt-2 gap-2">
                <button
                  onClick={() => setOpen(false)}
                  className="text-xs px-2 py-1 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSubmit}
                  disabled={!message.trim() || submitting}
                  className="text-xs px-3 py-1 bg-orange-500 hover:bg-orange-600 text-white rounded disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {submitting ? 'Submitting...' : 'Submit'}
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
