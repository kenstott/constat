import React, { useEffect, useState } from 'react'
import {
  CheckCircleIcon,
  XCircleIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  PencilIcon,
  TrashIcon,
  PlusIcon,
} from '@heroicons/react/24/outline'
import { useTestStore } from '@/store/testStore'
import type { GoldenQuestionExpectations, GoldenQuestionRequest, GoldenQuestionResponse } from '@/types/api'

interface Props {
  sessionId: string
}

/** Title-case a canonical entity name for display (mirrors Python display_entity_name). */
function dn(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

// ---------------------------------------------------------------------------
// Inline form for create / edit
// ---------------------------------------------------------------------------

function QuestionForm({
  initial,
  onSave,
  onCancel,
}: {
  initial?: GoldenQuestionResponse
  onSave: (body: GoldenQuestionRequest) => void
  onCancel: () => void
}) {
  const [question, setQuestion] = useState(initial?.question ?? '')
  const [tagsStr, setTagsStr] = useState((initial?.tags ?? []).join(', '))
  const [entitiesStr, setEntitiesStr] = useState((initial?.expect.entities ?? []).map(dn).join(', '))
  const [grounding, setGrounding] = useState<Array<Record<string, unknown>>>(
    (initial?.expect.grounding ?? []).map(g => ({ ...g, entity: g.entity ? dn(String(g.entity)) : '' }))
  )
  const [relationships, setRelationships] = useState<Array<Record<string, unknown>>>(
    (initial?.expect.relationships ?? []).map(r => ({
      ...r,
      subject: r.subject ? dn(String(r.subject)) : '',
      object: r.object ? dn(String(r.object)) : '',
    }))
  )
  const [glossary, setGlossary] = useState<Array<Record<string, unknown>>>(
    (initial?.expect.glossary ?? []).map(g => ({ ...g, name: g.name ? dn(String(g.name)) : '' }))
  )

  const handleSave = () => {
    const norm = (s: string) => s.trim().toLowerCase().replace(/\s+/g, '_')
    const tags = tagsStr.split(',').map(s => s.trim()).filter(Boolean)
    const entities = entitiesStr.split(',').map(s => s.trim()).filter(Boolean).map(norm)
    const expect: GoldenQuestionExpectations = {
      entities,
      grounding: grounding.map(g => ({ ...g, entity: norm(String(g.entity ?? '')) })),
      relationships: relationships.map(r => ({
        ...r,
        subject: norm(String(r.subject ?? '')),
        object: norm(String(r.object ?? '')),
      })),
      glossary: glossary.map(g => ({ ...g, name: norm(String(g.name ?? '')) })),
    }
    onSave({ question, tags, expect })
  }

  const inputClass = 'w-full px-2 py-1 text-xs border rounded dark:border-gray-600 dark:bg-gray-800 dark:text-gray-200'
  const labelClass = 'text-[10px] font-medium text-gray-500 dark:text-gray-400 uppercase mt-1.5'

  return (
    <div className="p-2 space-y-1.5 bg-gray-50 dark:bg-gray-800/50 border-t dark:border-gray-700">
      <div>
        <div className={labelClass}>Question</div>
        <input className={inputClass} value={question} onChange={e => setQuestion(e.target.value)} placeholder="What is the total revenue?" />
      </div>
      <div>
        <div className={labelClass}>Tags (comma-separated)</div>
        <input className={inputClass} value={tagsStr} onChange={e => setTagsStr(e.target.value)} placeholder="smoke, revenue" />
      </div>
      <div>
        <div className={labelClass}>Expected entities (comma-separated)</div>
        <input className={inputClass} value={entitiesStr} onChange={e => setEntitiesStr(e.target.value)} placeholder="revenue, orders" />
      </div>

      {/* Grounding rows */}
      <div>
        <div className="flex items-center justify-between">
          <div className={labelClass}>Grounding</div>
          <button onClick={() => setGrounding([...grounding, { entity: '', resolves_to: '' }])} className="text-[10px] text-blue-600 dark:text-blue-400 hover:underline">+ row</button>
        </div>
        {grounding.map((g, i) => (
          <div key={i} className="flex gap-1 mt-0.5">
            <input className={inputClass} placeholder="entity" value={String(g.entity ?? '')} onChange={e => { const next = [...grounding]; next[i] = { ...next[i], entity: e.target.value }; setGrounding(next) }} />
            <input className={inputClass} placeholder="resolves_to (comma-sep)" value={String(g.resolves_to ?? '')} onChange={e => { const next = [...grounding]; next[i] = { ...next[i], resolves_to: e.target.value }; setGrounding(next) }} />
            <button onClick={() => setGrounding(grounding.filter((_, j) => j !== i))} className="text-red-500 hover:text-red-700 px-1"><TrashIcon className="w-3 h-3" /></button>
          </div>
        ))}
      </div>

      {/* Relationship rows */}
      <div>
        <div className="flex items-center justify-between">
          <div className={labelClass}>Relationships</div>
          <button onClick={() => setRelationships([...relationships, { subject: '', verb: '', object: '', min_confidence: 0.5 }])} className="text-[10px] text-blue-600 dark:text-blue-400 hover:underline">+ row</button>
        </div>
        {relationships.map((r, i) => (
          <div key={i} className="flex gap-1 mt-0.5">
            <input className={inputClass} placeholder="subject" value={String(r.subject ?? '')} onChange={e => { const next = [...relationships]; next[i] = { ...next[i], subject: e.target.value }; setRelationships(next) }} />
            <input className={inputClass} placeholder="verb" value={String(r.verb ?? '')} onChange={e => { const next = [...relationships]; next[i] = { ...next[i], verb: e.target.value }; setRelationships(next) }} />
            <input className={inputClass} placeholder="object" value={String(r.object ?? '')} onChange={e => { const next = [...relationships]; next[i] = { ...next[i], object: e.target.value }; setRelationships(next) }} />
            <input className={`${inputClass} w-16`} placeholder="conf" type="number" step="0.1" min="0" max="1" value={String(r.min_confidence ?? '')} onChange={e => { const next = [...relationships]; next[i] = { ...next[i], min_confidence: parseFloat(e.target.value) || 0 }; setRelationships(next) }} />
            <button onClick={() => setRelationships(relationships.filter((_, j) => j !== i))} className="text-red-500 hover:text-red-700 px-1"><TrashIcon className="w-3 h-3" /></button>
          </div>
        ))}
      </div>

      {/* Glossary rows */}
      <div>
        <div className="flex items-center justify-between">
          <div className={labelClass}>Glossary</div>
          <button onClick={() => setGlossary([...glossary, { name: '', has_definition: true }])} className="text-[10px] text-blue-600 dark:text-blue-400 hover:underline">+ row</button>
        </div>
        {glossary.map((g, i) => (
          <div key={i} className="flex gap-1 mt-0.5 items-center">
            <input className={inputClass} placeholder="name" value={String(g.name ?? '')} onChange={e => { const next = [...glossary]; next[i] = { ...next[i], name: e.target.value }; setGlossary(next) }} />
            <label className="flex items-center gap-0.5 text-[10px] text-gray-500 whitespace-nowrap">
              <input type="checkbox" checked={!!g.has_definition} onChange={e => { const next = [...glossary]; next[i] = { ...next[i], has_definition: e.target.checked }; setGlossary(next) }} />
              def
            </label>
            <input className={inputClass} placeholder="domain" value={String(g.domain ?? '')} onChange={e => { const next = [...glossary]; next[i] = { ...next[i], domain: e.target.value }; setGlossary(next) }} />
            <input className={inputClass} placeholder="parent" value={String(g.parent ?? '')} onChange={e => { const next = [...glossary]; next[i] = { ...next[i], parent: e.target.value }; setGlossary(next) }} />
            <button onClick={() => setGlossary(glossary.filter((_, j) => j !== i))} className="text-red-500 hover:text-red-700 px-1"><TrashIcon className="w-3 h-3" /></button>
          </div>
        ))}
      </div>

      <div className="flex gap-2 pt-1">
        <button onClick={handleSave} disabled={!question.trim()} className="px-3 py-1 rounded bg-blue-600 text-white text-xs font-medium hover:bg-blue-700 disabled:opacity-50">Save</button>
        <button onClick={onCancel} className="px-3 py-1 rounded border text-xs text-gray-600 dark:text-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700">Cancel</button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export default function RegressionPanel({ sessionId }: Props) {
  const {
    testableDomains,
    results,
    loading,
    error,
    progress,
    selectedDomains,
    selectedTags,
    includeE2e,
    goldenQuestions,
    editingQuestion,
    loadTestableDomains,
    runTests,
    toggleDomain,
    toggleTag,
    setIncludeE2e,
    clearResults,
    loadGoldenQuestions,
    saveGoldenQuestion,
    deleteGoldenQuestion,
    setEditingQuestion,
    clearEditing,
  } = useTestStore()

  const [expandedDomains, setExpandedDomains] = useState<Set<string>>(new Set())
  const [expandedQuestions, setExpandedQuestions] = useState<Set<string>>(new Set())
  const [manageMode, setManageMode] = useState(false)
  const [manageDomains, setManageDomains] = useState<Set<string>>(new Set())

  useEffect(() => {
    loadTestableDomains(sessionId)
  }, [sessionId, loadTestableDomains])

  const allTags = Array.from(
    new Set(testableDomains.flatMap((d) => d.tags))
  ).sort()

  const toggleExpanded = (key: string, set: Set<string>, setter: React.Dispatch<React.SetStateAction<Set<string>>>) => {
    const next = new Set(set)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    setter(next)
  }

  const toggleManageDomain = (filename: string) => {
    const next = new Set(manageDomains)
    if (next.has(filename)) {
      next.delete(filename)
    } else {
      next.add(filename)
      // Load questions when expanding
      loadGoldenQuestions(sessionId, filename)
    }
    setManageDomains(next)
  }

  if (testableDomains.length === 0 && !loading) {
    return (
      <div className="px-3 py-4 text-sm text-gray-500 dark:text-gray-400">
        No domains with golden questions configured.
      </div>
    )
  }

  return (
    <div className="px-3 py-2 space-y-3 text-sm">
      {/* Mode toggle */}
      <div className="flex gap-1 border rounded dark:border-gray-700 p-0.5">
        <button
          onClick={() => setManageMode(false)}
          className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
            !manageMode
              ? 'bg-blue-600 text-white'
              : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
          }`}
        >
          Run
        </button>
        <button
          onClick={() => setManageMode(true)}
          className={`flex-1 px-2 py-1 rounded text-xs font-medium transition-colors ${
            manageMode
              ? 'bg-blue-600 text-white'
              : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
          }`}
        >
          Manage
        </button>
      </div>

      {/* ================================================================ */}
      {/* MANAGE MODE                                                      */}
      {/* ================================================================ */}
      {manageMode ? (
        <div className="space-y-2">
          {testableDomains.map((d) => {
            const expanded = manageDomains.has(d.filename)
            const questions = goldenQuestions[d.filename] ?? []
            const isAdding = editingQuestion?.domain === d.filename && editingQuestion.index === null

            return (
              <div key={d.filename} className="border rounded dark:border-gray-700">
                <button
                  onClick={() => toggleManageDomain(d.filename)}
                  className="w-full flex items-center gap-2 px-2 py-1.5 text-left hover:bg-gray-50 dark:hover:bg-gray-800"
                >
                  {expanded ? (
                    <ChevronDownIcon className="w-3.5 h-3.5 text-gray-400" />
                  ) : (
                    <ChevronRightIcon className="w-3.5 h-3.5 text-gray-400" />
                  )}
                  <span className="font-medium text-gray-800 dark:text-gray-200">{d.name}</span>
                  <span className="ml-auto text-xs text-gray-400">{d.question_count} questions</span>
                </button>

                {expanded && (
                  <div className="border-t dark:border-gray-700">
                    {questions.map((q) => {
                      const isEditing = editingQuestion?.domain === d.filename && editingQuestion.index === q.index
                      return (
                        <div key={q.index} className="border-b last:border-b-0 dark:border-gray-700">
                          {isEditing ? (
                            <QuestionForm
                              initial={q}
                              onSave={(body) => saveGoldenQuestion(sessionId, d.filename, q.index, body)}
                              onCancel={clearEditing}
                            />
                          ) : (
                            <div className="flex items-center gap-2 px-3 py-1.5 text-xs group">
                              <span className="text-gray-700 dark:text-gray-300 truncate flex-1">{q.question}</span>
                              <div className="flex gap-1 flex-shrink-0">
                                {q.tags.map(t => (
                                  <span key={t} className="px-1.5 py-0.5 rounded-full bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300 text-[10px]">{t}</span>
                                ))}
                              </div>
                              <button
                                onClick={() => setEditingQuestion(d.filename, q.index)}
                                className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-opacity"
                              >
                                <PencilIcon className="w-3.5 h-3.5" />
                              </button>
                              <button
                                onClick={() => {
                                  if (window.confirm(`Delete question: "${q.question}"?`)) {
                                    deleteGoldenQuestion(sessionId, d.filename, q.index)
                                  }
                                }}
                                className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-opacity"
                              >
                                <TrashIcon className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          )}
                        </div>
                      )
                    })}

                    {isAdding ? (
                      <QuestionForm
                        onSave={(body) => saveGoldenQuestion(sessionId, d.filename, null, body)}
                        onCancel={clearEditing}
                      />
                    ) : (
                      <button
                        onClick={() => setEditingQuestion(d.filename, null)}
                        className="w-full flex items-center gap-1.5 px-3 py-1.5 text-xs text-blue-600 dark:text-blue-400 hover:bg-gray-50 dark:hover:bg-gray-800"
                      >
                        <PlusIcon className="w-3.5 h-3.5" />
                        Add Question
                      </button>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      ) : (
        <>
          {/* ================================================================ */}
          {/* RUN MODE (existing UI)                                           */}
          {/* ================================================================ */}

          {/* Domain selector */}
          <div>
            <div className="font-medium text-gray-700 dark:text-gray-300 mb-1">Domains</div>
            {testableDomains.map((d) => (
              <label key={d.filename} className="flex items-center gap-2 py-0.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={selectedDomains.has(d.filename)}
                  onChange={() => toggleDomain(d.filename)}
                  className="rounded border-gray-300 dark:border-gray-600"
                />
                <span className="text-gray-800 dark:text-gray-200">{d.name}</span>
                <span className="text-gray-400 text-xs">({d.question_count})</span>
              </label>
            ))}
          </div>

          {/* Tag filter */}
          {allTags.length > 0 && (
            <div>
              <div className="font-medium text-gray-700 dark:text-gray-300 mb-1">Tags</div>
              <div className="flex flex-wrap gap-1">
                {allTags.map((tag) => (
                  <button
                    key={tag}
                    onClick={() => toggleTag(tag)}
                    className={`px-2 py-0.5 rounded-full text-xs border transition-colors ${
                      selectedTags.has(tag)
                        ? 'bg-blue-100 border-blue-300 text-blue-800 dark:bg-blue-900 dark:border-blue-700 dark:text-blue-200'
                        : 'bg-gray-100 border-gray-200 text-gray-600 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-300'
                    }`}
                  >
                    {tag}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* E2E toggle + Run button */}
          <div className="space-y-1.5">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={includeE2e}
                onChange={(e) => setIncludeE2e(e.target.checked)}
                className="rounded border-gray-300 dark:border-gray-600"
              />
              <span className="text-xs text-gray-600 dark:text-gray-400">Include E2E (slow, uses LLM)</span>
            </label>
            <button
              onClick={() => runTests(sessionId)}
              disabled={loading}
              className="w-full px-3 py-1.5 rounded bg-blue-600 text-white text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Running...
                </>
              ) : (
                'Run Tests'
              )}
            </button>

            {/* Live progress */}
            {loading && progress && (
              <div className="mt-2 p-2 rounded border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 space-y-1.5 animate-in fade-in">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                  <span className="text-xs font-medium text-blue-700 dark:text-blue-300">{progress.domainName}</span>
                </div>
                {progress.question && (
                  <div className="text-xs text-gray-600 dark:text-gray-400 pl-4 truncate">
                    {progress.questionIndex + 1}/{progress.questionTotal}: {progress.question}
                  </div>
                )}
                {progress.phase && (
                  <div className="flex items-center gap-1.5 pl-4">
                    <svg className="animate-spin h-3 w-3 text-blue-500" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                    <span className="text-[10px] uppercase font-medium text-blue-600 dark:text-blue-400">
                      {progress.phase === 'e2e' ? 'End-to-end (LLM)' : 'Checking metadata'}
                    </span>
                  </div>
                )}
                {progress.questionTotal > 0 && (
                  <div className="pl-4">
                    <div className="w-full h-1 rounded-full bg-blue-200 dark:bg-blue-800 overflow-hidden">
                      <div
                        className="h-full bg-blue-500 rounded-full transition-all duration-300"
                        style={{ width: `${((progress.questionIndex) / progress.questionTotal) * 100}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="text-red-600 dark:text-red-400 text-xs">{error}</div>
          )}

          {/* Results */}
          {results && (
            <div className="space-y-2">
              {/* Summary bar */}
              <div className="flex items-center gap-3 py-1.5 px-2 rounded bg-gray-50 dark:bg-gray-800">
                <span className="flex items-center gap-1 text-green-600 dark:text-green-400 font-medium">
                  <CheckCircleIcon className="w-4 h-4" />
                  {results.total_passed} passed
                </span>
                <span className="flex items-center gap-1 text-red-600 dark:text-red-400 font-medium">
                  <XCircleIcon className="w-4 h-4" />
                  {results.total_failed} failed
                </span>
                <button
                  onClick={clearResults}
                  className="ml-auto text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  Clear
                </button>
              </div>

              {/* Per-domain results */}
              {results.domains.map((dr) => (
                <div key={dr.domain} className="border rounded dark:border-gray-700">
                  <button
                    onClick={() => toggleExpanded(dr.domain, expandedDomains, setExpandedDomains)}
                    className="w-full flex items-center gap-2 px-2 py-1.5 text-left hover:bg-gray-50 dark:hover:bg-gray-800"
                  >
                    {expandedDomains.has(dr.domain) ? (
                      <ChevronDownIcon className="w-3.5 h-3.5 text-gray-400" />
                    ) : (
                      <ChevronRightIcon className="w-3.5 h-3.5 text-gray-400" />
                    )}
                    <span className="font-medium text-gray-800 dark:text-gray-200">{dr.domain_name || dr.domain}</span>
                    <span className="ml-auto text-xs">
                      <span className="text-green-600 dark:text-green-400">{dr.passed_count}</span>
                      {' / '}
                      <span>{dr.passed_count + dr.failed_count}</span>
                    </span>
                  </button>

                  {expandedDomains.has(dr.domain) && (
                    <div className="border-t dark:border-gray-700">
                      {dr.questions.map((qr, qi) => {
                        const qKey = `${dr.domain}:${qi}`
                        return (
                          <div key={qi} className="border-b last:border-b-0 dark:border-gray-700">
                            <button
                              onClick={() => toggleExpanded(qKey, expandedQuestions, setExpandedQuestions)}
                              className="w-full flex items-center gap-2 px-3 py-1 text-left text-xs hover:bg-gray-50 dark:hover:bg-gray-800"
                            >
                              {qr.passed ? (
                                <CheckCircleIcon className="w-4 h-4 text-green-500 flex-shrink-0" />
                              ) : (
                                <XCircleIcon className="w-4 h-4 text-red-500 flex-shrink-0" />
                              )}
                              <span className="text-gray-700 dark:text-gray-300 truncate flex-1">{qr.question}</span>
                              <div className="flex gap-1 flex-shrink-0">
                                {qr.layers.map((lr) => (
                                  <span
                                    key={lr.layer}
                                    className={`px-1.5 py-0.5 rounded text-[10px] ${
                                      lr.passed === lr.total
                                        ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                                        : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                                    }`}
                                  >
                                    {lr.layer}: {lr.passed}/{lr.total}
                                  </span>
                                ))}
                                {qr.end_to_end && (
                                  <span
                                    className={`px-1.5 py-0.5 rounded text-[10px] ${
                                      qr.end_to_end.passed
                                        ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                                        : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                                    }`}
                                  >
                                    e2e
                                  </span>
                                )}
                              </div>
                            </button>

                            {expandedQuestions.has(qKey) && (
                              <div className="px-3 py-1.5 bg-gray-50 dark:bg-gray-800/50">
                                {qr.layers.map((lr) =>
                                  lr.failures.length > 0 ? (
                                    <div key={lr.layer} className="mb-1">
                                      <div className="text-[10px] font-medium text-gray-500 uppercase">{lr.layer}</div>
                                      {lr.failures.map((f, fi) => (
                                        <div key={fi} className="text-xs text-red-600 dark:text-red-400 pl-2">
                                          {f}
                                        </div>
                                      ))}
                                    </div>
                                  ) : null
                                )}
                                {qr.end_to_end && (
                                  <div className="mb-1 border-t dark:border-gray-700 pt-1 mt-1">
                                    <div className="flex items-center gap-2">
                                      <div className="text-[10px] font-medium text-gray-500 uppercase">End-to-End</div>
                                      {qr.end_to_end.passed ? (
                                        <CheckCircleIcon className="w-3.5 h-3.5 text-green-500" />
                                      ) : (
                                        <XCircleIcon className="w-3.5 h-3.5 text-red-500" />
                                      )}
                                      <span className="text-[10px] text-gray-400">{qr.end_to_end.duration_s.toFixed(1)}s</span>
                                    </div>
                                    {qr.end_to_end.failures.map((f, fi) => (
                                      <div key={fi} className="text-xs text-red-600 dark:text-red-400 pl-2">{f}</div>
                                    ))}
                                    {qr.end_to_end.answer && (
                                      <div className="mt-1">
                                        <div className="text-[10px] font-medium text-gray-500 uppercase">Answer</div>
                                        <div className="text-xs text-gray-600 dark:text-gray-400 pl-2 line-clamp-3">{qr.end_to_end.answer}</div>
                                      </div>
                                    )}
                                    {qr.end_to_end.judge_reasoning && (
                                      <div className="mt-1">
                                        <div className="text-[10px] font-medium text-gray-500 uppercase">Judge</div>
                                        <div className="text-xs text-gray-600 dark:text-gray-400 pl-2">{qr.end_to_end.judge_reasoning}</div>
                                      </div>
                                    )}
                                  </div>
                                )}
                                {qr.layers.every((lr) => lr.failures.length === 0) && !qr.end_to_end && (
                                  <div className="text-xs text-green-600 dark:text-green-400">All assertions passed.</div>
                                )}
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}
