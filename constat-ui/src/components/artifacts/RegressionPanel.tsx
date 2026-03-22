import React, { useEffect, useState, useRef } from 'react'
import {
  CheckCircleIcon,
  XCircleIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  PencilIcon,
  TrashIcon,
  PlusIcon,
  ArrowsRightLeftIcon,
  ArrowsPointingOutIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline'
import { useTestStore } from '@/store/testStore'
import { SkeletonLoader } from '../common/SkeletonLoader'
import type { ExpectedOutput, GoldenQuestionExpectations, GoldenQuestionRequest, GoldenQuestionResponse, TestQuestionResult } from '@/types/api'

interface Props {
  sessionId: string
}

/** Title-case a canonical entity name for display (mirrors Python display_entity_name). */
function dn(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

// ---------------------------------------------------------------------------
// Indeterminate checkbox
// ---------------------------------------------------------------------------

function IndeterminateCheckbox({
  indeterminate,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement> & { indeterminate?: boolean }) {
  const ref = useRef<HTMLInputElement>(null)
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = !!indeterminate
  }, [indeterminate])
  return <input type="checkbox" ref={ref} {...props} />
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
  const [terms, setTerms] = useState<Array<Record<string, unknown>>>(
    (initial?.expect.terms ?? []).map(t => ({ ...t, name: t.name ? dn(String(t.name)) : '' }))
  )
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
  const [expectedOutputs, setExpectedOutputs] = useState<ExpectedOutput[]>(
    (initial?.expect.expected_outputs ?? []).map(o => ({ ...o }))
  )
  const [judgePrompt, setJudgePrompt] = useState(
    String(initial?.expect.end_to_end?.judge_prompt ?? '')
  )
  const [validatorCode, setValidatorCode] = useState(
    String(initial?.expect.end_to_end?.validator_code ?? '')
  )

  const handleSave = () => {
    const norm = (s: string) => s.trim().toLowerCase().replace(/\s+/g, '_')
    const tags = tagsStr.split(',').map(s => s.trim()).filter(Boolean)
    const expect: GoldenQuestionExpectations = {
      terms: terms.map(t => ({ ...t, name: norm(String(t.name ?? '')) })),
      grounding: grounding.map(g => ({ ...g, entity: norm(String(g.entity ?? '')) })),
      relationships: relationships.map(r => ({
        ...r,
        subject: norm(String(r.subject ?? '')),
        object: norm(String(r.object ?? '')),
      })),
      expected_outputs: expectedOutputs.filter(o => o.name.trim()),
      end_to_end: judgePrompt.trim() || validatorCode.trim()
        ? { judge_prompt: judgePrompt.trim(), validator_code: validatorCode.trim() }
        : undefined,
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
      {/* Term rows */}
      <div>
        <div className="flex items-center justify-between">
          <div className={labelClass}>Expected terms</div>
          <button onClick={() => setTerms([...terms, { name: '', has_definition: false }])} className="text-[10px] text-blue-600 dark:text-blue-400 hover:underline">+ row</button>
        </div>
        {terms.map((t, i) => (
          <div key={i} className="flex gap-1 mt-0.5 items-center">
            <input className={inputClass} placeholder="name" value={String(t.name ?? '')} onChange={e => { const next = [...terms]; next[i] = { ...next[i], name: e.target.value }; setTerms(next) }} />
            <label className="flex items-center gap-0.5 text-[10px] text-gray-500 whitespace-nowrap">
              <input type="checkbox" checked={!!t.has_definition} onChange={e => { const next = [...terms]; next[i] = { ...next[i], has_definition: e.target.checked }; setTerms(next) }} />
              def
            </label>
            <button onClick={() => setTerms(terms.filter((_, j) => j !== i))} className="text-red-500 hover:text-red-700 px-1"><TrashIcon className="w-3 h-3" /></button>
          </div>
        ))}
      </div>

      {/* Grounding rows */}
      <div>
        <div className="flex items-center justify-between">
          <div className={labelClass}>Grounding</div>
          <button onClick={() => setGrounding([...grounding, { entity: '', resolves_to: '', strict: true }])} className="text-[10px] text-blue-600 dark:text-blue-400 hover:underline">+ row</button>
        </div>
        {grounding.map((g, i) => (
          <div key={i} className="flex gap-1 mt-0.5">
            <input className={inputClass} placeholder="entity" value={String(g.entity ?? '')} onChange={e => { const next = [...grounding]; next[i] = { ...next[i], entity: e.target.value }; setGrounding(next) }} />
            <input className={inputClass} placeholder="resolves_to (comma-sep)" value={String(g.resolves_to ?? '')} onChange={e => { const next = [...grounding]; next[i] = { ...next[i], resolves_to: e.target.value }; setGrounding(next) }} />
            <label className="flex items-center gap-0.5 text-[10px] text-gray-500 whitespace-nowrap">
              <input type="checkbox" checked={!!g.strict} onChange={e => { const next = [...grounding]; next[i] = { ...next[i], strict: e.target.checked }; setGrounding(next) }} />
              strict
            </label>
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

      {/* Expected outputs */}
      <div>
        <div className="flex items-center justify-between">
          <div className={labelClass}>Expected outputs</div>
          <button onClick={() => setExpectedOutputs([...expectedOutputs, { name: '', type: 'table', columns: [] }])} className="text-[10px] text-blue-600 dark:text-blue-400 hover:underline">+ row</button>
        </div>
        {expectedOutputs.map((o, i) => (
          <div key={i} className="flex gap-1 mt-0.5 items-center">
            <input className={inputClass} placeholder="name" value={o.name} onChange={e => { const next = [...expectedOutputs]; next[i] = { ...next[i], name: e.target.value }; setExpectedOutputs(next) }} />
            <select className={`${inputClass} w-24`} value={o.type} onChange={e => { const next = [...expectedOutputs]; next[i] = { ...next[i], type: e.target.value }; setExpectedOutputs(next) }}>
              <option value="table">table</option>
              <option value="image">image</option>
              <option value="document">document</option>
              <option value="markdown">markdown</option>
              <option value="json">json</option>
              <option value="xml">xml</option>
              <option value="pdf">pdf</option>
            </select>
            {o.type === 'table' && (
              <input className={inputClass} placeholder="columns (comma-sep)" value={o.columns.join(', ')} onChange={e => { const next = [...expectedOutputs]; next[i] = { ...next[i], columns: e.target.value.split(',').map(s => s.trim()).filter(Boolean) }; setExpectedOutputs(next) }} />
            )}
            <button onClick={() => setExpectedOutputs(expectedOutputs.filter((_, j) => j !== i))} className="text-red-500 hover:text-red-700 px-1"><TrashIcon className="w-3 h-3" /></button>
          </div>
        ))}
      </div>

      {/* LLM-as-judge prompt */}
      <div>
        <div className={labelClass}>LLM judge prompt</div>
        <textarea className={`${inputClass} h-20`} value={judgePrompt} onChange={e => setJudgePrompt(e.target.value)} placeholder="System prompt for the LLM judge. Leave blank to skip integration test." />
      </div>

      {/* Python data validator */}
      <div>
        <div className={labelClass}>Data validator (Python)</div>
        <textarea
          className={`${inputClass} h-20 font-mono text-[11px]`}
          value={validatorCode}
          onChange={e => setValidatorCode(e.target.value)}
          placeholder={'# result = last table created, tables = all by name\nassert len(result) == 15'}
        />
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
    domainsLoading,
    error,
    progress,
    selectedTags,
    goldenQuestions,
    editingQuestion,
    loadTestableDomains,
    runTests,
    toggleTag,
    clearResults,
    loadGoldenQuestions,
    saveGoldenQuestion,
    deleteGoldenQuestion,
    moveGoldenQuestion,
    setEditingQuestion,
    clearEditing,
    includeE2e,
  } = useTestStore()

  const [expandedDomains, setExpandedDomains] = useState<Set<string>>(new Set())
  const [expandedQuestions, setExpandedQuestions] = useState<Set<string>>(new Set())
  const [deselected, setDeselected] = useState<Set<string>>(new Set())
  const [movingQuestion, setMovingQuestion] = useState<{ domain: string; index: number } | null>(null)
  const [expanded, setExpanded] = useState(false)
  const domainsInitRef = useRef(false)
  const loadedDomainsRef = useRef<Set<string>>(new Set())

  // Load domains on mount
  useEffect(() => {
    loadTestableDomains(sessionId)
  }, [sessionId, loadTestableDomains])

  // Expand all domains on first load & load questions for each
  useEffect(() => {
    if (testableDomains.length === 0) return
    if (!domainsInitRef.current) {
      domainsInitRef.current = true
      setExpandedDomains(new Set(testableDomains.map(d => d.filename)))
    }
    for (const d of testableDomains) {
      if (!loadedDomainsRef.current.has(d.filename)) {
        loadedDomainsRef.current.add(d.filename)
        loadGoldenQuestions(sessionId, d.filename)
      }
    }
  }, [testableDomains, sessionId, loadGoldenQuestions])

  const allTags = Array.from(
    new Set(testableDomains.flatMap(d => d.tags))
  ).sort()

  // Selection helpers — inverted logic: track deselected, everything else is selected
  const makeKey = (domain: string, index: number) => `${domain}:${index}`
  const isSelected = (key: string) => !deselected.has(key)

  const toggleQuestionSelection = (key: string) => {
    setDeselected(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const visibleQuestions = (domain: string): GoldenQuestionResponse[] => {
    const qs = goldenQuestions[domain] ?? []
    if (selectedTags.size === 0) return qs
    return qs.filter(q => q.tags.some(t => selectedTags.has(t)))
  }

  const toggleDomainSelection = (domain: string) => {
    const questions = visibleQuestions(domain)
    const keys = questions.map(q => makeKey(domain, q.index))
    const allSel = keys.length > 0 && keys.every(k => !deselected.has(k))
    setDeselected(prev => {
      const next = new Set(prev)
      if (allSel) keys.forEach(k => next.add(k))
      else keys.forEach(k => next.delete(k))
      return next
    })
  }

  const toggleSet = (key: string, setter: React.Dispatch<React.SetStateAction<Set<string>>>) => {
    setter(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  // Run tests — derive domains from selected questions
  const handleRun = (e2e: boolean) => {
    const domainsToRun = new Set<string>()
    for (const d of testableDomains) {
      const qs = visibleQuestions(d.filename)
      if (qs.some(q => !deselected.has(makeKey(d.filename, q.index)))) {
        domainsToRun.add(d.filename)
      }
    }
    if (domainsToRun.size === 0) return
    useTestStore.setState({
      selectedDomains: domainsToRun,
      selectedTags: new Set(selectedTags),
      includeE2e: e2e,
    })
    runTests(sessionId)
  }

  // Result lookup by domain + question text
  const getQuestionResult = (domain: string, questionText: string): TestQuestionResult | null => {
    if (!results) return null
    const dr = results.domains.find(d => d.domain === domain)
    if (!dr) return null
    return dr.questions.find(qr => qr.question === questionText) ?? null
  }

  if (testableDomains.length === 0 && !loading) {
    if (domainsLoading) {
      return <div className="px-3 py-2"><SkeletonLoader lines={3} /></div>
    }
    return (
      <div className="px-3 py-4 text-sm text-gray-500 dark:text-gray-400">
        No domains with golden questions configured.
      </div>
    )
  }

  const content = (
    <div className={expanded ? 'p-4 space-y-3 text-sm h-full overflow-y-auto' : 'px-3 py-2 space-y-3 text-sm'}>
      {/* Top controls */}
      <div className="flex gap-1.5 items-center">
        <button
          onClick={() => handleRun(false)}
          disabled={loading}
          className="px-2.5 py-1 rounded bg-blue-600 text-white text-xs font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5"
        >
          {loading && !includeE2e && (
            <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          )}
          Run Unit Tests
        </button>
        <button
          onClick={() => handleRun(true)}
          disabled={loading}
          className="px-2.5 py-1 rounded border border-blue-600 text-blue-600 dark:text-blue-400 dark:border-blue-500 text-xs font-medium hover:bg-blue-50 dark:hover:bg-blue-900/20 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5"
        >
          {loading && includeE2e && (
            <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          )}
          Run Integration
        </button>
        <button
          onClick={() => setExpanded(!expanded)}
          className="ml-auto p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
          title={expanded ? 'Collapse' : 'Expand'}
        >
          {expanded ? <XMarkIcon className="w-4 h-4" /> : <ArrowsPointingOutIcon className="w-4 h-4" />}
        </button>
      </div>

      {/* Tag filters */}
      {allTags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {allTags.map(tag => (
            <button
              key={tag}
              onClick={() => toggleTag(tag)}
              className={`px-2 py-0.5 rounded-full text-[10px] border transition-colors ${
                selectedTags.has(tag)
                  ? 'bg-blue-100 border-blue-300 text-blue-800 dark:bg-blue-900 dark:border-blue-700 dark:text-blue-200'
                  : 'bg-gray-100 border-gray-200 text-gray-500 dark:bg-gray-700 dark:border-gray-600 dark:text-gray-400'
              }`}
            >
              {tag}
            </button>
          ))}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="text-red-600 dark:text-red-400 text-xs px-1">{error}</div>
      )}

      {/* Summary bar */}
      {results && (
        <div className="flex items-center gap-3 py-1.5 px-2 rounded bg-gray-50 dark:bg-gray-800">
          <span className="flex items-center gap-1 text-green-600 dark:text-green-400 font-medium text-xs">
            <CheckCircleIcon className="w-4 h-4" />
            {results.total_passed} passed
          </span>
          <span className="flex items-center gap-1 text-red-600 dark:text-red-400 font-medium text-xs">
            <XCircleIcon className="w-4 h-4" />
            {results.total_failed} failed
          </span>
          <button onClick={clearResults} className="ml-auto text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
            Clear
          </button>
        </div>
      )}

      {/* Domain groups */}
      <div className="space-y-1.5">
        {testableDomains.map(d => {
          const questions = visibleQuestions(d.filename)
          const domainExpanded = expandedDomains.has(d.filename)
          const keys = questions.map(q => makeKey(d.filename, q.index))
          const allSel = keys.length > 0 && keys.every(k => !deselected.has(k))
          const someSel = keys.some(k => !deselected.has(k))
          const isAdding = editingQuestion?.domain === d.filename && editingQuestion.index === null

          return (
            <div key={d.filename} className="border rounded dark:border-gray-700">
              {/* Domain header */}
              <div className="flex items-center gap-2 px-2 py-1.5 hover:bg-gray-50 dark:hover:bg-gray-800">
                <button onClick={() => toggleSet(d.filename, setExpandedDomains)} className="flex-shrink-0">
                  {domainExpanded ? (
                    <ChevronDownIcon className="w-3.5 h-3.5 text-gray-400" />
                  ) : (
                    <ChevronRightIcon className="w-3.5 h-3.5 text-gray-400" />
                  )}
                </button>
                <IndeterminateCheckbox
                  checked={allSel}
                  indeterminate={someSel && !allSel}
                  onChange={() => toggleDomainSelection(d.filename)}
                  className="rounded border-gray-300 dark:border-gray-600 flex-shrink-0"
                />
                <span
                  className="font-medium text-gray-800 dark:text-gray-200 cursor-pointer flex-1 truncate"
                  onClick={() => toggleSet(d.filename, setExpandedDomains)}
                >
                  {d.name}
                </span>
                <span className="text-xs text-gray-400 flex-shrink-0">{questions.length} questions</span>
              </div>

              {/* Questions */}
              {domainExpanded && (
                <div className="border-t dark:border-gray-700">
                  {questions.map(q => {
                    const key = makeKey(d.filename, q.index)
                    const isEditing = editingQuestion?.domain === d.filename && editingQuestion.index === q.index
                    const qResult = getQuestionResult(d.filename, q.question)

                    if (isEditing) {
                      return (
                        <QuestionForm
                          key={q.index}
                          initial={q}
                          onSave={body => saveGoldenQuestion(sessionId, d.filename, q.index, body)}
                          onCancel={clearEditing}
                        />
                      )
                    }

                    return (
                      <div key={q.index} className="border-b last:border-b-0 dark:border-gray-700">
                        {/* Question row */}
                        <div
                          className="flex items-center gap-2 px-3 py-1.5 text-xs group cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 relative"
                          onClick={() => toggleSet(key, setExpandedQuestions)}
                        >
                          <input
                            type="checkbox"
                            checked={isSelected(key)}
                            onChange={() => toggleQuestionSelection(key)}
                            onClick={e => e.stopPropagation()}
                            className="rounded border-gray-300 dark:border-gray-600 flex-shrink-0"
                          />
                          {q.tags.map(t => (
                            <span key={t} className="px-1.5 py-0.5 rounded-full bg-gray-100 text-gray-500 dark:bg-gray-700 dark:text-gray-400 text-[10px] flex-shrink-0">{t}</span>
                          ))}
                          <span className="text-gray-700 dark:text-gray-300 truncate flex-1">{q.question}</span>
                          {/* Result badges inline */}
                          {qResult && (
                            <div className="flex gap-1 flex-shrink-0">
                              {qResult.layers.map(lr => (
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
                              {qResult.end_to_end && (
                                <span
                                  className={`px-1.5 py-0.5 rounded text-[10px] ${
                                    qResult.end_to_end.passed
                                      ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                                      : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                                  }`}
                                >
                                  e2e
                                </span>
                              )}
                            </div>
                          )}
                          {/* Action icons — visible on hover */}
                          <button
                            onClick={e => { e.stopPropagation(); setEditingQuestion(d.filename, q.index) }}
                            className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-opacity flex-shrink-0"
                          >
                            <PencilIcon className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={e => {
                              e.stopPropagation()
                              setMovingQuestion(
                                movingQuestion?.domain === d.filename && movingQuestion.index === q.index
                                  ? null
                                  : { domain: d.filename, index: q.index }
                              )
                            }}
                            className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-opacity flex-shrink-0"
                            title="Move to another domain"
                          >
                            <ArrowsRightLeftIcon className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={e => {
                              e.stopPropagation()
                              if (window.confirm(`Delete question: "${q.question}"?`)) {
                                deleteGoldenQuestion(sessionId, d.filename, q.index)
                              }
                            }}
                            className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-600 dark:hover:text-red-400 transition-opacity flex-shrink-0"
                          >
                            <TrashIcon className="w-3.5 h-3.5" />
                          </button>
                          {/* Move dropdown */}
                          {movingQuestion?.domain === d.filename && movingQuestion.index === q.index && (
                            <div className="absolute right-0 top-full z-10 mt-0.5 bg-white dark:bg-gray-800 border dark:border-gray-700 rounded shadow-lg py-1 min-w-[140px]">
                              <div className="px-2 py-0.5 text-[10px] text-gray-400 uppercase">Move to</div>
                              {testableDomains
                                .filter(td => td.filename !== d.filename)
                                .map(td => (
                                  <button
                                    key={td.filename}
                                    onClick={async e => {
                                      e.stopPropagation()
                                      const warnings = await moveGoldenQuestion(sessionId, d.filename, q.index, td.filename, true)
                                      if (warnings.length > 0) {
                                        const ok = window.confirm(`Warning:\n${warnings.join('\n')}\n\nMove anyway?`)
                                        if (!ok) { setMovingQuestion(null); return }
                                      }
                                      await moveGoldenQuestion(sessionId, d.filename, q.index, td.filename)
                                      setMovingQuestion(null)
                                    }}
                                    className="w-full text-left px-2 py-1 text-xs text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                                  >
                                    {td.name}
                                  </button>
                                ))
                              }
                            </div>
                          )}
                        </div>

                        {/* Expanded detail: definition + test results */}
                        {expandedQuestions.has(key) && (
                          <div className="px-3 py-2 bg-gray-50 dark:bg-gray-800/50 space-y-2 text-xs border-t dark:border-gray-700">
                            {/* Definition */}
                            <div className="space-y-1">
                              {(q.expect.terms ?? []).length > 0 && (
                                <div className="flex gap-1 items-center flex-wrap">
                                  <span className="text-[10px] font-medium text-gray-500 uppercase">Terms</span>
                                  {(q.expect.terms ?? []).map((t, i) => (
                                    <span key={i} className="px-1.5 py-0.5 rounded bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300 text-[10px]">
                                      {dn(String(t.name))}{t.has_definition ? ' +def' : ''}
                                    </span>
                                  ))}
                                </div>
                              )}
                              {(q.expect.grounding ?? []).length > 0 && (
                                <div className="flex gap-1 items-center flex-wrap">
                                  <span className="text-[10px] font-medium text-gray-500 uppercase">Grounding</span>
                                  {(q.expect.grounding ?? []).map((g, i) => (
                                    <span key={i} className="px-1.5 py-0.5 rounded bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300 text-[10px]">
                                      {dn(String(g.entity))} &rarr; {String(g.resolves_to)}
                                    </span>
                                  ))}
                                </div>
                              )}
                              {(q.expect.relationships ?? []).length > 0 && (
                                <div className="flex gap-1 items-center flex-wrap">
                                  <span className="text-[10px] font-medium text-gray-500 uppercase">Relationships</span>
                                  {(q.expect.relationships ?? []).map((r, i) => (
                                    <span key={i} className="px-1.5 py-0.5 rounded bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300 text-[10px]">
                                      {dn(String(r.subject))} {String(r.verb)} {dn(String(r.object))}
                                    </span>
                                  ))}
                                </div>
                              )}
                              {q.expect.end_to_end && (String(q.expect.end_to_end.judge_prompt ?? '').trim() || String(q.expect.end_to_end.validator_code ?? '').trim()) && (
                                <div className="space-y-0.5">
                                  {String(q.expect.end_to_end.judge_prompt ?? '').trim() && (
                                    <div>
                                      <span className="text-[10px] font-medium text-gray-500 uppercase">Judge prompt</span>
                                      <div className="text-gray-600 dark:text-gray-400 pl-2 line-clamp-2">{String(q.expect.end_to_end.judge_prompt)}</div>
                                    </div>
                                  )}
                                  {String(q.expect.end_to_end.validator_code ?? '').trim() && (
                                    <div>
                                      <span className="text-[10px] font-medium text-gray-500 uppercase">Validator</span>
                                      <pre className="text-gray-600 dark:text-gray-400 pl-2 line-clamp-2 font-mono text-[10px]">{String(q.expect.end_to_end.validator_code)}</pre>
                                    </div>
                                  )}
                                </div>
                              )}
                              {(q.expect.expected_outputs ?? []).length > 0 && (
                                <div className="flex gap-1 items-center flex-wrap">
                                  <span className="text-[10px] font-medium text-gray-500 uppercase">Expected outputs</span>
                                  {(q.expect.expected_outputs ?? []).map((o, i) => (
                                    <span key={i} className="px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-[10px]">
                                      {o.name}
                                      <span className="text-blue-500 dark:text-blue-400 ml-0.5">({o.type})</span>
                                      {o.type === 'table' && o.columns.length > 0 && (
                                        <span className="text-blue-400 dark:text-blue-500 ml-0.5">[{o.columns.join(', ')}]</span>
                                      )}
                                    </span>
                                  ))}
                                </div>
                              )}
                              {(q.expect.terms ?? []).length === 0 && (q.expect.grounding ?? []).length === 0 && (q.expect.relationships ?? []).length === 0 && (q.expect.expected_outputs ?? []).length === 0 && !q.expect.end_to_end && (
                                <div className="text-gray-400 italic">No expectations defined</div>
                              )}
                            </div>

                            {/* Test results */}
                            {qResult && (
                              <div className="border-t dark:border-gray-700 pt-1.5 space-y-1">
                                <div className="flex items-center gap-1">
                                  {qResult.passed ? (
                                    <CheckCircleIcon className="w-3.5 h-3.5 text-green-500" />
                                  ) : (
                                    <XCircleIcon className="w-3.5 h-3.5 text-red-500" />
                                  )}
                                  <span className="text-[10px] font-medium text-gray-500 uppercase">Test Results</span>
                                </div>
                                {qResult.layers.map(lr =>
                                  lr.failures.length > 0 ? (
                                    <div key={lr.layer}>
                                      <div className="text-[10px] font-medium text-gray-500 uppercase">{lr.layer}</div>
                                      {lr.failures.map((f, fi) => (
                                        <div key={fi} className="text-red-600 dark:text-red-400 pl-2">{f}</div>
                                      ))}
                                    </div>
                                  ) : null
                                )}
                                {qResult.end_to_end && (
                                  <div className="border-t dark:border-gray-700 pt-1 mt-1">
                                    <div className="flex items-center gap-2">
                                      <span className="text-[10px] font-medium text-gray-500 uppercase">Integration</span>
                                      {qResult.end_to_end.passed ? (
                                        <CheckCircleIcon className="w-3.5 h-3.5 text-green-500" />
                                      ) : (
                                        <XCircleIcon className="w-3.5 h-3.5 text-red-500" />
                                      )}
                                      <span className="text-[10px] text-gray-400">{qResult.end_to_end.duration_s.toFixed(1)}s</span>
                                    </div>
                                    {qResult.end_to_end.failures.map((f, fi) => (
                                      <div key={fi} className="text-red-600 dark:text-red-400 pl-2">{f}</div>
                                    ))}
                                    {qResult.end_to_end.answer && (
                                      <div className="mt-1">
                                        <span className="text-[10px] font-medium text-gray-500 uppercase">Answer</span>
                                        <div className="text-gray-600 dark:text-gray-400 pl-2 line-clamp-3">{qResult.end_to_end.answer}</div>
                                      </div>
                                    )}
                                    {qResult.end_to_end.judge_reasoning && (
                                      <div className="mt-1">
                                        <span className="text-[10px] font-medium text-gray-500 uppercase">Judge</span>
                                        <div className="text-gray-600 dark:text-gray-400 pl-2">{qResult.end_to_end.judge_reasoning}</div>
                                      </div>
                                    )}
                                  </div>
                                )}
                                {qResult.layers.every(lr => lr.failures.length === 0) && !qResult.end_to_end && (
                                  <div className="text-green-600 dark:text-green-400">All assertions passed.</div>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )
                  })}

                  {/* Add question */}
                  {isAdding ? (
                    <QuestionForm
                      onSave={body => saveGoldenQuestion(sessionId, d.filename, null, body)}
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

      {/* Live progress */}
      {loading && progress && (
        <div className="p-2 rounded border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 space-y-1.5">
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
            <div className="space-y-0.5 pl-4">
              <div className="flex items-center gap-1.5">
                <svg className="animate-spin h-3 w-3 text-blue-500" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <span className="text-[10px] uppercase font-medium text-blue-600 dark:text-blue-400">
                  {progress.phase === 'e2e' ? 'Integration test (LLM)' : 'Unit tests'}
                </span>
              </div>
              {progress.detail && (
                <div className="text-[10px] text-gray-500 dark:text-gray-400 pl-[18px] truncate">
                  {progress.detail}
                </div>
              )}
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
  )

  if (expanded) {
    return (
      <>
        <div className="px-3 py-2 text-xs text-gray-400 dark:text-gray-500">
          <button onClick={() => setExpanded(false)} className="hover:text-gray-600 dark:hover:text-gray-300">
            Viewing in expanded mode. Click to collapse.
          </button>
        </div>
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setExpanded(false)}>
          <div
            className="bg-white dark:bg-gray-900 rounded-lg shadow-xl border dark:border-gray-700 flex flex-col"
            style={{ width: '80%', height: '80%' }}
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-2 border-b dark:border-gray-700">
              <h2 className="text-sm font-semibold text-gray-800 dark:text-gray-200">Regression Tests</h2>
              <button onClick={() => setExpanded(false)} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto">
              {content}
            </div>
          </div>
        </div>
      </>
    )
  }

  return content
}
