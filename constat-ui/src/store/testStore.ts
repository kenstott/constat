import { create } from 'zustand'
import type { GoldenQuestionRequest, GoldenQuestionResponse, TestableDomainInfo, TestRunResponse } from '@/types/api'
import * as testingApi from '@/api/testing'
import type { TestProgressEvent } from '@/api/testing'

interface TestProgress {
  domain: string
  domainName: string
  question: string
  questionIndex: number
  questionTotal: number
  phase: string  // "metadata" | "e2e" | ""
}

interface TestState {
  testableDomains: TestableDomainInfo[]
  results: TestRunResponse | null
  loading: boolean
  error: string | null
  progress: TestProgress | null
  selectedDomains: Set<string>
  selectedTags: Set<string>
  includeE2e: boolean

  // Golden question CRUD
  goldenQuestions: Record<string, GoldenQuestionResponse[]>
  editingQuestion: { domain: string; index: number | null } | null

  loadTestableDomains: (sessionId: string) => Promise<void>
  runTests: (sessionId: string) => Promise<void>
  toggleDomain: (filename: string) => void
  toggleTag: (tag: string) => void
  setIncludeE2e: (value: boolean) => void
  clearResults: () => void

  loadGoldenQuestions: (sessionId: string, domain: string) => Promise<void>
  saveGoldenQuestion: (sessionId: string, domain: string, index: number | null, body: GoldenQuestionRequest) => Promise<void>
  deleteGoldenQuestion: (sessionId: string, domain: string, index: number) => Promise<void>
  setEditingQuestion: (domain: string, index: number | null) => void
  clearEditing: () => void
}

export const useTestStore = create<TestState>((set, get) => ({
  testableDomains: [],
  results: null,
  loading: false,
  error: null,
  progress: null,
  selectedDomains: new Set(),
  selectedTags: new Set(),
  includeE2e: false,
  goldenQuestions: {},
  editingQuestion: null,

  loadTestableDomains: async (sessionId: string) => {
    try {
      const domains = await testingApi.listTestableDomains(sessionId)
      set({ testableDomains: domains })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) })
    }
  },

  runTests: async (sessionId: string) => {
    const { selectedDomains, selectedTags, includeE2e } = get()
    set({ loading: true, error: null, progress: null, results: null })
    try {
      const results = await testingApi.runTestsStreaming(
        sessionId,
        [...selectedDomains],
        [...selectedTags],
        includeE2e,
        (evt: TestProgressEvent) => {
          if (evt.event === 'domain_start' || evt.event === 'question_start') {
            set({
              progress: {
                domain: evt.domain,
                domainName: evt.domain_name,
                question: evt.question,
                questionIndex: evt.question_index,
                questionTotal: evt.question_total,
                phase: evt.phase,
              },
            })
          }
        },
      )
      set({ results, loading: false, progress: null })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e), loading: false, progress: null })
    }
  },

  toggleDomain: (filename: string) => {
    const next = new Set(get().selectedDomains)
    if (next.has(filename)) next.delete(filename)
    else next.add(filename)
    set({ selectedDomains: next })
  },

  toggleTag: (tag: string) => {
    const next = new Set(get().selectedTags)
    if (next.has(tag)) next.delete(tag)
    else next.add(tag)
    set({ selectedTags: next })
  },

  setIncludeE2e: (value: boolean) => set({ includeE2e: value }),

  clearResults: () => set({ results: null, error: null, progress: null }),

  loadGoldenQuestions: async (sessionId: string, domain: string) => {
    try {
      const questions = await testingApi.listGoldenQuestions(sessionId, domain)
      set({ goldenQuestions: { ...get().goldenQuestions, [domain]: questions } })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) })
    }
  },

  saveGoldenQuestion: async (sessionId: string, domain: string, index: number | null, body: GoldenQuestionRequest) => {
    try {
      if (index === null) {
        await testingApi.createGoldenQuestion(sessionId, domain, body)
      } else {
        await testingApi.updateGoldenQuestion(sessionId, domain, index, body)
      }
      // Reload questions and testable domains to reflect changes
      await get().loadGoldenQuestions(sessionId, domain)
      await get().loadTestableDomains(sessionId)
      set({ editingQuestion: null })
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) })
    }
  },

  deleteGoldenQuestion: async (sessionId: string, domain: string, index: number) => {
    try {
      await testingApi.deleteGoldenQuestion(sessionId, domain, index)
      await get().loadGoldenQuestions(sessionId, domain)
      await get().loadTestableDomains(sessionId)
    } catch (e) {
      set({ error: e instanceof Error ? e.message : String(e) })
    }
  },

  setEditingQuestion: (domain: string, index: number | null) => {
    set({ editingQuestion: { domain, index } })
  },

  clearEditing: () => set({ editingQuestion: null }),
}))
