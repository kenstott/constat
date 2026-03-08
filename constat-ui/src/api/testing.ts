import { get, post, put, del } from '@/api/client'
import type { GoldenQuestionExpectations, GoldenQuestionRequest, GoldenQuestionResponse, TestableDomainInfo, TestRunResponse } from '@/types/api'
import { useAuthStore, isAuthDisabled } from '@/store/authStore'

export async function listTestableDomains(sessionId: string): Promise<TestableDomainInfo[]> {
  return get<TestableDomainInfo[]>(`/sessions/${sessionId}/tests/domains`)
}

export interface TestProgressEvent {
  event: 'domain_start' | 'question_start' | 'question_done' | 'domain_done' | 'complete'
  domain: string
  domain_name: string
  question: string
  question_index: number
  question_total: number
  phase: string
  result?: Record<string, unknown>
}

export async function runTestsStreaming(
  sessionId: string,
  domains: string[] = [],
  tags: string[] = [],
  includeE2e: boolean = false,
  onEvent: (event: TestProgressEvent) => void,
): Promise<TestRunResponse> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (!isAuthDisabled) {
    const token = await useAuthStore.getState().getToken()
    if (token) headers['Authorization'] = `Bearer ${token}`
  }

  const response = await fetch(`/api/sessions/${sessionId}/tests/run`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ domains, tags, include_e2e: includeE2e }),
  })

  if (!response.ok) {
    throw new Error(`Test run failed: ${response.status} ${response.statusText}`)
  }

  const reader = response.body?.getReader()
  if (!reader) throw new Error('No response body')

  const decoder = new TextDecoder()
  let buffer = ''
  let finalResult: TestRunResponse | null = null

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })

    // Parse SSE lines
    const lines = buffer.split('\n')
    buffer = lines.pop() ?? '' // keep incomplete line

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const jsonStr = line.slice(6)
      try {
        const evt = JSON.parse(jsonStr) as TestProgressEvent
        if (evt.event === 'complete') {
          finalResult = (evt as unknown as { result: TestRunResponse }).result
        }
        onEvent(evt)
      } catch {
        // skip malformed lines
      }
    }
  }

  if (!finalResult) throw new Error('Stream ended without complete event')
  return finalResult
}

export async function listGoldenQuestions(
  sessionId: string,
  domain: string,
): Promise<GoldenQuestionResponse[]> {
  return get<GoldenQuestionResponse[]>(`/sessions/${sessionId}/tests/${domain}/questions`)
}

export async function createGoldenQuestion(
  sessionId: string,
  domain: string,
  body: GoldenQuestionRequest,
): Promise<GoldenQuestionResponse> {
  return post<GoldenQuestionResponse>(`/sessions/${sessionId}/tests/${domain}/questions`, body)
}

export async function updateGoldenQuestion(
  sessionId: string,
  domain: string,
  index: number,
  body: GoldenQuestionRequest,
): Promise<GoldenQuestionResponse> {
  return put<GoldenQuestionResponse>(`/sessions/${sessionId}/tests/${domain}/questions/${index}`, body)
}

export async function deleteGoldenQuestion(
  sessionId: string,
  domain: string,
  index: number,
): Promise<void> {
  return del<void>(`/sessions/${sessionId}/tests/${domain}/questions/${index}`)
}

export interface ProofNodeInput {
  id: string
  name: string
  source?: string
  source_name?: string
  table_name?: string
  api_endpoint?: string
  status: string
}

export async function extractExpectations(
  sessionId: string,
  proofNodes?: ProofNodeInput[],
  originalQuestion?: string,
  proofSummary?: string,
): Promise<GoldenQuestionExpectations> {
  return post<GoldenQuestionExpectations>(
    `/sessions/${sessionId}/tests/expectations`,
    { proof_nodes: proofNodes ?? [], original_question: originalQuestion, proof_summary: proofSummary },
  )
}
