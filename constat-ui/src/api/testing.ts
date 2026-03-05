import { get, post, put, del } from '@/api/client'
import type { GoldenQuestionRequest, GoldenQuestionResponse, TestableDomainInfo, TestRunResponse } from '@/types/api'

export async function listTestableDomains(sessionId: string): Promise<TestableDomainInfo[]> {
  return get<TestableDomainInfo[]>(`/sessions/${sessionId}/tests/domains`)
}

export async function runTests(
  sessionId: string,
  domains: string[] = [],
  tags: string[] = [],
  includeE2e: boolean = false,
): Promise<TestRunResponse> {
  return post<TestRunResponse>(`/sessions/${sessionId}/tests/run`, { domains, tags, include_e2e: includeE2e })
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
