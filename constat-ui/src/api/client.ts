// API Client - base fetch wrapper with auth support

import { useAuthStore, isAuthDisabled } from '@/store/authStore'

const API_BASE = '/api'

export class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public data?: unknown
  ) {
    super(`API Error: ${status} ${statusText}`)
    this.name = 'ApiError'
  }
}

async function getAuthHeaders(): Promise<Record<string, string>> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  }

  // Add auth token if auth is enabled
  if (!isAuthDisabled) {
    const token = await useAuthStore.getState().getToken()
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }
  }

  return headers
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    // Handle 401 by triggering logout
    if (response.status === 401 && !isAuthDisabled) {
      useAuthStore.getState().logout()
    }
    let data: unknown
    try {
      data = await response.json()
    } catch {
      // Response body not JSON
    }
    throw new ApiError(response.status, response.statusText, data)
  }
  return response.json()
}

export async function get<T>(path: string): Promise<T> {
  const headers = await getAuthHeaders()
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'GET',
    headers,
  })
  return handleResponse<T>(response)
}

export async function post<T>(path: string, body?: unknown): Promise<T> {
  const headers = await getAuthHeaders()
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers,
    body: body ? JSON.stringify(body) : undefined,
  })
  return handleResponse<T>(response)
}

export async function put<T>(path: string, body: unknown): Promise<T> {
  const headers = await getAuthHeaders()
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'PUT',
    headers,
    body: JSON.stringify(body),
  })
  return handleResponse<T>(response)
}

export async function del<T>(path: string): Promise<T> {
  const headers = await getAuthHeaders()
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'DELETE',
    headers,
  })
  return handleResponse<T>(response)
}

export async function uploadFile(
  path: string,
  file: File
): Promise<unknown> {
  const formData = new FormData()
  formData.append('file', file)

  // For file uploads, don't set Content-Type (browser sets it with boundary)
  const headers: Record<string, string> = {}
  if (!isAuthDisabled) {
    const token = await useAuthStore.getState().getToken()
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }
  }

  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers,
    body: formData,
  })
  return handleResponse(response)
}

export async function uploadFiles(
  path: string,
  files: File[]
): Promise<unknown> {
  const formData = new FormData()
  files.forEach((file) => {
    formData.append('files', file)
  })

  // For file uploads, don't set Content-Type (browser sets it with boundary)
  const headers: Record<string, string> = {}
  if (!isAuthDisabled) {
    const token = await useAuthStore.getState().getToken()
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }
  }

  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers,
    body: formData,
  })
  return handleResponse(response)
}