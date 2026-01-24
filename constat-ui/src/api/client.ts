// API Client - base fetch wrapper

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

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
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
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  })
  return handleResponse<T>(response)
}

export async function post<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: body ? JSON.stringify(body) : undefined,
  })
  return handleResponse<T>(response)
}

export async function put<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  })
  return handleResponse<T>(response)
}

export async function del<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'DELETE',
    headers: {
      'Content-Type': 'application/json',
    },
  })
  return handleResponse<T>(response)
}

export async function uploadFile(
  path: string,
  file: File
): Promise<unknown> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    body: formData,
  })
  return handleResponse(response)
}