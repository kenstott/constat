// API Client - base fetch wrapper with auth support

import {useAuthStore, isAuthDisabled} from '@/store/authStore'

const API_BASE = '/api'

// Retry configuration
const MAX_RETRIES = 3
const INITIAL_RETRY_DELAY_MS = 1000
const RETRYABLE_STATUS_CODES = [500, 502, 503, 504]

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
            console.log('[API] Auth token attached (length:', token.length, ')')
        } else {
            console.warn('[API] Auth enabled but no token available')
        }
    }

    return headers
}

function isRetryableError(error: unknown): boolean {
    // Network errors (fetch failed)
    if (error instanceof TypeError && error.message.includes('fetch')) {
        return true
    }
    // Server errors (5xx)
    if (error instanceof ApiError && RETRYABLE_STATUS_CODES.includes(error.status)) {
        return true
    }
    return false
}

async function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
}

async function fetchWithRetry<T>(
    fetchFn: () => Promise<Response>,
    handleFn: (response: Response) => Promise<T>,
    retries = MAX_RETRIES
): Promise<T> {
    let lastError: unknown
    for (let attempt = 0; attempt <= retries; attempt++) {
        try {
            const response = await fetchFn()
            return await handleFn(response)
        } catch (error) {
            lastError = error
            if (attempt < retries && isRetryableError(error)) {
                const delay = INITIAL_RETRY_DELAY_MS * Math.pow(2, attempt)
                console.log(`[API] Request failed, retrying in ${delay}ms (attempt ${attempt + 1}/${retries})`, error)
                await sleep(delay)
            } else {
                throw error
            }
        }
    }
    throw lastError
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
    return fetchWithRetry(
        () => fetch(`${API_BASE}${path}`, { method: 'GET', headers }),
        handleResponse<T>
    )
}

export async function post<T>(path: string, body?: unknown): Promise<T> {
    const headers = await getAuthHeaders()
    return fetchWithRetry(
        () => fetch(`${API_BASE}${path}`, {
            method: 'POST',
            headers,
            body: body ? JSON.stringify(body) : undefined,
        }),
        handleResponse<T>
    )
}

export async function put<T>(path: string, body: unknown): Promise<T> {
    const headers = await getAuthHeaders()
    return fetchWithRetry(
        () => fetch(`${API_BASE}${path}`, {
            method: 'PUT',
            headers,
            body: JSON.stringify(body),
        }),
        handleResponse<T>
    )
}

export async function del<T>(path: string): Promise<T> {
    const headers = await getAuthHeaders()
    return fetchWithRetry(
        () => fetch(`${API_BASE}${path}`, { method: 'DELETE', headers }),
        handleResponse<T>
    )
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

    return fetchWithRetry(
        () => fetch(`${API_BASE}${path}`, {
            method: 'POST',
            headers,
            body: formData,
        }),
        handleResponse
    )
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

    return fetchWithRetry(
        () => fetch(`${API_BASE}${path}`, {
            method: 'POST',
            headers,
            body: formData,
        }),
        handleResponse
    )
}
