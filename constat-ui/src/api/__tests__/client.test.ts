import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// Mock authStore before importing client
vi.mock('@/store/authStore', () => ({
  isAuthDisabled: true,
  useAuthStore: {
    getState: () => ({
      getToken: vi.fn().mockResolvedValue(null),
      logout: vi.fn(),
    }),
  },
}))

import { ApiError, get, post, put, patch, del } from '../client'

describe('ApiError', () => {
  it('constructs with status, statusText, and optional data', () => {
    const error = new ApiError(404, 'Not Found', { detail: 'missing' })
    expect(error.status).toBe(404)
    expect(error.statusText).toBe('Not Found')
    expect(error.data).toEqual({ detail: 'missing' })
    expect(error.name).toBe('ApiError')
    expect(error.message).toBe('API Error: 404 Not Found')
  })

  it('works without data', () => {
    const error = new ApiError(500, 'Internal Server Error')
    expect(error.data).toBeUndefined()
  })

  it('is an instance of Error', () => {
    const error = new ApiError(400, 'Bad Request')
    expect(error).toBeInstanceOf(Error)
  })
})

describe('API client methods', () => {
  const mockFetch = vi.fn()

  beforeEach(() => {
    vi.stubGlobal('fetch', mockFetch)
    mockFetch.mockReset()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  function jsonResponse(data: unknown, status = 200) {
    return {
      ok: status >= 200 && status < 300,
      status,
      statusText: status === 200 ? 'OK' : 'Error',
      json: () => Promise.resolve(data),
    }
  }

  function errorResponse(status: number, statusText: string, data: unknown = {}) {
    return {
      ok: false,
      status,
      statusText,
      json: () => Promise.resolve(data),
    }
  }

  describe('get', () => {
    it('calls fetch with GET and correct URL', async () => {
      mockFetch.mockResolvedValue(jsonResponse({ result: 1 }))
      const result = await get('/sessions')
      expect(mockFetch).toHaveBeenCalledWith('/api/sessions', {
        method: 'GET',
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
      })
      expect(result).toEqual({ result: 1 })
    })
  })

  describe('post', () => {
    it('calls fetch with POST and JSON body', async () => {
      mockFetch.mockResolvedValue(jsonResponse({ id: 'abc' }))
      const result = await post('/sessions', { name: 'test' })
      expect(mockFetch).toHaveBeenCalledWith('/api/sessions', {
        method: 'POST',
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ name: 'test' }),
      })
      expect(result).toEqual({ id: 'abc' })
    })

    it('sends undefined body when no body provided', async () => {
      mockFetch.mockResolvedValue(jsonResponse({ ok: true }))
      await post('/sessions')
      expect(mockFetch).toHaveBeenCalledWith('/api/sessions', {
        method: 'POST',
        headers: expect.any(Object),
        body: undefined,
      })
    })
  })

  describe('put', () => {
    it('calls fetch with PUT and JSON body', async () => {
      mockFetch.mockResolvedValue(jsonResponse({ updated: true }))
      await put('/sessions/1', { name: 'updated' })
      expect(mockFetch).toHaveBeenCalledWith('/api/sessions/1', {
        method: 'PUT',
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ name: 'updated' }),
      })
    })
  })

  describe('patch', () => {
    it('calls fetch with PATCH and JSON body', async () => {
      mockFetch.mockResolvedValue(jsonResponse({ patched: true }))
      await patch('/sessions/1', { status: 'done' })
      expect(mockFetch).toHaveBeenCalledWith('/api/sessions/1', {
        method: 'PATCH',
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ status: 'done' }),
      })
    })
  })

  describe('del', () => {
    it('calls fetch with DELETE', async () => {
      mockFetch.mockResolvedValue(jsonResponse({ deleted: true }))
      await del('/sessions/1')
      expect(mockFetch).toHaveBeenCalledWith('/api/sessions/1', {
        method: 'DELETE',
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
      })
    })
  })

  describe('204 No Content', () => {
    it('returns undefined for 204 responses', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        status: 204,
        statusText: 'No Content',
        json: () => Promise.reject(new Error('no body')),
      })
      const result = await del('/sessions/1')
      expect(result).toBeUndefined()
    })
  })

  describe('error handling', () => {
    it('throws ApiError on non-ok response', async () => {
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: () => Promise.resolve({ detail: 'invalid' }),
      })
      await expect(get('/bad')).rejects.toThrow(ApiError)
      await expect(get('/bad')).rejects.toMatchObject({
        status: 400,
        statusText: 'Bad Request',
        data: { detail: 'invalid' },
      })
    })

    it('throws ApiError when error body is not JSON', async () => {
      // Use 400 (non-retryable) to avoid retry/timer complexity
      mockFetch.mockResolvedValue({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: () => Promise.reject(new Error('not json')),
      })
      const err = await get('/fail').catch((e: unknown) => e) as ApiError
      expect(err).toBeInstanceOf(ApiError)
      expect(err.status).toBe(400)
      expect(err.data).toBeUndefined()
    })
  })

  describe('retry logic', () => {
    it('retries on 500 and succeeds on second attempt', async () => {
      // First call fails with 500, second succeeds
      // The sleep delay is real but short enough with only 1 retry needed (1s)
      // Use a targeted approach: just verify the fetch was called twice
      mockFetch
        .mockResolvedValueOnce(errorResponse(500, 'Internal Server Error'))
        .mockResolvedValueOnce(jsonResponse({ ok: true }))

      const result = await get('/flaky')
      expect(result).toEqual({ ok: true })
      expect(mockFetch).toHaveBeenCalledTimes(2)
    }, 10000)

    it('retries on 502 and succeeds on second attempt', async () => {
      mockFetch
        .mockResolvedValueOnce(errorResponse(502, 'Bad Gateway'))
        .mockResolvedValueOnce(jsonResponse({ ok: true }))

      const result = await get('/flaky')
      expect(result).toEqual({ ok: true })
      expect(mockFetch).toHaveBeenCalledTimes(2)
    }, 10000)

    it('does not retry on 400', async () => {
      mockFetch.mockResolvedValue(errorResponse(400, 'Bad Request', { detail: 'bad' }))
      await expect(get('/bad')).rejects.toThrow(ApiError)
      expect(mockFetch).toHaveBeenCalledTimes(1)
    })

    it('does not retry on 401', async () => {
      mockFetch.mockResolvedValue(errorResponse(401, 'Unauthorized'))
      await expect(get('/unauth')).rejects.toThrow(ApiError)
      expect(mockFetch).toHaveBeenCalledTimes(1)
    })

    it('does not retry on 403', async () => {
      mockFetch.mockResolvedValue(errorResponse(403, 'Forbidden'))
      await expect(get('/forbidden')).rejects.toThrow(ApiError)
      expect(mockFetch).toHaveBeenCalledTimes(1)
    })

    it('does not retry on 404', async () => {
      mockFetch.mockResolvedValue(errorResponse(404, 'Not Found'))
      await expect(get('/missing')).rejects.toThrow(ApiError)
      expect(mockFetch).toHaveBeenCalledTimes(1)
    })
  })

  describe('401 logout', () => {
    it('does not trigger logout when auth is disabled', async () => {
      const { useAuthStore } = await import('@/store/authStore')
      const logoutFn = useAuthStore.getState().logout as ReturnType<typeof vi.fn>
      logoutFn.mockClear()

      mockFetch.mockResolvedValue(errorResponse(401, 'Unauthorized'))
      await expect(get('/protected')).rejects.toThrow(ApiError)
      expect(logoutFn).not.toHaveBeenCalled()
    })
  })
})
