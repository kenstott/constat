import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

// Mock auth-helpers
vi.mock('@/config/auth-helpers', () => ({
  isAuthDisabled: false,
  getAuthHeaders: vi.fn().mockResolvedValue({ Authorization: 'Bearer test-tok' }),
  getToken: vi.fn().mockResolvedValue('test-tok'),
}))

import { uploadFile, uploadFiles, get, ApiError } from '../client'

describe('API client coverage', () => {
  const mockFetch = vi.fn()

  beforeEach(() => {
    vi.stubGlobal('fetch', mockFetch)
    mockFetch.mockReset()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  describe('isRetryableError - network error (line 40-41)', () => {
    it('retries on TypeError with fetch in message', async () => {
      mockFetch
        .mockRejectedValueOnce(new TypeError('Failed to fetch'))
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: () => Promise.resolve({ ok: true }),
        })

      const result = await get('/retry-test')
      expect(result).toEqual({ ok: true })
      expect(mockFetch).toHaveBeenCalledTimes(2)
    }, 10000)
  })

  describe('fetchWithRetry exhausts retries (line 75)', () => {
    it('throws last error after all retries exhausted', async () => {
      // All attempts fail with retryable 500
      mockFetch.mockResolvedValue({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        json: () => Promise.resolve({ detail: 'server down' }),
      })

      await expect(get('/always-fail')).rejects.toThrow(ApiError)
      expect(mockFetch).toHaveBeenCalledTimes(4) // 1 initial + 3 retries
    }, 30000)
  })

  describe('handleResponse 401 warning (line 81-82)', () => {
    it('logs warning on 401 when auth is enabled', async () => {
      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
      mockFetch.mockResolvedValue({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        json: () => Promise.resolve({}),
      })

      await expect(get('/protected')).rejects.toThrow(ApiError)
      expect(warnSpy).toHaveBeenCalledWith('[API] 401 Unauthorized')
      warnSpy.mockRestore()
    })
  })

  describe('uploadFile (lines 155-169)', () => {
    it('uploads single file with FormData', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ uploaded: true }),
      })

      const file = new File(['content'], 'test.csv', { type: 'text/csv' })
      const result = await uploadFile('/upload', file)
      expect(result).toEqual({ uploaded: true })

      const [url, options] = mockFetch.mock.calls[0]
      expect(url).toBe('/api/upload')
      expect(options.method).toBe('POST')
      expect(options.body).toBeInstanceOf(FormData)
      // Should NOT have Content-Type (browser sets it with boundary)
      expect(options.headers?.['Content-Type']).toBeUndefined()
    })
  })

  describe('uploadFiles (lines 171-191)', () => {
    it('uploads multiple files with FormData', async () => {
      mockFetch.mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ count: 2 }),
      })

      const files = [
        new File(['a'], 'a.csv', { type: 'text/csv' }),
        new File(['b'], 'b.csv', { type: 'text/csv' }),
      ]
      const result = await uploadFiles('/upload-many', files)
      expect(result).toEqual({ count: 2 })

      const [url, options] = mockFetch.mock.calls[0]
      expect(url).toBe('/api/upload-many')
      expect(options.method).toBe('POST')
      expect(options.body).toBeInstanceOf(FormData)
    })
  })
})
