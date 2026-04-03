import { describe, it, expect, vi, beforeEach } from 'vitest'

const mockGetIdToken = vi.fn()

vi.mock('@/config/firebase', () => ({
  isAuthDisabled: false,
  getIdToken: (...args: unknown[]) => mockGetIdToken(...args),
}))

describe('auth-helpers', () => {
  beforeEach(() => {
    mockGetIdToken.mockReset()
  })

  describe('getToken', () => {
    it('returns token from getIdToken when auth enabled', async () => {
      mockGetIdToken.mockResolvedValue('tok-abc')
      const { getToken } = await import('../auth-helpers')
      const token = await getToken()
      expect(token).toBe('tok-abc')
    })

    it('returns null when getIdToken returns null', async () => {
      mockGetIdToken.mockResolvedValue(null)
      const { getToken } = await import('../auth-helpers')
      const token = await getToken()
      expect(token).toBeNull()
    })
  })

  describe('getAuthHeaders', () => {
    it('returns Authorization header when token available', async () => {
      mockGetIdToken.mockResolvedValue('tok-xyz')
      const { getAuthHeaders } = await import('../auth-helpers')
      const headers = await getAuthHeaders()
      expect(headers).toEqual({ Authorization: 'Bearer tok-xyz' })
    })

    it('returns empty headers when no token', async () => {
      mockGetIdToken.mockResolvedValue(null)
      const { getAuthHeaders } = await import('../auth-helpers')
      const headers = await getAuthHeaders()
      expect(headers).toEqual({})
    })
  })

  describe('isAuthDisabled re-export', () => {
    it('re-exports isAuthDisabled from firebase', async () => {
      const { isAuthDisabled } = await import('../auth-helpers')
      expect(typeof isAuthDisabled).toBe('boolean')
    })
  })
})

describe('auth-helpers (auth disabled)', () => {
  beforeEach(() => {
    vi.resetModules()
  })

  it('getToken returns null when auth disabled', async () => {
    vi.doMock('@/config/firebase', () => ({
      isAuthDisabled: true,
      getIdToken: vi.fn(),
    }))
    const { getToken } = await import('../auth-helpers')
    const token = await getToken()
    expect(token).toBeNull()
  })

  it('getAuthHeaders returns empty when auth disabled', async () => {
    vi.doMock('@/config/firebase', () => ({
      isAuthDisabled: true,
      getIdToken: vi.fn(),
    }))
    const { getAuthHeaders } = await import('../auth-helpers')
    const headers = await getAuthHeaders()
    expect(headers).toEqual({})
  })
})
