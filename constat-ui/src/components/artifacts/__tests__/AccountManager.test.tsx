import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { AccountManager } from '../AccountManager'
import type { AccountSummary } from '../AccountManager'

const mockAccounts: AccountSummary[] = [
  {
    name: 'email-google-ken',
    type: 'imap',
    provider: 'google',
    display_name: 'Gmail (ken@gmail.com)',
    email: 'ken@gmail.com',
    active: true,
    created_at: '2026-03-24T10:00:00Z',
  },
  {
    name: 'drive-google-ken',
    type: 'drive',
    provider: 'google',
    display_name: 'Google Drive (Analytics)',
    email: 'ken@gmail.com',
    active: true,
    created_at: '2026-03-24T10:05:00Z',
  },
  {
    name: 'calendar-microsoft-ken',
    type: 'calendar',
    provider: 'microsoft',
    display_name: 'Outlook Calendar',
    email: 'ken@company.com',
    active: false,
    created_at: '2026-03-24T10:10:00Z',
  },
]

describe('AccountManager', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
  }

  beforeEach(() => {
    vi.restoreAllMocks()
    defaultProps.onClose = vi.fn()
    // Default mock: return account list
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => mockAccounts,
    } as Response)
  })

  it('renders account list from API', async () => {
    render(<AccountManager {...defaultProps} />)

    await waitFor(() => {
      expect(screen.getByText('Gmail (ken@gmail.com)')).toBeInTheDocument()
    })

    expect(screen.getByText('Google Drive (Analytics)')).toBeInTheDocument()
    expect(screen.getByText('Outlook Calendar')).toBeInTheDocument()
  })

  it('does not render when isOpen is false', () => {
    render(<AccountManager {...defaultProps} isOpen={false} />)
    expect(screen.queryByText('My Accounts')).not.toBeInTheDocument()
  })

  it('shows active/paused badges', async () => {
    render(<AccountManager {...defaultProps} />)

    await waitFor(() => {
      expect(screen.getByText('Gmail (ken@gmail.com)')).toBeInTheDocument()
    })

    const activeBadges = screen.getAllByText('active')
    const pausedBadges = screen.getAllByText('paused')
    expect(activeBadges).toHaveLength(2)
    expect(pausedBadges).toHaveLength(1)
  })

  it('toggles active status via PUT', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    // First call returns list, second call is the PUT toggle
    fetchSpy
      .mockResolvedValueOnce({ ok: true, json: async () => mockAccounts } as Response)
      .mockResolvedValueOnce({ ok: true, json: async () => ({}) } as Response)

    render(<AccountManager {...defaultProps} />)

    await waitFor(() => {
      expect(screen.getByText('Gmail (ken@gmail.com)')).toBeInTheDocument()
    })

    const toggleBtn = screen.getByTestId('toggle-active-email-google-ken')
    fireEvent.click(toggleBtn)

    await waitFor(() => {
      expect(fetchSpy).toHaveBeenCalledWith(
        expect.stringContaining('/api/accounts/email-google-ken'),
        expect.objectContaining({
          method: 'PUT',
          body: JSON.stringify({ active: false }),
        }),
      )
    })
  })

  it('removes account via DELETE with confirmation', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    fetchSpy
      .mockResolvedValueOnce({ ok: true, json: async () => mockAccounts } as Response)
      .mockResolvedValueOnce({ ok: true, json: async () => ({ status: 'ok' }) } as Response)

    render(<AccountManager {...defaultProps} />)

    await waitFor(() => {
      expect(screen.getByText('Gmail (ken@gmail.com)')).toBeInTheDocument()
    })

    // Click delete to show confirmation
    const deleteBtn = screen.getByTestId('delete-email-google-ken')
    fireEvent.click(deleteBtn)

    // Confirm delete
    const confirmBtn = screen.getByTestId('confirm-delete-email-google-ken')
    fireEvent.click(confirmBtn)

    await waitFor(() => {
      expect(fetchSpy).toHaveBeenCalledWith(
        expect.stringContaining('/api/accounts/email-google-ken'),
        expect.objectContaining({ method: 'DELETE' }),
      )
    })

    // Account should be removed from list
    await waitFor(() => {
      expect(screen.queryByText('Gmail (ken@gmail.com)')).not.toBeInTheDocument()
    })
  })

  it('shows empty state when no accounts', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => [],
    } as Response)

    render(<AccountManager {...defaultProps} />)

    await waitFor(() => {
      expect(screen.getByText('No accounts connected yet.')).toBeInTheDocument()
    })
  })

  it('shows error on fetch failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: false,
      status: 500,
      json: async () => ({}),
    } as Response)

    render(<AccountManager {...defaultProps} />)

    await waitFor(() => {
      expect(screen.getByText(/Failed to load accounts/)).toBeInTheDocument()
    })
  })
})
