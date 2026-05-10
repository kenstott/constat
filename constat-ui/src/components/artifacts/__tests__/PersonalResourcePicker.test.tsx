import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { PersonalResourcePicker } from '../PersonalResourcePicker'

vi.mock('@/contexts/AuthContext', () => ({
  useAuth: vi.fn(),
}))

import { useAuth } from '@/contexts/AuthContext'

const mockUseAuth = useAuth as ReturnType<typeof vi.fn>

function mockFetchVault(hasVault: boolean) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ has_vault: hasVault }),
    }),
  )
}

describe('PersonalResourcePicker', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
    sessionId: 'test-session-123',
  }

  beforeEach(() => {
    vi.restoreAllMocks()
    defaultProps.onClose = vi.fn()
    mockUseAuth.mockReturnValue({ user: null, isAuthDisabled: true })
    mockFetchVault(true)
  })

  it('renders all 7 resource cards when auth is disabled and vault exists', async () => {
    render(<PersonalResourcePicker {...defaultProps} />)

    await waitFor(() => expect(screen.queryByTestId('vault-warning')).not.toBeInTheDocument())

    expect(screen.getByText('Gmail')).toBeInTheDocument()
    expect(screen.getByText('Outlook')).toBeInTheDocument()
    expect(screen.getByText('Google Drive')).toBeInTheDocument()
    expect(screen.getByText('OneDrive')).toBeInTheDocument()
    expect(screen.getByText('Google Calendar')).toBeInTheDocument()
    expect(screen.getByText('Outlook Calendar')).toBeInTheDocument()
    expect(screen.getByText('SharePoint')).toBeInTheDocument()

    // 7 Connect labels (one per card)
    expect(screen.getAllByText('Connect')).toHaveLength(7)
  })

  it('shows only Google resources when only google.com provider is linked', async () => {
    mockUseAuth.mockReturnValue({
      user: { providerData: [{ providerId: 'google.com' }] },
      isAuthDisabled: false,
    })
    render(<PersonalResourcePicker {...defaultProps} />)

    await waitFor(() => expect(screen.queryByTestId('vault-warning')).not.toBeInTheDocument())

    expect(screen.getByText('Gmail')).toBeInTheDocument()
    expect(screen.getByText('Google Drive')).toBeInTheDocument()
    expect(screen.getByText('Google Calendar')).toBeInTheDocument()
    expect(screen.queryByText('Outlook')).not.toBeInTheDocument()
    expect(screen.queryByText('OneDrive')).not.toBeInTheDocument()
    expect(screen.queryByText('Outlook Calendar')).not.toBeInTheDocument()
    expect(screen.queryByText('SharePoint')).not.toBeInTheDocument()
    expect(screen.getAllByText('Connect')).toHaveLength(3)
  })

  it('shows only Microsoft resources when only microsoft.com provider is linked', async () => {
    mockUseAuth.mockReturnValue({
      user: { providerData: [{ providerId: 'microsoft.com' }] },
      isAuthDisabled: false,
    })
    render(<PersonalResourcePicker {...defaultProps} />)

    await waitFor(() => expect(screen.queryByTestId('vault-warning')).not.toBeInTheDocument())

    expect(screen.getByText('Outlook')).toBeInTheDocument()
    expect(screen.getByText('OneDrive')).toBeInTheDocument()
    expect(screen.getByText('Outlook Calendar')).toBeInTheDocument()
    expect(screen.getByText('SharePoint')).toBeInTheDocument()
    expect(screen.queryByText('Gmail')).not.toBeInTheDocument()
    expect(screen.queryByText('Google Drive')).not.toBeInTheDocument()
    expect(screen.queryByText('Google Calendar')).not.toBeInTheDocument()
    expect(screen.getAllByText('Connect')).toHaveLength(4)
  })

  it('shows all 7 resources when both google.com and microsoft.com are linked', async () => {
    mockUseAuth.mockReturnValue({
      user: { providerData: [{ providerId: 'google.com' }, { providerId: 'microsoft.com' }] },
      isAuthDisabled: false,
    })
    render(<PersonalResourcePicker {...defaultProps} />)

    await waitFor(() => expect(screen.queryByTestId('vault-warning')).not.toBeInTheDocument())
    expect(screen.getAllByText('Connect')).toHaveLength(7)
  })

  it('shows empty-state message when no providers are linked', async () => {
    mockUseAuth.mockReturnValue({
      user: { providerData: [] },
      isAuthDisabled: false,
    })
    render(<PersonalResourcePicker {...defaultProps} />)

    await waitFor(() => expect(screen.queryByTestId('vault-warning')).not.toBeInTheDocument())
    expect(screen.queryByText('Connect')).not.toBeInTheDocument()
    expect(screen.getByText(/No OAuth providers/)).toBeInTheDocument()
  })

  it('does not render when isOpen is false', () => {
    render(<PersonalResourcePicker {...defaultProps} isOpen={false} />)
    expect(screen.queryByText('Connect Personal Resource')).not.toBeInTheDocument()
  })

  it('opens OAuth popup when Connect is clicked', async () => {
    const openSpy = vi.spyOn(window, 'open').mockReturnValue({} as Window)
    render(<PersonalResourcePicker {...defaultProps} />)

    await waitFor(() => expect(screen.queryByTestId('vault-warning')).not.toBeInTheDocument())

    const gmailCard = screen.getByTestId('resource-card-google-email')
    fireEvent.click(gmailCard)

    expect(openSpy).toHaveBeenCalledWith(
      expect.stringContaining('/api/oauth/authorize?provider=google&resource_type=email'),
      'oauth-popup',
      expect.any(String),
    )
  })

  it('shows error when popup is blocked', async () => {
    vi.spyOn(window, 'open').mockReturnValue(null)
    render(<PersonalResourcePicker {...defaultProps} />)

    await waitFor(() => expect(screen.queryByTestId('vault-warning')).not.toBeInTheDocument())

    const card = screen.getByTestId('resource-card-google-email')
    fireEvent.click(card)

    expect(screen.getByText(/Popup blocked/)).toBeInTheDocument()
  })

  it('calls onClose when X button is clicked', async () => {
    render(<PersonalResourcePicker {...defaultProps} />)
    await waitFor(() => expect(screen.queryByTestId('vault-warning')).not.toBeInTheDocument())
    const closeButtons = screen.getAllByRole('button')
    const closeBtn = closeButtons.find(btn => btn.querySelector('.w-5.h-5'))
    if (closeBtn) fireEvent.click(closeBtn)
    expect(defaultProps.onClose).toHaveBeenCalled()
  })

  it('calls onClose when backdrop is clicked', async () => {
    render(<PersonalResourcePicker {...defaultProps} />)
    await waitFor(() => expect(screen.queryByTestId('vault-warning')).not.toBeInTheDocument())
    const backdrop = screen.getByText('Connect Personal Resource').closest('.fixed')
    if (backdrop) fireEvent.click(backdrop)
    expect(defaultProps.onClose).toHaveBeenCalled()
  })

  it('calls onClose on Escape key', async () => {
    render(<PersonalResourcePicker {...defaultProps} />)
    await waitFor(() => expect(screen.queryByTestId('vault-warning')).not.toBeInTheDocument())
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(defaultProps.onClose).toHaveBeenCalled()
  })

  // ---------------------------------------------------------------------------
  // Vault warning tests
  // ---------------------------------------------------------------------------

  it('shows vault warning when has_vault is false', async () => {
    mockFetchVault(false)
    render(<PersonalResourcePicker {...defaultProps} />)

    await waitFor(() => expect(screen.getByTestId('vault-warning')).toBeInTheDocument())
    expect(screen.getByText(/vault password is required/i)).toBeInTheDocument()
    expect(screen.getByTestId('vault-password-input')).toBeInTheDocument()
    expect(screen.getByTestId('vault-confirm-input')).toBeInTheDocument()
    expect(screen.getByTestId('vault-set-password-btn')).toBeInTheDocument()
  })

  it('resource cards are disabled when vault is not established', async () => {
    mockFetchVault(false)
    render(<PersonalResourcePicker {...defaultProps} />)

    await waitFor(() => expect(screen.getByTestId('vault-warning')).toBeInTheDocument())

    const gmailCard = screen.getByTestId('resource-card-google-email')
    expect(gmailCard).toBeDisabled()
  })

  it('shows validation error when passwords do not match', async () => {
    mockFetchVault(false)
    render(<PersonalResourcePicker {...defaultProps} />)

    await waitFor(() => expect(screen.getByTestId('vault-warning')).toBeInTheDocument())

    fireEvent.change(screen.getByTestId('vault-password-input'), { target: { value: 'abc' } })
    fireEvent.change(screen.getByTestId('vault-confirm-input'), { target: { value: 'xyz' } })
    fireEvent.click(screen.getByTestId('vault-set-password-btn'))

    expect(screen.getByText(/Passwords do not match/i)).toBeInTheDocument()
  })

  it('shows validation error when password is empty', async () => {
    mockFetchVault(false)
    render(<PersonalResourcePicker {...defaultProps} />)

    await waitFor(() => expect(screen.getByTestId('vault-warning')).toBeInTheDocument())
    fireEvent.click(screen.getByTestId('vault-set-password-btn'))

    expect(screen.getByText('Password is required.')).toBeInTheDocument()
  })

  it('hides vault warning and enables cards after successful vault creation', async () => {
    mockFetchVault(false)
    const createResp = { ok: true, json: () => Promise.resolve({ status: 'ok' }) }
    vi.stubGlobal(
      'fetch',
      vi.fn()
        .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve({ has_vault: false }) })
        .mockResolvedValueOnce(createResp),
    )

    render(<PersonalResourcePicker {...defaultProps} />)

    await waitFor(() => expect(screen.getByTestId('vault-warning')).toBeInTheDocument())

    fireEvent.change(screen.getByTestId('vault-password-input'), { target: { value: 'secret' } })
    fireEvent.change(screen.getByTestId('vault-confirm-input'), { target: { value: 'secret' } })
    fireEvent.click(screen.getByTestId('vault-set-password-btn'))

    await waitFor(() => expect(screen.queryByTestId('vault-warning')).not.toBeInTheDocument())

    const gmailCard = screen.getByTestId('resource-card-google-email')
    expect(gmailCard).not.toBeDisabled()
  })
})
