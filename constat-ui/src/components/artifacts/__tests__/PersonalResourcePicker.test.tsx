import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { PersonalResourcePicker } from '../PersonalResourcePicker'

vi.mock('@/contexts/AuthContext', () => ({
  useAuth: vi.fn(),
}))

import { useAuth } from '@/contexts/AuthContext'

const mockUseAuth = useAuth as ReturnType<typeof vi.fn>

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
  })

  it('renders all 7 resource cards when auth is disabled', () => {
    render(<PersonalResourcePicker {...defaultProps} />)

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

  it('shows only Google resources when only google.com provider is linked', () => {
    mockUseAuth.mockReturnValue({
      user: { providerData: [{ providerId: 'google.com' }] },
      isAuthDisabled: false,
    })
    render(<PersonalResourcePicker {...defaultProps} />)

    expect(screen.getByText('Gmail')).toBeInTheDocument()
    expect(screen.getByText('Google Drive')).toBeInTheDocument()
    expect(screen.getByText('Google Calendar')).toBeInTheDocument()
    expect(screen.queryByText('Outlook')).not.toBeInTheDocument()
    expect(screen.queryByText('OneDrive')).not.toBeInTheDocument()
    expect(screen.queryByText('Outlook Calendar')).not.toBeInTheDocument()
    expect(screen.queryByText('SharePoint')).not.toBeInTheDocument()
    expect(screen.getAllByText('Connect')).toHaveLength(3)
  })

  it('shows only Microsoft resources when only microsoft.com provider is linked', () => {
    mockUseAuth.mockReturnValue({
      user: { providerData: [{ providerId: 'microsoft.com' }] },
      isAuthDisabled: false,
    })
    render(<PersonalResourcePicker {...defaultProps} />)

    expect(screen.getByText('Outlook')).toBeInTheDocument()
    expect(screen.getByText('OneDrive')).toBeInTheDocument()
    expect(screen.getByText('Outlook Calendar')).toBeInTheDocument()
    expect(screen.getByText('SharePoint')).toBeInTheDocument()
    expect(screen.queryByText('Gmail')).not.toBeInTheDocument()
    expect(screen.queryByText('Google Drive')).not.toBeInTheDocument()
    expect(screen.queryByText('Google Calendar')).not.toBeInTheDocument()
    expect(screen.getAllByText('Connect')).toHaveLength(4)
  })

  it('shows all 7 resources when both google.com and microsoft.com are linked', () => {
    mockUseAuth.mockReturnValue({
      user: { providerData: [{ providerId: 'google.com' }, { providerId: 'microsoft.com' }] },
      isAuthDisabled: false,
    })
    render(<PersonalResourcePicker {...defaultProps} />)

    expect(screen.getAllByText('Connect')).toHaveLength(7)
  })

  it('shows empty-state message when no providers are linked', () => {
    mockUseAuth.mockReturnValue({
      user: { providerData: [] },
      isAuthDisabled: false,
    })
    render(<PersonalResourcePicker {...defaultProps} />)

    expect(screen.queryByText('Connect')).not.toBeInTheDocument()
    expect(screen.getByText(/No OAuth providers/)).toBeInTheDocument()
  })

  it('does not render when isOpen is false', () => {
    render(<PersonalResourcePicker {...defaultProps} isOpen={false} />)
    expect(screen.queryByText('Connect Personal Resource')).not.toBeInTheDocument()
  })

  it('opens OAuth popup when Connect is clicked', () => {
    const openSpy = vi.spyOn(window, 'open').mockReturnValue({} as Window)
    render(<PersonalResourcePicker {...defaultProps} />)

    const gmailCard = screen.getByTestId('resource-card-google-email')
    fireEvent.click(gmailCard)

    expect(openSpy).toHaveBeenCalledWith(
      expect.stringContaining('/api/oauth/authorize?provider=google&resource_type=email'),
      'oauth-popup',
      expect.any(String),
    )
  })

  it('shows error when popup is blocked', () => {
    vi.spyOn(window, 'open').mockReturnValue(null)
    render(<PersonalResourcePicker {...defaultProps} />)

    const card = screen.getByTestId('resource-card-google-email')
    fireEvent.click(card)

    expect(screen.getByText(/Popup blocked/)).toBeInTheDocument()
  })

  it('calls onClose when X button is clicked', () => {
    render(<PersonalResourcePicker {...defaultProps} />)
    const closeButtons = screen.getAllByRole('button')
    // The X close button is the first button in the header
    const closeBtn = closeButtons.find(btn => btn.querySelector('.w-5.h-5'))
    if (closeBtn) fireEvent.click(closeBtn)
    expect(defaultProps.onClose).toHaveBeenCalled()
  })

  it('calls onClose when backdrop is clicked', () => {
    render(<PersonalResourcePicker {...defaultProps} />)
    // Click the backdrop (outermost overlay div)
    const backdrop = screen.getByText('Connect Personal Resource').closest('.fixed')
    if (backdrop) fireEvent.click(backdrop)
    expect(defaultProps.onClose).toHaveBeenCalled()
  })

  it('calls onClose on Escape key', () => {
    render(<PersonalResourcePicker {...defaultProps} />)
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(defaultProps.onClose).toHaveBeenCalled()
  })
})
