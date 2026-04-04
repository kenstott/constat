import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { PersonalResourcePicker } from '../PersonalResourcePicker'

describe('PersonalResourcePicker', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
    sessionId: 'test-session-123',
  }

  beforeEach(() => {
    vi.restoreAllMocks()
    defaultProps.onClose = vi.fn()
  })

  it('renders all 7 resource cards when open', () => {
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
