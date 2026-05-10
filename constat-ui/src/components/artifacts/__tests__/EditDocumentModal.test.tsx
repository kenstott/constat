// Copyright (c) 2025 Kenneth Stott
//
// Tests for EditDocumentModal: URI scheme detection, validation logic,
// and component rendering/validation behaviour.

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { detectScheme, isValidUri, URL_SCHEMES } from '../uriUtils'
import { EditDocumentModal } from '../EditDocumentModal'
import type { DocumentSourceInfo } from '@/types/api'

// ---------------------------------------------------------------------------
// Apollo mocks — prevent real network calls
// ---------------------------------------------------------------------------
vi.mock('@apollo/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@apollo/client')>()
  return {
    ...actual,
    useMutation: () => [vi.fn().mockResolvedValue({}), { loading: false }],
    useLazyQuery: () => [vi.fn(), { data: null, loading: false }],
  }
})

vi.mock('@/contexts/SessionContext', () => ({
  useSessionContext: () => ({ sessionId: 'test-session-id' }),
}))

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const BASE_DOC: DocumentSourceInfo = {
  name: 'hr-management',
  description: 'HR docs',
  path: 'https://hr.example.com/wiki',
  indexed: true,
  from_config: false,
  source: 'session',
}

function renderModal(doc = BASE_DOC, onSuccess = vi.fn(), onCancel = vi.fn()) {
  render(<EditDocumentModal doc={doc} onSuccess={onSuccess} onCancel={onCancel} />)
}

function setUri(value: string) {
  fireEvent.change(screen.getByPlaceholderText(/https:\/\/example\.com/i), { target: { value } })
}

// ---------------------------------------------------------------------------
// Pure unit tests: detectScheme
// ---------------------------------------------------------------------------
describe('detectScheme', () => {
  it('detects https://', () => expect(detectScheme('https://example.com')).toBe('https://'))
  it('detects http://', () => expect(detectScheme('http://example.com')).toBe('http://'))
  it('detects ftp://', () => expect(detectScheme('ftp://ftp.example.com/file')).toBe('ftp://'))
  it('detects sftp://', () => expect(detectScheme('sftp://host/path')).toBe('sftp://'))
  it('detects s3://', () => expect(detectScheme('s3://my-bucket/key.pdf')).toBe('s3://'))
  it('detects s3a://', () => expect(detectScheme('s3a://my-bucket/key.pdf')).toBe('s3a://'))
  it('detects file://', () => expect(detectScheme('file:///etc/passwd')).toBe('file://'))
  it('is case-insensitive', () => expect(detectScheme('HTTP://EXAMPLE.COM')).toBe('http://'))
  it('returns null for bare path', () => expect(detectScheme('/data/rules.md')).toBeNull())
  it('returns null for empty string', () => expect(detectScheme('')).toBeNull())
  it('URL_SCHEMES contains all 7 expected schemes', () =>
    expect(URL_SCHEMES).toEqual(['http://', 'https://', 'ftp://', 'sftp://', 's3://', 's3a://', 'file://']))
})

// ---------------------------------------------------------------------------
// Pure unit tests: isValidUri
// ---------------------------------------------------------------------------
describe('isValidUri', () => {
  // Valid cases
  it('accepts https URL', () => expect(isValidUri('https://example.com/doc.pdf')).toBe(true))
  it('accepts http URL', () => expect(isValidUri('http://example.com')).toBe(true))
  it('accepts ftp URL', () => expect(isValidUri('ftp://ftp.example.com/file.txt')).toBe(true))
  it('accepts sftp URL', () => expect(isValidUri('sftp://host.example.com/path')).toBe(true))
  it('accepts s3 with bucket and key', () => expect(isValidUri('s3://my-bucket/data/file.parquet')).toBe(true))
  it('accepts s3a with bucket', () => expect(isValidUri('s3a://my-bucket/key')).toBe(true))
  it('accepts file:// with path', () => expect(isValidUri('file:///etc/passwd')).toBe(true))
  it('accepts bare file path', () => expect(isValidUri('/data/docs/rules.md')).toBe(true))
  it('accepts relative path', () => expect(isValidUri('./docs/file.md')).toBe(true))
  it('trims whitespace before validation', () => expect(isValidUri('  https://example.com  ')).toBe(true))

  // Invalid cases
  it('rejects empty string', () => expect(isValidUri('')).toBe(false))
  it('rejects whitespace only', () => expect(isValidUri('   ')).toBe(false))
  it('rejects malformed http (no host)', () => expect(isValidUri('http://')).toBe(false))
  it('rejects malformed https', () => expect(isValidUri('https://')).toBe(false))
  it('rejects s3 without bucket', () => expect(isValidUri('s3:///key-without-bucket')).toBe(false))
  it('rejects s3a without bucket', () => expect(isValidUri('s3a:///key')).toBe(false))
  it('rejects file:// with no path', () => expect(isValidUri('file://')).toBe(false))
})

// ---------------------------------------------------------------------------
// Component tests: scheme hint label
// ---------------------------------------------------------------------------
describe('EditDocumentModal — scheme hint label', () => {
  beforeEach(() => renderModal())

  it('shows "Web page (secure)" for https://', () => {
    setUri('https://example.com/docs')
    expect(screen.getByText('Web page (secure)')).toBeInTheDocument()
  })

  it('shows "Web page" for http://', () => {
    setUri('http://example.com')
    expect(screen.getByText('Web page')).toBeInTheDocument()
  })

  it('shows "AWS S3 object" for s3://', () => {
    setUri('s3://my-bucket/data.pdf')
    expect(screen.getByText('AWS S3 object')).toBeInTheDocument()
  })

  it('shows "S3-compatible storage" for s3a://', () => {
    setUri('s3a://my-bucket/data.pdf')
    expect(screen.getByText('S3-compatible storage')).toBeInTheDocument()
  })

  it('shows "FTP server" for ftp://', () => {
    setUri('ftp://ftp.example.com/file.txt')
    expect(screen.getByText('FTP server')).toBeInTheDocument()
  })

  it('shows "SFTP server" for sftp://', () => {
    setUri('sftp://host/path')
    expect(screen.getByText('SFTP server')).toBeInTheDocument()
  })

  it('shows "Local file" for file://', () => {
    setUri('file:///etc/passwd')
    expect(screen.getByText('Local file')).toBeInTheDocument()
  })

  it('shows "Local file path" for bare path', () => {
    setUri('/data/rules.md')
    expect(screen.getByText('Local file path')).toBeInTheDocument()
  })

  it('hides hint when URI is empty', () => {
    setUri('')
    expect(screen.queryByText(/Web page|S3|FTP|SFTP|Local/)).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// Component tests: validation errors
// ---------------------------------------------------------------------------
describe('EditDocumentModal — validation', () => {
  it('shows "Name is required" when name cleared and Save clicked', () => {
    renderModal()
    fireEvent.change(screen.getByDisplayValue('hr-management'), { target: { value: '' } })
    fireEvent.click(screen.getByRole('button', { name: /save/i }))
    expect(screen.getByText('Name is required')).toBeInTheDocument()
  })

  it('shows "Description is required" when description cleared', () => {
    renderModal()
    fireEvent.change(screen.getByDisplayValue('HR docs'), { target: { value: '' } })
    fireEvent.click(screen.getByRole('button', { name: /save/i }))
    expect(screen.getByText('Description is required')).toBeInTheDocument()
  })

  it('shows "URI or path is required" when URI cleared', () => {
    renderModal()
    setUri('')
    fireEvent.click(screen.getByRole('button', { name: /save/i }))
    expect(screen.getByText('URI or path is required')).toBeInTheDocument()
  })

  it('shows format error for malformed https URL', () => {
    renderModal()
    setUri('https://')
    fireEvent.click(screen.getByRole('button', { name: /save/i }))
    expect(screen.getByText(/Enter a valid URI/)).toBeInTheDocument()
  })

  it('shows format error for s3 without bucket', () => {
    renderModal()
    setUri('s3:///no-bucket')
    fireEvent.click(screen.getByRole('button', { name: /save/i }))
    expect(screen.getByText(/Enter a valid URI/)).toBeInTheDocument()
  })

  it('clears name error when user types', () => {
    renderModal()
    fireEvent.change(screen.getByDisplayValue('hr-management'), { target: { value: '' } })
    fireEvent.click(screen.getByRole('button', { name: /save/i }))
    expect(screen.getByText('Name is required')).toBeInTheDocument()
    fireEvent.change(screen.getByDisplayValue(''), { target: { value: 'new-name' } })
    expect(screen.queryByText('Name is required')).toBeNull()
  })

  it('shows crawl options only for http/https', () => {
    renderModal()
    // Default URI is https — crawl options should be present
    expect(screen.getByText('Follow links')).toBeInTheDocument()
    // Switch to s3 — crawl options should disappear
    setUri('s3://bucket/key.pdf')
    expect(screen.queryByText('Follow links')).toBeNull()
  })

  it('Cancel calls onCancel', () => {
    const onCancel = vi.fn()
    render(<EditDocumentModal doc={BASE_DOC} onSuccess={vi.fn()} onCancel={onCancel} />)
    fireEvent.click(screen.getByRole('button', { name: /cancel/i }))
    expect(onCancel).toHaveBeenCalledOnce()
  })
})
