// Copyright (c) 2025 Kenneth Stott
//
// Tests for AddApiModal: auth type field rendering and onAdd payload.

import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { AddApiModal, type ApiAddFields } from '../AddApiModal'

function renderModal(onAdd = vi.fn(), onCancel = vi.fn()) {
  render(<AddApiModal onAdd={onAdd} onCancel={onCancel} uploading={false} />)
}

function fillBase(name = 'myapi', url = 'https://api.example.com') {
  fireEvent.change(screen.getByPlaceholderText('Name'), { target: { value: name } })
  fireEvent.change(screen.getByPlaceholderText(/Base URL/i), { target: { value: url } })
}

function selectAuth(value: string) {
  const selects = screen.getAllByRole('combobox')
  // Auth selector is the last combobox (after type selector)
  const authSelect = selects[selects.length - 1]
  fireEvent.change(authSelect, { target: { value } })
}

describe('AddApiModal', () => {
  let onAdd: ReturnType<typeof vi.fn>
  let onCancel: ReturnType<typeof vi.fn>

  beforeEach(() => {
    onAdd = vi.fn()
    onCancel = vi.fn()
  })

  // ---------------------------------------------------------------------------
  // Initial state
  // ---------------------------------------------------------------------------

  it('renders name, URL, type selector, auth selector', () => {
    renderModal(onAdd, onCancel)
    expect(screen.getByPlaceholderText('Name')).toBeInTheDocument()
    expect(screen.getByPlaceholderText(/Base URL/i)).toBeInTheDocument()
    expect(screen.getByText('Type')).toBeInTheDocument()
    expect(screen.getByText('Authentication')).toBeInTheDocument()
  })

  it('type selector has REST, GraphQL, OpenAPI options', () => {
    renderModal(onAdd, onCancel)
    const typeSelect = screen.getAllByRole('combobox')[0]
    const values = Array.from(typeSelect.querySelectorAll('option')).map((o) => (o as HTMLOptionElement).value)
    expect(values).toContain('rest')
    expect(values).toContain('graphql')
    expect(values).toContain('openapi')
  })

  it('auth selector has 5 options', () => {
    renderModal(onAdd, onCancel)
    const selects = screen.getAllByRole('combobox')
    const authSelect = selects[selects.length - 1]
    expect(authSelect.querySelectorAll('option').length).toBe(5)
  })

  it('Add button disabled when name missing', () => {
    renderModal(onAdd, onCancel)
    fireEvent.change(screen.getByPlaceholderText(/Base URL/i), { target: { value: 'https://api.example.com' } })
    expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled()
  })

  it('Add button disabled when URL missing', () => {
    renderModal(onAdd, onCancel)
    fireEvent.change(screen.getByPlaceholderText('Name'), { target: { value: 'myapi' } })
    expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled()
  })

  it('Add button enabled with name + URL (no auth)', () => {
    renderModal(onAdd, onCancel)
    fillBase()
    expect(screen.getByRole('button', { name: 'Add' })).not.toBeDisabled()
  })

  it('Cancel calls onCancel', () => {
    renderModal(onAdd, onCancel)
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(onCancel).toHaveBeenCalledOnce()
  })

  // ---------------------------------------------------------------------------
  // No auth
  // ---------------------------------------------------------------------------

  it('no-auth hides credential fields', () => {
    renderModal(onAdd, onCancel)
    // default is 'none'
    expect(screen.queryByPlaceholderText('Bearer token')).not.toBeInTheDocument()
    expect(screen.queryByPlaceholderText('user')).not.toBeInTheDocument()
    expect(screen.queryByPlaceholderText(/client_id/i)).not.toBeInTheDocument()
  })

  it('no-auth submits with authType none', () => {
    renderModal(onAdd, onCancel)
    fillBase()
    fireEvent.click(screen.getByRole('button', { name: 'Add' }))
    expect(onAdd).toHaveBeenCalledOnce()
    const fields: ApiAddFields = onAdd.mock.calls[0][0]
    expect(fields.authType).toBe('none')
    expect(fields.name).toBe('myapi')
    expect(fields.baseUrl).toBe('https://api.example.com')
    expect(fields.type).toBe('rest')
  })

  // ---------------------------------------------------------------------------
  // Bearer token
  // ---------------------------------------------------------------------------

  it('bearer shows token field', () => {
    renderModal(onAdd, onCancel)
    selectAuth('bearer')
    expect(screen.getByPlaceholderText('Bearer token')).toBeInTheDocument()
  })

  it('bearer Add disabled until token filled', () => {
    renderModal(onAdd, onCancel)
    fillBase()
    selectAuth('bearer')
    expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled()
    fireEvent.change(screen.getByPlaceholderText('Bearer token'), { target: { value: 'tok123' } })
    expect(screen.getByRole('button', { name: 'Add' })).not.toBeDisabled()
  })

  it('bearer submits authToken', () => {
    renderModal(onAdd, onCancel)
    fillBase()
    selectAuth('bearer')
    fireEvent.change(screen.getByPlaceholderText('Bearer token'), { target: { value: 'tok123' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add' }))
    const fields: ApiAddFields = onAdd.mock.calls[0][0]
    expect(fields.authType).toBe('bearer')
    expect(fields.authToken).toBe('tok123')
  })

  // ---------------------------------------------------------------------------
  // Basic auth
  // ---------------------------------------------------------------------------

  it('basic shows username and password fields', () => {
    renderModal(onAdd, onCancel)
    selectAuth('basic')
    expect(screen.getByPlaceholderText('user')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('password')).toBeInTheDocument()
  })

  it('basic Add disabled until username filled', () => {
    renderModal(onAdd, onCancel)
    fillBase()
    selectAuth('basic')
    expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled()
    fireEvent.change(screen.getByPlaceholderText('user'), { target: { value: 'admin' } })
    expect(screen.getByRole('button', { name: 'Add' })).not.toBeDisabled()
  })

  it('basic submits authUsername and authPassword', () => {
    renderModal(onAdd, onCancel)
    fillBase()
    selectAuth('basic')
    fireEvent.change(screen.getByPlaceholderText('user'), { target: { value: 'admin' } })
    fireEvent.change(screen.getByPlaceholderText('password'), { target: { value: 'pass123' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add' }))
    const fields: ApiAddFields = onAdd.mock.calls[0][0]
    expect(fields.authType).toBe('basic')
    expect(fields.authUsername).toBe('admin')
    expect(fields.authPassword).toBe('pass123')
  })

  // ---------------------------------------------------------------------------
  // API key
  // ---------------------------------------------------------------------------

  it('api_key shows header name and key value fields', () => {
    renderModal(onAdd, onCancel)
    selectAuth('api_key')
    expect(screen.getByPlaceholderText('X-API-Key')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('key value')).toBeInTheDocument()
  })

  it('api_key defaults header name to X-API-Key', () => {
    renderModal(onAdd, onCancel)
    selectAuth('api_key')
    const headerInput = screen.getByPlaceholderText('X-API-Key') as HTMLInputElement
    expect(headerInput.value).toBe('X-API-Key')
  })

  it('api_key Add disabled until key value filled', () => {
    renderModal(onAdd, onCancel)
    fillBase()
    selectAuth('api_key')
    expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled()
    fireEvent.change(screen.getByPlaceholderText('key value'), { target: { value: 'mysecret' } })
    expect(screen.getByRole('button', { name: 'Add' })).not.toBeDisabled()
  })

  it('api_key submits header name and token', () => {
    renderModal(onAdd, onCancel)
    fillBase()
    selectAuth('api_key')
    fireEvent.change(screen.getByPlaceholderText('X-API-Key'), { target: { value: 'Authorization' } })
    fireEvent.change(screen.getByPlaceholderText('key value'), { target: { value: 'sk-abc123' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add' }))
    const fields: ApiAddFields = onAdd.mock.calls[0][0]
    expect(fields.authType).toBe('api_key')
    expect(fields.authHeader).toBe('Authorization')
    expect(fields.authToken).toBe('sk-abc123')
  })

  // ---------------------------------------------------------------------------
  // OAuth2
  // ---------------------------------------------------------------------------

  it('oauth2 shows client_id, client_secret, token URL fields', () => {
    renderModal(onAdd, onCancel)
    selectAuth('oauth2')
    expect(screen.getByPlaceholderText('client_id')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('client_secret')).toBeInTheDocument()
    expect(screen.getByPlaceholderText(/token/i)).toBeInTheDocument()
  })

  it('oauth2 Add disabled until all three fields filled', () => {
    renderModal(onAdd, onCancel)
    fillBase()
    selectAuth('oauth2')
    expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled()
    fireEvent.change(screen.getByPlaceholderText('client_id'), { target: { value: 'cid' } })
    fireEvent.change(screen.getByPlaceholderText('client_secret'), { target: { value: 'csec' } })
    expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled()
    fireEvent.change(screen.getByPlaceholderText(/auth\.example\.com/), { target: { value: 'https://auth.example.com/token' } })
    expect(screen.getByRole('button', { name: 'Add' })).not.toBeDisabled()
  })

  it('oauth2 submits all credential fields', () => {
    renderModal(onAdd, onCancel)
    fillBase()
    selectAuth('oauth2')
    fireEvent.change(screen.getByPlaceholderText('client_id'), { target: { value: 'client123' } })
    fireEvent.change(screen.getByPlaceholderText('client_secret'), { target: { value: 'secret456' } })
    fireEvent.change(screen.getByPlaceholderText(/auth\.example\.com/), { target: { value: 'https://auth.example.com/token' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add' }))
    const fields: ApiAddFields = onAdd.mock.calls[0][0]
    expect(fields.authType).toBe('oauth2')
    expect(fields.authClientId).toBe('client123')
    expect(fields.authClientSecret).toBe('secret456')
    expect(fields.authTokenUrl).toBe('https://auth.example.com/token')
  })

  // ---------------------------------------------------------------------------
  // Type selection
  // ---------------------------------------------------------------------------

  it('selecting GraphQL passes type=graphql', () => {
    renderModal(onAdd, onCancel)
    fillBase()
    fireEvent.change(screen.getAllByRole('combobox')[0], { target: { value: 'graphql' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add' }))
    const fields: ApiAddFields = onAdd.mock.calls[0][0]
    expect(fields.type).toBe('graphql')
  })

  // ---------------------------------------------------------------------------
  // Uploading state
  // ---------------------------------------------------------------------------

  it('shows spinner and disabled Add when uploading=true', () => {
    render(<AddApiModal onAdd={vi.fn()} onCancel={vi.fn()} uploading={true} />)
    expect(screen.getByRole('button', { name: /adding/i })).toBeDisabled()
  })
})
