// Copyright (c) 2025 Kenneth Stott
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.

import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { MockedProvider } from '@apollo/client/testing'
import { HANDBOOK_QUERY } from '@/graphql/operations/handbook'

// Mock SessionContext
vi.mock('@/contexts/SessionContext', () => ({
  useSessionContext: () => ({ sessionId: 'test-session-1' }),
}))

// Mock useDomains
vi.mock('@/hooks/useDomains', () => ({
  useActiveDomains: () => ({
    activeDomains: ['sales-analytics', 'hr-reporting'],
    loading: false,
    toggleDomain: vi.fn(),
  }),
}))

// Import after mocks
import { DomainHandbookPage } from '../DomainHandbookPage'

const handbookMock = {
  request: {
    query: HANDBOOK_QUERY,
    variables: { sessionId: 'test-session-1', domain: 'sales-analytics' },
  },
  result: {
    data: {
      handbook: {
        domain: 'sales-analytics',
        generatedAt: '2025-06-01T12:00:00Z',
        summary: 'Sales analytics domain covers revenue tracking and forecasting.',
        sections: {
          overview: {
            title: 'Overview',
            content: [
              { key: 'description', display: 'Sales analytics domain', metadata: null, editable: false },
            ],
            last_updated: '2025-06-01T12:00:00Z',
          },
          glossary: {
            title: 'Glossary',
            content: [
              { key: 'revenue', display: 'Total sales income', metadata: null, editable: true },
            ],
            last_updated: '2025-06-01T11:00:00Z',
          },
        },
      },
    },
  },
}

function renderPage(mocks = [handbookMock]) {
  return render(
    <MockedProvider mocks={mocks} addTypename={false}>
      <DomainHandbookPage />
    </MockedProvider>,
  )
}

describe('DomainHandbookPage', () => {
  it('renders sidebar navigation with all section labels', async () => {
    renderPage()
    // Sidebar should show all 8 section labels
    expect(screen.getByText('Overview')).toBeInTheDocument()
    expect(screen.getByText('Data Sources')).toBeInTheDocument()
    expect(screen.getByText('Key Entities')).toBeInTheDocument()
    expect(screen.getByText('Glossary')).toBeInTheDocument()
    expect(screen.getByText('Learned Rules')).toBeInTheDocument()
    expect(screen.getByText('Common Patterns')).toBeInTheDocument()
    expect(screen.getByText('Agents & Skills')).toBeInTheDocument()
    expect(screen.getByText('Known Limitations')).toBeInTheDocument()
  })

  it('shows loading state initially', () => {
    renderPage()
    expect(screen.getByTestId('handbook-loading')).toBeInTheDocument()
  })

  it('renders handbook content after loading', async () => {
    renderPage()
    await waitFor(() => {
      expect(screen.getByText('Domain Handbook')).toBeInTheDocument()
    })
    await waitFor(() => {
      expect(screen.getByText(/Sales analytics domain covers/)).toBeInTheDocument()
    })
  })

  it('renders domain selector with active domains', () => {
    renderPage()
    const select = screen.getByLabelText('Domain') as HTMLSelectElement
    expect(select).toBeInTheDocument()
    expect(select.value).toBe('sales-analytics')
  })
})
