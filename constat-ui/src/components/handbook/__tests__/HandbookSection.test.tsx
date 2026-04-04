// Copyright (c) 2025 Kenneth Stott
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.

import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { MockedProvider } from '@apollo/client/testing'
import { HandbookSection, type HandbookSectionData } from '../HandbookSection'
import { UPDATE_HANDBOOK_ENTRY } from '@/graphql/operations/handbook'

const mockSection: HandbookSectionData = {
  title: 'Glossary',
  content: [
    { key: 'revenue', display: 'Total income from sales', metadata: null, editable: true },
    { key: 'churn', display: 'Customer attrition rate', metadata: null, editable: false },
  ],
  lastUpdated: '2025-01-15T10:00:00Z',
}

function renderSection(props?: Partial<Parameters<typeof HandbookSection>[0]>, mocks: any[] = []) {
  return render(
    <MockedProvider mocks={mocks} addTypename={false}>
      <HandbookSection
        section={mockSection}
        sectionKey="glossary"
        sessionId="test-session"
        defaultExpanded
        {...props}
      />
    </MockedProvider>,
  )
}

describe('HandbookSection', () => {
  it('renders section title and entry count', () => {
    renderSection()
    expect(screen.getByText('Glossary')).toBeInTheDocument()
    expect(screen.getByText('2 entries')).toBeInTheDocument()
  })

  it('renders entry keys and displays when expanded', () => {
    renderSection()
    expect(screen.getByText('revenue')).toBeInTheDocument()
    expect(screen.getByText('Total income from sales')).toBeInTheDocument()
    expect(screen.getByText('churn')).toBeInTheDocument()
  })

  it('shows edit button only for editable entries', () => {
    renderSection()
    const editButtons = screen.getAllByTitle('Edit entry')
    // Only 'revenue' is editable
    expect(editButtons).toHaveLength(1)
    expect(screen.getByLabelText('Edit revenue')).toBeInTheDocument()
  })

  it('enters inline edit mode and can cancel', () => {
    renderSection()
    fireEvent.click(screen.getByLabelText('Edit revenue'))
    // Should show textarea with current value
    const textarea = screen.getByDisplayValue('Total income from sales')
    expect(textarea).toBeInTheDocument()
    expect(screen.getByText('Cancel')).toBeInTheDocument()

    // Cancel should revert
    fireEvent.click(screen.getByText('Cancel'))
    expect(screen.queryByDisplayValue('Total income from sales')).not.toBeInTheDocument()
    expect(screen.getByText('Total income from sales')).toBeInTheDocument()
  })

  it('calls mutation on save', async () => {
    const mocks = [
      {
        request: {
          query: UPDATE_HANDBOOK_ENTRY,
          variables: {
            sessionId: 'test-session',
            section: 'glossary',
            key: 'revenue',
            fieldName: 'display',
            newValue: 'Updated definition',
            reason: undefined,
          },
        },
        result: { data: { updateHandbookEntry: { status: 'ok', section: 'glossary', key: 'revenue' } } },
      },
    ]

    renderSection({}, mocks)
    fireEvent.click(screen.getByLabelText('Edit revenue'))

    const textarea = screen.getByDisplayValue('Total income from sales')
    fireEvent.change(textarea, { target: { value: 'Updated definition' } })
    fireEvent.click(screen.getByText('Save'))

    await waitFor(() => {
      // After save, should exit edit mode
      expect(screen.queryByText('Cancel')).not.toBeInTheDocument()
    })
  })

  it('collapses and expands on header click', () => {
    renderSection({ defaultExpanded: false })
    // Content should not be visible
    expect(screen.queryByText('revenue')).not.toBeInTheDocument()

    // Click to expand
    fireEvent.click(screen.getByText('Glossary'))
    expect(screen.getByText('revenue')).toBeInTheDocument()
  })
})
