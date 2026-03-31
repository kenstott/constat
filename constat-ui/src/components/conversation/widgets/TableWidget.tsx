// Copyright (c) 2025 Kenneth Stott
// Canary: 76a8457d-cbff-4655-b917-9c86d72496a1
//
// This source code is licensed under the Business Source License 1.1
// found in the LICENSE file in the root directory of this source tree.
//
// NOTICE: Use of this software for training artificial intelligence or
// machine learning models is strictly prohibited without explicit written
// permission from the copyright holder.

// TableWidget - Editable grid for structured data entry

import { useState, useCallback } from 'react'
import { PlusIcon, TrashIcon } from '@heroicons/react/24/outline'

interface ColumnDef {
  key: string
  label: string
  type: 'text' | 'boolean' | 'select'
  options?: string[]
}

interface TableWidgetProps {
  config: Record<string, unknown>
  value: string
  structuredValue?: { rows: Record<string, unknown>[] }
  onAnswer: (freeform: string, structured: { rows: Record<string, unknown>[] }) => void
}

export function TableWidget({ config, structuredValue, onAnswer }: TableWidgetProps) {
  const columns = (config.columns as ColumnDef[]) || []
  const initialRows = (config.rows as Record<string, unknown>[]) || []

  const [rows, setRows] = useState<Record<string, unknown>[]>(() => {
    if (structuredValue?.rows?.length) return structuredValue.rows
    if (initialRows.length) return initialRows
    // Start with one empty row
    const emptyRow: Record<string, unknown> = {}
    columns.forEach(col => {
      emptyRow[col.key] = col.type === 'boolean' ? false : ''
    })
    return [emptyRow]
  })

  const emitAnswer = useCallback((newRows: Record<string, unknown>[]) => {
    const nonEmpty = newRows.filter(row =>
      columns.some(col => {
        const val = row[col.key]
        return val !== '' && val !== false && val != null
      })
    )
    const freeform = nonEmpty.length > 0
      ? `${nonEmpty.length} row(s): ${nonEmpty.map(r => columns.map(c => `${c.label}=${r[c.key]}`).join(', ')).join('; ')}`
      : 'No data entered'
    onAnswer(freeform, { rows: nonEmpty })
  }, [columns, onAnswer])

  const updateCell = (rowIndex: number, key: string, value: unknown) => {
    const newRows = rows.map((row, i) =>
      i === rowIndex ? { ...row, [key]: value } : row
    )
    setRows(newRows)
    emitAnswer(newRows)
  }

  const addRow = () => {
    const emptyRow: Record<string, unknown> = {}
    columns.forEach(col => {
      emptyRow[col.key] = col.type === 'boolean' ? false : ''
    })
    const newRows = [...rows, emptyRow]
    setRows(newRows)
  }

  const removeRow = (index: number) => {
    if (rows.length <= 1) return
    const newRows = rows.filter((_, i) => i !== index)
    setRows(newRows)
    emitAnswer(newRows)
  }

  const renderCell = (row: Record<string, unknown>, rowIndex: number, col: ColumnDef) => {
    const value = row[col.key]

    switch (col.type) {
      case 'boolean':
        return (
          <input
            type="checkbox"
            checked={!!value}
            onChange={(e) => updateCell(rowIndex, col.key, e.target.checked)}
            className="w-4 h-4 rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
          />
        )
      case 'select':
        return (
          <select
            value={(value as string) || ''}
            onChange={(e) => updateCell(rowIndex, col.key, e.target.value)}
            className="w-full px-2 py-1 text-sm rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
          >
            <option value="">--</option>
            {(col.options || []).map(opt => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        )
      default:
        return (
          <input
            type="text"
            value={(value as string) || ''}
            onChange={(e) => updateCell(rowIndex, col.key, e.target.value)}
            onKeyDown={(e) => e.stopPropagation()}
            onKeyUp={(e) => e.stopPropagation()}
            className="w-full px-2 py-1 text-sm rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
          />
        )
    }
  }

  return (
    <div className="space-y-2">
      <div className="overflow-x-auto border border-gray-200 dark:border-gray-700 rounded-lg">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-gray-50 dark:bg-gray-900/50">
              {columns.map(col => (
                <th
                  key={col.key}
                  className="px-3 py-2 text-left text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                >
                  {col.label}
                </th>
              ))}
              <th className="w-8" />
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="hover:bg-gray-50/50 dark:hover:bg-gray-800/50">
                {columns.map(col => (
                  <td key={col.key} className="px-3 py-2">
                    {renderCell(row, rowIndex, col)}
                  </td>
                ))}
                <td className="px-1 py-2">
                  <button
                    onClick={() => removeRow(rowIndex)}
                    disabled={rows.length <= 1}
                    className="p-1 text-gray-400 hover:text-red-500 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                    title="Remove row"
                  >
                    <TrashIcon className="w-3.5 h-3.5" />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <button
        onClick={addRow}
        className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-primary-600 dark:text-primary-400 hover:bg-primary-50 dark:hover:bg-primary-900/30 rounded-lg transition-colors"
      >
        <PlusIcon className="w-3.5 h-3.5" />
        Add row
      </button>
    </div>
  )
}
