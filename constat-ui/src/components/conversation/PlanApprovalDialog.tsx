// Plan Approval Dialog - modal for reviewing and approving execution plans

import { useState } from 'react'
import { Dialog, DialogPanel, DialogTitle } from '@headlessui/react'
import {
  ClipboardDocumentListIcon,
  XMarkIcon,
  CheckCircleIcon,
  ArrowPathIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  ClockIcon,
  TrashIcon,
} from '@heroicons/react/24/outline'
import { CheckCircleIcon as CheckCircleSolid, XCircleIcon } from '@heroicons/react/24/solid'
import { useSessionStore } from '@/store/sessionStore'
import type { Step } from '@/types/api'

const stepStatusIcons: Record<string, { icon: typeof ClockIcon; color: string }> = {
  pending: { icon: ClockIcon, color: 'text-gray-400' },
  running: { icon: ClockIcon, color: 'text-blue-500 animate-spin' },
  completed: { icon: CheckCircleSolid, color: 'text-green-500' },
  failed: { icon: XCircleIcon, color: 'text-red-500' },
  skipped: { icon: XCircleIcon, color: 'text-gray-400' },
}

function StepItem({ step, index, isExpanded, onToggle, modification, onModificationChange, onDelete }: {
  step: Step
  index: number
  isExpanded: boolean
  onToggle: () => void
  modification: string
  onModificationChange: (value: string) => void
  onDelete: () => void
}) {
  const status = stepStatusIcons[step.status] || stepStatusIcons.pending
  const StatusIcon = status.icon
  const stepNumber = step.number ?? index + 1

  const expectedInputs = step.expected_inputs || []
  const expectedOutputs = step.expected_outputs || []
  const dependsOn = step.depends_on || []
  // Only show as modified if text differs from original goal
  const hasModification = modification.trim().length > 0 && modification.trim() !== (step.goal || '').trim()

  return (
    <div className={`border rounded-lg overflow-hidden ${hasModification ? 'border-amber-400 dark:border-amber-600' : 'border-gray-200 dark:border-gray-700'}`}>
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
      >
        {isExpanded ? (
          <ChevronDownIcon className="w-4 h-4 text-gray-400 flex-shrink-0" />
        ) : (
          <ChevronRightIcon className="w-4 h-4 text-gray-400 flex-shrink-0" />
        )}
        <StatusIcon className={`w-4 h-4 ${status.color} flex-shrink-0`} />
        <span className="text-xs font-medium text-gray-500 dark:text-gray-400 flex-shrink-0">
          {stepNumber}.
        </span>
        <span className="text-sm text-gray-700 dark:text-gray-300 flex-1">
          {step.goal || 'Processing...'}
        </span>
        {step.role_id && (
          <span className="text-xs px-1.5 py-0.5 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 rounded">
            {step.role_id}
          </span>
        )}
        {hasModification && (
          <span className="text-xs px-1.5 py-0.5 bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 rounded">
            Modified
          </span>
        )}
        <button
          onClick={(e) => {
            e.stopPropagation()
            onDelete()
          }}
          className="p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition-colors"
          title="Delete step"
        >
          <TrashIcon className="w-4 h-4" />
        </button>
      </button>

      {isExpanded && (
        <div className="px-3 py-2 bg-gray-50 dark:bg-gray-800/50 border-t border-gray-200 dark:border-gray-700 space-y-3">
          <div className="space-y-1.5 text-xs">
            {expectedInputs.length > 0 && (
              <div>
                <span className="text-gray-500 dark:text-gray-400">Inputs: </span>
                <span className="text-gray-700 dark:text-gray-300">
                  {expectedInputs.join(', ')}
                </span>
              </div>
            )}
            {expectedOutputs.length > 0 && (
              <div>
                <span className="text-gray-500 dark:text-gray-400">Outputs: </span>
                <span className="text-gray-700 dark:text-gray-300">
                  {expectedOutputs.join(', ')}
                </span>
              </div>
            )}
            {dependsOn.length > 0 && (
              <div>
                <span className="text-gray-500 dark:text-gray-400">Depends on: </span>
                <span className="text-gray-700 dark:text-gray-300">
                  Step {dependsOn.join(', ')}
                </span>
              </div>
            )}
          </div>
          {/* Modification input */}
          <div>
            <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">
              Modify this step:
            </label>
            <textarea
              value={modification}
              onChange={(e) => onModificationChange(e.target.value)}
              placeholder="Describe how to change this step..."
              rows={2}
              className="w-full px-2 py-1.5 text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:border-amber-500 focus:ring-1 focus:ring-amber-500 resize-none"
            />
          </div>
        </div>
      )}
    </div>
  )
}

export function PlanApprovalDialog() {
  const { status, plan, approvePlan, rejectPlan } = useSessionStore()
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set())
  const [stepModifications, setStepModifications] = useState<Record<number, string>>({})
  const [deletedSteps, setDeletedSteps] = useState<Set<number>>(new Set())
  const [additionalInstructions, setAdditionalInstructions] = useState('')

  const isOpen = status === 'awaiting_approval' && plan !== null

  if (!isOpen || !plan) {
    return null
  }

  const steps = plan.steps || []
  // Transform clarification numbering from 0-indexed to 1-indexed for display
  const rawProblem = plan.problem || 'Processing...'
  const problem = rawProblem.replace(/Clarifications:\s*([\s\S]*)/g, (_match: string, clarifications: string) => {
    // Replace "0:", "1:", etc. with "1:", "2:", etc.
    const renumbered = clarifications.replace(/^(\d+):/gm, (_: string, num: string) => `${parseInt(num) + 1}:`)
    return `Clarifications:\n${renumbered}`
  })

  // Filter out deleted steps for display
  const visibleSteps = steps.filter((step, index) => {
    const stepNum = step.number ?? index + 1
    return !deletedSteps.has(stepNum)
  })

  // Check if any modifications exist (comparing against original goals)
  const hasModifications = Object.entries(stepModifications).some(([stepNumStr, mod]) => {
    const stepNum = parseInt(stepNumStr)
    const step = steps.find((s, i) => (s.number ?? i + 1) === stepNum)
    return mod.trim().length > 0 && mod.trim() !== (step?.goal || '').trim()
  })
  const hasDeletedSteps = deletedSteps.size > 0
  const hasAnyChanges = hasModifications || hasDeletedSteps || additionalInstructions.trim().length > 0

  const toggleStep = (stepNumber: number) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev)
      if (next.has(stepNumber)) {
        next.delete(stepNumber)
      } else {
        next.add(stepNumber)
        // Initialize modification with original step goal when first expanding
        if (stepModifications[stepNumber] === undefined) {
          const step = steps.find((s, i) => (s.number ?? i + 1) === stepNumber)
          if (step?.goal) {
            setStepModifications((prev) => ({ ...prev, [stepNumber]: step.goal }))
          }
        }
      }
      return next
    })
  }

  const setStepModification = (stepNumber: number, value: string) => {
    setStepModifications((prev) => ({ ...prev, [stepNumber]: value }))
  }

  const deleteStep = (stepNumber: number) => {
    setDeletedSteps((prev) => {
      const next = new Set(prev)
      next.add(stepNumber)
      return next
    })
    // Also remove any modifications for this step
    setStepModifications((prev) => {
      const next = { ...prev }
      delete next[stepNumber]
      return next
    })
  }

  const handleApprove = () => {
    // Pass deleted step numbers to backend if any
    const deletedStepNumbers = deletedSteps.size > 0 ? Array.from(deletedSteps) : undefined
    approvePlan(deletedStepNumbers)
    setAdditionalInstructions('')
    setStepModifications({})
    setDeletedSteps(new Set())
    setExpandedSteps(new Set())
  }

  const handleRevise = () => {
    // Build structured edited steps array for backend
    const editedStepsArray: Array<{ number: number; goal: string }> = []
    // Preserve original starting step number (e.g. 4 for follow-up plans)
    const firstStepNum = steps.length > 0 ? (steps[0].number ?? 1) : 1
    let stepCounter = firstStepNum

    steps.forEach((step) => {
      const stepNum = step.number ?? steps.indexOf(step) + 1
      // Skip deleted steps
      if (deletedSteps.has(stepNum)) return

      // Use modification if provided, otherwise original goal
      const mod = stepModifications[stepNum]?.trim()
      const stepText = mod || step.goal || ''
      editedStepsArray.push({ number: stepCounter, goal: stepText })
      stepCounter++
    })

    const commentary = additionalInstructions.trim()

    if (commentary) {
      // Has additional commentary - requires replanning
      rejectPlan(commentary, editedStepsArray)
    } else {
      // Just edits, no commentary - proceed directly with edited plan
      // Pass deleted step numbers for backend to filter
      const deletedStepNumbers = Array.from(deletedSteps)
      approvePlan(deletedStepNumbers, editedStepsArray)
    }

    setAdditionalInstructions('')
    setStepModifications({})
    setDeletedSteps(new Set())
    setExpandedSteps(new Set())
  }

  const handleCancel = () => {
    rejectPlan('Cancelled by user')
    setAdditionalInstructions('')
    setStepModifications({})
    setDeletedSteps(new Set())
    setExpandedSteps(new Set())
  }

  return (
    <Dialog open={true} onClose={() => {}} className="relative z-50">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/30 dark:bg-black/50" aria-hidden="true" />

      {/* Dialog container */}
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <DialogPanel className="w-full max-w-2xl max-h-[85vh] flex flex-col rounded-xl bg-white dark:bg-gray-800 shadow-2xl">
          {/* Header */}
          <div className="flex items-center gap-3 px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-blue-100 dark:bg-blue-900 flex items-center justify-center">
              <ClipboardDocumentListIcon className="w-6 h-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div className="flex-1">
              <DialogTitle className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Review Execution Plan
              </DialogTitle>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {visibleSteps.length} step{visibleSteps.length !== 1 ? 's' : ''} to execute
                {deletedSteps.size > 0 && (
                  <span className="text-red-500 dark:text-red-400">
                    {' '}({deletedSteps.size} deleted)
                  </span>
                )}
              </p>
            </div>
            <button
              onClick={handleCancel}
              className="p-2 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title="Cancel"
            >
              <XMarkIcon className="w-5 h-5" />
            </button>
          </div>

          {/* Content - scrollable */}
          <div className="flex-1 overflow-y-auto px-6 py-4">
            {/* Problem statement */}
            <div className="mb-4 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Problem:</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">{problem}</p>
            </div>

            {/* Steps */}
            <div className="space-y-2 mb-4">
              <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                Plan Steps
              </p>
              {visibleSteps.length === 0 ? (
                <p className="text-sm text-gray-500 dark:text-gray-400 italic">
                  {deletedSteps.size > 0 ? 'All steps deleted' : 'No steps defined'}
                </p>
              ) : (
                <div className="space-y-1.5">
                  {visibleSteps.map((step, index) => {
                    const stepNum = step.number ?? steps.indexOf(step) + 1
                    return (
                      <StepItem
                        key={stepNum}
                        step={step}
                        index={index}
                        isExpanded={expandedSteps.has(stepNum)}
                        onToggle={() => toggleStep(stepNum)}
                        modification={stepModifications[stepNum] || ''}
                        onModificationChange={(value) => setStepModification(stepNum, value)}
                        onDelete={() => deleteStep(stepNum)}
                      />
                    )
                  })}
                </div>
              )}
            </div>

            {/* Additional Instructions */}
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
                Additional Instructions (optional)
              </label>
              <textarea
                value={additionalInstructions}
                onChange={(e) => setAdditionalInstructions(e.target.value)}
                placeholder="Add modifications or clarifications to the plan..."
                rows={3}
                className="w-full px-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:border-primary-500 focus:ring-1 focus:ring-primary-500 resize-none"
              />
            </div>
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 flex justify-between items-center flex-shrink-0">
            <button
              onClick={handleCancel}
              className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
            >
              Cancel
            </button>
            <div className="flex items-center gap-2">
              {hasAnyChanges && (
                <button
                  onClick={handleRevise}
                  className="px-4 py-2 text-sm font-medium rounded-lg border border-amber-500 text-amber-600 dark:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-900/20 transition-colors flex items-center gap-2"
                >
                  <ArrowPathIcon className="w-4 h-4" />
                  Revise Plan
                </button>
              )}
              <button
                onClick={handleApprove}
                className="px-5 py-2 text-sm font-medium rounded-lg bg-green-600 text-white hover:bg-green-700 transition-colors flex items-center gap-2"
              >
                <CheckCircleIcon className="w-4 h-4" />
                Approve
              </button>
            </div>
          </div>
        </DialogPanel>
      </div>
    </Dialog>
  )
}