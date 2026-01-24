// Plan View component

import { useState } from 'react'
import { ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline'
import { CheckCircleIcon, XCircleIcon, ClockIcon } from '@heroicons/react/24/solid'
import type { Plan, Step } from '@/types/api'

interface PlanViewProps {
  plan: Plan
  onApprove?: () => void
  onReject?: (feedback: string) => void
  showActions?: boolean
}

const stepStatusIcons: Record<string, { icon: typeof ClockIcon; color: string }> = {
  pending: { icon: ClockIcon, color: 'text-gray-400' },
  running: { icon: ClockIcon, color: 'text-blue-500 animate-spin' },
  completed: { icon: CheckCircleIcon, color: 'text-green-500' },
  failed: { icon: XCircleIcon, color: 'text-red-500' },
  skipped: { icon: XCircleIcon, color: 'text-gray-400' },
}

function StepItem({ step, isExpanded, onToggle }: { step: Step; isExpanded: boolean; onToggle: () => void }) {
  const status = stepStatusIcons[step.status] || stepStatusIcons.pending
  const StatusIcon = status.icon

  // Handle missing fields
  const expectedInputs = step.expected_inputs || []
  const expectedOutputs = step.expected_outputs || []
  const dependsOn = step.depends_on || []

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
      >
        {isExpanded ? (
          <ChevronDownIcon className="w-4 h-4 text-gray-400" />
        ) : (
          <ChevronRightIcon className="w-4 h-4 text-gray-400" />
        )}
        <StatusIcon className={`w-5 h-5 ${status.color}`} />
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Step {step.number ?? '?'}
        </span>
        <span className="flex-1 text-sm text-gray-600 dark:text-gray-400 truncate">
          {step.goal || 'Processing...'}
        </span>
      </button>

      {isExpanded && (
        <div className="px-4 py-3 bg-gray-50 dark:bg-gray-800/50 border-t border-gray-200 dark:border-gray-700">
          <div className="space-y-2 text-sm">
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
            {step.code && (
              <div className="mt-2">
                <span className="text-gray-500 dark:text-gray-400 block mb-1">Code:</span>
                <pre className="bg-gray-900 text-gray-100 p-3 rounded-md text-xs overflow-x-auto">
                  {step.code}
                </pre>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export function PlanView({ plan, onApprove, onReject, showActions = true }: PlanViewProps) {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set())
  const [rejectFeedback, setRejectFeedback] = useState('')
  const [showRejectInput, setShowRejectInput] = useState(false)

  // Handle missing or malformed plan data
  const steps = plan.steps || []
  const completedSteps = plan.completed_steps || []
  const problem = plan.problem || 'Processing...'

  const toggleStep = (stepNumber: number) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev)
      if (next.has(stepNumber)) {
        next.delete(stepNumber)
      } else {
        next.add(stepNumber)
      }
      return next
    })
  }

  const handleReject = () => {
    if (rejectFeedback.trim()) {
      onReject?.(rejectFeedback)
      setShowRejectInput(false)
      setRejectFeedback('')
    }
  }

  return (
    <div className="space-y-4">
      {/* Problem statement */}
      <div className="text-sm text-gray-600 dark:text-gray-400">
        <span className="font-medium">Problem:</span> {problem}
      </div>

      {/* Progress */}
      {steps.length > 0 && (
        <div className="flex items-center gap-2 text-sm">
          <div className="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-primary-500 transition-all duration-300"
              style={{
                width: `${(completedSteps.length / steps.length) * 100}%`,
              }}
            />
          </div>
          <span className="text-gray-500 dark:text-gray-400">
            {completedSteps.length}/{steps.length}
          </span>
        </div>
      )}

      {/* Steps */}
      <div className="space-y-2">
        {steps.map((step, index) => (
          <StepItem
            key={step.number ?? index}
            step={step}
            isExpanded={expandedSteps.has(step.number ?? index)}
            onToggle={() => toggleStep(step.number ?? index)}
          />
        ))}
      </div>

      {/* Actions */}
      {showActions && onApprove && onReject && (
        <div className="flex items-center gap-3 pt-2">
          {showRejectInput ? (
            <div className="flex-1 flex gap-2">
              <input
                type="text"
                value={rejectFeedback}
                onChange={(e) => setRejectFeedback(e.target.value)}
                placeholder="Reason for rejection..."
                className="flex-1 input text-sm"
                autoFocus
              />
              <button onClick={handleReject} className="btn-secondary text-sm">
                Reject
              </button>
              <button
                onClick={() => setShowRejectInput(false)}
                className="btn-ghost text-sm"
              >
                Cancel
              </button>
            </div>
          ) : (
            <>
              <button onClick={onApprove} className="btn-primary text-sm">
                Approve Plan
              </button>
              <button
                onClick={() => setShowRejectInput(true)}
                className="btn-secondary text-sm"
              >
                Reject
              </button>
            </>
          )}
        </div>
      )}
    </div>
  )
}