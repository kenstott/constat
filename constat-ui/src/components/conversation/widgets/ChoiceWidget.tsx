// ChoiceWidget - Radio group for selecting from suggestions (extracted from ClarificationDialog)

import { RadioGroup } from '@headlessui/react'
import { CheckCircleIcon } from '@heroicons/react/24/outline'

interface ChoiceWidgetProps {
  config: Record<string, unknown>
  suggestions: string[]
  value: string
  onAnswer: (freeform: string, structured?: string) => void
  customAnswer: string
  onCustomAnswerChange: (value: string) => void
  onSubmitShortcut?: () => void
}

export function ChoiceWidget({
  suggestions,
  value,
  onAnswer,
  customAnswer,
  onCustomAnswerChange,
  onSubmitShortcut,
}: ChoiceWidgetProps) {
  const isOther = value === '__other__'

  const handleSelect = (selected: string) => {
    if (selected === '__other__') {
      onAnswer('__other__', undefined)
    } else {
      onAnswer(selected, selected)
    }
  }

  return (
    <RadioGroup
      value={isOther ? '__other__' : (value || '')}
      onChange={handleSelect}
      className="space-y-2"
    >
      {suggestions.map((suggestion, index) => (
        <RadioGroup.Option
          key={index}
          value={suggestion}
          className={({ checked }) =>
            `relative flex items-center gap-3 px-4 py-3 cursor-pointer rounded-lg border transition-colors ${
              checked
                ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30'
                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
            }`
          }
        >
          {({ checked }) => (
            <>
              <div
                className={`w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors ${
                  checked
                    ? 'border-primary-500 bg-primary-500'
                    : 'border-gray-300 dark:border-gray-600'
                }`}
              >
                {checked && <CheckCircleIcon className="w-3 h-3 text-white" />}
              </div>
              <span
                className={`text-sm ${
                  checked
                    ? 'text-primary-700 dark:text-primary-300 font-medium'
                    : 'text-gray-700 dark:text-gray-300'
                }`}
              >
                {suggestion}
              </span>
            </>
          )}
        </RadioGroup.Option>
      ))}

      {/* Other option */}
      <RadioGroup.Option
        value="__other__"
        className={({ checked }) =>
          `relative flex flex-col gap-2 px-4 py-3 cursor-pointer rounded-lg border transition-colors ${
            checked
              ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/30'
              : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
          }`
        }
      >
        {({ checked }) => (
          <>
            <div className="flex items-center gap-3">
              <div
                className={`w-5 h-5 rounded-full border-2 flex items-center justify-center transition-colors ${
                  checked
                    ? 'border-primary-500 bg-primary-500'
                    : 'border-gray-300 dark:border-gray-600'
                }`}
              >
                {checked && <CheckCircleIcon className="w-3 h-3 text-white" />}
              </div>
              <span
                className={`text-sm ${
                  checked
                    ? 'text-primary-700 dark:text-primary-300 font-medium'
                    : 'text-gray-700 dark:text-gray-300'
                }`}
              >
                Other
              </span>
            </div>
            {checked && (
              <input
                type="text"
                value={customAnswer}
                onChange={(e) => onCustomAnswerChange(e.target.value)}
                onKeyDown={(e) => {
                  e.stopPropagation()
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    onSubmitShortcut?.()
                  }
                }}
                onKeyUp={(e) => e.stopPropagation()}
                onKeyPress={(e) => e.stopPropagation()}
                placeholder="Type your answer..."
                className="ml-8 px-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 placeholder-gray-400 focus:border-primary-500 focus:ring-1 focus:ring-primary-500"
                autoFocus
                onClick={(e) => e.stopPropagation()}
              />
            )}
          </>
        )}
      </RadioGroup.Option>
    </RadioGroup>
  )
}
