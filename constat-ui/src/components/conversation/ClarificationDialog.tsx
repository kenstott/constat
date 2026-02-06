// Clarification Dialog - stepper dialog for answering clarification questions

import { useState, useEffect } from 'react'
import { Dialog, DialogPanel, DialogTitle, RadioGroup } from '@headlessui/react'
import {
  QuestionMarkCircleIcon,
  XMarkIcon,
  CheckCircleIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline'
import { useSessionStore } from '@/store/sessionStore'

export function ClarificationDialog() {
  const {
    clarification,
    answerClarification,
    skipClarification,
    setClarificationStep,
    setClarificationAnswer,
  } = useSessionStore()

  const [customAnswer, setCustomAnswer] = useState('')

  // Reset custom answer when step changes
  useEffect(() => {
    setCustomAnswer('')
  }, [clarification?.currentStep])

  if (!clarification?.needed || clarification.questions.length === 0) {
    return null
  }

  const currentStep = clarification.currentStep
  const totalSteps = clarification.questions.length
  const currentQuestion = clarification.questions[currentStep]
  const currentAnswer = clarification.answers[currentStep]
  const isOther = currentAnswer === '__other__'

  const handleOptionSelect = (value: string) => {
    setClarificationAnswer(currentStep, value)
    if (value !== '__other__') {
      setCustomAnswer('')
    }
  }

  const handleCustomAnswerChange = (value: string) => {
    setCustomAnswer(value)
  }

  // Helper to get the effective answer for current step (handles "Other" with custom text)
  const getCurrentEffectiveAnswer = (): string | undefined => {
    if (isOther && customAnswer.trim()) {
      return customAnswer.trim()
    }
    if (currentAnswer && currentAnswer !== '__other__') {
      return currentAnswer
    }
    return undefined
  }

  const handleNext = () => {
    // Save the effective answer (custom text if "Other" is selected)
    const effectiveAnswer = getCurrentEffectiveAnswer()
    if (effectiveAnswer) {
      setClarificationAnswer(currentStep, effectiveAnswer)
    }
    if (currentStep < totalSteps - 1) {
      setClarificationStep(currentStep + 1)
    }
  }

  const handleBack = () => {
    // Save current answer before going back
    const effectiveAnswer = getCurrentEffectiveAnswer()
    if (effectiveAnswer) {
      setClarificationAnswer(currentStep, effectiveAnswer)
    }
    if (currentStep > 0) {
      setClarificationStep(currentStep - 1)
    }
  }

  const handleSubmit = () => {
    // Build final answers, replacing any '__other__' with the custom text
    const finalAnswers: Record<number, string> = {}

    clarification.questions.forEach((_, i) => {
      if (i === currentStep) {
        // Current step - use effective answer (custom text if "Other")
        const effectiveAnswer = getCurrentEffectiveAnswer()
        if (effectiveAnswer) {
          finalAnswers[i] = effectiveAnswer
        }
      } else {
        // Previous steps - use stored answer (should already be the actual text)
        const storedAnswer = clarification.answers[i]
        if (storedAnswer && storedAnswer !== '__other__') {
          finalAnswers[i] = storedAnswer
        }
      }
    })

    answerClarification(finalAnswers)
  }

  const handleSkip = () => {
    skipClarification()
  }

  const canGoNext = currentStep < totalSteps - 1
  const canGoBack = currentStep > 0
  const hasCurrentAnswer = isOther
    ? customAnswer.trim().length > 0
    : currentAnswer && currentAnswer !== '__other__'


  return (
    <Dialog open={true} onClose={() => {}} className="relative z-50">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/30 dark:bg-black/50" aria-hidden="true" />

      {/* Dialog container */}
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <DialogPanel className="w-full max-w-lg rounded-xl bg-white dark:bg-gray-800 shadow-2xl">
          {/* Header */}
          <div className="flex items-center gap-3 px-6 py-4 border-b border-gray-200 dark:border-gray-700">
            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary-100 dark:bg-primary-900 flex items-center justify-center">
              <QuestionMarkCircleIcon className="w-6 h-6 text-primary-600 dark:text-primary-400" />
            </div>
            <div className="flex-1">
              <DialogTitle className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Clarification Needed
              </DialogTitle>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                {clarification.ambiguityReason}
              </p>
            </div>
            <button
              onClick={handleSkip}
              className="p-2 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title="Skip clarification"
            >
              <XMarkIcon className="w-5 h-5" />
            </button>
          </div>

          {/* Progress indicator */}
          {totalSteps > 1 && (
            <div className="px-6 pt-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  Question {currentStep + 1} of {totalSteps}
                </span>
                <div className="flex gap-1">
                  {clarification.questions.map((_, i) => (
                    <div
                      key={i}
                      className={`w-2 h-2 rounded-full transition-colors ${
                        i === currentStep
                          ? 'bg-primary-500'
                          : clarification.answers[i]
                          ? 'bg-primary-300 dark:bg-primary-700'
                          : 'bg-gray-200 dark:bg-gray-700'
                      }`}
                    />
                  ))}
                </div>
              </div>
              <div className="w-full h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary-500 transition-all duration-300"
                  style={{ width: `${((currentStep + 1) / totalSteps) * 100}%` }}
                />
              </div>
            </div>
          )}

          {/* Content */}
          <div className="px-6 py-5">
            {/* Original question context */}
            <div className="mb-4 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg">
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Your question:</p>
              <p className="text-sm text-gray-700 dark:text-gray-300">
                {clarification.originalQuestion}
              </p>
            </div>

            {/* Question */}
            <p className="text-base font-medium text-gray-900 dark:text-gray-100 mb-4">
              {currentQuestion.text}
            </p>

            {/* Radio options */}
            <RadioGroup
              value={isOther ? '__other__' : (currentAnswer || '')}
              onChange={handleOptionSelect}
              className="space-y-2"
            >
              {currentQuestion.suggestions.map((suggestion, index) => (
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
                        onChange={(e) => handleCustomAnswerChange(e.target.value)}
                        onKeyDown={(e) => {
                          e.stopPropagation()
                          if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault()
                            if (hasCurrentAnswer) {
                              if (canGoNext) {
                                handleNext()
                              } else {
                                handleSubmit()
                              }
                            }
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
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 flex justify-between items-center">
            <button
              onClick={handleSkip}
              className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
            >
              Skip and proceed anyway
            </button>
            <div className="flex items-center gap-2">
              {/* Back button */}
              {canGoBack && (
                <button
                  onClick={handleBack}
                  className="px-3 py-2 text-sm font-medium rounded-lg border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors flex items-center gap-1"
                >
                  <ChevronLeftIcon className="w-4 h-4" />
                  Back
                </button>
              )}

              {/* Submit button - always available if at least one answer */}
              {Object.keys(clarification.answers).length > 0 || hasCurrentAnswer ? (
                <button
                  onClick={handleSubmit}
                  className="px-4 py-2 text-sm font-medium rounded-lg bg-green-600 text-white hover:bg-green-700 transition-colors"
                >
                  Submit
                </button>
              ) : null}

              {/* Next button */}
              {canGoNext && (
                <button
                  onClick={handleNext}
                  disabled={!hasCurrentAnswer}
                  className="px-4 py-2 text-sm font-medium rounded-lg bg-primary-600 text-white hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-1"
                >
                  Next
                  <ChevronRightIcon className="w-4 h-4" />
                </button>
              )}

              {/* Final continue button if on last step */}
              {!canGoNext && !hasCurrentAnswer && (
                <button
                  disabled
                  className="px-5 py-2 text-sm font-medium rounded-lg bg-primary-600 text-white opacity-50 cursor-not-allowed"
                >
                  Continue
                </button>
              )}

              {!canGoNext && hasCurrentAnswer && Object.keys(clarification.answers).length === 0 && (
                <button
                  onClick={handleSubmit}
                  className="px-5 py-2 text-sm font-medium rounded-lg bg-primary-600 text-white hover:bg-primary-700 transition-colors"
                >
                  Continue
                </button>
              )}
            </div>
          </div>
        </DialogPanel>
      </div>
    </Dialog>
  )
}