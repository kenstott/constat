// Passkey registration flow — shown on first login when no passkey exists

import { useState } from 'react'
import { usePasskey } from '@/hooks/usePasskey'

interface PasskeySetupProps {
  userId: string
  onComplete: () => void
  onSkip: () => void
}

export function PasskeySetup({ userId, onComplete, onSkip }: PasskeySetupProps) {
  const { loading, error, registerPasskey } = usePasskey({ userId })
  const [registered, setRegistered] = useState(false)

  const handleRegister = async () => {
    try {
      await registerPasskey()
      setRegistered(true)
      onComplete()
    } catch {
      // Error displayed via hook state
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 px-4">
      <div className="max-w-md w-full space-y-6">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Set Up Passkey</h1>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
            Secure your data vault with a passkey. This uses your device's biometric or PIN authentication.
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 space-y-6">
          {error && (
            <div className="p-3 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-md">
              <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
            </div>
          )}

          {registered ? (
            <div className="text-center space-y-4">
              <div className="w-12 h-12 mx-auto bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center">
                <svg className="w-6 h-6 text-green-600 dark:text-green-400" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                </svg>
              </div>
              <p className="text-gray-700 dark:text-gray-300">Passkey registered successfully.</p>
            </div>
          ) : (
            <>
              <div className="space-y-3 text-sm text-gray-600 dark:text-gray-400">
                <div className="flex items-start gap-3">
                  <svg className="w-5 h-5 mt-0.5 text-primary-500 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
                  </svg>
                  <span>Your passkey encrypts personal data stored on the server</span>
                </div>
                <div className="flex items-start gap-3">
                  <svg className="w-5 h-5 mt-0.5 text-primary-500 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M7.864 4.243A7.5 7.5 0 0119.5 10.5c0 2.92-.556 5.709-1.568 8.268M5.742 6.364A7.465 7.465 0 004.5 10.5a48.667 48.667 0 00-6 .371m12 0a48.667 48.667 0 00-6-.371m12 0l.048.098c.197.403.39.81.578 1.225M4.117 16.773a48.394 48.394 0 005.928.371" />
                  </svg>
                  <span>Uses your device's biometric sensor or security key</span>
                </div>
              </div>

              <button
                onClick={handleRegister}
                disabled={loading}
                className="w-full flex justify-center py-2.5 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? (
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  'Register Passkey'
                )}
              </button>

              <button
                onClick={onSkip}
                disabled={loading}
                className="w-full text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
              >
                Skip for now
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
