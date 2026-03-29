// Passkey unlock prompt — shown when server returns 423 (vault locked)

import { usePasskey } from '@/hooks/usePasskey'

interface PasskeyUnlockProps {
  userId: string
  onUnlocked: () => void
}

export function PasskeyUnlock({ userId, onUnlocked }: PasskeyUnlockProps) {
  const { loading, error, authenticatePasskey } = usePasskey({ userId })

  const handleUnlock = async () => {
    try {
      const result = await authenticatePasskey()
      if (result.vault_unlocked) {
        onUnlocked()
      }
    } catch {
      // Error displayed via hook state
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 px-4">
      <div className="max-w-md w-full space-y-6">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Vault Locked</h1>
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
            Your data vault is encrypted. Authenticate with your passkey to unlock.
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 space-y-6">
          {error && (
            <div className="p-3 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-md">
              <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
            </div>
          )}

          <div className="flex justify-center">
            <svg className="w-16 h-16 text-gray-400 dark:text-gray-500" fill="none" viewBox="0 0 24 24" strokeWidth={1} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.5 10.5V6.75a4.5 4.5 0 10-9 0v3.75m-.75 11.25h10.5a2.25 2.25 0 002.25-2.25v-6.75a2.25 2.25 0 00-2.25-2.25H6.75a2.25 2.25 0 00-2.25 2.25v6.75a2.25 2.25 0 002.25 2.25z" />
            </svg>
          </div>

          <button
            onClick={handleUnlock}
            disabled={loading}
            className="w-full flex justify-center py-2.5 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              'Unlock with Passkey'
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
